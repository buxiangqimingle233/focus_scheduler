import os
import re
from copy import deepcopy
import networkx as nx
from compiler import global_control as gc

from op_graph.micro_op_graph import MicroOpGraph
# Fake trace generator
from fake_trace_generator.generator import gen_fake_trace
# Timeloop agents
from timeloop_agents.layer import TimeloopLayer
# Task Mapper
from mapping_algorithms.random_mapper import RandomMapper
from mapping_algorithms.hilbert_mapper import HilbertMapper
# Tree Generator
from compiler.routing_algorithms.meshtree_router import MeshTreeRouter, RPMTreeRouter, WhirlTreeRouter, BAMTreeRouter, Steiner_TreeRouter
# The backend to generate trace for spatial_sim
from compiler.spatialsim_agents.trace_generator import TraceGenerator
from compiler.spatialsim_agents.variables import Variables

class TaskCompiler():
    r'''This module compiles tasks for FOCUS-like spatial architectures. \
    A task is a set of NN layers, indentified with its model name and its serial number, e.g. bert_layer1. \
    This compiler has typically 3 passes: \
        1. Per-layer scheduling: Invoke timeloop, after which we get a graph of mini-operators. \
        2. Operator mapping: Assign each mini-operator with a core in the spatial array. \
        3. Traffic scheduling [ Optional ]: Generate the branching-tree for each multicast packet, globally route,
            and stragger packets at time to avoid conflicting use of channels. \
    The `binary` files are exported with different forms and places: spatial-sim like instruction lists at `sim_folder`/benchmark;
    The stream graph formatted as pandas DataFrame;
    '''

    # Deprecated
    # traffic = pd.DataFrame(columns=["layer", "src", "dst", "interval", "flit", "counts"])

    def __init__(self):
        self.layer_names = gc.layer_names
        self.cores = gc.cores
        self.model_names = [re.search(r"(^.+)_", layer).group(1) for layer in gc.layer_names]
        self.result_names = ["result_" + layer + ".yaml" for layer in gc.layer_names]
        self.prob_spec_names = [layer + ".yaml" for layer in gc.layer_names]


    def compile(self):

        if gc.dataflow_engine == "timeloop":
            op_graph = self._gen_op_graph()
        else: 
            assert False
            # Fake trace generator has not been compatible with op_graph
            vir_trace = gen_fake_trace()

        # map tasks to pe array
        op_graph = self._map_operators(op_graph)

        nx.write_gpickle(op_graph, "test.gpickle")

        op_graph.draw_graph(os.path.join(gc.visualization_root, "micro_operators.png"))
        op_graph.draw_mapping(os.path.join(gc.visualization_root, "mapping.png"))

        self.compute_cycle_lower_bound = op_graph.total_compute_cycles()

        for node, attr in op_graph.get_data().nodes(data=True):
            attr['count'] = 1
        start = 100000
        for u, v, eattr in op_graph.get_data().edges(data=True):
            if eattr['edge_type'] == "control":
                eattr['fid'] = start
                eattr['size'] = 1
                start += 1 

        nx.write_gpickle(op_graph.get_data(), f'./try.gpickle')

        # dump as spatialsim trace
        self._to_spatialsim_trace(op_graph)

    def get_compute_cycle(self):
        assert hasattr(self, "compute_cycle_lower_bound")
        return self.compute_cycle_lower_bound

    def _gen_op_graph(self):
        print("Generating the operator graph using timeloop")

        op_graph = MicroOpGraph()
        for layer, model, prob_spec, core in zip(self.layer_names, self.model_names, self.prob_spec_names, self.cores):
            print("Info:", "Working for", layer)    
            # Initialize the agent
            tlagent = TimeloopLayer(prob_spec, model_dir=model, dram_spatial_size=core, prj_root=gc.prj_root)
            # Invoke timeloop for dataflow reports
            timeloop_report = tlagent.run(TimeloopLayer.report_as_dataframe)
            op_graph.add_layer(timeloop_report, gc.batch)
            print("====================== FINISH =========================\n\n")

        return op_graph


    def _map_operators(self, op_graph):
        layout = self._gen_physical_layout()
        # mapper = RandomMapper(op_graph, layout)
        mapper = HilbertMapper(op_graph, layout, gc.array_diameter, gc.virtualization)
        return mapper.map()


    def _to_spatialsim_trace(self, op_graph):

        # Do some path handling works
        Variables.gen_working_dir(gc.spatial_sim_root, gc.taskname)

        trace_files = {key: open(value, "w") for key, value in \
            Variables.get_trace_file_path_dict(gc.spatial_sim_root, gc.taskname, gc.array_size).items()}

        routing_board_file = open(Variables.get_routing_board_path(gc.spatial_sim_root, gc.taskname), "w")
        specification_file = open(Variables.get_spec_path(gc.spatial_sim_root, gc.taskname), "w")
        specification_ref_file = open(Variables.get_ref_spec_path(gc.spatial_sim_root), "r")

        # Generate multicast tree for multi-end packets
        #router = MeshTreeRouter(gc.array_diameter)
        router = RPMTreeRouter(gc.array_diameter)
        #router = WhirlTreeRouter(gc.array_diameter)
        #router = BAMTreeRouter(gc.array_diameter)
        #router = Steiner_TreeRouter(gc.array_diameter)
        TraceGenerator().gen_trace(trace_files, routing_board_file, specification_file, \
            specification_ref_file, op_graph, router)

        for f in trace_files.values():
            f.close()
        routing_board_file.close()
        specification_ref_file.close()
        specification_file.close()


    def _gen_physical_layout(self):
        d = gc.array_diameter - 1
        mems = [
            d // 2, d // 2 + 1,
            d // 2 * gc.array_diameter, (d // 2 + 1) * gc.array_diameter, 
            d // 2 * gc.array_diameter + d, (d // 2 + 1) * gc.array_diameter + d, 
            d * gc.array_diameter + d // 2, d * gc.array_diameter + d // 2 + 1,
        ]

        # test
        mems.append(gc.array_size - 1)
        mems.append(gc.array_size - 2)
        mems.append(0)
        mems.append(1)

        cores = [i for i in range(gc.array_size) if i not in mems]
        layout = {i: "core" for i in cores}
        layout.update({i: "mem" for i in mems})
        return layout

    def _to_focus_trace(self, op_graph):
        pass

    def _dumpFocusTrace(self, traffic):
        # FIXME: Deprecated
        dest_dir = os.path.join(gc.focus_buffer, gc.taskname)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        trace_file = "trace_{}.json".format(gc.flit_size)
        traffic.to_json(os.path.join(dest_dir, trace_file))


    def _test(self, traffic):
        df = deepcopy(traffic)
        gc.debug_show(traffic)
        # gc.debug_show(df.drop_duplicates("layer"))
        for layer in gc.layer_names:
            streams = df[df["layer"] == layer]
            # streams = streams.drop_duplicates("datatype")
            o = streams[streams["datatype"] == "output"].iloc[0]
            w = streams[streams["datatype"] == "weight"].iloc[0]
            i = streams[streams["datatype"] == "input"].iloc[0]
            d = streams[streams["datatype"] == "www"]
            print(d.empty)
            if o["interval"] < w["interval"] and o["interval"] < i["interval"]: 
                print("O:{}, W:{}, I:{}".format(o["interval"], w["interval"], i["interval"]))
                exit()
            # print(o["interval"], w["interval"], )

