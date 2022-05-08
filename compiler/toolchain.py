import os
import re
import pandas as pd
from copy import deepcopy
import networkx as nx
import matplotlib.pyplot as plt
from compiler import global_control as gc
import numpy as np

from tracegen.generator import gen_fake_trace
from op_graph.micro_op_graph import MicroOpGraph
from mapper.task_map import ml_mapping
from ATM.gen_mapping import gen_mapping
from route_algorithms.mesh import MeshTreeRouter

from timeloop_agents.layer import TimeloopLayer
from spatialsim_agents.trace_gen import TraceGenerator


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
    traffic = pd.DataFrame(columns=["layer", "src", "dst", "interval", "flit", "counts"])
    

    def __init__(self, layers, cores):
        self.layer_names = layers
        self.cores = cores
        self.model_names = [re.search(r"(^.+)_", layer).group(1) for layer in layers]
        self.result_names = ["result_" + layer + ".yaml" for layer in layers]
        self.prob_spec_names = [layer + ".yaml" for layer in layers]
        self.exchange_file_name = "layer-set-exchange-info.yaml"
        self.counter = 0

    def compileTask(self):

        if gc.dataflow_engine == "timeloop":
            op_graph = self._gen_op_graph()
        else: 
            assert False
            # Fake trace generator has not been compatible with op_graph
            vir_trace = gen_fake_trace()
    
        nx.draw(op_graph.get_data())
        plt.savefig(os.path.join(gc.visualization_root, "micro_operators.png"))
        plt.close()

        # map tasks to pe array
        # positions = ml_mapping().map()
        # op_graph.set_physical_position(positions)

        mapping = np.load(gc.mapping)
        mapping = mapping.tolist()
        print(mapping)
        positions = gen_mapping().map(mapping)
        op_graph.set_physical_position(positions)

        # dump as spatialsim trace
        self._to_spatialsim_trace(op_graph)


        # # Dual-Phase Routing
        # dual_phase_trace = self._selectHubNode(mapped_trace)
        # dual_phase_trace = self._genSpanningTree(dual_phase_trace)

        # # Dump the trace for FOCUS scheduling.
        # trace_for_focus = dual_phase_trace
        # self._dumpFocusTrace(trace_for_focus)

        # # Dump the trace for simulation.
        # self._cvtAndDumpSimTrace(trace_for_focus)


    def _gen_op_graph(self):
        op_graph = MicroOpGraph()
        for layer, model, prob_spec, core in \
            zip(self.layer_names, self.model_names, self.prob_spec_names, self.cores):
            
            # Initialize the agent
            layer = TimeloopLayer(prob_spec, model_dir=model, dram_spatial_size=core, prj_root=gc.prj_root)
            # Invoke timeloop and get reports
            report = layer.run(TimeloopLayer.report_as_dataframe)
            op_graph.add_layer(report)

        return op_graph


    def _to_spatialsim_trace(self, op_graph):

        dest_dir = os.path.join(gc.spatial_sim_root, "tasks", gc.taskname)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        trace_files = {i: open(os.path.join(dest_dir, "c{}.inst".format(i)), "w") for i in range(gc.array_size)}
        routing_board_file = open(os.path.join(dest_dir, "routing_board"), "w")

        router = MeshTreeRouter(gc.array_diameter)
        generator = TraceGenerator(router)
        generator.gen_trace(trace_files, routing_board_file, op_graph)

        for f in trace_files.values():
            f.close()
        routing_board_file.close()

    def _to_focus_trace(self, op_graph):
        pass

    def _dumpFocusTrace(self, traffic):
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

