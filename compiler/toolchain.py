import os
import re
from matplotlib.pyplot import axis
from myhdl import instance
import pandas as pd
import numpy as np
from copy import deepcopy
from math import ceil
from compiler.mapper import task_map

from utils.layer import Layer
from utils import global_control as gc
from mapper.task_map import ml_mapping
from mapper.spanningtree import SpanningTree
from tracegen.generator import gen_fake_trace


class FocusToolChain():
    r'''This module is the compiling toolchain for FOCUS-like spatial architectures. \
    It takes task specifications as the input and generate the traffic trace to drive subsequent procedures. 
    Specifically, Focus-sim and Focus optimizer take its output. 
    The toolchain firstly schedules the dataflows for the tasks, then searches the core mapping,
    and selects the hub-node and generates spanning tree for each traffic flow. \
    The resulting trace file is stored in multiple places, including ./trace.json, `sim_folder`/benchmark/,
    and return values.
    '''
    traffic = pd.DataFrame(columns=["layer", "src", "dst", "interval", "flit", "counts"])

    def __init__(self, layers, cores):
        self.layer_names = layers
        self.cores = cores
        self.model_names = [re.search(r"(^.+)_", layer).group(1) for layer in layers]
        self.result_names = ["result_" + layer + ".yaml" for layer in layers]
        self.prob_spec_names = [layer + ".yaml" for layer in layers]
        self.exchange_file_name = "layer-set-exchange-info.yaml"

    def compileTask(self):
        r'''Invoke this compiling tool chain to generate the traffic trace. 

        ================================================================
        Parameters:
            dataflow_engine: str. It dictates the searching engine of dataflows, we just support
            `timeloop` and `fake` now.
            subdir_name: str. The directory where trace files locate at the simulation folder.
        '''
        if gc.dataflow_engine == "timeloop":
            vir_trace = self._invokeTimeloop()
        else:
            vir_trace = gen_fake_trace()

        # Mapping
        task_mapper = ml_mapping()
        mapped_trace = self._applyMapping(vir_trace, task_mapper.map())

    
        # FIXME: gnuplot does not work now
        # os.system("gnuplot ../mapper/mapping_vis.gp")

        # Dual-Phase Routing
        dual_phase_trace = self._selectHubNode(mapped_trace)
        dual_phase_trace = self._genSpanningTree(dual_phase_trace)

        # Dump the trace for FOCUS scheduling.
        trace_for_focus = dual_phase_trace
        self._dumpFocusTrace(trace_for_focus)

        # Dump the trace for simulation.
        self._cvtAndDumpSimTrace(trace_for_focus)

    def analyzeSimResult(self):
        result = pd.read_csv(
            os.path.join(gc.spt_sim_root, "test", gc.taskname, "brief_report.csv"),
            header=None, index_col=None,
            names=["name", "flit", "cycle"]
        )

        # restore original clock frequency
        result["cycle"] *= gc.overclock
        result = result.sort_values(by=["flit"], ascending=True)

        compute_time = (self.traffic["interval"] * self.traffic["counts"]).quantile(gc.quantile_)

        result.loc[:, "slowdown"] = result["cycle"] / compute_time
        result.to_csv(os.path.join("focus-final-out", "baseline_{}.csv".format(gc.taskname)))
        return result

    def analyzeFocusResult(self):
        result = pd.read_json(
            os.path.join("buffer", gc.taskname, "solution_{}.json".format(gc.flit_size))
        )

        compute_time = (result.groupby(by="layer").apply(lambda x: (x["interval"] * x["counts"]).max())).quantile(gc.quantile_)
        communicate_time = (result.groupby(by="layer").apply(lambda x: ((x["delay"] + x["interval"]) * x["counts"]).max())).quantile(gc.quantile_)
        slowdown = communicate_time / compute_time

        # Update the result file
        result_file = os.path.join("focus-final-out", "focus_{}.csv".format(gc.taskname))
        try:
            result = pd.read_csv(result_file, index_col=0)
        except FileNotFoundError:
            result = pd.DataFrame(columns=["name", "flit", "cycle", "slowdown"])
        
        inserted_line = ["{}_{}".format(gc.taskname, gc.flit_size), gc.flit_size, communicate_time, slowdown]

        # FUCK YOU Pandas !
        if not (result["flit"] == gc.flit_size).any():
            result = result.append(dict(zip(result.columns, inserted_line)), ignore_index=True)
        else:
            result[result["flit"] == gc.flit_size] = inserted_line

        result.to_csv(os.path.join("focus-final-out", "focus_{}.csv".format(gc.taskname)))

    @staticmethod
    def _extractTrafficTrace(layer: Layer, comm_bank):
        df = pd.DataFrame()
        for dti in range(3):
            for traffic_pertile in comm_bank[::-1]:
                traffic_perdatatype = traffic_pertile[dti]
                if traffic_perdatatype:
                    for flow in traffic_perdatatype:
                        df = df.append({
                            "layer": layer.layer_name,
                            "src": flow["srcs"],
                            "dst": flow["dsts"],
                            "interval": flow["pkt_interval"],
                            "flit": flow["bit_volume"] / gc.flit_size,
                            "counts": flow["cnt"],
                            "datatype": gc.datatype[dti]
                        }, ignore_index=True)
                    df = df[df["flit"] > 0]
                    df.loc[:, "flit"] = df["flit"].map(lambda x: int(max(x + 1, 2)))    # add headflits
                    break
        
        # debug_show(df["flit"].mean())

        # collapse all the datatypes
        bcast = df[(df["dst"].map(len) > 1) & (df["src"].map(lambda x: x[0]) == -1)]
        reduction = df[(df["dst"].map(lambda x: x[0]) == -1)]
        other = df[(df["src"].map(len) == 1) & (df["dst"].map(lambda x: x[0]) != -1)]

        # broadcast: keep destination, reduce source size to 1
        bcast.loc[:, "src"] = bcast["src"].map(lambda x: x[:1])

        # reduction: distribute
        tmp = pd.DataFrame(columns=reduction.columns)
        for idx, row in reduction.iterrows():
            srcs = row["src"]
            for s in srcs:
                new_row = deepcopy(row)
                new_row["src"] = [s]
                new_row["dst"] = [-2]
                new_row["delay"] = new_row["interval"]
                tmp = tmp.append(new_row)

        reduction = deepcopy(tmp)
        # other: keep still

        df = pd.concat([bcast, reduction, other])

        # always set weight source to reserved MC, and input source to mapper-defined nodees
        df.loc[df["datatype"] == "weight", "src"] = df[df["datatype"] == "weight"]["src"].map(lambda x: [-3] if x == [-1] else x)
        return df

    def _invokeTimeloop(self):
        for layer, model, result, prob_spec, core in \
            zip(self.layer_names, self.model_names, \
                self.result_names, self.prob_spec_names, self.cores):

            layer = Layer(prob_spec, model_dir=model, dram_spatial_size=core)
            traffic_per_layer = layer.run_with_gc(self._extractTrafficTrace)
            self.traffic = self.traffic.append(traffic_per_layer, ignore_index=True)

        return self.traffic

    def _applyMapping(self, traffic, core_map):

        src_sel = traffic.apply(lambda x: max(x["src"]) <= max(core_map[x["layer"]].keys()), axis=1)
        dst_sel = traffic.apply(lambda x: max(x["dst"]) <= max(core_map[x["layer"]].keys()), axis=1)

        traffic = traffic[(src_sel) & (dst_sel)]
        traffic.loc[:, "map_src"] = traffic.apply(lambda x: [core_map[x["layer"]][i] for i in x["src"]], axis=1)
        traffic.loc[:, "map_dst"] = traffic.apply(lambda x: [core_map[x["layer"]][i] for i in x["dst"]], axis=1)

        return traffic

    def _selectHubNode(self, traffic):
        sel = traffic["dst"].map(len) > 1
        rev_sel = ~sel
        bcast = traffic[sel]
        other = traffic[rev_sel]

        # left most captain
        bcast.loc[:, "captain"] = deepcopy(bcast["map_dst"].map(lambda x: min(x)))
        traffic = pd.concat([bcast, other])

        traffic.loc[:, "captain"] = traffic["captain"].astype("Int64")
        return traffic
    
    def _genSpanningTree(self, traffic):
        traffic["tree"] = np.NaN
        
        def genTree(row):
            if pd.isna(row["captain"]):
                return []
            else:
                gen = SpanningTree()
                return gen.wrapper_genST(int(row["captain"]), row["map_dst"])[0]

        def genEPFL(row):
            if pd.isna(row["captain"]):
                return np.NaN
            else:
                gen = SpanningTree()
                return gen.wrapper_genST(int(row["captain"]), row["map_dst"])[1]

        traffic.loc[:, "tree"] = traffic.apply(genTree, axis=1)
        traffic.loc[:, "epfl"] = traffic.apply(genEPFL, axis=1)
        
        return traffic

    def _cvtAndDumpSimTrace(self, traffic):
        df = deepcopy(traffic)

        # The traffic is the flow-centric data structure. We convert it to node-centric 
        # structure to drive the spatial simulator. 
        df = df.explode("map_src").explode("map_dst")
        # Remove in-place trasmission
        df = df[df["map_src"] != df["map_dst"]]
        df = df[["map_src", "map_dst", "datatype", "interval", "flit", "counts", "layer"]]
        # Allocate an unique flow id for each flow
        df.loc[:, "fid"] = range(df.shape[0])


        # Scale the trace with the accelerating ratio
        df["flit"] = df["flit"].map(lambda x: ceil(x / gc.overclock))
        df["interval"] = df["interval"].map(lambda x: ceil(x / gc.overclock))
        df["counts"] = df["counts"].map(lambda x: ceil(x * gc.shrink))

        # gc.debug_show(df)

        nodes = [{"nid": i} for i in range(gc.array_size)]
        for n in nodes:
            outf = df[df["map_src"] == n["nid"]]
            inf = df[df["map_dst"] == n["nid"]]
            outf["depend"], inf["depend"] = None, None

            # outflows can be output (compute PE), input (memory controller & hub node), and weight (MC)
            out_output = outf[outf["datatype"] == "output"]
            out_input = outf[outf["datatype"] == "input"]
            out_weight = outf[outf["datatype"] == "weight"]

            def tolist(flows):
                if not flows.empty:
                    return flows.tolist()
                else:
                    return []

            out_output["depend"] = out_output.apply(
                    lambda f: tolist(inf[(inf["layer"] == f["layer"]) & (inf["datatype"] != "output")] \
                        .apply(lambda x: (x["fid"], min(f["interval"]/x["interval"], x["counts"]/f["counts"])), axis=1)), 
            axis=1) 

            def prelayer(layer):
                layer_number = re.findall(r"\d+", layer)[-1]
                pre_layer_number = str(int(layer_number) - 1)
                pre_layer = re.sub(layer_number[::-1], layer[::-1], pre_layer_number[::-1])[::-1]
                return pre_layer

            out_input["depend"] = out_input.apply(
                lambda f: tolist(inf[(inf["datatype"] == "output") & (inf["layer"] == prelayer(f["layer"]))]  \
                    .apply(lambda x: (x["fid"], min(f["interval"]/x["interval"], x["counts"]/f["counts"])), axis=1)),
            axis=1)

            out_weight["depend"] = [[] for _ in range(out_weight.shape[0])]

            n["out_flows"] = pd.concat([out_input, out_output, out_weight])
            n["in_flows"] = inf

        trace_file = "trace_{}.txt".format(gc.flit_size)
        dest_dir = os.path.join(gc.spt_sim_root, "benchmark", gc.taskname)
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        with open(os.path.join(dest_dir, trace_file), "w") as wf:
            for n in nodes:
                print("{} {}".format(n["nid"], n["out_flows"].shape[0]), file=wf)
                for _, flow in n["out_flows"].iterrows():
                    # interval = min(flow["interval"], n["in_flows"]["interval"].min())
                    print("%d %d %d %d %d %d %d" % 
                        (flow["fid"], flow["interval"], flow["counts"], len(flow["depend"]), flow["flit"], flow["map_dst"], flow["map_src"]), 
                        file=wf)

                    for fid, ratio in flow["depend"]:
                        # FIXME: the float number leads to precision problem. We just scale down the ratio
                        # a bit to avoid insufficiency, but not consider exceed issues. 
                        print("%d    %.7f" % (fid, ratio * 0.99), file=wf)

    def _dumpFocusTrace(self, traffic):
        dest_dir = os.path.join("buffer/{}".format(gc.taskname))
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)
        trace_file = "trace_{}.json".format(gc.flit_size)
        traffic.to_json(os.path.join(dest_dir, trace_file))