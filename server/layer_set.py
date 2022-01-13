import sys
import os

from backup.global_control import debug_show
sys.path.append(".")

import re
import pandas as pd
import numpy as np
from copy import deepcopy

from mapper import task_map
from utils.layer import Layer
import utils.global_control as gc
from mapper.spanningtree import SpanningTree
from trace_generator import generator


class WorkingLayerSetDR():
    '''Glue module as the interface for the two trace generation backends
    genTraceFromTimeloop(): invoke timeloop for the traffic trace from real-world workloads. The \
        workload is specified in utils/global_control.py
    getTraceFromTraceGenerator(): invoke trace gererator for dummy traffic traces mixed from various \
        pre-defined data movement patterns. The specificiation of generator lies in utils/global_control.py as well.
    '''
    traffic = pd.DataFrame(columns=["layer", "src", "dst", "interval", "flit", "counts"])

    def __init__(self, layers, cores):
        self.layer_names = layers
        self.cores = cores
        self.model_names = [re.search(r"(^.+)_", layer).group(1) for layer in layers]
        self.result_names = ["result_" + layer + ".yaml" for layer in layers]
        self.prob_spec_names = [layer + ".yaml" for layer in layers]
        self.exchange_file_name = "layer-set-exchange-info.yaml"

    def genTrace(self, backend: str = "timeloop", subdir_name: str = "test"):
        if backend == "timeloop":
            trace_from_df_engine = self._getExecInfo()
        else:
            trace_from_df_engine = generator.gen_trace()

        focus_trace = self._schedule(trace_from_df_engine)
        # dump for focus simulation
        focus_trace.to_json("traceDR.json")
        if gc.simulate_baseline:
            sim_trace = self._convertToSimTrace(focus_trace)
            self._moveToSimFolder(sim_trace, subdir_name)

    def _schedule(self, traffic):
        # Instantiate the task mapper
        task_mapper = task_map.ml_mapping()
        # Generate task mapping
        core_map = task_mapper.map()
        # TODO: Doesn't work, visualize mapping results
        os.system("gnuplot mapper/mapping_vis.gp")
        traffic = self._applyMapping(traffic, core_map)
        # Only for focus
        traffic = self._selectHubNode(traffic)
        traffic = self._genSpanningTree(traffic)
        return traffic

    def _moveToSimFolder(self, trace_file, benchmark_name):
        simulator_folder = os.path.join(gc.booksim_working_path, "benchmark", benchmark_name)
        if not os.path.exists(simulator_folder):
            os.mkdir(simulator_folder)
        trace_in_simulator = os.path.join(simulator_folder, trace_file)
        os.system("mv {} {}".format(trace_file, trace_in_simulator))

    def _getExecInfo(self):
        for layer, model, result, prob_spec, core in \
            zip(self.layer_names, self.model_names, \
                self.result_names, self.prob_spec_names, self.cores):

            layer = Layer(prob_spec, model_dir=model, dram_spatial_size=core)
            traffic_per_layer = layer.run_with_gc(extractTrafficTrace)
            self.traffic = self.traffic.append(traffic_per_layer, ignore_index=True)

        return self.traffic        

    def _applyMapping(self, traffic, core_map):

        src_sel = traffic.apply(lambda x: max(x["src"]) < max(core_map[x["layer"]].keys()), axis=1)
        dst_sel = traffic.apply(lambda x: max(x["dst"]) < max(core_map[x["layer"]].keys()), axis=1)

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

    def _convertToSimTrace(self, traffic):
        df = deepcopy(traffic)

        # The traffic is the flow-centric data structure. We convert it to node-centric 
        # structure to drive the spatial simulator. 
        df = df.explode("map_src").explode("map_dst")
        df.loc[:, "fid"] = range(df.shape[0])

        nodes = [{"nid": i} for i in range(gc.array_size)]
        for n in nodes:
            n["out_flows"] = df[df["map_src"] == n["nid"]]
            n["in_flows"] = df[df["map_dst"] == n["nid"]]
        
        trace_file = "trace_{}.txt".format(gc.flit_size)
        with open(trace_file, "w") as wf:
            for n in nodes:
                print("{} {}".format(n["nid"], n["out_flows"].shape[0]), file=wf)
                for _, flow in n["out_flows"].iterrows():
                    # flow = flow.astype("int").astype("str")
                    if flow["datatype"] == "output":
                        depend = n["in_flows"].shape[0]
                    else:
                        depend = 0
                    print("%d %d %d %d %d %d" % 
                        (flow["interval"], flow["counts"], depend, flow["flit"], flow["map_dst"], flow["map_src"]), 
                        file=wf)

        return trace_file

        # with open(os.path.join(gc.booksim_working_path, "trace_{}.txt".format()), "w") as wf:
        #     for nid in range(gc.array_diameter**2):
        #         flows = df[df["src"] == nid]
        #         print("{} {}".format(nid, flows.shape[0]), file=wf)
        #         for _, f in flows.iterrows():
        #             f = f.astype("int").astype("str")
        #             print(" ".join([f["interval"], f["counts"], f["depend"], \
        #                             f["flit"], f["dst"], f["src"]]), file=wf)

    def _analyzeSimResult(self):
        booksim_out = os.path.join(gc.booksim_working_path, "out.txt")
        booksim_res = pd.read_csv(booksim_out, header=None, names=["id", "mean", "max", "min", "slowdown"], index_col=False)
        # booksim_res = booksim_res.iloc[:array_size, :]
        # booksim_res.loc[booksim_res["mean"].isna(), "mean"] = 1
        # booksim_res.loc[booksim_res["max"].isna(), "max"] = 1

        # max_slowdown = (booksim_res["max"] / self.trace.groupby("dst").agg({"interval": "max"})["interval"])
        # mean_slowdown = (booksim_res["mean"] / self.trace.groupby("dst").agg({"interval": "max"})["interval"])

        # mean_slowdown = mean_slowdown[~(mean_slowdown.isna()) & (mean_slowdown > 1)]

        # if mean_slowdown.shape[0] == 0:
        #     mean_slowdown = pd.Series([1])

        booksim_res.loc[booksim_res["slowdown"] == 0, "slowdown"] = 1
        mean_slowdown = booksim_res["slowdown"].mean()
        mean_delay = booksim_res["mean"].mean()

        with open(os.path.join("focus-final-out", f"booksim_{gc.result_file}"), "a") as wf:
            # print(flit_size, mean_slowdown, mean_delay, file=wf, sep="\t")
            print(mean_slowdown, file=wf)


def extractTrafficTrace(layer: Layer, comm_bank):
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
