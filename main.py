import sys
import re
import math
import yaml
import pandas as pd
import numpy as np
import seaborn as sns
from functools import reduce
from random import choice

from utils.latency_model import LatencyModel
from utils.layer import Layer
from utils.latency_model_wrapper import generate, analyze


array_diameter = 15
array_size = array_diameter**2
arch_config = {
    "p": 6, "cp_if": 6, "cp_of": 0, "tr": 1, "ts": 2, "tw": 1,
    "n": array_diameter**2,
    "d": array_diameter,
    "w": 512
}

cv = 2

controller_idx = 0

# Round Robin Style
def get_controller(memory_controllers):
    global controller_idx
    controller_idx = (controller_idx + 1) % len(memory_controllers)
    return memory_controllers[controller_idx]


class WorkingLayerSet():

    def __init__(self, layers, cores, core_map):    
        self.layer_names = layers
        self.cores = cores
        self.model_names = [re.search(r"(.+?)_", layer).group(1) for layer in layers]
        self.result_names = ["result_" + layer + ".yaml" for layer in layers]
        self.prob_spec_names = [layer + ".yaml" for layer in layers]
        self.exchange_file_name = "layer-set-exchange-info.yaml"
        self.core_map = core_map  # locations[layer] = {0:8, 1:9, 2:10, 3:13}
    
    def generate(self):
        exec_info = self.getExecInfo()
        transformed_info = self.transformExecInfo(exec_info)
        collapsed_info = self.collapseExecInfo(transformed_info)
        latency_result = self.invokeEstimator(collapsed_info)
        self.dump2HNOC(collapsed_info)

    def dump2HNOC(self, collapsed_info):
        comm_graph, interval, buffer_access = collapsed_info
        hnocs_dump_file = "trace.dat"
        with open(hnocs_dump_file, "w") as f:
            for comm_graph_per_datatype, interval_per_datatype, buffer_access_per_datatype \
                in zip(comm_graph, interval, buffer_access):

                for single_comm, single_interval, single_buffer_access \
                    in zip(comm_graph_per_datatype["graph"], interval_per_datatype, buffer_access_per_datatype):
                    # src, dst, interval, flits, count
                    print(",".join(map(lambda x: str(x), [single_comm[0], single_comm[1], single_interval, max(single_buffer_access/arch_config["w"], 4), 20])), file=f)

    def getPktSizes(self):
        return self.buffer_access

    def analyze(self):
        latency_result = yaml.load(open(self.exchange_file_name, "r"), Loader=yaml.FullLoader)
        
        slowdown_factors, bound_fractions, achieved_bandwidths, required_bandwidths = [], [], [], []
        result = pd.DataFrame()
        for result_per_core in latency_result:
            waiting_core, is_bound, ach_latency, req_latency, ach_bw, req_bw = zip(*result_per_core)
            ser = pd.Series(
                data=[
                    any(is_bound), max(ach_latency), max(req_latency), sum(ach_bw), sum(req_bw),
                    max([(ach / req) for req, ach in zip(req_latency, ach_latency)]), 
                    waiting_core[0]
                ]
            )
            result = result.append(ser, ignore_index=True)

        result = result.rename({0: "is_bound", 1: "achieved_latency", 2: "required_latency", \
                                3: "achieved_bandwidth", 4: "required_bandwidth", 5: "slow_down", 6:"wating_core"}, axis="columns")

        # refine
        result["is_bound"] = result["is_bound"] > 0
        result = result.groupby("wating_core").agg({"is_bound":"any", "achieved_latency": "max", "required_latency": "max", 
                                                    "achieved_bandwidth": "sum", "required_bandwidth": "sum", "slow_down": "max", })
        
        return result

    def invokeEstimator(self, collapsed_info):

        comm_graph, interval, buffer_access = collapsed_info
        self.comm_graph, self.interval, self.buffer_access = collapsed_info
        res = []
        for comm_graph_per_datatype, interval_per_datatype, buffer_access_per_datatype \
            in zip(comm_graph, interval, buffer_access):

            required_latencies = [x for x, y in \
                zip(interval_per_datatype, buffer_access_per_datatype)]

            required_bandwidths = [x / y for x, y in \
                zip(buffer_access_per_datatype, interval_per_datatype)]


            estimator = LatencyModel()
            waiting_cores = [req[1] for req in comm_graph_per_datatype["graph"]]

            for episode in range(10):
                try:
                    achieved_latencies = estimator.runModel(comm_graph_per_datatype, arch_config)
                    achieved_bandwidths = [data / latency for data, latency in \
                        zip(buffer_access_per_datatype, achieved_latencies)]
                    is_comm_bound = [l > q for l, q in \
                        zip(achieved_latencies, required_latencies)]
                    print("Estimating succeed!, cv: {}".format(comm_graph_per_datatype["cv"]))
                    break
                except Exception as e: 
                    print("CV: {}, raise an exception {}".format(comm_graph_per_datatype["cv"], "e"), file=sys.stderr)
                    comm_graph_per_datatype["cv"] *= 0.8
                    achieved_latencies = [-1] * len(comm_graph_per_datatype["graph"])
                    achieved_bandwidths = [0] * len(comm_graph_per_datatype["graph"])
                    is_comm_bound = [True] * len(comm_graph_per_datatype["graph"])
            
            res.append([list(factor) for factor in \
                zip(waiting_cores, is_comm_bound, achieved_latencies, required_latencies, \
                    achieved_bandwidths, required_bandwidths)])
            # Dump
            res = [list(factor) for factor in zip(*res)]
            yaml.dump(res, open(self.exchange_file_name, "w"))
        return res

    def applyCoreMap(self, layer, graph):
        core_map = self.core_map[layer]
        relocated_graph = [(core_map[req[0]], core_map[req[1]], req[2]) for req in graph]
        return relocated_graph

    def getExecInfo(self, search_dataflow=False, timeout=300):
        # exec_infos[layer]: (comm_graph, interval, access [bits])
        exec_info = []
        for layer, model, result, prob_spec, core in \
            zip(self.layer_names, self.model_names, \
                self.result_names, self.prob_spec_names, self.cores):

            layer = Layer(prob_spec, model_dir=model, dram_spatial_size=core)

            exec_info_per_layer = layer.run_with_gc(embeddedFunc)

            exec_info.append(list(exec_info_per_layer))

        return exec_info

    def transformExecInfo(self, exec_info):
        # FIXME: Hard-coded shape here
        layer_size, datatype_size, struct_size = len(exec_info), 3, 3

        transformed_info = [[[[] for k in range(layer_size)] for j in range(datatype_size)] for i in range(struct_size)]

        for layer_idx in range(len(exec_info)):
            for datatype_idx in range(len(exec_info[layer_idx])):
                for struct_idx in range(len(exec_info[layer_idx][datatype_idx])):
                    try:
                        transformed_info[struct_idx][datatype_idx][layer_idx] = \
                            exec_info[layer_idx][datatype_idx][struct_idx]
                    except IndexError:
                        pass
        
        for struct_idx in range(struct_size):
            for datatype_idx in range(datatype_size):
                for layer_idx in range(layer_size):
                    if not transformed_info[struct_idx][datatype_idx][layer_idx]:
                        raise Exception("No spatial parallelism is specified in layer {}".format(self.layer_names[layer_idx]))
                        pass

        return transformed_info

    def collapseExecInfo(self, exec_info):
        merged_comm_graph, merged_interval, merged_access = [], [], []

        for graph_per_datatype, interval_per_datatype, buffer_access_per_datatype in zip(*exec_info):
            total_cores = sum([len(graph_per_layer["graph"]) for \
                               graph_per_layer in graph_per_datatype if graph_per_layer])
            core_fractions = [len(graph_per_layer) / total_cores for graph_per_layer in graph_per_datatype]
            avg_l = sum([graph_per_layer["avg_l"] * fraction for \
                         graph_per_layer, fraction in zip(graph_per_datatype, core_fractions)])

            graph = reduce(
                lambda x, y: x + y,
                [self.applyCoreMap(layer, graph_per_layer["graph"]) for \
                    layer, graph_per_layer in zip(self.layer_names, graph_per_datatype)]
            )
            merged_comm_graph.append({"avg_l": avg_l, "cv": cv, "graph": graph})

            interval = reduce(lambda x, y: x + y, interval_per_datatype)
            merged_interval.append(interval)

            access = reduce(lambda x, y: x + y, buffer_access_per_datatype)
            merged_access.append(access)

        return (merged_comm_graph, merged_interval, merged_access)


def embeddedFunc(layer: Layer, comm_bank):

    def transform_format(comm_status):
        '''Transform the communication status to a graph\n
            NOTED: buffer in higher hierarchy (GLB & DRAM) are connected to router0 directly
        '''
        # according to "An analytical latency model for networks-on-chip", Table II
        comm_graph, interval_graph, volume_graph = [], [], []
        for datatype in comm_status:
            if not datatype:
                continue
            comm_graph_per_datatype = [(src, dst, request["pkt_rate"]) 
                                        for request in datatype
                                        for (src, dst) in zip(request["srcs"], request["dsts"])]
            interval_graph_per_datatype = [request["pkt_interval"]
                                        for request in datatype
                                        for (src, dst) in zip(request["srcs"], request["dsts"])]
            volume_graph_per_datatype = [math.ceil(request["bit_volume"])
                                        for request in datatype
                                        for (src, dst) in zip(request["srcs"], request["dsts"])]

            def avg(list_): 
                try:
                    return sum(list_) / len(list_)
                except ZeroDivisionError:
                    return 1e-10
            # calculate average packet length
            packet_lengths = [request["bit_volume"] for request in datatype]
            avg_packet_length = avg(packet_lengths)     # bit / packet
        
            
            # calculate coefficient variation
            avg_interval = avg(interval_graph_per_datatype)
            _, _, inject_rates = zip(*comm_graph_per_datatype)
            avg_inject_rate = avg(inject_rates)
            cv = math.sqrt(abs((
                    (avg_interval - avg_packet_length / arch_config["w"]) * avg_inject_rate**2
                    + (avg_packet_length / arch_config["w"]) * (arch_config["w"] / avg_packet_length - avg_inject_rate)**2
                )) / avg_interval
            ) / avg_inject_rate
            
            comm_graph.append({"graph": comm_graph_per_datatype, "cv": cv, "avg_l": avg_packet_length / arch_config["w"]})
            interval_graph.append(interval_graph_per_datatype)
            volume_graph.append(volume_graph_per_datatype)
        
        return comm_graph, interval_graph, volume_graph

    # Get communciation graph
    comm_graphs, intervals, buffer_accesses = zip(*[transform_format(comm_status) \
        for comm_status in comm_bank if any(comm_status)])

    # Do spatial mapping
    comm_graphs = [layer._locate_components(graph_per_tile) for graph_per_tile in comm_graphs]

    res = []
    comm_graph, interval, buffer_access = [], [], []
    if layer.get_dram_tile_spatial_size() > 1:

        # get data from the last tile of each datatype
        for datatype_index in range(3):
            for tile_index in range(len(comm_graphs)-1, -1, -1):
                try:
                    comm_graph.append(comm_graphs[tile_index][datatype_index])
                    interval.append(intervals[tile_index][datatype_index])
                    buffer_access.append(buffer_accesses[tile_index][datatype_index])
                    break
                except IndexError:
                    pass

    return zip(comm_graph, interval, buffer_access)


def plot_heatmap(analysis_result):
    lantencies = analysis_result["achieved_bandwidth"] / analysis_result["required_bandwidth"]
    lantencies = analysis_result["slow_down"]
    # lantencies[lantencies > 10] = 10
    supp = pd.Series([0] * (array_diameter**2 - len(lantencies)))
    lantencies = lantencies.append(supp)

    core_map = lantencies.values.reshape((array_diameter, -1))  
    fig_name = "heatmap.png"
    fig = sns.heatmap(data=core_map, cmap="RdBu_r", linewidths=0.3, center=1, annot=False)
    heatmap = fig.get_figure()
    heatmap.savefig(fig_name, dpi=400)


def plot_dist(pkt_sizes):

    array = np.asfarray(pkt_sizes)

    print("min: {}, avg: {}, max: {}".format(array.min(), array.mean(), array.max()))

    overhead = arch_config["w"] / (array.flatten() + arch_config["w"])
    fig = sns.displot(overhead)
    fig_name = "dist.png"
    fig.savefig(fig_name, dpi=400)


def run():
    layer_names = [
        "resnet50_layer43", "resnet50_layer44", "resnet50_layer45", "resnet50_layer46", \
        "vgg16_layer1", "vgg16_layer2", "vgg16_layer3", "vgg16_layer4", \
        "inception_layer1", "inception_layer2", "inception_layer3", "inception_layer4"
    ]
    cores = [
        16, 16, 16, 16, 
        16, 16, 32, 32,
        4, 4, 4, 4
    ]

    # Allocate memory controllers on the array
    memory_controllers = [
        0, array_diameter-1, 
        array_diameter, 2*array_diameter-1, 
        array_diameter*(array_diameter-2), array_diameter*(array_diameter-1)-1,
        array_diameter*(array_diameter-1), array_diameter*array_diameter-1
    ]

    assert(sum(cores) <= (array_diameter-2) * (array_diameter-1))

    # Setup core mapping
    remain_places = [i for i in range(array_diameter**2) if i not in memory_controllers]
    biases = [sum(cores[:i]) for i in range(len(cores))]

    core_map = {layer: {**{i: remain_places[i + bias] for i in range(core)}, **{-1: get_controller(memory_controllers)}} for \
        layer, bias, core in zip(layer_names, biases, cores)}

    # Instantiate the working set
    layer_set = WorkingLayerSet(layer_names, cores, core_map)

    # Invoke timeloop-model for extracting communication status, and invoke estimator
    # to analyze communication status. Intermediate results are stored in 
    # layer-set-exchange-info.yaml
    layer_set.generate()

    # Read the exchange file and organize it as a dataframe
    result = layer_set.analyze()

    # Plot the heatmap
    plot_heatmap(result)

    # plot the histogram
    plot_dist(layer_set.getPktSizes())


    # avg_slowdown = result["slow_down"].sum() / len(result["slow_down"])
    avg_slowdown = result["slow_down"].max()
    print("max slowdown:", result["slow_down"].max())
    print("bounded fraction:", result["is_bound"].sum())
    bw_util = result["achieved_bandwidth"].sum() / (arch_config["w"] * 160 * 3)
    print("bandwidth utilization", bw_util)
    print("bandwidth requirement", result["required_bandwidth"].sum() / (1024*3))

    return bw_util, avg_slowdown


if __name__ == "__main__": 

    res = pd.DataFrame()
    run()
    # for width in range(2048, 2049, 128):
    #     arch_config["w"] = width
    #     bw_util, avg_slowdown = run()
    #     res = res.append({"width": width, "average slowdown": avg_slowdown, "bandwidth utilization": bw_util}, ignore_index=True)
    res.to_csv("fk.csv")
    print(res)

    # print(result.head(10))
