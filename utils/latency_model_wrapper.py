import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
import math
import pandas as pd
import yaml
from pprint import PrettyPrinter
from utils.model import Model
from utils.layer import Layer
from utils.latency_model import LatencyModel


def evaluate_top_level_comm(layer: Layer, comm_bank):

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
                    (avg_interval - avg_packet_length / flit_size) * avg_inject_rate**2
                    + (avg_packet_length / flit_size) * (flit_size / avg_packet_length - avg_inject_rate)**2
                )) / avg_interval
            ) / avg_inject_rate
            
            cv = 0
            comm_graph.append({"graph": comm_graph_per_datatype, "cv": cv, "avg_l": avg_packet_length / flit_size})
            interval_graph.append(interval_graph_per_datatype)
            volume_graph.append(volume_graph_per_datatype)
        
        return comm_graph, interval_graph, volume_graph

    # Get architecture configuration
    obj, _ = layer._get_top_arch_specification()
    top_tile_obj = obj["architecture"]["subtree"][0]["subtree"][0]

    tile_name = top_tile_obj["name"]
    tile_component_cnt = int(re.search(r"0..(\d+)", tile_name).group(1)) + 1
    # array_diameter = math.ceil(math.sqrt(tile_component_cnt))
    array_diameter = math.ceil(math.sqrt(512))

    array_size = array_diameter**2
    # print(array_size)
    network_bandwidth = obj["architecture"]["subtree"][0]["subtree"][0]["local"][0]["attributes"]["width"]
    arch_config = {
        "p": 6, "cp_if": 6, "cp_of": 0, "tr": 1, "ts": 2, "tw": 1,
        "n": array_diameter**2,
        "d": array_diameter,
        "w": network_bandwidth,
    }

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

    # return zip(comm_graph, interval, buffer_access)
        # Invoke NoC estimator for data transmission latencies
        for comm_graph_per_datatype, interval_per_datatype, buffer_access_per_datatype \
            in zip(comm_graph, interval, buffer_access):

            required_latencies = [x for x, y in \
                zip(interval_per_datatype, buffer_access_per_datatype)]

            required_bandwidths = [x / y for x, y in \
                zip(buffer_access_per_datatype, interval_per_datatype)]


            estimator = LatencyModel()
            try:
                achieved_latencies = estimator.runModel(comm_graph_per_datatype, arch_config)
                achieved_bandwidths = [data / latency for data, latency in \
                    zip(buffer_access_per_datatype, achieved_latencies)]
                is_comm_bound = [l > q for l, q in \
                    zip(achieved_latencies, required_latencies)]
            except Exception: 
                achieved_latencies = [-1] * len(comm_graph_per_datatype)
                achieved_bandwidths = [0] * len(comm_graph_per_datatype)
                is_comm_bound = [True] * len(comm_graph_per_datatype)

            res.append([list(factor) for factor in \
                zip(is_comm_bound, achieved_latencies, required_latencies, \
                    achieved_bandwidths, required_bandwidths)])
    
    return res

def generate(prob_file_name, result_file_name, model_name, \
             search_dataflow=False, minimum_logsize=0, maximum_logsize=9, \
             timeout=300):
    # backup[top_level_pe][datatype]
    backups = []
    for top_level_pe in [2**i for i in range(minimum_logsize, maximum_logsize + 1)]:
        print("================== WORKING FOR SIZE {} ======================\n".format(top_level_pe))
        try:
            layer = Layer(prob_file_name, model_dir=model_name, dram_spatial_size=top_level_pe)
            results = layer.run(evaluate_top_level_comm, top_level_pe_cnt=top_level_pe, \
                search_dataflow=search_dataflow, timeout=timeout)
            backups.append(results)
            yaml.dump(backups, open(result_file_name, "w"))
        except Exception as e:  
            print("ERROR: Exception {} happens, while we keep processing ...".format(e))
    return backups

def analyze(prob_file_name, result_file_name, model_dir, minimum_logsize=0, maximum_logsize=9):
    result_obj = yaml.load(open(result_file_name, "r"), Loader=yaml.FullLoader)
    # result_obj = yaml.load(open("vgg16layer5_256bit.yaml", "r"), Loader=yaml.FullLoader)

    cycles = []
    for top_level_pe in [2**i for i in range(minimum_logsize, maximum_logsize + 1)]:
        layer = Layer(prob_file_name, model_dir=model_dir, dram_spatial_size=top_level_pe)
        cycle = layer.get_cycles()
        cycles.append(cycle)

    slowdown_factors, bound_fractions, a_bds, r_bds = [], [], [], []

    for result_per_topsize in result_obj:
        slowdown, bound, a_bds_per_topsize, r_bds_per_topsize = 1, 0, [], []
        for result_per_datatype in result_per_topsize:
            is_bounds, a_l, r_l, a_bd, r_bd = zip(*result_per_datatype)

            bound += is_bounds.count(True) / len(is_bounds) * 0.33
            try:
                slowdown = max([max(1, achieved / required) \
                    for achieved, required in zip(a_l, r_l)] + [slowdown])
            except ZeroDivisionError:
                print(a_l, r_l)
                exit(-1)

            a_bds_per_topsize.append(a_bd)
            r_bds_per_topsize.append(r_bd)

        a_bds.append([sum(component) for component in zip(*a_bds_per_topsize)])
        r_bds.append([sum(component) for component in zip(*r_bds_per_topsize)])
        slowdown_factors.append(slowdown)
        bound_fractions.append(bound)

    # return cycles, bound_fractions

    baseline = cycles[0]
    relative_latencies = [(slow_down * cycle) / baseline for slow_down, cycle in zip(slowdown_factors, cycles)]

    return relative_latencies, bound_fractions, a_bds, r_bds


if __name__ == "__main__":
    
    df = pd.DataFrame(index=[2**i for i in range(10)])

    for i in [44]:
        layer_name = "resnet50_layer" + str(i)
        model_name = "resnet50"
        result_file_name = "result_" + layer_name + ".yaml"
        prob_file_name = layer_name + ".yaml"

        # generate
        generate(prob_file_name, result_file_name, model_name, False)

        # analyze
        relative_latency, comm_bound_fraction, achieved_bandwidth, required_bandwidth = \
            analyze(prob_file_name, result_file_name, model_name)

        # store
        # df[layer_name] = [sum(bd_per_tile) for bd_per_tile in required_bandwidth]
        df[layer_name] = required_bandwidth

        # print
        pp = PrettyPrinter(indent=2)
        print("*********** Analysis Results for {} ***************".format(layer_name))
        pp.pprint({"Relative latencies": relative_latency, 
               "Communication bound fractions": comm_bound_fraction,
               "Achieved total bandwidth": achieved_bandwidth,
               "Required bandwidth": required_bandwidth})

    df.to_csv("relative_latency.csv", index=False)
