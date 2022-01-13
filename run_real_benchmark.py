import os
from utils import global_control as gc
import focus


root_dir = os.path.dirname(os.path.abspath(__file__))
global_control_bc = os.path.join(root_dir, "utils", "global_control_bc.py")
global_control = os.path.join(root_dir, "utils", "global_control.py")

benchmark_modelss = [
    ["wide_resnet50_2", "resnext50_32x4d", "resnet50", "vgg16"],
    ["bert", "bert"],
    ["unet", "resnet50", "bert-large", "ssd_r34"],
    ["unet", "vgg16", "mnasnet", "inception"]
]

biass = [
    [0, 0, 0, 0],
    [0, 33],
    [0, 0, 0, 0],
    [0, 0, 0, 0]
]

allocate_coress = [
    [64, 64, 64, 64],
    [66, 80],
    [64, 64, 64, 64],
    [128, 64, 32, 32]
]

pipeline_layerss = [
    [4, 4, 8, 4],
    [33, 40],
    [8, 4, 32, 4],
    [16, 4, 4, 8]
]

# w_candidate = range(512, 4097, 512)
w_candidate = range(1024, 1025)
search_dataflow = False

def run():
    global search_dataflow

    gc.trace_gen_backend = "timeloop"

    for w in w_candidate:
        cnt = 0
        for benchmark_models, allocate_cores, bias, pipeline_layers in zip(benchmark_modelss, allocate_coress, biass, pipeline_layerss):
            # FIXME: for debugging
            cnt += 1
            if cnt != 4:
                continue

            # calculate names of layers
            layer_names = ["{}_layer{}".format(benchmark_models[idm], idl+1+bias[idm]) 
                    for idm in range(len(benchmark_models)) for idl in range(pipeline_layers[idm])]
            cores = [int(allocate_cores[idm] / pipeline_layers[idm]) for idm in range(len(benchmark_models)) for _ in range(pipeline_layers[idm])]

            # update global control infos
            gc.search_dataflow = search_dataflow
            gc.dump_comm_status = search_dataflow
            gc.layer_names = layer_names
            gc.cores = cores
            gc.flit_size = w
            gc.result_file = "slowdown_{}.csv".format("_".join(benchmark_models))

            # invoke focus engine
            focus.run()

        search_dataflow = False

if __name__ == "__main__":
    run()