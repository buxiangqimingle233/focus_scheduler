from functools import reduce
import os
import yaml
import re


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
    [132, 120],
    [64, 64, 64, 64],
    [128, 64, 32, 32]
]

pipeline_layerss = [
    [4, 4, 8, 4],
    [33, 40],
    [8, 4, 32, 4],
    [16, 4, 4, 8]
]

# # # just for search
# benchmark_modelss = [
#     ["unet"]
# ]

# biass = [
#     [8]
# ]

# allocate_coress = [
#     [64]
# ]

# pipeline_layerss = [
#     [8]
# ]

w_candidate = range(256, 2049, 256)
# w_candidate = [256]
# w_candidate = [2**i for i in range(6, 8)]
# w_candidate = [32, 64, 128]
search = False

if __name__ == "__main__":

    # run! run! run!
    for w in w_candidate:
        cnt = 0
        for benchmark_models, allocate_cores, bias, pipeline_layers in zip(benchmark_modelss, allocate_coress, biass, pipeline_layerss):
            cnt += 1
            if cnt != 4:
                continue
            layer_names = ["{}_layer{}".format(benchmark_models[idm], idl+1+bias[idm]) 
                            for idm in range(len(benchmark_models)) for idl in range(pipeline_layers[idm])]

            layer_names = list(map(lambda x: "\"{}\"".format(x), layer_names))

            cores = [str(int(allocate_cores[idm] / pipeline_layers[idm])) for idm in range(len(benchmark_models)) for _ in range(pipeline_layers[idm])]

            with open(global_control_bc, "r") as rf:
                with open(global_control, "w") as wf:
                    for line in rf:
                        if line[0] != "#":
                            if re.search(r"layer_names = .*", line) is not None:
                                print("layer_names = [{}]".format(", ".join(layer_names)), file=wf)
                            elif re.search(r"cores = .*", line) is not None:
                                print("cores = [{}]".format(", ".join(cores)), file=wf)
                            elif re.search("\s\"w\":.*?", line) is not None:
                                line = re.sub("[0-9]+", str(w), line)
                                print(line, file=wf)
                            elif re.search(r"slowdown_result.*?", line) is not None:
                                line = "slowdown_result = \"slowdown_{}.csv\"".format("_".join(benchmark_models))
                                print(line, file=wf)
                            elif re.search(r"search_dataflow = .*", line) is not None:
                                if search:
                                    print("search_dataflow = {}".format("True"), file=wf)
                                else:
                                    print("search_dataflow = {}".format("False"), file=wf)
                            elif re.search(r"dump_comm_status = .*", line) is not None:
                                if search:
                                    print("dump_comm_status = {}".format("True"), file=wf)
                                else:
                                    print("dump_comm_status = {}".format("False"), file=wf)
                            else:
                                print(line, file=wf)
                        elif line:
                            print(line, file=wf)

            try:
                os.system("python focus.py")
            except Exception as e:
                print("Exception: {} happend".format(e))

        search = False


# # --------------------- scalablility -----------------------------------
# benchmark_modelss = [
#     ["wide_resnet50_2", "resnext50_32x4d"],
#     ["wide_resnet50_2", "resnext50_32x4d"],
#     ["wide_resnet50_2", "resnext50_32x4d"],
#     ["wide_resnet50_2", "resnext50_32x4d"],
#     ["wide_resnet50_2", "resnext50_32x4d"],
#     ["wide_resnet50_2", "resnext50_32x4d"],
# ]

# biass = [
#     [33, 30] for _ in benchmark_modelss
# ]

# base_cores = [16, 48]

# allocate_coress = [
#     [base_cores[0] * i, base_cores[1] * i] for i in [2**i for ]
# ]

# pipeline_layerss = [
#     [8, 4] for _ in benchmark_models
# ]

# benchmark 1: resnet50 + vgg16
# benchmark_models = ["resnet50", "vgg16"]
# bias = [40, 8]
# allocate_cores = [32, 32]
# pipeline_layers = [2, 4]

# benchmark 2: wide_resnet50_2 + resnext50
# benchmark_models = ["wide_resnet50_2", "resnext50_32x4d"]
# allocate_cores = [16, 48]
# pipeline_layers = [8, 4]
# bias = [33, 30]

# benchmark 3: vgg16
# benchmark_models = ["vgg16"]
# bias = [0]
# allocate_cores = [64]
# pipeline_layers = [16]

# benchmark 4: 宇宙无敌终极象拔蚌之大杂烩

# benchmark_models = ["mnasnet", "vgg16", "resnet50", "inception", "alexnet", "wide_resnet50_2"]
# bias = [5 for _ in benchmark_models]
# pipeline_layers = [1 for _ in benchmark_models]
# allocate_cores = [4, 16, 16, 8, 4, 16]
