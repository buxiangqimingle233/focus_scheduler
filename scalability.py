from functools import reduce
import os
import yaml
import re


root_dir = os.path.dirname(os.path.abspath(__file__))
global_control_bc = os.path.join(root_dir, "utils", "global_control_bc.py")
global_control = os.path.join(root_dir, "utils", "global_control.py")

benchmark_models = ["wide_resnet50_2", "resnext50_32x4d"]

bias = [33, 30]
allocate_cores = [16, 48]
pipeline_layers = [8, 4]
w_candidate = [1024]

base_allocate_cores = [4, 12]
base_pipeline_layers = [2, 1]

for scale in [4**i for i in range(5)]:
    pipeline_layers = [i * scale for i in base_pipeline_layers]
    allocate_cores = [i * scale for i in base_allocate_cores]



layer_names = ["{}_layer{}".format(benchmark_models[idm], idl+1+bias[idm]) 
                for idm in range(len(benchmark_models)) for idl in range(pipeline_layers[idm])]

layer_names = list(map(lambda x: "\"{}\"".format(x), layer_names))

cores = [str(int(allocate_cores[idm] / pipeline_layers[idm])) for idm in range(len(benchmark_models)) for _ in range(pipeline_layers[idm])]


def modify_gc(w):
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
                    elif re.search(r"array_diameter = .*", line) is not None:
                        print("array_diameter = {}".format(int(sum(allocate_cores) ** 0.5)), file=wf)
                    else:
                        print(line, file=wf)
                elif line:
                    print(line, file=wf)


# run! run! run!
for w in w_candidate:
    modify_gc(w)
    try:
        os.system("python focus.py")
    except Exception as e:
        print("Exception: {} happend".format(e))
