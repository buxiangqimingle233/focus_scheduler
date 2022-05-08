import os
import yaml
from compiler import global_control
from ATM.gen_benchmark import gen_benchmark
import numpy as np

def convert2mapping(mapping):
    mapping_res = []
    for i in range(global_control.array_diameter):
        for j in range(global_control.array_diameter):
            mapping_res.append(-1)
    
    for i in range(global_control.array_diameter):
        for j in range(global_control.array_diameter):
            if i == 0 or j == 0 or i == global_control.array_diameter - 1 or j == global_control.array_diameter - 1:
                mapping_res[i * global_control.array_diameter + j] = -1
            else:
                mapping_res[i * global_control.array_diameter + j] = mapping[(i - 1) * (global_control.array_diameter - 2) + (j - 1)]
    return mapping_res
    # print(mapping_res)



def get_performance(mapping):
    task_list = "./ATM/task_list.yaml"
    bm_file = "./benchmark/test.yaml"
    mapping_file = "./ATM/mapping.npy"

    gen_benchmark(task_list, mapping, bm_file)
    global_control.array_diameter = 6
    global_control.mapping = convert2mapping(mapping)

    mf = np.array(convert2mapping(mapping))
    np.save(mapping_file, mf)

    obj = yaml.load(open(task_list, "r"), Loader=yaml.FullLoader)
    models = list(obj.keys())
    # os.system("python focus.py -bm benchmark/test.yaml -d 6 tes")
    os.system("python focus.py -bm benchmark/test.yaml -d 6 es")

    os.chdir(f"{os.getcwd()}/simulator")
    # os.system(f"build/bin/spatialsim tasks/resnet50_resnext50_32x4d_vgg16_wide_resnet50_2/spatial_spec")
    os.system(f"build/bin/spatialsim tasks/bert/spatial_spec")

if __name__ == "__main__":
    mapping = [1, 1, 2, 2, 1, 1, 0, 0, 3, 3, 2, 2, 4, 4, 4, 4]
    get_performance(mapping)