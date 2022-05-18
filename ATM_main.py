import os
import yaml
from compiler import global_control as gc
from ATM.gen_benchmark import gen_benchmark
import numpy as np

def convert2mapping(mapping):
    mapping_res = []
    for i in range(gc.array_diameter):
        for j in range(gc.array_diameter):
            mapping_res.append(-1)
    
    # for i in range(gc.array_diameter):
    #     for j in range(gc.array_diameter):
    #         if i == 0 or j == 0 or i == gc.array_diameter - 1 or j == gc.array_diameter - 1:
    #             mapping_res[i * gc.array_diameter + j] = -1
    #         else:
    #             mapping_res[i * gc.array_diameter + j] = mapping[(i - 1) * (gc.array_diameter - 2) + (j - 1)]

    for i in range(gc.array_diameter):
        for j in range(gc.array_diameter):
            if i == 0:
                mapping_res[i * gc.array_diameter + j] = -1
            else:
                mapping_res[i * gc.array_diameter + j] = mapping[(i - 1) * gc.array_diameter + j]
    print(mapping_res)

    return mapping_res

def parse_performance(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        if last_line.split(" ")[0] == "":
            return 30000
        return int(last_line.split(" ")[0])

def get_performance(mapping):
    task_list = "./ATM/task_list.yaml"
    bm_file = "./benchmark/test.yaml"
    mapping_file = "./ATM/mapping.npy"
    log_file = "./simulator/out.log"
    array_diameter = 6

    for i in range(0, 5):
        if i not in mapping:
            return 30000

    gen_benchmark(task_list, mapping, bm_file)
    gc.array_diameter = array_diameter

    mf = np.array(convert2mapping(mapping))
    # mf = np.array(mapping)
    np.save(mapping_file, mf)

    obj = yaml.load(open(task_list, "r"), Loader=yaml.FullLoader)
    models = list(obj.keys())
    os.system(f"python focus.py -bm benchmark/test.yaml -d {gc.array_diameter} es")
    # os.system("python focus.py -bm benchmark/test.yaml -d 6 tes > out.log")

    os.chdir(f"{os.getcwd()}/simulator")
    # os.system(f"build/bin/spatialsim tasks/resnet50_resnext50_32x4d_vgg16_wide_resnet50_2/spatial_spec > ../{log_file}")
    # os.system(f"build/bin/spatialsim tasks/resnet50_resnext50_32x4d_vgg16_wide_resnet50_2/spatial_spec")
    os.system(f"build/bin/spatialsim tasks/bert/spatial_spec > ../{log_file}")
    # os.system(f"build/bin/spatialsim tasks/bert/spatial_spec")
    os.chdir(f"{os.getcwd()}/..")
    return parse_performance(log_file)

if __name__ == "__main__":
    # mapping = [1, 1, 1, 1, 2, 2, 2, 0, 2, 3, 0, 3, 4, 4, 4, 4]
    mapping = [3, 1, 0, 2, 4, 1, 3, 0, 2, 4, 3, 0, 4, 1, 0, 1, 4, 3, 0, 2, 1, 4, 2, 3, 0, 2, 3, 4, 4, 0]
    # mapping = []
    # for i in range(5):
    #     for _ in range(32):
    #         mapping.append(i)
    print(f"*** Performance: {get_performance(mapping)} cycles ***")