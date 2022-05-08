import os
import yaml
from compiler import global_control
from gen_benchmark import gen_benchmark

def convert2mapping(mapping):
    mapping_res = []
    for i in range(global_control.array_diameter):
        for j in range(global_control.array_diameter):
            mapping_res.append(-1)
    
    print(mapping_res)

    if i == 0 or j == 0 or i == global_control.array_diameter - 1 or j == global_control.array_diameter - 1:
        return



def get_performance(mapping):
    task_list = "task_list.yaml"
    bm_file = "../benchmark/test.yaml"

    gen_benchmark(task_list, mapping, bm_file)
    global_control.array_diameter = 6
    global_control.mapping = convert2mapping(mapping)

    obj = yaml.load(open(task_list, "r"), Loader=yaml.FullLoader)
    models = list(obj.keys())

    os.chdir(f"{os.getcwd()}/..")
    os.system("python focus.py -bm benchmark/test.yaml -d 6 tes")
    # os.system("python focus.py -bm benchmark/test.yaml -d 6 es")

    os.chdir(f"{os.getcwd()}/simulator")
    # os.system(f"build/bin/spatialsim tasks/resnet50_resnext50_32x4d_vgg16_wide_resnet50_2/spatial_spec")
    os.system(f"build/bin/spatialsim tasks/bert/spatial_spec")




if __name__ == "__main__":
    mapping = [1, 1, 2, 2, 1, 1, 0, 0, 3, 3, 2, 2, 4, 4, 4, 4]
    get_performance(mapping)