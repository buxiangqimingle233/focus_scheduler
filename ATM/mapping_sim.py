import os
import yaml
from gen_benchmark import gen_benchmark

def get_performance(mapping):
    task_list = "task_list.yaml"
    bm_file = "../benchmark/test.yaml"
    gen_benchmark(task_list, mapping, bm_file)

    obj = yaml.load(open(task_list, "r"), Loader=yaml.FullLoader)
    models = list(obj.keys())

    os.chdir(f"{os.getcwd()}/..")
    # os.system("python focus.py -bm benchmark/test.yaml -d 6 tes")
    os.system("python focus.py -bm benchmark/test.yaml -d 6 es")

    os.chdir(f"{os.getcwd()}/simulator")
    os.system(f"build/bin/spatialsim tasks/inception_resnet50_resnext50_32x4d_vgg16/spatial_spec")




if __name__ == "__main__":
    mapping = [1, 1, 2, 2, 1, 1, 0, 0, 3, 3, 2, 2, 4, 4, 4, 4]
    get_performance(mapping)