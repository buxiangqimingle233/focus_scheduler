import os
import yaml
from functools import reduce

def gen_benchmark(task_list, mapping_res, bm_file):
    obj = yaml.load(open(task_list, "r"), Loader=yaml.FullLoader)
    # print(obj)
    models = list(obj.keys())
    layer_names, cores = [], []
    for model in obj.values():
        layer_names += reduce(lambda x, y: x + y, map(lambda x: list(x.keys()), model))
        cores += reduce(lambda x, y: x + y, map(lambda x: list(x.values()), model))
    for task_id in mapping_res:
        cores[task_id] += 1

    for model in obj.values():
        for layers in model:
            for layer_name in layers.keys():
                layers[layer_name] = cores[layer_names.index(layer_name)]
    # print(obj)
    benchmark_file = open(bm_file, "w")
    yaml.dump(obj, benchmark_file)


if __name__ == "__main__":
    task_list = "task_list.yaml"
    bm_file = "../benchmark/test.yaml"
    mapping_res = [1, 1, 2, 2, 1, 1, 0, 0, 2, 3, 2, 2, 4, 4, 4, 4]
    # 
    # 1 1 2 2
    # 1 1 0 0
    # 3 3 2 2
    # 4 4 4 4
    gen_benchmark(task_list, mapping_res, bm_file)