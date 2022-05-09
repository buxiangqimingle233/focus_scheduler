import datetime
import random
import unittest
import numpy as np
import yaml
import os

from compiler import global_control as gc
import ATM.genetic as genetic
from ATM.gen_benchmark import gen_benchmark

def parse_performance(log_file):
    with open(log_file, 'r') as f:
        lines = f.readlines()
        last_line = lines[-1]
        if last_line.split(" ")[0] == "":
            return 30000
        return int(last_line.split(" ")[0])

def convert2mapping(mapping):
    mapping_res = []
    for i in range(gc.array_diameter):
        for j in range(gc.array_diameter):
            mapping_res.append(-1)
    
    for i in range(gc.array_diameter):
        for j in range(gc.array_diameter):
            if i == 0 or j == 0 or i == gc.array_diameter - 1 or j == gc.array_diameter - 1:
                mapping_res[i * gc.array_diameter + j] = -1
            else:
                mapping_res[i * gc.array_diameter + j] = mapping[(i - 1) * (gc.array_diameter - 2) + (j - 1)]
    return mapping_res

def get_fitness(mapping_res, target):
    return -1 * get_performance(mapping_res)
    # get_performance(mapping_res)
    # return sum(1 for expected, actual in zip(mapping_res, target) if expected == actual)
    
def display(candidate, startTime):
    timeDiff = datetime.datetime.now() - startTime
    print("{}\t{}\t{}".format(
        candidate.Genes, -1 * candidate.Fitness, timeDiff))

def get_performance(mapping):
    task_list = "./ATM/task_list.yaml"
    bm_file = "./benchmark/test.yaml"
    mapping_file = "./ATM/mapping.npy"
    log_file = "./simulator/out.log"

    array_diameter = 6

    print(f"Current Mapping: {mapping}")
    for i in range(0, 5):
        if i not in mapping:
            return 30000

    gen_benchmark(task_list, mapping, bm_file)
    gc.array_diameter = array_diameter

    mf = np.array(convert2mapping(mapping))
    np.save(mapping_file, mf)

    obj = yaml.load(open(task_list, "r"), Loader=yaml.FullLoader)
    models = list(obj.keys())
    os.system(f"python focus.py -bm benchmark/test.yaml -d {gc.array_diameter} tes")
    # os.system("python focus.py -bm benchmark/test.yaml -d 6 tes > out.log")

    os.chdir(f"{os.getcwd()}/simulator")
    os.system(f"build/bin/spatialsim tasks/bert/spatial_spec > ../{log_file}")
    # os.system(f"build/bin/spatialsim tasks/resnet50_resnext50_32x4d_vgg16_wide_resnet50_2/spatial_spec > ../{log_file}")
    os.chdir(f"{os.getcwd()}/..")
    # os.system(f"build/bin/spatialsim tasks/bert/spatial_spec")
    return parse_performance(log_file)

class GuessmappingTests():
    geneset = []
    for i in range(0, 5):
        geneset.append(i)

    def test_Hello_World(self):
        target = list(-1 for i in range(16))
        # target = [1, 1, 2, 2, 1, 1, 0, 0, 3, 3, 2, 2, 4, 4, 4, 4]
        # 1 1 2 2
        # 1 1 0 0
        # 3 3 2 2
        # 4 4 4 4
        self.guess_mapping(target)

    def guess_mapping(self, target):
        startTime = datetime.datetime.now()

        def fnGetFitness(genes):
            return get_fitness(genes, target)

        def fnDisplay(candidate):
            display(candidate, startTime)

        # optimalFitness = len(target)
        optimalFitness = 0
        best = genetic.get_best(fnGetFitness, len(target), optimalFitness,
                                self.geneset, fnDisplay)

if __name__ == '__main__':
    gpt = GuessmappingTests()
    gpt.test_Hello_World()