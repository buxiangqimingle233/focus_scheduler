from re import L
import sys
import configparser as cp
import numpy as np
import math

from utils.global_control import *

class ml_mapping():
    controller_idx = 0

    memory_controllers = [
        array_size-1, array_size-2
    ]

    def __init__(self):
        self.layer_num = 0
        self.layer_MACs = {}
        self.layer_tile_num = {}
    
    def parse_config(self):
        config = cp.ConfigParser()
        config.read(conf_filename)

        self.layer_num = int(config.get("layer_config", "layer_num"))
        layer_MACs = config.get("layer_config", "layer_MACs").split(':')

        for i in range(self.layer_num):
            self.layer_MACs[i] = int(layer_MACs[i])

        # The first row is mapped as memory controllers
        self.tile_array_height = int(config.get("accelerator_config", "tile_array_height")) - 1
        self.tile_array_width = int(config.get("accelerator_config", "tile_array_width"))

    def cal_tile_num(self):
        total_MACs = 0
        for layer_i in self.layer_MACs:
            total_MACs += self.layer_MACs[layer_i]
        
        for layer_i in self.layer_MACs:
            self.layer_tile_num[layer_i] = round((self.layer_MACs[layer_i] * self.tile_array_height * self.tile_array_width) / total_MACs)
            # self.layer_tile_num[layer_i] = (self.layer_MACs[layer_i] * self.tile_array_height * self.tile_array_width) // total_MACs
            # print(f"Layer {layer_i} gets {self.layer_tile_num[layer_i]} tiles")

        # TODO: We do not wish this module to change the sizes of layers
        self.layer_tile_num = self.layer_MACs

        self.layer_tile_num = sorted(self.layer_tile_num.items(), key=lambda x:x[1], reverse=True)

    def get_controller(self, memory_controllers):
        self.controller_idx = (self.controller_idx + 1) % len(memory_controllers)
        return memory_controllers[self.controller_idx]

    def map(self):
        self.parse_config()
        self.cal_tile_num()

        mapping_result = np.full((self.tile_array_height, self.tile_array_width), -1, dtype=np.int)

        for layer in self.layer_tile_num:
            height = int(math.sqrt(layer[1]))
            width = math.ceil(math.sqrt(layer[1]))
            if width * (width + height) < 2 * layer[1]:
                height = width

            flag = False
            while(not flag):
                for i in range(self.tile_array_height):
                    for j in range(self.tile_array_width):
                        if not flag:
                            # print(height, width)
                            if i + height <= self.tile_array_height and j + width <= self.tile_array_width and mapping_result[i : i + height, j : j + width].max() == -1:
                                mapping_result[i : i + height, j : j + width] = layer[0]
                                print(f'layer{layer[0]} requires: {layer[1]} height: {height} width: {width} latency: {self.layer_MACs[layer[0]] / (height * width)}')
                                flag = True
                            elif i + width <= self.tile_array_height and j + height <= self.tile_array_width and mapping_result[i : i + width, j : j + height].max() == -1:
                                mapping_result[i : i + width, j : j + height] = layer[0]
                                print(f'layer{layer[0]} requires: {layer[1]} height: {width} width: {height} latency: {self.layer_MACs[layer[0]] / (height * width)}')
                                flag = True
                if not flag:
                    width -= 1
                    if width == 0:
                        print(f"{layer[0]} failed")
                        return
                    height = layer[1] // width

        if mapper_verbose:
            print(mapping_result)

        f = open('mapping_result.dat', 'w')
        for i in range(self.tile_array_height):
            for j in range(self.tile_array_width):
                print(i, j, mapping_result[i][j], file=f)

        res = {}
        for idl in range(len(layer_names)):
            mapped_mc = {-1: self.get_controller(self.memory_controllers)}
            mapped_core = [x * self.tile_array_width + y for x, y in zip(*np.where(mapping_result == idl))]
            mapped_core = {i: mapped_core[i] for i in range(cores[idl])}

            res[layer_names[idl]] = {**mapped_mc, **mapped_core}
        
        return res

if __name__ == '__main__':
    ml_m = ml_mapping()
    print(ml_m.map())