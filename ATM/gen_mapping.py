import numpy as np
import math
import sys
import os

from pandas import array
from compiler import global_control as gc
import matplotlib.pyplot as plt
import seaborn as sns

class gen_mapping():
    
    def __init__(self):
        self.layer_num = 0
        self.layer_tile_num = {}
        self.controller_idx = [0, 0]
        self.array_diameter = gc.array_diameter

        self.mc_idx0 = self.array_diameter//2 - 1
        self.mc_idx1 = self.array_diameter//2
        self.memory_controllers = [
            self.mc_idx0, self.mc_idx1,
            self.mc_idx0 * self.array_diameter, self.mc_idx1 * self.array_diameter,
            (self.mc_idx0 + 1) * self.array_diameter - 1, (self.mc_idx1 + 1) * self.array_diameter - 1,
            (self.array_diameter - 1) * self.array_diameter + self.mc_idx0, (self.array_diameter - 1) * self.array_diameter + self.mc_idx1
        ]
        # self.memory_controllers = []
        # An indicator for allocating memory controllers in round-robin style
        self.pointer = 0

    def _parse_config(self):
        self.tile_array_height = gc.array_diameter
        self.tile_array_width = gc.array_diameter
        self.mapping_style = gc.mapping_style
        self.layer_num = len(gc.layer_names)
        self.layer_tile_num = sorted([(i, gc.cores[i]) for i in range(len(gc.cores))], key=lambda x:x[1], reverse=True)
        if gc.array_size < len(self.memory_controllers) + sum(gc.cores):
            print("Mapping error: allocated cores exceed free cores", file=sys.stderr)
            exit(-1)

    def get_controller(self, memory_controllers, layer):
        self.pointer = (self.pointer + 1) % len(self.memory_controllers)
        return self.memory_controllers[self.pointer]

    def map(self, mapping):
        self._parse_config()

        board = np.full((self.tile_array_height, self.tile_array_width), -1, dtype=np.int)
        mapping_res = mapping
        for i in range(self.tile_array_height):
            for j in range(self.tile_array_width):
                board[i][j] = mapping_res[i * self.tile_array_width + j]

        self.mapping_result = board

        if gc.mapper_verbose:
            fig_name = os.path.join(gc.visualization_root, "mapping.png")
            fig = sns.heatmap(data=self.mapping_result, cmap="RdBu_r", linewidths=0.3, annot=True)
            heatmap = fig.get_figure()
            heatmap.savefig(fig_name, dpi=400)
            plt.close()

        # f = open(f'mapping_vis/self.mapping_result_{self.mapping_style}.dat', 'w')
        # for i in range(self.tile_array_height):
        #     for j in range(self.tile_array_width):
        #         print(i, j, self.mapping_result[i][j], file=f)

        res = {}
        layer_names, cores = gc.layer_names, gc.cores
        is_pipelined = [False] + [layer_names[idl-1][:6] == layer_names[idl][:6] for idl in range(1, len(layer_names))]
        for idl in range(len(layer_names)):
            mapped_core = [x * self.tile_array_width + y for x, y in zip(*np.where(self.mapping_result == idl))]
            
            mapped_core = {i: mapped_core[i] for i in range(cores[idl])}

            # Select the largest tile to be the exit port if you're not the last layer
            if (idl < len(layer_names) - 1) and (is_pipelined[idl+1] is True):
                mapped_out = {-2: mapped_core[max(mapped_core.keys())]}
            else:
                mapped_out = {-2: self.get_controller(self.memory_controllers, self.layer_tile_num[idl])}

            # If pipelined, input is fetched from exit port of last layer, weight is fetched from memory controllers
            # For simplicity, we always allocate a memory controller for each layer, which is denoted as -3
            if is_pipelined[idl]:
            # if False:
                mapped_mc = {-1: res[layer_names[idl-1]][-2],
                             -3: self.get_controller(self.memory_controllers, self.layer_tile_num[idl])}
            else:
                mapped_mc = {-1: self.get_controller(self.memory_controllers, self.layer_tile_num[idl]),
                             -3: self.get_controller(self.memory_controllers, self.layer_tile_num[idl])}

            res[layer_names[idl]] = {**mapped_mc, **mapped_core, **mapped_out}

        return res

if __name__ == "__main__":
    task_mapper = gen_mapping()