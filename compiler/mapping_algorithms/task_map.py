import numpy as np
import math
import os
import sys

from .hilbert import hilbert_map
from compiler import global_control as gc
import matplotlib.pyplot as plt
import seaborn as sns

class ml_mapping():


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

        x_sum = 0
        y_sum = 0
        array_diameter = self.array_diameter
        for i in range(array_diameter):
            for j in range(array_diameter):
                if self.mapping_result[i][j] == layer[0]:
                    x_sum += i
                    y_sum += j
        x_sum /= layer[1]
        y_sum /= layer[1]

        mc_distance = 100
        for mc in self.memory_controllers:
            mc_index = [mc // array_diameter, mc % array_diameter]
            if abs(x_sum - mc_index[0]) + abs(y_sum - mc_index[1]) < mc_distance:
                self.controller_idx = [mc_index[0], mc_index[1]]
                mc_distance = abs(x_sum - mc_index[0]) + abs(y_sum - mc_index[1])
        return self.controller_idx[0] * array_diameter + self.controller_idx[1]

    def map_tetris(self, mapping_result):
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
                            if i + height <= self.tile_array_height and j + width <= self.tile_array_width and self.mapping_result[i : i + height, j : j + width].max() == -1:
                                mapping_result[i : i + height, j : j + width] = layer[0]
                                print(f'layer{layer[0]} requires: {layer[1]} height: {height} width: {width} latency: {self.layer_tile_num[layer[0]] / (height * width)}')
                                flag = True
                            elif i + width <= self.tile_array_height and j + height <= self.tile_array_width and self.mapping_result[i : i + width, j : j + height].max() == -1:
                                mapping_result[i : i + width, j : j + height] = layer[0]
                                print(f'layer{layer[0]} requires: {layer[1]} height: {width} width: {height} latency: {self.layer_tile_num[layer[0]] / (height * width)}')
                                flag = True
                if not flag:
                    width -= 1
                    if width == 0:
                        print(f"{layer[0]} failed")
                        return
                    height = layer[1] // width
        return mapping_result

    def map_hilbert(self, mapping_result):
        mapping_list = list(hilbert_map(np.log2(self.array_diameter)))

        for lid, core_num in self.layer_tile_num:
            for _ in range(core_num):
                x, y = mapping_list.pop(0)
                while x * self.array_diameter + y in self.memory_controllers:
                    x, y = mapping_list.pop(0)
                mapping_result[x][y] = lid

        return mapping_result

    def map_zigzag(self, mapping_result):
        mapping_list = []
        for i in range(self.array_diameter):
            for j in range(self.array_diameter):
                if i % 2 == 0:
                    mapping_list.append([i, j])
                else:
                    mapping_list.append([i, self.array_diameter - 1 - j])

        for lid, core_num in self.layer_tile_num:
            for _ in range(core_num):
                x, y = mapping_list.pop(0)
                while x * self.array_diameter + y in self.memory_controllers:
                    x, y = mapping_list.pop(0)
                mapping_result[x][y] = lid
        return mapping_result


    def map(self):
        self._parse_config()

        board = np.full((self.tile_array_height, self.tile_array_width), -1, dtype=np.int)

        if self.mapping_style == "Tetris":
            self.mapping_result = self.map_tetris(board)
        elif self.mapping_style == "Zig-Zag":
            self.mapping_result = self.map_zigzag(board)
        elif self.mapping_style == "Hilbert":
            self.mapping_result = self.map_hilbert(board)

        if gc.mapper_verbose:
            fig_name = os.path.join(gc.visualization_root, "mapping.png")
            fig = sns.heatmap(data=self.mapping_result, cmap="RdBu_r", linewidths=0.3, annot=True)
            heatmap = fig.get_figure()
            heatmap.savefig(fig_name, dpi=400)
            plt.close()

        # TODO: This does not work 
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

if __name__ == '__main__':
    ml_m = ml_mapping()
    print(ml_m.map())