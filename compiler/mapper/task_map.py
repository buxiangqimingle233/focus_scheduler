import numpy as np
import math
import copy
from .hilbert import hilbert_map
from utils.global_control import debug_show
from utils import global_control as gc
import seaborn as sns


class ml_mapping():
    controller_idx = [0, 0]
    array_diameter = gc.array_diameter

    mc_idx0 = array_diameter//2 - 1
    mc_idx1 = array_diameter//2
    memory_controllers = [
        mc_idx0, mc_idx1,
        mc_idx0 * array_diameter, mc_idx1 * array_diameter,
        (mc_idx0 + 1) * array_diameter - 1, (mc_idx1 + 1) * array_diameter - 1,
        (array_diameter - 1) * array_diameter + mc_idx0, (array_diameter - 1) * array_diameter + mc_idx1
    ]

    def __init__(self):
        self.layer_num = 0
        self.layer_MACs = {}
        self.layer_tile_num = {}
    
    def parse_config(self):
        self.layer_num = len(gc.layer_names)
        self.layer_MACs = {i: gc.cores[i] for i in range(self.layer_num)}
        self.tile_array_height = gc.array_diameter
        self.tile_array_width = gc.array_diameter
        self.mapping_style = gc.mapping_style

    def cal_tile_num(self):
        # total_MACs = 0
        # for layer_i in self.layer_MACs:
        #     total_MACs += self.layer_MACs[layer_i]

        # for layer_i in self.layer_MACs:
        #     self.layer_tile_num[layer_i] = round((self.layer_MACs[layer_i] * self.tile_array_height * self.tile_array_width) / total_MACs)
            # self.layer_tile_num[layer_i] = (self.layer_MACs[layer_i] * self.tile_array_height * self.tile_array_width) // total_MACs
            # print(f"Layer {layer_i} gets {self.layer_tile_num[layer_i]} tiles")

        # TODO: We do not wish this module to change the sizes of layers
        self.layer_tile_num = copy.copy(self.layer_MACs)

        self.layer_tile_num = sorted(self.layer_tile_num.items(), key=lambda x:x[1], reverse=True)

    def get_controller(self, memory_controllers, layer):
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

    def map(self):
        self.parse_config()
        self.cal_tile_num()

        self.mapping_result = np.full((self.tile_array_height, self.tile_array_width), -1, dtype=np.int)
        mapping_style = gc.mapping_style
        array_diameter = gc.array_diameter
        if mapping_style == "Tetris":
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
                                    self.mapping_result[i : i + height, j : j + width] = layer[0]
                                    print(f'layer{layer[0]} requires: {layer[1]} height: {height} width: {width} latency: {self.layer_MACs[layer[0]] / (height * width)}')
                                    flag = True
                                elif i + width <= self.tile_array_height and j + height <= self.tile_array_width and self.mapping_result[i : i + width, j : j + height].max() == -1:
                                    self.mapping_result[i : i + width, j : j + height] = layer[0]
                                    print(f'layer{layer[0]} requires: {layer[1]} height: {width} width: {height} latency: {self.layer_MACs[layer[0]] / (height * width)}')
                                    flag = True
                    if not flag:
                        width -= 1
                        if width == 0:
                            print(f"{layer[0]} failed")
                            return
                        height = layer[1] // width

        else:
            mapping_list = []
            if self.mapping_style == "Zig-Zag":
                for i in range(array_diameter):
                    for j in range(array_diameter):
                        if i % 2 == 0:
                            mapping_list.append([i, j])
                        else:
                            mapping_list.append([i, array_diameter - 1 - j])

            elif self.mapping_style == "Hilbert":
                mapping_list = list(hilbert_map(np.log2(array_diameter)))

            for layer in self.layer_tile_num:
                tile_num = layer[1]
                while(tile_num > 0):
                    self.mapping_result[mapping_list[0][0]][mapping_list[0][1]] = layer[0]
                    mapping_list.pop(0)
                    tile_num -= 1

        if gc.mapper_verbose:
            print(self.mapping_result)
            fig_name = "mapping_vis/map_result.png"
            fig = sns.heatmap(data=self.mapping_result, cmap="RdBu_r", linewidths=0.3, annot=True)
            heatmap = fig.get_figure()
            heatmap.savefig(fig_name, dpi=400) 

        f = open(f'mapping_vis/self.mapping_result_{self.mapping_style}.dat', 'w')
        for i in range(self.tile_array_height):
            for j in range(self.tile_array_width):
                print(i, j, self.mapping_result[i][j], file=f)

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