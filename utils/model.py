import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import re
from layer import Layer


class Model():
    layers = []

    def __init__(self, model_dir):
        self.prj_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.database_path = os.path.join(self.prj_root, "db")
        for root, _, files in os.walk(os.path.join(self.database_path, model_dir)):
            for file in files:
                self.layers.append(Layer(layer_file=file, model_dir=model_dir, prj_root=self.prj_root))

    def _allocate_top_level_pe(self):
        mac_cnts = [layer.get_mac_number() for layer in self.layers]
        mac_cnt_ratio = [cnt / sum(mac_cnts) for cnt in mac_cnts]
        obj, _ = self.layers[0]._get_top_arch_specification()
        top_level_name = obj["architecture"]["subtree"][0]["subtree"][0]["name"]
        total_top_level_pe_cnt = int(re.search(r"0..(\d+)", top_level_name).group(1)) + 1

        total_top_level_pe_cnt -= len(mac_cnt_ratio)
        top_level_pes = [int(ratio * total_top_level_pe_cnt) + 1 for ratio in mac_cnt_ratio]
        return top_level_pes

    def execute(self, apply_to_comm_bank, collect_result, search_dataflow=False, timeout=False, spatially_compute=False):
        '''Iterate for each layer and invoke `apply_to_comm_bank` at Layer.run\n
            Inputs:\n
                apply_to_comm_bank: a function taking `Layer` and comm_bank as inputs
                collect_result: collecting results for each layer
        '''

        if spatially_compute:
            top_level_pes = self._allocate_top_level_pe()
        else:
            top_level_pes = [None] * len(self.layers)

        res = {}
        for layer, pe_cnt in zip(self.layers, top_level_pes):
            res[layer.layer_name] = layer.run(apply_to_comm_bank, top_level_pe_cnt=pe_cnt,
                                              search_dataflow=search_dataflow, timeout=timeout)
        sys.stdout.flush()
        sys.stderr.flush()
        collect_result(res)


if __name__ == "__main__":
    model = Model("resnet50")
    model.execute(lambda x, y: y, lambda x: x, search_dataflow=True, timeout=600)
