import re
import yaml
import argparse
import os
from functools import reduce


class Loop2Map():
    def __init__(self):
        self.keeps = []
        self.tiling = []
        self.spatial_component_sizes = []

    def transform(self, in_file_path, problem_file_path, out_file_path):
        self.__parseTxt(in_file_path)
        self.__exportYaml(problem_file_path, out_file_path)

    def getSpatialComponentSize(self, tile_order):
        try:
            return self.spatial_component_sizes[tile_order]
        except Exception:
            raise Exception("Please transform stats file first!")

    def __parseTxt(self, in_file_path):
        judge_tile_title = re.compile(r"^[^|].*?]\s*$")
        extract_tile_title = re.compile(r"^(\w*).*?$")
        extract_tile_keep = re.compile(r"Weights|Inputs|Outputs")
        judge_dim = re.compile(r"^\|.*$")
        extract_diminfo = re.compile(r"^[^A-Z]*?([A-Z]).*?:([0-9]+).*$")
        with open(in_file_path, 'r') as f:
            line = f.readline()

            while line:
                # remove empty lines
                while line and not judge_tile_title.match(line):
                    line = f.readline()

                if line:
                    title = extract_tile_title.match(line).group(1)
                    keep_datatypes = extract_tile_keep.findall(line)
                    self.keeps.append((title, keep_datatypes))
                # remove divider lines
                while line and not judge_dim.match(line):
                    line = f.readline()

                while line and judge_dim.match(line):
                    match_obj = extract_diminfo.match(line)
                    is_spatial = False
                    if re.search("Spatial", line):
                        is_spatial = True
                    is_spatial_Y = False
                    if re.search("Spatial-Y", line):
                        is_spatial_Y = True
                    self.tiling.append({"tile": title, "name": match_obj.group(1),
                                        "size": int(match_obj.group(2)), "is_spatial": is_spatial, "is_spatial_Y": is_spatial_Y})
                    line = f.readline()

    def __exportYaml(self, problem_file_path, out_file_path):
        prb_root_node = yaml.load(open(problem_file_path), Loader=yaml.FullLoader)
        dimensions = prb_root_node["problem"]["shape"]["dimensions"]

        map_root_node = {"mapping": []}
        for tile, keep in self.keeps:
            temporal_dims = [dim for dim in self.tiling if dim["tile"] == tile and not dim["is_spatial"]]
            spatial_dims = [dim for dim in self.tiling if dim["tile"] == tile and dim["is_spatial"]]

            # Parse temporal dimenstions
            dim_names = [it["name"] for it in temporal_dims]
            permutation = "".join(dim_names + [name for name in dimensions if name not in dim_names])
            permutation = permutation[::-1]

            factors = {i: 1 for i in dimensions}
            dim_sizes = {it["name"]: it["size"] for it in temporal_dims}
            factors.update(dim_sizes)
            factors = " ".join([name + "=" + str(size) for name, size in factors.items()])

            target = tile
            type_ = "temporal"
            tile_yaml_obj = {"target": target, "type": type_, "factors": factors, "permutation": permutation}
            map_root_node["mapping"].append(tile_yaml_obj)


            # Parse spatial dimensions
            dim_names = [it["name"] for it in spatial_dims]
            permutation = "".join(dim_names + [name for name in dimensions if name not in dim_names])
            permutation = permutation[::-1]     # Reverse it

            factors = {i: 1 for i in dimensions}
            dim_sizes = {it["name"]: it["size"] for it in spatial_dims}
            factors.update(dim_sizes)
            factors = " ".join([name + "=" + str(size) for name, size in factors.items()])
            try:
                self.spatial_component_sizes.append(reduce(lambda x, y: x * y, dim_sizes.values()))
            except TypeError:
                self.spatial_component_sizes.append(1)

            target = tile
            type_ = "spatial"
            tile_yaml_obj = {"target": target, "type": type_, "factors": factors, "permutation": permutation}
            try:
                first_spatial_Y = next(dim for dim in spatial_dims if dim["is_spatial_Y"])
                split = permutation.index(first_spatial_Y["name"])
                tile_yaml_obj["split"] = split - 1
            except:
                pass
            map_root_node["mapping"].append(tile_yaml_obj)

            if len(keep) != 3:
                datatype_bank = ["Weights", "Inputs", "Outputs"]
                bypass = [datatype for datatype in datatype_bank if datatype not in keep]
                bypass_yaml_obj = {"target": tile, "type": "bypass", "bypass": bypass}
                map_root_node["mapping"].append(bypass_yaml_obj)

        # print(map_root_node)
        yaml.dump(map_root_node, open(out_file_path, "w"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("p", help="problem specification yaml")
    args = parser.parse_args()

    root = os.path.dirname(os.path.abspath(__file__))
    mapping_dir = root + "/mapping"
    if not os.path.exists(mapping_dir):
        os.makedirs(mapping_dir)
    mapping_file = mapping_dir + "/dump_mapping.yaml"

    # FIXME: specified root
    map_report = root + "/timeloop-mapper.map.txt"

    l2m = Loop2Map()
    l2m.transform(map_report, args.p, mapping_file)
