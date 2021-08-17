import yaml
from functools import reduce


def get_mac_number(self):
    obj = yaml.load(open(self.prob_specs, "r"), Loader=yaml.FullLoader)

    # mac_cnt = multiplicative of output dims and weight dims
    weight_obj = obj["problem"]["shape"]["data-spaces"][0]
    output_obj = obj["problem"]["shape"]["data-spaces"][2]

    if not output_obj["name"] == "Outputs" and weight_obj["name"] == "Weight":
        raise Exception("Datatypes in problem spec misordered!")

    def extract_dim_names(datatype): return [wrapper[0][0] for wrapper in datatype["projection"]]
    weight_dim_names, output_dim_names = extract_dim_names(weight_obj), extract_dim_names(output_obj)
    mac_related_dim_names = list(set(weight_dim_names + output_dim_names))

    mac_related_dim_sizes = [obj["problem"]["instance"][dim_name] for dim_name in mac_related_dim_names]
    mac_cnt = reduce(lambda x, y: x * y, mac_related_dim_sizes)
    return mac_cnt
