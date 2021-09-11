# - We just focus on spatial / temporal reuse states of input activations 
# - Spatial reuse is also called `` replication '' 
# - We keep the task unchanged ? 
# - 128 TOPs per core ( 64 MACs under 1 GHz )
# - Parallel dimension order: kernel (R & S), input channel (M), output channel (C), pixel (X & Y)

import numpy as np
import pandas as pd
import seaborn as sns
from math import log2
import copy
import yaml
import re
import os

tile = {
    "ic": 4 * 1,
    "oc": 4 * 4,
    "r": 1,
    "s": 3,
    "x": 4,
    "y": 8
}

job = {
    "oc": 256,
    "ic": 256,
    "r": 3,
    "s": 3,
    "x": 64,
    "y": 64
}

inputs = job["ic"] * job["x"] * job["y"]
outputs = job["oc"] * job["x"] * job["y"]
weights = job["ic"] * job["oc"] * job["r"] * job["s"]


def get_replicated_factor():

    with open("timeloop-model.stats.txt", "r") as rf:
        ptn_network = re.compile(r"Network 0(.*?)Network 1", re.DOTALL)
        content = ptn_network.search(rf.read()).group(1)
        
        ptn_input = re.compile(r"Inputs:.*?@multicast ([0-9]+): ([0-9]+)", re.DOTALL)
        bst_factor, bst_volume = map(int, ptn_input.search(content).groups())

    replicated_factor = bst_factor * bst_volume / inputs

    return replicated_factor


def modify_dataflow(oc, ic, r, s, x, y):
    obj = yaml.load(open("dataflow.yaml", "r"), Loader=yaml.FullLoader)

    oc, ic, r, s, x, y = map(int, [oc, ic, r, s, x, y])
    temporal = f"C={ic} M={oc} R={r} S={s} N=1 P={x} Q={y}"
    oc, ic, r, s, x, y = tile["oc"]/oc, tile["ic"]/ic, tile["r"]/r, tile["s"]/s, tile["x"]/x, tile["y"]/y
    oc, ic, r, s, x, y = map(int, [oc, ic, r, s, x, y])
    spatial = f"C={ic} M={oc} R={r} S={s} N=1 P={x} Q={y}"

    obj["mapping"][0]["factors"] = temporal
    obj["mapping"][1]["factors"] = spatial

    print(temporal, spatial, sep="\n")
    yaml.dump(obj, open("dataflow_working.yaml", "w"))


def get_factors(num):
    if num == 3:
        return [3]
    else:
        return [2 for _ in range(int(log2(num)))]


split_dims = ["r", "s", "ic", "oc" , "x", "y"]
temporal = copy.copy(tile)


db = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "db")

input_repl_factors = pd.DataFrame(columns=["performance", "factor"])
peak_performance = 128 / 1024
for dim in split_dims:
    factors = get_factors(tile[dim])
    for f in factors:
        temporal[dim] /= f
        peak_performance *= f
        modify_dataflow(temporal["oc"], temporal["ic"], temporal["r"], temporal["s"], temporal["x"], temporal["y"])
        
        os.system(f"timeloop-model {db}/arch/components/* arch.yaml {db}/vgg16/vgg16_layer6.yaml dataflow_working.yaml")
        factor = get_replicated_factor()
        input_repl_factors = input_repl_factors.append({"performance": peak_performance, "factor": factor}, ignore_index=True)
        input_repl_factors.to_csv("result.csv")

print(input_repl_factors)
# modify_dataflow(1, 1, 1, 1, 1, 1)