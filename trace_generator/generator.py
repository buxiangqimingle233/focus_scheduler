import os
from random import sample
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import seaborn as sns
import pattern as P
import utils.global_control as gc
from math import floor
# import matplotlib.pyplot as plt

intensity_distributions = {0.5: (30, 5)}
interval_distributions = {0.5: (30, 5)}
zipf_alpha = 1.5

def sample_from_compound_distribution(sample_cnt: int, distribution_list: list) -> np.ndarray:
    samples = np.asarray_chkfinite([np.random.normal(para[0], para[1], sample_cnt) * scale 
        for scale, para in distribution_list.items()])
    ret = np.sum(samples, axis=0)
    return ret


def gen_inference_trace(core_list: list) -> dict:
    intensity = sample_from_compound_distribution(3, intensity_distributions).astype(int)
    # FIXME: Approximation: we assume intensity and interval are indepentent
    interval = sample_from_compound_distribution(3, interval_distributions).astype(int)

    global region_name
    _weight = P.DataDistribution([], core_list, intensity[0] * interval[0], interval[0], "weight", region_name)
    _input = P.DataDistribution([], core_list, intensity[1] * interval[1], interval[1], "input", region_name)
    _output = P.DataReduction(core_list, [], intensity[2] * interval[2], interval[2], "output", region_name)
    
    region_trace = pd.concat([_weight.asDataFrame(), _input.asDataFrame(), _output.asDataFrame()])
    # gc.debug_show(region_trace.columns)
    return region_trace


def gen_trace():
    sum_sampled_core, iter_cnt = 0, 0
    trace = pd.DataFrame()
    gc.layer_names, gc.cores = [], []

    # Sample a consequtive region once a time
    while sum_sampled_core < gc.array_size:
        # sample the region parameters
        region_size = min(2 ** floor(np.random.zipf(zipf_alpha, size=1)),
            gc.array_size - sum_sampled_core)
        sum_sampled_core += region_size
        global region_name
        region_name = "dummy_layer" + str(iter_cnt)

        # generate trace      
        region_trace = gen_inference_trace(range(region_size))
        trace = trace.append(region_trace, ignore_index=True)

        # update global control 
        gc.layer_names.append(region_name)
        gc.cores.append(region_size)

        iter_cnt += 1
    # gc.debug_show(gc.layer_names)
    return trace

if __name__ == "__main__":
    gen_trace()