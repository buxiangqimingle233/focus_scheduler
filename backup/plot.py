import numpy as np
import seaborn as sns
import pandas as pd

from utils.global_control import *

def plot_heatmap(analysis_result):
    lantencies = analysis_result["achieved_bandwidth"] / analysis_result["required_bandwidth"]
    lantencies = analysis_result["slow_down"]
    # lantencies[lantencies > 10] = 10
    supp = pd.Series([0] * (array_diameter**2 - len(lantencies)))
    lantencies = lantencies.append(supp)

    core_map = lantencies.values.reshape((array_diameter, -1))  
    fig_name = "heatmap.png"
    fig = sns.heatmap(data=core_map, cmap="RdBu_r", linewidths=0.3, center=1, annot=False)
    heatmap = fig.get_figure()
    heatmap.savefig(fig_name, dpi=400)


def plot_dist(pkt_sizes):

    array = np.asfarray(pkt_sizes)

    print("min: {}, avg: {}, max: {}".format(array.min(), array.mean(), array.max()))

    overhead = arch_config["w"] / (array.flatten() + arch_config["w"])
    fig = sns.displot(overhead)
    fig_name = "dist.png"
    fig.savefig(fig_name, dpi=400)



# x = np.array([[1,2,3,4], [2,3,4,6], [10,2,3,6], [8,9,7,3]])
# fig_name = 'heatmap.png'
# fig = sns.heatmap(x, annot = True)
# heatmap = fig.get_figure()
# heatmap.savefig(fig_name, dpi = 400)