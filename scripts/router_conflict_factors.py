import os
import sys
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root, "compiler"))

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from compiler import global_control as gc


def plot_router_conflict_factors(conflict_factor: list):
    plt.cla()
    plt.clf()
    board = np.asarray_chkfinite(conflict_factor)
    board = board.reshape((gc.array_diameter, gc.array_diameter))
    fig = sns.heatmap(data=board, cmap="RdBu_r", linewidths=0.3, annot=True)
    heatmap = fig.get_figure()
    heatmap.savefig(os.path.join(gc.visualization_root, "conflicts_per_router_{}_{}.png".format(gc.taskname, gc.benchmark_name[10:])), dpi=500)
    plt.close()