import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from compiler import global_control as gc


def plot_core_busy_ratio(data, ignore):
    plt.cla()
    plt.clf()
    busy_ratio = np.asarray_chkfinite([i for i in data if i not in ignore])
    plot = sns.histplot(busy_ratio, bins=10, binrange=(0, 1), common_norm=True)
    fig = plot.get_figure()

    fig_path = os.path.join(gc.visualization_root, "core_busy_ratio_{}_{}.png".format(gc.taskname, gc.benchmark_name[10:]))
    fig.savefig(fig_path)
    plt.close()

    data_path = os.path.join(gc.op_graph_buffer, "core_busy_ratio_{}_{}.npy".format(gc.taskname, gc.benchmark_name[10:]))
    with open(data_path, "wb") as f:
        np.save(f, busy_ratio)