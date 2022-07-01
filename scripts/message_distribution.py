import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from compiler.op_graph.micro_op_graph import MicroOpGraph
from compiler import global_control as gc

def plot_msg_size_dist(op_graph: MicroOpGraph):
    plt.cla()
    plt.clf()
    # plt.ylim(0, 200)
    # plt.xlim(0, 1000)
    fig_path = os.path.join(gc.visualization_root, "msg_size_dist_{}.png".format(gc.taskname))
    msg_size = [attr["size"] for _, __, attr in op_graph.get_data().edges(data=True) if attr["edge_type"] == "data"]
    msg_size = np.asarray_chkfinite(msg_size)

    fig = sns.histplot(msg_size, bins=30)
    hist = fig.get_figure()
    hist.savefig(fig_path, dpi=500)
    plt.close()

    # also save original data
    data_path = os.path.join(gc.op_graph_buffer, "msg_size_dist_{}.npy".format(gc.taskname))
    with open(data_path, "wb") as f:
        np.save(f, msg_size)
