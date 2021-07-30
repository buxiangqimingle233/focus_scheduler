import os
import pandas as pd

from utils.global_control import *
from trace_gen import trace_gen
from mapper import task_map

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


def run():

    # Instantiate the task mapper
    task_mapper = task_map.ml_mapping()

    # Generate task mapping
    core_map = task_mapper.map()

    # Instantiate the traffic trace generator
    trace_generator = trace_gen.WorkingLayerSet(layer_names, cores, core_map)

    # Generate trace file
    trace_generator.generate()

    # Baseline test: invoke HNOCs

    if simulate_baseline:
        
        os.system(f"cp trace.dat {hnocs_working_path}")
        prev_cmd = os.getcwd()
        os.chdir(hnocs_working_path)
        os.system("./run.sh")
        os.chdir(prev_cmd)

    # FOCUS optimizations: invoke focus scheduler



if __name__ == "__main__":
    run()
