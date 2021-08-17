import os
import pandas as pd
import seaborn as sns
import numpy as np

from trace_gen import trace_gen
from mapper import task_map
from ts_scheduler import EA
from ts_scheduler import individual

from utils.global_control import *

pd.set_option('mode.chained_assignment', None)

def run():

    # Instantiate the task mapper
    task_mapper = task_map.ml_mapping()

    # Generate task mapping
    core_map = task_mapper.map()

    # FIXME: Doesn't work, visualize mapping results
    os.system("gnuplot mapper/mapping_vis.gp")

    # Instantiate the original traffic trace generator
    org_trace_generator = trace_gen.WorkingLayerSet(layer_names, cores, core_map)
    # Generate trace file for HNOCS
    org_trace_generator.generate()    

    # Baseline test: invoke HNOCs
    if simulate_baseline:
        
        # os.system(f"cp trace.dat {hnocs_working_path}")
        # prev_cmd = os.getcwd()
        # os.chdir(hnocs_working_path)
        # os.system("./run.sh")
        # os.chdir(prev_cmd)

        # Invoke Booksim
        prev_cwd = os.getcwd()
        os.chdir(booksim_working_path)
        os.system("./run.sh")
        os.chdir(prev_cwd)

        org_trace_generator.analyzeBookSim()
    

    # FOCUS optimizations: invoke focus scheduler
    if focus_schedule:
        # Instantiate the focus traffic trace generator
        focus_trace_generator = trace_gen.WorkingLayerSetDR(layer_names, cores, core_map)

        # Generate trace file for focus
        focus_trace_generator.generate()

        # generate scheduling
        ea_controller = EA.ParallelEvolutionController(n_workers=n_workers, population_size=population_size, n_evolution=n_evolution)
        # ea_controller = EA.EvolutionController()
        ea_controller.init_population(individual.individual_generator)
        best_individual, _ = ea_controller.run_evolution_search(scheduler_verbose)

        # dump & print
        best_trace = best_individual.getTrace()
        best_trace.to_csv("best_scheduling.csv")
        slowdown = (best_trace["issue_time"] / (best_trace["interval"] * best_trace["count"]))
        best_mean = slowdown[slowdown > 1].mean()
        print("Sum Exceeded Latency: {}".format(best_mean))
        with open(slowdown_result, "a") as wf:
            # print(arch_config["w"], best_mean, best_mean, sep=",", file=wf)
            print(best_mean)


if __name__ == "__main__":
    run()
