import os
import pandas as pd
import seaborn as sns
import numpy as np

from trace_gen import trace_gen
from mapper import task_map
from ts_scheduler import EA
from ts_scheduler import individual

from utils.global_control import *


def run():

    # Instantiate the task mapper
    task_mapper = task_map.ml_mapping()

    # Generate task mapping
    core_map = task_mapper.map()

    # Visualize mapping results
    os.system("gnuplot mapper/mapping_vis.gp")

    # Instantiate the original traffic trace generator
    org_trace_generator = trace_gen.WorkingLayerSet(layer_names, cores, core_map)

    # Generate trace file for HNOCS
    org_trace_generator.generate()    

    # Baseline test: invoke HNOCs
    if simulate_baseline:
        os.system(f"cp trace.dat {hnocs_working_path}")
        prev_cmd = os.getcwd()
        os.chdir(hnocs_working_path)
        os.system("./run.sh")
        os.chdir(prev_cmd)
    
    # Instantiate the focus traffic trace generator
    focus_trace_generator = trace_gen.WorkingLayerSetDR(layer_names, cores, core_map)

    # Generate trace file for focus
    focus_trace_generator.generate()

    # FOCUS optimizations: invoke focus scheduler
    if focus_schedule:
        ea_controller = EA.ParallelEvolutionController(n_workers=n_workers, population_size=population_size)
        # ea_controller = EA.EvolutionController()
        ea_controller.init_population(individual.individual_generator)
        best_individual, best_score = ea_controller.run_evolution_search(scheduler_verbose)
        best_individual.getTrace().to_csv("best_scheduling.csv")
        print("Sum Exceeded Latency: {}".format(best_score))



if __name__ == "__main__":
    run()
