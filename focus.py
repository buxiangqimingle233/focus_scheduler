import os
import pandas as pd
import seaborn as sns
import numpy as np

from server import layer_set
from mapper import task_map
from ts_scheduler import EA
from ts_scheduler import individual
import utils.global_control as gc

pd.set_option('mode.chained_assignment', None)

def run():

    # Instantiate the focus traffic trace generator
    working_layer_set = layer_set.WorkingLayerSetDR(gc.layer_names, gc.cores)

    # Generate traffic trace from real-world workloads, feeding the backends of focus and booksim
    # working_layer_set.getTraceFromTimeloop()

    # Generate traffic trace by randomly mixing traffic operations
    working_layer_set.getTraceFromTraceGenerator()

    # Invoke Booksim
    if gc.simulate_baseline:
        prev_cwd = os.getcwd()
        os.chdir(gc.booksim_working_path)
        os.system("./run.sh")
        os.chdir(prev_cwd)
        working_layer_set.analyzeBookSim()

    if gc.focus_schedule:
        # generate scheduling
        ea_controller = EA.ParallelEvolutionController(n_workers=gc.n_workers, population_size=gc.population_size, n_evolution=n_evolution)
        # for debugging
        # ea_controller = EA.EvolutionController()
        ea_controller.init_population(individual.individual_generator)
        best_individual, _ = ea_controller.run_evolution_search(gc.scheduler_verbose)

        # dump & print
        best_trace = best_individual.getTrace()
        best_trace.to_csv("best_scheduling.csv")
        slowdown = (best_trace["issue_time"] / (best_trace["interval"] * best_trace["count"]))
        best_mean = slowdown[slowdown > 1].mean()
        print("Sum Exceeded Latency: {}".format(best_mean))
        with open(os.path.join("focus-final-out", gc.slowdown_result), "a") as wf:
            # print(arch_config["w"], best_mean, best_mean, sep=",", file=wf)
            print(best_mean, file=wf)


if __name__ == "__main__":
    run()
