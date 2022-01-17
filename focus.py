import os
import argparse
import yaml
import pandas as pd
from functools import reduce

from utils import global_control as gc
from compiler.toolchain import FocusToolChain
from scheduler import EA, individual

pd.set_option('mode.chained_assignment', None)


def getArgumentParser():
    parser = argparse.ArgumentParser(description="FOCUS Testing")
    parser.add_argument("-bm", "--benchmark", dest="bm", type=str, metavar="runfiles/test.yaml",
                        default="runfiles/test.yaml", help="Spec file of task to run")
    parser.add_argument("-d", "--array_diameter", dest="d", type=int, metavar="D",
                        default=8, help="Diameter of the PE array")
    parser.add_argument("-f", "--flit_size", dest="f", type=int, metavar="F",
                        default=1024, help="Flit size")
    parser.add_argument("mode", type=str, metavar="tgesf", default="tesf",
                        help="Running mode, t: invoke timeloop-mapper, g: use fake trace generator, \
                              e: invoke timeloop-model, s: simulate baseline, f: invoke focus software")
    return parser


def setEnvSpecs(args: argparse.Namespace):

    # set architecture parameters
    gc.array_diameter = args.d
    gc.array_size = args.d ** 2
    gc.flit_size = args.f

    # set running mode
    gc.search_dataflow = "t" in args.mode
    gc.extract_traffic = "e" in args.mode
    gc.simulate_baseline = "s" in args.mode
    gc.focus_schedule = "f" in args.mode

    # set dataflow engine
    gc.dataflow_engine = "fake" if "g" in args.mode else "timeloop"

    # set task specifications
    obj = yaml.load(open(args.bm, "r"), Loader=yaml.FullLoader)
    gc.models = list(obj.keys())
    gc.layer_names, gc.cores = [], []
    for model in obj.values():
        gc.layer_names += reduce(lambda x, y: x + y, map(lambda x: list(x.keys()), model))
        gc.cores += reduce(lambda x, y: x + y, map(lambda x: list(x.values()), model))

    # set task name and result file
    if gc.dataflow_engine is "timeloop":
        gc.taskname = "_".join(gc.models)
    else:
        gc.taskname = "fake_task"


def printSpecs():
    print("\n")
    print("*"*20, "Running Environment", "*"*20)
    print("array size: {}, flit size: {}".format(gc.array_size, gc.flit_size))
    print("dataflow engine: {}".format(gc.dataflow_engine))
    print("invoke timeloop-mapper: {}, invoke timeloop-model: {}"
          .format(gc.search_dataflow, gc.extract_traffic))
    print("invoke baseline simulator: {}, invoke focus software: {}"
          .format(gc.simulate_baseline, gc.focus_schedule))
    print("task name: {}".format(gc.taskname))
    print("task layers: {}".format(gc.models))
    print("PE Utilization: {:.2f}".format(sum(gc.cores) / gc.array_size))
    print("*"*60, "\n")


def run_single_task():
    '''An E2E flow for the task specified in `global_control.py`.
    '''

    printSpecs()

    # Invoke the FOCUS compiling toolchain to generate the original traffic trace.
    toolchain = FocusToolChain(gc.layer_names, gc.cores)
    toolchain.compileTask()

    # Invoke simulator to estimate the performance of baseline interconnection architectures.
    if gc.simulate_baseline:
        prev_cwd = os.getcwd()
        os.chdir(gc.spt_sim_root)
        os.system("python run.py single --bm {}".format(gc.taskname))
        os.chdir(prev_cwd)
        toolchain.analyzeSimResult()

    # Invoke the FOCUS software procedure to schedule the traffic.
    if gc.focus_schedule:
        # Generate an engine for heuristic search
        ea_controller = EA.ParallelEvolutionController(n_workers=gc.n_workers,
            population_size=gc.population_size, n_evolution=gc.n_evolution)

        # for debugging
        # ea_controller = EA.EvolutionController()

        ea_controller.init_population(individual.individual_generator)
        best_individual, _ = ea_controller.run_evolution_search(gc.scheduler_verbose)
        # dump & print
        best_trace = best_individual.getTrace()
        best_trace.to_json("buffer/best_scheduling.json")


if __name__ == "__main__":
    parser = getArgumentParser()
    args = parser.parse_args()
    setEnvSpecs(args)
    run_single_task()
