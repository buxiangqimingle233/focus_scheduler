import os
import argparse
import yaml
import pandas as pd
from functools import reduce
from time import time

from compiler import global_control as gc
from compiler.toolchain import TaskCompiler
from compiler.focus import EA, individual
from simulator.pyAPI.agent import Simulator
from compiler.spatialsim_agents.variables import Variables

pd.set_option('mode.chained_assignment', None)


def getArgumentParser():
    example_text = '''example:
    
    Generate trace files: 
        python focus.py -bm benchmark/test.yaml -d 4 -fr 1024-4096-512 d
    Run the simulator with the already generated trace files: 
        python focus.py -bm benchmark/test.yaml -d 4 s
    '''

    parser = argparse.ArgumentParser(description="FOCUS Testing", 
                                     epilog=example_text, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-bm", "--benchmark", dest="bm", type=str, metavar="benchmark/test.yaml",
                        default="benchmark/test.yaml", help="Spec file of task to run")
    parser.add_argument("-d", "--array_diameter", dest="d", type=int, metavar="D",
                        default=8, help="Diameter of the PE array")
    parser.add_argument("-fr", "--flit_size_range", dest="fr", type=str, metavar="Fmin-Fmax-Step",
                        default="1024-1024-512", help="Flit size range from Fmin to Fmax, interleave with Step")
    parser.add_argument("-debug", dest="debug", action="store_true")
    parser.add_argument("mode", type=str, metavar="tgesf", default="",
                        help="Running mode, t: invoke timeloop-mapper, g: use fake trace generator, \
                              e: invoke timeloop-model, s: simulate baseline, f: invoke focus scheduler \
                              d: ONLY dump the trace file, do nothing else")
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

    # set debug flags
    gc.timeloop_verbose = args.debug
    gc.scheduler_verbose = args.debug
    gc.mapper_verbose = args.debug

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
    print("task layers: {}".format(gc.layer_names))
    print("PE Utilization: {:.2f}".format(sum(gc.cores) / gc.array_size))
    print("*"*60, "\n")


def run_single_task():
    '''An E2E flow for the task specified in `global_control.py`.
    '''

    printSpecs()

    # Invoke the FOCUS compiling toolchain to generate the original traffic trace.
    toolchain = TaskCompiler()
    start_time = time()
    toolchain.compileTask()

    if gc.simulate_baseline:
        working_dir = Variables.gen_working_dir(gc.spatial_sim_root, gc.taskname)
        spec = Variables.get_spec_path(gc.spatial_sim_root, gc.taskname)
        Simulator(working_dir, spec).run()

    # # Invoke simulator to estimate the performance of baseline interconnection architectures.
    # if gc.simulate_baseline:
    #     prev_cwd = os.getcwd()
    #     os.chdir(gc.spatial_sim_root)
    #     os.system("python run.py single --bm {}".format(gc.taskname))
    #     os.chdir(prev_cwd)
    #     toolchain.analyzeSimResult()

    # # Invoke the FOCUS software procedure to schedule the traffic.
    # if gc.focus_schedule:
    #     # Generate working directory
    #     working_dir = os.path.join(gc.focus_buffer, gc.taskname)
    #     if not os.path.exists(working_dir):
    #         os.mkdir(working_dir)

    #     # Generate an engine for heuristic search
    #     # for debugging
    #     if gc.scheduler_verbose:
    #         ea_controller = EA.EvolutionController(population_size=gc.population_size, n_evolution=gc.n_evolution, 
    #                                             log_path=os.path.join(gc.focus_buffer, gc.taskname, "ea_output"))
    #     else:
    #         ea_controller = EA.ParallelEvolutionController(n_workers=gc.n_workers,
    #             population_size=gc.population_size, n_evolution=gc.n_evolution,
    #             log_path=gc.get_ea_logpath())

    #         ea_controller.init_population(individual.individual_generator)
    #         best_individual, _ = ea_controller.run_evolution_search(gc.scheduler_verbose)
    #     # dump the EA's results
    #     solution = best_individual.getTrace()
    #     dump_file = os.path.join(gc.focus_buffer, gc.taskname, "solution_{}.json".format(gc.flit_size))
    #     solution.to_json(dump_file)

    #     toolchain.analyzeFocusResult()

    end_time = time()
    print("METRO software takes: {} seconds".format(end_time - start_time))


if __name__ == "__main__":
    parser = getArgumentParser()
    args = parser.parse_args()
    fmin, fmax, fstep = map(int, args.fr.split("-"))

    for f in range(fmin, fmax + fstep, fstep):
        vars(args)["f"] = f
        setEnvSpecs(args)
        run_single_task()