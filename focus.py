import os
import argparse
import random
from sys import stderr
from time import time
from functools import reduce
import pandas as pd
import numpy as np
import yaml

from compiler import global_control as gc
from compiler.toolchain import TaskCompiler
from compiler.spatialsim_agents.variables import Variables
from scripts.message_distribution import plot_msg_size_dist
from simulator.pyAPI.agent import Simulator

from scripts.channel_utilization import plot_channel_load
from scripts.router_conflict_factors import plot_router_conflict_factors
from scripts.core_busy_ratio import plot_core_busy_ratio


pd.set_option('mode.chained_assignment', None)
random.seed(114514)

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
    parser.add_argument("-d", "--array_diameter", dest="d", type=int, metavar="8",
                        default=8, help="Diameter of the PE array")
    parser.add_argument("-fr", "--flit_size_range", dest="fr", type=str, metavar="Fmin-Fmax-Step",
                        default="1024-1024-512", help="Flit size range from Fmin to Fmax, interleave with Step")
    parser.add_argument("-b", "--batch", dest="b", type=int, default=1, metavar="4")
    parser.add_argument("-debug", dest="debug", action="store_true")
    parser.add_argument("mode", type=str, metavar="tgesf", default="",
                        help="Running mode, t: invoke timeloop-mapper, g: use fake trace generator, \
                              e: invoke timeloop-model, s: simulate baseline, f: invoke focus scheduler \
                              d: ONLY dump the trace file, do nothing else")
    return parser


def setEnvSpecs(args: argparse.Namespace):

    gc.benchmark_name = args.bm

    # set architecture parameters
    gc.array_diameter = args.d
    gc.array_size = args.d ** 2
    gc.flit_size = args.f 
    gc.batch = args.b

    # set running mode
    gc.search_dataflow = "t" in args.mode
    gc.extract_traffic = "e" in args.mode
    gc.simulate_baseline = "s" in args.mode
    gc.focus_schedule = "f" in args.mode
    gc.compile_task = "d" in args.mode

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

    # set the task name that exclusively identify the task
    if gc.dataflow_engine == "timeloop":
        gc.taskname = "_".join(gc.models) + "_b{}w{}".format(gc.batch, gc.flit_size) \
                                          + "_{}x{}".format(gc.array_diameter, gc.array_diameter)
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


def run_single_task(args):
    '''An E2E flow for the task specified in `global_control.py`.
    '''

    printSpecs()

    start_time = time()
    # Invoke FOCUS compiling toolchain 
    if gc.compile_task:
        toolchain = TaskCompiler()
        toolchain.compile()
        compute_cycle = toolchain.get_compute_cycle() # * gc.overclock
        maeri_cycle = toolchain.get_maeri_cycle()
        eyeriss_cycle = toolchain.get_eyeriss_cycle()
        print("{} {} compute cycle: {}, maeri cycle: {}, eyeriss_cycle: {}" \
              .format(gc.taskname, maeri_cycle/compute_cycle, compute_cycle, maeri_cycle, eyeriss_cycle), file=stderr)

        # plot channel loads
        plot_channel_load(toolchain.get_working_graph())
        # plot message size distribution
        plot_msg_size_dist(toolchain.get_working_graph())

    # Invoke simulator
    if gc.simulate_baseline:
        working_dir = Variables.gen_working_dir(gc.spatial_sim_root, gc.taskname)
        sim_config = Variables.get_spec_path(gc.spatial_sim_root, gc.taskname)
        simulator = Simulator(working_dir, sim_config)

        # activate simulator
        simulate_cycle = simulator.run()
        
        # plot core busy ratios
        mems = [k for k, v in TaskCompiler().gen_physical_layout().items() if v != "mems"]
        plot_core_busy_ratio(simulator.core_busy_ratio(), mems)

        # plot router conflict factors
        plot_router_conflict_factors(simulator.router_conflict_factor())


    if gc.compile_task and gc.simulate_baseline:
        print("{} {} {} {} {} {}".format(gc.taskname, gc.array_diameter, gc.flit_size, (simulate_cycle-compute_cycle)/compute_cycle, compute_cycle, simulate_cycle), file=stderr)

    end_time = time()
    print("METRO software takes: {} seconds".format(end_time - start_time))


if __name__ == "__main__":
    parser = getArgumentParser()
    args = parser.parse_args()
    fmin, fmax, fstep = map(int, args.fr.split("-"))

    for f in range(fmin, fmax + fstep, fstep):
        vars(args)["f"] = f
        setEnvSpecs(args)
        run_single_task(args)

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