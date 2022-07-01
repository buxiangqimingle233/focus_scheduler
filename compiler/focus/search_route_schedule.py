import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import argparse
import compiler.EA as EA
from .individual import Individual
from copy import deepcopy
import numpy as np

parser=argparse.ArgumentParser()
parser.add_argument('trace_file',type=str)
parser.add_argument('--verbose',action='store_true')
parser.add_argument('--population_size',type=int,default=100)
parser.add_argument('--n_workers',type=int,default=20)
args=parser.parse_args()
trace = pd.read_csv(args.trace_file, header=0)


# trace = pd.read_csv("focus/ts_scheduler/trace.dat", header=0)

# ea_controller=EA.EvolutionController()
ea_controller=EA.ParallelEvolutionController(n_workers=args.n_workers,population_size=args.population_size)

def individual_generator():
    p = Individual(deepcopy(trace),(8, 8),)
    for i in range(np.random.randint(100)):
        p.mutate(inplace=True)
    return p

ea_controller.init_population(individual_generator)
best_individual, best_score=ea_controller.run_evolution_search(args.verbose)
best_individual.getTrace().to_csv("best_solution.csv")