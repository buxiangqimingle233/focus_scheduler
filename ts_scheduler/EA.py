import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import random
import numpy as np
import pandas as pd
import time
from tqdm import tqdm
from copy import deepcopy
from individual import Individual

from utils.global_control import *

class EvolutionController:
    def __init__(self, mutate_prob=0.1, population_size=100, n_evolution=50, parent_fraction=0.5, mutation_fraction=0.25, crossover_fraction=0.25, log_path='output/'):
        # evolution hyper-parameters
        # self.n_blocks_mutate_prob = kwargs.get('n_blocks_mutate_prob', 0.1)
        # self.n_base_channels_mutate_prob = kwargs.get('n_base_channels_mutate_prob', 0.5)
        self.mutate_prob = mutate_prob
        # self.n_blocks_mutate_prob = kwargs.get('n_blocks_mutate_prob', 0.1)
        # self.n_channels_mutate_prob = kwargs.get(
        #     'n_base_channels_mutate_prob', 0.1)
        self.population_size = population_size
        self.n_generations = n_evolution
        self.parent_num = int(self.population_size*parent_fraction)
        self.mutation_num = int(self.population_size*mutation_fraction)
        self.crossover_num = int(self.population_size*crossover_fraction)
        self.log_path = log_path
        if not os.path.exists(self.log_path):
            os.makedirs(self.log_path)
        self.log_file = open(os.path.join(self.log_path,time.strftime('%Y%m%d_%H%M%S')), 'a+')

        self.population=[]
        self.scores=[]

    def add_individual(self,individual,score=None):
        self.population.append(individual)
        if score is None:
            score=individual.evaluate()
        self.scores.append(score)

    def init_population(self,individual_generator,allow_repeat=False,max_sample_times=1000):
        self.population.clear()
        print(f"Generate {self.population_size} individuals")
        if allow_repeat:
            n=1
        else:
            n=max_sample_times
        for i in tqdm(range(self.population_size),desc="Generate individuals"):
            repeat_flag = False
            for i in range(n):
                individual = individual_generator()
                # break
                if individual not in self.population:
                    self.add_individual(individual)
                    repeat_flag=False
                    break
            if repeat_flag==True:
                self.add_individual(individual)
                print(f"WARNING: sample {n} times but all the sampled individuals are repeatted in population")

    def mutation(self,parents):
        for _ in range(self.mutation_num):
            selected_parent = parents[np.random.randint(self.parent_num)]
            # Mutate
            child = selected_parent.mutate()
            self.add_individual(child)
    
    def crossover(self,parents):
        for _ in range(self.crossover_num):
            selected_parent1=parents[np.random.randint(self.parent_num)]
            selected_parent2=parents[np.random.randint(self.parent_num)]
            child=selected_parent1.crossover(selected_parent1,selected_parent2)
            self.add_individual(child)

    def run_evolution_search(self, verbose=False):

        best_score_history = []

        print('Start Evolution Search...')
        t=tqdm(range(self.n_generations),desc="Evolutionary Search")
        for generation in t: 
            print(f'Start Generation={generation}')
            # sort
            sorted_inds=np.argsort(self.scores)[::-1]
            parents = [self.population[_] for _ in sorted_inds[:self.parent_num]]

            now_best_score = self.scores[sorted_inds[0]]
            t.set_postfix({'new_best_score': now_best_score})
            print(f'Now Best score: {now_best_score} Best Individual {parents[0]}')

            if now_best_score > -1e-5:
                print("We have found the best solution, break now")
                break

            self.log_file.write(
                f"==={generation}/{self.n_generations}===\n")
            for i in sorted_inds[:3]:
                self.log_file.write(f"{self.scores[i]} {self.population[i]}\n")
            self.log_file.flush()
            
            best_score_history.append(now_best_score)

            # remove individuals
            self.population=parents
            self.scores=[self.scores[_] for _ in sorted_inds[:self.parent_num]]

            # mutation and crossover
            print("Start Mutation")
            self.mutation(parents)
            print("Start Crossover")
            self.crossover(parents)

        print('Finish Evolution Search')
        ind=np.argmax(self.scores)
        return self.population[ind], self.scores[ind]

import multiprocessing as mp


def individual_generator():
    p = Individual(pd.read_csv("trace.dat", header=0), (array_diameter, array_diameter),)
    for i in range(np.random.randint(100)):
        p.mutate(inplace=True)
    return p

def individual_gen_process(pid,individual_generator):
    # print(f"start {pid}")
    with open("output/individual.out", "a+") as outf:
        sys.stdout = outf
        individual=individual_generator()
        score=individual.evaluate()
    return individual,score

def individual_mutation_process(pid,parent):
    # print(f"start {pid}")
    with open("output/individual.out", "a+") as outf:
        sys.stdout = outf
        child=parent.mutate()
        score=child.evaluate()
    return child,score

def individual_crossover_process(pid,parents):
    # print(f"start {pid}")
    with open("output/individual.out", "a+") as outf:
        sys.stdout = outf
        child=parents[0].crossover(*parents)
        score=child.evaluate()
    return child,score

class ParallelEvolutionController(EvolutionController):
    def __init__(self, n_workers=8, mutate_prob=0.1, population_size=100, n_evolution=50, parent_fraction=0.5, mutation_fraction=0.25, crossover_fraction=0.25, log_path='output/'):
        super().__init__(mutate_prob=mutate_prob, population_size=population_size, n_evolution=n_evolution, parent_fraction=parent_fraction, mutation_fraction=mutation_fraction, crossover_fraction=crossover_fraction, log_path=log_path)
        self.n_workers=n_workers

    def init_population(self,individual_generator,allow_repeat=False,max_sample_times=1000):
        self.population.clear()
        print(f"Generate {self.population_size} individuals Parallel")
        if allow_repeat:
            n=1
        else:
            n=max_sample_times
        
        pool=mp.Pool(processes=self.n_workers)
        rst=pool.starmap(individual_gen_process,[ (pid,individual_generator) for pid in range(self.population_size)])
        for i,s in rst:
            self.add_individual(i,s)
        pool.close()
            

    def mutation(self, parents):
        pool=mp.Pool(processes=self.n_workers)
        selected_parents=[]
        for _ in range(self.mutation_num):
            selected_parent = parents[np.random.randint(self.parent_num)]
            print(selected_parent)
            selected_parents.append(selected_parent)
        
        rst=pool.starmap(individual_mutation_process,[ (pid,parent) for pid,parent in enumerate(selected_parents)])
        for i,s in rst:
            self.add_individual(i,s)
        pool.close()

    def crossover(self, parents):
        pool = mp.Pool(processes=self.n_workers)
        selected_parents=[]
        for _ in range(self.mutation_num):
            selected_parent1=parents[np.random.randint(self.parent_num)]
            selected_parent2=parents[np.random.randint(self.parent_num)]
            selected_parents.append([selected_parent1,selected_parent2])
        
        rst = pool.starmap(individual_crossover_process,[ (pid,parent) for pid,parent in enumerate(selected_parents)])
        for i,s in rst:
            self.add_individual(i,s)
        pool.close()