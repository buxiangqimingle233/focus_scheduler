import copy
import random
import numpy as np
import time
import os
from tqdm import tqdm
import sys
import scipy.stats
from routing_algorithms.router import Router
from routing_algorithms.meshtree_router import MeshTreeRouter, RPMTreeRouter, WhirlTreeRouter
import networkx as nx
import functools

def edge_cmp(a, b):
    if a[0] < b[0]:
        return -1
    elif a[0] > b[0]:
        return 1
    else:
        if a[1] < b[1]:
            return -1
        elif a[1] > b[1]:
            return 1
        else:
            return 0


class RouterIndividual:
    def __init__(self, graph, diameter, init_method=WhirlTreeRouter) -> None:
        self.graph = graph
        self.diameter = diameter
        self.router = {}
        self.init_router = init_method(self.diameter)
        
        nodes_lists = []
        edges_lists = []
        hyper_edges_list = []
        graph_backup = copy.deepcopy(self.graph)
        while graph_backup.nodes():
            temp_node_list = []
            temp_edge_list = []    #edges in a layer
            
            for v in graph_backup.nodes():
                if graph_backup.in_degree(v) == 0:
                    temp_node_list.append(v)
                    edge_list_a_node = list(graph_backup.edges(v))
                    edge_list_a_node.sort(key=self.edge_fid_func)

                    pre_fid = None
                    if edge_list_a_node:
                        pre_fid = self.graph.edges[edge_list_a_node[0]]['fid']
                    hyper_edge = []
                    for e in edge_list_a_node:
                        if self.graph.edges[e]['fid'] == pre_fid:
                            hyper_edge.append(e)
                        else:
                            temp_edge_list.append(hyper_edge)
                            hyper_edge = []
                            pre_fid = self.graph.edges[e]['fid']
                            hyper_edge.append(e)
                    if hyper_edge:
                        temp_edge_list.append(hyper_edge)
                        hyper_edges_list.append(hyper_edge)

            nodes_lists.append(temp_node_list)
            edges_lists.append(temp_edge_list)
            graph_backup.remove_nodes_from(temp_node_list)

        for e in hyper_edges_list:
            e.sort(key=functools.cmp_to_key(edge_cmp))
            source = self.graph.nodes[e[0][0]]['p_pe']
            dests = []
            for i in e:
                dests.append(self.graph.nodes[i[1]]['p_pe'])
            self.router[tuple(e)] = self.init_router.route(source=source, dests=dests, xy_format=False)

        self.hyper_edges_list = hyper_edges_list

    def mutate():
        pass

            
        


        


    


class StrategyIndividual:
    def __init__(self) -> None:
        pass

        
    
    def evaluate(self):
        for i in range(self.convs_num):
            self.convs[i].use_index_weight(change_row_index=self.reorder_index[i])
            self.convs[i+1].use_index_weight(change_col_index=self.reorder_index[i])

        out=self.block.forward_backup(self.x)
        similarity = self.get_similarity(out, self.raw_out)

        for i in range(self.convs_num):
            self.convs[i].restore_index_weight(change_row_index=self.reorder_index[i])
            self.convs[i+1].restore_index_weight(change_col_index=self.reorder_index[i])
        return similarity
    
    def mutate(self):
        child = ReorderIndividual(self.block, self.conv_wrapper, self.raw_inputs, self.raw_outs, self.x, self.similarity_mode, self.convs_num)
        child.reorder_index = copy.deepcopy(self.reorder_index)
        child.reorder_move()
        return child
        


class EvolutionController:
    def __init__(self, mutate_prob=0.1, population_size=100, n_evolution=25, parent_fraction=0.5, mutation_fraction=0.5, crossover_fraction=0, log_path='output/'):
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
        self.log_file = open(os.path.join(self.log_path,'ea_accuracy.txt'), 'a+')

        self.population=[]
        self.scores=[]

    def add_individual(self,individual,score=None):
        self.population.append(individual)
        if score is None:
            score=individual.evaluate()
        self.scores.append(score)
        print(score)

    def init_population(self,individual_generator,allow_repeat=False,max_sample_times=1000):
        self.population.clear()
        print(f"Generate {self.population_size} individuals")
        if allow_repeat:
            n=1
        else:
            n=max_sample_times
        for i in tqdm(range(self.population_size),desc="Generate individuals"):
            repeat_flag=True
            for i in range(n):
                individual=individual_generator()
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

        #print('Start Evolution Search...')
        t=tqdm(range(self.n_generations),desc="Evolutionary Search")
        for generation in t: 
            #print(f'Start Generation={generation}')
            # sort
            sorted_inds=np.argsort(self.scores)[::-1]
            parents = [self.population[_] for _ in sorted_inds[:self.parent_num]]

            now_best_score = self.scores[sorted_inds[0]]
            t.set_postfix({'new_best_score': now_best_score})
            #print(f'Now Best score: {now_best_score} Best Individual {parents[0]}')
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
            #print("Start Mutation")
            self.mutation(parents)
            #print("Start Crossover")
            self.crossover(parents)

        #print('Finish Evolution Search')
        ind=np.argmax(self.scores)
        return self.population[ind],self.scores[ind]

import multiprocessing as mp


def individual_gen_process(pid,individual_generator):
    # print(f"start {pid}")
    sys.stdout = open("output/individual.out", "a+")
    individual=individual_generator()
    score=individual.evaluate()
    return individual,score

def individual_mutation_process(pid,parent):
    # print(f"start {pid}")
    sys.stdout = open("output/individual.out", "a+")
    child=parent.mutate()
    score=child.evaluate()
    return child,score

def individual_crossover_process(pid,parents):
    # print(f"start {pid}")
    sys.stdout = open("output/individual.out", "a+")
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
    
    def crossover(self, parents):
        pool=mp.Pool(processes=self.n_workers,)
        selected_parents=[]
        for _ in range(self.mutation_num):
            selected_parent1=parents[np.random.randint(self.parent_num)]
            selected_parent2=parents[np.random.randint(self.parent_num)]
            selected_parents.append([selected_parent1,selected_parent2])
        
        rst=pool.starmap(individual_crossover_process,[ (pid,parent) for pid,parent in enumerate(selected_parents)])
        for i,s in rst:
            self.add_individual(i,s)