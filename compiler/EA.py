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
import random
from graph_analyzer import Graph_analyzer

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
    def edge_fid_func(self, e):
        return self.graph.edges[e]['fid']


    def __init__(self, graph, diameter, init_method=WhirlTreeRouter, mutate_k = 1, tree_generate_rate = 0.5) -> None:
        self.graph = graph
        self.diameter = diameter
        self.router = {}
        self.init_router = init_method(self.diameter)
        self.mutate_k = mutate_k
        self.tree_generate_rate = tree_generate_rate
        
        nodes_lists = []
        edges_lists = []
        hyper_edges_list = []
        graph_backup = copy.deepcopy(graph)
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
                        pre_fid = graph.edges[edge_list_a_node[0]]['fid']
                    hyper_edge = []
                    for e in edge_list_a_node:
                        if graph.edges[e]['fid'] == pre_fid:
                            hyper_edge.append(e)
                        else:
                            temp_edge_list.append(hyper_edge)
                            hyper_edges_list.append(hyper_edge)
                            hyper_edge = []
                            pre_fid = graph.edges[e]['fid']
                            hyper_edge.append(e)
                    if hyper_edge:
                        temp_edge_list.append(hyper_edge)
                        hyper_edges_list.append(hyper_edge)

            nodes_lists.append(temp_node_list)
            edges_lists.append(temp_edge_list)
            graph_backup.remove_nodes_from(temp_node_list)

        
        # print(hyper_edges_list)
        # print(edges_lists)

        for e in hyper_edges_list:
            e.sort(key=functools.cmp_to_key(edge_cmp))
            source = graph.nodes[e[0][0]]['p_pe']
            dests = []
            for i in e:
                dests.append(graph.nodes[i[1]]['p_pe'])
            self.router[tuple(e)] = self.init_router.route(source=source, dests=dests, xy_format=False)
    

        self.hyper_edges_list = hyper_edges_list
        self.routing_strategies = len(list(self.router.keys()))

    def mutate(self):
        keys_list = list(self.router.keys())
        mutate_obj = random.sample(keys_list, self.mutate_k)

        child = copy.deepcopy(self)

        for i in mutate_obj:
            child.router[i] = self.disturb(child.router[i])
        
        return child

    def crossover(self, A, B):
        paths_to_change = random.sample(list(A.router.keys()), A.routing_strategies // 2)
        child1 = copy.deepcopy(A)
        child2 = copy.deepcopy(B)
        for i in paths_to_change:
            temp = child1.router[i]
            child1.router[i] = child2.router[i]
            child2.router[i] = temp

        temp = random.random()
        child = child1
        if temp < 0.5:
            child = child2
        
        return child


    
    def disturb(self, tree):
        nodes_list = list(tree.nodes())
        # center_node = (random.sample(nodes_list, 1))[0]
        start_node = None
        if len(list(tree.nodes())) == 1:
            return tree
        while start_node == None or tree.nodes[start_node]['dest'] == True:
            start_node = (random.sample(nodes_list, 1))[0]
            


        sub_tree_nodes = []
        edge_nodes = []

        self.point_to_tree(tree, start_node, sub_tree_nodes, edge_nodes)

        if sub_tree_nodes:
            #delete subtree add a new node and treat it as a source to build a tree
            for v in sub_tree_nodes:
                tree.remove_node(v)
            
            temp = True
            while temp:
                
                center_p = random.sample(sub_tree_nodes, 1)[0]
                movement = random.sample([self.diameter, -self.diameter, 1, -1, 0], 1)[0]
                center_p += movement
                if center_p >= 0 and center_p < self.diameter**2 and (not center_p in edge_nodes):
                    
                    center_to_edge = self.init_router.route(center_p, edge_nodes, xy_format=False)
                    start_to_center = None
                    if center_p != start_node:
                        start_to_center = self.init_router.route(start_node, [center_p], xy_format=False)

                    edges_list = list(tree.edges()) + list(center_to_edge.edges())
                    if start_to_center:
                        edges_list += list(start_to_center.edges())
                    
                    ## make sure there is no loop in the new graph
                    temp_edges_list = []
                    for e in edges_list:
                        temp_edges_list.append((e[1], e[0]))
                    
                    edges_list += temp_edges_list
                    
                    edges_set = set(edges_list)
                    if len(edges_list) == len(edges_set):
                        temp = False
                        for v in center_to_edge.nodes():
                            tree.add_node(v)
                            tree.nodes[v]['dest'] = False
                        for e in center_to_edge.edges():
                            tree.add_edge(*e)
                        
                        if start_to_center:
                            for v in start_to_center.nodes():
                                tree.add_node(v)
                                tree.nodes[v]['dest'] = False
                            for e in start_to_center.edges():
                                tree.add_edge(*e)

        # for v in tree.nodes():
        #     print(tree.nodes[v], v, sep=' ')

        # print("####")
            
        return tree
                    


    def point_to_tree(self, tree, point, sub_tree_nodes, edge_nodes):
        for e in tree.edges(point):
            if_continue = False
            temp = random.random()
            if temp < self.tree_generate_rate:
                if_continue = True

            if if_continue and (not tree.nodes[e[1]]['dest']):
                sub_tree_nodes.append(e[1])
                self.point_to_tree(tree, e[1], sub_tree_nodes,edge_nodes=edge_nodes)

            else:
                edge_nodes.append(e[1])




class StrategyIndividual:
    def __init__(self, graph, diameter, init_method = WhirlTreeRouter, mutate_k = 1, tree_generate_rate=0.5) -> None:
        self.router = RouterIndividual(graph, diameter, init_method=init_method, mutate_k=mutate_k, tree_generate_rate=tree_generate_rate)
        self.mutate_k = mutate_k
        self.graph = graph
        self.diameter = diameter
        # self.graph_analyzer = Graph_analyzer(9, graph=graph, router=RPMTreeRouter, per_layer_topology=True)
        


        
    
    def evaluate(self):
        graph_analyzer = Graph_analyzer(self.diameter, graph=self.graph, router=self.router, per_layer_topology=True,edge_priority=True, user_defined_router=True)

        return -graph_analyzer.analyze()
    
    def mutate(self):
        child = copy.deepcopy(self)
        child.router = child.router.mutate()

        mutate_edges = random.sample(list(child.graph.edges()), 2*self.mutate_k)
        
        for i in range(self.mutate_k):
            temp = child.graph.edges[mutate_edges[i]]['priority']
            child.graph.edges[mutate_edges[i]]['priority'] = child.graph.edges[mutate_edges[2*self.mutate_k - i - 1]]['priority']
            child.graph.edges[mutate_edges[2*self.mutate_k - i - 1]]['priority'] = temp

        return child

    def crossover(self, A, B):
        child = copy.deepcopy(A)
        child.router = child.router.crossover(A.router, B.router)
        
        temp = A.graph
        r = random.random()
        if r < 0.5:
            temp = B.graph

        child.graph = copy.deepcopy(temp)
        return child


class EvolutionController:
    def __init__(self, mutate_prob=0.1, population_size=100, n_evolution=25, parent_fraction=0.5, mutation_fraction=0.5, crossover_fraction=0, log_path='output/'):
        self.mutate_prob = mutate_prob
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



if __name__ == "__main__":
    # graph = nx.DiGraph()
    # graph.add_nodes_from([(1,{'delay':0, 'p_pe':12}), (2,{'delay':0, 'p_pe':12}), (3,{'delay':0, 'p_pe':12}), (4,{'delay':0, 'p_pe':12}), \
    #                       (5,{'delay':3, 'p_pe':8}), (6,{'delay':4, 'p_pe':15}), (7,{'delay':5, 'p_pe':2}), (8,{'delay':0, 'p_pe':12})])
    # graph.add_edges_from([(1,5,{'fid':0, 'size':4, 'priority':4}), (2,5,{'fid':0, 'size':3, 'priority':3}), (3,6,{'fid':0, 'size':2, 'priority':2}), (4,6,{'fid':0, 'size':1, 'priority':4}), \
    #                       (5,7,{'fid':0, 'size':1, 'priority':1}), (6,7,{'fid':0, 'size':2, 'priority':2}), (7,8,{'fid':0, 'size':3, 'priority':3})])
    graph = nx.read_gpickle('./try.gpickle')
    for e in graph.edges():
        graph.edges[e]['priority'] = -1


    controller = EvolutionController(population_size=100, n_evolution=2, parent_fraction=0.5, mutation_fraction=0.3, crossover_fraction=0.2)
    
    for individual_i in range(controller.population_size):
        individual=StrategyIndividual(graph=graph, diameter=5)
        controller.add_individual(individual)
    best_individual,best_similarity=controller.run_evolution_search()