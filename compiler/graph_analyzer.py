import networkx as nx
import copy
import networkx as nx
# from op_graph.micro_op_graph import MicroOpGraph
from routing_algorithms.meshtree_router import MeshTreeRouter, RPMTreeRouter, WhirlTreeRouter, Steiner_TreeRouter
import matplotlib.pyplot as plt
import functools
import argparse

import time

time_start = time.time()


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

def first_value(a):
    return a[0]

class Graph_analyzer:

    def __init__(self, diameter, graph = None, router = RPMTreeRouter, multi_task_per_core = False, per_layer_topology = False, \
                 edge_priority = False, user_defined_router = False) -> None:
        # super.__init__()
        self.diameter = diameter
        self.multi_task_per_core = multi_task_per_core
        self.per_layer_topology = per_layer_topology
        self.edge_priority = edge_priority
        self.edge_occupied = {}
        self.p_node_occupied = {}
        if not user_defined_router:
            self.router = router(self.diameter)
        #user defined router: a mapping from hyper edge to route path
        self.user_defined_router = user_defined_router
        if user_defined_router:
            self.router = router
        self.max_time = 0
        if graph != None:
            self.graph = copy.deepcopy(graph)

        self.channel_loads = {}
        self.size_length = 0
        self.raw_graph = copy.deepcopy(self.graph)

        

    def init_graph(self):
        # print('init_graph start:', time.time()-time_start)
        G_backup = copy.deepcopy(self.graph)
        topo_node_list = None
        if self.per_layer_topology:
            topo_node_list = list(self.per_layer_topological_sort(G_backup))
        else:
            topo_node_list = list(nx.topological_sort(G_backup))

        # print('0', time.time()-time_start)
        self.p_pe_earliest = {}
        self.p_pe_latest = {}
        self.p_pe_delay_sum = {}

        for i in range(self.diameter**2):
            self.p_pe_earliest[i] = None
            self.p_pe_latest[i] = 0
            self.p_pe_delay_sum[i] = 0


        for v in topo_node_list:
            G_backup.nodes[v]['start'] = 0
            G_backup.nodes[v]['should_start'] = 0

        # print('1', time.time()-time_start)

        for e in G_backup.edges():
            G_backup.edges[e]['waiting_time'] = 0

        # print('2', time.time()-time_start)

        for p_v in range(self.diameter**2):
            self.p_node_occupied[p_v] = (0, 0)

        for i in range(self.diameter**2):
            for j in range(self.diameter**2):
                if (abs(i-j) == 1 or abs(i-j) == self.diameter) :
                    self.edge_occupied[(i, j)] = []

        # print('init_graph end:', time.time()-time_start)
        return G_backup, topo_node_list

    def edge_priority_func(self, edge):
        return -self.graph.edges[edge[0]]['priority']

    def edge_fid_func(self, edge):
        return self.graph.edges[edge]['fid']

    def edge_end_time(self, hyper_edge):
        return self.graph.edges[hyper_edge[0]]['end_time']

    def node_start_time(self, hyper_edge):
        return self.graph.nodes[hyper_edge[0][0]]['start'] 

    def analyze(self, critical_path = False, endtime_sort=False):
        # print('analyze start:', time.time()-time_start)
        G_backup, topo_node_list = self.init_graph()
        self.graph = copy.deepcopy(G_backup)
        self.max_time = 0
        #debug
        # print(topo_node_list)
        #if we give each edge a priority, we can use it to decide excute order per-layer
        if self.edge_priority:
        #     nodes_lists = []
        #     edges_lists = []
        #     graph_backup = copy.deepcopy(self.graph)
        #     while graph_backup.nodes():
        #         temp_node_list = []
        #         temp_edge_list = []    #edges in a layer
        #         for v in graph_backup.nodes():
        #             if graph_backup.in_degree(v) == 0:
        #                 temp_node_list.append(v)
        #                 edge_list_a_node = list(graph_backup.edges(v))
        #                 edge_list_a_node.sort(key=self.edge_fid_func)

        #                 pre_fid = None
        #                 if edge_list_a_node:
        #                     pre_fid = self.graph.edges[edge_list_a_node[0]]['fid']
        #                 hyper_edge = []
        #                 for e in edge_list_a_node:
        #                     if self.graph.edges[e]['fid'] == pre_fid:
        #                         hyper_edge.append(e)
        #                     else:
        #                         temp_edge_list.append(hyper_edge)
        #                         hyper_edge = []
        #                         pre_fid = self.graph.edges[e]['fid']
        #                         hyper_edge.append(e)
        #                 if hyper_edge:
        #                     temp_edge_list.append(hyper_edge)

        #         nodes_lists.append(temp_node_list)
        #         edges_lists.append(temp_edge_list)
        #         graph_backup.remove_nodes_from(temp_node_list)

            
            # for l in edges_lists:
            #     l.sort(key=self.edge_priority_func)

            # # print(edges_lists, "%%%%%%%")

            # for l in edges_lists:
            #     l.sort(key=self.node_start_time)
            #     for e in l:
            #         print(f'hyper_edge{e}', time.time()-time_start)

            #         edge = e[0]
            #         id = G_backup.edges[e[0]]['fid']
            #         dest = [u[1] for u in e]

            #         if critical_path:
            #             G_backup = self.critical_path_update(e[0][0], dest, G_backup)
            #         else:
            #             G_backup = self.update(e[0][0], id, edge, dest, G_backup, hyper_edge=e)
            fid_dict = {}
            counter = 0
            for v in topo_node_list:
                temp_edges_list = list(G_backup.edges(v))
                if self.graph.nodes[v]['op_type'] == 'worker':
                    self.p_pe_delay_sum[self.graph.nodes[v]['p_pe']] += self.graph.nodes[v]['delay']
                    # print(self.graph.nodes[v]['p_pe'], " ", self.p_pe_delay_sum[self.graph.nodes[v]['p_pe']])
                for e in temp_edges_list:
                    if (not G_backup.edges[e]['fid'] in fid_dict.keys()):
                        id = G_backup.edges[e]['fid']
                        dest = [u for u in G_backup[v].keys() if self.graph.edges[(v, u)]['fid'] == id]

                        if critical_path:
                            G_backup = self.critical_path_update(v, dest, G_backup)
                        else:
                            G_backup = self.update(v, id, e, dest, G_backup)

                        fid_dict[id] = True

                        # counter += 1
                        # if counter % 1000 == 0:
                        #     print('1000finished:', time.time()-time_start)
            
            # print('analyze end:', time.time()-time_start)
            self.graph = G_backup
            
            
        else:
            for v in topo_node_list:
                

                while G_backup.edges(v):
                    edge = list(G_backup.edges(v))[0]
                    id = G_backup.edges[edge]['fid']
                    dest = [u for u in G_backup[v].keys()]

                    if critical_path:
                        G_backup = self.critical_path_update(v, dest, G_backup)
                    else:
                        G_backup = self.update(v, id, edge, dest, G_backup)

                    G_backup.remove_edges_from([e for e in G_backup.edges(v) if G_backup.edges()[e]['fid'] == id])
                G_backup.remove_node(v)

        return self.max_time
    
    #update mesh_edge occupied time, node start, self.max_time, edge waiting time, mesh_node occupied time
    def update(self, vector, edge_id, edge, dest, graph, hyper_edge=None, ignore_tranfering = False):
        routing_tree = None
        # print('update start:', time.time()-time_start)
        if not self.user_defined_router:
            routing_tree = self.router.route(graph.nodes[vector]['p_pe'], [graph.nodes[u]['p_pe'] for u in dest], xy_format=False)
        else:
            routing_tree = self.router.route(vector, dest, xy_format=False)
        p_source = graph.nodes[vector]['p_pe']
        edges_list = self.tree_to_edges(routing_tree)
        temp_edge_occupied = {}
        transfer_start = None
        # print('routing tree and edge lists generate:', time.time()-time_start)
        #transfer_start = max(graph.nodes[vector]['start'], self.p_node_occupied[graph.nodes[vector]['p_pe']][1]) + graph.nodes[vector]['delay']
        if self.multi_task_per_core:
            transfer_start = graph.nodes[vector]['start'] + graph.nodes[vector]['delay']
        else:
            transfer_start = max(graph.nodes[vector]['start'], self.p_node_occupied[graph.nodes[vector]['p_pe']][1]) + graph.nodes[vector]['delay']
        size = (graph.edges[edge]['size']+10)*4

        #update mesh_node occupied time
        self.p_node_occupied[graph.nodes[vector]['p_pe']] = (max(graph.nodes[vector]['start'], self.p_node_occupied[graph.nodes[vector]['p_pe']][1]), transfer_start)

        max_waiting_time = 0
        max_waiting_e = None

        # print('core start time updated:', time.time()-time_start)

        if not ignore_tranfering:
            #update mesh_edge occupied time
            
            for e in edges_list:
                # print(routing_tree.edges(), " ", p_source, " ", e[0])

                path_from_start2v = self.path_in_tree(routing_tree, p_source, e[0], [], 0)
                transfering_time = 0
                if path_from_start2v:
                    for ee in path_from_start2v:
                        transfering_time += abs(ee[0]%self.diameter - ee[1]%self.diameter) + abs(ee[0]//self.diameter - ee[1]//self.diameter)

                # transfering_time = edges_list.index(e)


                occupy_start = transfer_start + transfering_time
                occupy_end = occupy_start + size
                temp_edge_occupied[e] = (occupy_start, occupy_end)    #here occupy_end don't need to wait, [ , )

                if not self.edge_occupied[e] or self.edge_occupied[e][0][0] < occupy_end:
                    for i in range(len(self.edge_occupied[e])):
                        if i == len(self.edge_occupied[e]) - 1 or (self.edge_occupied[e][i+1][0] - max(occupy_start, self.edge_occupied[e][i][1])) >= transfering_time:
                            max_waiting_time = max(0, self.edge_occupied[e][i][1] - occupy_start)
                            max_waiting_e = e
                            break

            # print('1:', time.time()-time_start)


            for e in edges_list:
                self.edge_occupied[e].append((temp_edge_occupied[e][0] + max_waiting_time, temp_edge_occupied[e][1] + max_waiting_time))
                self.edge_occupied[e].sort(key=first_value)
                if len(self.edge_occupied[e]) > 10:
                    self.edge_occupied[e] = self.edge_occupied[e][-10:-1]
                self.max_time = max(self.max_time, temp_edge_occupied[e][1] + max_waiting_time) + 1 #we suppose there will be a output node whose delay is 0

            #update edge waiting time
            for e in graph.edges(vector):
                if graph.edges[e]['fid'] == edge_id:
                    graph.edges[e]['waiting_time'] = max_waiting_time

        # print('edge updated:', time.time()-time_start)
        # print(max_waiting_time)
        # if max_waiting_e:
        #     print("e:",self.edge_occupied[max_waiting_e])
        #
        for v in dest:
            p_pe = graph.nodes[v]['p_pe']
            
            path_from_start2v = self.path_in_tree(routing_tree, p_source, p_pe, [], 0)
            transfering_time = 0
            if path_from_start2v:
                for e in path_from_start2v:
                    transfering_time += abs(e[0]%self.diameter - e[1]%self.diameter) + abs(e[0]//self.diameter - e[1]//self.diameter)

            
            graph.nodes[v]['start'] = max(graph.nodes[v]['start'], transfering_time + transfer_start + max_waiting_time + size)
            # print(graph.nodes[vector]['op_type'])
            if graph.nodes[vector]['op_type'] == 'worker':
                graph.nodes[v]['should_start'] = max(graph.nodes[v]['should_start'], transfering_time + transfer_start + max_waiting_time + size)
            
            if graph.nodes[v]['op_type'] == 'worker':
                if self.p_pe_earliest[graph.nodes[v]['p_pe']] == None:
                    self.p_pe_earliest[graph.nodes[v]['p_pe']] = graph.nodes[v]['start']
                
                self.p_pe_latest[graph.nodes[v]['p_pe']] = max(self.p_pe_latest[graph.nodes[v]['p_pe']], graph.nodes[v]['start'] + graph.nodes[v]['delay'])


            self.graph.edges[(vector, v)]['end_time'] = transfering_time + transfer_start + max_waiting_time + size
            # if in the same core, transfermation is unnecessary
            if p_source == p_pe:
                graph.nodes[v]['start'] = max(graph.nodes[v]['start'], transfer_start)
                if graph.nodes[vector]['op_type'] == 'worker':
                    graph.nodes[v]['should_start'] = max(graph.nodes[v]['should_start'], transfer_start)
                self.graph.edges[(vector, v)]['end_time'] = transfer_start

        # print('start time updated:', time.time()-time_start)
        
        # if vector in [5,6]:
        #     print(vector, p_source, transfering_time, transfer_start, max_waiting_time, size, sep=' ')

        return graph

    def path_in_tree(self, tree, v, dest, path, depth):
        if depth > self.diameter * 3:
            return []
        if v == dest:
            return path
        
        if tree.out_degree(v) == 0:
            return None
        
        for e in tree.edges(v):
            path2 = copy.deepcopy(path)
            path2.append(e)
            path3 = self.path_in_tree(tree, e[1], dest, path2, depth + 1)

            if path3:
                return path3



    #transform the x-y format to full format
    def tree_to_edges(self, tree):
        edges_list = []
        for e in tree.edges():
            edges_list += self.xy_to_full(e)
        return edges_list

    def xy_to_full(self,edge):
        edges_list = []
        
        mid_node = edge[0] - edge[0] % self.diameter + edge[1] % self.diameter
        
        #if same col
        if mid_node != edge[0]:
            dilta = (mid_node - edge[0]) // int(abs(mid_node - edge[0]))
            curr_p = edge[0]
            next_p = edge[0]
            while next_p != mid_node:
                next_p += dilta
                edges_list.append((curr_p, next_p))
                curr_p = next_p

        if mid_node != edge[1]:
            dilta = ((edge[1] - mid_node) // int(abs(mid_node - edge[1]))) * self.diameter
            curr_p = mid_node
            next_p = mid_node
            while next_p != edge[1]:
                next_p += dilta
                edges_list.append((curr_p, next_p))
                curr_p = next_p
        
        return edges_list

    
    def critical_path_update(self, v, dest, G_backup):
        for d in dest:
            G_backup.nodes[d]['start'] = max(G_backup.nodes[d]['start'], G_backup.nodes[v]['start'] + G_backup.nodes[v]['delay'])
            self.max_time = max(self.max_time, G_backup.nodes[d]['start'] + G_backup.nodes[d]['delay'])
        return G_backup


    def per_layer_topological_sort(self, graph):
        nodes_list = []
        # print('per_layer_top:', time.time()-time_start)
        graph_backup = copy.deepcopy(graph)
        while graph_backup.nodes():
            temp_node_list = []
            for v in graph_backup.nodes():
                if graph_backup.in_degree(v) == 0:
                    temp_node_list.append(v)
            nodes_list += temp_node_list
            # print(temp_node_list)
            graph_backup.remove_nodes_from(temp_node_list)
        return nodes_list


    def get_channel_loads_and_size_length(self, use_cnt=False):
        topo_node_list = list(nx.topological_sort(self.graph))
        fid_dict = {}
        counter = 0
        

        for v in topo_node_list:
            temp_edges_list = list(self.graph.edges(v))
            for e in temp_edges_list:
                if (not self.graph.edges[e]['fid'] in fid_dict.keys()):
                    id = self.graph.edges[e]['fid']
                    dest = [u for u in self.graph[v].keys() if self.graph.edges[(v, u)]['fid'] == id]

                    self.graph = self.accumulate(v, id, e, dest, self.graph, use_cnt)

                    fid_dict[id] = True

                    counter += 1
                    if counter % 1000 == 0:
                        print('1000finished:', time.time()-time_start)

        return self.channel_loads, self.size_length

    def accumulate(self, vector, edge_id, edge, dest, graph, use_cnt):
        p_dest = [graph.nodes[u]['p_pe'] for u in dest]
        p_dest.sort()
        routing_tree = None
        if not self.user_defined_router:
            routing_tree = self.router.route(graph.nodes[vector]['p_pe'], p_dest, xy_format=False)
        else:
            routing_tree = self.router.route(vector, dest, xy_format=False)
        for e in routing_tree.edges():
            delta = 1
            if use_cnt:
                delta = self.graph.nodes[vector]['cnt']

            if e in list(self.channel_loads.keys()):
                self.channel_loads[e] += delta
            else:
                self.channel_loads[e] = delta
        
        self.size_length += graph.edges[edge]['size'] * len(routing_tree.edges())

        return graph
        
    
    def core_utilization(self):
        sum = 0
        cnt = 0
        # for v in self.graph.nodes():
        #     if 'op_type' in self.graph.nodes[v].keys():
        #         if self.graph.nodes[v]['op_type'] == 'worker':
        #             utilization = (self.graph.nodes[v]['delay']) / (self.graph.nodes[v]['delay'] + self.graph.nodes[v]['start'] - self.graph.nodes[v]['should_start'])
        #             sum += utilization
        #             cnt += 1
        
        # mean = sum / cnt

        for i in range(self.diameter**2):
            if self.p_pe_earliest[i] and (self.p_pe_delay_sum[i] + self.p_pe_latest[i] - self.p_pe_earliest[i]):
                sum += self.p_pe_delay_sum[i] / (self.p_pe_delay_sum[i] + self.p_pe_latest[i] - self.p_pe_earliest[i])
            if self.p_pe_delay_sum[i] > 0:
                cnt += 1
        
        mean = sum / cnt
        return mean





if __name__ == "__main__":
    # graph = nx.DiGraph()
    # graph.add_nodes_from([(1,{'delay':0, 'p_pe':12}), (2,{'delay':0, 'p_pe':12}), (3,{'delay':0, 'p_pe':12}), (4,{'delay':0, 'p_pe':12}), \
    #                       (5,{'delay':3, 'p_pe':8}), (6,{'delay':4, 'p_pe':15}), (7,{'delay':5, 'p_pe':2}), (8,{'delay':0, 'p_pe':12})])
    # graph.add_edges_from([(1,5,{'fid':0, 'size':4, 'priority':4}), (2,5,{'fid':0, 'size':3, 'priority':3}), (3,6,{'fid':0, 'size':2, 'priority':2}), (4,6,{'fid':0, 'size':1, 'priority':4}), \
    #                       (5,7,{'fid':0, 'size':1, 'priority':1}), (6,7,{'fid':0, 'size':2, 'priority':2}), (7,8,{'fid':0, 'size':3, 'priority':3})])

    # a = Graph_analyzer(4, graph=graph, router=WhirlTreeRouter, per_layer_topology=True, edge_priority=True)
    # print("total cycles:",a.analyze())
    # print("critical cycles:", a.analyze(critical_path=True))

    # temp = Graph_analyzer(4)
    # print(temp.per_layer_topological_sort(graph))

    #######################################################################3
    # router = WhirlTreeRouter(8)
    # tree = router.route(6, [22, 23, 30])
    # print(tree.edges())
    # A = Graph_analyzer(8)
    # print(A.path_in_tree(tree, 6, 30, []))


    graph = nx.read_gpickle('./Steiner_TreeRouter_mobilenet_v3_large_b1w1024_20x20_mobilenet_v3_large_8.yaml_small_channel_loads.gpickle')
    # for e in graph.edges():
    #     graph.edges[e]['priority'] = -1

    print(len(graph.edges()))


    # # for v in graph.nodes():
    # #     print(graph.nodes[v])
    
    # mapping = {}
    # label = 1
    # for v in graph.nodes():
    #     mapping[v] = f"{label}:{graph.nodes[v]['delay']}"
    #     # print(label, graph.nodes[v], sep = ' ')
    #     label += 1
    # graph = nx.relabel_nodes(graph, mapping=mapping)

    # # for e in graph.edges():
    # #     print(e, graph.edges[e], sep=' ')

    

    # nx.draw_networkx(graph)  # networkx draw()
    
    # plt.draw()  # pyplot draw()
    # plt.savefig('./figure.png')

    a = Graph_analyzer(20, graph=graph, router=Steiner_TreeRouter, per_layer_topology=False, multi_task_per_core=True, edge_priority=True)
    print("total cycles:", a.analyze())
    print('core utilization:', a.core_utilization())
    print("critical cycles:", a.analyze(critical_path=True))

    # print(a.get_channel_loads_and_size_length())
    