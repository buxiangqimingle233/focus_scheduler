import networkx as nx
import copy
import networkx as nx
# from op_graph.micro_op_graph import MicroOpGraph
from routing_algorithms.meshtree_router import MeshTreeRouter, RPMTreeRouter, WhirlTreeRouter, Steiner_TreeRouter
import matplotlib.pyplot as plt
import functools
import argparse
from sys import stderr
import numpy as np

import time

time_start = time.time()
args = None
list_maintain = 10

def getArgumentParser():
    example_text = '''example:

        python3 graph_analyzer.py --op_file ./op_graph_output/mobilenet_v3_large_8.gpickle --diameter 32 --reticle_size 16 --reticle_cycle 5
    '''

    parser = argparse.ArgumentParser(description="graph analyzer", 
                                     epilog=example_text, 
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument("-of", "--op_file", dest="of", type=str, metavar="./op_graph_output/",
                        default="benchmark/test.yaml", help="Op_graph file of task to simulate")
    parser.add_argument("-d", "--diameter", dest="d", type=int, metavar="8",
                        default=8, help="Diameter of the mesh of PEs")
    parser.add_argument("-rs", "--reticle_size", dest="rs", type=int, metavar="8",
                        default=8, help="Diameter of the reticle array")
    parser.add_argument("-rc", "--reticle_cycle", dest="rc", type=int, metavar="8",
                        default=8, help="cycles for transmitting a flit between reticles")
    parser.add_argument("-debug", dest="debug", action="store_true")
    return parser


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

    def __init__(self, diameter, reticle_size, graph = None, router = RPMTreeRouter, multi_task_per_core = False,  \
                 user_defined_router = False, reticle_cycle = 1) -> None:
        # super.__init__()
        self.diameter = diameter
        self.reticle_size = reticle_size
        self.reticle_cycle = reticle_cycle
        self.multi_task_per_core = multi_task_per_core
        self.edge_occupied = {}
        self.edge_occupied_sum = {}
        self.edge_occupied_max = {}

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
        self.updata_cnt = 0

        

    def init_graph(self):
        topo_node_list = list(nx.topological_sort(self.graph))

        for v in topo_node_list:
            self.graph.nodes[v]['start'] = 0

        for e in self.graph.edges():
            self.graph.edges[e]['waiting_time'] = 0

        for p_v in range(self.diameter**2):
            self.p_node_occupied[p_v] = (0, 0)

        for i in range(self.diameter**2):
            for j in range(self.diameter**2):
                if (abs(i-j) == 1 or abs(i-j) == self.diameter) :
                    self.edge_occupied[(i, j)] = []
                    if args.debug:
                        self.edge_occupied_sum[(i, j)] = 0
                        self.edge_occupied_max[(i, j)] = 0

        # print('init_graph end:', time.time()-time_start)
        return self.graph, topo_node_list

    def edge_priority_func(self, edge):
        return -self.graph.edges[edge[0]]['priority']

    def edge_fid_func(self, edge):
        return self.graph.edges[edge]['fid']

    def edge_end_time(self, hyper_edge):
        return self.graph.edges[hyper_edge[0]]['end_time']

    def node_start_time(self, hyper_edge):
        return self.graph.nodes[hyper_edge[0][0]]['start'] 

    def analyze(self, critical_path = False):
        self.graph, topo_node_list = self.init_graph()
        self.max_time = 0
        self.updata_cnt = 0
        
        fid_dict = {}
        counter = 0
        for v in topo_node_list:
            temp_edges_list = list(self.graph.edges(v))
            for e in temp_edges_list:
                if (not self.graph.edges[e]['fid'] in fid_dict.keys()):
                    id = self.graph.edges[e]['fid']
                    dest = [u for u in self.graph[v].keys() if self.graph.edges[(v, u)]['fid'] == id]

                    if critical_path:
                        self.graph = self.critical_path_update(v, dest, self.graph)
                    else:
                        self.graph = self.update(v, id, e, dest, self.graph)

                    fid_dict[id] = True
                    self.updata_cnt += 1

                    # counter += 1
                    # if counter % 1000 == 0:
                    #     print('1000finished:', time.time()-time_start)

        return self.max_time
    
    #update mesh_edge occupied time, node start, self.max_time, edge waiting time, mesh_node occupied time
    def update(self, vector, edge_id, edge, dest, graph):
        routing_tree = None
        if not self.user_defined_router:
            routing_tree = self.router.route(graph.nodes[vector]['p_pe'], [graph.nodes[u]['p_pe'] for u in dest], xy_format=False)
        else:
            routing_tree = self.router.route(vector, dest, xy_format=False)

        p_source = graph.nodes[vector]['p_pe']
        edges_list = self.tree_to_edges(routing_tree)
        temp_edge_occupied = {}
        transfer_start = None
        if self.multi_task_per_core:
            transfer_start = graph.nodes[vector]['start'] + graph.nodes[vector]['delay']
        else:
            transfer_start = max(graph.nodes[vector]['start'], self.p_node_occupied[graph.nodes[vector]['p_pe']][1]) + graph.nodes[vector]['delay']
        size = (graph.edges[edge]['size']+10)*4

        #update mesh_node occupied time
        if not self.multi_task_per_core:
            self.p_node_occupied[graph.nodes[vector]['p_pe']] = (max(graph.nodes[vector]['start'], self.p_node_occupied[graph.nodes[vector]['p_pe']][1]), transfer_start)

        max_waiting_time = 0
        
        #update mesh_edge occupied time
        
        for e in edges_list:
            transfering_time = 0
            if self.user_defined_router:
                path_from_start2v = self.path_in_tree(routing_tree, p_source, e[0], [], 0)
                if path_from_start2v:
                    for ee in path_from_start2v:
                        transfering_time += abs(ee[0]%self.diameter - ee[1]%self.diameter) + abs(ee[0]//self.diameter - ee[1]//self.diameter)
            else:
                transfering_time = abs(e[0]%self.diameter - p_source%self.diameter) + abs(e[0]//self.diameter - p_source//self.diameter)
                transfering_time += int((abs((e[0]%self.diameter)//self.reticle_size - (p_source%self.diameter)//self.reticle_size) + \
                                    abs((e[0]//self.diameter)//self.reticle_size - (p_source//self.diameter)//self.reticle_size)) * (self.reticle_cycle - 1))
            

            occupy_start = transfer_start + transfering_time
            occupy_end = occupy_start + size
            reticle_edge = (abs((e[0]%self.diameter)//self.reticle_size - (e[1]%self.diameter)//self.reticle_size) + \
                            abs((e[0]//self.diameter)//self.reticle_size - (e[1]//self.diameter)//self.reticle_size)) 
            if reticle_edge:
                occupy_end += self.reticle_cycle - 1
            temp_edge_occupied[e] = (occupy_start, occupy_end)    #here occupy_end don't need to wait, [ , )

            index_temp = 0
            if len(self.edge_occupied[e]) > list_maintain:
                index_temp = -list_maintain
            if not self.edge_occupied[e] or self.edge_occupied[e][index_temp][0] < occupy_end:
                length_temp = min(len(self.edge_occupied[e]), list_maintain)
                for i in range(length_temp):
                    if i == length_temp - 1 or (self.edge_occupied[e][-length_temp+i+1][0] - max(occupy_start, self.edge_occupied[e][-length_temp+i][1])) >= transfering_time:
                        max_waiting_time = max(max_waiting_time, self.edge_occupied[e][-length_temp+i][1] - occupy_start)
                        break

        # print('1:', time.time()-time_start)


        for e in edges_list:
            if args.debug:
                # print('update cnt: ', self.updata_cnt)
                # print('edge: ', e)
                # print('occupied periods: ', self.edge_occupied[e])
                # print('current period: ', (temp_edge_occupied[e][0] + max_waiting_time, temp_edge_occupied[e][1] + max_waiting_time))
                # print()
                self.edge_occupied_max[e] = max(self.edge_occupied_max[e], temp_edge_occupied[e][1] + max_waiting_time)
                self.edge_occupied_sum[e] += temp_edge_occupied[e][1] - temp_edge_occupied[e][0]


            self.edge_occupied[e].append((temp_edge_occupied[e][0] + max_waiting_time, temp_edge_occupied[e][1] + max_waiting_time))
            self.edge_occupied[e][-11:].sort(key=first_value)
            # if len(self.edge_occupied[e]) > 10:
            #     self.edge_occupied[e] = self.edge_occupied[e][-10:-1]
            self.max_time = max(self.max_time, temp_edge_occupied[e][1] + max_waiting_time) #we suppose there will be a output node whose delay is 0

        #update edge waiting time
        for e in graph.edges(vector):
            if graph.edges[e]['fid'] == edge_id:
                graph.edges[e]['waiting_time'] = max_waiting_time

        
        for v in dest:
            p_pe = graph.nodes[v]['p_pe']
            
            if self.user_defined_router:
                path_from_start2v = self.path_in_tree(routing_tree, p_source, p_pe, [], 0)
                transfering_time = 0
                if path_from_start2v:
                    for e in path_from_start2v:
                        transfering_time += abs(e[0]%self.diameter - e[1]%self.diameter) + abs(e[0]//self.diameter - e[1]//self.diameter)
            else:
                transfering_time = abs(p_pe%self.diameter - p_source%self.diameter) + abs(p_pe//self.diameter - p_source//self.diameter)
                transfering_time += int((abs((p_pe%self.diameter)//self.reticle_size - (p_source%self.diameter)//self.reticle_size) + \
                                    abs((p_pe//self.diameter)//self.reticle_size - (p_source//self.diameter)//self.reticle_size)) * (self.reticle_cycle - 1))

            
            graph.nodes[v]['start'] = max(graph.nodes[v]['start'], transfering_time + transfer_start + max_waiting_time + size)

            self.graph.edges[(vector, v)]['end_time'] = transfering_time + transfer_start + max_waiting_time + size
            # if in the same core, transfermation is unnecessary
            if p_source == p_pe:
                graph.nodes[v]['start'] = max(graph.nodes[v]['start'], transfer_start)

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

    def print_channel_utilization(self):
        for i in range(self.diameter**2):
            for j in range(self.diameter**2):
                if ((abs(i%self.diameter-j%self.diameter) == 1 and abs(i-j) == 1) or abs(i-j) == self.diameter) :
                    print(i, '-', j, ': ', self.edge_occupied_sum[(i, j)] / max(self.edge_occupied_max[(i, j)], 1), sep='')
                    
    def print_channel_utilization_fig(self, colums=100, output_dir = './visualization_output'):
        for i in range(self.diameter**2):
            for j in range(self.diameter**2):
                if ((abs(i%self.diameter-j%self.diameter) == 1 and abs(i-j) == 1) or abs(i-j) == self.diameter) :
                    self.edge_occupied[(i,j)].sort(key=first_value)
                    interval = self.max_time // colums
                    occupied_periods = np.zeros(colums+3)
                    for p in self.edge_occupied[(i,j)]:
                        start = p[0]
                        start_index = int(p[0] // interval)
                        # print(start_index)
                        if p[1] < (start_index+1)*interval:
                            occupied_periods[start_index] += p[1]-p[0]
                        else:
                            occupied_periods[start_index] += (start_index+1)*interval-p[0]
                            temp_index = start_index + 1
                            while (temp_index + 1)*interval < p[1]:
                                occupied_periods[temp_index] += interval
                                temp_index += 1
                            occupied_periods[temp_index] += p[1] - temp_index*interval
                            
                    plt.plot(list(range(colums+3)), occupied_periods)
                    # plt.savefig(output_dir + f'/{i}-{j}.png')
                    # plt.clf()
                        
                            


if __name__ == "__main__":

    parser = getArgumentParser()
    args = parser.parse_args()

    graph = nx.read_gpickle(args.of)

    print('graph edges num:', len(graph.edges()))
    print('graph nodes num:', len(graph.nodes()))


    a = Graph_analyzer(diameter=args.d, reticle_size=args.rs, reticle_cycle=args.rc, graph=graph, router=Steiner_TreeRouter, multi_task_per_core=True)
    print("total cycles:", a.analyze(), file=stderr)
    a.print_channel_utilization_fig()
    plt.savefig('./visualization_output' + f'/gpt2-xl_channel_loads.png')
    
    # if args.debug:
    #     a.print_channel_utilization()
    print("critical cycles:", a.analyze(critical_path=True), file=stderr)

    