import networkx as nx
import copy
import networkx as nx
# from op_graph.micro_op_graph import MicroOpGraph
from routing_algorithms.meshtree_router import MeshTreeRouter, RPMTreeRouter, WhirlTreeRouter
import matplotlib.pyplot as plt
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

    def init_graph(self):
        G_backup = copy.deepcopy(self.graph)
        topo_node_list = None
        if self.per_layer_topology:
            topo_node_list = list(self.per_layer_topological_sort(G_backup))
        else:
            topo_node_list = list(nx.topological_sort(G_backup))

        for v in topo_node_list:
            G_backup.nodes[v]['start'] = 0

        for e in G_backup.edges():
            G_backup.edges[e]['waiting_time'] = 0

        for p_v in range(self.diameter**2):
            self.p_node_occupied[p_v] = (0, 0)

        for i in range(self.diameter**2):
            for j in range(self.diameter**2):
                if (abs(i-j) == 1 or abs(i-j) == self.diameter) :
                    self.edge_occupied[(i, j)] = (0, 0)

        return G_backup, topo_node_list

    def edge_priority_func(self, edge):
        return -self.graph.edges[edge[0]]['priority']

    def edge_fid_func(self, edge):
        return self.graph.edges[edge]['fid']

    def analyze(self, critical_path = False):
        G_backup, topo_node_list = self.init_graph()
        self.graph = copy.deepcopy(G_backup)
        self.max_time = 0
        #debug
        # print(topo_node_list)
        #if we give each edge a priority, we can use it to decide excute order per-layer
        if self.edge_priority:
            nodes_lists = []
            edges_lists = []
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

                nodes_lists.append(temp_node_list)
                edges_lists.append(temp_edge_list)
                graph_backup.remove_nodes_from(temp_node_list)
            
            for l in edges_lists:
                l.sort(key=self.edge_priority_func)

            # print(edges_lists, "%%%%%%%")

            for l in edges_lists:
                for e in l:
                    edge = e[0]
                    id = G_backup.edges[e[0]]['fid']
                    dest = [u[1] for u in e]

                    if critical_path:
                        G_backup = self.critical_path_update(e[0][0], dest, G_backup)
                    else:
                        G_backup = self.update(e[0][0], id, edge, dest, G_backup, hyper_edge=e)
                    
            
            
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
    def update(self, vector, edge_id, edge, dest, graph, hyper_edge=None):
        routing_tree = None
        if not self.user_defined_router:
            routing_tree = self.router.route(graph.nodes[vector]['p_pe'], [graph.nodes[u]['p_pe'] for u in dest])
        else:
            hyper_edge.sort(key=functools.cmp_to_key(edge_cmp))
            # print(list(self.router.router.keys()))
            routing_tree = self.router.router[tuple(hyper_edge)]
        p_source = graph.nodes[vector]['p_pe']
        edges_list = self.tree_to_edges(routing_tree)
        temp_edge_occupied = {}
        transfer_start = None
        #transfer_start = max(graph.nodes[vector]['start'], self.p_node_occupied[graph.nodes[vector]['p_pe']][1]) + graph.nodes[vector]['delay']
        if self.multi_task_per_core:
            transfer_start = graph.nodes[vector]['start'] + graph.nodes[vector]['delay']
        else:
            transfer_start = max(graph.nodes[vector]['start'], self.p_node_occupied[graph.nodes[vector]['p_pe']][1]) + graph.nodes[vector]['delay']
        size = graph.edges[edge]['size']

        #update mesh_node occupied time
        self.p_node_occupied[graph.nodes[vector]['p_pe']] = (max(graph.nodes[vector]['start'], self.p_node_occupied[graph.nodes[vector]['p_pe']][1]), transfer_start)

        #update mesh_edge occupied time
        max_waiting_time = 0
        for e in edges_list:
            occupy_start = transfer_start + abs(e[0]%self.diameter - p_source%self.diameter) + abs(e[0]//self.diameter - p_source//self.diameter)
            occupy_end = occupy_start + size
            temp_edge_occupied[e] = (occupy_start, occupy_end)    #here occupy_end don't need to wait, [ , )
            max_waiting_time = max(max_waiting_time, self.edge_occupied[e][1] - occupy_start)
        
        for e in edges_list:
            self.edge_occupied[e] = (temp_edge_occupied[e][0] + max_waiting_time, temp_edge_occupied[e][1] + max_waiting_time)
            self.max_time = max(self.max_time, temp_edge_occupied[e][1] + max_waiting_time) + 1 #we suppose there will be a output node whose delay is 0

        #update edge waiting time
        for e in graph.edges(vector):
            if graph.edges[e]['fid'] == edge_id:
                graph.edges[e]['waiting_time'] = max_waiting_time

                self.graph.edges[e]['waiting_time'] = max_waiting_time
        
        #
        for v in dest:
            p_pe = graph.nodes[v]['p_pe']
            transfering_time = abs(p_pe%self.diameter - p_source%self.diameter) + abs(p_pe//self.diameter - p_source//self.diameter)
            graph.nodes[v]['start'] = transfering_time + transfer_start + max_waiting_time + size

            self.graph.nodes[v]['start'] = transfering_time + transfer_start + max_waiting_time + size
            # if in the same core, transfermation is unnecessary
            if p_source == p_pe:
                self.graph.nodes[v]['start'] = transfer_start
        
        # if vector in [5,6]:
        #     print(vector, p_source, transfering_time, transfer_start, max_waiting_time, size, sep=' ')

        return graph


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
        graph_backup = copy.deepcopy(graph)
        while graph_backup.nodes():
            temp_node_list = []
            for v in graph_backup.nodes():
                if graph_backup.in_degree(v) == 0:
                    temp_node_list.append(v)
            nodes_list += temp_node_list
            graph_backup.remove_nodes_from(temp_node_list)
        return nodes_list



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
    graph = nx.read_gpickle('./try.gpickle')
    for e in graph.edges():
        graph.edges[e]['priority'] = -1


    for v in graph.nodes():
        print(graph.nodes[v])
    
    mapping = {}
    label = 1
    for v in graph.nodes():
        mapping[v] = f"{label}:{graph.nodes[v]['delay']}"
        # print(label, graph.nodes[v], sep = ' ')
        label += 1
    graph = nx.relabel_nodes(graph, mapping=mapping)

    for e in graph.edges():
        print(e, graph.edges[e], sep=' ')

    

    nx.draw_networkx(graph)  # networkx draw()
    
    plt.draw()  # pyplot draw()
    plt.savefig('./figure.png')

    a = Graph_analyzer(9, graph=graph, router=RPMTreeRouter, per_layer_topology=True)
    print("total cycles:",a.analyze())
    print("critical cycles:", a.analyze(critical_path=True))
    