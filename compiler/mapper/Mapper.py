import networkx as nx
from copy import deepcopy
from random import sample
from math import ceil, log2
from hilbert import hilbert_map

class Mapper:
    # "wsrc", "insrc", "worker", "sink"
    def __init__(self, physical_cores: list, memory_controllers: list) -> None:
        self.cores = physical_cores
        self.mems = memory_controllers

    def map(self, op_graph: nx.DiGraph):
        
        G = deepcopy(op_graph)
        clustered_nodes = self._clustering(G)

        # TODO: its too ugly ... 
        self.sorted_components = [c for c in sorted(clustered_nodes, key=len, reverse=True)]

        nodes = self._select(G, clustered_nodes)
        while nodes != False:
            self._map(G, nodes)
            nodes = self._select(G, clustered_nodes)

        self._apply(G, op_graph)


    def _clustering(self, G: nx.DiGraph) -> list:
        # Instead of storing itermediate outputs back to the DRAM, 
        # the sink PE directly passes them to the downstreaming isources via on-chip networks. 
        # To achieve this goal, we add a constraint to the mapping algorithms: 
        # isources and its corresponding sinks should be mapped to the same physical PE.

        isources = [node for node, op_type in G.nodes(data="op_type") if op_type == "insrc"]
        for insrc in isources:
            sinks = [u for u, _ in G.in_edges(insrc) if G.nodes[u]["op_type"] == "sink"]
            for s in sinks:
                G.add_edge(insrc, s)    # Add the reverse edge to build strong connected components

        return list(nx.strongly_connected_components(G))


    def _select(self, G: nx.DiGraph, components) -> set:
        '''Select a sub-graph to map without repetition
        '''
        if not self.sorted_components:
            return False
        else:
            return self.sorted_components.pop(0)


    def _map(self, G: nx.DiGraph, sub_graph: set):
        '''Map the selected sub_graph to a physical pe
        '''
        op_types = set(map(lambda x: G.nodes[x]["op_type"], sub_graph))
        if {"worker", "sink"} & op_types:
            pe = sample(self.cores, 1)[0]
        elif {"wsrc", "insrc"} & op_types:
            pe = sample(self.mems, 1)[0]
        
        for node in sub_graph:
            G.nodes[node]["p_pe"] = pe


    def _apply(self, working_graph: nx.DiGraph, dest_graph: nx.DiGraph):
        for node, attr in dest_graph.nodes(data=True):
            attr["p_pe"] = working_graph.nodes[node]["p_pe"]
