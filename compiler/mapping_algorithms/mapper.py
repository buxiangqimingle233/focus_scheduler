import networkx as nx
from sqlalchemy import false
from op_graph.micro_op_graph import MicroOpGraph
from copy import deepcopy

class Mapper:
    def __init__(self, op_graph: MicroOpGraph, physical_layout: dict) -> None:
        self.original_graph = op_graph
        self.working_graph = deepcopy(op_graph)
        assert "core" in physical_layout.values() and "mem" in physical_layout.values()
        self.cores = [i for i in physical_layout if physical_layout[i] == "core"]
        self.mems = [i for i in physical_layout if physical_layout[i] == "mem"]
        self.clusters = self._clustering()


    def map(self):
    
        clusters = self.clusters
        cluster_to_map = self._select(clusters)
        while cluster_to_map:
            pe = self._map(cluster_to_map)
            self.working_graph.set_physical_pe(cluster_to_map, pe)
            cluster_to_map = self._select(clusters)
        
        self._apply_mapping_to_oringal_graph()


    def _clustering(self) -> list:
        # Instead of storing itermediate outputs back to the DRAM, 
        # the sink PE directly passes them to the downstreaming isources via on-chip networks. 
        # To achieve this goal, we add a constraint to the mapping algorithms: 
        # isources and its corresponding sinks should be mapped to the same physical PE.
        G = self.working_graph.get_data()
        isources = [node for node, op_type in G.nodes(data="op_type") if op_type == "insrc"]
        for insrc in isources:
            sinks = [u for u, _ in G.in_edges(insrc) if G.nodes[u]["op_type"] == "sink"]
            for s in sinks:
                G.add_edge(insrc, s)    # Add the reverse edge to build strong connected components
        return list(nx.strongly_connected_components(G))


    def _select(self, clusters: list) -> set:
        assert(False)


    def _map(self, selected_cluster: set) -> int:
        assert(False)


    def _apply_mapping_to_oringal_graph(self):
        G_to = self.original_graph.get_data()
        G_from = self.working_graph.get_data()
        for node, attr in G_to.nodes(data=True):
            attr["p_pe"] = G_from.nodes[node]["p_pe"]