import networkx as nx
from op_graph.micro_op_graph import MicroOpGraph
from copy import deepcopy


class Mapper:
    def __init__(self, op_graph: MicroOpGraph, physical_layout: dict) -> None:
        self.backup = deepcopy(op_graph)
        self.working_graph = deepcopy(op_graph)

        assert "core" in physical_layout.values() and "mem" in physical_layout.values()
        self.cores = [i for i in physical_layout if physical_layout[i] == "core"]
        self.mems = [i for i in physical_layout if physical_layout[i] == "mem"]
        self.clusters = self._clustering()


    def map(self) -> nx.DiGraph:
        clusters = self.clusters
        cluster_to_map = self._select(clusters)
        while cluster_to_map:
            pe = self._map(cluster_to_map)
            for s in cluster_to_map:
                self.working_graph.set_physical_pe(s, pe)
            cluster_to_map = self._select(clusters)

        return self._generate_ret()


    def _clustering(self) -> list:
        # TODO: need refactor
        
        # Sink nodes at layer x and isource nodes at layer x+1 should be mapped to the 
        # same physical processing element to exploit data reuse. 
        G = deepcopy(self.working_graph.get_data())
        isources = [node for node, op_type in G.nodes(data="op_type") if op_type == "insrc"]
        for insrc in isources:
            sinks = [u for u, _ in G.in_edges(insrc) if G.nodes[u]["op_type"] == "sink"]
            for s in sinks:
                G.add_edge(insrc, s)    # Add the reverse edge to build strong connected components
        
        constraint_edges = [(u, v) for u, v, type_ in G.edges(data="edge_type") if type_ == "map_constraint"]
        for u, v in constraint_edges:
            G.add_edge(v, u)

        return list(nx.strongly_connected_components(G))


    def _generate_ret(self):
        
        # Change mapping constraint edges to real control edges
        constraint_edges = [(u, v) for u, v, type_ in self.backup.get_data().edges(data="edge_type") \
                            if type_ == "map_constraint"]

        for u, v in constraint_edges:
            self.backup.remove_edge(u, v)
            self.backup.add_control_edge(u, v)

        for node, p_pe in self.working_graph.get_data().nodes(data="p_pe"):
            self.backup.set_physical_pe(node, p_pe)

        return self.backup


    def _select(self, clusters: list) -> set:
        assert(False)

    def _map(self, selected_cluster: set) -> int:
        assert(False)
