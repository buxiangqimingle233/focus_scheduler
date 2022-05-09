from random import sample
from mapper import Mapper
from op_graph.micro_op_graph import MicroOpGraph


class RandomMapper(Mapper):
    '''At each iteration, it selects the largest cluster from the remaining operators and maps it to a randomly chosen PE.
    '''

    # "wsrc", "insrc", "worker", "sink"
    def __init__(self, op_graph: MicroOpGraph, physical_layout: dict) -> None:
        super().__init__(op_graph, physical_layout)
        self.sorted_clusters = [c for c in sorted(self.clusters, key=len, reverse=True)]


    def _select(self, clusters: list) -> set:
        '''Select a sub-graph to map without repetition
        '''
        if not self.sorted_clusters:
            return []
        else:
            return self.sorted_clusters.pop(0)


    def _map(self, selected_cluster: set) -> int:
        '''Find a physical pe to map the operators on
        '''
        G = self.working_graph
        op_types = set(map(G.get_operator_type, selected_cluster))

        if {"worker", "sink"} & op_types:
            pe = sample(self.cores, 1)[0]
        elif {"wsrc", "insrc"} & op_types:
            pe = sample(self.mems, 1)[0]

        return pe
