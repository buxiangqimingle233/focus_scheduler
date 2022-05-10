import sys
from math import ceil, log2
from random import sample

from compiler.mapping_algorithms.hilbert import hilbert_map
from op_graph.micro_op_graph import MicroOpGraph
from mapper import Mapper

class HilbertMapper(Mapper):
    '''To guarantee adjacent PEs are allocated continuously, we sort PEs in the order of Hilbert Curve, \
        and map the operators of the same layer continuously according to that curve. 
    '''

    def __init__(self, op_graph: MicroOpGraph, physical_layout: dict, diameter: int) -> None:
        super().__init__(op_graph, physical_layout)
    
        print("core: ", self.cores)
        print("mem controller: ", self.mems)
        # Check whether the playboard has enough pes to map.
        pe_cnt = self.__get_required_pe_num()
        if pe_cnt > len(self.cores):
            print("ERROR: the array doesn't have enough processing elements to run the task requiring"\
                  " {} pes while we just have {}".format(pe_cnt, len(self.cores)), 
                  file=sys.stderr)
            print("Mapping fails, abort. ")
            exit(0)

        self.sorted_clusters = sorted(self.clusters, key=self.__belonging_layer)

        self.diameter = diameter
        if diameter & (diameter - 1) == 0:
            quantilized_diameter = diameter
        else:
            quantilized_diameter = 2 ** ceil(log2(diameter))
        self.hilbert_curve = list(hilbert_map(log2(quantilized_diameter)))


    def _select(self, clusters: list) -> set:
        if not self.sorted_clusters:
            return []
        else:
            return self.sorted_clusters.pop(0)


    def _map(self, selected_cluster: set) -> int:
        if self.__need_pe(selected_cluster):
            return self.__map_to_core(selected_cluster)
        else:
            return self.__map_to_mem_ctrl(selected_cluster)


    def __map_to_core(self, selected_cluster):
        while True:
            assert self.hilbert_curve
            pe = self.__pe_position(*self.hilbert_curve.pop(0))
            if pe in self.cores:
                return pe


    def __map_to_mem_ctrl(self, selected_cluster):
        return sample(self.mems, 1)[0]


    def __pe_position(self, x, y) -> int:
        if x >= self.diameter or y >= self.diameter:
            return -1
        else:
            return x * self.diameter + y


    def __belonging_layer(self, cluster: set) -> str:
        G = self.working_graph.get_data()
        # For the component with just single node, simply check its layer from G; 
        # For the component with more than one node, return the layer of its first node.
        if len(cluster) == 1:
            return G.nodes[list(cluster)[0]]["layer"]
        else:
            op_priority = {"sink": 0, "insrc": 1, "wsrc": 2, "worker": 3}
            leading_op = list(sorted(cluster, key=lambda x: op_priority[G.nodes[x]["op_type"]]))[0]
            return G.nodes[leading_op]["layer"]


    def __get_required_pe_num(self):
        comp_ops = list(filter(self.__need_pe, self.clusters))
        return len(comp_ops)


    def __need_pe(self, component: set) -> bool:
        G = self.working_graph.get_data()
        if len(component) > 1:
            return True
        else:
            ele = list(component)[0]
            return G.in_degree(ele) > 0 and G.nodes[ele]["op_type"] not in ["wsrc", "insrc"]
