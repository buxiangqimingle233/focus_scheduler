import sys
from math import ceil, log2
from random import sample, uniform

from compiler.mapping_algorithms.hilbert import hilbert_curve, hilbert_map
from op_graph.micro_op_graph import MicroOpGraph
from mapper import Mapper

class HilbertMapper(Mapper):
    '''To guarantee adjacent PEs are allocated continuously, we sort PEs in the order of Hilbert Curve, \
        and map the operators of the same layer continuously according to that curve. 
    '''

    def __init__(self, op_graph: MicroOpGraph, physical_layout: dict, diameter: int, virtualization=True) -> None:
        super().__init__(op_graph, physical_layout)
    
        print("core: ", self.cores)
        print("mem controller: ", self.mems)
        # Check whether the playboard has enough pes to map.
        pe_cnt = self.__get_required_pe_num()

        if not virtualization and pe_cnt > len(self.cores):
            print("ERROR: the array doesn't have enough processing elements to run the task requiring"\
                  " {} pes while we just have {}".format(pe_cnt, len(self.cores)), 
                  file=sys.stderr)
            print("Mapping fails, abort. ")
            exit(0)

        # self.sorted_clusters = sorted(self.clusters, key=self.__belonging_layer)

        self.diameter = diameter
        if diameter & (diameter - 1) == 0:
            quantilized_diameter = diameter
        else:
            quantilized_diameter = 2 ** ceil(log2(diameter))
        
        self.virtualization = virtualization
        self.hilbert_curve = list(hilbert_map(log2(quantilized_diameter)))
        self.idx = 0


    def _select(self, clusters: list) -> set:
        if not self.clusters:
            return []
        else:
            return self.clusters.pop(0)


    def _map(self, selected_cluster: set) -> int:
        if self.__need_pe(selected_cluster):
            return self.__map_to_core(selected_cluster)
        else:
            return self.__map_to_mem_ctrl(selected_cluster)


    def __map_to_core(self, selected_cluster):
        while True:
            assert self.idx < len(self.hilbert_curve)
            # pe = self.__pe_position(*self.hilbert_curve.pop(0))
            pe = self.__pe_position(*self.hilbert_curve[self.idx])
            self.idx = self.idx + 1
            if self.virtualization:
                self.idx = self.idx % len(self.hilbert_curve)
            if pe in self.cores:
                # return sample(self.cores, 1)[0]
                return pe

    def __map_to_mem_ctrl(self, selected_cluster):
        return sample(self.mems, 1)[0]


    def __pe_position(self, x, y) -> int:
        if x >= self.diameter or y >= self.diameter:
            return -1
        else:
            return x * self.diameter + y


    def __belonging_layer(self, cluster: set) -> str:
        G = self.working_graph.get_graph()
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
        G = self.working_graph.get_graph()
        types = {G.nodes[ele]["op_type"] for ele in component}
        # pure "insrc" or "wsrc" do not need cores
        if not types & {"worker", "sink"}:
            return False
        else:
            return True


class RandomizedHilbertMapper(HilbertMapper):
    def __init__(self, op_graph: MicroOpGraph, physical_layout: dict, diameter: int, virtualization=True) -> None:
        super().__init__(op_graph, physical_layout, diameter, virtualization)
        # randomize the starting index
        self.idx = int(uniform(0, len(self.hilbert_curve)))

    def _map(self, selected_cluster: set) -> int:
        if self.__need_pe(selected_cluster):
            return self.__map_to_core(selected_cluster)
        else:
            return self.__map_to_mem_ctrl(selected_cluster)

    def __map_to_core(self, selected_cluster):
        while True:
            assert self.idx < len(self.hilbert_curve)
            # pe = self.__pe_position(*self.hilbert_curve.pop(0))
            pe = self.__pe_position(*self.hilbert_curve[self.idx])
            self.idx = self.idx + 1
            if self.virtualization:
                self.idx = self.idx % len(self.hilbert_curve)
            if pe in self.cores:
                # return sample(self.cores, 1)[0]
                return pe

    def __map_to_mem_ctrl(self, selected_cluster):
        return sample(self.mems, 1)[0]


    def __pe_position(self, x, y) -> int:
        if x >= self.diameter or y >= self.diameter:
            return -1
        else:
            return x * self.diameter + y

    def __need_pe(self, component: set) -> bool:
        G = self.working_graph.get_graph()
        types = {G.nodes[ele]["op_type"] for ele in component}
        # for single layer, only "wsrc" does not need cores
        if types & {"wsrc"}:
            return False
        else:
            return True