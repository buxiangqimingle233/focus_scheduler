import networkx as nx

class Mapper:
    
    def __init__(self, physical_cores: list, memory_controllers: list) -> None:
        self.cores = physical_cores
        self.mems = memory_controllers

    def map(op_graph: nx.DiGraph):
        compute_operators = nx.subgraph_view(op_graph, \
            filter_node=lambda x: op_graph.nodes[x]["op_type"] in ["wsrc", "insrc"])

        data_source_operators = nx.subgraph_view(op_graph, \
            filter_node=lambda x: op_graph.nodes[x]["op_type"] in ["worker", "sink"])

    def _map_core() -> dict:
        pass

    # def _map_m():
        # pass