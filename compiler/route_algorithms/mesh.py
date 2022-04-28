import networkx as nx
from router import Router

class MeshTreeRouter(Router):

    def __init__(self, diameter) -> None:
        super().__init__(diameter)
    
    def route(self, source: int, dests: list) -> nx.DiGraph:
        tree = nx.DiGraph()
        # Connect dests directly to the source. 
        # This is very likely to violate the spatial-sim's tree-generation rules.
        tree.add_node(source, root=True)
        tree.add_nodes_from(dests, root=False)
        tree.add_edges_from((source, d) for d in dests)

        return tree


if __name__ == "__main__":
    router = MeshTreeRouter(4)
    print(router.route(1, [1, 2, 5, 6]).edges())