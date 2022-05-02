import networkx as nx
from router import Router
import copy

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






class RPMTreeRouter(Router):

    def __init__(self, diameter) -> None:
        super().__init__(diameter)

    def get_tree(self, tree, node: int, dests: list) -> None:
        if node in dests:
            dests.remove(node)
        # partition
        dests_parts = [[] for i in range(8)]
        dests_di = [[] for i in range(4)] #0,1,2,3 corresponding to N,S,W,E
        
        for i in dests:
            if i % self.diameter == node % self.diameter: #part 1,5
                if i // self.diameter < node // self.diameter: #part 1. attention, the first node is 0 here
                    dests_parts[1].append(i)
                if i // self.diameter > node // self.diameter: #part 5
                    dests_parts[5].append(i)

            if i % self.diameter < node % self.diameter: #part 2,3,4
                if i // self.diameter < node // self.diameter: #part 2
                    dests_parts[2].append(i)
                if i // self.diameter == node // self.diameter: #part 3
                    dests_parts[3].append(i)
                if i // self.diameter > node // self.diameter: #part 4
                    dests_parts[4].append(i)

            if i % self.diameter > node % self.diameter: #part 0,7,6
                if i // self.diameter < node // self.diameter: #part 0
                    dests_parts[0].append(i)
                if i // self.diameter == node // self.diameter: #part 7
                    dests_parts[7].append(i)
                if i // self.diameter > node // self.diameter: #part 6
                    dests_parts[6].append(i)
        
        # compulsory direction
        dests_di[0] += dests_parts[1]
        dests_di[1] += dests_parts[5]
        dests_di[2] += dests_parts[3]
        dests_di[3] += dests_parts[7]

        # rules
        if dests_parts[1] or dests_parts[2] or (not dests_parts[7]):
            dests_di[0] += dests_parts[0]
        else:
            dests_di[3] += dests_parts[0]

        if dests_parts[5] or dests_parts[6] or (not dests_parts[3]):
            dests_di[1] += dests_parts[4]
        else:
            dests_di[2] += dests_parts[4]

        if dests_parts[0] or (dests_parts[1] and dests_parts[2] and (not dests_parts[3])):
            dests_di[0] += dests_parts[2]
        else:
            dests_di[2] += dests_parts[2]

        if dests_parts[4] or (dests_parts[5] and dests_parts[6] and (not dests_parts[7])):
            dests_di[1] += dests_parts[6]
        else:
            dests_di[3] += dests_parts[6]

        #add nodes
        if dests_di[0]:
            tree.add_node(node-self.diameter, root=False)
            tree.add_edge(node, node-self.diameter)
            self.get_tree(tree, node-self.diameter, dests_di[0])
        
        if dests_di[1]:
            tree.add_node(node+self.diameter, root=False)
            tree.add_edge(node, node+self.diameter)
            self.get_tree(tree, node+self.diameter, dests_di[1])

        if dests_di[2]:
            tree.add_node(node-1, root=False)
            tree.add_edge(node, node-1)
            self.get_tree(tree, node-1, dests_di[2])

        if dests_di[3]:
            tree.add_node(node+1, root=False)
            tree.add_edge(node, node+1)
            self.get_tree(tree, node+1, dests_di[3])

    
    def route(self, source: int, dests: list) -> nx.DiGraph:
        tree = nx.DiGraph()
        tree.add_node(source, root=True)
        self.get_tree(tree, source, dests)

        return tree


if __name__ == "__main__":
    router = RPMTreeRouter(4)
    print(router.route(9, [0, 2, 3, 13, 15]).edges())