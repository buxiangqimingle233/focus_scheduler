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

    
    def tree_pruner(self, tree, v, p_node, pp_node):
        neighbors = nx.neighbors(tree, v)
        if_pruned_all = True
        if (p_node and pp_node):
            if ((p_node // self.diameter == pp_node // self.diameter) and (p_node // self.diameter == v // self.diameter))  \
                or ((p_node % self.diameter == pp_node % self.diameter) and (p_node % self.diameter == v % self.diameter))  \
                or ((p_node // self.diameter == pp_node // self.diameter) and (p_node % self.diameter == v % self.diameter)):
                if len(list(nx.neighbors(tree, p_node))) == 1 and (not tree.nodes[p_node]['dest']):
                    tree.remove_edge(pp_node, p_node)
                    tree.remove_edge(p_node, v)
                    tree.remove_node(p_node)
                    tree.add_edge(pp_node, v)

        for i in list(neighbors):
            if not self.tree_pruner(tree, i, v, p_node):
                if_pruned_all = False
        return if_pruned_all

        


    def get_tree(self, tree, node: int, dests: list) -> None:
        if node in dests:
            tree.nodes[node]['dest'] = True
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
            tree.nodes[node-self.diameter]['dest'] = False
            tree.add_edge(node, node-self.diameter)
            self.get_tree(tree, node-self.diameter, dests_di[0])
        
        if dests_di[1]:
            tree.add_node(node+self.diameter, root=False)
            tree.nodes[node+self.diameter]['dest'] = False
            tree.add_edge(node, node+self.diameter)
            self.get_tree(tree, node+self.diameter, dests_di[1])

        if dests_di[2]:
            tree.add_node(node-1, root=False)
            tree.nodes[node-1]['dest'] = False
            tree.add_edge(node, node-1)
            self.get_tree(tree, node-1, dests_di[2])

        if dests_di[3]:
            tree.add_node(node+1, root=False)
            tree.nodes[node+1]['dest'] = False
            tree.add_edge(node, node+1)
            self.get_tree(tree, node+1, dests_di[3])

    
    def route(self, source: int, dests: list) -> nx.DiGraph:
        tree = nx.DiGraph()
        tree.add_node(source, root=True)
        self.get_tree(tree, source, dests)
        while not self.tree_pruner(tree, source, None, None):
            pass

        return tree


class WhirlTreeRouter(Router):

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
    router = RPMTreeRouter(4)
    print(router.route(9, [0, 2, 3, 13, 15]).edges())