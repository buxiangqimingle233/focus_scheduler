#from tkinter.tix import MAX
from xmlrpc.client import MAXINT
import networkx as nx
from router import Router
import copy
import random
import numpy as np

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
        if (p_node != None and pp_node != None):
            if (p_node // self.diameter == pp_node // self.diameter) or (p_node % self.diameter == v % self.diameter):
                if len(list(nx.neighbors(tree, p_node))) == 1 and (not tree.nodes[p_node]['dest']):
                    tree.remove_edge(pp_node, p_node)
                    tree.remove_edge(p_node, v)
                    tree.remove_node(p_node)
                    tree.add_edge(pp_node, v)
                    p_node = pp_node
                    if_pruned_all = False

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

    def tree_pruner(self, tree, v, p_node, pp_node):
        neighbors = nx.neighbors(tree, v)
        if_pruned_all = True
        if (p_node != None and pp_node != None):
            if (p_node // self.diameter == pp_node // self.diameter) or (p_node % self.diameter == v % self.diameter):
                if len(list(nx.neighbors(tree, p_node))) == 1 and (not tree.nodes[p_node]['dest']):
                    tree.remove_edge(pp_node, p_node)
                    tree.remove_edge(p_node, v)
                    tree.remove_node(p_node)
                    tree.add_edge(pp_node, v)
                    p_node = pp_node
                    if_pruned_all = False

        for i in list(neighbors):
            if not self.tree_pruner(tree, i, v, p_node):
                if_pruned_all = False
        return if_pruned_all

    def dest_pruner(self, tree, v, p_node):
        neighbors = list(nx.neighbors(tree, v))
        if_pruned_all = True
        if len(neighbors) == 0 and p_node and (not tree.nodes[v]['dest']):
            tree.remove_edge(p_node, v)
            tree.remove_node(v)
            if_pruned_all = False
        
        for i in neighbors:
            if not self.dest_pruner(tree, i, v):
                if_pruned_all = False
        return if_pruned_all
    
    def route(self, source: int, dests: list, x = -1) -> nx.DiGraph:
        tree = nx.DiGraph()
        tree.add_node(source, root=True)

        if x == -1: #we allow setting x manually
            x = random.randint(0, 15) #four bit corresponding to LTBw, LTBn, LTBe, LTBs
        # LTBw = x & 8, LTBn = x & 4, LTBe = x & 2, LTBs = x & 1
        # RTBs = not LTBw, RTBw = not LTBn, RTBn = not LTBe, RTBe = not LTBs
        
        delta = [self.diameter, 1, -self.diameter, -1]
        for i in range(4): #to traversing from s to w
            LTB = x & (1 << i)
            RTB = not (x & 1 << ((i + 3) % 4))

            current_node = source
            next_node = source + delta[i]
            while ((next_node % self.diameter == current_node % self.diameter) or (next_node // self.diameter == current_node // self.diameter)) \
                  and next_node >= 0 and next_node < self.diameter * self.diameter:
                tree.add_node(current_node)
                tree.add_node(next_node)
                tree.add_edge(current_node, next_node)
                if LTB:
                    current_node2 = next_node
                    next_node2 = current_node2 + delta[(i+1)%4]
                    while ((next_node2 % self.diameter == current_node2 % self.diameter) or (next_node2 // self.diameter == current_node2 // self.diameter))\
                          and next_node2 >= 0 and next_node2 < self.diameter * self.diameter:
                        tree.add_node(current_node2)
                        tree.add_node(next_node2)
                        tree.add_edge(current_node2, next_node2)

                        current_node2 += delta[(i+1)%4]
                        next_node2 += delta[(i+1)%4]
                if RTB:
                    current_node2 = next_node
                    next_node2 = current_node2 + delta[(i+3)%4]
                    while ((next_node2 % self.diameter == current_node2 % self.diameter) or (next_node2 // self.diameter == current_node2 // self.diameter))\
                          and next_node2 >= 0 and next_node2 < self.diameter * self.diameter:
                        tree.add_node(current_node2)
                        tree.add_node(next_node2)
                        tree.add_edge(current_node2, next_node2)

                        current_node2 += delta[(i+3)%4]
                        next_node2 += delta[(i+3)%4]
                
                current_node += delta[i]
                next_node += delta[i]
        
        for i in tree.nodes():
            if i in dests:
                tree.nodes[i]['dest'] = True
            else:
                tree.nodes[i]['dest'] = False
        
        while not self.dest_pruner(tree, source, None):
            pass

        while not self.tree_pruner(tree, source, None, None):
            pass

        return tree


class BAMTreeRouter(Router):

    def __init__(self, diameter) -> None:
        super().__init__(diameter)

    
    def tree_pruner(self, tree, v, p_node, pp_node):
        neighbors = nx.neighbors(tree, v)
        if_pruned_all = True
        if (p_node != None and pp_node != None):
            if (p_node // self.diameter == pp_node // self.diameter) or (p_node % self.diameter == v % self.diameter):
                if len(list(nx.neighbors(tree, p_node))) == 1 and (not tree.nodes[p_node]['dest']):
                    tree.remove_edge(pp_node, p_node)
                    tree.remove_edge(p_node, v)
                    tree.remove_node(p_node)
                    tree.add_edge(pp_node, v)
                    p_node = pp_node
                    if_pruned_all = False

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
        
        #we randomly set the congestion bit
        N_ne = random.randint(0,1)
        E_ne = not N_ne
        N_nw = random.randint(0,1)
        W_nw = not N_nw
        S_sw = random.randint(0,1)
        W_sw = not S_sw
        S_se = random.randint(0,1)
        E_se = not S_se

        # rules
        p = np.zeros(8)
        n_p = np.zeros(8)
        for i in range(8):
            p[i] = len(dests_parts[i])
            n_p[i] = not p[i]
        N_p0 = p[1]*n_p[7] + p[1]*p[7]*N_ne + n_p[1]*n_p[7]*N_ne
        E_p0 = n_p[1]*p[7] + p[1]*p[7]*E_ne + n_p[1]*n_p[7]*E_ne
        N_p2 = p[1]*n_p[3] + p[1]*p[3]*N_nw + n_p[1]*n_p[3]*N_nw
        W_p2 = n_p[1]*p[3] + p[1]*p[3]*W_nw + n_p[1]*n_p[7]*W_nw
        S_p4 = p[5]*n_p[3] + p[5]*p[3]*S_sw + n_p[5]*n_p[3]*S_sw
        W_p4 = n_p[5]*p[3] + p[5]*p[3]*W_sw + n_p[5]*n_p[3]*W_sw
        S_p6 = p[5]*n_p[7] + p[5]*p[7]*S_se + n_p[5]*n_p[7]*S_se
        E_p6 = n_p[5]*p[7] + p[5]*p[7]*E_se + n_p[5]*n_p[7]*E_se

        if N_p0:
            dests_di[0] += dests_parts[0]
        else:
            dests_di[3] += dests_parts[0]

        if N_p2:
            dests_di[0] += dests_parts[2]
        else:
            dests_di[2] += dests_parts[2]

        if S_p4:
            dests_di[1] += dests_parts[4]
        else:
            dests_di[2] += dests_parts[4]

        if S_p6:
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

class Steiner_TreeRouter(Router):

    def __init__(self, diameter) -> None:
        super().__init__(diameter)

    def tree_pruner(self, tree, v, p_node, pp_node):
        neighbors = nx.neighbors(tree, v)
        if_pruned_all = True
        if (p_node != None and pp_node != None):
            if (p_node // self.diameter == pp_node // self.diameter) or (p_node % self.diameter == v % self.diameter):
                if len(list(nx.neighbors(tree, p_node))) == 1 and (not tree.nodes[p_node]['dest']):
                    tree.remove_edge(pp_node, p_node)
                    tree.remove_edge(p_node, v)
                    tree.remove_node(p_node)
                    tree.add_edge(pp_node, v)
                    
                    p_node = pp_node
                    if_pruned_all = False

        for i in list(neighbors):
            if not self.tree_pruner(tree, i, v, p_node):
                if_pruned_all = False
        return if_pruned_all
    
    def p2p_distance(self, x, y):
        return abs(x // self.diameter - y // self.diameter) + abs(x % self.diameter - y % self.diameter)

    def p2t_distance(self, x, tree):
        min_dis = MAXINT
        for v in tree.nodes():
            min_dis = min(self.p2p_distance(v, x), min_dis)
        return min_dis

    def add_point(self, x, tree):
        min_dis = MAXINT
        connect_v = None
        for v in tree.nodes():
            if min_dis > self.p2p_distance(v, x):
                connect_v = v
                min_dis = self.p2p_distance(v, x)

        temp = random.randint(0,1) #0:x-y path, 1:y-x path
        if temp:
            mid_v = x - x % self.diameter + connect_v % self.diameter
            pre_node = connect_v
            step = self.diameter
            if connect_v != mid_v:
                step = ((mid_v-connect_v)//(abs(mid_v-connect_v))) * self.diameter
            for i in range(connect_v, mid_v, step):
                if i != connect_v:
                    tree.add_node(i)
                    tree.add_edge(pre_node, i)
                    pre_node = i

            if mid_v != connect_v:
                tree.add_node(mid_v)
                tree.add_edge(pre_node, mid_v)

            pre_node = mid_v
            step = 1
            if x != mid_v:
                step = (x-mid_v)//(abs(x-mid_v))
            for i in range(mid_v, x, step):
                if i != mid_v:
                    tree.add_node(i)
                    tree.add_edge(pre_node, i)
                    pre_node = i
            if mid_v != x:   #to avoid add self-to-self edge
                tree.add_node(x)
                tree.add_edge(pre_node, x)

        else:
            mid_v = connect_v - connect_v % self.diameter + x % self.diameter
            pre_node = connect_v

            step = 1
            if connect_v != mid_v:
                step = (mid_v-connect_v)//(abs(mid_v-connect_v))

            for i in range(connect_v, mid_v, step):
                if i != connect_v:
                    tree.add_node(i)
                    tree.add_edge(pre_node, i)
                    pre_node = i
            
            if mid_v != connect_v:
                tree.add_node(mid_v)
                tree.add_edge(pre_node, mid_v)

            pre_node = mid_v
            step = self.diameter
            if x != mid_v:
                step = ((x-mid_v)//(abs(x-mid_v))) * self.diameter
            for i in range(mid_v, x, step):
                if i != mid_v:
                    tree.add_node(i)
                    tree.add_edge(pre_node, i)
                    pre_node = i
            if mid_v != x:
                tree.add_node(x)
                tree.add_edge(pre_node, x)
    
    def route(self, source: int, dests: list) -> nx.DiGraph:
        tree = nx.DiGraph()
        tree.add_node(source, root=True)
        dests_temp = copy.deepcopy(dests)

        while dests_temp:
            min_dis = MAXINT
            add_p = None
            for i in dests_temp:  #to choose a point which is closest to the tree
                dis = self.p2t_distance(i, tree)
                if min_dis > dis:
                    min_dis = dis
                    add_p = i
            self.add_point(add_p, tree)
            dests_temp.remove(add_p)

        print(tree.edges())
        for i in tree.nodes():
            if i in dests:
                tree.nodes[i]['dest'] = True
            else:
                tree.nodes[i]['dest'] = False

        while not self.tree_pruner(tree, source, None, None):
            pass

        return tree


if __name__ == "__main__":
    #router = WhirlTreeRouter(4)
    #router = RPMTreeRouter(4)
    #router = BAMTreeRouter(4)
    router = Steiner_TreeRouter(4)
    print(router.route(9, [0, 2, 3, 13, 15]).edges())