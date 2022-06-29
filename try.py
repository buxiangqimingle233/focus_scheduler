import networkx as nx

G = nx.DiGraph()
G.add_node(1, delay = 2)
G.add_node(2, delay = 4)
G.add_node(3, delay = 5)
G.add_node(4, delay = 5)
G.add_edge(1,2, id = 1, size = 5)
G.add_edge(1,3, id = 2, size = 5)
G.add_edge(2,4, id = 1, size = 5)




# G.remove_edges_from([e for e in G.edges() if (e[0] == 1 and G.edges()[e]['id'] == 1)])
# while G.edges(1):
#     x = list(G.edges(1))[0]
#     id = G.edges()[x]['id']
#     G.remove_edges_from([e for e in G.edges() if (e[0] == 1 and G.edges()[e]['id'] == id)])

print(G.edges())
G.remove_nodes_from([2])
print(G.edges())

class A:
    def __init__(self) -> None:
        l = [3,4,5,6,7,8,9,1,2]
        l.sort(key=self.func)
        print(l)
        pass

    def func(self, e):
        return e


temp = A()

