import networkx as nx
import pickle

G = nx.DiGraph()
G.add_node(1, delay = 2)
G.add_node(2, delay = 4)
G.add_node(3, delay = 5)
G.add_node(4, delay = 5)
G.add_edge(1,2, id = 1, size = 5)
G.add_edge(1,3, id = 2, size = 5)
G.add_edge(2,4, id = 1, size = 5)

for _, __, ettr in G.edges(data=True):
    print(_)
    print(__)
    print(ettr)
    print()


# class A:
#     def __init__(self):
#         self.a = 10
#         self.b = 333

# # x = A()
# # pickle.dump(x, open('try.pickle', 'wb'))

# x = pickle.load(open('try.pickle', 'rb'))

# print(x.a, " ", x.b)

