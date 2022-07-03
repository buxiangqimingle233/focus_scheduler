import networkx as nx
import matplotlib.pyplot as plt

class Router:


    def __init__(self, diameter) -> None:
        self.diameter = diameter
        self.cnt = 0
    
    def print_tree(self, tree, path):
        for i in tree.edges():
            plt.arrow(i[0] % self.diameter, i[0] // self.diameter, i[1] % self.diameter-i[0] % self.diameter, i[1] // self.diameter-i[0] // self.diameter, head_width=0.2)
            # plt.plot([i[0] % self.diameter, i[1] % self.diameter],[i[0] // self.diameter, i[1] // self.diameter])
        print(str(self.cnt), " ", tree.edges())
        
        plt.xlim(-1,8)
        plt.ylim(-1,8)
        plt.savefig('temp/'+'-'+str(self.cnt)+'.png')
        self.cnt += 1
        plt.clf()
    
    def route(self, source: int, dests: list) -> nx.DiGraph:
        pass