import networkx as nx

class Router:

    def __init__(self, diameter) -> None:
        self.diameter = diameter
    
    def route(self, source: int, dests: list) -> nx.DiGraph:
        pass