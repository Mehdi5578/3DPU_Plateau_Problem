import networkx as nx
from .CreatingGraph import *
from tqdm import tqdm

class Graph_Cycles:
    def __init__(self, Edges, Marked_edges):
        self.cycles = []
        self.cycles_index = {}
        self.dict_blocked_edges = {}
        self.Graph_cycles = nx.Graph()
        self.edge_in_cycle = {}

        G = GraphGrid3D(Edges,Marked_edges)
        self.mapping_GC = G.mapping
        self.edges = G.edges
        self.marked_edges = Marked_edges

        self.blocked_edges = set(G.blocked_edges)
        new_G = nx.Graph()
        self.b_1 = set()
        self.b_2 = set()
        new_G.add_edges_from(list(G.edges))

        cycles_base = nx.cycle_basis(new_G)
        
        cycles_base = [tuple(cl+[cl[0]]) for cl in cycles_base]
        self.cycles = cycles_base

        for cycle in self.cycles:
            parite = self.f(cycle)
            if parite == 1:
                self.b_1.add(cycle)
            elif parite == 0:
                self.b_2.add(cycle)
        
    def f(self,cycle):
        cpt  = 0
        for i in range(len(cycle)):
            edge = (min(cycle[i],cycle[(i+1)%len(cycle)]),max(cycle[i],cycle[(i+1)%len(cycle)]))
            cpt = cpt + (edge in self.blocked_edges)
        return cpt % 2





            
        
        