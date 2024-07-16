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


        # for i in range(len(cycles_base)):
        #     self.cycles_index[cycles_base[i]] = i
        
        # for blocked_edge in G.blocked_edges:
        #     self.dict_blocked_edges[blocked_edge] = []
        
        # for edge in G.edges:
        #     self.edge_in_cycle[edge] = []
        
        # for cycle in self.cycles:
        #     edges = self.get_edges(self.cycles_index[cycle])
        #     for edge in edges:
        #         self.edge_in_cycle[edge].append(self.cycles_index[cycle])

        # for cycle in self.cycles:
        #     for i in range(len(cycle)-1):
        #         edge = (min(cycle[i],cycle[i+1]),max(cycle[i],cycle[i+1]))
        #         if edge in self.dict_blocked_edges:
        #             self.dict_blocked_edges[edge].append(cycle)
        
        # for blocked_edge in G.blocked_edges:
        #     if self.dict_blocked_edges[blocked_edge] == []:
        #         del self.dict_blocked_edges[blocked_edge]

        # for edge in tqdm(G.edges):
        #     for cycle1 in self.edge_in_cycle[edge]:
        #         for cycle2 in self.edge_in_cycle[edge]:
        #             if cycle1 != cycle2:
        #                 self.Graph_cycles.add_edge(cycle1,cycle2)

    
    
    # def get_edges(self,cycle_index):
    #     cycle = self.cycles[cycle_index]
    #     edges = set()
    #     for i in range(len(cycle)-1):
    #         edges.add((min(cycle[i],cycle[(i+1)]),max(cycle[i],cycle[(i+1)])))
    #     return edges
    
    # def construct_cycle(self,edges):
    #     a,b = edges.pop(0)
    #     cycle = [a,b]
    #     while edges != []:
    #         x,y = edges.pop(0)
    #         if x == cycle[-1]:
    #             cycle.append(y)
    #         elif y == cycle[-1]:
    #             cycle.append(x)
    #         else:
    #             edges.append((x,y))
    #     return cycle

    

    # def get_full_cycle(self,path):
    #     Edges = set()
    #     E = []
    #     for cycle_index in path:
    #         Edges = Edges.symmetric_difference(self.get_edges(cycle_index))

    #     cpt = 0
    #     for edge in Edges:
    #         if edge in self.blocked_edges:
    #             cpt += 1 

    #     if cpt % 2 == 0:
    #         return False
    #     else:
    #         E = list(Edges)
    #         return self.construct_cycle(E)
    
    # def find_cycle_impair(self):
    #     path = nx.shortest_path(self.Graph_cycles,0,1)

    #     while not self.get_full_cycle(path):
    #         cpt1 = np.random.randint(0,len(self.dict_blocked_edges))
    #         blocked_edge = list(self.blocked_edges)[cpt1]
    #         cycle_blocked = self.dict_blocked_edges[blocked_edge]
    #         cycle_blocked_index = self.cycles_index[cycle_blocked[np.random.randint(0,len(cycle_blocked))]]
    #         cycle_index = np.random.randint(0,len(self.cycles))
    #         path = nx.shortest_path(self.Graph_cycles,cycle_index,cycle_blocked_index)
    #     return self.get_full_cycle(path)


            
        
        