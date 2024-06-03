from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import random
import csv
import os
import pickle
from typing import Union, Optional

from tqdm import tqdm
import networkx as nx
dim = int(3)


# Here each residual is stored in a tuple of five elements fist three are the smallest cooridnates of 
# the face it lays upon, the second is the the axis = 0,1,2 and the third is the value = -1 or 1


class Resiuals():
    
    def __init__(self,data_phase):
        self.data = data_phase
        self.Res  = {}
        self.list_res = []
        self.Res_graph = {}
        self.X,self.Y, self.Z = self.data.shape
        self.connex = {}
        self.indirected_graph = {}
        self.connected_components = {}
        self.mapping = []
        self.res_ordre = {}
        self.Separate_graphs = {}
        self.cycles = []
        self.visited = []
        self.incycles = []
        self.graph_res_networkx = nx.DiGraph()
        self.starting_open_paths = []
        self.inverted_dictionnary = dict()
        self.closing_edges = []
        self.open_paths = []

    def wrap(self,phi) :
        return np.round(phi / (2 * np.pi)).astype(int)

    def grad(self,psi, a: int):
        return np.diff(psi, axis=a)

    def wrap_grad(self,psi, a: int):
        return self.wrap(self.grad(psi, a))

    def residuals(self, a: int):
        assert(a >= 0 and a < dim)
        ax, ay = (a + np.arange(1, dim)) % dim
        gx = self.wrap_grad(self.data, a=ax)
        gy = self.wrap_grad(self.data, a=ay)
        self.Res[a] = np.diff(gy, axis=ax) - np.diff(gx, axis=ay)

    def list_residuals(self) -> None:
        for a in range(dim):
            self.residuals(a)
            I,J,K = np.where(self.Res[a] != 0)
            for ind in range(len(I)):
                x = I[ind]
                y = J[ind]
                z = K[ind]
                value = self.Res[a][x,y,z]
                self.list_res.append((x,y,z,a,value))
    
    def map_nodes(self):
        self.list_residuals()
        self.mapping = self.list_res
        self.res_ordre = {residual: index for index, residual in enumerate(self.mapping)}


    def nodes_of_not_boundary(self, Residual):
        "Add the sons of the Residual"  
        i, j, k, axe, value = Residual
        if axe == 0:
            if value == 1 and i < self.X - 1:
                if self.Res[0][i + 1, j, k] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i + 1, j, k, 0, 1)])
                if self.Res[1][i, j, k] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j, k, 1, -1)])
                if self.Res[2][i, j, k] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j, k, 2, -1)])
                if self.Res[1][i, j + 1, k] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j + 1, k, 1, 1)])
                if self.Res[2][i, j, k + 1] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j, k + 1, 2, 1)])

            if value == -1 and i > 0:
                if self.Res[0][i - 1, j, k] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i - 1, j, k, 0, -1)])
                if self.Res[1][i - 1, j, k] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i - 1, j, k, 1, -1)])
                if self.Res[2][i - 1, j, k] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i - 1, j, k, 2, -1)])
                if self.Res[1][i - 1, j + 1, k] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i - 1, j + 1, k, 1, 1)])
                if self.Res[2][i - 1, j, k + 1] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i - 1, j, k + 1, 2, 1)])

        elif axe == 1:
            if value == 1 and j < self.Y - 1 :
                if self.Res[1][i, j + 1, k] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j + 1, k, 1, 1)])
                if self.Res[0][i, j, k] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j, k, 0, -1)])
                if self.Res[2][i, j, k] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j, k, 2, -1)])
                if self.Res[0][i + 1, j, k] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i + 1, j, k, 0, 1)])
                if self.Res[2][i, j, k + 1] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j, k + 1, 2, 1)])

            if value == -1 and j > 0:
                if self.Res[1][i, j - 1, k] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j - 1, k, 1, -1)])
                if self.Res[0][i, j - 1, k] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j - 1, k, 0, -1)])
                if self.Res[2][i, j - 1, k] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j - 1, k, 2, -1)])
                if self.Res[0][i + 1, j - 1, k] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i + 1, j - 1, k, 0, 1)])
                if self.Res[2][i, j - 1, k + 1] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j - 1, k + 1, 2, 1)])
        
        
        elif axe == 2:
            if value == 1 and k < self.Z - 1:
                if self.Res[2][i, j, k + 1] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j, k + 1, 2, 1)])
                if self.Res[0][i, j, k] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j, k, 0, -1)])
                if self.Res[1][i, j, k] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j, k, 1, -1)])
                if self.Res[0][i + 1, j, k] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i + 1, j, k, 0, 1)])
                if self.Res[1][i, j + 1, k] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j + 1, k, 1, 1)])

            if value == -1 and k > 0:
                if self.Res[2][i, j, k - 1] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j, k - 1, 2, -1)])
                if self.Res[0][i, j, k - 1] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j, k - 1, 0, -1)])
                if self.Res[1][i, j, k - 1] == -1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j, k - 1, 1, -1)])
                if self.Res[0][i + 1, j, k - 1] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i + 1, j, k - 1, 0, 1)])
                if self.Res[1][i, j + 1, k - 1] == 1:
                    self.Res_graph[self.res_ordre[Residual]].append(self.res_ordre[(i, j + 1, k - 1, 1, 1)])





    def create_graph(self):
        for res in self.list_res:
            self.Res_graph[self.res_ordre[res]] = []
        for res in tqdm(self.list_res):
            self.nodes_of_not_boundary(res)
        
        for val in self.Res_graph.values():
            self.inverted_dictionnary[tuple(sorted(val))] = []

        for key,value in self.Res_graph.items():
            self.inverted_dictionnary[tuple(sorted(value))].append(key)



    def untangle_graph(self):
        triples = [value for key,value in self.inverted_dictionnary.items() if len(key) == 3]
        doubles = [value for key,value in self.inverted_dictionnary.items() if len(key) == 2]
        for double in tqdm(doubles):
            assert len(self.Res_graph[double[0]]) == 2, "Error this one has not two"
            children = self.Res_graph[double[0]]
            self.Res_graph[double[0]] =  [children[0]]
            self.Res_graph[double[1]] =  [children[1]]
        for triple in tqdm(triples):
            assert(len(self.Res_graph[triple[0]]) == 3), "Error this has not three"
            children = self.Res_graph[triple[0]]
            self.Res_graph[triple[0]] = [children[0]]
            self.Res_graph[triple[1]] = [children[1]]
            self.Res_graph[triple[2]] = [children[2]]
    
    def create_graph_networkx(self):
        for key in self.Res_graph.keys():
            for value in self.Res_graph[key]:
                self.graph_res_networkx.add_edge(key,value)

    def reduire_cycle(self,cycle):
        Cycles = []
        subgraph = self.graph_res_networkx.subgraph(cycle).copy()
        base_cycle = nx.cycle_basis(subgraph)
        while len(base_cycle) > 0:
            smallest_base_cycle = min(base_cycle, key=len)
            Cycles.append(smallest_base_cycle + [smallest_base_cycle[0]])
            subgraph.remove_node(smallest_base_cycle[0])
            base_cycle = nx.cycle_basis(subgraph)
        return Cycles

    def reduire_cycles(self):
        new_Cycles = []
        for cycle in tqdm(self.cycles):
            Cycles = self.reduire_cycle(cycle)
            for cycle in Cycles:
                new_Cycles.append(cycle)
        self.cycles = new_Cycles
    


    def detect_connex(self ,node ,colour):
        visited = set()
        stack = set()
        stack.add(node)
        while len(stack) != 0:
            new_stack = set()
            for next in stack :
                self.connex[next] = colour
                visited.add(next)
                for neighbour in self.indirected_graph[next]:
                    if neighbour not in visited:
                        new_stack.add(neighbour)
                        visited.add(neighbour)
                        self.connex[neighbour] = colour
                stack = new_stack


    

    def identify_connected_component(self):

        for node in range(len(self.mapping)):
            self.connex[node] = 0
        colour = 1

        for node in tqdm(range(len(self.mapping))):
            if self.connex[node] == 0:
                self.detect_connex(node,colour)
                colour += 1

        for k in range(colour):
            self.connected_components[k+1] = []

        for node in range(len(self.mapping)):
            self.connected_components[self.connex[node]].append(node)
    
    
    def group_by_connected_compo(self):
        
        colours = len(self.connected_components)

        for colour in range(colours):
            self.Separate_graphs[colour + 1] = {}
            for node in self.connected_components[colour +1]:
                self.Separate_graphs[colour +1][node]= []

        for node in range(len(self.mapping)):
            colour = self.connex[node]
            self.Separate_graphs[colour][node] = self.Res_graph[node]
    
    def is_boundary(self,node):
        i,j,k,axe,value = self.mapping[node]
        if axe == 0 and i in [self.X - 1, 0]:
            return True
        if axe == 1 and j in [self.Y - 1, 0]:
            return True
        if axe == 2 and k in [self.Z - 1, 0]:
            return True
        return False
    
    def is_entering_from_boundary(self,node):
        i,j,k,axe,value = self.mapping[node]
        if axe == 0 and i == 0 and value == 1:
            return True
        if axe == 0 and i == self.X - 1 and value == -1:
            return True
        if axe == 1 and j == 0 and value == 1:
            return True
        if axe == 1 and j == self.Y - 1 and value == -1:
            return True
        if axe == 2 and k == 0 and value == 1:
            return True
        if axe == 2 and k == self.Z - 1 and value == -1:
            return True
        return False
   
    def iterative_dfs(self, start):
        stack = [(start, [start])]
        self.visited[start] = True
        path_dic = {start: 0}

        while stack:
            v, path, = stack.pop()
            self.visited[v] = True
            path_dic[v] = len(path) - 1

            for neighbour in self.Res_graph[v]:
                if not self.visited[neighbour]:
                    newPath = path + [neighbour]
                    stack.append((neighbour, newPath))
                elif neighbour in path:
                    cycle_start_index = path_dic[neighbour]
                    cycle = path[cycle_start_index:] + [neighbour]

                    self.cycles.append(cycle)
                    for point in cycle:
                        self.visited[point] = True
                        self.incycles[point] = True
                    return True

        return stack

    def detect_cycles_iterative(self):
        for i in tqdm(range(len(self.mapping))):
            if not self.visited[i]:
                self.iterative_dfs(i)
                
    def fill_starting_open_paths(self):
        """This should be after detecting the closed cycles"""
        self.starting_open_paths = []
        for node in tqdm(range(len(self.mapping))):
            if self.is_entering_from_boundary(node):
                self.starting_open_paths.append(node)
        


    def fill_open_paths(self):
        self.incycles = [False]*len(self.mapping)
        self.open_paths = []
        self.fill_starting_open_paths()
        layer = set(self.starting_open_paths)
        antecedant = dict()
        paths = dict()
        visited = set()
        for point in tqdm(layer):
            paths[point] = [point]
            antecedant[point] = point
        cpt = 0
        while layer != {-1}:
            cpt += 1
            new_layer = set()
            for point in (layer):
                if point != -1 :
                    next_nodes = [n for n in self.Res_graph[point] if (n not in visited)]
                    if next_nodes:
                        next_node = next_nodes[0]
                        paths[antecedant[point]].append(next_node)
                        self.incycles[point] = True
                        antecedant[next_node] = antecedant[point]
                        visited.add(next_node)
                    else:
                        next_node = -1
                    new_layer.add(next_node)
            assert layer != new_layer, "there is a repetition"
            layer = new_layer

        self.open_paths = list(paths.values())
        for cyc in self.open_paths:
            self.graph_res_networkx.add_edge(cyc[-1],cyc[0])
            self.closing_edges.append((cyc[-1],cyc[0]))
        
        # print("now separating the open paths")
        # new_cycles = []
        # for cycle in tqdm(self.open_paths):
        #     new_cycles += self.untangle_2(cycle)
        
        # self.open_paths = []

        # for open_path in tqdm(new_cycles):
        #     ind = 0
        #     if open_path[ind+1] not in self.graph_res_networkx.neighbors(open_path[ind]):
        #         self.open_paths.append(open_path[::-1])
        #     else:
        #         self.open_paths.append(open_path)
    
        
        


    def detect_cycles(self):
        not_visited = set(self.Res_graph.keys())
        for p in self.open_paths:
            not_visited.difference_update(p)
        while not_visited:
            node = not_visited.pop()
            path = [node]
            next_node  = self.Res_graph[node][0]
            while next_node != node:
                if not self.Res_graph[next_node]:
                    print(next_node)
                path.append(next_node)
                next_node = self.Res_graph[next_node][0]
            path.append(node)
            not_visited.difference_update(path)
            self.cycles.append(path)
        new_cycles = []
        for cycle in tqdm(self.cycles):
            new_cycles += self.untangle_2(cycle)

        self.cycles = []

        for cycle in tqdm(new_cycles):
            ind = 0
            if cycle[ind+1] not in self.graph_res_networkx.neighbors(cycle[ind]):
                self.cycles.append(cycle[::-1])
            else:
                self.cycles.append(cycle)



    
    def untangle_partial(self,cycle):
        Cycle_edges = set([(cycle[i],cycle[(i+1)%len(cycle)]) for i in range(len(cycle))])
        subgraph = self.graph_res_networkx.subgraph(cycle).copy()
        degree_2 = [point for point in subgraph if subgraph.out_degree(point) == 2]
        
        old_edges = []
        new_edges = []
        Dict_couples = dict()
        for point in degree_2:
            e1,e2 = subgraph.neighbors(point)
            if (min(e1,e2),max(e1,e2)) in Dict_couples:
                Dict_couples[(min(e1,e2),max(e1,e2))].append(point)
            else:
                Dict_couples[(min(e1,e2),max(e1,e2))] = [point]
        couple = list(Dict_couples)[0]
        s1,s2 = couple
        if len(Dict_couples[couple]) == 1:
            print(cycle)
        e1,e2 = Dict_couples[couple]

        if (e1,s1) in Cycle_edges :
            subgraph.remove_edge(e1,s1)
            subgraph.remove_edge(e2,s2)
            old_edges += [(e1,s1),(e2,s2)]
            new_edges += [(e2,s1),(e1,s2)]
        else:
            assert (e1,s2) in Cycle_edges, ((e1,s2),cycle,(e1,s1))
            subgraph.remove_edge(e1,s2)
            subgraph.remove_edge(e2,s1)
            old_edges += [(e1,s2),(e2,s1)]
            new_edges += [(e1,s1),(e2,s2)]
        
            # degree_3 = [point for point in subgraph if subgraph.out_degree(point) == 3]
            # Dict_trouples = dict()

            # for point in degree_3:
            #     e1,e2,e3 = subgraph.neighbors(point)
            #     e1,e2,e3 = sorted([e1,e2,e3])
            #     if (e1,e2,e3) in Dict_trouples:
            #         Dict_trouples[(e1,e2,e3)].append(point)
            #     else:
            #         Dict_trouples[(e1,e2,e3)] = [point]
            

            # if Dict_trouples:
            #     trouple = list(Dict_trouples.keys())[0]
            #     s1,s2,s3 = trouple
            #     e1,e2,e3 = Dict_trouples[trouple]
            #     possible_edges = [(e,s) for e in [e1,e2,e3] for s in [s1,s2,s3]]
            #     subgraph.remove_edges_from(possible_edges)
            #     to_keep_cycles = [edge for edge in possible_edges if nx.has_path(subgraph,edge[1],edge[0]) ]
            #     subgraph.add_edges_from(to_keep_cycles)

        return subgraph,new_edges,old_edges
    
    def is_cycle(self,subgraph):
        return all(subgraph.out_degree(point) != 2 for point in subgraph)

    def untangle_2(self,cycle):
        cycles_to_keep = []
        cycles_to_test = []
        if self.is_cycle(nx.subgraph(self.graph_res_networkx,cycle)):
            cycles_to_keep.append(cycle)
        else:
            cycle = cycle[::-1]
            cycles_to_test.append(cycle)

        while cycles_to_test:
            C_C = []
            for c in (cycles_to_test):
                if c[0] != c[-1]:
                    c = c + [c[0]]
                if c[1] not in self.graph_res_networkx.neighbors(c[0]):
                    c = c[::-1]
                assert c[1] in self.graph_res_networkx.neighbors(c[0]),c
                _,new_edges,old_edges = self.untangle_partial(c)
                assert len(old_edges) == len(new_edges) and len(old_edges) == 2, c
                edges_cycle = [(c[i],c[i+1]) for i in range(len(c)-1) if (c[i],c[(i+1)]) not in old_edges]
                edges_cycle = edges_cycle + new_edges
                graph_cycle = nx.Graph()
                graph_cycle.add_edges_from(edges_cycle)
                if len(list(nx.cycle_basis(graph_cycle))) != 2:
                    print("Error in number of cycles in untangle_2")
                    if  len(list(nx.connected_components(graph_cycle))) != 2 or 1 not in [len(comp) for comp in nx.connected_components(graph_cycle)]:
                        print("Error in number of connected components")
                        print(list(nx.connected_components(graph_cycle)))
                        print(list(nx.cycle_basis(graph_cycle)),c,cycle)
            
                for b in list(nx.cycle_basis(graph_cycle)):
                    C_C.append(b)
                
            cycles_to_test = []
            for c in C_C:
                sub_c = nx.subgraph(self.graph_res_networkx,c)
                if self.is_cycle(sub_c):
                    cycles_to_keep.append(c)
                else:
                    cycles_to_test.append(c)
        
        return cycles_to_keep
    


    
        

        
    
    




        
        
def main():
    pass

        



        
        
            
