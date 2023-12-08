from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import random
import csv
import os
import pickle
from typing import Union, Optional
from numpy.typing import NDArray
from tqdm import tqdm

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
        self.starting_open_paths = []

        self.open_paths = []

    def wrap(self,phi) :
        return np.round(phi / (2 * np.pi)).astype(int)

    def grad(self,psi, a: int):
        return np.diff(psi, axis=a)

    def wrap_grad(self,psi: NDArray, a: int):
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
        
        self.indirected_graph = deepcopy(self.Res_graph)
        for point in tqdm(range(len(self.mapping))):
            sons = self.Res_graph[point]
            self.indirected_graph[point] = deepcopy(self.Res_graph[point])
            for son in sons:
                self.indirected_graph[son].append(point)

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
        print(layer)
        for path in self.cycles:
            for point in path:
                self.incycles[point] = True
        antecedant = dict()
        paths = dict()
        visited = set()
        for point in tqdm(layer):
            paths[point] = [point]
            antecedant[point] = point
        cpt = 0
        while layer != {-1}:
            cpt += 1
            if cpt % 100 == 0:
                print(len(layer))
            new_layer = set()
            for point in (layer):
                if point != -1 :
                    next_nodes = [n for n in self.Res_graph[point] if (not self.incycles[n] and n not in visited)]
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
        





        



        
        
            
