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
        self.connected_components = {}
        self.mapping = []
        self.res_ordre = {}
        self.Separate_graphs = {}
        self.cycles = []


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
            if value == 1 and j < self.Y - 1 - 1:
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
        for res in (self.list_res):
            i,j,k,axe,value = res
            self.nodes_of_not_boundary(res)

    def dfs(self,node, colour):
        stack = [node]
        self.connex[node] = colour
        while stack:
            node = stack.pop()
            for neighbor in self.Res_graph[node]:
                if self.connex[neighbor] == 0:
                    self.connex[neighbor] = colour
                    stack.append(neighbor)
    
    
    def identify_connected_component(self):

        for node in range(len(self.mapping)):
            self.connex[node] = 0
        colour = 1

        for node in range(len(self.mapping)):
            if self.connex[node] == 0:
                self.dfs(node,colour)
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

    def detect_cycles(self):
        for colour in (self.Separate_graphs.keys()):
            pop = []
            for node in tqdm(self.connected_components[colour]):
                if self.connex[node] != -1 :
                    continue
                pop.append(node)
                path = []
                while pop :
                    next = pop.pop()
                    if self.connex[next] != -1 :
                        if path and path[-1] == node:
                            path.pop()
                        continue
                    self.connex[next] = -1
                    path.append(next)
                    pop.append(next)
                    path_set  = set(path)
                    for neighbour in self.Separate_graphs[colour][next]:
                        if neighbour in path_set: # this is O(n) funny
                            # Continue the logic or rebegin it is DFS now we 
                            # are 
                            cycle_index = path.index(neighbour)
                            self.cycles.append(path[cycle_index:] + [neighbour])
                        elif self.connex[neighbour] != -1:
                            pop.append(neighbour)





        





        


        
        
            
