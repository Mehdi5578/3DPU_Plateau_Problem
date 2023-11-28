from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import random
import csv
import os
import pickle
from typing import Union, Optional
from numpy.typing import NDArray

dim = int(3)


# Here each residual is stored in a tuple of five elements fist three are the smallest cooridnates of 
# the face it lays upon, the second is the the axis = 0,1,2 and the third is the value = -1 or 1


class Resiuals():
    
    def __init__(self,data_phase):
        self.data = data_phase
        self.Res  = {}
        self.list_res = []
        self.Res_graph = {}
        self.grid_size = self.data.shape



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
    

    def nodes_of_not_boundary(self,Residual):
        "Add the sons of the Residual "  
        #REVOIR LES VOISINS PAR -1
        i,j,k,axe,value = Residual
        if axe == 0 :
            if value == 1:
                if self.Res[0][i+1,j,k] == 1:
                    self.Res_graph[Residual].append((i+1,j,k,0,1))
                if self.Res[1][i,j,k] == -1 :
                    self.Res_graph[Residual].append((i,j,k,1,-1))
                if self.Res[2][i,j,k] == -1 :
                    self.Res_graph[Residual].append((i,j,k,2,-1))
                if self.Res[1][i,j+1,k] == 1:
                    self.Res_graph[Residual].append((i,j+1,k,1,1))
                if self.Res[2][i,j,k+1] == 1:
                    self.Res_graph[Residual].append((i,j,k+1,2,1))
            
            if value == -1:
                if self.Res[0][i-1,j,k] == -1:
                    self.Res_graph[Residual].append((i-1,j,k,0,-1))
                if self.Res[1][i-1,j,k] == -1:
                    self.Res_graph[Residual].append((i-1,j,k,1,-1))
                if self.Res[2][i-1,j,k] == -1:
                    self.Res_graph[Residual].append((i-1,j,k,2,-1))
                if self.Res[1][i-1,j+1,k] == 1:
                    self.Res_graph[Residual].append((i-1,j+1,k,0,1))
                if self.Res[2][i-1,j,k+1] == 1:
                    self.Res_graph[Residual].append((i-1,j,k+1,2,1))
        if axe == 1:
            if value == 1:
                if self.Res[1][i,j+1,k] == 1:
                    self.Res_graph[Residual].append((i,j+1,k,1,1))
                if self.Res[0][i,j,k] == -1:
                    self.Res_graph[Residual].append((i,j,k,0,-1))
                if self.Res[2][i,j,k] == -1:
                    self.Res_graph[Residual].append((i,j,k,2,-1))
                if self.Res[0][i+1,j,k] == 1:
                    self.Res_graph[Residual].append((i+1,j,k,0,1))
                if self.Res[2][i,j,k+1] == 1:
                    self.Res_graph[Residual].append((i,j,k+1,2,1))

            if value == -1:
                if self.Res[1][i,j-1,k] == -1:
                    self.Res_graph[Residual].append((i,j-1,k,1,-1))
                if self.Res[0][i,j-1,k] == -1:
                    self.Res_graph[Residual].append((i,j-1,k,0,-1))
                if self.Res[2][i,j-1,k] == -1:
                    self.Res_graph[Residual].append((i,j-1,k,2,-1))
                if self.Res[0][i+1,j-1,k] == 1:
                    self.Res_graph[Residual].append((i+1,j-1,k,0,1))
                if self.Res[2][i,j-1,k+1] == 1:
                    self.Res_graph[Residual].append((i,j-1,k+1,2,1))
        if axe == 2:
            if value == 1:
                if self.Res[2][i,j,k+1] == 1:
                    self.Res_graph[Residual].append((i,j,k+1,2,1))
                if self.Res[0][i,j,k] == -1:
                    self.Res_graph[Residual].append((i,j,k,0,-1))
                if self.Res[1][i,j,k] == -1:
                    self.Res_graph[Residual].append((i,j,k,1,-1))
                if self.Res[0][i+1,j,k] == 1:
                    self.Res_graph[Residual].append((i+1,j,k,0,1))
                if self.Res[1][i,j+1,k] == 1:
                    self.Res_graph[Residual].append((i,j+1,k,1,1))

            if value == -1:
                if self.Res[2][i,j,k-1] == -1:
                    self.Res_graph[Residual].append((i,j,k-1,2,-1))
                if self.Res[0][i,j,k-1] == -1:
                    self.Res_graph[Residual].append((i,j,k-1,0,-1))
                if self.Res[1][i,j,k-1] == -1:
                    self.Res_graph[Residual].append((i,j,k-1,1,-1))
                if self.Res[0][i+1,j,k-1] == 1:
                    self.Res_graph[Residual].append((i+1,j,k-1,0,1))
                if self.Res[1][i,j+1,k-1] == 1:
                    self.Res_graph[Residual].append((i,j+1,k-1,1,1))
