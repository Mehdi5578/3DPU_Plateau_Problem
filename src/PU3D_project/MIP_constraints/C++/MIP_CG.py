#!/usr/bin/env python
"""Provides some essential functions for the MIP Column Generation for one loop .
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import random
from _3DLoops._3dpu import *



from typing import Union, Optional
from numpy.typing import NDArray

dim = int(3)

axe = int

Square = ((int,int,int,int),int,axe)

Cube = ((int,int,int,int),int,int,axe)



# def extract_residuals (loop : Loop ):
#     lasso_positions = []
#     for r in loop:
#         pos = [r.res.ax] + list(r.res.pos)
#         if pos[0] == 2:
#             new_pos = [pos[1]+0.5,pos[2] + 0.5,pos[3]]
#         elif pos[0] == 1:
#             new_pos = [pos[1]+0.5,pos[2],pos[3]+0.5]
#         elif pos[0] == 0:
#             new_pos = [pos[1],pos[2] + 0.5,pos[3]+0.5]
#         lasso_positions.append(new_pos)
#     return np.array(lasso_positions) 


# def defined_cage(loop : Loop):
#     positions = extract_residuals(loop)
#     dict = {}

#     return dict

def closed_paths_2D(carre : Square) -> list[Square]:
    "Gives the closed paths that are derived from a 2D square"
    (x1,x2,y1,y2),z,ax = carre
    debut = np.array([x1,x2,y1,y2])
    assert(x2-x1 == y2-y1 and x1 < x2 and y1 < y2)

    L = []
    for step in range(x2 - x1):
        L.append((tuple(debut + step*np.array([1,0,1,0])),z,ax))
        L.append((tuple(debut + step*np.array([0,-1,0,-1])),z,ax))

    return L[1:]

def shift_cube(cube :Cube, iter : int) -> Cube :
    "shifts the representation of the Cube"
    (x1,x2,y1,y2),z1,z2,ax = cube
    ax  = (ax - iter ) % dim
    return (z1,z2,x1,x2),y1,y2,ax


def closed_paths_3D_bis(cube : Cube) -> list[Square]:
    (x1,x2,y1,y2),z1,z2,ax = cube
    L = []
    for z in range(z1,z2+1):
        L = L + closed_paths_2D(((x1,x2,y1,y2),z,ax))
    return L

def closed_paths_3D(cube :Cube) -> list[Square]:
    L = []
    L = L + closed_paths_3D_bis(cube)
    L = L + closed_paths_3D_bis(shift_cube(cube,1))
    L = L + closed_paths_3D_bis(shift_cube(cube,2))
    return L





















    
    

    








    
