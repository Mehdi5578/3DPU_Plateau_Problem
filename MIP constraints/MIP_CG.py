#!/usr/bin/env python
"""Provides some essential functions for the MIP Column Generation for one loop .
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import random
from _3dpu import *

from typing import Union, Optional
from numpy.typing import NDArray

dim = int(3)

Square = tuple(int,int,int,int)
Cube = tuple(Square,Square)



def extract_residuals (loop : Loop ):
    lasso_positions = []
    for r in loop:
        pos = [r.res.ax] + list(r.res.pos)
        if pos[0] == 2:
            new_pos = [pos[1]+0.5,pos[2] + 0.5,pos[3]]
        elif pos[0] == 1:
            new_pos = [pos[1]+0.5,pos[2],pos[3]+0.5]
        elif pos[0] == 0:
            new_pos = [pos[1],pos[2] + 0.5,pos[3]+0.5]
        lasso_positions.append(new_pos)
    return np.array(lasso_positions) 


def defined_cage(loop : Loop):
    positions = extract_residuals(loop)
    dict = {}

    return dict

def closed_paths_2D(carre : Square) -> list[Square]:
    "Gives the closed paths that are derived from a 2D square"
    x1,x2,y1,y2 = carre
    debut = np.array([x1,x2,y1,y2])
    assert(x2-x1 == y2-y1)
    L = []
    for step in range(x2 - x1):
        L.append(debut + step*np.array([1,0,1,0]))
        L.append(debut + step*np.array([0,-1,0,-1]))
    return L


# def closed_paths_3D(cube : Cube) -> list[Square]:
#     start_carre, end_carre = cube

    








    
