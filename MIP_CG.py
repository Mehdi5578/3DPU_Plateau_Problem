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

def extract_residuals (loop : Loop ):
    l = []
    for k in loop:
        l.append(k.res.pos)
    return l



    
