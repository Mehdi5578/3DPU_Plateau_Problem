
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

class Resiuals():
    def __init__(self,data_phase):
        self.data = data_phase


    def wrap(self,phi) -> NDArray:
        return np.round(phi / (2 * np.pi)).astype(int)

    def grad(self,psi, a: int) -> NDArray:
        return np.diff(psi, axis=a)

    def wrap_grad(self,psi: NDArray, a: int) -> NDArray:
        return self.wrap(self.grad(psi, a))

    def residuals(self,psi: NDArray, a: int) -> NDArray:
        assert(a >= 0 and a < dim)
        ax, ay = (a + np.arange(1, dim)) % dim
        gx = wrap_grad(psi, a=ax)
        gy = wrap_grad(psi, a=ay)
        return np.diff(gy, axis=ax) - np.diff(gx, axis=ay)

