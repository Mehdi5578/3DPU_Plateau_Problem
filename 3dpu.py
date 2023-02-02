#!/usr/bin/env python
"""Provides multiple functions for 3D phase unwrapping.
"""
import numpy as np

__author__ = "Youssouf Emine and El Mehdi Oudaoud"
__copyright__ = "Copyright 2023, 3DPU project"
__credits__ = ["Youssouf Emine", "El Mehdi Oudaoud", "Thibaut Vidal"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Youssouf Emine"
__email__ = "youssouf.emine@polymtl.ca"
__status__ = "Production"

dim = int(3)

def wrap(phi):
    return np.round(phi / (2 * np.pi)).astype(int)

def grad(psi, a: int):
    return np.diff(psi, axis=a)

def wrap_grad(psi, a: int):
    return wrap(grad(psi, a))

def residuals(psi, a: int):
    assert(a >= 0 and a < dim)
    ax, ay = (1 + np.arange(dim-1) + a) % dim
    gx = wrap_grad(psi, a=ax)
    gy = wrap_grad(psi, a=ay)
    return np.diff(gy, a=ax) - np.diff(gx, a=ay)

def residual_loops(psi):
    rx = residuals(psi, 0)
    ry = residuals(psi, 1)
    rz = residuals(psi, 2)

    # TODO: loop through the residuals and create
    # residual loops.

if __name__ == '__main__':
    pass