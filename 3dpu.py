#!/usr/bin/env python
"""Provides multiple functions for 3D phase unwrapping.
"""
from copy import deepcopy
import numpy as np
import random

from typing import Union, Optional
from numpy.typing import NDArray

__author__ = "Youssouf Emine and El Mehdi Oudaoud"
__copyright__ = "Copyright 2023, 3DPU project"
__credits__ = ["Youssouf Emine", "El Mehdi Oudaoud", "Thibaut Vidal"]
__license__ = "GPL"
__version__ = "1.0.1"
__maintainer__ = "Youssouf Emine"
__email__ = "youssouf.emine@polymtl.ca"
__status__ = "Production"

Residual = tuple[int, int, tuple[int,...]]
SpinnedResidual = tuple[int, Residual]
ResidualMarker = dict[Residual, int]
Loop = list[SpinnedResidual]
FlaggedLoop = tuple[int, Loop]

dim = int(3)

def wrap(phi: NDArray) -> NDArray:
    return np.round(phi / (2 * np.pi)).astype(int)

def grad(psi: NDArray, a: int) -> NDArray:
    return np.diff(psi, axis=a)

def wrap_grad(psi: NDArray, a: int) -> NDArray:
    return wrap(grad(psi, a))

def residuals(psi: NDArray, a: int) -> NDArray:
    assert(a >= 0 and a < dim)
    ax, ay = (a + np.arange(1, dim)) % dim
    gx = wrap_grad(psi, a=ax)
    gy = wrap_grad(psi, a=ay)
    return np.diff(gy, a=ax) - np.diff(gx, a=ay)

def unprocess_all(res: list[NDArray], marker: ResidualMarker) -> None:
    for a, ra in enumerate(res):
        for idx, r in np.ndenumerate(ra):
            if r != 0:
                marker[(a, r, idx)] = -1

def unprocessed_residual(marker: ResidualMarker) -> Optional[Residual]:
    for r, m in marker.items():
        if m == -1:
            return r
    return None

def is_boundary_residual(curr: SpinnedResidual, res: list[NDArray], reverse: bool = False) -> bool:
    spin, (a, _, pos) = curr  
    if spin == 1:
        return pos[a] + 1 == res[a].shape[a]
    elif spin == -1:
        return pos[a] == 0

def potential_neighbors(curr: SpinnedResidual, reverse: bool = False) -> list[SpinnedResidual]:
    # Retreive the axis, the sign and the position
    # of the current residual.
    s, (a, r, pos) = curr

    # List of potential neighbors.
    neighbors = []

    # Add the neighbor with the same axis as
    # the current residual.
    npos = deepcopy(pos)
    if reverse:
        npos[a] -= s
    else:
        npos[a] += s
    neighbors.append((s, (a, r, npos)))

    # Add the neighbors for other axes.
    for d in range(1, dim):
        # Get the axis of the neighbor.
        na = (a + d) % dim
        for i, ns in zip([0, 1], [-1, 1]):
            npos = deepcopy(pos)
            if reverse:
                npos[a] -= s
                npos[na] -= i
            else:
                npos[na] += i
            neighbors.append((ns, (na, ns * r, npos)))

    return neighbors

def next_residual(curr: SpinnedResidual, marker: ResidualMarker, reverse: bool = False, shuffle: bool = True) -> Optional[SpinnedResidual]:
    # List of potential neighbors.
    neighbors = potential_neighbors(curr, reverse)

    # Shuffle the list of neighbors if asked to.
    if shuffle:
        random.shuffle(neighbors)

    # Look for the first neighbor that has not been
    # yet processed.
    for s, r in neighbors:
        if r in marker and marker[r] == -1:
            return s, r

    # Return None.
    return None

def mark_loop(loop: Loop, marker: ResidualMarker, m: int) -> None:
    for _, s in loop:
        marker[s] = m

def close_loop(curr: SpinnedResidual, loop: Loop, reverse: bool = False) -> int:
    # List of potential neighbors.
    neighbors = potential_neighbors(curr, reverse)

    # Look for the first neighbor that has not been
    # yet processed.
    for i, item in enumerate(loop):
        if item in neighbors:
            return i

    # Return None.
    return -1

def search_loop(start: SpinnedResidual, res: list[NDArray], marker: ResidualMarker, reverse: bool = False) -> FlaggedLoop:
    loop = []
    curr = start
    while True:
        _, r = curr
        loop.append(curr)
        marker[r] = 0
        if is_boundary_residual(curr, res):
            # Boundary residual.
            return 0, loop
        else:
            i = close_loop(curr, loop, reverse)
            if i != -1:
                # Unmark all residuals before i.
                mark_loop(loop[:i], marker, -1)

                # Mark all residuals after i as processed.
                mark_loop(loop[i:], marker, 1)

                # Retreive only the loop.
                return 1, loop[i:]
            else:
                # Get the next residual.
                neighbor = next_residual(curr, marker, reverse)
                if neighbor:
                    curr = neighbor
                else:
                    # TODO: handle this.
                    raise ValueError()

def join_open_loops(ploop: Loop, nloop: Loop) -> Loop:
    # The end of the first loop has to be
    # the start of the second loop.
    assert(ploop[-1] == nloop[1])

    # Join the two loops.
    loop = ploop + nloop[1:]
    
    return loop

def residual_loops(psi: NDArray) -> list[FlaggedLoop]:
    # Store the list of all residuals.
    res = []
    for a in range(dim):
        res.append(residuals(psi, a))

    # A dict to store the mark of the residuals:
    # -1: unprocessed
    # 0: in process
    # 1: processed
    marker = {}
    unprocess_all(res, marker)

    # A loop will have (flag, items):
    # flag: 0 if the loop is open and
    # 1 if it is closed.
    # items contains the list of residuals
    # in the loop.
    loops = []

    # Loop until all the residual has been processed
    while True:
        # Look for unprocessed residual.
        r = unprocessed_residual(marker)

        # If all residuals were processed.
        if not r:
            break

        s, loop = search_loop((1, r), res, marker, False)

        # If the loop is closed.
        if s == 1:
            loops.append((1, loop))
        else: # The loop is open.
            # Unmark the loop.
            mark_loop(loop, marker, -1)

            # Get the reversed loop.
            s, rloop = search_loop((1, r), res, marker, True)
            rloop = list(reversed(rloop))

            # If the reversed loop is closed.
            if s == 1:
                loops.append((1, rloop))
            else:
                # Unmark the reversed loop.
                mark_loop(rloop, marker, -1)

                # Join the two loops.
                loop = join_open_loops(rloop, loop)

                # Mark the loop.
                mark_loop(loop, marker, 1)

                loops.append((0, loop))
    return loops

if __name__ == '__main__':
    pass
