#!/usr/bin/env python
"""Provides multiple functions for 3D phase unwrapping.
"""
from copy import deepcopy
from dataclasses import dataclass
import numpy as np
import random
import csv
import os

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
    return np.diff(gy, axis=ax) - np.diff(gx, axis=ay)


@dataclass
class Residual:
    ax: int
    ori: int
    pos: NDArray
    
    def __eq__(self, other) -> bool:
        return self.ax == other.ax and tuple(self.pos) == tuple(other.pos) and self.ori == other.ori

    def __hash__(self) -> int:
        return tuple((self.ax, tuple(self.pos), self.ori)).__hash__()

@dataclass
class SpinnedResidual:
    spin: int
    res: Residual
    
    def __eq__(self, other) -> bool:
        return self.spin == other.spin and self.res.__eq__(other.res)

    def __hash__(self) -> int:
        return tuple((self.spin, self.res.ax, self.res.ori, tuple(self.res.pos))).__hash__()
    
    def is_boundary(self, shape, reverse: bool = False) -> bool:
        s = self.spin
        if reverse:
            s = -self.spin
        if s == 1:
            return self.res.pos[self.res.ax] + 1 == shape[self.res.ax]
        else:
            return self.res.pos[self.res.ax] == 0

ResidualMarker = dict[Residual, int]
Loop = list[SpinnedResidual]

@dataclass
class FlaggedLoop:
    closed: bool
    loop: Loop

def unprocess_all(res: list[NDArray], marker: ResidualMarker) -> None:
    for ax, rax in enumerate(res):
        for idx, ori in np.ndenumerate(rax):
            
            if ori != 0:
                pos = np.array(idx)
                marker[Residual(ax, ori, pos)] = -1

def unprocessed_residual(marker: ResidualMarker) -> Optional[Residual]:
    for r, m in marker.items():
        if m == -1:
            return r
    return None

    
# def potential_neighbors(self: SpinnedResidual, reverse: bool = False) -> list[SpinnedResidual]:
#     # List of potential neighbors.
#     neighbors = []

#     # Add the neighbor with the same axis as
#     # the current residual.
#     npos = deepcopy(self.res.pos)
#     if reverse:
#         npos[self.res.ax] -= self.spin
#     else:
#         npos[self.res.ax] += self.spin
#     neighbors.append(SpinnedResidual(self.spin, Residual(self.res.ax, self.res.ori, npos)))


def potential_neighbors_no_reverse(self: SpinnedResidual) -> list[SpinnedResidual]:
    # List of potential neighbors.
    neighbors = []
    # Add the neighbor with the same axis as
    # the current residual.
    npos = deepcopy(self.res.pos)
    npos[self.res.ax] += self.spin
    neighbors.append(SpinnedResidual(self.spin, Residual(self.res.ax, self.res.ori, npos)))

    # Add the neighbors for other axes.
    for d in range(1, dim):
        na = (self.res.ax + d) % dim
        if self.spin == 1:
            for i, ns in zip([0, 1], [-1, 1]):
                npos = deepcopy(self.res.pos)
                npos[na] += i*self.spin
                neighbors.append(SpinnedResidual(ns, Residual(na, self.res.ori * ns, npos)))
        if self.spin == -1:
            for i, ns in zip([0, 1], [1, -1]):
                npos = deepcopy(self.res.pos)
                npos[na] += i
                npos[self.res.ax] -= 1
                neighbors.append(SpinnedResidual(self.spin*ns, Residual(na, self.res.ori * ns, npos)))
    return neighbors

def potential_neighbors(self : SpinnedResidual, reverse:bool = False ) -> list[SpinnedResidual]:
    if not(reverse) :
        return potential_neighbors_no_reverse(self)
    
    else:
        curr = deepcopy(self)
        curr.spin = -self.spin
        return potential_neighbors_no_reverse(curr)

def next_residual(curr: SpinnedResidual, marker: ResidualMarker, reverse: bool = False, shuffle: bool = True) -> Optional[SpinnedResidual]:
    # List of potential neighbors.
    neighbors = potential_neighbors(curr, reverse)

    # Shuffle the list of neighbors if asked to.
    if shuffle:
        random.shuffle(neighbors)

    # Look for the first neighbor that has not been
    # yet processed.
    for spinned_residual in neighbors:
        if spinned_residual.res in marker and marker[spinned_residual.res] == -1:
            return spinned_residual

    # Return None.
    return None

def mark_loop(loop: Loop, marker: ResidualMarker, m: int) -> None:
    for spinned_residual in loop:
        marker[spinned_residual.res] = m

def close_loop(curr: SpinnedResidual, loop: Loop, reverse: bool = False) -> int:
    # List of potential neighbors.
    neighbors = potential_neighbors(curr, reverse)

    # Look for the first neighbor that has not been
    # yet processed.
    for i, item in enumerate(loop):
        if item in neighbors :
            return i

    # Return -1.
    return -1

def search_loop(start: SpinnedResidual, shape, marker: ResidualMarker, rev: bool = False) -> FlaggedLoop:
    loop = []
    curr = deepcopy(start)
    
    if rev :
        curr.spin = -start.spin
    while True:
        loop.append(curr)
        marker[curr.res] = 0
        if curr.is_boundary(shape):
            # Boundary residual. !! NB : les deux bouts doivent etre aux bords)
            return FlaggedLoop(False, loop)
        else:
            i = close_loop(curr, loop)
            # print(i,curr,loop)
            if i != -1 and len(loop[i:]) > 2:
                # print("this loop is discarded")
                # Unmark all residuals before i.
                mark_loop(loop[:i], marker, -1)

                # Mark all residuals after i as processed.
                mark_loop([curr] + loop[i:], marker, 1)
                loop = [curr] + loop[i:] 
                
                # Retreive only the loop.
                return FlaggedLoop(True, loop)
            else:
                # Get the next residual.
                neighbor = next_residual(curr, marker)
                if neighbor:
                    curr = neighbor
                    
                else:
                    print(curr)
                    print()
                    print(rev)
                    # TODO: handle this.
                    raise ValueError()

def join_open_loops(ploop: Loop, nloop: Loop) -> Loop:
    # The end of the first loop has to be
    # the start of the second loop.
    assert(ploop[-1].res == nloop[0].res)

    # Join the two loops.
    loop = ploop + nloop[1:]
    
    return loop


def save_loops_to_csv(loops, filename):
    # if not os.path.exists(filename):
    #     os.makedirs(filename)
    with open(filename, 'a', newline='') as csvfile:  # 'a' for append mode
        csvwriter = csv.writer(csvfile)
        for loop in loops:
            # Convert the loop to a format suitable for CSV writing
            # This example assumes you want to write the 'pos' attribute of each residual
            loop_data = [[res.res.ax, res.res.ori] + list(res.res.pos) for res in loop.loop]
            csvwriter.writerows(loop_data)


def residual_loops(loops, marker, psi: NDArray,csv_filename) -> list[FlaggedLoop]:
    # Store the shape of psi.
    print("begining the creation of the loops")
    shape = psi.shape
    # Store the list of all residuals.
    res = []
    for a in range(dim):
        res.append(residuals(psi, a))
    print("unprocessing of the residuals started")
    unprocess_all(res, marker)
    print("unprocessing of the residuals ended")
    # Total number of residuals that need processing.

    total_residuals = sum(r.size for r in res)
    print((total_residuals))
    processed_residuals = 0
    last_reported_progress = -1

    # Loop until all the residual has been processed
    while True:
        # Look for unprocessed residual.
        r = unprocessed_residual(marker)
        if not r:
            break  # Exit if all residuals have been processed.

        flagged_loop = search_loop(SpinnedResidual(1, r), shape, marker, False)
        loop = flagged_loop.loop

        if flagged_loop.closed:
            loops.append(flagged_loop)
            processed_residuals += len(loop)
        else:
            mark_loop(loop, marker, -1)
            rflagged_loop = search_loop(SpinnedResidual(1, r), shape, marker, True)
            rloop = list(reversed(rflagged_loop.loop))
            if rflagged_loop.closed:
                loops.append(FlaggedLoop(True, rloop))
                processed_residuals += len(rloop)
            else:
                mark_loop(rloop, marker, -1)
                loop = join_open_loops(rloop, loop)
                loop_res = [pos.res for pos in loop]
                if len(set(loop_res)) == len(loop_res):
                    mark_loop(loop, marker, 1)
                    loops.append(FlaggedLoop(False, loop))
                else:
                    mark_loop(loop, marker, -1)
                processed_residuals += len(loop)

        # Calculate and print the progress if there is at least a 1% advancement.
        progress = (processed_residuals / total_residuals) * 100
        if progress >= last_reported_progress + 0.01:
            print(f"Progress: {progress:.2f}% complete")
            last_reported_progress = progress
            save_loops_to_csv(loops,  csv_filename)
            loops.clear() 

    
    return loops


if __name__ == '__main__':
    pass
