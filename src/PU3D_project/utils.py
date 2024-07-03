import numpy as np
from src.PU3D_project.Plateau_Problem.Triangulation_Meshing import *
from src.PU3D_project.Block_edges import *


def transform_res_to_point(res):
    x,y,z,ax,_ = res
    if ax == 0:
        return (x,y+0.5,z+0.5)
    elif ax == 1:
        return (x+0.5,y,z+0.5)
    else:
        return (x+0.5,y+0.5,z)
    

def fill_cycle(cycle,C):
    points = PointList()
    for res in cycle:
        points.add_point(transform_res_to_point(C.mapping[res]))
    M = Edge_Flipping(points,10*len(points.points))
    M.create_quadrilaterals()
    M.split_quadrilateral()
    M.canonic_representation_from_mesh()
    M.clean_triangles()
    M.fill_edges()
    M.update_weights()
    M.mapping = [np.array(o) for o in M.mapping]
    E = Block_edges(M.triangles,M.mapping)
    E.block_all_the_edges()
    Edges = (E.blocked_edges)
    return Edges

def compute_absolute_phase_gradients(wrapped_phase, unwrapped_phase):

    gradient_x_unwrapped = np.diff(unwrapped_phase, axis=0, append=unwrapped_phase[0:1,:,:])
    gradient_y_unwrapped = np.diff(unwrapped_phase, axis=1, append=unwrapped_phase[:,0:1,:])
    gradient_z_unwrapped = np.diff(unwrapped_phase, axis=2, append=unwrapped_phase[:,:,0:1])
    
    gradient_x_wrapped = np.diff(wrapped_phase, axis=0, append=wrapped_phase[0:1,:,:])
    gradient_y_wrapped = np.diff(wrapped_phase, axis=1, append=wrapped_phase[:,0:1,:])
    gradient_z_wrapped = np.diff(wrapped_phase, axis=2, append=wrapped_phase[:,:,0:1])
    
    
    abs_diff_x = np.abs(gradient_x_unwrapped - gradient_x_wrapped)
    abs_diff_y = np.abs(gradient_y_unwrapped - gradient_y_wrapped)
    abs_diff_z = np.abs(gradient_z_unwrapped - gradient_z_wrapped)

    total_diff = np.count_nonzero(abs_diff_x) + np.count_nonzero(abs_diff_y) + np.count_nonzero(abs_diff_z)
    
    return total_diff


