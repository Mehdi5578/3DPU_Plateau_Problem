import numpy as np
from tqdm import tqdm
from queue import PriorityQueue

def wrapping_operator(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def spatial_phase_coherence_weight(phi_i, phi_j):
    return 1 - abs(wrapping_operator(phi_i - phi_j) / np.pi)

def temporal_phase_coherence_weight(phi_i1, phi_j1, phi_i2, phi_j2, TE_i1, TE_i2):
    return max(0, 1 - abs(wrapping_operator(phi_i1 - phi_j1) - wrapping_operator(phi_i2 - phi_j2) * TE_i1 / TE_i2))

def magnitude_coherence_weight(M_i, M_j):
    return (min(M_i, M_j) / max(M_i, M_j)) ** 2

def all_edges(shape):
    """Generate all possible edges in a 3D grid."""
    edges = []
    for i in range(shape[0]):
        for j in range(shape[1]):
            for k in range(shape[2]):
                if i + 1 < shape[0]:
                    edges.append(((i, j, k), (i + 1, j, k)))
                if j + 1 < shape[1]:
                    edges.append(((i, j, k), (i, j + 1, k)))
                if k + 1 < shape[2]:
                    edges.append(((i, j, k), (i, j, k + 1)))
    return edges

def quality_map(phi):
    edges = all_edges(phi.shape)
    quality_map = dict()
    for edge in tqdm(edges):
        i, j = edge
        quality_map[edge] = spatial_phase_coherence_weight(phi[i], phi[j])
    return quality_map

def quality_to_cost(quality_map):
    """Transform a real-valued quality map to integer cost values."""
    cost_map = {edge: max(round(255 * (1 - quality)), 1) for edge, quality in quality_map.items()}
    for edge in quality_map:
        if quality_map[edge] == 0:
            cost_map[edge] = 0  # Special case for no connection
    return cost_map

def phase_unwrapping(quality_map, phases, forbidden_edges):
    """Perform phase unwrapping using the quality maps and phase values."""
    directions = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    shape = phases.shape
    unwrapped_phases = np.zeros_like(phases)
    visited = np.zeros_like(phases, dtype=bool)
    
    # Priority queue for voxel edges
    pq = PriorityQueue()
    
    # Function to check if an edge is forbidden
    def is_forbidden(edge):
        return (edge in forbidden_edges) or (edge[::-1] in forbidden_edges)
    
    # Initialize the queue with the six edges around the seed voxel (0, 0, 0)
    seed = (0, 0, 0)
    for d in directions:
        neighbor = (seed[0] + d[0], seed[1] + d[1], seed[2] + d[2])
        if 0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1] and 0 <= neighbor[2] < shape[2]:
            edge = (min(seed, neighbor),max(seed, neighbor))
            if not is_forbidden(edge):
                cost = quality_map[edge]
                pq.put((cost, seed, neighbor))
    
    visited[seed] = True
    unwrapped_phases[seed] = phases[seed]
    
    while not pq.empty():
        cost, v1, v2 = pq.get()
        
        if visited[v2]:
            continue
        
        # Phase unwrapping formula
        phase_jump = phases[v2] - unwrapped_phases[v1]
        n = round(phase_jump / (2 * np.pi))
        unwrapped_phases[v2] = phases[v2] - 2 * np.pi * n
        visited[v2] = True
        
        # Add new edges to the queue
        for d in directions:
            neighbor = (v2[0] + d[0], v2[1] + d[1], v2[2] + d[2])
            if 0 <= neighbor[0] < shape[0] and 0 <= neighbor[1] < shape[1] and 0 <= neighbor[2] < shape[2]:
                edge = (min(v2, neighbor),max(v2, neighbor))
                if not visited[neighbor] and not is_forbidden(edge):
                    cost = quality_map[edge]
                    pq.put((cost, v2, neighbor))
    
    return unwrapped_phases


def unwrapp_phase(phase,forbidden_edges):
    quality_map_ = quality_map(phase)
    cost_map = quality_to_cost(quality_map_)
    unwrapped_phase = phase_unwrapping(cost_map, phase, forbidden_edges)
    return unwrapped_phase
