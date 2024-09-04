import pickle
from tqdm import tqdm
from src.PU3D_project.MIP_constraints.Python.MIP_formulation import minimize_edges_MIP

print('Loading data')

with open('src/tests/Blocked_edges_cycles_init.pkl', 'rb') as f:
    Blocked_edges_cycles_init = pickle.load(f)
with open('src/tests/Blocked_edges_open_paths_init.pkl', 'rb') as f:
    Blocked_edges_open_paths_init = pickle.load(f)

print('Minimizing edges')

MIP_cycle_edges = []
for edges in tqdm(Blocked_edges_cycles_init):
    smaller_edges = minimize_edges_MIP(edges,time_limit = 15 ,num_threads = 10)
    MIP_cycle_edges.append(smaller_edges)

with open('Blocked_edges_cycles_MIP.pkl', 'wb') as f:
    pickle.dump(MIP_cycle_edges, f)

MIP_open_path_edges = []
for edges in tqdm(Blocked_edges_open_paths_init):
    smaller_edges = minimize_edges_MIP(edges,time_limit = 15 ,num_threads = 10)
    MIP_open_path_edges.append(smaller_edges)

with open('Blocked_edges_open_paths_MIP.pkl', 'wb') as f:
    pickle.dump(MIP_open_path_edges, f)
