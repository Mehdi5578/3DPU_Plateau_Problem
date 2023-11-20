import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import time

import numpy as np

# Parameters for the grid and the ball
grid_size = 100
radius = 40
center = [grid_size // 2, grid_size // 2, grid_size // 2]

# Create a 3D grid
x, y, z = np.ogrid[:grid_size, :grid_size, :grid_size]

# Equation for a sphere
mask = (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 <= radius**2

# Create varying phase data inside the ball
# For instance, a linear gradient from -10*pi to 10*pi across one dimension
phase_data = np.zeros((grid_size, grid_size, grid_size))
phase_data[mask] = np.linspace(-10 * np.pi, 10 * np.pi, np.count_nonzero(mask))

# Wrap the phase data
wrapped_phase = np.angle(np.exp(1j * phase_data))

if __name__ == '__main__':
    print("begining the data creation")
    deb = time.time()
    with open(r'/home/mehdii/projects/def-vidalthi/mehdii/3dPU/3dPU/Data/artificial_data/data_ball_orig.pkl', 'wb') as f:
       pickle.dump(phase_data, f)
    with open(r'/home/mehdii/projects/def-vidalthi/mehdii/3dPU/3dPU/Data/artificial_data/data_ball_wrapped.pkl', 'wb') as f:
       pickle.dump(wrapped_phase, f)
    fin = time.time()
    print(fin-deb)