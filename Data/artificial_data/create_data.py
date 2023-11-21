import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle
import time

import numpy as np

import numpy as np
from scipy.special import sph_harm
import numpy as np
import matplotlib.pyplot as plt

# Parameters for the grid and the sphere
grid_size = 100
radius = 40
center = [grid_size // 2, grid_size // 2, grid_size // 2]

# Create a 3D grid
X, Y, Z = np.meshgrid(np.arange(grid_size), np.arange(grid_size), np.arange(grid_size), indexing='ij')

# Equation for a sphere
mask = (X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2 <= radius**2

# Convert grid indices to radial and angular coordinates
r = np.sqrt((X - center[0])**2 + (Y - center[1])**2 + (Z - center[2])**2)
theta = np.arccos((Z - center[2]) / np.maximum(r, 1))  # Avoid division by zero for r = 0
phi = np.arctan2((Y - center[1]), (X - center[0]))

# Complex phase pattern using the recalculated coordinates
phase_data = np.zeros((grid_size, grid_size, grid_size))
phase_data[mask] = np.sin(3 * theta[mask]) + np.cos(5 * phi[mask]) + np.sin(r[mask])

# Increase the scale of the noise for more pronounced residuals
increased_noise = np.random.normal(scale=0.6, size=phase_data.shape)  # Increased noise scale
noisy_phase_data_with_more_noise = phase_data + increased_noise

# Ensure that the noise is only added within the sphere
noisy_phase_data_with_more_noise[~mask] = 0

# Wrap the noisy phase data with more noise
wrapped_noisy_phase_data_with_more_noise = np.angle(np.exp(1j * noisy_phase_data_with_more_noise))


print('the data is getting generated as a 3D ball')

if __name__ == '__main__':
    print("begining the data creation")
    deb = time.time()
    with open(r'/home/mehdii/projects/def-vidalthi/mehdii/3dPU/3dPU/Data/artificial_data/data_ball_ori.pkl', 'wb') as file:
       pickle.dump(phase_data, file)
    with open(r'/home/mehdii/projects/def-vidalthi/mehdii/3dPU/3dPU/Data/artificial_data/data_ball_wrapped.pkl', 'wb') as file:
       pickle.dump(wrapped_noisy_phase_data_with_more_noise, file)
    fin = time.time()
    print(fin-deb)