import pickle
import sys
import os

# Step 1: Load the Pickle File
pickle_file_path = '/home/mehdii/projects/def-vidalthi/mehdii/3dPU/3dPU/Data/artificial_data/data_ball_wrapped.pkl'

with open(pickle_file_path, 'rb') as file:
    data = pickle.load(file)

sys.path.append('/home/mehdii/projects/def-vidalthi/mehdii/3dPU/3dPU/Plateau_Problem/Triangulation_Meshing')
from Final_surface import *



# Apply the function(s) to your data
# processed_data = your_function(data)

# Step 3: Store the Result
output_file_path = '/path/to/your/output/file.pkl'  # specify your output file path

with open(output_file_path, 'wb') as file:
    pickle.dump(processed_data, file)

print("Processing complete. Data saved to", output_file_path)
