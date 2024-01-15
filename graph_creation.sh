#!/bin/bash
#SBATCH --job-name=my_cpp_job     # Job name
#SBATCH --nodes=1                 # Run all processes on a single node
#SBATCH --ntasks=1                # Run a single task
#SBATCH --time=01:00:00           # Time limit hrs:min:sec
  # Standard output and error log (%j expands to jobId)

# Load the C++ module (if required)
module load gcc/11.3.0
module load gurobi

export GRB_LICENSE_FILE="/home/mehdii/gurobi.lic"
#Decare the path variable of the c++ file
path_to_script="/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/MIP_constraints/"

g++ "${path_to_script}MIP_model.cpp" -o "${path_to_script}your_program" -I"$GUROBI_HOME/include" -L"$GUROBI_HOME/lib" -lgurobi_c++ -lgurobi110
cd $path_to_script
# Execute the program and redirect output to a file
./your_program > output_file.txt

cp output_file.txt "${path_to_script}Results/output_file.txt"