#!/bin/bash
#SBATCH --job-name=solve_MIP    # Job name
#SBATCH --nodes=1             # Run all processes on a single node
#SBATCH --ntasks=1           # Run a single task
#SBATCH --cpus-per-task=30   # Number of CPU cores per task
#SBATCH --time=100:00:00                # Infinite time limit
#SBATCH --mem=10G                # Memory limit per node
  # Standard output and error log (%j expands to jobId)
# FILEPATH: /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/scripts_folder/test_loops.sh
# Description: This script is used for testing loops.

module load StdEnv/2023
module load gurobi/11.0.1

source /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/env_gurobi_HZ/bin/activate


# pip install -r requirements.txt 

python /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/scripts_folder/create_loops.py > /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/scripts_folder/create_loops.txt