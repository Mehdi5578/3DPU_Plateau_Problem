#!/bin/bash
#SBATCH --job-name=create_loops    # Job name
#SBATCH --nodes=1           # Run all processes on a single node
#SBATCH --ntasks=8           # Run a single task
#SBATCH --cpus-per-task=8  # Number of CPU cores per task
#SBATCH --time=2:00:00                # Infinite time limit
  # Standard output and error log (%j expands to jobId)
# FILEPATH: /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/scripts_folder/test_loops.sh
# Description: This script is used for creating loops.

module load StdEnv/2023
module load gurobi/11.0.1

source /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/env_gurobi_HZ/bin/activate


# pip install -r requirements.txt 

python /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/scripts_folder/create_loops.py 8 50 > /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/scripts_folder/create_loops.txt