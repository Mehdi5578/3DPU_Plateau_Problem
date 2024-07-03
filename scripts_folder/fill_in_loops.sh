#!/bin/bash
#SBATCH --job-name=mip_gurobi    # Job name
#SBATCH --output=%x_%j.out             # Standard output log (%x expands to job name, %j to jobId)
#SBATCH --error=%x_%j.err              # Standard error log
#SBATCH --nodes=1                      # Run all processes on a single node
#SBATCH --ntasks=1                     # Run a single task
#SBATCH --cpus-per-task=8           # Number of CPU cores per task
#SBATCH --time=10:00:00                # Time limit hrs:min:sec
#SBATCH --mem=10G                      # Memory per node
 # Standard output and error log (%j expands to jobId)
# FILEPATH: /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/scripts_folder/test_loops.sh
# Description: This script is used for creating loops.

module load StdEnv/2023
module load gurobi/11.0.1

source /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/env_gurobi_HZ/bin/activate


# pip install -r requirements.txt 

python /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Plateau_Problem/Triangulation_Meshing/tests/fill_in_loop.py 8 50 > /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/scripts_folder/fill_in_loops_100.txt