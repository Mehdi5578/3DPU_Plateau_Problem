#!/bin/bash
#SBATCH --job-name=solve_MIP    # Job name
#SBATCH --nodes=1                 # Run all processes on a single node
#SBATCH --ntasks=1                # Run a single task
#SBATCH --time=100:00:00                # Infinite time limit
#SBATCH --mem=20G                  # Memory limit per node
  # Standard output and error log (%j expands to jobId)
# FILEPATH: /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/scripts_folder/test_loops.sh
# Description: This script is used for testing loops.
module load StdEnv/2020
module load python/3.11


/cvmfs/soft.computecanada.ca/easybuild/software/2020/avx2/Core/python/3.11.5/bin/python /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Plateau_Problem/Triangulation_Meshing/tests/test_MIP.py > /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/scripts_folder/double_loop.txt