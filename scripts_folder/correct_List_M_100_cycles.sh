#!/bin/bash
#SBATCH --job-name=fill_loops    # Job name
#SBATCH --output=%x_%j.out             # Standard output log (%x expands to job name, %j to jobId)
#SBATCH --error=%x_%j.err              # Standard error log
#SBATCH --mem=50G                      # Memory per node
#SBATCH --time=1-00:10                 # Time limit in D-HH:MM	
 # Standard output and error log (%j expands to jobId)
 #SBATCH --mail-type=BEGIN,END,FAIL       # Send an email at start, end, and failure
#SBATCH --mail-user=oudaoud.mehdi@gmail.com  # Replace with your actual email address
# FILEPATH: /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/scripts_folder/test_loops.sh
# Description: This script is used for creating loops.

module load StdEnv/2023
module load gurobi/11.0.1

# source /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/env_gurobi_HZ/bin/activate

# pip install -r requirements.txt 

python /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/src/tests/correct_loops_List_M.py > /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/Results/correct_loops_List_M_100_cycles.txt

