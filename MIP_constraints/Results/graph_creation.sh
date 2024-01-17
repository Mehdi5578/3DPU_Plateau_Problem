#!/bin/bash
#SBATCH --job-name=my_cpp_job     # Job name
#SBATCH --nodes=1                 # Run all processes on a single node
#SBATCH --ntasks=1                # Run a single task
#SBATCH --time=01:00:00           # Time limit hrs:min:sec
#SBATCH --output=result_%j.out    # Standard output log
#SBATCH --error=error_%j.out      # Error log



# Load the C++ module (if required)
# module load StdEnv/2020
# module load gcc
# module load gurobi

grbgetkey 9462b60b-4bac-4999-8cee-af81af738219 > output.txt

# GRB_LICENSE_FILE=/cvmfs/restricted.computecanada.ca/config/licenses/gurobi/clusters/beluga.lic
# #Decare the path variable of the c++ file
# grbprobe > grbprobe.txt
# gurobi_cl > gurobi_cl.txt

