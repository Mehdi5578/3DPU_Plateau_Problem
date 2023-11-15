#!/bin/bash

# Load the Python 3.11.5 module
module load python/3.11.5

#Load the variable of the path of the project in your repo
export projects="$HOME/projects/def-vidalthi/mehdii/3dPU/3dPU/"

#Activate the virual environement
source $projects/virtual_3dpu/bin/activate

# Any additional commands you want to run after loading the module
pip install -r requirements.txt

# add the path of the project to your pythonpath. 
export PYTHONPATH=$PYTHONPATH:$projects

#To run the creation of the loops
cd CreateLoops
nohup python -u create_loops.py > output.log 2>&1 &
