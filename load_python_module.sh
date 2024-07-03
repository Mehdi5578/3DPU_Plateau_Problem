#!/bin/bash

#Load the variable of the path of the project in your repo
export projects="$HOME/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem"

#Activate the virual environement
source $projects/env_gurobi_hz/bin/activate

# Any additional commands you want to run after loading the module
pip install -r requirements.txt

# add the path of the project to your pythonpath. 
export PYTHONPATH="/home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem:$PYTHONPATH"

