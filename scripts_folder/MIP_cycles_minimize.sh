#!/bin/bash

# Define pairs of seuil_begin and seuil_end
declare -a seuil_pairs=(
  "0 1000"
  "1000 2000"
  "2000 4000"
  "4000 6000"
  "6000 10000"
  "20000 50000"

  # Add more pairs as needed
)

# Submit a job for each pair
for pair in "${seuil_pairs[@]}"; do
  # Extract seuil_begin and seuil_end from the pair
  set -- $pair
  seuil_begin=$1
  seuil_end=$2
  
  # Define the job name including seuil_begin and seuil_end
  job_name="fill_loops_${seuil_begin}_${seuil_end}"

  # Submit the script with each pair of seuils and set the job name
  sbatch --job-name="$job_name" fill_MIP_cycles.sh "$seuil_begin" "$seuil_end"
done
