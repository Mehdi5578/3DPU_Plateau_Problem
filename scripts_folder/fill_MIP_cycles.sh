#!/bin/bash
#SBATCH --job-name=MIP_loops             # Nom du job
#SBATCH --output=%x_%j.out                 # Log de sortie standard (%x pour nom du job, %j pour jobId)
#SBATCH --error=%x_%j.err                  # Log d'erreur standard
#SBATCH --mem=50G                         # Mémoire par noeud
#SBATCH --cpus-per-task=10                 # CPU par tâche
#SBATCH --time=2-00:10                     # Limite de temps en J-HH:MM	
#SBATCH --mail-type=BEGIN,END,FAIL         # Notifications par mail
#SBATCH --mail-user=oudaoud.mehdi@gmail.com  # Votre adresse e-mail

# Charger les modules nécessaires
module load StdEnv/2023
module load gurobi/11.0.1

# Récupérer les arguments de seuil
seuil_begin=$1
seuil_end=$2

# Exécuter le script Python en redirigeant la sortie vers un fichier nommé avec le JobID et les valeurs de seuil
python /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/src/tests/test_ROMEO_MIP.py $seuil_begin $seuil_end > /home/mehdii/projects/def-vidalthi/mehdii/3DPU_Plateau_Problem/scripts_folder/MIP_cycles/loops_filled_${seuil_begin}_${seuil_end}_${SLURM_JOB_ID}.txt
