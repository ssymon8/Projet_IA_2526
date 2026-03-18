#!/bin/bash
#SBATCH --job-name=demucs_l40
#SBATCH --partition=ENSTA-l40s       # <-- Changement ici
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --time=04:00:00              # Rappel: max 4h pour les étudiants 
#SBATCH --output=demucs_l40_%j.log

source ~/Projet_IA_2526/envSon/bin/activate
cd ~/Projet_IA_2526
bash fine-tuning_demucs.sh