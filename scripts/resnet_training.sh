#!/bin/bash
# This is an example batch script for Slurm on Hydra
#
# To submit this script to Slurm, use: sbatch batch_script.sh
#
# Once the job starts, you will see a file MyPythonJob-****.out
# The **** will be the Slurm JobID

#--- Start of Slurm commands ---
#SBATCH --partition=gpus
#SBATCH --gres=gpu:4
#SBATCH --time=6:00:00
#SBATCH --mem=32G
#SBATCH -J MyPythonJob
#SBATCH -o MyPythonJob-%j.out
#SBATCH -e MyPythonJob-%j.err
#--- End of Slurm commands ---

# Load any necessary modules here (if needed)
# module load python

# Activate your Python virtual environment (if applicable)
# source /path/to/your/venv/bin/activate

# Run your Python script
python resnet_training.py