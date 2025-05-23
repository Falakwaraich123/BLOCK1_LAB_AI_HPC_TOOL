#!/bin/bash -l
# SLURM SUBMIT SCRIPT
#SBATCH -N 2             # This needs to match Trainer(num_nodes=...)
#SBATCH --gres=gpu:a100:2
#SBATCH --ntasks-per-node=2   # This needs to match Trainer(devices=...)
#SBATCH --mem=8G
#SBATCH -c 32
#SBATCH --time=01:30:00          # total run time limit (HH:MM:SS)

# Load and activate your Python environment
source $STORE/mypython/bin/deactivate
source $STORE/mypython/bin/activate


# debugging flags (optional)
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Verify the Python path and run the Python script
which python

# Use srun to distribute the Python script execution across the resources
srun python BASELINE.py

