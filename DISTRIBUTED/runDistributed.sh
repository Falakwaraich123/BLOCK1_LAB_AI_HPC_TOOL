#!/bin/bash -l
#SBATCH --nodes=2               # node count
#SBATCH --ntasks-per-node=2     # total number of tasks per node
#SBATCH --cpus-per-task=32        # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem=64G                # total memory per node (4 GB per cpu-core is default)
#SBATCH --gres=gpu:a100:2             # number of gpus per node
#SBATCH --time=01:30:00          # total run time limit (HH:MM:SS)


# Load and activate your Python environment
source $STORE/mypython/bin/activate

# Debugging flags
export NCCL_DEBUG=INFO
export PYTHONFAULTHANDLER=1

# Work distribution strategies
STRATEGY="ddp"
# STRATEGY="fsdp"
# STRATEGY="deepspeed"

# Use srun to distribute the Python script execution across the resources
which python
srun python DISTRIBUTED.py --strategy $STRATEGY
