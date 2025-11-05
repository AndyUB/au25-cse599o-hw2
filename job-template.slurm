#!/bin/bash
SBATCH --job-name=andy-hw2               # Job name

SBATCH --account=yhruan22                # Replace with your netid
SBATCH --partition=gpu                   # Partition name (default is gpu)
SBATCH --nodes=1                         # One node (tomago/tempura)
SBATCH --ntasks-per-node=8               # Number of CPU cores (vary it as needed)
SBATCH --mem=16G                         # Memory per node
SBATCH --gpus=8                          # Number of GPUs (for HW1, let it be 1)
SBATCH --time=02:00:00                   # Max runtime = 2 hours (cluster limit)

SBATCH --chdir=/homes/iws/yhruan22/au25-cse599o-hw2/cse599o_systems  # Working directory
SBATCH --export=ALL                      # Export environment variables
SBATCH --output=yhruan22-slurm-%j.out    # STDOUT log file (%j = JobID)
SBATCH --error=yhruan22-slurm-%j.err     # STDERR log file (%j = JobID)

# --- Example Command to run ---
srun uv run pytest