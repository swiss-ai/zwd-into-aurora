#!/bin/bash
#SBATCH --job-name=mswep_6h_acc
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --time=00:30:00
#SBATCH --partition=normal
#SBATCH --output=logs/mswep_6h_acc_%j.out
#SBATCH --error=logs/mswep_6h_acc_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zarr_env

python3 compute_mswep_6h_accumulation.py
