#!/bin/bash
#SBATCH --job-name=compare_zwd_int
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --output=logs/compare_zwd_int_%j.out
#SBATCH --error=logs/compare_zwd_int_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zarr_env

export DATA_ROOT="${DATA_ROOT:-/path/to/data}"
WORK_DIR="${WORK_DIR:-/path/to/checkpoints}"

python compare_zwd_integration.py \
    --target ${WORK_DIR}/zwd/lr_1e-4_lw_scheduler/target_2020-07-25_600steps.zarr \
    --external zwd_20200801.zarr \
    --static static_vars.nc \
    --date 2020-08-01 \
    --plot-dir compare_zwd_output
