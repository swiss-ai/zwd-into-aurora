#!/bin/bash
#SBATCH --job-name=check_zarr_for_zwd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --time=00:30:00
#SBATCH --partition=debug
#SBATCH --output=logs/check_zarr_for_zwd_%j.out
#SBATCH --error=logs/check_zarr_for_zwd_%j.err

source ~/miniconda3/etc/profile.d/conda.sh
conda activate zarr_env

# python check_zarr_for_zwd.py --pred /path/to/checkpoints/precip_small/lw_0.1/pred_2020-07-25_600steps.zarr --target /path/to/checkpoints/precip_small/lw_0.1/target_2020-07-25_600steps.zarr --baseline /path/to/checkpoints/precip_small/lw_0.1/pred_baseline_2020-07-25_600steps.zarr --static "static_vars.nc" --every-n 10 --plot-dir "precip_small"
python check_zarr_for_zwd.py --pred /path/to/checkpoints/zwd/lr_1e-4_lw_scheduler/pred_2020-07-25_600steps.zarr --target /path/to/checkpoints/zwd/lr_1e-4_lw_scheduler/target_2020-07-25_600steps.zarr --baseline /path/to/checkpoints/zwd/lr_1e-4_lw_scheduler/pred_baseline_2020-07-25_600steps.zarr --static "static_vars.nc" --every-n 50 --plot-dir "zwd_large"
# python check_zarr_for_zwd.py --pred /path/to/checkpoints/zwd_small/lr_5e-5_lw_scheduler/pred_2020-07-25_600steps.zarr --target /path/to/checkpoints/zwd_small/lr_5e-5_lw_scheduler/target_2020-07-25_600steps.zarr --baseline /path/to/checkpoints/zwd_small/lr_5e-5_lw_scheduler/pred_baseline_2020-07-25_600steps.zarr --static "static_vars.nc" --every-n 10 --plot-dir "zwd_small"
