#!/bin/bash
#SBATCH --job-name=inference_precip         # short name for the job
#SBATCH --nodes=1                   # number of nodes
#SBATCH --ntasks-per-node=1         # run 1 task per node
#SBATCH --gpus-per-node=1           # GPUs per node
#SBATCH -c 72                       # CPU cores per task
#SBATCH --mem=460000                # memory per node
#SBATCH --exclusive
#SBATCH --time=01:50:00             # total run time (HH:MM:SS)
#SBATCH --account=YOUR_SLURM_ACCOUNT
#SBATCH --partition=normal
#SBATCH --output=inference_precip_twelve.out  # output log file


##### Number of total processes 
echo "Nodelist:= " $SLURM_JOB_NODELIST
echo "Number of nodes:= " $SLURM_JOB_NUM_NODES
echo "Ntasks per node:= "  $SLURM_NTASKS_PER_NODE

# If you want to load things from your .bashrc profile, e.g. cuda drivers, singularity etc 
source ~/.bashrc
export WANDB_KEY=$WANDB_API_KEY
export WANDB__SERVICE_WAIT=300
export WANDB_CACHE_DIR="/iopsstor/scratch/cscs/$USER/wandb/cache"
export WANDB_ARTIFACT_LOCATION="/iopsstor/scratch/cscs/$USER/wandb/artifact_location"
export WANDB_ARTIFACT_DIR="/iopsstor/scratch/cscs/$USER/wandb/artifact_dir"
export WANDB_CONFIG_DIR="/iopsstor/scratch/cscs/$USER/wandb/config"
export WANDB_DATA_DIR="/iopsstor/scratch/cscs/$USER/wandb/data_dir"
export OMP_NUM_THREADS=1
ulimit -c 0  # disable core dumps
ulimit -t unlimited

set -x

# ------------------------------------------------------------------------------
# Rendezvous setup
# ------------------------------------------------------------------------------
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR="$master_addr"
export MASTER_PORT=29501

node_rank=$SLURM_NODEID
nnodes=$SLURM_JOB_NUM_NODES

echo "================= SLURM Info ================="
echo "Run started at:    $(date)"
echo "SLURM_JOB_ID:      $SLURM_JOB_ID"
echo "SLURM_NODELIST:    $SLURM_JOB_NODELIST"
echo "MASTER_ADDR:       $MASTER_ADDR"
echo "MASTER_PORT:       $MASTER_PORT"
echo "nnodes:            $nnodes"
echo "node_rank:         $node_rank"
echo "================================================"


CURRENT_TIME=$(date +"%Y-%m-%d--%H-%M-%S")

workdir="/users/$USER/SwissCliM_aurora"
tomlpath="/users/$USER/.edf/torchcontainer_clariden_yun_root.toml"
set -x

NTRAIN=1
NVAL=1

srun --ntasks=$nnodes \
     --export=ALL \
     --environment=$tomlpath \
     --container-workdir=$workdir \
     -u -l torchrun \
      --nnodes=$SLURM_JOB_NUM_NODES \
      --node_rank=$SLURM_PROCID \
      --nproc_per_node=1 \
      --rdzv_id=42 \
      --rdzv_backend=c10d \
      --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
      inference_direct.py \
      --batch_size 1 \
        --num_workers 1 \
        --epochs 10 \
        --devices 1 \
        --num_nodes $SLURM_JOB_NUM_NODES \
        --log_dir "/path/to/checkpoints/precip_new/lw_2_new/" \
        --name_ckpt "model_ckpt-step=6700-loss_train=0.07.ckpt" \
        --dataset_config_path "dataset_config.yaml" \
        --data_sources "era5_zwd_precip" \
        --Ntest 600\
        --output_prefix "twelve_checkpoint" \
        --save_baseline \
        --baseline_ckpt "/path/to/checkpoints/precip_new/without_zwd_new/model_ckpt-step=6700-loss_train=0.07.ckpt" \
        --baseline_data_sources "era5_zwd_precip_without_zwd" \
        # --start_time_train "2002-05-02T00:00:00" \
        # --end_time_train "2014-06-30T23:00:00" \
        # --start_time_val "2014-07-01T00:00:00" \
        # --end_time_val "2015-12-31T23:00:00"


echo "Run finished at: $(date)"
