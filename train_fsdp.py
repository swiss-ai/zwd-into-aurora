## import modules

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
import os
import glob
import sys
import math
import pickle
import numpy as np
import pandas as pd
import wandb
from datetime import datetime, timedelta
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist
# torch.backends.cuda.enable_mem_efficient_sdp(False)
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_math_sdp(True)

# Import FSDP and Mixed Precision from PyTorch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

import lightning as L
from lightning.pytorch.callbacks import Callback, ModelCheckpoint, DeviceStatsMonitor, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import FSDPStrategy, DDPStrategy
from huggingface_hub import hf_hub_download

from aurora import Aurora, Batch, Metadata
from aurora.model import swin3d as aurora_swin3d_module
from aurora.model.decoder import Perceiver3DDecoder

## import custom modules
import yaml
from config import parse_args
from utils import dataset, logging_utils, losses, model_freezer
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR, ConstantLR
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler
from torch.utils.data.distributed import DistributedSampler
import psutil
import time

from utils.gradient_logging import log_gradient_norms,log_weight_norms
from lightning.pytorch.callbacks import ThroughputMonitor
from lightning.fabric.utilities.throughput import measure_flops


# add environment variables for memory management
# expandable_segments:True -> Allows memory segments to grow
# garbage_collection_threshold:0.8 -> Triggers cleanup when 80% memory is used
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True,garbage_collection_threshold:0.8'

_DEBUG = True
is_rank0 = False
if _DEBUG:
    if int(os.getenv("LOCAL_RANK", "0")) == 0:
        is_rank0 = True
        logging_utils.log_entire_script(__file__)

args = parse_args()

## 
cmip6 = False 
is_bristen = False
is_clariden = True
is_switch = False

assert (is_bristen + is_clariden + is_switch) == 1 # only 1 can be valid
    
wandb_key = os.getenv("WANDB_KEY")
if is_rank0:  # Only attempt login on rank 0
    if wandb_key:
        wandb.login(key=wandb_key)
    else:
        if args.wnb_mode != 'disabled':
            print(f'WANDB_KEY is not set. Please set WANDB_KEY to use wandb. not logging to WANDB.')
            args.wnb_mode = 'disabled'

def get_total_gpus():
    # Get number of GPUs on the current node
    gpus_per_node = torch.cuda.device_count()
    # Get total number of nodes (ensure this is set externally)
    total_gpus = int(os.getenv("WORLD_SIZE", "1"))  # Default to 1 node if not set
    # total_gpus = gpus_per_node * num_nodes
    return total_gpus

## setup dataset & dataloader
# Dataset scheme to use. For now, will use raw .zarr from weatherbench2, but this is to be changed later.
if is_switch:
    start_time = datetime(2021, 1, 1, 0, 0, 0)
    end_time = datetime(2021, 1, 5, 23, 0, 0)

    # List of datetime objects with 1-hour steps
    time_steps = [np.datetime64(start_time + timedelta(hours=i)) for i in range(int((end_time - start_time).total_seconds() // 3600) + 1)]
    path_weatherbench2 = '/path/to/data/weatherbench2_subset.zarr'
    inds_train = time_steps
    inds_val = inds_train

if is_bristen:
    DATA_PATH_PREFIX = '/path/to/data/'
    # start_time_train = datetime(1979, 1, 1, 0, 0, 0)
    start_time_train = datetime(2020, 12, 21, 0, 0, 0) ## making it 10 days for testing it now
    end_time_train = datetime(2020, 12, 31, 23, 0, 0)
    start_time_val = datetime(2021, 1, 1, 0, 0, 0)
    # end_time_val = datetime(2021, 12, 31, 23, 0, 0) ## last date on wb2 is 2021-12-31
    end_time_val = datetime(2021, 1, 2, 23, 0, 0)
    path_weatherbench2 = '/path/to/data/weatherbench2_original'
    path_cmip6 = '' # TODO: add path in briston
    path_slt = '/path/to/data/ERA5_Soiltype.npy'
    slt = np.load(path_slt)
    inds_train = [np.datetime64(start_time_train + timedelta(hours=i)) for i in range(int((end_time_train - start_time_train).total_seconds() // 3600) + 1)]
    inds_val = [np.datetime64(start_time_val + timedelta(hours=i)) for i in range(int((end_time_val - start_time_val).total_seconds() // 3600) + 1)]

if is_clariden:
    DATA_PATH_PREFIX = '/path/to/data/'
    start_time_train = datetime(1979, 1, 1, 0, 0, 0)
    #start_time_train = datetime(2020, 11, 1, 0, 0, 0) ## making it 2 months for testing it now
    end_time_train = datetime(2020, 12, 31, 23, 0, 0)
    start_time_val = datetime(2021, 1, 1, 0, 0, 0)
    end_time_val = datetime(2021, 12, 31, 23, 0, 0) ## last date on wb2 is 2021-12-31
    #end_time_val = datetime(2021, 1, 2, 23, 0, 0)
    path_weatherbench2 = '/path/to/data/weatherbench2_original'
    path_cmip6 = ''
    path_slt = '/path/to/data/ERA5_Soiltype.npy'
    slt = np.load(path_slt)
    inds_train = [np.datetime64(start_time_train + timedelta(hours=i)) for i in range(int((end_time_train - start_time_train).total_seconds() // 3600) + 1)]
    inds_val = [np.datetime64(start_time_val + timedelta(hours=i)) for i in range(int((end_time_val - start_time_val).total_seconds() // 3600) + 1)]


def get_device(use_gpu):
    return "cuda" if use_gpu and torch.cuda.is_available() else "cpu"


def get_time_indices(yml_file, dataset_type):
    """ Create time indices for training and validation datasets. Default to 1 hour sampling."""
    # training time setup
    start_time_train = datetime.strptime(yml_file[dataset_type]['start_time_train'], '%Y-%m-%dT%H:%M:%S') if 'start_time_train' in yml_file[dataset_type] else datetime(1979, 1, 1, 0, 0, 0)
    end_time_train = datetime.strptime(yml_file[dataset_type]['end_time_train'], '%Y-%m-%dT%H:%M:%S') if 'end_time_train' in yml_file[dataset_type] else datetime(2020, 12, 31, 23, 0, 0)
    step_time_train = yml_file[dataset_type]['step_time_train'] if 'step_time_train' in yml_file[dataset_type] else 1

    # validation time setup
    start_time_val = datetime.strptime(yml_file[dataset_type]['start_time_val'], '%Y-%m-%dT%H:%M:%S') if 'start_time_val' in yml_file[dataset_type] else datetime(2021, 1, 1, 0, 0, 0)
    end_time_val = datetime.strptime(yml_file[dataset_type]['end_time_val'], '%Y-%m-%dT%H:%M:%S') if 'end_time_val' in yml_file[dataset_type] else datetime(2021, 12, 31, 23, 0, 0)
    step_time_val = yml_file[dataset_type]['step_time_val'] if 'step_time_val' in yml_file[dataset_type] else 1

    inds_train = np.array(pd.date_range(start=start_time_train, end=end_time_train, freq=f'{step_time_train}h'))
    inds_val = np.array(pd.date_range(start=start_time_val, end=end_time_val, freq=f'{step_time_val}h'))

    return inds_train, inds_val


_TIME_FMT = "%Y-%m-%dT%H:%M:%S.%f"           # keeps the format in one place
def _make_batch(batch_dict, *, normalize_target: bool = True):
    """
    Convert the default-collated dictionary into (batch_x, batch_y)
    Batch objects – exactly what you were building inside train_step.
    """
    # time & level metadata

    x_time = tuple(datetime.strptime(t[:26], _TIME_FMT) for t in batch_dict["x_time"])
    y_time = tuple(datetime.strptime(t[:26], _TIME_FMT) for t in batch_dict["y_time"])
    atmos_levels = tuple(batch_dict["atmos_levels"][0].cpu().numpy().tolist())
    lead_time_seconds = batch_dict.get("lead_time_seconds", [timedelta(hours=6).total_seconds()])[0]  # Default to 6 hours if not provided
    if isinstance(lead_time_seconds, timedelta):
        lead_time_seconds = lead_time_seconds.total_seconds()


    # Build Batch objects
    batch_x = Batch(
        surf_vars=batch_dict["x_srf"],
        static_vars={k: v[0] for k, v in batch_dict["x_static"].items()},
        atmos_vars=batch_dict["x_atmos"],
        metadata=Metadata(
            dataset_name=batch_dict['name'][0],
            lat=batch_dict["lat"][0],
            lon=batch_dict["lon"][0],
            time=x_time,
            atmos_levels=atmos_levels,
            locations={k: v[0] for k, v in batch_dict["locations"].items()},
            scales={k: v[0] for k, v in batch_dict["scales"].items()},
            grid_resolution=batch_dict["grid_resolution"][0],
            is_global_observation=batch_dict["is_global_observation"][0],
            lead_time_seconds=lead_time_seconds,
        ),
    )

    batch_y = Batch(
        surf_vars=batch_dict["y_srf"],
        static_vars={k: v for k, v in batch_dict["y_static"].items()},
        atmos_vars=batch_dict["y_atmos"],
        metadata=Metadata(
            dataset_name=batch_dict['name'][0],
            lat=batch_dict["lat"][0],
            lon=batch_dict["lon"][0],
            time=y_time,
            atmos_levels=atmos_levels,
            locations={k: v[0] for k, v in batch_dict["locations"].items()},
            scales={k: v[0] for k, v in batch_dict["scales"].items()},
            grid_resolution=batch_dict["grid_resolution"][0],
            is_global_observation=batch_dict["is_global_observation"][0],
            lead_time_seconds=lead_time_seconds,
        ),
    )

    if normalize_target:
        batch_y = batch_y.normalise(surf_stats=None)

    return batch_x, batch_y

# # Define collate functions for training and validation
collate_fn_train = lambda samples: _make_batch(torch.utils.data.default_collate(samples), normalize_target=True)
collate_fn_val   = lambda samples: _make_batch(torch.utils.data.default_collate(samples), normalize_target=False)


def load_data(yaml_path, datasets_type):
    # TODO: replace this function with LightningDataModule
    """Load datasets based on YAML configuration."""
    # path_slt = os.path.join(DATA_PATH_PREFIX, 'ERA5_Soiltype.npy')
    # slt = np.load(path_slt)
    surf_vars, atmos_vars, static_vars = (), (), ()
    dataset_train, dataset_val = [], []
    batch_sizes = []
    subset_sizes = []

    for dataset_type in datasets_type:
        with open(yaml_path, 'r') as file:
            yml_file = yaml.safe_load(file)
            data_cls = getattr(dataset, yml_file[dataset_type]['class'])
            conf_train = yml_file[dataset_type]['conf']
            conf_train['path'] = os.path.join(DATA_PATH_PREFIX, conf_train['path'])
            
            if dataset_type in ['era5', 'era5_climate', 'era5_spatial_unmask', 'era5_vertical_unmask', 'era5_variable_unmask']:
                conf_train['inds'] = inds_train
                # conf_train['slt'] = slt
                conf_val = conf_train.copy()
                conf_val['inds'] = inds_val
                conf_val['step_time'] = yml_file[dataset_type].get('step_time_val', 1)  # Default to 1 hour step time

            elif dataset_type in ['era5_without_zwd', 'era5_precip', 'era5_hydro', 'era5_zwd', 'era5_zwd_precip', 'era5_zwd_precip_without_zwd', 'era5_zwd_1h', 'era5_without_zwd_1h']:
                inds_train, inds_val = get_time_indices(yml_file, dataset_type)
                conf_train['inds'] = inds_train
                conf_train['atmos_levels'] = np.asarray(conf_train['atmos_levels'], dtype=np.int32)
                conf_train['slt'] = slt
                conf_train['step_time'] = yml_file[dataset_type].get('step_time_train', 1)  # Default to 1 hour step time
                conf_val = conf_train.copy()
                conf_val['inds'] = inds_val
                conf_val['step_time'] = yml_file[dataset_type].get('step_time_val', 1)  # Default to 1 hour step time
                
            else:
                conf_train['start_idx'] = yml_file[dataset_type]['start_train']
                conf_train['end_idx'] = yml_file[dataset_type]['end_train']
                if 'wb2_path' in conf_train:
                    conf_train['wb2_path'] = os.path.join(DATA_PATH_PREFIX, conf_train['wb2_path'])
                conf_val = conf_train.copy()
                conf_val['start_idx'] = yml_file[dataset_type]['start_val']
                conf_val['end_idx'] = yml_file[dataset_type]['end_val']

            surf_vars += tuple([yml_file[dataset_type]['conf']['variable_name_mapping'].get(var, var) 
                              for var in yml_file[dataset_type]['conf']['surf_vars']] 
                              if 'variable_name_mapping' in yml_file[dataset_type]['conf'] 
                              else yml_file[dataset_type]['conf']['surf_vars'])
            atmos_vars += tuple([yml_file[dataset_type]['conf']['variable_name_mapping'].get(var, var) 
                                for var in yml_file[dataset_type]['conf']['atmos_vars']]
                                if 'variable_name_mapping' in yml_file[dataset_type]['conf']
                                else yml_file[dataset_type]['conf']['atmos_vars'])
            static_vars += tuple([yml_file[dataset_type]['conf']['variable_name_mapping'].get(var, var) 
                                for var in yml_file[dataset_type]['conf']['static_vars']]
                                if 'variable_name_mapping' in yml_file[dataset_type]['conf']
                                else yml_file[dataset_type]['conf']['static_vars'])

            batch_sizes.append(yml_file[dataset_type]['batch_size'])
            subset_sizes.append(yml_file[dataset_type].get('subset_size', 256))

            dataset_train.append(data_cls(**conf_train))
            dataset_val.append(data_cls(**conf_val))
            
    ## keep only unique var names and remove repetitions in tuple. (Note: set() op. is not deterministic.)
    surf_vars = tuple(sorted(set(surf_vars)))
    atmos_vars = tuple(sorted(set(atmos_vars)))
    static_vars = tuple(sorted(set(static_vars)))
    if args.devices > 1:
        dist.init_process_group(backend=args.backend)
    if len(dataset_train) == 1:
        dataloader_train = StatefulDataLoader(
            dataset_train[0], 
            sampler=StatefulDistributedSampler(dataset_train[0], seed=args.seed, drop_last=True) if args.devices > 1 else None,
            batch_size=batch_sizes[0],
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,  
            persistent_workers=True,
            collate_fn=collate_fn_train,
            prefetch_factor=4,
        )
        dataloader_val = DataLoader(
            dataset_val[0],
            sampler=DistributedSampler(dataset_val[0], shuffle=False, drop_last=True) if args.devices > 1 else None,
            batch_size=batch_sizes[0],
            num_workers=args.num_workers,
            # shuffle=False,  # Ensure deterministic for validation
            drop_last=True,
            pin_memory=True,  
            persistent_workers=True,
            collate_fn=collate_fn_val  
        )
        if args.dump_datasampler_indices and args.devices > 1:
            logging_utils.save_sampled_indices_across_ranks(dataloader_train.sampler, seed=args.seed, rank=int(dist.get_rank()), output_dir=os.path.join(args.log_dir, 'data_sampler_indices'))  # Save sampled indices for the first epoch
            logging.info(f"saved sampler indices for rank: {dist.get_rank()}.")
    else:
        dataloader_train = dataset.StatefulMultiDatasetLoader(
            datasets=dataset_train,
            samplers=[StatefulDistributedSampler(ds) for ds in dataset_train] if args.devices > 1 else None,
            batch_sizes=batch_sizes,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            collate_fns=[collate_fn_train for _ in dataset_train],
        ) 
        
        dataloader_val = dataset.StatefulMultiDatasetLoader(
            datasets=dataset_val,
            samplers=[StatefulDistributedSampler(ds) for ds in dataset_val] if args.devices > 1 else None,
            batch_sizes=batch_sizes,
            num_workers=args.num_workers,
            drop_last=True,
            pin_memory=True,
            persistent_workers=True,
            collate_fns=[collate_fn_val for _ in dataset_val],
        ) 

    return dataloader_train, dataloader_val, surf_vars, atmos_vars, static_vars



def main():
    L.seed_everything(args.seed, workers=True)
    logging_utils.copy_exp_params(log_dir=args.log_dir, config_file=args.config, args=args)
    logging.info("args = %s", args)
    device = get_device(use_gpu=not args.no_gpu)

    if not args.no_gpu and not torch.cuda.is_available():
        logging.info("GPU training is requested, but no GPU device available.")
        sys.exit(1)

    # Replace the existing dataset setup with:
    dataloader_train, dataloader_val, surf_vars, atmos_vars, static_vars = load_data(
        args.dataset_config_path, args.data_sources
    )
    
    ## Need to check if the target checkpoint exists when --resume is passed.
    ## Use --ckpt_name (or an absolute path) instead of hardcoding last.ckpt.
    if args.resume: 
        ckpt_fname = args.ckpt_name if os.path.isabs(args.ckpt_name) else os.path.join(args.log_dir, args.ckpt_name)
        if os.path.isfile(ckpt_fname):
            trainer_fit_ckpt_path = ckpt_fname
            checkpoint = torch.load(trainer_fit_ckpt_path, weights_only=False)
            if 'dataloader_state' in checkpoint.keys():
                dataloader_train.load_state_dict(checkpoint['dataloader_state'])
                sampler_train = dataloader_train.sampler
                if args.dump_datasampler_indices and args.devices > 1:
                    logging_utils.save_sampled_indices_across_ranks(sampler_train, seed=args.seed, rank=int(dist.get_rank()), output_dir=os.path.join(args.log_dir, 'resumed_data_sampler_indices')) 
            else:
                logging.info("Warning: No dataloader state found in checkpoint. Training will start observations from scratch!")
        else:
            logging.info(
                f"\n\n\n--resume is passed, but could not find ckpt at {ckpt_fname}. Starting from scratch.\n\n\n"
            )
            trainer_fit_ckpt_path = None
            args.resume = False
    else:
        trainer_fit_ckpt_path = None

    ## setup model architecture
    encoder_act_checkpointing = args.act_checkpointing_encoder
    backbone_act_checkpointing = args.act_checkpointing_backbone
    decoder_act_checkpointing = args.act_checkpointing_decoder
    use_small_model = False
    if use_small_model:
        # Minimal model architecture
        model = Aurora(
            use_lora=False, 
            autocast=False, # Use AMP (mixed precision to fit to GPU)
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            encoder_depths=(3, 5, 4),
            encoder_num_heads=(4, 8, 16),
            decoder_depths=(4, 5, 3),
            decoder_num_heads=(16, 8, 4),
            embed_dim=128,
        )
    else:
        # Default model architecture
        str_architecture_size = args.str_architecture_size
        if str_architecture_size == "small":
            encoder_depths = (2,6,2)
            encoder_num_heads = (4,8,16)
            decoder_depths = (2, 6, 2)
            decoder_num_heads = (16, 8, 4)
            embed_dim = 256
            num_heads = 8
            hf_pretrain_fname = 'aurora-0.25-small-pretrained.ckpt'
        elif str_architecture_size == "large":
            encoder_depths = (6, 10, 8)
            encoder_num_heads = (8, 16, 32)
            decoder_depths = (8, 10, 6)
            decoder_num_heads = (32, 16, 8)
            embed_dim= 512
            num_heads = 16
            hf_pretrain_fname = 'aurora-0.25-pretrained.ckpt'
        else:
            raise ValueError(f"Unknown architecture size: {str_architecture_size}. Choose 'small' or 'large'.")
            
        model = Aurora(
            use_lora=False, 
            autocast=True, # Use AMP (mixed precision to fit to GPU)
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_path=0.2,
            num_ensemble = args.num_ensemble,  # Number of ensemble members
            use_smoe = args.use_smoe,  # use SMoEs
            num_experts = args.num_experts,   # New parameter for SMoE
            rc_loss = args.rc_loss,  # Enable routing classification loss
            save_error_signal=True,  # Enable error signal saving for RC loss
            block_gate_grad = args.block_gate_grad,  # Allow gate gradients for RC loss
            variable_aggregation= args.variable_aggregation,
            use_resolution_specific_patch_tokenizers = args.use_resolution_specific_patch_tokenizers,
            do_not_use_var_specific_bias_in_patch_tokenizer = args.do_not_use_var_specific_bias_in_patch_tokenizer, ## temporary feature, will be removed in the future.
            disable_flashattention=args.disable_flashattention,
            stabilise_level_agg=args.stabilise_level_agg,
            add_qk_norm_to_swin3d=args.add_qk_norm_to_swin3d,
            encoder_activation_checkpointing=encoder_act_checkpointing,  # Enable extensive checkpointing for large models
        )
        
        # Load the pretrained weights (aurora our custom).
        if not args.resume:
            if args.load_aurora_pretrain_weights:
                # Load pretrained weights from HuggingFace Hub
                path = hf_hub_download(repo_id="microsoft/aurora", filename=hf_pretrain_fname)
                if not cmip6:
                    # TODO: Add support for chkpt loading based on a parameter
                    model.load_checkpoint_local(path, strict=False) ## we do not yet have slt
            elif args.load_custom_pretrain_weights_str is not None:
                pretrained_weights_path = args.load_custom_pretrain_weights_str
                checkpoint = torch.load(pretrained_weights_path, map_location=next(model.parameters()).device, weights_only=False)
                # Handle Lightning checkpoint format: model weights are under 'state_dict' key with 'net.' prefix
                state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
                state_dict = {k[4:] if k.startswith('net.') else k: v for k, v in state_dict.items()}
                load_result = model.load_state_dict(state_dict, strict=False)
                logging.info(f"Loaded custom pretrain weights from {pretrained_weights_path}.")
                if load_result.missing_keys:
                    s = f"Missing keys when loading checkpoint {pretrained_weights_path}:"
                    for key in load_result.missing_keys:
                        s += f"\n{key}"
                    logging.warning(s)
                
                if load_result.unexpected_keys:
                    s = f"Unexpected keys when loading checkpoint {pretrained_weights_path}:"
                    for key in load_result.unexpected_keys:
                        s += f"\n{key}"
                    logging.warning(s)
        if backbone_act_checkpointing: # checkpointing only necessary for large model.
            model.configure_activation_checkpointing() # recalculates backbone activations on the backprop.

        # initialize freezer and freeze parts of the model, default is to un-freeze everything
        freezer = model_freezer.ModelFreezer(model)
        freezer.freeze_parts(
            freeze_encoder = args.freeze_encoder, 
            freeze_backbone = args.freeze_backbone,
            freeze_decoder = args.freeze_decoder,
            unfreeze_last_n_backbone_layers = 1
        )


    ## setup loss function    
    with open(args.loss_config_path, 'r') as file:
        yml_file = yaml.safe_load(file)
    surf_var_weights = yml_file['default']['surf_var_weights']
    atmos_var_weights = yml_file['default']['atmos_var_weights']

    #loss_obj = losses.MSELoss(d_var_weights=None, d_pres_weights=None) ## TODO: Define appropriate weights
    # loss_obj = losses.WeightedMAELoss()
    loss_obj = losses.CombinedLoss(
        mae_weight=1.0,
        nll_weight=args.nll_weight,
        crps_weight=args.crps_weight,
        kernel_crps_weight=args.kernel_crps_weight,
        stats_loss_weight=args.stats_loss_weight,
        kill_if_nan_in_preds= args.kill_on_nan_detection,
        surf_var_weights=surf_var_weights,
        atmos_var_weights=atmos_var_weights
    )
    
    def convert_to_wandb_image(x):
        ## normalize numpy array and then scale to 0-255 range and uint8.
        x = (x - x.min()) / (x.max() - x.min())
        x = (x * 255).astype(np.uint8)
        return x
    
    ## setup lightning module & trainer
    class LightningModule(L.LightningModule):
        def __init__(self, net, loss_fn, **kwargs):
            super().__init__()
            self.net = net
            self.loss_fn = loss_fn
            self.rc_quantile = kwargs.pop('rc_quantile', 0.7)  # Add RC loss quantile parameter
            self.example_input_array = kwargs.pop('example_input_array', None)
            self.lr_scheduler_interval = kwargs.pop('lr_scheduler_interval', 'step')
            self.batch_size = kwargs.pop('batch_size', None)
            self.learning_rate = kwargs.pop('learning_rate', 5e-4)  # Changed base learning rate to 5e-4
            self.warmup_steps = kwargs.pop('warmup_steps', 1000)    # 1k warmup steps
            self.weight_decay = kwargs.pop('weight_decay', 5e-6)    # AdamW weight decay
            self.rc_weight = kwargs.pop('rc_weight', 0.1)    # rc_weight for smoe
            self.aux_weight = kwargs.pop('aux_weight', 0.1)    # aux_weight for smoe
            for key, val in kwargs.items():
                setattr(self, key, val)
            self.save_hyperparameters(ignore=['net'])
            self.worst_metrics_train, self.worst_metrics_val, self.worst_metrics_test = {}, {}, {}
            self.is_ybatch_images_logged = False
            self.flops_per_batch = 95820522520576 * 2 # the ThoughtputMonitor callback will check this value
            self.use_constant_lr = kwargs.pop('use_constant_lr', True)  # Flag to use constant LR after warmup

        def on_train_start(self):
            """Initialize timing variables when training starts"""
            self.last_step_time = time.time()

        def configure_optimizers(self):
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.learning_rate,  # base_lr = 5e-4
                weight_decay=self.weight_decay,  # weight decay = 5e-6
                betas=tuple(args.opt_betas),
                eps=args.opt_eps,   
            )

            total_steps = self.trainer.estimated_stepping_batches

            # 1. Linear warm‑up: 1e‑8 → base_lr over warmup_steps
            warmup_scheduler = LinearLR(
                optimizer,
                start_factor=1e-8 / self.learning_rate,
                end_factor=1.0,
                total_iters=self.warmup_steps
            )

            # 2a. Cosine decay: base_lr → 0.1× base_lr
            cosine_scheduler = CosineAnnealingLR(
                optimizer,
                T_max=total_steps - self.warmup_steps,
                eta_min=self.learning_rate * 0.1
            )

            # 2b. Constant: keep base_lr for the rest of training
            constant_scheduler = ConstantLR(
                optimizer,
                factor=1.0,                                       # stay at base_lr
                total_iters=total_steps - self.warmup_steps
            )

            # Choose what to do after warm‑up
            if self.use_constant_lr:     # ← True ⇒ constant after warm‑up
                print("Using constant learning rate after warmup")
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, constant_scheduler],
                    milestones=[self.warmup_steps]
                )
            else:                        # False ⇒ cosine decay after warm‑up
                scheduler = SequentialLR(
                    optimizer,
                    schedulers=[warmup_scheduler, cosine_scheduler],
                    milestones=[self.warmup_steps]
                )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1
                }
            }



        def forward(self, x):
            return self.net.forward(x)

        @staticmethod
        def _make_batch(batch):
            # Create input batch
            x_time = tuple([datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f') for t in batch['x_time']])
            atmos_levels = tuple(batch['atmos_levels'][0].cpu().numpy().tolist())
            static_vars = {k: batch['x_static'][k][0] for k in batch['x_static'].keys()}

            batch_obj_x = Batch(
                surf_vars={k: batch['x_srf'][k] for k in batch['x_srf'].keys()},
                static_vars=static_vars,
                atmos_vars={k: batch['x_atmos'][k] for k in batch['x_atmos'].keys()},
                metadata=Metadata(
                    dataset_name=batch['name'][0],
                    lat=batch['lat'][0],
                    lon=batch['lon'][0],
                    time=x_time,
                    atmos_levels=atmos_levels,
                    locations={k: v[0] for k, v in batch['locations'].items()},
                    scales={k: v[0] for k, v in batch['scales'].items()},
                    grid_resolution=batch['grid_resolution'][0],
                    is_global_observation=batch['is_global_observation'][0]
                )
            )

            # Create target batch
            y_time = tuple([datetime.strptime(t, '%Y-%m-%dT%H:%M:%S.%f') for t in batch['y_time']])
            batch_obj_y = Batch(
                surf_vars={k: batch['y_srf'][k] for k in batch['y_srf'].keys()},
                static_vars={k: batch['y_static'][k] for k in batch['y_static'].keys()},
                atmos_vars={k: batch['y_atmos'][k] for k in batch['y_atmos'].keys()},
                metadata=Metadata(
                    dataset_name=batch['name'][0],
                    lat=batch['lat'][0],
                    lon=batch['lon'][0],
                    time=y_time,
                    atmos_levels=atmos_levels,
                    locations={k: v[0] for k, v in batch['locations'].items()},
                    scales={k: v[0] for k, v in batch['scales'].items()},
                    grid_resolution=batch['grid_resolution'][0],
                    is_global_observation=batch['is_global_observation'][0]
                )
            )
            del batch
            return batch_obj_x, batch_obj_y


        
        def _get_system_metrics(self):
            """Gather system metrics including GPU, CPU, memory, and throughput."""
            try:
                # Calculate throughput
                current_time = time.time()
                batch_size = self.trainer.train_dataloader.batch_size
                world_size = self.trainer.world_size
                step_elapsed_time = current_time - self.last_step_time
                step_throughput = (batch_size * world_size) / step_elapsed_time

              
                        
                # Prepare metrics dictionary with proper WandB formatting
                metrics = {
                    'performance/throughput': step_throughput,  # Changed naming for better WandB organization
                    'performance/cpu_usage': psutil.cpu_percent(),
                    'performance/memory_usage_percent': psutil.virtual_memory().percent,
                    'performance/memory_used_gb': psutil.virtual_memory().used / (1024**3),
                    'performance/memory_available_gb': psutil.virtual_memory().available / (1024**3)
                }
                
                # Add GPU metrics
                if torch.cuda.is_available():
                    for i in range(torch.cuda.device_count()):
                        metrics.update({
                            f'performance/gpu_{i}/memory_used_mb': torch.cuda.memory_allocated(i) / 1024**2,
                            f'performance/gpu_{i}/memory_reserved_mb': torch.cuda.memory_reserved(i) / 1024**2,
                            f'performance/gpu_{i}/max_memory_mb': torch.cuda.max_memory_allocated(i) / 1024**2
                        })
                
                self.last_step_time = current_time
                return metrics
                
            except Exception as e:
                print(f"Error collecting system metrics: {str(e)}")
                return {}
            
        def _check_for_nan_weights(self):
            is_nans = False
            for name, param in self.named_parameters():
                # print(f'{name}: {param}')
                if torch.isnan(param).any():
                    logging.error(f"NaN detected in layer: {name}")
                    is_nans = True
                if torch.isinf(param).any():
                    logging.error(f"Inf detected in layer: {name}")
                    is_nans = True
            return is_nans

        def measure_model_flops(self, batch_obj_x):
            """Measure FLOPs for the current model with the given batch."""
            # Simple lambda for forward pass
            forward_only = lambda: self.net(batch_obj_x)
            # Measure FLOPs
            flops = measure_flops(self.net, forward_only)
            # Set the attribute for ThroughputMonitor
            self.flops_per_batch = flops * 2 
                
            return flops
        
        def _is_normalized_input_magnitude_above_threshold(self, x_normalized, threshold, return_xtime=True):
            """Check if the normalized input magnitude exceeds a threshold.
            x_normalized: Batch object with normalized input data.
            threshold: Magnitude threshold to check against.
            return_xtime: If True, return xtime of the batch if any variable exceeds the threshold as a tuple (bool, xtime).
            """
            is_above_th = False
            for var in x_normalized.surf_vars.keys():
                val = x_normalized.surf_vars[var].flatten().cpu().numpy() # (b, t, h, w) -> (l)
                if np.abs(val).max() > threshold:
                    is_above_th = True
                    if return_xtime:
                        xtimes = x_normalized.metadata.xtime[0]  # Return xtime for all minibatch.
                        return is_above_th, xtimes
                    else:
                        return is_above_th
            for var in x_normalized.atmos_vars.keys():
                val = x_normalized.atmos_vars[var].flatten().cpu().numpy() # (b, t, c, h, w) -> (l)
                if np.abs(val).max() > threshold:
                    is_above_th = True
                    if return_xtime:
                        xtimes = x_normalized.metadata.xtime[0]  # Return xtime for all minibatch.
                        return is_above_th, xtimes
                    else:
                        return is_above_th
            # Explicit return if no variable exceeds the threshold
            if return_xtime:
                return False, None
            else:
                return False
        def training_step(self, batch, batch_idx):
            if args.kill_on_nan_detection and batch_idx % args.check_interval_nan_model_weights == 0:
                if self._check_for_nan_weights():
                    logging.error("NaN or Inf detected in model weights. Stopping training.")
                    raise ValueError("NaN or Inf detected in model weights. Stopping training.")
            
            # batch_obj_x, batch_obj_y = self._make_batch(batch)
            # batch_obj_y = batch_obj_y.normalise(surf_stats =None) # Normalise target batch in the training step
            batch_obj_x, batch_obj_y = batch
            del batch
            
            if args.max_norm_val_before_loss_reweighting != -1.:
                loss_weight = 1.0
                loss_reweight_th = args.max_norm_val_before_loss_reweighting
                x_normalized = batch_obj_x.normalise(surf_stats=None)  # Normalise input batch
                is_above_th = self._is_normalized_input_magnitude_above_threshold(x_normalized=x_normalized, threshold=loss_reweight_th, return_xtime=False)
                if is_above_th: 
                    loss_weight = args.lambda_loss_reweight_max_x_mag_above_th
                del x_normalized

            # Only measure FLOPs once on first batch
            if batch_idx == 0  and not hasattr(self, '_flops_measured'):
                self.measure_model_flops(batch_obj_x)
                self._flops_measured = True

            # Forward pass
            batch_pred, batch_std, batch_ens = self.net.forward(batch_obj_x)
            del batch_obj_x

            # Main task loss
            task_loss, loss_dict = self.loss_fn(batch_pred, batch_std, batch_ens, batch_obj_y)
            del batch_std, batch_ens
            
            # Compute RC loss if enabled
            if hasattr(self.net, 'rc_loss') and self.net.rc_loss:
                rc_loss = losses.RoutingClassificationLoss.compute_loss_from_predictions(
                    self.net,
                    batch_pred,  # predictions
                    batch_obj_y, # targets
                    abstol=1e-3, # absolute tolerance for correctness
                    reltol=0.0   # relative tolerance for correctness
                ).to(self.device)
                # Add RC loss to the loss dict for logging
                loss_dict['smoe/rc_loss'] = rc_loss # logged to smoe namespace
                # Combine losses
                total_loss = task_loss + self.rc_weight * rc_loss
            else:
                total_loss = task_loss
            del batch_pred,batch_obj_y
            
            def aggregate_aux_losses_ignore_nan(net):
                """Gather and sum all auxiliary losses, ignoring any NaN or Inf."""
                aux_loss_sum = None
                for module in net.modules():
                    if hasattr(module, 'aux_losses'):
                        for loss in getattr(module, 'aux_losses').values():
                            if loss is not None and torch.isfinite(loss):
                                if aux_loss_sum is None:
                                    aux_loss_sum = loss
                                else:
                                    aux_loss_sum = aux_loss_sum + loss
                return aux_loss_sum

            
            aux_loss = aggregate_aux_losses_ignore_nan(self.net)
            #print(f'task_loss: {task_loss},aux_loss: {aux_loss}, rc_loss: {rc_loss}')
            if aux_loss is not None:
                loss_dict['smoe/aux_loss'] = aux_loss # logged to smoe namespace
                total_loss = total_loss + self.aux_weight * aux_loss
                
            if args.max_norm_val_before_loss_reweighting != -1.:
                total_loss = total_loss * loss_weight

            # Get system metrics and add to loss_dict 
            system_metrics = self._get_system_metrics()
            loss_dict.update(system_metrics)

                
            train_keys_replicated = {}
            train_keys_replicated['loss_train'] = total_loss
            for key in loss_dict: 
                train_keys_replicated[f'train/{key}'] = loss_dict[key]
            loss_dict.update(train_keys_replicated)

            # Log all losses
            self.log_dict(loss_dict, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
            # if sync and torch.distributed.is_initialized():
                # TODO: Could be deleted
                # torch.distributed.barrier()

            return total_loss

        def validation_step(self, val_batch, batch_idx):
            #batch_obj_x, batch_obj_y = self._make_batch(val_batch) # Don't normalise target batch in the validation step
            batch_obj_x, batch_obj_y = val_batch
            del val_batch
            batch_pred, batch_std, batch_ens = self.net.forward(batch_obj_x)
            del batch_obj_x
            
            if args.log_val_predictions_as_images and batch_idx == 0:
                if not self.is_ybatch_images_logged:
                    ## surface vars:
                    y_srf_ = {f'val/y_surf_{var}': wandb.Image(convert_to_wandb_image(batch_obj_y.surf_vars[var][0, 0,:,:].cpu().numpy()), caption=f'val/y_surf_{var}', file_type="jpg") for var in batch_obj_y.surf_vars.keys()}
                    y_atmos_ = {f'val/y_atmos_{var}': wandb.Image(convert_to_wandb_image(batch_obj_y.atmos_vars[var][0, 0,2,:,:].cpu().numpy()), caption=f'val/y_atmos_{var}', file_type="jpg") for var in batch_obj_y.atmos_vars.keys()}
                    wandb_images_y = {**y_srf_, **y_atmos_}
                    self.logger.experiment.log(
                        wandb_images_y,
                        step=self.global_step,
                    )
                    self.is_ybatch_images_logged = True
                yp_srf_ = {f'val/y_pred_surf_{var}': wandb.Image(convert_to_wandb_image(batch_pred.surf_vars[var][0,0,:,:].cpu().numpy()), caption=f'val/y_pred_surf_{var}', file_type="jpg") for var in batch_obj_y.surf_vars.keys()}
                yp_atmos_ = {f'val/y_pred_atmos_{var}': wandb.Image(convert_to_wandb_image(batch_pred.atmos_vars[var][0,0,2,:,:].cpu().numpy()), caption=f'val/y_pred_atmos_{var}', file_type="jpg") for var in batch_obj_y.atmos_vars.keys()}
                wandb_images_yp = {**yp_srf_, **yp_atmos_}
                self.logger.experiment.log(
                    wandb_images_yp,
                    step=self.global_step,
                )
            
            # Main task loss
            task_loss, loss_dict = self.loss_fn(batch_pred, batch_std, batch_ens, batch_obj_y)
            del batch_std, batch_ens
            
            # Compute RC loss if enabled
            if hasattr(self.net, 'rc_loss') and self.net.rc_loss:
                rc_loss = losses.RoutingClassificationLoss.compute_loss_from_predictions(
                    self.net,
                    batch_pred,  # predictions
                    batch_obj_y, # targets
                    abstol=1e-3, # absolute tolerance for correctness
                    reltol=0.0   # relative tolerance for correctness
                ).to(self.device)
                # Add RC loss to the loss dict for logging
                loss_dict['rc_loss'] = rc_loss
                # Combine losses
                total_loss = task_loss + self.rc_weight * rc_loss
            else:
                total_loss = task_loss

            del batch_pred,batch_obj_y
            
            def aggregate_aux_losses_ignore_nan(net):
                """Gather and sum all auxiliary losses, ignoring any NaN or Inf."""
                aux_loss_sum = None
                for module in net.modules():
                    if hasattr(module, 'aux_losses'):
                        for loss in getattr(module, 'aux_losses').values():
                            if loss is not None and torch.isfinite(loss):
                                if aux_loss_sum is None:
                                    aux_loss_sum = loss
                                else:
                                    aux_loss_sum = aux_loss_sum + loss
                return aux_loss_sum

            
            aux_loss = aggregate_aux_losses_ignore_nan(self.net)
            #print(f'task_loss: {task_loss},aux_loss: {aux_loss}, rc_loss: {rc_loss}')
            if aux_loss is not None:
                loss_dict['aux_loss'] = aux_loss
                total_loss = total_loss + self.aux_weight * aux_loss
        

            # Log metrics
            loss_dict = {f'{key}_val': value for key, value in loss_dict.items()}
            loss_dict['loss_val'] = total_loss
            self.log_dict(loss_dict, batch_size=self.batch_size, sync_dist=True, prog_bar=True)
        
        def on_after_backward(self):
            """Override to check for NaN in gradients if enabled."""
            if args.kill_on_nan_detection:
                # Check for NaN in loss and skip optimizer step if detected.
                nan_or_inf_found = False
                for param in self.parameters():
                    if param.grad is not None:
                        if torch.isnan(param.grad).any():
                            logging.error("NaN detected in gradients! Resetting gradients for step...")
                            nan_or_inf_found = True
                            break
                        if torch.isinf(param.grad).any():
                            logging.error("Inf detected in gradients! Resetting gradients for step...")
                            nan_or_inf_found = True
                            break
                if nan_or_inf_found:
                    raise ValueError("NaN or Inf detected in gradients. Stopping training.")
                # if nan_or_inf_found:
                #     self.zero_grad(set_to_none=True)  # Prevent NaNs from propagating, not sure how it works with FSDP.
                
            super().on_after_backward()  # Call the parent method to ensure any additional behavior is executed
        
        def optimizer_step(
            self,
            epoch=None,
            batch_idx=None,
            optimizer=None,
            optimizer_closure=None,
            optimizer_idx=None,
            on_tpu=None,
            using_native_amp=None,
            using_lbfgs=None,
        ):  
            # Run the closure to get the loss and compute gradients
            if optimizer_closure is not None:
                optimizer_closure()
            
            # Clip gradients using FSDP's clip_grad_norm_: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.fsdp.FullyShardedDataParallel.clip_grad_norm_
            if hasattr(self, 'net'):
                # Check if we're using FSDP strategy
                using_fsdp = isinstance(self.trainer.strategy, FSDPStrategy)
                if using_fsdp:
                    #print("using fsdp clip...")
                    # Get the FSDP wrapper from the strategy
                    fsdp_wrapper = self.trainer.strategy.model
                    if args.log_norms:
                        pre_clip_norm = fsdp_wrapper.clip_grad_norm_(max_norm=float('inf')) 
                    fsdp_wrapper.clip_grad_norm_(max_norm=args.max_grad_norm)  # Apply clipping
                    
                    if args.log_norms:
                        if int(self.trainer.global_step) % args.log_norm_every_n_steps == 0:
                            
                            # Measure the norm again to confirm it's now ≤ 1.0
                            with torch.no_grad():
                                post_clip_norm = torch.norm(torch.stack([
                                    torch.norm(p.grad.detach(), 2) 
                                    for p in fsdp_wrapper.parameters() 
                                    if p.grad is not None
                                ]), 2)

                            if is_rank0:
                                grad_metrics = log_gradient_norms(fsdp_wrapper)
                                weight_metrics = log_weight_norms(fsdp_wrapper)
                                self.log('grad_overall/grad_norm_pre_clip', pre_clip_norm)
                                self.log('grad_overall/grad_norm_post_clip', post_clip_norm)
                                self.log_dict(grad_metrics, sync_dist=False)
                                self.log_dict(weight_metrics, sync_dist=False)

                else:
                    # Fallback to regular gradient clipping
                    parameters = [p for p in self.net.parameters() if p.requires_grad and p.grad is not None]
                    if parameters:
                        grad_norms = [
                            torch.norm(p.grad.detach(), 2) 
                            for p in parameters
                        ] if args.log_norms else None
                        pre_clip_norm = torch.norm(torch.stack(grad_norms), 2) if args.log_norms else None
                        torch.nn.utils.clip_grad_norm_(parameters, max_norm=args.max_grad_norm)
                        if args.log_norms:
                            if int(self.trainer.global_step) % args.log_norm_every_n_steps == 0:
                                with torch.no_grad():
                                    post_clip_norm = torch.norm(torch.stack(grad_norms), 2)
                                if is_rank0:
                                    grad_metrics = log_gradient_norms(self.net)
                                    weight_metrics = log_weight_norms(self.net)
                                    if pre_clip_norm is not None:
                                        self.log('grad_overall/grad_norm_pre_clip', pre_clip_norm)
                                    self.log('grad_overall/grad_norm_post_clip', post_clip_norm)
                                    self.log_dict(grad_metrics, sync_dist=False)
                                    self.log_dict(weight_metrics, sync_dist=False)
            
            # Update parameters
            optimizer.step()
            
            # Zero gradients
            optimizer.zero_grad()

    # Initialize LightningModule
    model_lightning = LightningModule(
        net=model, 
        loss_fn=loss_obj.get_loss,  
        example_input_array=None, 
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=5e-6,
        warmup_steps=1000,
        use_constant_lr=args.use_constant_lr,  # Use constant LR after warmup
    )
    
    # Setup WandB
    # wnb_id = f'{args.wnb_project}_{str(args.log_dir).split("/")[-1]}' #_{time.strftime("%Y%m%d-%H%M%S")}'
    # lightning_logs_path = os.path.join(args.log_dir, 'lightning_logs', 'version_1')
    # wandb.tensorboard.patch(root_logdir=lightning_logs_path, pytorch=True)
    # wandb.init(
    #     project=args.wnb_project, 
    #     name=args.wnb_name, 
    #     id=wnb_id, 
    #     config=args, 
    #     save_code=True, 
    #     resume='allow', 
    #     mode=args.wnb_mode
    # )

    ### setup, callbacks,  logging and checkpoints
    class PushProfilingTraceToWB(Callback):
        def __init__(self, lightning_logs_path):
            super().__init__()
            self.lightning_logs_path = lightning_logs_path
            self.profiling_files = glob.glob(os.path.join(lightning_logs_path, f'.*.pt.trace.json'))
        def on_train_epoch_end(self, *args, **kwargs):
            profiling_files = glob.glob(os.path.join(self.lightning_logs_path, f'.*.pt.trace.json'))
            new_files = []
            for file in profiling_files:
                if file not in self.profiling_files:
                    new_files.append(file)
                    self.profiling_files.append(file)
            for f in new_files:
                wandb.save(os.path.basename(f), base_path=self.lightning_logs_path, policy='now')

    class StatefulDataLoaderCallback(Callback):
        """
        PyTorch Lightning callback to save and restore dataloader state during training.

        Args:
            dataloader (DataLoader): The dataloader instance to manage
            checkpoint_dir (str): Directory to save dataloader state checkpoints
        """
        def __init__(self, dataloader: DataLoader, checkpoint_dir: str = './dataloader_checkpoints'):
            super().__init__()
            self.dataloader = dataloader
            self.checkpoint_dir = checkpoint_dir
            os.makedirs(checkpoint_dir, exist_ok=True)

        def _save_dataloader_state(self, trainer, checkpoint_path):
            """Save dataloader state to a file"""
            state_path = os.path.join(self.checkpoint_dir, f"dataloader_state_{trainer.global_step}.pkl")
            with open(state_path, 'wb') as f:
                pickle.dump(self.dataloader.state_dict(), f)
            return state_path

        def on_save_checkpoint(self, trainer, pl_module, checkpoint):
            """Save dataloader state when a checkpoint is saved"""
            # Only save on rank 0 to prevent file conflicts
            if trainer.is_global_zero:
                # Check if the checkpoint is triggered by modelcheckpoint_callback_regular_step_save
                for callback in trainer.callbacks:
                    if isinstance(callback, ModelCheckpoint) and callback == modelcheckpoint_callback_regular_step_save:
                        # Add the dataloader state in the checkpoint
                        checkpoint['dataloader_state'] = self.dataloader.state_dict()
                        break

    ## Define Callbacks
    lr_monitor = LearningRateMonitor(logging_interval='step', log_momentum=True, log_weight_decay=True)
    modelcheckpoint_callback_regular_step_save = ModelCheckpoint(
        dirpath=args.log_dir, 
        filename="model_ckpt-{step}-{loss_train:.2f}", 
        every_n_train_steps=100, 
        save_last=True,
        save_top_k = -1, ## save all ckpts.
    )
    modelcheckpoint_callback_regular_epoch_save = ModelCheckpoint(
        dirpath=args.log_dir, 
        filename="model_ckpt-{epoch}-{loss_train:.2f}", 
        save_on_train_epoch_end=True, 
        save_last=True
    )
    modelcheckpoint_callback_best_val_save = ModelCheckpoint(
        dirpath=args.log_dir, 
        filename="model_best_val_ckpt-{epoch}-{step}-{loss_val:.2f}", 
        monitor="loss_val", 
        save_top_k=3, 
        mode='min', 
        save_on_train_epoch_end=False
    )
    # Best train-loss checkpoint
    modelcheckpoint_callback_best_train_save = ModelCheckpoint(
        dirpath=os.path.join(args.log_dir, "best_train_ckpts"),  # <--- unique folder
        filename="{epoch:02d}-{loss_train:.4f}",
        monitor="loss_train",
        mode="min",
        save_top_k=3,
        save_last=True,
        save_on_train_epoch_end=False,
        every_n_train_steps=10

    )
    # 1) define a small helper to pull batch‐size out of your Batch object:
    def batch_size_fn(batch):
        batch_obj_x, _ = batch
        surf_var = batch_obj_x.surf_vars.get('2t', next(iter(batch_obj_x.surf_vars.values())))
        return surf_var.shape[0]

    # 2) instantiate the ThroughputMonitor:
    throughput_cb = ThroughputMonitor(batch_size_fn=batch_size_fn) # starts to log after "window_size=100" steps
    callbacks = [
        modelcheckpoint_callback_regular_step_save,
        modelcheckpoint_callback_regular_epoch_save, 
        modelcheckpoint_callback_best_val_save,
        # modelcheckpoint_callback_best_train_save,
        # PushProfilingTraceToWB(lightning_logs_path=lightning_logs_path),
        StatefulDataLoaderCallback(dataloader=dataloader_train, checkpoint_dir=args.log_dir),
        lr_monitor,
        # DeviceStatsMonitor(),  # Logs CPU stats
        # throughput_cb,
    ]
    if len(args.data_sources) == 1:
        # ThroughputMonitor not working properly on multi-dataset setting
        callbacks.append(throughput_cb)

    ## Setup Loggers
    # logger = TensorBoardLogger(
    #     save_dir=args.log_dir, 
    #     version=1, 
    #     name='lightning_logs', 
    #     log_graph=True
    # )
    # Alternatively, you can use WandbLogger if preferred
    logger = WandbLogger(
        save_dir=args.log_dir,
        #entity=args.wnb_entity,
        name=args.wnb_name,
        project=args.wnb_project,
        id=args.wnb_id,
        log_model=False, #True to save models on wandb
        save_code=True,
        resume='allow',
        mode=args.wnb_mode,
        config=args
    )

    logger.experiment.save(os.path.abspath("loss_config.yaml"))  # Save the current script to W&B
    print(f"Saved loss_config.yaml to wandb from {os.path.abspath('loss_config.yaml')}")

    ## Setup and Launch Lightning Trainer
    deterministic_trainer = False  # Might make training slower
    check_val_every_n_epoch = 1  # Use val_check_interval if you want to run val every N steps
    total_train_minibatches = int(len(dataloader_train))
    print(f'get_total_gpus returns {get_total_gpus()}. len(dataloader_train): {len(dataloader_train)}')
    if total_train_minibatches >= 900:
        val_check_interval = 900
    else:
        val_check_interval = total_train_minibatches
    if is_rank0:
        print(f'val_check_interval: {val_check_interval}.')
    log_every_n_steps = args.log_every_n_steps  # How often to add logging rows
    max_epochs = args.epochs
    accelerator = 'gpu' if not args.no_gpu else 'cpu'
    devices = args.devices  # Number of GPUs on each node
    num_nodes = args.num_nodes  # Number of nodes. Total GPUs = num_nodes x devices

    ### Strategy
    num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if not hasattr(args, 'strategy'):
        strategy_str = 'full_fsdp'
    else:
        strategy_str = args.strategy.lower()
    if num_gpus > 1:
        if strategy_str == 'full_fsdp':
            # Define FSDPStrategy with Mixed Precision
            if decoder_act_checkpointing:
                activation_ckpt_policy = { Perceiver3DDecoder,}
            else:
                activation_ckpt_policy = None
            fsdp_strategy = FSDPStrategy(
                activation_checkpointing_policy=activation_ckpt_policy,
                cpu_offload=False,
                sharding_strategy="FULL_SHARD", 
                #sharding_strategy="HYBRID_SHARD", 
                backward_prefetch=None,
                use_orig_params=True,
                timeout = timedelta(seconds=6000), # set NCCL timeout to 100 mins
                process_group_backend=args.backend
            )
            strategy = fsdp_strategy
        elif strategy_str == 'ddp':
            
            strategy = DDPStrategy(
                find_unused_parameters=args.ddp_find_unused_parameters,  # Set to True if your model has unused parameters
                process_group_backend=args.backend
            )
        else:
            raise ValueError(f"Unsupported strategy: {strategy_str}. Supported strategies are 'full_fsdp' and 'ddp'.")
    else:
        strategy = 'auto'
    if is_rank0:
        logging.info(f"Strategy: {strategy}, num_gpus: {num_gpus}.")

    # Keep precision as float32 in trainer
    trainer = L.Trainer(
        accelerator=accelerator, 
        devices=devices, 
        num_nodes=num_nodes, 
        strategy=strategy, 
        precision='32-true',  # Keep this as 32-bit
        deterministic=deterministic_trainer, 
        callbacks=callbacks, 
        check_val_every_n_epoch=check_val_every_n_epoch, 
        # val_check_interval = val_check_interval,
        log_every_n_steps=log_every_n_steps, 
        logger=logger, 
        min_epochs=10, 
        max_epochs=max_epochs, 
        profiler="pytorch", 
        enable_progress_bar=True, 
        num_sanity_val_steps=2, 
        #gradient_clip_val=1.0, 
        #gradient_clip_algorithm='value'
        use_distributed_sampler=False,
    )
    logging.info(f"trainer state fn: {trainer.state.fn}, status: {trainer.state.status}.")
        
    trainer.fit(model_lightning, dataloader_train, dataloader_val, ckpt_path=trainer_fit_ckpt_path)

    wandb.finish()
    
    logging.info("Training is completed.")

if __name__ == "__main__":
    main()
