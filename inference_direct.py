import os, sys
_repo_root = os.path.dirname(os.path.abspath(__file__))
if _repo_root not in sys.path:
    sys.path.append(_repo_root)
import torch
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd
import json
import yaml
from datetime import datetime, timedelta
import timeit
import argparse

from config import get_parser
from aurora import Aurora, Batch, Metadata
from aurora.model import swin3d as aurora_swin3d_module
from aurora.model.decoder import Perceiver3DDecoder
from utils import dataset
import postprocessing_esfm
# from utils.dataset import read_co2
from aurora.normalisation import load_normalization_stats
from huggingface_hub import hf_hub_download

parser = get_parser()
parser.add_argument("--use_lora", action=argparse.BooleanOptionalAction, default=False, help="Enable or disable LoRA ")
parser.add_argument("--lora_steps", type=int, default=16, help="Number of LoRA adaptation steps")
parser.add_argument("--lora_mode", type=str, default="all", help="LoRA mode of application [Default: 'all']")
parser.add_argument("--name_ckpt", type=str, default="/path/to/checkpoints/last.ckpt", help="Checkpoint filename to load")
parser.add_argument("--start_time_test", type=str, nargs='+', default=["2020-07-25T00:00:00"], help="Start datetime for testing (YYYY-MM-DDTHH)")
parser.add_argument("--Ntest", type=int, default=5, help="Number of test samples to run (= number of lead times for rollout)")
parser.add_argument("--rollout", action=argparse.BooleanOptionalAction, default=False, help="Make one time step predictions or rollouts [Default: False]")
parser.add_argument("--save_baseline", action=argparse.BooleanOptionalAction, default=False, help="Save baseline model predictions [Default: False]")
parser.add_argument("--baseline_ckpt", type=str, default=None, help="Path to the baseline model checkpoint (if None, it will download the pretrained model from Hugging Face Hub)")

parser.add_argument("--lead_time_h", type=int, default=6, help="Number of hours of lead time for the forecasting task")

parser.add_argument("--baseline_data_sources", nargs='+', default=["era5_without_zwd"], help="Data source for the baseline model")
parser.add_argument("--output_prefix", type=str, default="", help="Optional prefix for output zarr filenames to avoid overwriting existing files")


args = parser.parse_args()


with open(os.path.join(_repo_root, "dataset_config.yaml"), 'r') as file:
    yml_file = yaml.safe_load(file)

if len(args.data_sources) == 1:
    data_source = args.data_sources[0]
    surf_vars = yml_file[data_source]['surf_vars']
    static_vars = yml_file[data_source]['static_vars']
    atmos_vars = yml_file[data_source]['atmos_vars']
else:
    raise Exception("You need to implement the variables for multi datasets")

if args.save_baseline:
    if len(args.baseline_data_sources) == 1:
        baseline_data_source = args.baseline_data_sources[0]
        print(f"Using baseline yaml file config vars: {yml_file[baseline_data_source]}")
        baseline_surf_vars = yml_file[baseline_data_source]['surf_vars']
    else:
        raise Exception("You need to implement the variables for multi datasets for the baseline model")

path_save = args.log_dir
if not os.path.exists(path_save):
    os.makedirs(path_save)

output_prefix = f"{args.output_prefix}_" if args.output_prefix else ""
print(f"Saving predictions to {path_save} (prefix: '{args.output_prefix}')")



### DATA
locations, scales = load_normalization_stats(os.path.join(_repo_root, 'aurora/normalization_stats_1979_2021.json'))

d_srf = {k: dataset.d_srf_abr2full[k] for k in surf_vars}
d_static = dict(zip(("lsm", "z", "slt"), ("land_sea_mask", "geopotential_at_surface", "soil_type")))
d_atmos = dict(zip(("z", "u", "v", "t", "q"), ("geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity")))


# for zenith wet delay, we need to see the effects on specific humidity, so we need to save also the atmospheric variables
if 'zwd' in surf_vars or 'precip' in surf_vars:
    # we save only specific humidity for memory efficiency
    with_atmos = True
    atmos_vars_to_save = {"geopotential", "u_component_of_wind", "v_component_of_wind", "temperature", "specific_humidity"}
    d_atmos_to_save = {k: v for k, v in d_atmos.items() if v in atmos_vars_to_save}
else:
    with_atmos = False
    d_atmos_to_save = {}


atmos_levels = np.asarray([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], dtype=np.int32)

tmp1a = xr.open_zarr("/path/to/data/weatherbench2_original", chunks=None)
tmp1b = xr.open_zarr("/path/to/data/weatherbench2_2022_2023.zarr", chunks=None)
tmp2 = xr.open_zarr("/path/to/data/weatherbench2_additionalvariables.zarr", chunks=None, drop_variables=['divergence', 'volumetric_soil_water_layer_4', 'potential_vorticity', 'vorticity'])
# tmp_zwd = xr.open_zarr("/path/to/data/ZWDX/era5/zwd_data.zarr", chunks=None)
ds = xr.merge([tmp1a, tmp2.sel(time=tmp1a.time)])
ds_2022 = xr.merge([tmp1b, tmp2.sel(time=tmp1b.time)])
# ds = xr.merge([tmp1a.sel(time=tmp_zwd.time), tmp_zwd], compat='override', join='override')
del tmp1a, tmp1b, tmp2
# del tmp_zwd
ds = ds.sel(level=atmos_levels)
ds = ds.sel(latitude=ds.latitude.values[:-1])

if 'extended_path' in yml_file[data_source]['conf']:
    dict_ds_extended = dict()
    for k in yml_file[data_source]['conf']['extended_vars']:
        tmp = xr.open_zarr(yml_file[data_source]['conf']['extended_path'][k], chunks=None)[dataset.d_srf_abr2full[k]].sel(latitude=ds.latitude, longitude=ds.longitude)
        # if k == 'tp_mswep':
        #     time = np.array(tmp.time)
        #     time[95678] = time[95678] + np.timedelta64(3, 'h')
        #     tmp = tmp.assign_coords(time=time)
        dict_ds_extended[dataset.d_srf_abr2full[k]] = tmp
else:
    dict_ds_extended = None

if args.save_baseline:
    if 'extended_path' in yml_file[baseline_data_source]['conf']:
        dict_ds_extended_baseline = dict()
        for k in yml_file[baseline_data_source]['conf']['extended_vars']:
            tmp = xr.open_zarr(yml_file[baseline_data_source]['conf']['extended_path'][k], chunks=None)[dataset.d_srf_abr2full[k]].sel(latitude=ds.latitude, longitude=ds.longitude)
            dict_ds_extended_baseline[dataset.d_srf_abr2full[k]] = tmp
    else:
        dict_ds_extended_baseline = None

# da_co2 = read_co2(ds.time.values, ds.latitude, ds.longitude, dataset.d_srf_abr2full['co2'], lead_time_h=args.lead_time_h)
# dict_ds_extended[dataset.d_srf_abr2full['co2']] = da_co2


### MODEL
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
    use_lora=args.use_lora, 
    # lora_steps=16,
    # lora_mode='all',
    autocast=True, # Use AMP (mixed precision to fit to GPU)
    surf_vars=surf_vars,
    static_vars=static_vars,
    atmos_vars=atmos_vars,
    timestep = timedelta(hours=args.lead_time_h),
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
)

checkpoint_path = os.path.join(args.log_dir, args.name_ckpt)

print(f"Loading checkpoint from {checkpoint_path}")

checkpoint = torch.load(checkpoint_path, weights_only=False)

model.load_state_dict({k[4:]: v for k, v in checkpoint['state_dict'].items()})
model.to("cuda")
model.eval()

if args.save_baseline:
    ### BASELINE MODEL 
    d_srf_baseline = {k: dataset.d_srf_abr2full[k] for k in baseline_surf_vars}
    model_baseline = Aurora(
                use_lora=args.use_lora, 
                autocast=True, # Use AMP (mixed precision to fit to GPU)
                surf_vars=baseline_surf_vars,
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
    )

    if args.baseline_ckpt is not None:
        print(f"Loading baseline checkpoint from {args.baseline_ckpt}")

        baseline_checkpoint = torch.load(args.baseline_ckpt, weights_only=False)    
        model_baseline.load_state_dict({k[4:]: v for k, v in baseline_checkpoint['state_dict'].items()})
    else: # Load the pre-trained weights from Hugging Face Hub (for comparisons)
        path_baseline_model = hf_hub_download(repo_id="microsoft/aurora", filename="aurora-0.25-pretrained.ckpt") # float32
        model_baseline.load_checkpoint_local(path_baseline_model, strict=False)
    model_baseline.to("cuda")
    model_baseline.eval()


### INFERENCE
if not args.rollout:
    assert len(args.start_time_test) == 1, "Without rollout, you need to give a single start_time_test"
    start_time_test = datetime.strptime(args.start_time_test[0], '%Y-%m-%dT%H:%M:%S')
    inds_test = pd.date_range(start=start_time_test, freq=f'{args.lead_time_h}h', periods=args.Ntest).values

    print(f"Making predictions for {len(inds_test)} steps starting from {start_time_test} with lead time of {args.lead_time_h} hours and {args.Ntest} total steps.")

    save_every = 100
    comp_time1 = timeit.default_timer()
    mode, append_dim = 'w', None
    for i, t1 in enumerate(inds_test):
        print('I am making prediction on', t1)
        t1 = t1.astype('M8[ms]').astype(datetime)
        target = postprocessing_esfm.make_target_batch(ds=ds, 
                                    times=t1+timedelta(hours=args.lead_time_h), 
                                    d_srf=d_srf, 
                                    d_static=d_static, 
                                    d_atmos=d_atmos,
                                    locations=locations,
                                    scales=scales, 
                                    dict_ds_extended=dict_ds_extended,
                                    lead_time_h=args.lead_time_h,
                                    device='cpu')
        batch_obj_x = postprocessing_esfm.make_input_batch(ds=ds, 
                                    times=[t1-timedelta(hours=args.lead_time_h), t1], 
                                    d_srf=d_srf, 
                                    d_static=d_static, 
                                    d_atmos=d_atmos,
                                    locations=locations,
                                    scales=scales, 
                                    dict_ds_extended=dict_ds_extended,
                                    lead_time_h=args.lead_time_h,
                                    device='cuda')
        
        if args.save_baseline:
            target_baseline = postprocessing_esfm.make_target_batch(ds=ds, 
                                        times=t1+timedelta(hours=args.lead_time_h), 
                                        d_srf=d_srf_baseline, 
                                        d_static=d_static, 
                                        d_atmos=d_atmos,
                                        locations=locations,
                                        scales=scales, 
                                        dict_ds_extended=dict_ds_extended_baseline,
                                        lead_time_h=args.lead_time_h,
                                        device='cpu')
            
            batch_obj_x_baseline = postprocessing_esfm.make_input_batch(ds=ds, 
                                        times=[t1-timedelta(hours=args.lead_time_h), t1], 
                                        d_srf=d_srf_baseline, 
                                        d_static=d_static, 
                                        d_atmos=d_atmos,
                                        locations=locations,
                                        scales=scales, 
                                        dict_ds_extended=dict_ds_extended_baseline,
                                        lead_time_h=args.lead_time_h,
                                        device='cuda')

        with torch.inference_mode():
            pred, pred_std, pred_ens = model(batch_obj_x)
            if args.save_baseline:
                pred_baseline, pred_std_baseline, pred_ens_baseline = model_baseline(batch_obj_x_baseline)

        if i % save_every == 0:
            if i > 0:
                ds_pred.to_zarr(f"{path_save}/{output_prefix}pred_{start_time_test.strftime('%Y-%m-%d')}_{args.Ntest}steps.zarr",
                    mode=mode, append_dim=append_dim, zarr_format=2)
                if args.num_ensemble > 1:
                    # TODO: add ensemble support for baseline model
                    ds_pred_ens.to_zarr(f"{path_save}/{output_prefix}predensemble_{start_time_test.strftime('%Y-%m-%d')}_{args.Ntest}steps.zarr",
                        mode=mode, append_dim=append_dim, zarr_format=2)
                ds_target.to_zarr(f"{path_save}/{output_prefix}target_{start_time_test.strftime('%Y-%m-%d')}_{args.Ntest}steps.zarr",
                    mode=mode, append_dim=append_dim, zarr_format=2)

                if args.save_baseline:
                    ds_pred_baseline.to_zarr(f"{path_save}/{output_prefix}pred_baseline_{start_time_test.strftime('%Y-%m-%d')}_{args.Ntest}steps.zarr",
                    mode=mode, append_dim=append_dim, zarr_format=2)
                
                mode, append_dim = 'a', 'init_time'
            # For non-rollout: t1 is the forecast initialization time, lead_time_h is constant
            ds_target, ds_pred = postprocessing_esfm.batch2xr(batch_obj_x, target, pred, d_srf, d_atmos=d_atmos_to_save, with_atmos=with_atmos, lead_time_h=args.lead_time_h)
            
            # Clip negative precipitation values (precipitation cannot be negative)
            if 'total_precipitation' in ds_pred.data_vars:
                ds_pred['total_precipitation'] = xr.where(ds_pred['total_precipitation'] > 0, ds_pred['total_precipitation'], 0)
            if 'total_precipitation_MSWEP' in ds_pred.data_vars:
                ds_pred['total_precipitation_MSWEP'] = xr.where(ds_pred['total_precipitation_MSWEP'] > 0, ds_pred['total_precipitation_MSWEP'], 0)
            
            if args.save_baseline:
                ds_target_baseline, ds_pred_baseline = postprocessing_esfm.batch2xr(batch_obj_x_baseline, target_baseline, pred_baseline, d_srf_baseline, d_atmos=d_atmos_to_save, with_atmos=with_atmos, lead_time_h=args.lead_time_h)
                # Clip negative values for baseline too
                if 'total_precipitation' in ds_pred_baseline.data_vars:
                    ds_pred_baseline['total_precipitation'] = xr.where(ds_pred_baseline['total_precipitation'] > 0, ds_pred_baseline['total_precipitation'], 0)
                if 'total_precipitation_MSWEP' in ds_pred_baseline.data_vars:
                    ds_pred_baseline['total_precipitation_MSWEP'] = xr.where(ds_pred_baseline['total_precipitation_MSWEP'] > 0, ds_pred_baseline['total_precipitation_MSWEP'], 0)
            if args.num_ensemble > 1:
                # TODO: add ensemble support for baseline model
                ds_pred_ens = postprocessing_esfm.batch2xr_ensemble(batch_obj_x, pred_ens, d_srf, args.num_ensemble, lead_time_h=args.lead_time_h)
                # Clip negative precipitation in ensemble predictions too
                if 'total_precipitation' in ds_pred_ens.data_vars:
                    ds_pred_ens['total_precipitation'] = xr.where(ds_pred_ens['total_precipitation'] > 0, ds_pred_ens['total_precipitation'], 0)
                if 'total_precipitation_MSWEP' in ds_pred_ens.data_vars:
                    ds_pred_ens['total_precipitation_MSWEP'] = xr.where(ds_pred_ens['total_precipitation_MSWEP'] > 0, ds_pred_ens['total_precipitation_MSWEP'], 0)
        else:
            # For non-rollout: t1 is the forecast initialization time, lead_time_h is constant
            tmp_target, tmp_pred = postprocessing_esfm.batch2xr(batch_obj_x, target, pred, d_srf, d_atmos=d_atmos_to_save, with_atmos=with_atmos, lead_time_h=args.lead_time_h)
            
            # Clip negative precipitation values
            if 'total_precipitation' in tmp_pred.data_vars:
                tmp_pred['total_precipitation'] = xr.where(tmp_pred['total_precipitation'] > 0, tmp_pred['total_precipitation'], 0)
            if 'total_precipitation_MSWEP' in tmp_pred.data_vars:
                tmp_pred['total_precipitation_MSWEP'] = xr.where(tmp_pred['total_precipitation_MSWEP'] > 0, tmp_pred['total_precipitation_MSWEP'], 0)
            
            if args.num_ensemble > 1:
                tmp_pred_ens = postprocessing_esfm.batch2xr_ensemble(batch_obj_x, pred_ens, d_srf, args.num_ensemble, lead_time_h=args.lead_time_h)
                # Clip negative precipitation in ensemble
                if 'total_precipitation' in tmp_pred_ens.data_vars:
                    tmp_pred_ens['total_precipitation'] = xr.where(tmp_pred_ens['total_precipitation'] > 0, tmp_pred_ens['total_precipitation'], 0)
                if 'total_precipitation_MSWEP' in tmp_pred_ens.data_vars:
                    tmp_pred_ens['total_precipitation_MSWEP'] = xr.where(tmp_pred_ens['total_precipitation_MSWEP'] > 0, tmp_pred_ens['total_precipitation_MSWEP'], 0)
            ds_target = xr.concat([ds_target, tmp_target], dim='init_time')
            ds_pred = xr.concat([ds_pred, tmp_pred], dim='init_time')

            if args.save_baseline:
                tmp_target_baseline, tmp_pred_baseline = postprocessing_esfm.batch2xr(batch_obj_x_baseline, target_baseline, pred_baseline, d_srf_baseline, d_atmos=d_atmos_to_save, with_atmos=with_atmos, lead_time_h=args.lead_time_h)
                # Clip negative values for baseline
                if 'total_precipitation' in tmp_pred_baseline.data_vars:
                    tmp_pred_baseline['total_precipitation'] = xr.where(tmp_pred_baseline['total_precipitation'] > 0, tmp_pred_baseline['total_precipitation'], 0)
                if 'total_precipitation_MSWEP' in tmp_pred_baseline.data_vars:
                    tmp_pred_baseline['total_precipitation_MSWEP'] = xr.where(tmp_pred_baseline['total_precipitation_MSWEP'] > 0, tmp_pred_baseline['total_precipitation_MSWEP'], 0)
                ds_pred_baseline = xr.concat([ds_pred_baseline, tmp_pred_baseline], dim='init_time')
            if args.num_ensemble > 1:
                #TODO: add ensemble support for baseline model
                ds_pred_ens = xr.concat([ds_pred_ens, tmp_pred_ens], dim='init_time')

    ds_pred.to_zarr(f"{path_save}/{output_prefix}pred_{start_time_test.strftime('%Y-%m-%d')}_{args.Ntest}steps.zarr",
        mode=mode, append_dim=append_dim, zarr_format=2)
    if args.num_ensemble > 1:
        ds_pred_ens.to_zarr(f"{path_save}/{output_prefix}predensemble_{start_time_test.strftime('%Y-%m-%d')}_{args.Ntest}steps.zarr",
            mode=mode, append_dim=append_dim, zarr_format=2)
    ds_target.to_zarr(f"{path_save}/{output_prefix}target_{start_time_test.strftime('%Y-%m-%d')}_{args.Ntest}steps.zarr",
        mode=mode, append_dim=append_dim, zarr_format=2)

    if args.save_baseline:
        ds_pred_baseline.to_zarr(f"{path_save}/{output_prefix}pred_baseline_{start_time_test.strftime('%Y-%m-%d')}_{args.Ntest}steps.zarr",
        mode=mode, append_dim=append_dim, zarr_format=2)

    comp_time2 = timeit.default_timer()
    
    if args.save_baseline:
        print(f"Total inference time (for finetuned and baseline models)= {comp_time2 - comp_time1:.1f}s")
    else:
        print(f"Total inference time (for finetuned model)= {comp_time2 - comp_time1:.1f}s")


else:
    comp_time1 = timeit.default_timer()
    mode, append_dim = 'w', None
    for start_time_test in args.start_time_test:
        start_time_test = datetime.strptime(start_time_test, '%Y-%m-%dT%H:%M:%S')
        inds_test = pd.date_range(start=start_time_test, freq=f'{args.lead_time_h}h', periods=args.Ntest).values

        ds_batch_input = ds.sel(time=[start_time_test-timedelta(hours=args.lead_time_h),
                                start_time_test])
        for var in dict_ds_extended.keys():
            ds_batch_input = ds_batch_input.assign({var: dict_ds_extended[var].sel(time=ds_batch_input.time)})

        if args.save_baseline:
            ds_batch_input_baseline = ds.sel(time=[start_time_test-timedelta(hours=args.lead_time_h),
                                    start_time_test])
            if dict_ds_extended_baseline is not None:
                for var in dict_ds_extended_baseline.keys():
                    ds_batch_input_baseline = ds_batch_input_baseline.assign({var: dict_ds_extended_baseline[var].sel(time=ds_batch_input_baseline.time)})

        save_every = 100
        for i, t1 in enumerate(inds_test):
            print('I am making prediction on', t1)
            t1 = t1.astype('M8[ms]').astype(datetime)
            target = postprocessing_esfm.make_target_batch(ds=ds,
                                        times=t1+timedelta(hours=args.lead_time_h),
                                        d_srf=d_srf,
                                        d_static=d_static,
                                        d_atmos=d_atmos,
                                        locations=locations,
                                        scales=scales,
                                        dict_ds_extended=dict_ds_extended,
                                        lead_time_h=args.lead_time_h,
                                        device='cpu')

            tmp_input = postprocessing_esfm.make_input_batch(ds=ds,
                                        times=[t1-timedelta(hours=args.lead_time_h), t1],
                                        d_srf=d_srf,
                                        d_static=d_static,
                                        d_atmos=d_atmos,
                                        locations=locations,
                                        scales=scales,
                                        dict_ds_extended=dict_ds_extended,
                                        lead_time_h=args.lead_time_h,
                                        device='cuda')

            if args.save_baseline:
                target_baseline = postprocessing_esfm.make_target_batch(ds=ds,
                                            times=t1+timedelta(hours=args.lead_time_h),
                                            d_srf=d_srf_baseline,
                                            d_static=d_static,
                                            d_atmos=d_atmos,
                                            locations=locations,
                                            scales=scales,
                                            dict_ds_extended=dict_ds_extended_baseline,
                                            lead_time_h=args.lead_time_h,
                                            device='cpu')

                tmp_input_baseline = postprocessing_esfm.make_input_batch(ds=ds,
                                            times=[t1-timedelta(hours=args.lead_time_h), t1],
                                            d_srf=d_srf_baseline,
                                            d_static=d_static,
                                            d_atmos=d_atmos,
                                            locations=locations,
                                            scales=scales,
                                            dict_ds_extended=dict_ds_extended_baseline,
                                            lead_time_h=args.lead_time_h,
                                            device='cuda')

            if i == 0: # to initialize, we need inputs from the reference dataset
                batch_obj_x = tmp_input
                if args.save_baseline:
                    batch_obj_x_baseline = tmp_input_baseline
            else: # from the 2nd step, we use predictions as new inputs but need to add forcings
                for k in ['co2', 'ci', 'sst']:
                    if k in tmp_input.surf_vars:
                        batch_obj_x.surf_vars[k] = tmp_input.surf_vars[k]
                    if args.save_baseline and k in tmp_input_baseline.surf_vars:
                        batch_obj_x_baseline.surf_vars[k] = tmp_input_baseline.surf_vars[k]

            with torch.inference_mode():
                pred, pred_std, pred_ens = model(batch_obj_x)
                if args.save_baseline:
                    pred_baseline, pred_std_baseline, pred_ens_baseline = model_baseline(batch_obj_x_baseline)

            current_lead_time = i * args.lead_time_h

            if i == 0:
                # For rollout: init_time is constant (start_time_test), lead_time represents the rollout step
                ds_target, ds_pred = postprocessing_esfm.batch2xr(batch_obj_x, target, pred, d_srf, lead_time_h=current_lead_time, init_time=start_time_test)

                # Clip negative precipitation
                if 'total_precipitation' in ds_pred.data_vars:
                    ds_pred['total_precipitation'] = xr.where(ds_pred['total_precipitation'] > 0, ds_pred['total_precipitation'], 0)
                if 'total_precipitation_MSWEP' in ds_pred.data_vars:
                    ds_pred['total_precipitation_MSWEP'] = xr.where(ds_pred['total_precipitation_MSWEP'] > 0, ds_pred['total_precipitation_MSWEP'], 0)
                if args.num_ensemble > 1:
                    ds_pred_ens = postprocessing_esfm.batch2xr_ensemble(batch_obj_x, pred_ens, d_srf, args.num_ensemble, lead_time_h=current_lead_time, init_time=start_time_test)

                if args.save_baseline:
                    _, ds_pred_baseline = postprocessing_esfm.batch2xr(batch_obj_x_baseline, target_baseline, pred_baseline, d_srf_baseline, lead_time_h=current_lead_time, init_time=start_time_test)
                    if 'total_precipitation' in ds_pred_baseline.data_vars:
                        ds_pred_baseline['total_precipitation'] = xr.where(ds_pred_baseline['total_precipitation'] > 0, ds_pred_baseline['total_precipitation'], 0)
                    if 'total_precipitation_MSWEP' in ds_pred_baseline.data_vars:
                        ds_pred_baseline['total_precipitation_MSWEP'] = xr.where(ds_pred_baseline['total_precipitation_MSWEP'] > 0, ds_pred_baseline['total_precipitation_MSWEP'], 0)

            elif i % save_every == 0 and len(args.start_time_test)==1: # save intermediate only if we are doing a long rollout
                ds_pred.to_zarr(f"{path_save}/{output_prefix}rollout_pred_{args.start_time_test[0][:-3]}_{args.Ntest}steps.zarr",
                    mode=mode, append_dim=append_dim, zarr_format=2)
                if args.num_ensemble > 1:
                    ds_pred_ens.to_zarr(f"{path_save}/{output_prefix}rollout_predensemble_{args.start_time_test[0][:-3]}_{args.Ntest}steps.zarr",
                        mode=mode, append_dim=append_dim, zarr_format=2)
                ds_target.to_zarr(f"{path_save}/{output_prefix}rollout_target_{args.start_time_test[0][:-3]}_{args.Ntest}steps.zarr",
                    mode=mode, append_dim=append_dim, zarr_format=2)
                if args.save_baseline:
                    ds_pred_baseline.to_zarr(f"{path_save}/{output_prefix}rollout_pred_baseline_{args.start_time_test[0][:-3]}_{args.Ntest}steps.zarr",
                        mode=mode, append_dim=append_dim, zarr_format=2)
                mode, append_dim = 'a', 'lead_time'

                ds_target, ds_pred = postprocessing_esfm.batch2xr(batch_obj_x, target, pred, d_srf, lead_time_h=current_lead_time, init_time=start_time_test)
                # Clip negative precipitation
                if 'total_precipitation' in ds_pred.data_vars:
                    ds_pred['total_precipitation'] = xr.where(ds_pred['total_precipitation'] > 0, ds_pred['total_precipitation'], 0)
                if 'total_precipitation_MSWEP' in ds_pred.data_vars:
                    ds_pred['total_precipitation_MSWEP'] = xr.where(ds_pred['total_precipitation_MSWEP'] > 0, ds_pred['total_precipitation_MSWEP'], 0)
                if args.num_ensemble > 1:
                    ds_pred_ens = postprocessing_esfm.batch2xr_ensemble(batch_obj_x, pred_ens, d_srf, args.num_ensemble, lead_time_h=current_lead_time, init_time=start_time_test)

                if args.save_baseline:
                    _, ds_pred_baseline = postprocessing_esfm.batch2xr(batch_obj_x_baseline, target_baseline, pred_baseline, d_srf_baseline, lead_time_h=current_lead_time, init_time=start_time_test)
                    if 'total_precipitation' in ds_pred_baseline.data_vars:
                        ds_pred_baseline['total_precipitation'] = xr.where(ds_pred_baseline['total_precipitation'] > 0, ds_pred_baseline['total_precipitation'], 0)
                    if 'total_precipitation_MSWEP' in ds_pred_baseline.data_vars:
                        ds_pred_baseline['total_precipitation_MSWEP'] = xr.where(ds_pred_baseline['total_precipitation_MSWEP'] > 0, ds_pred_baseline['total_precipitation_MSWEP'], 0)

            else:
                tmp_target, tmp_pred = postprocessing_esfm.batch2xr(batch_obj_x, target, pred, d_srf, lead_time_h=current_lead_time, init_time=start_time_test)
                # Clip negative precipitation
                if 'total_precipitation' in tmp_pred.data_vars:
                    tmp_pred['total_precipitation'] = xr.where(tmp_pred['total_precipitation'] > 0, tmp_pred['total_precipitation'], 0)
                if 'total_precipitation_MSWEP' in tmp_pred.data_vars:
                    tmp_pred['total_precipitation_MSWEP'] = xr.where(tmp_pred['total_precipitation_MSWEP'] > 0, tmp_pred['total_precipitation_MSWEP'], 0)
                if args.num_ensemble > 1:
                    tmp_pred_ens = postprocessing_esfm.batch2xr_ensemble(batch_obj_x, pred_ens, d_srf, args.num_ensemble, lead_time_h=current_lead_time, init_time=start_time_test)
                ds_target = xr.concat([ds_target, tmp_target], dim='lead_time')
                ds_pred = xr.concat([ds_pred, tmp_pred], dim='lead_time')
                if args.num_ensemble > 1:
                    ds_pred_ens = xr.concat([ds_pred_ens, tmp_pred_ens], dim='lead_time')

                if args.save_baseline:
                    _, tmp_pred_baseline = postprocessing_esfm.batch2xr(batch_obj_x_baseline, target_baseline, pred_baseline, d_srf_baseline, lead_time_h=current_lead_time, init_time=start_time_test)
                    if 'total_precipitation' in tmp_pred_baseline.data_vars:
                        tmp_pred_baseline['total_precipitation'] = xr.where(tmp_pred_baseline['total_precipitation'] > 0, tmp_pred_baseline['total_precipitation'], 0)
                    if 'total_precipitation_MSWEP' in tmp_pred_baseline.data_vars:
                        tmp_pred_baseline['total_precipitation_MSWEP'] = xr.where(tmp_pred_baseline['total_precipitation_MSWEP'] > 0, tmp_pred_baseline['total_precipitation_MSWEP'], 0)
                    ds_pred_baseline = xr.concat([ds_pred_baseline, tmp_pred_baseline], dim='lead_time')

            # prepare new inputs for finetuned model
            _, ds_pred_batch = postprocessing_esfm.batch2xr(batch_obj_x, target, pred, d_srf, d_atmos=d_atmos, with_atmos=True, lead_time_h=current_lead_time, init_time=start_time_test)
            # Clip negative precipitation values before using them as input for next step
            if 'total_precipitation' in ds_pred_batch.data_vars:
                ds_pred_batch['total_precipitation'] = xr.where(ds_pred_batch['total_precipitation'] > 0, ds_pred_batch['total_precipitation'], 0)
            if 'total_precipitation_MSWEP' in ds_pred_batch.data_vars:
                ds_pred_batch['total_precipitation_MSWEP'] = xr.where(ds_pred_batch['total_precipitation_MSWEP'] > 0, ds_pred_batch['total_precipitation_MSWEP'], 0)
            # Convert the init_time and lead_time back to time dimension for the next input
            ds_pred_batch_time = ds_pred_batch.squeeze('lead_time').squeeze('init_time').assign_coords(time=t1 + timedelta(hours=args.lead_time_h))
            ds_pred_batch_time = ds_pred_batch_time.drop_vars(['lead_time', 'init_time']).expand_dims('time')
            ds_batch_input = ds_batch_input.drop_vars(['lead_time', 'init_time'], errors='ignore')
            ds_batch_input = xr.merge([ds_batch_input, ds_pred_batch_time], compat='no_conflicts', join='outer') # now time contains t-1, t, t+1
            ds_batch_input = ds_batch_input.sel(time=[t1, t1 + timedelta(hours=args.lead_time_h)])
            batch_obj_x = postprocessing_esfm.make_input_batch(ds=ds_batch_input,
                                            times=ds_batch_input.time.values.astype('M8[ms]').astype(datetime),
                                            d_srf=d_srf,
                                            d_static=d_static,
                                            d_atmos=d_atmos,
                                            locations=locations,
                                            scales=scales,
                                            lead_time_h=args.lead_time_h,
                                            device='cuda')

            # prepare new inputs for baseline model
            if args.save_baseline:
                _, ds_pred_batch_baseline = postprocessing_esfm.batch2xr(batch_obj_x_baseline, target_baseline, pred_baseline, d_srf_baseline, d_atmos=d_atmos, with_atmos=True, lead_time_h=current_lead_time, init_time=start_time_test)
                if 'total_precipitation' in ds_pred_batch_baseline.data_vars:
                    ds_pred_batch_baseline['total_precipitation'] = xr.where(ds_pred_batch_baseline['total_precipitation'] > 0, ds_pred_batch_baseline['total_precipitation'], 0)
                if 'total_precipitation_MSWEP' in ds_pred_batch_baseline.data_vars:
                    ds_pred_batch_baseline['total_precipitation_MSWEP'] = xr.where(ds_pred_batch_baseline['total_precipitation_MSWEP'] > 0, ds_pred_batch_baseline['total_precipitation_MSWEP'], 0)
                ds_pred_batch_baseline_time = ds_pred_batch_baseline.squeeze('lead_time').squeeze('init_time').assign_coords(time=t1 + timedelta(hours=args.lead_time_h))
                ds_pred_batch_baseline_time = ds_pred_batch_baseline_time.drop_vars(['lead_time', 'init_time']).expand_dims('time')
                ds_batch_input_baseline = ds_batch_input_baseline.drop_vars(['lead_time', 'init_time'], errors='ignore')
                ds_batch_input_baseline = xr.merge([ds_batch_input_baseline, ds_pred_batch_baseline_time], compat='no_conflicts', join='outer')
                ds_batch_input_baseline = ds_batch_input_baseline.sel(time=[t1, t1 + timedelta(hours=args.lead_time_h)])
                batch_obj_x_baseline = postprocessing_esfm.make_input_batch(ds=ds_batch_input_baseline,
                                                times=ds_batch_input_baseline.time.values.astype('M8[ms]').astype(datetime),
                                                d_srf=d_srf_baseline,
                                                d_static=d_static,
                                                d_atmos=d_atmos,
                                                locations=locations,
                                                scales=scales,
                                                lead_time_h=args.lead_time_h,
                                                device='cuda')

        if len(args.start_time_test)==1:
            ds_pred.to_zarr(f"{path_save}/{output_prefix}rollout_pred_{args.start_time_test[0][:-3]}_{args.Ntest}steps.zarr",
                mode=mode, append_dim=append_dim, zarr_format=2)
            if args.num_ensemble > 1:
                ds_pred_ens.to_zarr(f"{path_save}/{output_prefix}rollout_predensemble_{args.start_time_test[0][:-3]}_{args.Ntest}steps.zarr",
                    mode=mode, append_dim=append_dim, zarr_format=2)
            ds_target.to_zarr(f"{path_save}/{output_prefix}rollout_target_{args.start_time_test[0][:-3]}_{args.Ntest}steps.zarr",
                mode=mode, append_dim=append_dim, zarr_format=2)
            if args.save_baseline:
                ds_pred_baseline.to_zarr(f"{path_save}/{output_prefix}rollout_pred_baseline_{args.start_time_test[0][:-3]}_{args.Ntest}steps.zarr",
                    mode=mode, append_dim=append_dim, zarr_format=2)
        else:
            ds_pred.to_zarr(f"{path_save}/{output_prefix}rollout_pred_{args.start_time_test[0][:-3]}x{len(args.start_time_test)}_{args.Ntest}steps.zarr",
                mode=mode, append_dim=append_dim, zarr_format=2)
            if args.num_ensemble > 1:
                ds_pred_ens.to_zarr(f"{path_save}/{output_prefix}rollout_predensemble_{args.start_time_test[0][:-3]}x{len(args.start_time_test)}_{args.Ntest}steps.zarr",
                    mode=mode, append_dim=append_dim, zarr_format=2)
            ds_target.to_zarr(f"{path_save}/{output_prefix}rollout_target_{args.start_time_test[0][:-3]}x{len(args.start_time_test)}_{args.Ntest}steps.zarr",
                mode=mode, append_dim=append_dim, zarr_format=2)
            if args.save_baseline:
                ds_pred_baseline.to_zarr(f"{path_save}/{output_prefix}rollout_pred_baseline_{args.start_time_test[0][:-3]}x{len(args.start_time_test)}_{args.Ntest}steps.zarr",
                    mode=mode, append_dim=append_dim, zarr_format=2)

        mode, append_dim = 'a', 'init_time'

    comp_time2 = timeit.default_timer()
    if args.save_baseline:
        print(f"Total inference time (for finetuned and baseline models)= {comp_time2 - comp_time1:.1f}s")
    else:
        print(f"Total inference time (for finetuned model)= {comp_time2 - comp_time1:.1f}s")