import argparse

import yaml


def parse_config(config_file):
    with open(config_file) as f:
        config = yaml.safe_load(f)
        yaml_args = argparse.Namespace()
        yaml_args.__dict__.update(config)
    return yaml_args


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--net",
        type=str,
        default="aurora",
        help="Network architecture to use. [default: aurora] Options: [aurora]",
    )
    parser.add_argument("--config", help="Load settings from yaml.")
    parser.add_argument(
        "--dataset_config_path", 
        default='dataset_config.yaml',
        help="Load dataset configs from yaml.")
    parser.add_argument("--no_gpu", action="store_true", default=False, help="Explicitly use CPU [default: uses gpu]")
    parser.add_argument("--num_nodes", type=int, default=1, help="num nodes to train on")
    parser.add_argument("--devices", type=int, default=1, help="num GPU devices on each node to train on")
    parser.add_argument("--fix_seedcudnn", action="store_false", default=True, help="true if fixing cudnn")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--num_workers", type=int, default=1, help="#threads to run for dataloaders")
    parser.add_argument("--backend", type=str, default='nccl', help="Backend for distributed trianing ")

    # parser.add_argument('--category', default=None, help='Which single class to train on [default: None]')
    parser.add_argument("--log_dir", default="checkpoints/", help="Log dir [default: log]")

    parser.add_argument("--data", type=str, default="./data", help="dataset path")
    
    parser.add_argument("--stats_trainingset_name",
        type=str,
        default=None,
        help="Filename of npz file which stores training set statistics.",
    )
    
    parser.add_argument("--epochs", type=int, default=100, help="Epoch to run [default: 200]")
    parser.add_argument(
        "--batch_size", type=int, default=100, help="Batch Size during training [default: 100]"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-4, help="Initial learning rate [default: 5e-4]"
    )
    parser.add_argument("--optimizer", default="adamW", help="adam or sgd [default: adamW]")

    parser.add_argument(
        "--opt_eps", type=float, default=1e-6, help="AdamW epsilon [default: 1e-6]"
    )
    parser.add_argument(
        "--opt_betas",
        type=float,
        nargs=2,                 # expect two floats
        default=(0.9, 0.95),     # default tuple
        metavar=("BETA1", "BETA2"),
        help="AdamW betas (beta1, beta2)."
    )

    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Resume training using --ckpt_name (resolved under --log_dir unless absolute).",
    )
    
    parser.add_argument(
        "--dump_datasampler_indices",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Dump the sampled indices of the dataset sampler to a csv file. [default: False]",
    )

    # load checkpoint for inference
    parser.add_argument('--ckpt_name', type=str, 
                    default="last.ckpt",
                    help='Name of the checkpoint file to load')
    
    parser.add_argument(
        "--list_ckpt_names",
        nargs='+',
        default=None,
        help="List of ckpt files to use for validation. Leave it as None to use all ckpts in the log_dir. [default: None]",
    )
    
    parser.add_argument(
        "--load_aurora_pretrain_weights",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize model weights to Aurora pretrained weights, where applicable. [default: True]",
    )
    
    parser.add_argument(
        "--load_custom_pretrain_weights_str",
        type=str,
        default=None,
        help="Load custom pretrained weights from an absolute path. If None, no custom weights are loaded. [default: None]",
    )
        
    parser.add_argument(
        "--freeze_encoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freezes model weights after loading checkpoint (or pretrained weights). [default: False]",
    )
    
    parser.add_argument(
        "--freeze_backbone",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freezes model weights after loading checkpoint (or pretrained weights). [default: False]",
    )
    
    parser.add_argument(
        "--freeze_decoder",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Freezes model weights after loading checkpoint (or pretrained weights). [default: False]",
    )
    parser.add_argument(
        "--stabilise_level_agg",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Applies additional layer norm to perceiver modules. [default: False]",
    )
    
    parser.add_argument(
        "--act_checkpointing_encoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation checkpointing for encoder. [default: True]",
    )
    
    parser.add_argument(
        "--act_checkpointing_backbone",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation checkpointing for backbone. [default: True]",
    )
    
    parser.add_argument(
        "--act_checkpointing_decoder",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable activation checkpointing for decoder. [default: True]",
    )

    parser.add_argument("--restore_checkpoint", default=None, help="restore_model")

    parser.add_argument("--valid_checkpoint", default=None, help="restore_model")

    parser.add_argument("--report_freq", type=int, default=10, help="report batch frequency")
    parser.add_argument("--wnb_entity", type=str, default="climate_fm", help="W&B project name")
    parser.add_argument("--wnb_project", type=str, default="aurora_era5", help="W&B project name")
    parser.add_argument("--wnb_name", type=str, default="", help="W&B run name")
    parser.add_argument("--wnb_id", type=str, default=None, help="W&B project id")
    parser.add_argument(
        "--wnb_mode", type=str, default="online", help="W&B mode. use online or disabled"
    )
    parser.add_argument("--log_every_n_steps", type=int, default=5, help="log freq for wandb")
    
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="max_grad_norm")
    parser.add_argument("--log_norms", action=argparse.BooleanOptionalAction, default=False, help="Logs gradient and weight norms of model [default: False]",)
    parser.add_argument("--log_norm_every_n_steps", type=int, default=100, help="log freq for weight and gradient norms on wandb")
    parser.add_argument("--log_val_predictions_as_images", action=argparse.BooleanOptionalAction, default=False, help="Logs the first sample in validation step as pictures to W&B [default: False]",)

    ## ensemble args
    parser.add_argument("--num_ensemble", type=int, default=1, help="number of ensembles")
    parser.add_argument("--nll_weight", type=float, default=0.0, help="nll_weight")
    parser.add_argument("--crps_weight", type=float, default=0.0, help="crps_weight")
    parser.add_argument("--kernel_crps_weight", type=float, default=0.0, help="kernel_crps_weight")
    parser.add_argument("--stats_loss_weight", type=float, default=0.0, help="nll_wstats_loss_weighteight")

    ## smoe args 
    parser.add_argument("--use_smoe", action="store_true", default=False, help="Enable SMoE (default: False)")
    parser.add_argument("--num_experts", type=int, default=1, help="number of experts")
    parser.add_argument("--rc_loss", action="store_true", default=False, help="Enable rc_loss (default: False)")
    parser.add_argument("--block_gate_grad", action="store_true", default=False, help="Enable block_gate_grad (default: False)")
    
    ## args for forcing variables
    parser.add_argument(
        "--loss_config_path", 
        default='loss_config.yaml',
        help="Load loss weights for each surface variable from yaml.")
    
    ## args for stable training
    parser.add_argument(
        "--max_norm_val_before_loss_reweighting",
        type=float,
        default=-1.0,
        help="Maximum norm value before loss reweighting. If -1, no reweighting is applied [default: -1.0]",
    )
    parser.add_argument(
        "--lambda_loss_reweight_max_x_mag_above_th",
        type=float,
        default=1e-5,
        help="Lambda value for total loss reweighting based on maximum input var magnitude above threshold defined as max_norm_val_before_loss_reweighting. This is a mitigation means for outliers in input. [default: 1e-5]",
    )

    parser.add_argument("--check_interval_nan_model_weights", type=int, default=100, help="Frequency to check for NaN or Inf in model weights. [default: 100]")
    parser.add_argument(
        "--kill_on_nan_detection",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Kill the process if NaN or Inf is detected in gradients or model weights. [default: False]",
    )
    
    parser.add_argument("--strategy", type=str, default='full_fsdp', help="training strategy.")
    parser.add_argument(
        "--ddp_find_unused_parameters",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable find_unused_parameters in DDP if that's the training strategy. [default: False]",
    )
    
    parser.add_argument(
        "--str_architecture_size",
        type=str,
        default="large",
        choices=["small", "large",],
        help="Size of the architecture. Options: [small ,large] [default: large]", ## consider expanding with tiny, small, base, large, huge for future, in case we want to use them.
    )
    
    parser.add_argument("--note", type=str, default="", help="extra note about the run")
    parser.add_argument(
        "--machine_name",
        type=str,
        default='clariden',
        choices=["bristen", "todi", "switch", "clariden"],
        help="Name of the machine. Options: [bristen, todi, switch, clariden]",
    )
    parser.add_argument(
        "--use_resolution_specific_patch_tokenizers",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable multi-(de-)tokenizers [default: False]",
    )
    
    parser.add_argument(
        "--variable_aggregation",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable variable aggregation [default: False]",
    )
    
    parser.add_argument(
        "--axial_attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable axial attention. This works if variable_aggregation is True [default: True]",
    )
    
    parser.add_argument(
        "--do_not_use_var_specific_bias_in_patch_tokenizer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="A temporary argument to keep using shared bias across patch tokenizers. Will be removed in the future. [default: False]",
    )
    
    parser.add_argument(
        "--disable_flashattention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disables all flash attention implementations and uses native pytorch self-attention instead. [default: False]",
    )

    parser.add_argument(
        "--add_qk_norm_to_swin3d",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Applies additional qk layer norm to swin3d. [default: False]",
    )    

    parser.add_argument(
        "--data_sources",
        nargs='+',
        default=['era5'],
        help="List of data sources to use.",
    )

    parser.add_argument(
        "--use_constant_lr",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use constant learning rate after warmup [default: True]",
    )

    return parser

def parse_args():
    parser = get_parser()

    args = parser.parse_args()

    if args.config:
        print(f"Loading config from {args.config}. Will overwrite any command line arguments with yaml content.")
        yaml_args = parse_config(args.config)
        args.__dict__.update({**args.__dict__, **yaml_args.__dict__})

    return args