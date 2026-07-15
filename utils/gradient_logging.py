import torch
import numpy as np
import wandb
import logging
import re
import matplotlib.pyplot as plt
import os
from collections import defaultdict
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
import time

def log_gradient_norms(model, save_plot=False, use_fsdp_context=True):
    """
    Log gradient norms for each layer in the model with architectural prefixes.
    Supports both DDP and FSDP models.
    
    Args:
        model: The model (can be wrapped in DDP or FSDP)
        save_plot: Whether to save a plot of backbone gradient norms by depth
        use_fsdp_context: Whether to call this function inside an FSDP.summon_full_params context
    
    Returns:
        dict: Metrics dictionary with gradient norm information
    """
    # Check if we're using FSDP
    is_fsdp_model = isinstance(model, FSDP)
    
    # Warning if using FSDP without context
    if is_fsdp_model and not use_fsdp_context:
        logging.warning("Logging gradient norms on FSDP model without summon_full_params context!")
        logging.warning("You'll only see gradients for parameters sharded to this rank.")
        logging.warning("Wrap this call with: with FSDP.summon_full_params(model, writeback=False):")
    
    # Get the actual model (unwrap DDP or FSDP)
    if hasattr(model, 'module'):
        model_obj = model.module  # For DDP model
    else:
        model_obj = model
    
    # Dictionary to store gradient norms
    grad_norms = {}
    
    # Structure to store backbone depth information
    backbone_grad_by_depth = {
        'encoder': defaultdict(lambda: defaultdict(list)),
        'decoder': defaultdict(lambda: defaultdict(list))
    }
    
    # Function to recursively get layer names and gradient norms
    def get_grad_norms(module, prefix=''):
        for name, p in module.named_parameters():
            if p.grad is not None:
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Compute norm - use separate context for each parameter to reduce memory
                with torch.no_grad():
                    if p.grad.is_cuda:
                        # Process on GPU to avoid expensive transfers
                        norm = torch.norm(p.grad).item()
                    else:
                        norm = torch.norm(p.grad).item()
                
                if not np.isfinite(norm):
                    norm = -1.0  # Mark NaN/Inf as -1
                
                # Determine architectural component and add useful tags
                if "encoder" in full_name:
                    if "encoder_layers" in full_name or "swin" in full_name.lower():
                        arch_prefix = "backbone:encoder:"
                        component = "encoder"
                    else:
                        arch_prefix = "encoder:"
                        component = None
                elif "decoder" in full_name:
                    if "decoder_layers" in full_name or "swin" in full_name.lower():
                        arch_prefix = "backbone:decoder:"
                        component = "decoder"
                    else:
                        arch_prefix = "decoder:"
                        component = None
                elif "backbone" in full_name:
                    arch_prefix = "backbone:"
                    component = None
                elif "expert" in full_name:
                    arch_prefix = "expert:"
                    component = None
                elif "gate" in full_name:
                    arch_prefix = "gate:"
                    component = None
                elif "router" in full_name:
                    arch_prefix = "router:"
                    component = None
                else:
                    arch_prefix = "other:"
                    component = None
                
                # Get layer type
                if "attn" in full_name:
                    if "window_attn" in full_name or "WindowAttention" in full_name:
                        layer_type = "window_attention"
                    elif "axial_attn" in full_name:
                        layer_type = "axial_attention"
                    else:
                        layer_type = "attention"
                elif "mlp" in full_name:
                    layer_type = "mlp"
                elif "norm" in full_name:
                    layer_type = "norm"
                elif "head" in full_name:
                    layer_type = "head"
                elif "embed" in full_name:
                    layer_type = "embed"
                elif "expert" in full_name:
                    layer_type = "expert"
                elif "gate" in full_name:
                    layer_type = "gate"
                elif "router" in full_name:
                    layer_type = "router"
                else:
                    layer_type = "other"
                
                # Create detailed key with architectural prefix
                key_detailed = f"{arch_prefix}{full_name}"
                
                # Store values - only store specific parameter values to save memory
                # For all parameters, we'll just store aggregates
                if norm > 1.0 or "gate" in full_name or "router" in full_name or "expert" in full_name:
                    grad_norms[f"grad_norm/{key_detailed}"] = norm
                
                # Track component aggregates
                component_key = f"grad_norm/{arch_prefix}{layer_type}"
                if component_key not in grad_norms:
                    grad_norms[component_key] = []
                grad_norms[component_key].append(norm)
                
                # Extract layer depth information
                if component in ['encoder', 'decoder']:
                    # Look for patterns like encoder_layers.0.blocks.2 or decoder_layers.1.blocks.3
                    depth_match = re.search(r'_layers\.(\d+)(?:\.blocks\.(\d+))?', full_name)
                    if depth_match:
                        stage = int(depth_match.group(1))
                        block = int(depth_match.group(2)) if depth_match.group(2) else 0
                        
                        # Store for visualization
                        backbone_grad_by_depth[component][stage][block].append(norm)
                        
                        # Also store in regular metrics
                        depth_key = f"grad_norm/{arch_prefix}stage_{stage}/block_{block}/{layer_type}"
                        if depth_key not in grad_norms:
                            grad_norms[depth_key] = []
                        grad_norms[depth_key].append(norm)
        
        # Recursively process child modules
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            get_grad_norms(child, child_prefix)
    
    # Get all gradient norms
    get_grad_norms(model_obj)
    
    # Compute averages for component summaries
    metrics = {}
    for key, value in list(grad_norms.items()):
        if isinstance(value, list):
            # Filter out -1 values (NaN/Inf)
            valid_values = [v for v in value if v >= 0]
            if valid_values:
                #print(key,valid_values)
                metrics[key] = np.mean(valid_values)
            else:
                metrics[key] = -1
        else:
            metrics[key] = value
    
    # Add special gradient statistics for debugging
    large_gradients = [v for v in metrics.values() if isinstance(v, (int, float)) and v > 1.0]
    if large_gradients:
        metrics["grad_norm/large_gradient_count"] = len(large_gradients)
        metrics["grad_norm/large_gradient_mean"] = np.mean(large_gradients)
    
    # Find maximum gradients
    max_grad = max([v for v in metrics.values() if isinstance(v, (int, float)) and v >= 0], default=0)
    metrics["grad_norm/max"] = max_grad
    
    # Create visualization of backbone gradient norms by depth
    if save_plot:
        # Use a timestamp instead of step number for plot filenames
        timestamp = int(time.time())
        plot_backbone_gradients(backbone_grad_by_depth, timestamp, max_grad)
    
    # Keep console logging with a simple message (no step)
    logging.info(f"Max gradient norm: {max_grad:.6f}")
    
    # Log top 5 largest gradients to console
    top_grads = sorted([(k, v) for k, v in metrics.items()], 
                     key=lambda x: x[1], reverse=True)[:5]
    for key, value in top_grads:
        logging.info(f"  {key}: {value:.6f}")
    
    return metrics

def log_weight_norms(model, save_plot=False, use_fsdp_context=True):
    """
    Log parameter weight norms for each layer in the model with architectural prefixes.
    Supports both DDP and FSDP models.
    
    Args:
        model: The model (can be wrapped in DDP or FSDP)
        save_plot: Whether to save a plot of backbone weight norms by depth
        use_fsdp_context: Whether to call this function inside an FSDP.summon_full_params context
    
    Returns:
        dict: Metrics dictionary with weight norm information
    """
    # Check if we're using FSDP
    is_fsdp_model = isinstance(model, FSDP)
    
    # Warning if using FSDP without context
    if is_fsdp_model and not use_fsdp_context:
        logging.warning("Logging weight norms on FSDP model without summon_full_params context!")
        logging.warning("You'll only see weights for parameters sharded to this rank.")
        logging.warning("Wrap this call with: with FSDP.summon_full_params(model, writeback=False):")
    
    # Get the actual model (unwrap DDP or FSDP)
    if hasattr(model, 'module'):
        model_obj = model.module  # For DDP model
    else:
        model_obj = model
    
    # Dictionary to store weight norms
    weight_norms = {}
    
    # Structure to store backbone depth information
    backbone_weight_by_depth = {
        'encoder': defaultdict(lambda: defaultdict(list)),
        'decoder': defaultdict(lambda: defaultdict(list))
    }
    
    # Function to recursively get layer names and weight norms
    def get_weight_norms(module, prefix=''):
        for name, p in module.named_parameters():
            full_name = f"{prefix}.{name}" if prefix else name
            
            # Compute norm - use separate context for each parameter to reduce memory
            with torch.no_grad():
                if p.is_cuda:
                    # Process on GPU to avoid expensive transfers
                    norm = torch.norm(p).item()
                else:
                    norm = torch.norm(p).item()
            
            if not np.isfinite(norm):
                norm = -1.0  # Mark NaN/Inf as -1
            
            # Determine architectural component and add useful tags
            if "encoder" in full_name:
                if "encoder_layers" in full_name or "swin" in full_name.lower():
                    arch_prefix = "backbone:encoder:"
                    component = "encoder"
                else:
                    arch_prefix = "encoder:"
                    component = None
            elif "decoder" in full_name:
                if "decoder_layers" in full_name or "swin" in full_name.lower():
                    arch_prefix = "backbone:decoder:"
                    component = "decoder"
                else:
                    arch_prefix = "decoder:"
                    component = None
            elif "backbone" in full_name:
                arch_prefix = "backbone:"
                component = None
            elif "expert" in full_name:
                arch_prefix = "expert:"
                component = None
            elif "gate" in full_name:
                arch_prefix = "gate:"
                component = None
            elif "router" in full_name:
                arch_prefix = "router:"
                component = None
            else:
                arch_prefix = "other:"
                component = None
            
            # Get layer type
            if "attn" in full_name:
                if "window_attn" in full_name or "WindowAttention" in full_name:
                    layer_type = "window_attention"
                elif "axial_attn" in full_name:
                    layer_type = "axial_attention"
                else:
                    layer_type = "attention"
            elif "mlp" in full_name:
                layer_type = "mlp"
            elif "norm" in full_name:
                layer_type = "norm"
            elif "head" in full_name:
                layer_type = "head"
            elif "embed" in full_name:
                layer_type = "embed"
            elif "expert" in full_name:
                layer_type = "expert"
            elif "gate" in full_name:
                layer_type = "gate"
            elif "router" in full_name:
                layer_type = "router"
            else:
                layer_type = "other"
            
            # Create detailed key with architectural prefix
            key_detailed = f"{arch_prefix}{full_name}"
            
            # Store values - only store specific parameter values to save memory
            # For all parameters, we'll just store aggregates
            if norm > 10.0 or "gate" in full_name or "router" in full_name or "expert" in full_name:
                weight_norms[f"weight_norm/{key_detailed}"] = norm
            
            # Track component aggregates
            component_key = f"weight_norm/{arch_prefix}{layer_type}"
            if component_key not in weight_norms:
                weight_norms[component_key] = []
            weight_norms[component_key].append(norm)
            
            # Extract layer depth information
            if component in ['encoder', 'decoder']:
                # Look for patterns like encoder_layers.0.blocks.2 or decoder_layers.1.blocks.3
                depth_match = re.search(r'_layers\.(\d+)(?:\.blocks\.(\d+))?', full_name)
                if depth_match:
                    stage = int(depth_match.group(1))
                    block = int(depth_match.group(2)) if depth_match.group(2) else 0
                    
                    # Store for visualization
                    backbone_weight_by_depth[component][stage][block].append(norm)
                    
                    # Also store in regular metrics
                    depth_key = f"weight_norm/{arch_prefix}stage_{stage}/block_{block}/{layer_type}"
                    if depth_key not in weight_norms:
                        weight_norms[depth_key] = []
                    weight_norms[depth_key].append(norm)
    
        # Recursively process child modules
        for name, child in module.named_children():
            child_prefix = f"{prefix}.{name}" if prefix else name
            get_weight_norms(child, child_prefix)
    
    # Get all weight norms
    get_weight_norms(model_obj)
    
    # Compute averages for component summaries
    metrics = {}
    for key, value in list(weight_norms.items()):
        if isinstance(value, list):
            # Filter out -1 values (NaN/Inf)
            valid_values = [v for v in value if v >= 0]
            if valid_values:
                metrics[key] = np.mean(valid_values)
            else:
                metrics[key] = -1
        else:
            metrics[key] = value
    
    # Add special weight statistics for debugging
    large_weights = [v for v in metrics.values() if isinstance(v, (int, float)) and v > 10.0]
    if large_weights:
        metrics["weight_norm/large_weight_count"] = len(large_weights)
        metrics["weight_norm/large_weight_mean"] = np.mean(large_weights)
    
    # Find maximum weights
    max_weight = max([v for v in metrics.values() if isinstance(v, (int, float)) and v >= 0], default=0)
    metrics["weight_norm/max"] = max_weight
    
    # Create visualization of backbone weight norms by depth
    if save_plot:
        # Use a timestamp instead of step number for plot filenames
        timestamp = int(time.time())
        plot_backbone_weights(backbone_weight_by_depth, timestamp, max_weight)
    
    # Keep console logging with a simple message (no step)
    logging.info(f"Max weight norm: {max_weight:.6f}")
    
    # Log top 5 largest weights to console
    top_weights = sorted([(k, v) for k, v in metrics.items()], 
                     key=lambda x: x[1], reverse=True)[:5]
    for key, value in top_weights:
        logging.info(f"  {key}: {value:.6f}")
    
    return metrics

def plot_backbone_gradients(backbone_grad_by_depth, step, max_grad_value=None):
    """
    Create visualization of backbone gradient norms by stage and block depth.
    
    Args:
        backbone_grad_by_depth: Dictionary with gradient norms by component, stage, and block
        step: Current training step
        max_grad_value: Maximum gradient value for color scaling
    """
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Calculate average gradient norm for each stage and block
    encoder_data = {}
    decoder_data = {}
    
    for component, stages in backbone_grad_by_depth.items():
        for stage, blocks in stages.items():
            for block, norms in blocks.items():
                # Filter out invalid norms
                valid_norms = [n for n in norms if n >= 0]
                if valid_norms:
                    avg_norm = np.mean(valid_norms)
                    if component == 'encoder':
                        encoder_data[(stage, block)] = avg_norm
                    else:
                        decoder_data[(stage, block)] = avg_norm
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle(f'Backbone Gradient Norms by Depth (Step {step})', fontsize=16)
    
    # Function to plot heatmap for a component
    def plot_component_heatmap(ax, data, title):
        if not data:
            ax.text(0.5, 0.5, "No gradient data available", ha='center', va='center')
            ax.set_title(title)
            return
            
        # Get all stages and blocks
        stages = sorted(set(k[0] for k in data.keys()))
        blocks_per_stage = {}
        for stage in stages:
            blocks_per_stage[stage] = max([k[1] for k in data.keys() if k[0] == stage]) + 1
        
        # Create a 2D grid for the heatmap
        max_blocks = max(blocks_per_stage.values())
        heatmap_data = np.zeros((len(stages), max_blocks))
        
        # Fill in the data
        for (stage, block), norm in data.items():
            heatmap_data[stages.index(stage), block] = norm
        
        # Define color scale
        vmax = max_grad_value if max_grad_value is not None else np.max(heatmap_data)
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
        
        # Add stage and block labels
        ax.set_yticks(range(len(stages)))
        ax.set_yticklabels([f'Stage {s}' for s in stages])
        ax.set_xticks(range(max_blocks))
        ax.set_xticklabels([f'Block {b}' for b in range(max_blocks)])
        
        # Add value annotations
        for i in range(len(stages)):
            for j in range(blocks_per_stage[stages[i]]):
                value = heatmap_data[i, j]
                if value > 0:
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                            color='white' if value > vmax/2 else 'black')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Gradient Norm')
        ax.set_title(title)
    
    # Plot encoder and decoder heatmaps
    plot_component_heatmap(ax1, encoder_data, 'Encoder Gradient Norms by Stage and Block')
    plot_component_heatmap(ax2, decoder_data, 'Decoder Gradient Norms by Stage and Block')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'plots/backbone_grads_step_{step}.png', dpi=100)
    plt.close()
    
    logging.info(f"Saved backbone gradient visualization to plots/backbone_grads_step_{step}.png")

def plot_backbone_weights(backbone_weight_by_depth, step, max_weight_value=None):
    """
    Create visualization of backbone weight norms by stage and block depth.
    
    Args:
        backbone_weight_by_depth: Dictionary with weight norms by component, stage, and block
        step: Current training step
        max_weight_value: Maximum weight value for color scaling
    """
    # Create plots directory if it doesn't exist
    os.makedirs("plots", exist_ok=True)
    
    # Calculate average weight norm for each stage and block
    encoder_data = {}
    decoder_data = {}
    
    for component, stages in backbone_weight_by_depth.items():
        for stage, blocks in stages.items():
            for block, norms in blocks.items():
                # Filter out invalid norms
                valid_norms = [n for n in norms if n >= 0]
                if valid_norms:
                    avg_norm = np.mean(valid_norms)
                    if component == 'encoder':
                        encoder_data[(stage, block)] = avg_norm
                    else:
                        decoder_data[(stage, block)] = avg_norm
    
    # Create the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))
    fig.suptitle(f'Backbone Weight Norms by Depth (Step {step})', fontsize=16)
    
    # Function to plot heatmap for a component
    def plot_component_heatmap(ax, data, title):
        if not data:
            ax.text(0.5, 0.5, "No weight data available", ha='center', va='center')
            ax.set_title(title)
            return
            
        # Get all stages and blocks
        stages = sorted(set(k[0] for k in data.keys()))
        blocks_per_stage = {}
        for stage in stages:
            blocks_per_stage[stage] = max([k[1] for k in data.keys() if k[0] == stage]) + 1
        
        # Create a 2D grid for the heatmap
        max_blocks = max(blocks_per_stage.values())
        heatmap_data = np.zeros((len(stages), max_blocks))
        
        # Fill in the data
        for (stage, block), norm in data.items():
            heatmap_data[stages.index(stage), block] = norm
        
        # Define color scale
        vmax = max_weight_value if max_weight_value is not None else np.max(heatmap_data)
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='viridis', aspect='auto', vmin=0, vmax=vmax)
        
        # Add stage and block labels
        ax.set_yticks(range(len(stages)))
        ax.set_yticklabels([f'Stage {s}' for s in stages])
        ax.set_xticks(range(max_blocks))
        ax.set_xticklabels([f'Block {b}' for b in range(max_blocks)])
        
        # Add value annotations
        for i in range(len(stages)):
            for j in range(blocks_per_stage[stages[i]]):
                value = heatmap_data[i, j]
                if value > 0:
                    ax.text(j, i, f'{value:.2f}', ha='center', va='center', 
                            color='white' if value > vmax/2 else 'black')
        
        # Add colorbar
        plt.colorbar(im, ax=ax, label='Weight Norm')
        ax.set_title(title)
    
    # Plot encoder and decoder heatmaps
    plot_component_heatmap(ax1, encoder_data, 'Encoder Weight Norms by Stage and Block')
    plot_component_heatmap(ax2, decoder_data, 'Decoder Weight Norms by Stage and Block')
    
    # Adjust layout and save
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f'plots/backbone_weights_step_{step}.png', dpi=100)
    plt.close()
    
    logging.info(f"Saved backbone weight visualization to plots/backbone_weights_step_{step}.png")
