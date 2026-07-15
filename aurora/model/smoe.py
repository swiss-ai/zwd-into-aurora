# refer to the implementation: https://github.com/spcl/smoe

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Any, Tuple

##############################################################################
#                               SMoE CONFIG                                  #
##############################################################################

@dataclass
class SMoEConfig:
    """Configuration for SMoE layer."""
    in_planes: int
    out_planes: int
    num_experts: int
    kernel_size: int = 3
    padding: str = 'same'
    gate_type: str = 'conv'
    input_shape: Optional[Tuple[int, int]] = None
    gate_kernel_size: int = 3
    gate_act: Optional[Any] = None

    # Gating / weighting options
    norm_weighted: bool = True
    unweighted: bool = False
    absval_routing: bool = False

    # Noise / regularization
    noise: bool = False
    noise_std: float = 0.1
    noise_std_scale: float = 1.0  # factor to scale noise if desired

    # Routing backprop / error
    block_gate_grad: bool = False
    save_error_signal: bool = True

    # If dampen_expert_error=True, large expert gradients can be scaled.
    dampen_expert_error: bool = False
    dampen_expert_error_factor: float = 0.1
    routing_error_quantile: float = 0.7

    # Auxiliary losses
    importance_weight: float = 0.0
    load_weight: float = 0.0
    spatial_agreement_weight: float = 0.0

    # Additional flags
    smooth_gate_error: bool = False
    gate_mask: Optional[torch.Tensor] = None
    rc_loss: bool = True
    rc_loss_from_loss: bool = False


##############################################################################
#                              SMoERouting                                   #
##############################################################################

class SMoERouting(torch.autograd.Function):
    """
    Custom routing implementation with a straight-through estimator for gating.
    1) Forward: top-k selection of experts for each spatial location
    2) Backward: routes gradients to selected experts and the gating weights
    """

    @staticmethod
    @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
    def forward(ctx, experts: torch.Tensor, routing_weights: torch.Tensor, 
                smoe_config: SMoEConfig, save_module: Any = None):
        # Save config/module for backward
        ctx.smoe_config = smoe_config
        ctx.save_module = save_module

        # 1) top-k along expert dimension
        # topk(...) returns (values, indices)
        vals, indices = routing_weights.topk(k=smoe_config.out_planes, dim=1)
        ctx.indices = indices

        # Save for backward
        ctx.save_for_backward(experts, routing_weights)

        # 2) Create a routing map (all zeros except top-k positions)
        routing_map = torch.zeros_like(routing_weights).scatter_(
            dim=1, index=indices, src=vals
        )
        ctx.mark_non_differentiable(routing_map)
        ctx.mark_non_differentiable(indices)

        # 3) Select or scale experts
        if smoe_config.unweighted:
            # Just gather the top-k experts
            selected = torch.gather(experts, dim=1, index=indices)
        else:
            # Multiply by routing map
            scaled_experts = experts * routing_map
            selected = torch.gather(scaled_experts, dim=1, index=indices)

        return selected, routing_map, indices

    @staticmethod
    @torch.cuda.amp.custom_bwd
    def backward(ctx, grad_selected, grad_routing_map, grad_indices):
        experts, routing_weights = ctx.saved_tensors
        smoe_config: SMoEConfig = ctx.smoe_config

        grad_experts = grad_routing_weights = None

        # Optionally store the error signal
        if ctx.save_module and smoe_config.save_error_signal:
            ctx.save_module.saved_error_signal = grad_selected
            if not smoe_config.unweighted:
                # multiply by selected experts
                selected_experts = torch.gather(experts, dim=1, index=ctx.indices)
                ctx.save_module.saved_error_signal = grad_selected * selected_experts

        # 1) Scatter gradient back to expert dimension
        scattered_grads = torch.zeros_like(experts).scatter_(
            dim=1, index=ctx.indices, src=grad_selected
        )

        # 2) Grad wrt. experts
        if ctx.needs_input_grad[0]:
            if smoe_config.unweighted:
                grad_experts = scattered_grads
            else:
                grad_experts = scattered_grads * routing_weights

        # 3) Grad wrt. routing_weights
        if ctx.needs_input_grad[1] and not smoe_config.block_gate_grad:
            if smoe_config.unweighted:
                grad_routing_weights = scattered_grads
            else:
                grad_routing_weights = scattered_grads * experts
        else:
            grad_routing_weights = None

        # 4) Optionally dampen large gradients (quantile-based approach)
        if (smoe_config.dampen_expert_error and grad_experts is not None and 
            scattered_grads.size(1) > smoe_config.out_planes):

            orig_dtype = grad_selected.dtype
            grad_selected = grad_selected.float()
            # Compute quantile
            incorrect = torch.quantile(
                grad_selected.abs(),
                smoe_config.routing_error_quantile,
                dim=1,
                keepdim=True
            ) # Shape: [B, 1, H, W]
            incorrect = incorrect.to(orig_dtype)
            grad_selected = grad_selected.to(orig_dtype)

            # Mark large gradients
            scaling_factors = torch.ones_like(grad_selected)
            scaling_factors[grad_selected.abs() > incorrect] = smoe_config.dampen_expert_error_factor

            # Scatter that damping back
            damping = torch.ones_like(scattered_grads) # Shape: [B, num_experts, H, W]
            damping.scatter_(dim=1, index=ctx.indices, src=scaling_factors)
            grad_experts *= damping

        return grad_experts, grad_routing_weights, None, None


##############################################################################
#                          BASE GATE: SpatialGate2d                          #
##############################################################################

class SpatialGate2d(nn.Module):
    """
    Base class for gating layers in Spatial Mixture of Experts.

    Handles:
      - Optional mask concat
      - Activation
      - Noise
      - abs routing
      - Softmax
      - Auxiliary losses: importance, load, spatial_agreement

    Children should set self._gate_fn(x) to produce the raw logits.
    """

    def __init__(self, smoe_config: SMoEConfig) -> None:
        super().__init__()
        self.smoe_config = smoe_config

        # If we have a mask, add an input channel
        if smoe_config.gate_mask is not None:
            smoe_config.in_planes += 1

        # Possibly create an activation
        self.gate_act = smoe_config.gate_act() if smoe_config.gate_act else None

        # Noise
        self.noise = smoe_config.noise
        self.noise_std = smoe_config.noise_std_scale * smoe_config.noise_std

        # This will hold a callable for raw logits
        self._gate_fn = None

        # Container for aux losses
        self.aux_losses = {
            'importance_loss': None,
            'load_loss': None,
            'spatial_agreement_loss': None,
        }

    ########################
    #    AUX LOSSES LOGIC  #
    ########################

    @staticmethod
    def importance_loss(routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Encourages experts to have balanced importance.
        Coefficient of variation squared over sum of weights per expert.
        """
        # Sum over batch and spatial dims => we get a per-expert sum
        expert_importance = routing_weights.sum(dim=(0,) + tuple(range(2, routing_weights.ndim)))
        imp_std = expert_importance.std()
        imp_mean = expert_importance.mean().clamp_min(1e-8)
        return (imp_std / imp_mean) ** 2


    def load_loss(self, routing_weights: torch.Tensor, noisy_routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Approximates how often each expert is "selected" if noise is re-sampled.
        Then measures coefficient of variation (squared) for 'load'.
        """
        # Get the smallest weight among top-k
        threshold = noisy_routing_weights.topk(k=self.smoe_config.out_planes, dim=1).values[:, -1, :]
        
        # Subtract to get distance
        distance_to_selection = threshold.unsqueeze(1) - routing_weights

        # 1) Replace NaNs/Infs with finite numbers
        distance_to_selection = torch.nan_to_num(distance_to_selection, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 2) Optionally clamp to a range to avoid extreme magnitude
        #distance_to_selection = torch.clamp(distance_to_selection, min=-1e5, max=1e5)

        # Probability of selection if we re-sample noise
        dist = torch.distributions.normal.Normal(0.0, self.noise_std)
        p = 1.0 - dist.cdf(distance_to_selection)

        # Summation => per-expert load
        load = p.sum(dim=(0,) + tuple(range(2, p.ndim)))
        load_std = load.std()
        load_mean = load.mean().clamp_min(1e-8)
        return (load_std / load_mean) ** 2




    @staticmethod
    def spatial_agreement_loss(routing_weights: torch.Tensor) -> torch.Tensor:
        """
        Encourages gating weights to be consistent across the batch or spatial dimension.
        Minimizes stdev across some dimension. Reference code sums stdev across batch dim.
        """
        # reference approach: expert_spatial_std = routing_weights.std(dim=0)
        expert_spatial_std = routing_weights.std(dim=0)  # => shape [expert, H, W]
        mean_expert_std = expert_spatial_std.mean(dim=tuple(range(1, expert_spatial_std.ndim)))
        return mean_expert_std.sum()

    ########################
    #     FORWARD PASS     #
    ########################

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Gate mask, expected x shape: [B, in_planes, H, W]
        if self.smoe_config.gate_mask is not None:
            gate_mask = self.smoe_config.gate_mask.to(dtype=x.dtype, device=x.device)
            expanded_mask = gate_mask.repeat(x.size(0), 1, 1, 1)  # shape [B, 1, H, W]
            x = torch.cat((x, expanded_mask), dim=1)

        # 2) Raw routing logits
        if self._gate_fn is None:
            raise RuntimeError("Child class must define self._gate_fn.")
        routing_weights = self._gate_fn(x)

        # 3) Activation
        if self.gate_act is not None:
            routing_weights = self.gate_act(routing_weights)

        # 4) Compute pre-noise aux losses
        self.aux_losses['importance_loss'] = None
        self.aux_losses['load_loss'] = None
        self.aux_losses['spatial_agreement_loss'] = None

        if self.training:
            # Importance
            if self.smoe_config.importance_weight > 0.0:
                if self.smoe_config.norm_weighted:
                    routing_softmax = F.softmax(routing_weights, dim=1)
                    imp_loss_val = self.importance_loss(routing_softmax)
                else:
                    imp_loss_val = self.importance_loss(routing_weights)
                self.aux_losses['importance_loss'] = self.smoe_config.importance_weight * imp_loss_val

            # Spatial agreement
            if self.smoe_config.spatial_agreement_weight > 0.0:
                spa_loss_val = self.spatial_agreement_loss(routing_weights)
                self.aux_losses['spatial_agreement_loss'] = self.smoe_config.spatial_agreement_weight * spa_loss_val

        # 5) Noise injection & load loss
        if self.noise and self.training:
            noisy_routing_weights = routing_weights + torch.randn_like(routing_weights) * self.noise_std
            if self.smoe_config.load_weight > 0.0:
                load_loss_val = self.load_loss(routing_weights, noisy_routing_weights)
                self.aux_losses['load_loss'] = self.smoe_config.load_weight * load_loss_val
            routing_weights = noisy_routing_weights

        # 6) Absolute value if requested
        if self.smoe_config.absval_routing:
            routing_weights = routing_weights.abs()

        # 7) Softmax if requested
        if self.smoe_config.norm_weighted:
            routing_weights = F.softmax(routing_weights, dim=1)

        return routing_weights


##############################################################################
#                          CHILD GATE: Conv Gate                             #
##############################################################################

class SpatialConvGate2d(SpatialGate2d):
    """Convolution-based gating function."""

    def __init__(self, smoe_config: SMoEConfig) -> None:
        super().__init__(smoe_config)
        pad_val = smoe_config.gate_kernel_size // 2 if smoe_config.padding == 'same' else 0
        # Convolution to produce [B, num_experts, H, W]
        self.conv = nn.Conv2d(
            smoe_config.in_planes,
            smoe_config.num_experts,
            kernel_size=smoe_config.gate_kernel_size,
            padding=pad_val,
            bias=False
        )
        # Assign the raw gate function
        self._gate_fn = self._forward_gate

    def _forward_gate(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


##############################################################################
#                      CHILD GATE: Latent Tensor Gate                        #
##############################################################################

class SpatialLatentTensorGate2d(SpatialGate2d):
    """
    A gate that uses a learnable latent 2D tensor plus a 1x1 projection 
    of the input features, then multiplies them for final routing weights.
    """

    def __init__(self, smoe_config: SMoEConfig) -> None:
        super().__init__(smoe_config)
        if smoe_config.input_shape is None:
            raise ValueError("input_shape is required for the latent gate type")

        # 1) Latent tensor: shape [1, num_experts, H, W]
        self.latent_tensors = nn.Parameter(
            torch.empty(1, smoe_config.num_experts, *smoe_config.input_shape)
        )

        # 2) 1x1 projection
        self.proj = nn.Conv2d(
            smoe_config.in_planes,
            smoe_config.num_experts,
            kernel_size=1,
            bias=False
        )

        # Initialize
        self.reset_parameters()

        # Define the raw gate function
        self._gate_fn = self._forward_gate

    def reset_parameters(self):
        """
        Initializes latent_tensors in a Kaiming-uniform-like fashion,
        assuming gain=1.0.
        """
        gain = 1.0
        # As seen in the reference snippet:
        # fan = out_planes / num_experts. 
        # For the gate, let's approximate:
        fan = self.smoe_config.out_planes / float(self.smoe_config.num_experts)
        bound = gain * math.sqrt(3.0 / fan)
        nn.init.uniform_(self.latent_tensors, -bound, bound)

        # (Optional) also init self.proj if you like:
        # nn.init.kaiming_uniform_(self.proj.weight, a=math.sqrt(5))

    def _forward_gate(self, x: torch.Tensor) -> torch.Tensor:
        #print(f'debug aux_losses: {self.aux_losses}')
        # x shape: [B, in_planes, H, W]
        batch_size = x.shape[0]

        # 1) Project input => [B, num_experts, H, W]
        proj_features = self.proj(x)

        # 2) Expand latent => [B, num_experts, H, W]
        latent = self.latent_tensors.expand(batch_size, -1, -1, -1)

        # 3) Multiply
        routing_weights = proj_features * latent
        return routing_weights


##############################################################################
#                             SMoELayer                                      #
##############################################################################

class SMoELayer(nn.Module):
    """
    Spatial Mixture-of-Experts Layer:
      - A gating network (conv or latent or others)
      - A set of experts (Conv2d producing num_experts channels)
      - SMoERouting for top-k selection
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_experts: int,
        kernel_size: int = 3,
        gate_type: str = 'conv',
        input_shape: Optional[Tuple[int, int]] = None,
        gate_kernel_size: int = 3,
        gate_act: Optional[Any] = None,
        **kwargs
    ):
        super().__init__()
        self.smoe_config = SMoEConfig(
            in_planes=in_channels,
            out_planes=out_channels,
            num_experts=num_experts,
            kernel_size=kernel_size,
            gate_type=gate_type,
            input_shape=input_shape,
            gate_kernel_size=gate_kernel_size,
            gate_act=gate_act,
            **kwargs
        )

        # Choose gate
        if gate_type == 'latent':
            self.gate = SpatialLatentTensorGate2d(self.smoe_config)
        else:
            self.gate = SpatialConvGate2d(self.smoe_config)

        # Experts
        pad_val = kernel_size // 2 if self.smoe_config.padding == 'same' else 0
        self.experts = nn.Conv2d(
            in_channels,
            num_experts,
            kernel_size=kernel_size,
            padding=pad_val,
            bias=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Gating
        routing_weights = self.gate(x)  # [B, num_experts, H, W]

        # 2) Expert outputs
        expert_outputs = self.experts(x)  # [B, num_experts, H, W]

        # 3) Route
        output, routing_map, routing_indices = SMoERouting.apply(
            expert_outputs, routing_weights, self.smoe_config, self
        )

        # Store for reference
        self.routing_weights = routing_weights
        self.routing_map = routing_map
        self.routing_indices = routing_indices

        # Also store the gate's auxiliary losses so the user can retrieve them
        self.aux_losses = self.gate.aux_losses

        return output


##############################################################################
#                         EXAMPLE USAGE / TEST                               #
##############################################################################

if __name__ == "__main__":
    # Example usage: latent gate
    layer_latent = SMoELayer(
        in_channels=64,
        out_channels=4,
        num_experts=8,
        gate_type='latent',
        input_shape=(32, 32),  # required for latent
        importance_weight=0.1,
        load_weight=0.05,
        spatial_agreement_weight=0.05,
        noise=True,
        noise_std=0.01,
    )

    # Example input
    x = torch.randn(16, 64, 32, 32)

    # Forward
    out_latent = layer_latent(x)
    print("Output shape (latent gate):", out_latent.shape)

    # Retrieve auxiliary losses
    aux_losses = layer_latent.aux_losses
    print("Aux losses:", aux_losses)

    # Combine them with a dummy main task loss
    task_loss = out_latent.mean()
    total_loss = task_loss
    for name, loss_val in aux_losses.items():
        if loss_val is not None:
            total_loss += loss_val

    total_loss.backward()
    print(f"Total loss backprop successful. Final total loss = {total_loss.item():.4f}")
