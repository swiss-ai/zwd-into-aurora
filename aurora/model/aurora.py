"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

import contextlib
import dataclasses
import warnings
from datetime import timedelta
from functools import partial
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
)

from aurora.batch import Batch
from aurora.model.decoder import Perceiver3DDecoder
from aurora.model.encoder import Perceiver3DEncoder, Perceiver3DEncoderWithVariableAggregation
from aurora.model.lora import LoRAMode
from aurora.model.swin3d import BasicLayer3D, Swin3DTransformerBackbone


__all__ = ["Aurora", "AuroraSmall", "AuroraHighRes"]

class ResolutionSpecificPatchTokenizers:
    def __init__(self, ):
        self.resolutions_str = ['very_coarse', 'coarse', 'medium', 'fine', 'very_fine']
        self.num_resolutions = len(self.resolutions_str)
        self.patch_sizes = [4, 4, 4, 4, 4]
        self.resolutions_degree = [1.5, 0.50, 0.15, 0.05]
        self.d_patchsize = dict(zip(self.resolutions_str, self.patch_sizes))
        if self.num_resolutions != len(self.resolutions_degree) + 1:
            raise ValueError('Number of resolutions and degrees do not match')
        if not all(self.resolutions_degree[i] > self.resolutions_degree[i + 1] for i in range(len(self.resolutions_degree) - 1)):
            raise ValueError('Resolutions are not in decreasing order')
    def get_resolution_str(self, grid_resolution):
        if grid_resolution > self.resolutions_degree[0]: #(1.5deg, inf)
            return self.resolutions_str[0]
        elif grid_resolution > self.resolutions_degree[1]: #(0.5, 1.5]
            return self.resolutions_str[1]
        elif grid_resolution > self.resolutions_degree[2]: #(0.15, 0.5]
            return self.resolutions_str[2]
        elif grid_resolution > self.resolutions_degree[3]:  #(0.05, 0.15]
            return self.resolutions_str[3]
        else: # (0, 0.05]
            return self.resolutions_str[4]
    def get_patch_size(self, grid_resolution):
        return self.patch_sizes[self.resolutions_str.index(self.get_resolution_str(grid_resolution))]
    



class Aurora(torch.nn.Module):
    """The Aurora model.

    Defaults to to the 1.3 B parameter configuration.
    """

    def __init__(
        self,
        surf_vars: tuple[str, ...] = ("2t", "10u", "10v", "msl"),
        static_vars: tuple[str, ...] = ("lsm", "z", "slt"),
        atmos_vars: tuple[str, ...] = ("z", "u", "v", "t", "q"),
        window_size: tuple[int, int, int] = (2, 6, 12),
        encoder_depths: tuple[int, ...] = (6, 10, 8),
        encoder_num_heads: tuple[int, ...] = (8, 16, 32),
        decoder_depths: tuple[int, ...] = (8, 10, 6),
        decoder_num_heads: tuple[int, ...] = (32, 16, 8),
        latent_levels: int = 4,
        patch_size: int = 4,
        embed_dim: int = 512,
        num_heads: int = 16,
        head_dim: int = 32,
        mlp_ratio: float = 4.0,
        drop_path: float = 0.0,
        drop_rate: float = 0.0,
        enc_depth: int = 1,
        dec_depth: int = 1,
        dec_mlp_ratio: float = 2.0,
        perceiver_ln_eps: float = 1e-5,
        max_history_size: int = 2,
        timestep: timedelta = timedelta(hours=6),
        stabilise_level_agg: bool = False,
        use_lora: bool = True,
        lora_steps: int = 40,
        lora_mode: LoRAMode = "single",
        surf_stats: Optional[dict[str, tuple[float, float]]] = None,
        autocast: bool = False,
        num_ensemble: int = 1,  # Number of ensemble members
        rc_loss=False,  # Enable routing classification loss
        save_error_signal=True,  # Enable error signal saving for RC loss
        block_gate_grad=False,  # Allow gate gradients for RC loss
        use_smoe: bool = False,  # use SMoEs
        num_experts: int = 32,   # New parameter for SMoE
        variable_aggregation=False,
        axial_attention=True,
        use_resolution_specific_patch_tokenizers: bool = False,
        do_not_use_var_specific_bias_in_patch_tokenizer: bool = False, # this arg will be removed in the future. Is only here for backwards compatibility for the time being.
        disable_flashattention: bool = False,
        add_qk_norm_to_swin3d: bool = False,
        encoder_activation_checkpointing: bool = True,
        surf_vars_nan_to_zero: tuple[str, ...] = ("sst", "ci"), # NaN content within these vars will be replaced with zero
    ) -> None:
        """Construct an instance of the model.

        Args:
            surf_vars (tuple[str, ...], optional): All surface-level variables supported by the
                model.
            static_vars (tuple[str, ...], optional): All static variables supported by the
                model.
            atmos_vars (tuple[str, ...], optional): All atmospheric variables supported by the
                model.
            window_size (tuple[int, int, int], optional): Vertical height, height, and width of the
                window of the underlying Swin transformer.
            encoder_depths (tuple[int, ...], optional): Number of blocks in each encoder layer.
            encoder_num_heads (tuple[int, ...], optional): Number of attention heads in each encoder
                layer. The dimensionality doubles after every layer. To keep the dimensionality of
                every head constant, you want to double the number of heads after every layer. The
                dimensionality of attention head of the first layer is determined by `embed_dim`
                divided by the value here. For all cases except one, this is equal to `64`.
            decoder_depths (tuple[int, ...], optional): Number of blocks in each decoder layer.
                Generally, you want this to be the reversal of `encoder_depths`.
            decoder_num_heads (tuple[int, ...], optional): Number of attention heads in each decoder
                layer. Generally, you want this to be the reversal of `encoder_num_heads`.
            latent_levels (int, optional): Number of latent pressure levels.
            patch_size (int, optional): Patch size.
            embed_dim (int, optional): Patch embedding dimension.
            num_heads (int, optional): Number of attention heads in the aggregation and
                deaggregation blocks. The dimensionality of these attention heads will be equal to
                `embed_dim` divided by this value.
            mlp_ratio (float, optional): Hidden dim. to embedding dim. ratio for MLPs.
            drop_rate (float, optional): Drop-out rate.
            drop_path (float, optional): Drop-path rate.
            enc_depth (int, optional): Number of Perceiver blocks in the encoder.
            dec_depth (int, optioanl): Number of Perceiver blocks in the decoder.
            dec_mlp_ratio (float, optional): Hidden dim. to embedding dim. ratio for MLPs in the
                decoder. The embedding dimensionality here is different, which is why this is a
                separate parameter.
            perceiver_ln_eps (float, optional): Epsilon in the perceiver layer norm. layers. Used
                to stabilise the model.
            max_history_size (int, optional): Maximum number of history steps. You can load
                checkpoints with a smaller `max_history_size`, but you cannot load checkpoints
                with a larger `max_history_size`.
            timestep (timedelta, optional): Timestep of the model. Defaults to 6 hours.
            stabilise_level_agg (bool, optional): Stabilise the level aggregation by inserting an
                additional layer normalisation. Defaults to `False`.
            use_lora (bool, optional): Use LoRA adaptation.
            lora_steps (int, optional): Use different LoRA adaptation for the first so-many roll-out
                steps.
            lora_mode (str, optional): LoRA mode. `"single"` uses the same LoRA for all roll-out
                steps, and `"all"` uses a different LoRA for every roll-out step. Defaults to
                `"single"`.
            surf_stats (dict[str, tuple[float, float]], optional): For these surface-level
                variables, adjust the normalisation to the given tuple consisting of a new location
                and scale.
            autocast (bool, optional): Use `torch.autocast` to reduce memory usage. Defaults to
                `False`.
            variable_aggregation (bool, optional): specify if we want to use self-attention mechanism
                across variables and them apply variable aggregation
            add_qk_norm_to_swin3d (bool, optional): Add layer norm layers to Q & K in the Swin3D transformer blocks of the backbone.
            surf_vars_nan_to_zero: tuple[str, ...]: Normalise a surface-level variable. NaN content within these vars will be replaced with zero
        """
        super().__init__()
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.patch_size = patch_size
        self.surf_stats = surf_stats or dict()
        self.autocast = autocast
        self.max_history_size = max_history_size
        self.timestep = timestep
        self.rc_loss = rc_loss
        self.axial_attention = axial_attention
        self.variable_aggregation = variable_aggregation
        self.use_resolution_specific_patch_tokenizers = use_resolution_specific_patch_tokenizers
        self.do_not_use_var_specific_bias_in_patch_tokenizer = do_not_use_var_specific_bias_in_patch_tokenizer
        self.disable_flashattention = disable_flashattention
        self.surf_vars_nan_to_zero = surf_vars_nan_to_zero
        if self.use_resolution_specific_patch_tokenizers:
            self.patch_tokenizer_identifier = ResolutionSpecificPatchTokenizers()
        else:
            self.patch_tokenizer_identifier = None
        self.encoder_activation_checkpointing = encoder_activation_checkpointing

        if self.surf_stats:
            warnings.warn(
                f"The normalisation statics for the following surface-level variables are manually "
                f"adjusted: {', '.join(sorted(self.surf_stats.keys()))}. "
                f"Please ensure that this is right!",
                stacklevel=2,
            )

        encoder = Perceiver3DEncoderWithVariableAggregation if variable_aggregation else Perceiver3DEncoder 
        self.encoder = encoder(
            surf_vars=surf_vars,
            static_vars=static_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            drop_rate=drop_rate,
            mlp_ratio=mlp_ratio,
            head_dim=head_dim,
            depth=enc_depth,
            latent_levels=latent_levels,
            max_history_size=max_history_size,
            perceiver_ln_eps=perceiver_ln_eps,
            stabilise_level_agg=stabilise_level_agg,
            axial_attention=axial_attention,
            autocast=autocast,
            patch_tokenizer_identifier=self.patch_tokenizer_identifier,
            do_not_use_var_specific_bias_in_patch_tokenizer=do_not_use_var_specific_bias_in_patch_tokenizer,
            disable_flashattention=disable_flashattention,
            extensive_checkpointing=encoder_activation_checkpointing,
        )

        self.backbone = Swin3DTransformerBackbone(
            window_size=window_size,
            encoder_depths=encoder_depths,
            encoder_num_heads=encoder_num_heads,
            decoder_depths=decoder_depths,
            decoder_num_heads=decoder_num_heads,
            embed_dim=embed_dim,
            mlp_ratio=mlp_ratio,
            drop_path_rate=drop_path,
            drop_rate=drop_rate,
            use_lora=use_lora,
            lora_steps=lora_steps,
            lora_mode=lora_mode,
            disable_flashattention=disable_flashattention,
            adding_qk_norm=add_qk_norm_to_swin3d,
        )

        self.decoder = Perceiver3DDecoder(
            surf_vars=surf_vars,
            atmos_vars=atmos_vars,
            patch_size=patch_size,
            # Concatenation at the backbone end doubles the dim.
            embed_dim=embed_dim * 2,
            head_dim=embed_dim * 2 // num_heads,
            num_heads=num_heads,
            depth=dec_depth,
            # Because of the concatenation, high ratios are expensive.
            # We use a lower ratio here to keep the memory in check.
            mlp_ratio=dec_mlp_ratio,
            perceiver_ln_eps=perceiver_ln_eps,
            num_ensemble=num_ensemble,
            use_smoe=use_smoe,
            num_experts=num_experts,
            rc_loss=rc_loss,
            save_error_signal=save_error_signal,
            block_gate_grad=block_gate_grad,
            patch_tokenizer_identifier=self.patch_tokenizer_identifier,
        )

    def forward(self, batch: Batch) -> Batch:
        """Forward pass.

        Args:
            batch (:class:`Batch`): Batch to run the model on.

        Returns:
            :class:`Batch`: Prediction for the batch.
        """
        # Get the first parameter. We'll derive the data type and device from this parameter.
        p = next(self.parameters())
        batch = batch.type(p.dtype)
        batch = batch.normalise(surf_stats=self.surf_stats)

        if hasattr(batch.metadata, 'lead_time_seconds'):
            timestep_seconds = batch.metadata.lead_time_seconds
            if not isinstance(timestep_seconds, (int, float, torch.Tensor)):
                raise ValueError(f'batch.metadata.lead_time_seconds must be a number. Found {type(timestep_seconds)} instead.')
            if isinstance(timestep_seconds, torch.Tensor):
                timestep_seconds = timestep_seconds.item()
            timestep = timedelta(seconds=timestep_seconds)
        else:
            timestep = self.timestep

        if not self.variable_aggregation:
            for var in self.surf_vars_nan_to_zero:
                if var in batch.surf_vars:
                    batch.surf_vars[var] = torch.nan_to_num(batch.surf_vars[var], nan=0.0)

        if self.use_resolution_specific_patch_tokenizers:
            patch_size = self.patch_tokenizer_identifier.get_patch_size(batch.metadata.grid_resolution)
        else:
            patch_size = self.patch_size
        batch = batch.crop(patch_size=patch_size)
        batch = batch.to(p.device)
        is_global_observation = batch.metadata.is_global_observation

        H, W = batch.spatial_shape
        patch_res = (
            self.encoder.latent_levels,
            H // patch_size,
            W // patch_size,
        )

        # Insert batch and history dimension for static variables.
        B, T = batch.batch_and_history_dims

        batch = dataclasses.replace(
            batch,
            static_vars={k: v[None, None].repeat(B, T, 1, 1) for k, v in batch.static_vars.items()},
        )

        x = self.encoder(
            batch,
            lead_time=timestep,
        )
        with torch.autocast(device_type="cuda",dtype=torch.bfloat16) if self.autocast else contextlib.nullcontext():    
            x = self.backbone(
                x,
                lead_time=timestep,
                patch_res=patch_res,
                rollout_step=batch.metadata.rollout_step,
                is_global_observation=is_global_observation,
            )

        # get all ensemble preds 
        preds = self.decoder(
            x,
            batch,
            lead_time=timestep,
            patch_res=patch_res,
        )



        # Process each prediction in the ensemble
        # First remove batch and history dimension from static variables
        preds = dataclasses.replace(
            preds,
            static_vars={k: v[0, 0] for k, v in batch.static_vars.items()},
        )

        # Insert history dimension in prediction for both surface and atmospheric variables
        preds = dataclasses.replace(
            preds,
            surf_vars={k: v[:, :, None] for k, v in preds.surf_vars.items()},  # [B, E, 1, H, W]
            atmos_vars={k: v[:, :, None] for k, v in preds.atmos_vars.items()}, # [B, E, 1, L, H, W]
        )

        # Unnormalize all ensemble predictions if not in training mode
        preds = preds if self.training else preds.unnormalise(surf_stats=self.surf_stats)
        
        # Compute mean and std across ensemble dimension (dim=1)
        mean_surf_vars = {k: v.mean(dim=1) for k, v in preds.surf_vars.items()}  # [B, 1, H, W]
        # unbiased is set to zero to handel issue with single ensamble
        std_surf_vars = {k: v.std(dim=1, unbiased=False) for k, v in preds.surf_vars.items()}    # [B, 1, H, W]
        mean_atmos_vars = {k: v.mean(dim=1) for k, v in preds.atmos_vars.items()} # [B, 1, L, H, W]
        std_atmos_vars = {k: v.std(dim=1, unbiased=False) for k, v in preds.atmos_vars.items()}   # [B, 1, L, H, W]

        # Create final prediction and std Batch objects
        pred = dataclasses.replace(
            preds,  # Use preds as template
            surf_vars=mean_surf_vars,
            atmos_vars=mean_atmos_vars,
        )

        std = dataclasses.replace(
            preds,  # Use preds as template
            surf_vars=std_surf_vars,
            atmos_vars=std_atmos_vars,
        )

        return pred, std, preds

    def load_checkpoint(self, repo: str, name: str, strict: bool = True) -> None:
        """Load a checkpoint from HuggingFace.

        Args:
            repo (str): Name of the repository of the form `user/repo`.
            name (str): Path to the checkpoint relative to the root of the repository, e.g.
                `checkpoint.cpkt`.
            strict (bool, optional): Error if the model parameters are not exactly equal to the
                parameters in the checkpoint. Defaults to `True`.
        """
        path = hf_hub_download(repo_id=repo, filename=name)
        self.load_checkpoint_local(path, strict=strict)

    @staticmethod
    def orthogonal_perturbation(v, lambda_=0.1):
        """Apply orthogonal perturbation to a weight matrix v."""
        G = torch.randn_like(v)  
        U, _, Vt = torch.linalg.svd(G, full_matrices=False)  # SVD decomposition
        R = Vt[:v.shape[0], :]  # Use right singular vectors to perturb along feature space
        
        return v + lambda_ * R  # Apply perturbation

    def process_checkpoint_for_ensemble(self, checkpoint_dict: dict) -> dict:
        """Process checkpoint dictionary to handle ensemble heads with random perturbations.
        
        Args:
            checkpoint_dict (dict): Original checkpoint state dictionary
            
        Returns:
            dict: Processed checkpoint with duplicated and perturbed weights for ensemble heads
        """
        new_dict = checkpoint_dict.copy()
        
        # Process heads that need to be duplicated for ensemble members
        for k, v in list(checkpoint_dict.items()):
            if k.startswith("decoder.surf_heads.") or k.startswith("decoder.atmos_heads."):
                # Extract the base name (e.g., "decoder.surf_heads.2t")
                base_name = k.rsplit('.', 1)[0]
                param_type = k.rsplit('.', 1)[1]  # "weight" or "bias"
                
                # Remove the original key
                del new_dict[k]
                
                # For each ensemble member
                for i in range(self.decoder.num_ensemble):
                    new_key = f"{base_name}.{i}.{param_type}"
                    if i == 0 or param_type == "bias":
                        # First ensemble member uses original weights
                        new_dict[new_key] = v.clone()
                        print(f"Copying {k} -> {new_key}")
                    else:
                        # Other members get perturbed weights
                        # Add small random noise and orthogonal perturbation
                        random_perturbation = torch.randn_like(v) * 0.01 # small noise
                        v2 = self.orthogonal_perturbation(v, lambda_=0.1)
                        new_dict[new_key] = v2 + random_perturbation
                        print(f"Perturbing {k} -> {new_key}")
                            
        return new_dict


    def load_checkpoint_local(self, path: str, strict: bool = True) -> None:
        """Load a checkpoint directly from a file.

        Args:
            path (str): Path to the checkpoint.
            strict (bool, optional): Error if the model parameters are not exactly equal to the
                parameters in the checkpoint. Defaults to `True`.
        """
        # Assume that all parameters are either on the CPU or on the GPU.
        device = next(self.parameters()).device

        d = torch.load(path, map_location=device, weights_only=True)

        # You can safely ignore all cumbersome processing below. We modified the model after we
        # trained it. The code below manually adapts the checkpoints, so the checkpoints are
        # compatible with the new model.

        # Remove possibly prefix from the keys.
        for k, v in list(d.items()):
            if k.startswith("net."):
                del d[k]
                d[k[4:]] = v

        if self.use_resolution_specific_patch_tokenizers:
            pretained_patch_size = self.patch_tokenizer_identifier.get_patch_size(0.25) # Aurora was trained on 0.25 degree resolution data.
        else:
            pretained_patch_size = self.patch_size
        # Convert the ID-based parametrization to a name-based parametrization.
        if "encoder.surf_token_embeds.weight" in d:
            weight = d["encoder.surf_token_embeds.weight"]
            del d["encoder.surf_token_embeds.weight"]
            bias = d["encoder.surf_token_embeds.bias"]
            if self.variable_aggregation or self.use_resolution_specific_patch_tokenizers:
                del d["encoder.surf_token_embeds.bias"]

            assert weight.shape[1] == 4 + 3
            if self.use_resolution_specific_patch_tokenizers and not self.variable_aggregation:
                for k in self.patch_tokenizer_identifier.resolutions_str:
                    d[f"encoder.surf_token_embeds.{k}.bias"] = bias
            for i, name in enumerate(("2t", "10u", "10v", "msl", "lsm", "z", "slt")):
                if self.use_resolution_specific_patch_tokenizers:
                    for k in self.patch_tokenizer_identifier.resolutions_str:
                        d[f"encoder.surf_token_embeds.{k}.weights.{name}"] = weight[:, [i]]
                        if self.variable_aggregation:
                            d[f"encoder.surf_token_embeds.{k}.bias.{name}"] = bias ##duplicate pretrained bias to all vars separately.
                else:
                    d[f"encoder.surf_token_embeds.weights.{name}"] = weight[:, [i]]
                    if self.variable_aggregation:
                        d[f"encoder.surf_token_embeds.bias.{name}"] = bias ##duplicate pretrained bias to all vars separately.

        if "encoder.atmos_token_embeds.weight" in d:
            weight = d["encoder.atmos_token_embeds.weight"]
            del d["encoder.atmos_token_embeds.weight"]
            bias = d["encoder.atmos_token_embeds.bias"] 
            if self.variable_aggregation or self.use_resolution_specific_patch_tokenizers:
                del d["encoder.atmos_token_embeds.bias"]

            assert weight.shape[1] == 5
            if self.use_resolution_specific_patch_tokenizers and not self.variable_aggregation:
                for k in self.patch_tokenizer_identifier.resolutions_str:
                    d[f"encoder.atmos_token_embeds.{k}.bias"] = bias
            for i, name in enumerate(("z", "u", "v", "t", "q")):
                if self.use_resolution_specific_patch_tokenizers:
                    for k in self.patch_tokenizer_identifier.resolutions_str:
                        d[f"encoder.atmos_token_embeds.{k}.weights.{name}"] = weight[:, [i]]
                        if self.variable_aggregation:
                            d[f"encoder.atmos_token_embeds.{k}.bias.{name}"] = bias
                else:
                    d[f"encoder.atmos_token_embeds.weights.{name}"] = weight[:, [i]]
                    if self.variable_aggregation:
                        d[f"encoder.atmos_token_embeds.bias.{name}"] = bias

        ### Do not be alarmed that the naming logic changes from weights.{name} to {name}.weight. This is how it was in base repo, we follow the same.
        if "decoder.surf_head.weight" in d: ##TODO: need to check this
            weight = d["decoder.surf_head.weight"]
            bias = d["decoder.surf_head.bias"]
            del d["decoder.surf_head.weight"]
            del d["decoder.surf_head.bias"]

            assert weight.shape[0] == 4 * pretained_patch_size**2
            assert bias.shape[0] == 4 * pretained_patch_size**2
            weight = weight.reshape(pretained_patch_size**2, 4, -1)
            bias = bias.reshape(pretained_patch_size**2, 4)

            for i, name in enumerate(("2t", "10u", "10v", "msl")):
                if self.use_resolution_specific_patch_tokenizers:
                    for k in self.patch_tokenizer_identifier.resolutions_str:
                        d[f"decoder.surf_heads.{k}.{name}.weight"] = weight[:, i]
                        d[f"decoder.surf_heads.{k}.{name}.bias"] = bias[:, i]
                else:
                    d[f"decoder.surf_heads.{name}.weight"] = weight[:, i]
                    d[f"decoder.surf_heads.{name}.bias"] = bias[:, i]

        if "decoder.atmos_head.weight" in d:
            weight = d["decoder.atmos_head.weight"]
            bias = d["decoder.atmos_head.bias"]
            del d["decoder.atmos_head.weight"]
            del d["decoder.atmos_head.bias"]

            assert weight.shape[0] == 5 * pretained_patch_size**2
            assert bias.shape[0] == 5 * pretained_patch_size**2
            weight = weight.reshape(pretained_patch_size**2, 5, -1)
            bias = bias.reshape(pretained_patch_size**2, 5)

            for i, name in enumerate(("z", "u", "v", "t", "q")):
                if self.use_resolution_specific_patch_tokenizers:
                    for k in self.patch_tokenizer_identifier.resolutions_str:
                        d[f"decoder.atmos_heads.{k}.{name}.weight"] = weight[:, i]
                        d[f"decoder.atmos_heads.{k}.{name}.bias"] = bias[:, i]
                else:
                    d[f"decoder.atmos_heads.{name}.weight"] = weight[:, i]
                    d[f"decoder.atmos_heads.{name}.bias"] = bias[:, i]

        # Check if the history size is compatible and adjust weights if necessary.
        if self.use_resolution_specific_patch_tokenizers:
            current_history_size = d["encoder.surf_token_embeds.medium.weights.2t"].shape[2]
        else:
            current_history_size = d["encoder.surf_token_embeds.weights.2t"].shape[2]
        if self.max_history_size > current_history_size:
            self.adapt_checkpoint_max_history_size(d)
        elif self.max_history_size < current_history_size:
            raise AssertionError(
                f"Cannot load checkpoint with `max_history_size` {current_history_size} "
                f"into model with `max_history_size` {self.max_history_size}."
            )

        #self.load_state_dict(d, strict=strict)
        # Load state dict and capture results
        # check if the checkpoint is for ensemble. If not, process it so that it will have the first ensemble head as the pretrained aurora weights
        if self.use_resolution_specific_patch_tokenizers:
            if not "decoder.surf_heads.medium.2t.bias.0" in d:
                d = self.process_checkpoint_for_ensemble(d)
        else:
            if not "decoder.surf_heads.2t.bias.0" in d:
                d = self.process_checkpoint_for_ensemble(d)
        load_result = self.load_state_dict(d, strict=strict)

        # Log any missing or unexpected keys
        if load_result.missing_keys:
            print("Missing keys when loading checkpoint:")
            for key in load_result.missing_keys:
                print(f"  {key}")
        
        if load_result.unexpected_keys:
            print("Unexpected keys in checkpoint:")
            for key in load_result.unexpected_keys:
                print(f"  {key}")

    def adapt_checkpoint_max_history_size(self, checkpoint: dict[str, torch.Tensor]) -> None:
        """Adapt a checkpoint with smaller `max_history_size` to a model with a larger
        `max_history_size` than the current model.

        If a checkpoint was trained with a larger `max_history_size` than the current model,
        this function will assert fail to prevent loading the checkpoint. This is to
        prevent loading a checkpoint which will likely cause the checkpoint to degrade is
        performance.

        This implementation copies weights from the checkpoint to the model and fills zeros
        for the new history width dimension. It mutates `checkpoint`.
        """
        for name, weight in list(checkpoint.items()):
            # We only need to adapt the patch embedding in the encoder.
            enc_surf_embedding = name.startswith("encoder.surf_token_embeds.weights.")
            enc_atmos_embedding = name.startswith("encoder.atmos_token_embeds.weights.")
            if enc_surf_embedding or enc_atmos_embedding:
                # This shouldn't get called with current logic but leaving here for future proofing
                # and in cases where its called outside current context.
                if not (weight.shape[2] <= self.max_history_size):
                    raise AssertionError(
                        f"Cannot load checkpoint with `max_history_size` {weight.shape[2]} "
                        f"into model with `max_history_size` {self.max_history_size}."
                    )

                # Initialize the new weight tensor.
                new_weight = torch.zeros(
                    (weight.shape[0], 1, self.max_history_size, weight.shape[3], weight.shape[4]),
                    device=weight.device,
                    dtype=weight.dtype,
                )
                # Copy the existing weights to the new tensor by duplicating the histories provided
                # into any new history dimensions. The rest remains at zero.
                new_weight[:, :, : weight.shape[2]] = weight

                checkpoint[name] = new_weight

    def configure_activation_checkpointing(
        self,
        module_names: tuple[str, ...] = (
            "Basic3DDecoderLayer",
            "Basic3DEncoderLayer",
            "Perceiver3DDecoder",
            "Perceiver3DEncoder",
            "Swin3DTransformerBackbone",
            "Swin3DTransformerBlock",
        ),
    ) -> None:
        """Configure activation checkpointing.

        This is required in order to compute gradients without running out of memory.

        Args:
            module_names (tuple[str, ...], optional): Names of the modules to checkpoint
                on.

        Raises:
            RuntimeError: If any module specifies in `module_names` was not found and
                thus could not be checkpointed.
        """

        found: set[str] = set()

        def check(x: torch.nn.Module) -> bool:
            name = x.__class__.__name__
            if name in module_names:
                found.add(name)
                return True
            else:
                return False

        apply_activation_checkpointing(self, check_fn=check)

        if found != set(module_names):
            raise RuntimeError(
                f'Could not checkpoint on the following modules: '
                f'{", ".join(sorted(set(module_names) - found))}.'
            )


AuroraSmall = partial(
    Aurora,
    encoder_depths=(2, 6, 2),
    encoder_num_heads=(4, 8, 16),
    decoder_depths=(2, 6, 2),
    decoder_num_heads=(16, 8, 4),
    embed_dim=256,
    num_heads=8,
    use_lora=False,
)

AuroraHighRes = partial(
    Aurora,
    patch_size=10,
    encoder_depths=(6, 8, 8),
    decoder_depths=(8, 8, 6),
)
