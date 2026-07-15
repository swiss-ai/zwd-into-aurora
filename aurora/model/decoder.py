"""Copyright (c) Microsoft Corporation. Licensed under the MIT license."""

from datetime import timedelta

import torch
from einops import rearrange
from torch import nn

from aurora.batch import Batch, Metadata
from aurora.model.fourier import levels_expansion
from aurora.model.perceiver import PerceiverResampler
from aurora.model.util import (
    check_lat_lon_dtype,
    init_weights,
    unpatchify,
)
from aurora.model.smoe import SMoELayer

__all__ = ["Perceiver3DDecoder"]


class Perceiver3DDecoder(nn.Module):
    """Multi-scale multi-source multi-variable decoder based on the Perceiver architecture."""

    def __init__(
        self,
        surf_vars: tuple[str, ...],
        atmos_vars: tuple[str, ...],
        patch_size: int = 4,
        embed_dim: int = 1024,
        depth: int = 1,
        head_dim: int = 64,
        num_heads: int = 8,
        mlp_ratio: float = 4.0,
        drop_rate: float = 0.0,
        perceiver_ln_eps: float = 1e-5,
        num_ensemble: int = 1,  # Number of ensemble members
        use_smoe: bool = False,  # New parameter
        num_experts: int = 32,   # New parameter for SMoE
        rc_loss=False,  # Enable routing classification loss
        save_error_signal=True,  # Enable error signal saving for RC loss
        block_gate_grad=False,  # Allow gate gradients for RC loss
        patch_tokenizer_identifier=None, 
        disable_flashattention: bool = False,
    ) -> None:
        """Initialise.

        Args:
            surf_vars (tuple[str, ...]): All supported surface-level variables.
            atmos_vars (tuple[str, ...]): All supported atmospheric variables.
            patch_size (int, optional): Patch size. Defaults to `4`.
            embed_dim (int, optional): Embedding dim.. Defaults to `1024`.
            depth (int, optional): Number of Perceiver cross-attention and feed-forward blocks.
                Defaults to `1`.
            head_dim (int, optional): Dimension of the attention heads used in the aggregation
                blocks. Defaults to `64`.
            num_heads (int, optional): Number of attention heads used in the aggregation blocks.
                Defaults to `8`.
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding dimensionality.
                Defaults to `4.0`.
            drop_rate (float, optional): Drop-out rate for input patches. Defaults to `0.0`.
            perceiver_ln_eps (float, optional): Layer norm. epsilon for the Perceiver blocks.
                Defaults to `1e-5`.
        """
        super().__init__()

        self.patch_size = patch_size
        self.surf_vars = surf_vars
        self.atmos_vars = atmos_vars
        self.embed_dim = embed_dim
        self.num_ensemble = num_ensemble
        self.patch_tokenizer_identifier = patch_tokenizer_identifier
        self.use_smoe = use_smoe

        self.level_decoder = PerceiverResampler(
            latent_dim=embed_dim,
            context_dim=embed_dim,
            depth=depth,
            head_dim=head_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            drop=drop_rate,
            residual_latent=True,
            ln_eps=perceiver_ln_eps,
            disable_flashattention=disable_flashattention,
        )

        # # Create ensemble of heads for each variable
        # self.surf_heads = nn.ModuleDict()
        # self.atmos_heads = nn.ModuleDict()
        
        # for name in surf_vars:
        #     if use_smoe:
        #         self.surf_heads[name] = nn.ModuleList([
        #             SMoELayer(
        #                 in_channels=embed_dim,
        #                 out_channels=patch_size**2,
        #                 num_experts=num_experts,
        #                 kernel_size=1,  # Use 1x1 conv to mimic linear layer
        #                 norm_weighted=True,
        #                 rc_loss=True,  # Enable routing classification loss
        #                 save_error_signal=True,  # Enable error signal saving for RC loss
        #                 block_gate_grad=False,  # Allow gate gradients for RC loss 
        #                 rc_loss_from_loss=False
        #             ) for _ in range(num_ensemble)
        #         ])
        #     else:
        #         self.surf_heads[name] = nn.ModuleList([
        #             nn.Linear(embed_dim, patch_size**2) 
        #             for _ in range(num_ensemble)
        #         ])

        # for name in atmos_vars:
        #     if use_smoe:
        #         self.atmos_heads[name] = nn.ModuleList([
        #             SMoELayer(
        #                 in_channels=embed_dim,
        #                 out_channels=patch_size**2,
        #                 num_experts=num_experts,
        #                 kernel_size=1,  # Use 1x1 conv to mimic linear layer
        #                 norm_weighted=True,
        #                 rc_loss=True,  # Enable routing classification loss
        #                 save_error_signal=True,  # Enable error signal saving for RC loss
        #                 block_gate_grad=False,  # Allow gate gradients for RC loss
        #                 rc_loss_from_loss=False
        #             ) for _ in range(num_ensemble)
        #         ])
        #     else:
        #         self.atmos_heads[name] = nn.ModuleList([
        #             nn.Linear(embed_dim, patch_size**2)
        #             for _ in range(num_ensemble)
        #         ])


        # create smoe layers, one for each variable, for surface and atmospheric variables
        self.surf_smoe = nn.ModuleDict()
        self.atmos_smoe = nn.ModuleDict()

        for name in surf_vars:
            if use_smoe:
                self.surf_smoe[name] = SMoELayer(
                        in_channels=1,
                        out_channels=1,
                        num_experts=num_experts,
                        kernel_size=3,  # default to 3x3 conv
                        norm_weighted=True,
                        rc_loss=True,  # Enable routing classification loss
                        save_error_signal=True,  # Enable error signal saving for RC loss
                        block_gate_grad=False,  # Allow gate gradients for RC loss 
                        rc_loss_from_loss=False,
                        gate_type='latent',
                        input_shape=(720, 1440), # this should be updated
                        importance_weight=0.0,
                        load_weight=0.0,
                        spatial_agreement_weight=0.0,
                        noise=False,
                        noise_std=0.1,   
                    ) 
            else:
                self.surf_smoe[name] = nn.Identity()
        
        for name in atmos_vars:
            if use_smoe:
                self.atmos_smoe[name] = SMoELayer(
                        in_channels=1,
                        out_channels=1,
                        num_experts=num_experts,
                        kernel_size=3,  # default to 3x3 conv
                        norm_weighted=True,
                        rc_loss=True,  # Enable routing classification loss
                        save_error_signal=True,  # Enable error signal saving for RC loss
                        block_gate_grad=False,  # Allow gate gradients for RC loss
                        rc_loss_from_loss=False,
                        gate_type='latent',
                        input_shape=(720, 1440), # this should be updated
                        importance_weight=0.0,
                        load_weight=0.0,
                        spatial_agreement_weight=0.0,
                        noise=False,
                        noise_std=0.1, 
                    ) 
            else:
                self.atmos_smoe[name] = nn.Identity()
          
        # Create ensemble of heads for each variable
        self.surf_heads = nn.ModuleDict()
        self.atmos_heads = nn.ModuleDict()
        
        if self.patch_tokenizer_identifier is None:
            for name in surf_vars:
                self.surf_heads[name] = nn.ModuleList([
                    nn.Linear(embed_dim, patch_size**2) 
                    for _ in range(num_ensemble)
                ])

            for name in atmos_vars:
                self.atmos_heads[name] = nn.ModuleList([
                    nn.Linear(embed_dim, patch_size**2)
                    for _ in range(num_ensemble)
                ])
        else:
            # Create heads for each grid size input
            for key, patch_size in self.patch_tokenizer_identifier.d_patchsize.items():
                self.surf_heads[key] = nn.ModuleDict()
                self.atmos_heads[key] = nn.ModuleDict()

                for name in surf_vars:
                    self.surf_heads[key][name] = nn.ModuleList([
                        nn.Linear(embed_dim, patch_size**2) 
                        for _ in range(num_ensemble)
                    ])

                for name in atmos_vars:
                    self.atmos_heads[key][name] = nn.ModuleList([
                        nn.Linear(embed_dim, patch_size**2)
                        for _ in range(num_ensemble)
                    ])


        self.atmos_levels_embed = nn.Linear(embed_dim, embed_dim)
        self.apply(init_weights)

    def deaggregate_levels(self, level_embed: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Deaggregate pressure level information.

        Args:
            level_embed (torch.Tensor): Level embedding of shape `(B, L, C, D)`.
            x (torch.Tensor): Aggregated input of shape `(B, L, C', D)`.

        Returns:
            torch.Tensor: Deaggregate output of shape `(B, L, C, D)`.
        """
        B, L, C, D = level_embed.shape
        level_embed = level_embed.flatten(0, 1)  # (BxL, C, D)
        x = x.flatten(0, 1)  # (BxL, C', D)
        _msg = f"Batch size mismatch. Found {level_embed.size(0)} and {x.size(0)}."
        assert level_embed.size(0) == x.size(0), _msg
        assert len(level_embed.shape) == 3, f"Expected 3 dims, found {level_embed.dims()}."
        assert x.dim() == 3, f"Expected 3 dims, found {x.dim()}."

        x = self.level_decoder(level_embed, x)  # (BxL, C, D)
        x = x.reshape(B, L, C, D)
        return x

    def forward(
        self,
        x: torch.Tensor,
        batch: Batch,
        patch_res: tuple[int, int, int],
        lead_time: timedelta,
    ) -> tuple[Batch, Batch, Batch]:  # Returns (mean_batch, std_batch, all_preds_batch)
        surf_vars = tuple(batch.surf_vars.keys())
        atmos_vars = tuple(batch.atmos_vars.keys())
        atmos_levels = batch.metadata.atmos_levels

        B, L, D = x.shape
        lat, lon = batch.metadata.lat, batch.metadata.lon
        dataset_name = batch.metadata.dataset_name
        check_lat_lon_dtype(lat, lon)
        lat, lon = lat.to(dtype=torch.float32), lon.to(dtype=torch.float32)
        H, W = lat.shape[0], lon.shape[-1]
        
        if self.patch_tokenizer_identifier is None:
            patch_size = self.patch_size
            grid_resolution_str = None
            surf_heads = self.surf_heads
            atmos_heads = self.atmos_heads
        else:
            patch_size = self.patch_tokenizer_identifier.get_patch_size(batch.metadata.grid_resolution)
            grid_resolution_str = self.patch_tokenizer_identifier.get_resolution_str(batch.metadata.grid_resolution)
            surf_heads = self.surf_heads[grid_resolution_str]
            atmos_heads = self.atmos_heads[grid_resolution_str]
        # Unwrap the latent level dimension.
        x = rearrange(
            x,
            "B (C H W) D -> B (H W) C D",
            C=patch_res[0],
            H=patch_res[1],
            W=patch_res[2],
        )

        if len(surf_vars) > 0:
            # Run ensemble predictions for surface variables
            surf_preds_ensemble = []
            for i in range(self.num_ensemble):
                if isinstance(surf_heads[list(surf_vars)[0]][0], SMoELayer):
                    # Reshape for SMoE (expects 4D input: B*var_dim, D, H, W)
                    x_surf = x[..., :1, :].squeeze(2)  # Remove the var_dim of 1
                    x_surf = x_surf.permute(0, 2, 1)  # [B, D, H*W]
                    x_surf = x_surf.reshape(B, self.embed_dim, patch_res[1], patch_res[2])  # [B, D, H, W]

                    x_surf = torch.stack([
                        surf_heads[name][i](x_surf)  # [B, patch_size**2, H, W]
                        .permute(0, 2, 3, 1)  # [B, H, W, patch_size**2]
                        .reshape(B, patch_res[1]*patch_res[2], 1, -1)  # [B, H*W, 1, patch_size**2]
                        for name in surf_vars
                    ], dim=-1)  # Final: [B, H*W, 1, patch_size**2, num_vars]
                else:
                    x_surf = torch.stack([
                        surf_heads[name][i](x[..., :1, :]) 
                        for name in surf_vars
                    ], dim=-1)
                    #x_surf: torch.Size([1, 64800, 1, 16, 4])
                x_surf = x_surf.reshape(*x_surf.shape[:3], -1)
                surf_preds = unpatchify(x_surf, len(surf_vars), H, W, patch_size)
                surf_preds = surf_preds.squeeze(2) # [B, V_S, H, W]
                
                ### SMoE ###
                if self.use_smoe:
                    surf_outputs = []
                    for i, name in enumerate(surf_vars):
                        # 1) slice single channel data for this variable
                        var_data = surf_preds[:, i : i+1, :, :]   # => [B, 1, H, W]
                        # 2) Pass through that var's SMoE
                        out = self.surf_smoe[name](var_data) + var_data  # => [B, 1, H, W]
                        surf_outputs.append(out)

                    # 3) Stack the outputs => [B, V_surf, H, W]
                    surf_preds = torch.cat(surf_outputs, dim=1)
                    ### SMoE ###

                surf_preds_ensemble.append(surf_preds)
            # Stack ensemble predictions
            surf_preds_all = torch.stack(surf_preds_ensemble, dim=1)  # [B, E, V_S, H, W]
            # surf_preds_mean = torch.mean(surf_preds_all, dim=1)  # [B, V_S, H, W]
            # surf_preds_std = torch.std(surf_preds_all, dim=1)    # [B, V_S, H, W]


        # Process atmospheric variables
        atmos_levels_encode = levels_expansion(
            torch.tensor(atmos_levels, device=x.device), self.embed_dim
        ).to(dtype=x.dtype)
        levels_embed = self.atmos_levels_embed(atmos_levels_encode)
        levels_embed = levels_embed.expand(B, x.size(1), -1, -1)
        x_atmos = self.deaggregate_levels(levels_embed, x[..., 1:, :])

        # Run ensemble predictions for atmospheric variables
        atmos_preds_ensemble = []
        for i in range(self.num_ensemble):
            if isinstance(atmos_heads[list(atmos_vars)[0]][0], SMoELayer):
                # Reshape for SMoE (expects 4D input: B*L, D, H, W), where L is pressure levels
                x_atmos_flat = x_atmos.reshape(B * x_atmos.size(2), self.embed_dim, patch_res[1], patch_res[2])  # [B*L, D, H, W]
                
                x_atmos_i = torch.stack([
                    atmos_heads[name][i](x_atmos_flat)  # [B*L, patch_size**2, H, W]
                    .permute(0, 2, 3, 1)  # [B*L, H, W, patch_size**2]
                    .reshape(B,  x_atmos.size(2), patch_res[1]*patch_res[2], -1)  # [B, L, H*W, patch_size**2]
                    .permute(0,2,1,3)
                    for name in atmos_vars
                ], dim=-1)  # Final: [B, H*W, L, patch_size**2, num_vars]
            else:
                x_atmos_i = torch.stack([
                    atmos_heads[name][i](x_atmos) 
                    for name in atmos_vars
                ], dim=-1)
                #x_atmos_i: torch.Size([1, 64800, 8, 16, 5])
            
            x_atmos_i = x_atmos_i.reshape(*x_atmos_i.shape[:3], -1)
            atmos_preds = unpatchify(x_atmos_i, len(atmos_vars), H, W, patch_size) # [B, V_A, L, H, W]

            ### SMoE ###
            if self.use_smoe:
                B, V, L, H, W = atmos_preds.shape # [B, V_A, L, H, W]
                atmos_smoe = atmos_preds.permute(0, 2, 1, 3, 4) # [B, L, V_A, H, W]
                # 2) reshape to [B*L, V, H, W]
                atmos_smoe = atmos_smoe.reshape(B * L, V, H, W) 
                atmos_outputs = []
                for i, name in enumerate(atmos_vars):
                    # 1) slice single channel data for this variable
                    var_data = atmos_smoe[:, i : i+1, :, :]   # => [B*L, 1, H, W]
                    # 2) Pass through that var's SMoE
                    out = self.atmos_smoe[name](var_data) + var_data # => [B*L, 1, H, W]
                    atmos_outputs.append(out)
                # 3) Stack the outputs => [B*L, V_A, H, W]
                atmos_preds = torch.cat(atmos_outputs, dim=1)
                # 4) Reshape back to [B, V_A, L, H, W]
                atmos_preds = atmos_preds.reshape(B, L, V, H, W)
                # permute back to [B, L, V_A, H, W]    
                atmos_preds = atmos_preds.permute(0, 2, 1, 3, 4)  # [B, L, V_A, H, W]
                ### SMoE ###

            atmos_preds_ensemble.append(atmos_preds)
        
        # Stack ensemble predictions
        atmos_preds_all = torch.stack(atmos_preds_ensemble, dim=1)  # [B, E, V_A, H, W]
        # atmos_preds_mean = torch.mean(atmos_preds_all, dim=1)  # [B, V_A, H, W]
        # atmos_preds_std = torch.std(atmos_preds_all, dim=1)   # [B, V_A, H, W]

        # Create three batches for mean, std, and all predictions
        # mean_batch = Batch(
        #     {v: surf_preds_mean[:, i] for i, v in enumerate(surf_vars)},
        #     batch.static_vars,
        #     {v: atmos_preds_mean[:, i] for i, v in enumerate(atmos_vars)},
        #     Metadata(
        #         lat=lat,
        #         lon=lon,
        #         time=tuple(t + lead_time for t in batch.metadata.time),
        #         atmos_levels=atmos_levels,
        #         rollout_step=batch.metadata.rollout_step + 1,
        #     ),
        # )

        # std_batch = Batch(
        #     {v: surf_preds_std[:, i] for i, v in enumerate(surf_vars)},
        #     batch.static_vars,
        #     {v: atmos_preds_std[:, i] for i, v in enumerate(atmos_vars)},
        #     mean_batch.metadata,
        # )

        all_preds_batch = Batch(
            {v: surf_preds_all[:, :, i] for i, v in enumerate(surf_vars)},
            batch.static_vars,
            {v: atmos_preds_all[:, :, i] for i, v in enumerate(atmos_vars)},
            Metadata(
                dataset_name=dataset_name,
                lat=lat,
                lon=lon,
                time=tuple(t + lead_time for t in batch.metadata.time),
                atmos_levels=atmos_levels,
                locations=batch.metadata.locations,
                scales=batch.metadata.scales,
                rollout_step=batch.metadata.rollout_step + 1,
            ),
        )

        #return mean_batch, std_batch, all_preds_batch  
        return all_preds_batch
