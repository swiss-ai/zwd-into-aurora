from typing import Optional, Dict
import logging
import torch
from aurora import Aurora
from aurora.model import swin3d as aurora_swin3d_module

class ModelFreezer:
    """Utility class to manage freezing/unfreezing of Aurora model parameters."""
    
    def __init__(self, model: Aurora):
        """Initialize the ModelFreezer.
        
        Args:
            model (Aurora): The Aurora model instance to manage
        """
        self.model = model
        self.frozen_states: Dict[str, bool] = {}
        self._save_initial_states()
        
    def _save_initial_states(self):
        """Save the initial requires_grad states of all parameters."""
        for name, param in self.model.named_parameters():
            self.frozen_states[name] = not param.requires_grad
            
    def freeze_parts(
        self,
        freeze_encoder: bool = False,
        freeze_backbone: bool = True,
        freeze_decoder: bool = False,
        unfreeze_last_n_backbone_layers: Optional[int] = None,
    ) -> None:
        """Selectively freeze parts of the Aurora model.
        
        Args:
            freeze_encoder (bool): Whether to freeze the encoder
            freeze_backbone (bool): Whether to freeze the backbone
            freeze_decoder (bool): Whether to freeze the decoder
            unfreeze_last_n_backbone_layers (Optional[int]): Number of final backbone layers to unfreeze
        """
        # First clear all gradients and empty CUDA cache
        self.model.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Handle encoder
        if freeze_encoder:
            self._set_requires_grad(self.model.encoder, False)
        else:
            self._set_requires_grad(self.model.encoder, True)
        
        # Handle backbone
        if freeze_backbone:
            self._set_requires_grad(self.model.backbone, False)
            if unfreeze_last_n_backbone_layers is not None:
                self._unfreeze_last_n_backbone_layers(unfreeze_last_n_backbone_layers)
        else:
            self._set_requires_grad(self.model.backbone, True)
        
        # Handle decoder
        if freeze_decoder:
            self._set_requires_grad(self.model.decoder, False)
        else:
            self._set_requires_grad(self.model.decoder, True)
        
        # Update frozen states and log stats
        self._update_frozen_states()
        self.log_parameter_stats()
        
    def _set_requires_grad(self, module: torch.nn.Module, requires_grad: bool):
        """Set requires_grad for all parameters in a module."""
        for param in module.parameters():
            if param is not None:  # Check for None parameters
                param.requires_grad = requires_grad

            
    def _unfreeze_last_n_encoder_layers(self, n: int):
        """Unfreeze the last n layers of the encoder."""
        encoder_layers = []
        
        # Get all relevant layer types from the Perceiver3DEncoder
        for name, module in self.model.encoder.named_modules():
            if isinstance(module, (PerceiverResampler, MLP, nn.LayerNorm)):
                encoder_layers.append(module)
        
        # Unfreeze last n layers
        num_layers = len(encoder_layers)
        start_idx = max(0, num_layers - n)
        for i in range(start_idx, num_layers):
            self._set_requires_grad(encoder_layers[i], True)
            
    def _unfreeze_last_n_backbone_layers(self, n: int):
        """Unfreeze the last n layers of both encoder and decoder in backbone."""
        # Handle encoder layers
        total_enc_layers = len(self.model.backbone.encoder_layers)
        start_idx = max(0, total_enc_layers - n)
        for i in range(start_idx, total_enc_layers):
            self._set_requires_grad(self.model.backbone.encoder_layers[i], True)
            
        # Handle decoder layers
        total_dec_layers = len(self.model.backbone.decoder_layers)
        start_idx = max(0, total_dec_layers - n)
        for i in range(start_idx, total_dec_layers):
            self._set_requires_grad(self.model.backbone.decoder_layers[i], True)
            
    def _unfreeze_last_n_decoder_layers(self, n: int):
        """Unfreeze the last n layers of the decoder."""
        decoder_layers = []
        
        # Get all relevant layer types from the Decoder
        for name, module in self.model.decoder.named_modules():
            if isinstance(module, (ResnetBlock, AttnBlock, Normalize)):
                decoder_layers.append(module)
        
        # Unfreeze last n layers
        num_layers = len(decoder_layers)
        start_idx = max(0, num_layers - n)
        for i in range(start_idx, num_layers):
            self._set_requires_grad(decoder_layers[i], True)
            
    def _update_frozen_states(self):
        """Update the frozen states dictionary."""
        for name, param in self.model.named_parameters():
            self.frozen_states[name] = not param.requires_grad
            
    def reset_to_initial_states(self):
        """Reset all parameters to their initial frozen/unfrozen states."""
        for name, param in self.model.named_parameters():
            param.requires_grad = not self.frozen_states[name]
            
    def log_parameter_stats(self):
        """Log statistics about frozen/unfrozen parameters."""
        total_params = 0
        trainable_params = 0
        
        # Count parameters by component
        stats = {
            'encoder': {'total': 0, 'trainable': 0},
            'backbone': {'total': 0, 'trainable': 0},
            'decoder': {'total': 0, 'trainable': 0}
        }
        
        for name, param in self.model.named_parameters():
            param_count = param.numel()
            total_params += param_count
            
            # Determine component
            if name.startswith('encoder'):
                component = 'encoder'
            elif name.startswith('backbone'):
                component = 'backbone'
            elif name.startswith('decoder'):
                component = 'decoder'
            else:
                continue
                
            # Update stats
            stats[component]['total'] += param_count
            if param.requires_grad:
                stats[component]['trainable'] += param_count
                trainable_params += param_count
        
        # Simplified layer_info - only tracking backbone
        layer_info = {
            'backbone': {'layers': [], 'frozen': 0}
        }

        # Collect layer information - only for backbone
        for name, module in self.model.named_modules():
            if isinstance(module, aurora_swin3d_module.BasicLayer3D):
                layer_info['backbone']['layers'].append(name)
                if not any(p.requires_grad for p in module.parameters()):
                    layer_info['backbone']['frozen'] += 1

        # Log existing stats
        logging.info(f"Parameter Statistics:")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Frozen parameters: {total_params - trainable_params:,}")
        
        # Log component-wise stats - detailed for backbone, simplified for encoder/decoder
        for component, counts in stats.items():
            trainable_percent = (counts['trainable'] / counts['total'] * 100) if counts['total'] > 0 else 0
            
            logging.info(f"\n{component.capitalize()} parameters:")
            logging.info(f"  Total: {counts['total']:,}")
            logging.info(f"  Trainable: {counts['trainable']:,} ({trainable_percent:.1f}%)")
            
            # Only show detailed layer information for backbone
            if component == 'backbone':
                total_layers = len(layer_info['backbone']['layers'])
                frozen_layers = layer_info['backbone']['frozen']
                logging.info(f"  Total layers: {total_layers}")
                logging.info(f"  Frozen/Total layers: {frozen_layers}/{total_layers}")
                logging.info("  Layer names:")
                for layer_name in layer_info['backbone']['layers']:
                    logging.info(f"    - {layer_name}")