import torch
import logging
import torch.nn.functional as F
from aurora import Batch
import numpy as np

epsilon = 1e-6

class MSELoss:
        def __init__(self, d_var_weights=None, d_pres_weights=None):
            '''
            if d_var_weights=None, all vars are weighted equally. Any vars not in dict keys are weighted 1.
            if d_pres_weights=None, all pres levels are weighted equally. Any pres levels not in dict keys are weighted 1. weights in d_pres_weights are multiplied with d_var_weights.
            '''
            self.d_var_weights = d_var_weights if d_var_weights is not None else {}
            self.d_pres_weights = d_pres_weights if d_pres_weights is not None else {}
            self.loss_fn = torch.nn.MSELoss()
            self.loss_no_reduction = torch.nn.MSELoss(reduction='none')

        def get_loss(self, pred_batch, target_batch):
            
            loss_individual_vars = {}
            ## surface vars
            loss_srf = []
            for k in target_batch.surf_vars.keys():
                if k not in self.d_var_weights.keys():
                    self.d_var_weights[k] = 1
                loss_srf_ = self.loss_fn(pred_batch.surf_vars[k], target_batch.surf_vars[k]) * self.d_var_weights[k]
                print('loss_srf_', loss_srf_)
                loss_individual_vars[f'srf_{k}'] = loss_srf_
                loss_srf.append(loss_srf_)
            loss_srf = torch.stack(loss_srf).mean() ## take average over all surf_vars
            ###
            # ## static vars
            ### Static vars do not have gradients.
            # loss_static = []
            # for k in target_batch.static_vars.keys():
            #     if k not in self.d_var_weights.keys():
            #         self.d_var_weights[k] = 1
            #     loss_static_ = self.loss_fn(pred_batch.static_vars[k], target_batch.static_vars[k]) * self.d_var_weights[k]
            #     loss_individual_vars[f'static_{k}'] = loss_static_
            #     loss_static.append(loss_static_)
            # loss_static = torch.stack(loss_static).mean() ## take average over all static_vars
            ###
            ## atmos vars
            loss_atmos = []
            ## check pressure level weights dict
            for c in pred_batch.metadata.atmos_levels:
                if c not in self.d_pres_weights.keys():
                    self.d_pres_weights[c] = 1
            if all(v == 1 for v in self.d_pres_weights.values()):
                scale_pres_lvls = False
            else:
                scale_pres_lvls = True
                weights_pres_lvl = torch.tensor([self.d_pres_weights[c] for c in pred_batch.metadata.atmos_levels], device=pred_batch.atmos_vars[list(pred_batch.atmos_vars.keys())[0]].device)
                pres_lvl_len = torch.tensor(weights_pres_lvl.size(0), dtype=torch.float32, device=weights_pres_lvl.device)
            for k in target_batch.atmos_vars.keys():
                if k not in self.d_var_weights.keys():
                    self.d_var_weights[k] = 1
                if scale_pres_lvls:
                    loss_atmos_ = self.loss_no_reduction(pred_batch.atmos_vars[k], target_batch.atmos_vars[k]) * self.d_var_weights[k] ## shape: (batch, pressure levels, lat, lon)
                    loss_atmos_ = torch.einsum('bchw,c->bchw', loss_atmos_, weights_pres_lvl) / pres_lvl_len ## shape: (batch, pressure levels, lat, lon)
                else:
                    loss_atmos_ = self.loss_fn(pred_batch.atmos_vars[k], target_batch.atmos_vars[k]) * self.d_var_weights[k]
                print('loss_atmos_', loss_atmos_)
                loss_individual_vars[f'atmos_{k}'] = loss_atmos_
                loss_atmos.append(loss_atmos_)
            loss_atmos = torch.stack(loss_atmos).mean() ## take average over all atmos_vars
            loss_total = loss_atmos + loss_srf # + loss_static
            dict_losses = {**{'loss_srf': loss_srf, 'loss_atmos': loss_atmos}, **loss_individual_vars} #, 'loss_static': loss_static
            # l_nans = []
            # for k,v in dict_losses.items():
            #     if torch.isnan(v):
            #         l_nans.append(k)
            # if len(l_nans) > 0:
            #     logging.info(f'Following losses are NaN: {str(",").join(l_nans)}')
            return loss_total, dict_losses


class WeightedMAELoss:
    def __init__(
        self,
        surf_weight: float = 1/4,
        atmos_weight: float = 1.0,
        surf_var_weights: dict[str, float] = None,
        atmos_var_weights: dict[str, float] = None,
        dataset_weight: int = 2,
        reduction: bool = True,
        latitude_weight: bool = True,
        replace_nan_with_zero: bool = False,
    ) -> None:
        if surf_var_weights is None:
            surf_var_weights = {
                'msl': 1.5,
                '10u': 0.77,
                '2t': 3.0,
            }
        if atmos_var_weights is None:
            atmos_var_weights = {
                'z': 2.8,
                'q': 0.78,
                't': 1.7,
                'u': 0.87,
                'v': 0.6
            }

        self.surf_weight = surf_weight
        self.atmos_weight = atmos_weight
        self.dataset_weight = dataset_weight
        self.surf_var_weights = surf_var_weights
        self.atmos_var_weights = atmos_var_weights
        self.reduction = reduction
        if not reduction:
            raise NotImplementedError('reduction=False is not implemented for WeightedMAELoss. Please use reduction=True.')
        self.latitude_weight = latitude_weight
        self.replace_nan_with_zero = replace_nan_with_zero

    def __call__(self, pred_batch, target_batch):
        latitudes = torch.deg2rad(pred_batch.metadata.lat)
        latitude_weight = torch.cos(latitudes) / torch.cos(latitudes).mean()
        latitude_weight = latitude_weight[None, None, :, None] # shape (1, 1, lat, 1)

        num_vars = (len(target_batch.surf_vars) + len(target_batch.atmos_vars))

        groups = [
            (target_batch.surf_vars, pred_batch.surf_vars, self.surf_weight, self.surf_var_weights), 
            (target_batch.atmos_vars, pred_batch.atmos_vars, self.atmos_weight, self.atmos_var_weights)
        ]
        loss_dict = {}
        total_loss = 0
        for target, pred, group_weight, var_weights in groups:
            group_loss = 0
            for var_name in sorted(target.keys()):
                pred_var = pred[var_name]
                target_var = target[var_name]
                
                # Create mask for non-NaN values in target_var
                mask_nonnan = ~torch.isnan(target_var)
                
                # Handle dimensional mismatch between pred_var and target_var
                mask_expanded = mask_nonnan
                if pred_var.shape != target_var.shape:
                    # Expand mask to match pred_var shape for proper indexing
                    while mask_expanded.ndim < pred_var.ndim:
                        mask_expanded = mask_expanded.unsqueeze(1)
                    # Broadcast the mask to match pred_var shape
                    mask_expanded = mask_expanded.expand_as(pred_var)
                    
                    # Compute error on full tensors first, then mask
                    err_full = abs(pred_var - target_var)  # Broadcasting handles dimension mismatch
                    err = err_full[mask_expanded]  # Select only valid elements
                else:
                    # Same shape case - direct masking
                    err = abs(pred_var[mask_nonnan] - target_var[mask_nonnan])
                
                # Handle case where all target values are NaN
                if err.numel() == 0:
                    err = torch.tensor(0.0, device=target_var.device)
                
                if self.latitude_weight:
                    lat_weights_expanded = latitude_weight
                    while lat_weights_expanded.ndim < pred_var.ndim:
                        lat_weights_expanded = lat_weights_expanded.unsqueeze(1)
                    lat_weights_expanded = lat_weights_expanded.expand_as(pred_var)
                    lat_weights_masked = lat_weights_expanded[mask_expanded]
                    
                    if err.numel() > 0:  # Only apply if err is not empty
                        err = err * lat_weights_masked
 
                # calculate the average loss over entire loss
                if self.replace_nan_with_zero:
                    mean = err.nanmean()
                    mean = torch.tensor(0, device=target_var.device) if mean.isnan() else mean
                else:
                    mean = err.mean()

                loss_dict[var_name] = mean
                group_loss += mean * var_weights.get(var_name, torch.tensor(1.0, device=target_var.device))

            total_loss += group_weight * group_loss

        loss_dict['total_mae'] = total_loss
        #return (self.dataset_weight / num_vars) * total_loss, loss_dict # leave the normalization to CombinedLoss
        return  total_loss, loss_dict

    def get_loss(self, pred_batch, target_batch, **kwargs): # this is here only to support backward compatibility in the code.
        return self.__call__(pred_batch, target_batch)


class CombinedLoss:
    def __init__(
        self,
        surf_weight: float = 1/4,
        atmos_weight: float = 1.0,
        surf_var_weights: dict[str, float] = None,
        atmos_var_weights: dict[str, float] = None,
        dataset_weight: int = 2,
        reduction: bool = True,
        latitude_weight: bool = True,
        mae_weight: float = 1.0,
        nll_weight: float = 1.0,
        crps_weight: float = 1.0,
        kernel_crps_weight: float = 1.0,
        stats_loss_weight: float = 1.0,
        kill_if_nan_in_preds: bool = True,
    ) -> None:
        # Define default weights if None is provided
        if surf_var_weights is None:
            surf_var_weights = {
                'msl': 1.5,
                '10u': 0.77,
                '2t': 3.0,
            }
        if atmos_var_weights is None:
            atmos_var_weights = {
                'z': 2.8,
                'q': 0.78,
                't': 1.7,
                'u': 0.87,
                'v': 0.6
            }

        # Initialize the WeightedMAELoss with the same weights
        if kill_if_nan_in_preds:
            replace_mae_nan_with_zero = False
        else:
            replace_mae_nan_with_zero = True
        self.mae_loss = WeightedMAELoss(
            surf_weight=surf_weight,
            atmos_weight=atmos_weight,
            surf_var_weights=surf_var_weights,
            atmos_var_weights=atmos_var_weights,
            dataset_weight=dataset_weight,
            reduction=reduction,
            latitude_weight=latitude_weight,
            replace_nan_with_zero=replace_mae_nan_with_zero,
        )
        
        # Store parameters
        self.surf_weight = surf_weight
        self.atmos_weight = atmos_weight
        self.surf_var_weights = surf_var_weights  # Now guaranteed to not be None
        self.atmos_var_weights = atmos_var_weights  # Now guaranteed to not be None
        self.dataset_weight = dataset_weight
        self.reduction = reduction
        self.latitude_weight = latitude_weight
        
        # Weights for different loss components
        self.mae_weight = mae_weight
        self.nll_weight = nll_weight
        self.crps_weight = crps_weight
        self.kernel_crps_weight = kernel_crps_weight        
        self.kernel_crps_weight = kernel_crps_weight
        self.stats_loss_weight = stats_loss_weight
        
        self.kill_if_nan_in_preds = kill_if_nan_in_preds
        
    def stats_loss(self, x, mu, std):
        ### adapted from stats loss from atmorep: https://github.com/clessig/atmorep/blob/main/atmorep/utils/utils.py#L347 ###
        stats_loss = torch.exp(-0.5 * (x-mu)*(x-mu) / (std*std + epsilon))
        diff = stats_loss - 1.
        stats_loss = torch.mean(diff * diff) + torch.mean(torch.sqrt(torch.abs(std)))
        return stats_loss

    def gaussian_nll(self, y, mean, std):
        """Gaussian Negative Log Likelihood"""
        return 0.5 * torch.log(2 * torch.pi * (std**2 + epsilon)) + \
               0.5 * ((y - mean)**2) / (std**2 + epsilon)

    def crps_loss(self, y, mean, std):
        """Continuous Ranked Probability Score for Gaussian distribution"""
        normalized_diff = (y - mean) / (std + epsilon)
        # Convert pi to a tensor with the same device as the input tensors
        pi_tensor = torch.tensor(torch.pi, device=y.device)
        return std * (normalized_diff * (2 * torch.erf(normalized_diff / torch.sqrt(torch.tensor(2.0, device=y.device))) - 1) + \
               2 / torch.sqrt(pi_tensor) * torch.exp(-(normalized_diff**2 / 2)))

    def expensive_kernel_crps(self, y, ensemble_preds):
        """Kernel CRPS using ensemble predictions"""
        diff_ey = torch.abs(ensemble_preds - y.unsqueeze(1))  # [B, E, ...]
        diff_ee = torch.abs(ensemble_preds.unsqueeze(2) - ensemble_preds.unsqueeze(1))  # [B, E, E, ...]
        return torch.mean(diff_ey, dim=1) - 0.5 * torch.mean(diff_ee, dim=(1,2))

    # Efficient kernel CRPS
    # https://docs.nvidia.com/deeplearning/modulus/modulus-core/_modules/modulus/metrics/general/crps.html#:~:text=torch.Tensor-,%40torch.jit.script,-def%20_kernel_crps_implementation(
    def kernel_crps(self, y, ensemble_preds, biased: bool = False):
        """Efficient kernel CRPS implementation with O(m log m) complexity."""
        if ensemble_preds.ndim == 5 and y.ndim == 3:
            skill = torch.abs(ensemble_preds - y[:, None, None, ...]).mean(1)  # Mean over ensemble dimension
        else:
            skill = torch.abs(ensemble_preds - y.unsqueeze(1)).mean(1)  # Mean over ensemble dimension
        pred, _ = torch.sort(ensemble_preds, dim=1)  # Sort along ensemble dimension
        
        # Efficient spread calculation
        m = pred.size(1)  # Ensemble size
        i = torch.arange(1, m + 1, device=pred.device, dtype=pred.dtype)
        denom = m * m if biased else m * (m - 1)
        factor = (2 * i - m - 1) / denom
        spread = torch.sum(factor.view(1, -1, *([1] * (pred.dim() - 2))) * pred, dim=1)
        
        return skill - spread
    
    def _check_for_nan(self, pred_batch):
        any_nans = False
        nan_str = ''
        for k in pred_batch.atmos_vars.keys():
            nan = torch.isnan(pred_batch.atmos_vars[k])
            if nan.any():
                any_nans = True
                nan_str += f'Found NaN in pred_batch.atmos_vars[{k}]. Number of NaNs: {nan.sum()}/{nan.numel()}, pred feature shape: {pred_batch.atmos_vars[k].shape}\n'
        for k in pred_batch.surf_vars.keys():
            nan = torch.isnan(pred_batch.surf_vars[k])
            if nan.any():
                any_nans = True
                nan_str += f'Found NaN in pred_batch.surf_vars[{k}]. Number of NaNs: {nan.sum()}/{nan.numel()}, pred feature shape: {pred_batch.surf_vars[k].shape}\n'
        if any_nans:
            logging.warning(f'NaNs found in predictions:\n{nan_str}')
        return any_nans

    def get_loss(self, pred_batch, std_batch, ens_batch, target_batch):
        
        mae_total, mae_dict = self.mae_loss(pred_batch, target_batch)
        total_loss = self.mae_weight * mae_total
        
        if self.kill_if_nan_in_preds and self._check_for_nan(pred_batch):
            raise ValueError('NaNs found in predictions. Aborting training.')
        
        latitudes = torch.deg2rad(pred_batch.metadata.lat)
        latitude_weight = torch.cos(latitudes) / torch.cos(latitudes).mean() if self.latitude_weight else 1.0

        losses = mae_dict  # Start with MAE losses
        num_vars = (len(target_batch.surf_vars) + len(target_batch.atmos_vars))

        # Process both surface and atmospheric variables for other losses
        groups = [
            (target_batch.surf_vars, pred_batch.surf_vars, std_batch.surf_vars, 
             ens_batch.surf_vars, self.surf_weight, self.surf_var_weights),
            (target_batch.atmos_vars, pred_batch.atmos_vars, std_batch.atmos_vars, 
             ens_batch.atmos_vars, self.atmos_weight, self.atmos_var_weights)
        ]

        for target, pred, std, ens, group_weight, var_weights in groups:
            group_loss = 0
            for var_name in sorted(target.keys()):
                weight = var_weights.get(var_name, 1.0)
                
                # Apply latitude weighting if enabled
                if self.latitude_weight:
                    target_var = target[var_name] * latitude_weight[..., None]
                    pred_var = pred[var_name] * latitude_weight[..., None]
                    std_var = std[var_name] * latitude_weight[..., None]
                    ens_var = ens[var_name] * latitude_weight[..., None]
                else:
                    target_var = target[var_name]
                    pred_var = pred[var_name]
                    std_var = std[var_name]
                    ens_var = ens[var_name]

                # Only compute losses if their weights are non-zero
                stats_loss = torch.nan_to_num(self.stats_loss(target_var, pred_var, std_var).nanmean(), nan=0.0, posinf=0., neginf=0.) if self.stats_loss_weight != 0 else torch.zeros_like(total_loss)
                nll = torch.nan_to_num(self.gaussian_nll(target_var, pred_var, std_var).nanmean(), nan=0.0, posinf=0., neginf=0.) if self.nll_weight != 0 else torch.zeros_like(total_loss)
                crps = torch.nan_to_num(self.crps_loss(target_var, pred_var, std_var).nanmean(), nan=0.0, posinf=0., neginf=0.) if self.crps_weight != 0 else torch.zeros_like(total_loss)
                kernel_crps = torch.nan_to_num(self.kernel_crps(target_var, ens_var).nanmean(), nan=0.0, posinf=0., neginf=0.) if self.kernel_crps_weight != 0 else torch.zeros_like(total_loss)

                # Store individual losses only if their weights are non-zero
                if self.stats_loss_weight != 0:
                    losses[f'tail/{var_name}_stats'] = stats_loss
                if self.nll_weight != 0:
                    losses[f'tail/{var_name}_nll'] = nll
                if self.crps_weight != 0:
                    losses[f'tail/{var_name}_crps'] = crps
                if self.kernel_crps_weight != 0:
                    losses[f'tail/{var_name}_kcrps'] = kernel_crps

                # Combine statistical losses with their respective weights
                var_loss = (self.stats_loss_weight * stats_loss +
                            self.nll_weight * nll + 
                            self.crps_weight * crps + 
                            self.kernel_crps_weight * kernel_crps)

                group_loss += var_loss * weight

            total_loss += group_weight * group_loss

        # Before final normalization
        total_loss = (self.dataset_weight / num_vars) * total_loss
        if not isinstance(total_loss, torch.Tensor):
            total_loss = torch.tensor(total_loss, requires_grad=True, device=pred_batch.metadata.lat.device)
 
        losses['total'] = total_loss
        # assert total_loss > 0, 'total loss is zero'

        return total_loss, losses







# the RC loss: https://github.com/spcl/smoe/blob/main/smoe/models/loss.py
class RoutingClassificationLoss:
    """Loss for training SMoE gates using routing classification"""
    
    @staticmethod
    @torch.no_grad()
    def construct_routing_labels(correct: torch.Tensor,
                               routing_indices: torch.Tensor,
                               num_experts: int) -> torch.Tensor:
        """Construct labels for gate routing classification.
        
        Args:
            correct: Tensor indicating if each expert was correct (1) or not (0)
            routing_indices: Indices of selected experts
            num_experts: Total number of experts
            
        Returns:
            Labels tensor for training gates
        """
        out_planes = correct.size(1)
        # Number of experts not selected at each point
        unselected = num_experts - out_planes
        if unselected == 0:
            raise RuntimeError('0 unselected experts not supported')
            
        # Get number of wrong selections at each point
        num_wrong = out_planes - correct.sum(dim=1, keepdim=True)
        
        # Compute weight for unselected experts
        unselected_weight = num_wrong / unselected
        
        # Build initial labels tensor
        labels = unselected_weight.repeat(1, num_experts, *([1]*(correct.ndim - 2)))
        
        # Set correct/incorrect labels for selected experts
        labels.scatter_(dim=1, index=routing_indices, src=correct)
        
        return labels

    @staticmethod
    def compute_loss_from_predictions(net, output: Batch, target: Batch, abstol=1e-5, reltol=0.0):
        """Compute routing classification loss using predictions vs targets."""
        
        # Get latitude weights like in WeightedMAELoss
        latitudes = torch.deg2rad(output.metadata.lat)
        latitude_weight = torch.cos(latitudes) / torch.cos(latitudes).mean()
        # Add necessary dimensions for broadcasting
        latitude_weight = latitude_weight[:, None]  # Shape: (lat, 1)

        # Process both surface and atmospheric variables like in WeightedMAELoss
        groups = [
            (target.surf_vars, output.surf_vars), 
            (target.atmos_vars, output.atmos_vars)
        ]
        
        # Initialize correctness tensor
        all_correct = []
        
        for target_vars, pred_vars in groups:
            group_correct = []
            for var_name in sorted(target_vars.keys()):
                # Skip if variable doesn't exist in predictions
                if var_name not in pred_vars:
                    continue

                # Compute absolute difference
                err = abs(pred_vars[var_name] - target_vars[var_name])
                
                # Apply latitude weighting with proper broadcasting
                if err.dim() == 4:  # (batch, channel, lat, lon)
                    err = err * latitude_weight[None, None, :, None]
                elif err.dim() == 3:  # (batch, lat, lon)
                    err = err * latitude_weight[None, :, None]
                
                # Determine correctness based on tolerance
                if reltol > 0:
                    rel_err = err / (target_vars[var_name].abs().clamp(min=1e-8))
                    is_correct = ((err <= abstol) | (rel_err <= reltol)).float()
                else:
                    is_correct = (err <= abstol).float()
                
                # Average across all dimensions except batch and spatial (lat, lon)
                while is_correct.dim() > 3:  # Reduce to (batch, lat, lon)
                    is_correct = is_correct.mean(dim=1)
                
                group_correct.append(is_correct)
            
            # Combine correctness for this group
            if group_correct:
                # Stack and mean across variables within the group
                group_correct = torch.stack(group_correct).mean(dim=0)  # Average across variables
                all_correct.append(group_correct)

        if not all_correct:
            raise ValueError("No matching variables found between output and target")

        # Average across all groups
        correct = torch.mean(torch.stack(all_correct, dim=0), dim=0)  # Average across groups

        # Initialize losses list
        losses = []

        # Compute routing classification loss for each SMoE layer
        for module in net.modules():
            if hasattr(module, 'routing_weights'):
                # Get the shape of routing indices
                routing_shape = module.routing_indices.shape
                
                # Reshape correctness tensor to match routing indices shape
                correct_expanded = correct.unsqueeze(1)  # Add channel dimension
                if correct_expanded.shape != routing_shape:
                    correct_expanded = correct_expanded.expand_as(module.routing_indices)
                
                # Construct labels based on correctness
                labels = RoutingClassificationLoss.construct_routing_labels(
                    correct_expanded,
                    module.routing_indices,
                    module.smoe_config.num_experts
                )
                
                # Compute binary cross entropy loss
                loss = F.binary_cross_entropy_with_logits(
                    module.routing_weights, 
                    labels
                )
                losses.append(loss)

        # Return mean of all routing losses
        if losses:
            return torch.stack(losses).mean()
        return torch.tensor(0.0, device=net.device)

    @staticmethod 
    def compute_loss_from_error(net: torch.nn.Module,
                              quantile: float = 0.3,
                              expert_range: tuple = None) -> torch.Tensor:
        """Compute routing classification loss based on error signals"""
        losses = []
        
        for module in net.modules():
            if hasattr(module, 'saved_error_signal'):
                # Get correct routing based on error magnitude
                with torch.no_grad():
                    # Get the actual batch size from the saved error signal
                    error_signal = module.saved_error_signal
                    error_threshold = torch.quantile(
                        error_signal.abs(),
                        quantile, dim=1, keepdim=True)
                    correct = (error_signal.abs() <= error_threshold).float()
                
                # Ensure routing_weights matches the batch size of error_signal
                routing_weights = module.routing_weights[:error_signal.size(0)]
                # Get routing labels
                labels = RoutingClassificationLoss.construct_routing_labels(
                    correct, module.routing_indices[:error_signal.size(0)], 
                    module.smoe_config.num_experts)
                
                # Compute BCE loss
                loss = F.binary_cross_entropy_with_logits(
                    routing_weights, labels)
                losses.append(loss)
                
        return sum(losses) if losses else torch.tensor(0.0)



class KD_Loss_activations:
    """Knowledge Distillation Loss on activations. """
    
    def __init__(self, criterion='l1'):
        if str(criterion).lower() == 'l2':
            self.loss_fn = torch.nn.MSELoss(reduction='mean')
        elif str(criterion).lower() == 'l1':
            self.loss_fn = torch.nn.L1Loss(reduction='mean')
        else:
            raise ValueError(f"Unknown criterion: {criterion}. Supported: 'l2', 'l1'.")
    
    def __call__(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor):
        """Compute the knowledge distillation loss.
        
        Args:
            student_logits: Logits from the student model.
            teacher_logits: Logits from the teacher model.
        
        Returns:
            Computed KD loss.
        """
        loss = self.loss_fn(student_logits, teacher_logits)
        
        return loss
    
    def get_loss(self, pred_batch, target_batch):
        """Get the loss for list of student and teacher logits.
        
        Args:
            pred_batch: List of student logits.
            target_batch: List of teacher logits.
        
        Returns:
            Computed KD loss.
        """
        return self.__call__(student_logits=pred_batch, teacher_logits=target_batch)



