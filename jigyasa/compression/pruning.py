"""
Model Pruning for Compression
Implements structured and unstructured pruning techniques
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
from dataclasses import dataclass
import logging


@dataclass
class PruningConfig:
    """Configuration for pruning"""
    pruning_ratio: float = 0.5
    structured: bool = True
    importance_metric: str = "magnitude"  # magnitude, gradient, taylor
    iterative: bool = True
    iterations: int = 10
    recovery_epochs: int = 5


class PruningScheduler:
    """
    Schedules pruning across training iterations
    """
    
    def __init__(
        self,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.9,
        begin_step: int = 0,
        end_step: int = 1000,
        frequency: int = 100
    ):
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.begin_step = begin_step
        self.end_step = end_step
        self.frequency = frequency
        self.current_step = 0
    
    def get_sparsity(self, step: Optional[int] = None) -> float:
        """Get sparsity level for current step"""
        if step is None:
            step = self.current_step
        
        if step < self.begin_step:
            return self.initial_sparsity
        elif step >= self.end_step:
            return self.final_sparsity
        else:
            # Linear interpolation
            progress = (step - self.begin_step) / (self.end_step - self.begin_step)
            sparsity = self.initial_sparsity + progress * (self.final_sparsity - self.initial_sparsity)
            return sparsity
    
    def should_prune(self, step: Optional[int] = None) -> bool:
        """Check if pruning should occur at this step"""
        if step is None:
            step = self.current_step
        
        if step < self.begin_step or step > self.end_step:
            return False
        
        return (step - self.begin_step) % self.frequency == 0
    
    def step(self):
        """Increment step counter"""
        self.current_step += 1


class StructuredPruner:
    """
    Structured pruning - removes entire channels, filters, or heads
    """
    
    def __init__(self, model: nn.Module, config: Optional[PruningConfig] = None):
        self.model = model
        self.config = config or PruningConfig()
        self.pruned_modules = []
        self.importance_scores = {}
        self.logger = logging.getLogger(__name__)
    
    def compute_importance_scores(self, dataloader=None) -> Dict[str, torch.Tensor]:
        """Compute importance scores for structured components"""
        importance_scores = {}
        
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Score output neurons
                if self.config.importance_metric == "magnitude":
                    scores = torch.norm(module.weight.data, p=2, dim=1)
                elif self.config.importance_metric == "gradient":
                    if hasattr(module.weight, 'grad') and module.weight.grad is not None:
                        scores = torch.norm(module.weight.grad, p=2, dim=1)
                    else:
                        scores = torch.norm(module.weight.data, p=2, dim=1)
                else:
                    scores = torch.norm(module.weight.data, p=2, dim=1)
                
                importance_scores[name] = scores
                
            elif isinstance(module, nn.Conv2d):
                # Score output channels
                weight = module.weight.data
                scores = torch.norm(weight.view(weight.size(0), -1), p=2, dim=1)
                importance_scores[name] = scores
                
            elif isinstance(module, nn.MultiheadAttention):
                # Score attention heads
                # This is simplified - actual implementation would be more complex
                if hasattr(module, 'out_proj'):
                    scores = torch.norm(module.out_proj.weight.data, p=2, dim=0)
                    # Group by heads
                    embed_dim = module.embed_dim
                    num_heads = module.num_heads
                    head_dim = embed_dim // num_heads
                    scores = scores.view(num_heads, head_dim).mean(dim=1)
                    importance_scores[name] = scores
        
        self.importance_scores = importance_scores
        return importance_scores
    
    def prune(self, pruning_ratio: Optional[float] = None) -> Dict[str, Any]:
        """Apply structured pruning"""
        if pruning_ratio is None:
            pruning_ratio = self.config.pruning_ratio
        
        # Compute importance scores if not already done
        if not self.importance_scores:
            self.compute_importance_scores()
        
        pruning_stats = {
            'modules_pruned': 0,
            'parameters_removed': 0,
            'compression_ratio': 0.0
        }
        
        total_params_before = sum(p.numel() for p in self.model.parameters())
        
        for name, module in self.model.named_modules():
            if name in self.importance_scores:
                scores = self.importance_scores[name]
                num_to_prune = int(len(scores) * pruning_ratio)
                
                if num_to_prune == 0:
                    continue
                
                # Get indices to prune (lowest scores)
                _, indices = torch.topk(scores, num_to_prune, largest=False)
                
                if isinstance(module, nn.Linear):
                    self._prune_linear(module, indices)
                elif isinstance(module, nn.Conv2d):
                    self._prune_conv2d(module, indices)
                elif isinstance(module, nn.MultiheadAttention):
                    self._prune_attention_heads(module, indices)
                
                self.pruned_modules.append(name)
                pruning_stats['modules_pruned'] += 1
        
        # Calculate compression
        total_params_after = sum(p.numel() for p in self.model.parameters())
        pruning_stats['parameters_removed'] = total_params_before - total_params_after
        pruning_stats['compression_ratio'] = 1 - (total_params_after / total_params_before)
        
        self.logger.info(f"Structured pruning complete: {pruning_stats}")
        return pruning_stats
    
    def _prune_linear(self, module: nn.Linear, indices: torch.Tensor):
        """Prune linear layer neurons"""
        # Create mask for keeping neurons
        keep_mask = torch.ones(module.out_features, dtype=torch.bool)
        keep_mask[indices] = False
        keep_indices = torch.where(keep_mask)[0]
        
        # Create new smaller layer
        new_out_features = len(keep_indices)
        new_module = nn.Linear(module.in_features, new_out_features, bias=module.bias is not None)
        
        # Copy weights
        new_module.weight.data = module.weight.data[keep_indices]
        if module.bias is not None:
            new_module.bias.data = module.bias.data[keep_indices]
        
        # Replace module (this is simplified - actual implementation would handle connections)
        module.weight = new_module.weight
        module.bias = new_module.bias
        module.out_features = new_out_features
    
    def _prune_conv2d(self, module: nn.Conv2d, indices: torch.Tensor):
        """Prune convolutional layer channels"""
        # Create mask for keeping channels
        keep_mask = torch.ones(module.out_channels, dtype=torch.bool)
        keep_mask[indices] = False
        keep_indices = torch.where(keep_mask)[0]
        
        # Create new smaller layer
        new_out_channels = len(keep_indices)
        new_module = nn.Conv2d(
            module.in_channels,
            new_out_channels,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.groups,
            module.bias is not None
        )
        
        # Copy weights
        new_module.weight.data = module.weight.data[keep_indices]
        if module.bias is not None:
            new_module.bias.data = module.bias.data[keep_indices]
        
        # Replace module
        module.weight = new_module.weight
        module.bias = new_module.bias
        module.out_channels = new_out_channels
    
    def _prune_attention_heads(self, module: nn.MultiheadAttention, head_indices: torch.Tensor):
        """Prune attention heads"""
        # This is a simplified implementation
        # Actual implementation would need to handle Q, K, V projections properly
        num_heads = module.num_heads
        keep_mask = torch.ones(num_heads, dtype=torch.bool)
        keep_mask[head_indices] = False
        num_kept_heads = keep_mask.sum().item()
        
        # Update module (simplified)
        module.num_heads = num_kept_heads
        self.logger.info(f"Pruned {len(head_indices)} attention heads, {num_kept_heads} remaining")


class UnstructuredPruner:
    """
    Unstructured pruning - removes individual weights
    """
    
    def __init__(self, model: nn.Module, config: Optional[PruningConfig] = None):
        self.model = model
        self.config = config or PruningConfig()
        self.pruned_modules = []
        self.masks = {}
        self.logger = logging.getLogger(__name__)
    
    def compute_importance_scores(self) -> Dict[str, torch.Tensor]:
        """Compute importance scores for individual weights"""
        importance_scores = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                if self.config.importance_metric == "magnitude":
                    scores = torch.abs(module.weight.data)
                elif self.config.importance_metric == "gradient":
                    if hasattr(module.weight, 'grad') and module.weight.grad is not None:
                        scores = torch.abs(module.weight.grad)
                    else:
                        scores = torch.abs(module.weight.data)
                elif self.config.importance_metric == "taylor":
                    # Taylor expansion approximation
                    if hasattr(module.weight, 'grad') and module.weight.grad is not None:
                        scores = torch.abs(module.weight.data * module.weight.grad)
                    else:
                        scores = torch.abs(module.weight.data)
                else:
                    scores = torch.abs(module.weight.data)
                
                importance_scores[name] = scores.flatten()
        
        return importance_scores
    
    def prune(self, pruning_ratio: Optional[float] = None) -> Dict[str, Any]:
        """Apply unstructured pruning"""
        if pruning_ratio is None:
            pruning_ratio = self.config.pruning_ratio
        
        pruning_stats = {
            'modules_pruned': 0,
            'weights_pruned': 0,
            'total_weights': 0,
            'sparsity': 0.0
        }
        
        # Get global importance scores
        importance_scores = self.compute_importance_scores()
        all_scores = torch.cat(list(importance_scores.values()))
        threshold = torch.quantile(all_scores, pruning_ratio)
        
        # Apply pruning
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                # Use PyTorch's pruning functionality
                prune.custom_from_mask(
                    module,
                    name='weight',
                    mask=(torch.abs(module.weight.data) > threshold)
                )
                
                # Track statistics
                mask = module.weight_mask if hasattr(module, 'weight_mask') else None
                if mask is not None:
                    self.masks[name] = mask
                    weights_pruned = (~mask).sum().item()
                    total_weights = mask.numel()
                    
                    pruning_stats['modules_pruned'] += 1
                    pruning_stats['weights_pruned'] += weights_pruned
                    pruning_stats['total_weights'] += total_weights
                
                self.pruned_modules.append(name)
        
        # Calculate overall sparsity
        if pruning_stats['total_weights'] > 0:
            pruning_stats['sparsity'] = pruning_stats['weights_pruned'] / pruning_stats['total_weights']
        
        self.logger.info(f"Unstructured pruning complete: {pruning_stats}")
        return pruning_stats
    
    def remove_pruning(self):
        """Remove pruning and make it permanent"""
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight_mask'):
                prune.remove(module, 'weight')
    
    def get_sparsity(self) -> float:
        """Calculate current model sparsity"""
        total_weights = 0
        pruned_weights = 0
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight'):
                weight = module.weight.data
                total_weights += weight.numel()
                pruned_weights += (weight == 0).sum().item()
                
                if hasattr(module, 'weight_mask'):
                    mask = module.weight_mask
                    pruned_weights += (~mask).sum().item()
        
        return pruned_weights / total_weights if total_weights > 0 else 0.0
    
    def iterative_prune(
        self,
        dataloader,
        criterion,
        optimizer,
        final_sparsity: float = 0.9,
        iterations: int = 10
    ):
        """Iterative magnitude pruning with recovery"""
        initial_sparsity = self.get_sparsity()
        sparsity_per_iteration = (final_sparsity - initial_sparsity) / iterations
        
        for iteration in range(iterations):
            # Calculate target sparsity for this iteration
            target_sparsity = initial_sparsity + (iteration + 1) * sparsity_per_iteration
            current_sparsity = self.get_sparsity()
            pruning_ratio = (target_sparsity - current_sparsity) / (1 - current_sparsity)
            
            # Prune
            self.prune(pruning_ratio)
            
            # Recovery training
            self.logger.info(f"Iteration {iteration + 1}/{iterations}: Recovery training...")
            for epoch in range(self.config.recovery_epochs):
                total_loss = 0.0
                for batch in dataloader:
                    # Training step (simplified)
                    optimizer.zero_grad()
                    outputs = self.model(batch['input'])
                    loss = criterion(outputs, batch['target'])
                    loss.backward()
                    optimizer.step()
                    total_loss += loss.item()
                
                avg_loss = total_loss / len(dataloader)
                self.logger.info(f"  Recovery epoch {epoch + 1}: Loss = {avg_loss:.4f}")
            
            # Log progress
            self.logger.info(f"Iteration {iteration + 1} complete: Sparsity = {self.get_sparsity():.2%}")
        
        # Make pruning permanent
        self.remove_pruning()
        
        return {
            'final_sparsity': self.get_sparsity(),
            'iterations': iterations,
            'initial_sparsity': initial_sparsity
        }