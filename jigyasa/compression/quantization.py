"""
Model Quantization for Deployment
Implements post-training quantization and quantization-aware training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import logging
import os
from pathlib import Path
import struct
from dataclasses import dataclass
import numpy as np

from ..core.model import JigyasaModel


@dataclass
class QuantizationConfig:
    """Configuration for model quantization"""
    weight_bits: int = 4
    activation_bits: int = 8
    symmetric: bool = True
    per_channel: bool = True
    calibration_samples: int = 128
    output_format: str = "gguf"  # "gguf", "onnx", "trt"


class ModelQuantizer:
    """
    Main quantizer for converting models to lower precision
    """
    
    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig()
        
        # Quantization statistics
        self.calibration_data = []
        self.quantization_stats = {}
        
    def calibrate(self, model: JigyasaModel, calibration_dataloader):
        """
        Calibrate quantization parameters using sample data
        
        Args:
            model: Model to calibrate
            calibration_dataloader: Data loader for calibration
        """
        logging.info("Starting quantization calibration...")
        
        model.eval()
        self.calibration_data = []
        
        # Collect activation statistics
        hooks = []
        activation_stats = {}
        
        def collect_stats(name):
            def hook(module, input, output):
                if isinstance(output, torch.Tensor):
                    if name not in activation_stats:
                        activation_stats[name] = []
                    activation_stats[name].append(output.detach().cpu())
            return hook
        
        # Register hooks for key layers
        for name, module in model.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                hook = module.register_forward_hook(collect_stats(name))
                hooks.append(hook)
        
        # Run calibration
        with torch.no_grad():
            for i, batch in enumerate(calibration_dataloader):
                if i >= self.config.calibration_samples:
                    break
                
                input_ids = batch['input_ids'].to(model.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(model.device)
                
                _ = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Calculate quantization parameters
        self.quantization_stats = self._calculate_quantization_params(activation_stats)
        
        logging.info(f"Calibration completed with {len(activation_stats)} layers")
    
    def _calculate_quantization_params(self, activation_stats: Dict[str, List[torch.Tensor]]) -> Dict[str, Dict]:
        """Calculate scale and zero-point for quantization"""
        quant_params = {}
        
        for name, activations in activation_stats.items():
            # Concatenate all activations for this layer
            all_activations = torch.cat(activations, dim=0).flatten()
            
            # Calculate min/max
            min_val = all_activations.min().item()
            max_val = all_activations.max().item()
            
            # Calculate quantization parameters
            if self.config.symmetric:
                # Symmetric quantization
                abs_max = max(abs(min_val), abs(max_val))
                scale = (2 * abs_max) / (2 ** self.config.activation_bits - 1)
                zero_point = 0
            else:
                # Asymmetric quantization
                scale = (max_val - min_val) / (2 ** self.config.activation_bits - 1)
                zero_point = -round(min_val / scale)
            
            quant_params[name] = {
                'scale': scale,
                'zero_point': zero_point,
                'min_val': min_val,
                'max_val': max_val
            }
        
        return quant_params
    
    def quantize_weights(self, model: JigyasaModel) -> JigyasaModel:
        """
        Apply post-training quantization to model weights
        
        Args:
            model: Model to quantize
        Returns:
            Quantized model
        """
        logging.info("Applying weight quantization...")
        
        quantized_model = model
        
        for name, module in quantized_model.named_modules():
            if isinstance(module, nn.Linear):
                # Quantize linear layer weights
                weight = module.weight.data
                
                if self.config.per_channel:
                    # Per-channel quantization (separate scale per output channel)
                    scales = []
                    zero_points = []
                    
                    for i in range(weight.size(0)):
                        channel_weight = weight[i]
                        scale, zero_point = self._compute_weight_quantization_params(
                            channel_weight, self.config.weight_bits
                        )
                        scales.append(scale)
                        zero_points.append(zero_point)
                    
                    scales = torch.tensor(scales, device=weight.device)
                    zero_points = torch.tensor(zero_points, device=weight.device)
                    
                    # Quantize and dequantize
                    quantized_weight = self._quantize_tensor(weight, scales.unsqueeze(-1), zero_points.unsqueeze(-1))
                    
                else:
                    # Per-tensor quantization
                    scale, zero_point = self._compute_weight_quantization_params(
                        weight, self.config.weight_bits
                    )
                    quantized_weight = self._quantize_tensor(weight, scale, zero_point)
                
                # Update module weight
                module.weight.data = quantized_weight
                
                # Store quantization parameters as module attributes
                module.weight_scale = scales if self.config.per_channel else scale
                module.weight_zero_point = zero_points if self.config.per_channel else zero_point
        
        logging.info("Weight quantization completed")
        return quantized_model
    
    def _compute_weight_quantization_params(self, tensor: torch.Tensor, bits: int) -> Tuple[float, int]:
        """Compute quantization scale and zero-point for a tensor"""
        min_val = tensor.min().item()
        max_val = tensor.max().item()
        
        if self.config.symmetric:
            abs_max = max(abs(min_val), abs(max_val))
            scale = (2 * abs_max) / (2 ** bits - 1)
            zero_point = 0
        else:
            scale = (max_val - min_val) / (2 ** bits - 1)
            zero_point = -round(min_val / scale)
        
        return scale, zero_point
    
    def _quantize_tensor(
        self, 
        tensor: torch.Tensor, 
        scale: Union[float, torch.Tensor], 
        zero_point: Union[int, torch.Tensor]
    ) -> torch.Tensor:
        """Quantize and dequantize a tensor"""
        if isinstance(scale, (int, float)):
            scale = torch.tensor(scale, device=tensor.device)
        if isinstance(zero_point, (int, float)):
            zero_point = torch.tensor(zero_point, device=tensor.device)
        
        # Quantize
        quantized = torch.round(tensor / scale + zero_point)
        
        # Clamp to valid range
        if self.config.symmetric:
            q_min = -(2 ** (self.config.weight_bits - 1))
            q_max = 2 ** (self.config.weight_bits - 1) - 1
        else:
            q_min = 0
            q_max = 2 ** self.config.weight_bits - 1
        
        quantized = torch.clamp(quantized, q_min, q_max)
        
        # Dequantize
        dequantized = (quantized - zero_point) * scale
        
        return dequantized


class PTQConverter:
    """
    Post-Training Quantization converter with multiple output formats
    """
    
    def __init__(self, config: QuantizationConfig = None):
        self.config = config or QuantizationConfig()
        self.quantizer = ModelQuantizer(config)
    
    def convert_to_gguf(
        self, 
        model: JigyasaModel, 
        output_path: str,
        calibration_dataloader=None
    ):
        """
        Convert model to GGUF format for llama.cpp compatibility
        
        Args:
            model: Model to convert
            output_path: Output file path
            calibration_dataloader: Data for calibration (optional)
        """
        if calibration_dataloader:
            self.quantizer.calibrate(model, calibration_dataloader)
        
        quantized_model = self.quantizer.quantize_weights(model)
        
        # Convert to GGUF format
        self._write_gguf_file(quantized_model, output_path)
        
        logging.info(f"Model converted to GGUF format: {output_path}")
    
    def _write_gguf_file(self, model: JigyasaModel, output_path: str):
        """
        Write model in GGUF format
        This is a simplified implementation - real GGUF format is more complex
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            # GGUF Magic number
            f.write(b'GGUF')
            
            # Version
            f.write(struct.pack('<I', 3))  # Version 3
            
            # Tensor count
            tensor_count = sum(1 for _ in model.parameters())
            f.write(struct.pack('<Q', tensor_count))
            
            # Metadata count
            f.write(struct.pack('<Q', 0))  # No metadata for now
            
            # Write tensors
            for name, param in model.named_parameters():
                # Tensor name
                name_bytes = name.encode('utf-8')
                f.write(struct.pack('<Q', len(name_bytes)))
                f.write(name_bytes)
                
                # Tensor dimensions
                dims = param.shape
                f.write(struct.pack('<I', len(dims)))
                for dim in dims:
                    f.write(struct.pack('<Q', dim))
                
                # Data type (simplified - using float16)
                f.write(struct.pack('<I', 1))  # GGML_TYPE_F16
                
                # Tensor offset (will be filled later)
                offset_pos = f.tell()
                f.write(struct.pack('<Q', 0))
            
            # Write tensor data
            for name, param in model.named_parameters():
                # Convert to float16 and write
                data = param.detach().cpu().numpy().astype(np.float16)
                f.write(data.tobytes())
    
    def convert_to_onnx(
        self, 
        model: JigyasaModel, 
        output_path: str,
        example_input: Dict[str, torch.Tensor],
        calibration_dataloader=None
    ):
        """
        Convert model to ONNX format
        
        Args:
            model: Model to convert
            output_path: Output file path
            example_input: Example input for tracing
            calibration_dataloader: Data for calibration (optional)
        """
        try:
            import onnx
            import torch.onnx
        except ImportError:
            raise ImportError("ONNX not available. Install with: pip install onnx")
        
        if calibration_dataloader:
            self.quantizer.calibrate(model, calibration_dataloader)
        
        quantized_model = self.quantizer.quantize_weights(model)
        quantized_model.eval()
        
        # Export to ONNX
        torch.onnx.export(
            quantized_model,
            (example_input['input_ids'], example_input.get('attention_mask')),
            output_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'seq_len'},
                'attention_mask': {0: 'batch_size', 1: 'seq_len'},
                'logits': {0: 'batch_size', 1: 'seq_len'}
            }
        )
        
        logging.info(f"Model converted to ONNX format: {output_path}")


class QATTrainer:
    """
    Quantization-Aware Training for better quantized model performance
    """
    
    def __init__(self, model: JigyasaModel, config: QuantizationConfig = None):
        self.model = model
        self.config = config or QuantizationConfig()
        
        # Prepare model for QAT
        self._prepare_model_for_qat()
        
        # Training setup
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        self.training_history = []
    
    def _prepare_model_for_qat(self):
        """Prepare model for quantization-aware training"""
        # Add fake quantization modules
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # Add fake quantization for weights and activations
                module.weight_fake_quant = self._create_fake_quantize(
                    self.config.weight_bits, symmetric=True
                )
                module.activation_fake_quant = self._create_fake_quantize(
                    self.config.activation_bits, symmetric=False
                )
    
    def _create_fake_quantize(self, bits: int, symmetric: bool = True):
        """Create fake quantization function"""
        def fake_quantize(x):
            if symmetric:
                abs_max = x.abs().max()
                scale = (2 * abs_max) / (2 ** bits - 1)
                zero_point = 0
            else:
                min_val = x.min()
                max_val = x.max()
                scale = (max_val - min_val) / (2 ** bits - 1)
                zero_point = -torch.round(min_val / scale)
            
            if scale == 0:
                return x
            
            # Quantize and dequantize
            quantized = torch.round(x / scale + zero_point)
            
            if symmetric:
                q_min = -(2 ** (bits - 1))
                q_max = 2 ** (bits - 1) - 1
            else:
                q_min = 0
                q_max = 2 ** bits - 1
            
            quantized = torch.clamp(quantized, q_min, q_max)
            dequantized = (quantized - zero_point) * scale
            
            return dequantized
        
        return fake_quantize
    
    def train_qat(
        self,
        train_dataloader,
        num_epochs: int = 3,
        eval_dataloader=None
    ) -> Dict[str, Any]:
        """
        Perform quantization-aware training
        
        Args:
            train_dataloader: Training data
            num_epochs: Number of training epochs
            eval_dataloader: Validation data (optional)
        """
        logging.info("Starting quantization-aware training...")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0
            
            for batch in train_dataloader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)
                labels = batch.get('labels', None)
                if labels is not None:
                    labels = labels.to(self.model.device)
                
                # Forward pass with fake quantization
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss'] if 'loss' in outputs else self._compute_loss(outputs, labels)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            avg_loss = epoch_loss / num_batches
            
            # Evaluation
            eval_results = {}
            if eval_dataloader:
                eval_results = self._evaluate_qat(eval_dataloader)
            
            epoch_stats = {
                'epoch': epoch + 1,
                'train_loss': avg_loss,
                **eval_results
            }
            
            self.training_history.append(epoch_stats)
            logging.info(f"QAT Epoch {epoch + 1}: {epoch_stats}")
        
        return {
            'training_history': self.training_history,
            'final_model': self.model
        }
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        """Compute training loss"""
        logits = outputs['logits']
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
    
    def _evaluate_qat(self, eval_dataloader) -> Dict[str, float]:
        """Evaluate model during QAT"""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                input_ids = batch['input_ids'].to(self.model.device)
                attention_mask = batch.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.model.device)
                labels = batch.get('labels', None)
                if labels is not None:
                    labels = labels.to(self.model.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs['loss'] if 'loss' in outputs else self._compute_loss(outputs, labels)
                total_loss += loss.item()
                num_batches += 1
        
        self.model.train()
        
        return {'eval_loss': total_loss / num_batches}
    
    def finalize_quantization(self) -> JigyasaModel:
        """Convert QAT model to actual quantized model"""
        # Remove fake quantization and apply real quantization
        quantizer = ModelQuantizer(self.config)
        return quantizer.quantize_weights(self.model)


# High-level convenience functions
def quantize_model_ptq(
    model: JigyasaModel,
    output_path: str,
    calibration_dataloader=None,
    config: QuantizationConfig = None
) -> str:
    """
    High-level function for post-training quantization
    
    Args:
        model: Model to quantize
        output_path: Output file path
        calibration_dataloader: Calibration data
        config: Quantization configuration
    
    Returns:
        Path to quantized model
    """
    converter = PTQConverter(config)
    
    if output_path.endswith('.gguf'):
        converter.convert_to_gguf(model, output_path, calibration_dataloader)
    elif output_path.endswith('.onnx'):
        # Need example input for ONNX
        example_input = {
            'input_ids': torch.randint(0, 1000, (1, 64)),
            'attention_mask': torch.ones(1, 64)
        }
        converter.convert_to_onnx(model, output_path, example_input, calibration_dataloader)
    else:
        raise ValueError("Unsupported output format. Use .gguf or .onnx")
    
    return output_path


def quantize_model_qat(
    model: JigyasaModel,
    train_dataloader,
    output_path: str,
    config: QuantizationConfig = None,
    num_epochs: int = 3
) -> JigyasaModel:
    """
    High-level function for quantization-aware training
    
    Args:
        model: Model to quantize
        train_dataloader: Training data
        output_path: Output path for final model
        config: Quantization configuration
        num_epochs: Number of QAT epochs
    
    Returns:
        Quantized model
    """
    trainer = QATTrainer(model, config)
    trainer.train_qat(train_dataloader, num_epochs)
    
    quantized_model = trainer.finalize_quantization()
    
    # Save quantized model
    quantized_model.save_pretrained(output_path)
    
    return quantized_model