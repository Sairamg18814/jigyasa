#!/usr/bin/env python3
"""
Adaptive Training Optimizer
Automatically adjusts training parameters based on hardware capabilities and performance
"""

import math
import time
import logging
import threading
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import torch
import torch.nn as nn
from pathlib import Path
import json

from .hardware_detector import HardwareSpecs, PerformanceMetrics, hardware_detector

@dataclass
class TrainingConfig:
    """Adaptive training configuration"""
    # Model Parameters
    batch_size: int
    learning_rate: float
    max_sequence_length: int
    gradient_accumulation_steps: int
    
    # Training Optimization
    mixed_precision: bool
    gradient_checkpointing: bool
    dataloader_num_workers: int
    pin_memory: bool
    
    # Hardware Specific
    device: str
    use_multi_gpu: bool
    cpu_threads: int
    memory_limit_gb: float
    
    # Performance Tuning
    optimizer_type: str  # AdamW, SGD, Lion, etc.
    scheduler_type: str
    warmup_steps: int
    
    # Adaptive Features
    auto_scale_batch_size: bool
    auto_adjust_lr: bool
    dynamic_memory_management: bool
    performance_monitoring: bool
    
    # Safety Limits
    max_memory_usage_percent: float
    max_cpu_usage_percent: float
    thermal_throttle_temp: float

@dataclass
class AdaptiveMetrics:
    """Metrics for adaptive training optimization"""
    timestamp: float
    throughput_samples_per_sec: float
    memory_efficiency_mb_per_sample: float
    gpu_utilization_percent: float
    training_stability_score: float
    convergence_speed_factor: float
    
    # Adaptation triggers
    should_increase_batch_size: bool
    should_decrease_batch_size: bool
    should_adjust_learning_rate: bool
    should_enable_optimizations: bool

class AdaptiveTrainingOptimizer:
    """Automatically optimizes training parameters based on hardware and performance"""
    
    def __init__(self, hardware_specs: Optional[HardwareSpecs] = None):
        self.hardware_specs = hardware_specs or hardware_detector.detect_hardware()
        self.training_config = None
        self.baseline_config = None
        self.adaptation_history = []
        
        # Performance tracking
        self.training_start_time = None
        self.samples_processed = 0
        self.adaptation_active = True
        
        # Optimization presets for different hardware classes
        self.optimization_presets = self._load_optimization_presets()
        
        logging.info(f"🎯 Initialized adaptive optimizer for {self.hardware_specs.performance_class} class hardware")
    
    def _load_optimization_presets(self) -> Dict[str, Dict[str, Any]]:
        """Load optimization presets for different hardware classes"""
        return {
            "low": {
                "batch_size": 2,
                "learning_rate": 1e-5,
                "max_sequence_length": 256,
                "gradient_accumulation_steps": 8,
                "mixed_precision": False,
                "gradient_checkpointing": True,
                "dataloader_num_workers": 1,
                "pin_memory": False,
                "use_multi_gpu": False,
                "optimizer_type": "AdamW",
                "scheduler_type": "linear",
                "memory_limit_gb": min(4.0, self.hardware_specs.available_ram * 0.6),
                "max_memory_usage_percent": 60.0,
                "max_cpu_usage_percent": 70.0
            },
            "medium": {
                "batch_size": 8,
                "learning_rate": 3e-5,
                "max_sequence_length": 512,
                "gradient_accumulation_steps": 4,
                "mixed_precision": True,
                "gradient_checkpointing": True,
                "dataloader_num_workers": 2,
                "pin_memory": True,
                "use_multi_gpu": self.hardware_specs.gpu_count > 1,
                "optimizer_type": "AdamW",
                "scheduler_type": "cosine",
                "memory_limit_gb": min(8.0, self.hardware_specs.available_ram * 0.7),
                "max_memory_usage_percent": 70.0,
                "max_cpu_usage_percent": 80.0
            },
            "high": {
                "batch_size": 16,
                "learning_rate": 5e-5,
                "max_sequence_length": 1024,
                "gradient_accumulation_steps": 2,
                "mixed_precision": True,
                "gradient_checkpointing": False,
                "dataloader_num_workers": 4,
                "pin_memory": True,
                "use_multi_gpu": self.hardware_specs.gpu_count > 1,
                "optimizer_type": "Lion",
                "scheduler_type": "cosine_with_restarts",
                "memory_limit_gb": min(16.0, self.hardware_specs.available_ram * 0.8),
                "max_memory_usage_percent": 80.0,
                "max_cpu_usage_percent": 85.0
            },
            "extreme": {
                "batch_size": 32,
                "learning_rate": 1e-4,
                "max_sequence_length": 2048,
                "gradient_accumulation_steps": 1,
                "mixed_precision": True,
                "gradient_checkpointing": False,
                "dataloader_num_workers": 8,
                "pin_memory": True,
                "use_multi_gpu": self.hardware_specs.gpu_count > 1,
                "optimizer_type": "Lion",
                "scheduler_type": "cosine_with_restarts",
                "memory_limit_gb": min(32.0, self.hardware_specs.available_ram * 0.85),
                "max_memory_usage_percent": 85.0,
                "max_cpu_usage_percent": 90.0
            }
        }
    
    def generate_optimal_config(self) -> TrainingConfig:
        """Generate optimal training configuration for current hardware"""
        try:
            logging.info("⚙️ Generating optimal training configuration...")
            
            # Get base preset for hardware class
            preset = self.optimization_presets[self.hardware_specs.performance_class].copy()
            
            # Hardware-specific adjustments
            preset.update(self._adjust_for_cpu())
            preset.update(self._adjust_for_memory())
            preset.update(self._adjust_for_gpu())
            preset.update(self._adjust_for_storage())
            
            # Create training config
            config = TrainingConfig(
                # Core parameters from preset
                batch_size=preset["batch_size"],
                learning_rate=preset["learning_rate"],
                max_sequence_length=preset["max_sequence_length"],
                gradient_accumulation_steps=preset["gradient_accumulation_steps"],
                
                # Optimization settings
                mixed_precision=preset["mixed_precision"],
                gradient_checkpointing=preset["gradient_checkpointing"],
                dataloader_num_workers=preset["dataloader_num_workers"],
                pin_memory=preset["pin_memory"],
                
                # Hardware configuration
                device=self._select_optimal_device(),
                use_multi_gpu=preset["use_multi_gpu"],
                cpu_threads=min(self.hardware_specs.cpu_threads, 8),
                memory_limit_gb=preset["memory_limit_gb"],
                
                # Training configuration
                optimizer_type=preset["optimizer_type"],
                scheduler_type=preset["scheduler_type"],
                warmup_steps=self._calculate_warmup_steps(preset["batch_size"]),
                
                # Adaptive features
                auto_scale_batch_size=True,
                auto_adjust_lr=True,
                dynamic_memory_management=True,
                performance_monitoring=True,
                
                # Safety limits
                max_memory_usage_percent=preset["max_memory_usage_percent"],
                max_cpu_usage_percent=preset["max_cpu_usage_percent"],
                thermal_throttle_temp=85.0
            )
            
            self.training_config = config
            self.baseline_config = config
            
            logging.info(f"✅ Generated optimal config: batch_size={config.batch_size}, lr={config.learning_rate:.2e}")
            return config
            
        except Exception as e:
            logging.error(f"❌ Failed to generate optimal config: {e}")
            return self._create_safe_fallback_config()
    
    def _adjust_for_cpu(self) -> Dict[str, Any]:
        """Adjust configuration based on CPU capabilities"""
        adjustments = {}
        
        # Adjust workers based on CPU cores
        if self.hardware_specs.cpu_cores >= 8:
            adjustments["dataloader_num_workers"] = min(8, self.hardware_specs.cpu_cores // 2)
        elif self.hardware_specs.cpu_cores >= 4:
            adjustments["dataloader_num_workers"] = 2
        else:
            adjustments["dataloader_num_workers"] = 1
        
        # Adjust based on CPU frequency
        if self.hardware_specs.cpu_frequency >= 3.5:
            # High-frequency CPU can handle more
            adjustments["max_cpu_usage_percent"] = min(90.0, adjustments.get("max_cpu_usage_percent", 80.0) + 10)
        elif self.hardware_specs.cpu_frequency <= 2.0:
            # Low-frequency CPU needs more conservative settings
            adjustments["max_cpu_usage_percent"] = max(60.0, adjustments.get("max_cpu_usage_percent", 80.0) - 20)
        
        return adjustments
    
    def _adjust_for_memory(self) -> Dict[str, Any]:
        """Adjust configuration based on memory capabilities"""
        adjustments = {}
        
        # Adjust batch size based on available memory
        available_gb = self.hardware_specs.available_ram
        
        if available_gb >= 32:
            # High memory system
            adjustments["batch_size"] = min(64, adjustments.get("batch_size", 16) * 2)
            adjustments["max_sequence_length"] = 2048
        elif available_gb >= 16:
            # Medium-high memory
            adjustments["batch_size"] = min(32, adjustments.get("batch_size", 8) * 2)
            adjustments["max_sequence_length"] = 1024
        elif available_gb >= 8:
            # Medium memory
            adjustments["max_sequence_length"] = 512
        else:
            # Low memory - conservative settings
            adjustments["batch_size"] = max(1, adjustments.get("batch_size", 4) // 2)
            adjustments["max_sequence_length"] = 256
            adjustments["gradient_checkpointing"] = True
        
        # Memory limit
        adjustments["memory_limit_gb"] = min(available_gb * 0.8, 32.0)
        
        return adjustments
    
    def _adjust_for_gpu(self) -> Dict[str, Any]:
        """Adjust configuration based on GPU capabilities"""
        adjustments = {}
        
        if not self.hardware_specs.has_gpu:
            # CPU-only training
            adjustments["device"] = "cpu"
            adjustments["mixed_precision"] = False
            adjustments["batch_size"] = max(1, adjustments.get("batch_size", 4) // 4)
            adjustments["gradient_accumulation_steps"] = 16
            return adjustments
        
        # GPU available
        total_gpu_memory = sum(self.hardware_specs.gpu_memory)
        
        if total_gpu_memory >= 24:
            # High-end GPU (RTX 4090, A100, etc.)
            adjustments["batch_size"] = min(128, adjustments.get("batch_size", 16) * 4)
            adjustments["mixed_precision"] = True
            adjustments["gradient_checkpointing"] = False
        elif total_gpu_memory >= 12:
            # Mid-high GPU (RTX 4070 Ti, RTX 3080, etc.)
            adjustments["batch_size"] = min(64, adjustments.get("batch_size", 16) * 2)
            adjustments["mixed_precision"] = True
        elif total_gpu_memory >= 8:
            # Mid-range GPU (RTX 4060 Ti, RTX 3070, etc.)
            adjustments["batch_size"] = min(32, adjustments.get("batch_size", 8) * 2)
            adjustments["mixed_precision"] = True
            adjustments["gradient_checkpointing"] = True
        else:
            # Low-end GPU or integrated graphics
            adjustments["batch_size"] = max(2, adjustments.get("batch_size", 8) // 2)
            adjustments["mixed_precision"] = True
            adjustments["gradient_checkpointing"] = True
        
        # Multi-GPU configuration
        if self.hardware_specs.gpu_count > 1:
            adjustments["use_multi_gpu"] = True
            # Scale batch size with GPU count
            adjustments["batch_size"] = adjustments.get("batch_size", 16) * self.hardware_specs.gpu_count
        
        return adjustments
    
    def _adjust_for_storage(self) -> Dict[str, Any]:
        """Adjust configuration based on storage capabilities"""
        adjustments = {}
        
        # Adjust data loading based on storage speed
        if self.hardware_specs.storage_type in ["NVMe", "SSD"]:
            # Fast storage - can use more workers and larger batches
            adjustments["pin_memory"] = True
            if self.hardware_specs.storage_speed >= 1000:  # NVMe
                adjustments["dataloader_num_workers"] = min(12, self.hardware_specs.cpu_cores)
        else:
            # Slower storage - be more conservative
            adjustments["pin_memory"] = False
            adjustments["dataloader_num_workers"] = min(2, adjustments.get("dataloader_num_workers", 4))
        
        return adjustments
    
    def _select_optimal_device(self) -> str:
        """Select the optimal device for training"""
        if self.hardware_specs.has_gpu:
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return "mps"
        return "cpu"
    
    def _calculate_warmup_steps(self, batch_size: int) -> int:
        """Calculate optimal warmup steps based on batch size"""
        # General rule: more warmup for larger batch sizes
        base_warmup = 500
        if batch_size >= 32:
            return base_warmup * 2
        elif batch_size >= 16:
            return int(base_warmup * 1.5)
        else:
            return base_warmup
    
    def _create_safe_fallback_config(self) -> TrainingConfig:
        """Create a safe fallback configuration"""
        return TrainingConfig(
            batch_size=2,
            learning_rate=1e-5,
            max_sequence_length=256,
            gradient_accumulation_steps=8,
            mixed_precision=False,
            gradient_checkpointing=True,
            dataloader_num_workers=1,
            pin_memory=False,
            device="cpu",
            use_multi_gpu=False,
            cpu_threads=2,
            memory_limit_gb=4.0,
            optimizer_type="AdamW",
            scheduler_type="linear",
            warmup_steps=500,
            auto_scale_batch_size=False,
            auto_adjust_lr=False,
            dynamic_memory_management=True,
            performance_monitoring=True,
            max_memory_usage_percent=60.0,
            max_cpu_usage_percent=70.0,
            thermal_throttle_temp=85.0
        )
    
    def adapt_during_training(self, current_metrics: PerformanceMetrics) -> bool:
        """Adapt training parameters based on current performance metrics"""
        if not self.adaptation_active or not self.training_config:
            return False
        
        try:
            adaptations_made = False
            
            # Check if adaptation is needed
            adaptation_metrics = self._analyze_adaptation_needs(current_metrics)
            
            # Memory pressure adaptation
            if current_metrics.memory_usage > self.training_config.max_memory_usage_percent:
                adaptations_made |= self._handle_memory_pressure(current_metrics)
            
            # Performance optimization
            if adaptation_metrics.should_increase_batch_size:
                adaptations_made |= self._increase_batch_size()
            elif adaptation_metrics.should_decrease_batch_size:
                adaptations_made |= self._decrease_batch_size()
            
            # Learning rate adaptation
            if adaptation_metrics.should_adjust_learning_rate:
                adaptations_made |= self._adjust_learning_rate(adaptation_metrics)
            
            # Enable additional optimizations if performance is stable
            if adaptation_metrics.should_enable_optimizations:
                adaptations_made |= self._enable_additional_optimizations()
            
            # Log adaptations
            if adaptations_made:
                self._log_adaptation(adaptation_metrics)
                self.adaptation_history.append({
                    'timestamp': time.time(),
                    'metrics': asdict(adaptation_metrics),
                    'config_after': asdict(self.training_config)
                })
            
            return adaptations_made
            
        except Exception as e:
            logging.error(f"❌ Error during training adaptation: {e}")
            return False
    
    def _analyze_adaptation_needs(self, metrics: PerformanceMetrics) -> AdaptiveMetrics:
        """Analyze current metrics to determine adaptation needs"""
        # Calculate throughput and efficiency
        throughput = metrics.training_speed
        memory_efficiency = metrics.memory_efficiency
        gpu_util = metrics.gpu_usage[0] if metrics.gpu_usage else 0.0
        
        # Determine adaptation needs
        should_increase_batch = (
            metrics.memory_usage < 70 and 
            gpu_util < 80 and 
            throughput > 0 and
            metrics.cpu_usage < 80
        )
        
        should_decrease_batch = (
            metrics.memory_usage > 85 or
            any(temp > 80 for temp in metrics.temperature.values()) or
            metrics.cpu_usage > 90
        )
        
        should_adjust_lr = abs(throughput - getattr(self, '_last_throughput', throughput)) > throughput * 0.2
        
        should_enable_optimizations = (
            throughput > 0 and
            metrics.memory_usage < 60 and
            metrics.cpu_usage < 70 and
            not any(temp > 75 for temp in metrics.temperature.values())
        )
        
        # Training stability score (simplified)
        stability_score = min(1.0, (100 - metrics.memory_usage) / 100 * (100 - metrics.cpu_usage) / 100)
        
        return AdaptiveMetrics(
            timestamp=time.time(),
            throughput_samples_per_sec=throughput,
            memory_efficiency_mb_per_sample=memory_efficiency,
            gpu_utilization_percent=gpu_util,
            training_stability_score=stability_score,
            convergence_speed_factor=1.0,  # Would need loss tracking
            should_increase_batch_size=should_increase_batch,
            should_decrease_batch_size=should_decrease_batch,
            should_adjust_learning_rate=should_adjust_lr,
            should_enable_optimizations=should_enable_optimizations
        )
    
    def _handle_memory_pressure(self, metrics: PerformanceMetrics) -> bool:
        """Handle high memory usage"""
        logging.warning(f"🔥 Memory pressure detected: {metrics.memory_usage:.1f}%")
        
        adaptations = False
        
        # Reduce batch size
        if self.training_config.batch_size > 1:
            new_batch_size = max(1, self.training_config.batch_size // 2)
            self.training_config.batch_size = new_batch_size
            # Compensate with gradient accumulation
            self.training_config.gradient_accumulation_steps *= 2
            adaptations = True
            logging.info(f"🔧 Reduced batch size to {new_batch_size}")
        
        # Enable gradient checkpointing
        if not self.training_config.gradient_checkpointing:
            self.training_config.gradient_checkpointing = True
            adaptations = True
            logging.info("🔧 Enabled gradient checkpointing")
        
        # Reduce sequence length
        if self.training_config.max_sequence_length > 128:
            new_length = max(128, self.training_config.max_sequence_length // 2)
            self.training_config.max_sequence_length = new_length
            adaptations = True
            logging.info(f"🔧 Reduced sequence length to {new_length}")
        
        return adaptations
    
    def _increase_batch_size(self) -> bool:
        """Increase batch size for better throughput"""
        if self.training_config.batch_size < 128:  # Reasonable upper limit
            new_batch_size = min(128, self.training_config.batch_size * 2)
            # Reduce gradient accumulation proportionally
            self.training_config.gradient_accumulation_steps = max(1, self.training_config.gradient_accumulation_steps // 2)
            self.training_config.batch_size = new_batch_size
            logging.info(f"📈 Increased batch size to {new_batch_size}")
            return True
        return False
    
    def _decrease_batch_size(self) -> bool:
        """Decrease batch size to reduce resource usage"""
        if self.training_config.batch_size > 1:
            new_batch_size = max(1, self.training_config.batch_size // 2)
            # Increase gradient accumulation to maintain effective batch size
            self.training_config.gradient_accumulation_steps *= 2
            self.training_config.batch_size = new_batch_size
            logging.info(f"📉 Decreased batch size to {new_batch_size}")
            return True
        return False
    
    def _adjust_learning_rate(self, metrics: AdaptiveMetrics) -> bool:
        """Adjust learning rate based on performance"""
        # Simple adaptive learning rate (could be more sophisticated)
        if metrics.training_stability_score > 0.8:
            # Stable training, can try higher learning rate
            new_lr = min(1e-3, self.training_config.learning_rate * 1.1)
            self.training_config.learning_rate = new_lr
            logging.info(f"📈 Increased learning rate to {new_lr:.2e}")
            return True
        elif metrics.training_stability_score < 0.5:
            # Unstable training, reduce learning rate
            new_lr = max(1e-6, self.training_config.learning_rate * 0.9)
            self.training_config.learning_rate = new_lr
            logging.info(f"📉 Decreased learning rate to {new_lr:.2e}")
            return True
        return False
    
    def _enable_additional_optimizations(self) -> bool:
        """Enable additional optimizations when system is stable"""
        adaptations = False
        
        # Enable mixed precision if not already enabled
        if not self.training_config.mixed_precision and self.hardware_specs.has_gpu:
            self.training_config.mixed_precision = True
            adaptations = True
            logging.info("🔧 Enabled mixed precision training")
        
        # Disable gradient checkpointing if memory allows
        if self.training_config.gradient_checkpointing and self.hardware_specs.total_ram > 16:
            self.training_config.gradient_checkpointing = False
            adaptations = True
            logging.info("🔧 Disabled gradient checkpointing for speed")
        
        return adaptations
    
    def _log_adaptation(self, metrics: AdaptiveMetrics):
        """Log adaptation changes"""
        logging.info(f"🎯 Training adaptation: "
                    f"throughput={metrics.throughput_samples_per_sec:.1f}sps, "
                    f"stability={metrics.training_stability_score:.2f}, "
                    f"batch_size={self.training_config.batch_size}")
    
    def get_current_config(self) -> Optional[TrainingConfig]:
        """Get current training configuration"""
        return self.training_config
    
    def reset_to_baseline(self):
        """Reset configuration to baseline"""
        if self.baseline_config:
            self.training_config = self.baseline_config
            logging.info("🔄 Reset to baseline configuration")
    
    def save_config(self, file_path: str):
        """Save current configuration to file"""
        if self.training_config:
            with open(file_path, 'w') as f:
                json.dump(asdict(self.training_config), f, indent=2)
    
    def load_config(self, file_path: str) -> Optional[TrainingConfig]:
        """Load configuration from file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.training_config = TrainingConfig(**data)
                return self.training_config
        except Exception as e:
            logging.warning(f"Could not load config from {file_path}: {e}")
            return None

# Global adaptive optimizer
adaptive_optimizer = None

def initialize_adaptive_optimizer(hardware_specs: Optional[HardwareSpecs] = None) -> AdaptiveTrainingOptimizer:
    """Initialize the adaptive training optimizer"""
    global adaptive_optimizer
    adaptive_optimizer = AdaptiveTrainingOptimizer(hardware_specs)
    return adaptive_optimizer

def get_optimal_training_config() -> TrainingConfig:
    """Get optimal training configuration for current hardware"""
    global adaptive_optimizer
    if adaptive_optimizer is None:
        adaptive_optimizer = initialize_adaptive_optimizer()
    return adaptive_optimizer.generate_optimal_config()

def adapt_training_during_runtime(metrics: PerformanceMetrics) -> bool:
    """Adapt training parameters during runtime"""
    global adaptive_optimizer
    if adaptive_optimizer is None:
        return False
    return adaptive_optimizer.adapt_during_training(metrics)