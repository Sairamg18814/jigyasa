"""
JIGYASA Adaptive Training System
Automatically adapts to hardware capabilities and optimizes training parameters
"""

from .hardware_detector import (
    HardwareSpecs, 
    PerformanceMetrics, 
    hardware_detector,
    detect_system_hardware,
    start_hardware_monitoring,
    get_hardware_specs,
    get_performance_metrics
)

from .training_optimizer import (
    TrainingConfig,
    AdaptiveMetrics,
    adaptive_optimizer,
    initialize_adaptive_optimizer,
    get_optimal_training_config,
    adapt_training_during_runtime
)

__all__ = [
    'HardwareSpecs',
    'PerformanceMetrics', 
    'TrainingConfig',
    'AdaptiveMetrics',
    'hardware_detector',
    'adaptive_optimizer',
    'detect_system_hardware',
    'start_hardware_monitoring',
    'get_hardware_specs',
    'get_performance_metrics',
    'initialize_adaptive_optimizer',
    'get_optimal_training_config',
    'adapt_training_during_runtime'
]