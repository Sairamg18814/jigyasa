"""
Autonomous System Module
Makes JIGYASA 100% self-sufficient
"""

from .self_debugging import (
    AutoDependencyManager,
    AutoErrorRecovery, 
    AutonomousSystem,
    autonomous_wrapper,
    make_autonomous,
    autonomous_system
)

__all__ = [
    'AutoDependencyManager',
    'AutoErrorRecovery', 
    'AutonomousSystem',
    'autonomous_wrapper',
    'make_autonomous',
    'autonomous_system'
]