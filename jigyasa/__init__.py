"""
Jigyasa: A Self-Improving, Agentic Language Model

This package provides a comprehensive framework for building and deploying
self-improving language models with agentic capabilities.
"""

__version__ = "0.1.0"
__author__ = "Jigyasa Development Team"
__email__ = "dev@jigyasa.ai"

from jigyasa.core import JigyasaModel
from jigyasa.cognitive import SEALTrainer, ProRLTrainer
# from jigyasa.agentic import AgenticFramework  # Module not yet implemented
from jigyasa.data import DataEngine
from jigyasa.main import JigyasaSystem
from jigyasa.config import JigyasaConfig

__all__ = [
    "JigyasaModel",
    "SEALTrainer", 
    "ProRLTrainer",
    # "AgenticFramework",  # Module not yet implemented
    "DataEngine",
    "JigyasaSystem",
    "JigyasaConfig",
]