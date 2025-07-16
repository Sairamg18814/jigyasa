"""
Cognitive module for Jigyasa
Implements SEAL, ProRL, and self-correction mechanisms
"""

from .seal import SEALTrainer, SelfEditGenerator, AdaptationEngine
from .prorl import ProRLTrainer, RewardModel, VerifiableTaskDataset
from .self_correction import SelfCorrectionModule, ChainOfVerification, ReverseCOT
from .meta_learning import MetaLearningEngine

__all__ = [
    "SEALTrainer",
    "SelfEditGenerator", 
    "AdaptationEngine",
    "ProRLTrainer",
    "RewardModel",
    "VerifiableTaskDataset",
    "SelfCorrectionModule",
    "ChainOfVerification",
    "ReverseCOT",
    "MetaLearningEngine",
]