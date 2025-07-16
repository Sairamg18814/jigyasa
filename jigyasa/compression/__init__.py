"""
Model compression module for Jigyasa
Implements knowledge distillation, pruning, and quantization for on-device deployment
"""

from .distillation import KnowledgeDistillationTrainer, TeacherStudentPair
# from .pruning import StructuredPruner, UnstructuredPruner, PruningScheduler  # Not yet implemented
from .quantization import ModelQuantizer, QATTrainer, PTQConverter
# from .compression_pipeline import CompressionPipeline  # Not yet implemented

__all__ = [
    "KnowledgeDistillationTrainer",
    "TeacherStudentPair",
    # "StructuredPruner",  # Not yet implemented
    # "UnstructuredPruner",  # Not yet implemented
    # "PruningScheduler",  # Not yet implemented
    "ModelQuantizer",
    "QATTrainer",
    "PTQConverter",
    # "CompressionPipeline",  # Not yet implemented
]