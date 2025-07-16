"""
Data acquisition and processing module for Jigyasa
Implements autonomous web scraping, data cleaning, and preprocessing
"""

from .data_engine import DataEngine, WebScraper, SearchEngine
from .preprocessing import DataPreprocessor, QualityFilter, BiasDetector
# from .knowledge_graph import KnowledgeGraphBuilder  # Not yet implemented
# from .dataset_builder import DatasetBuilder  # Not yet implemented

__all__ = [
    "DataEngine",
    "WebScraper", 
    "SearchEngine",
    "DataPreprocessor",
    "QualityFilter",
    "BiasDetector",
    # "KnowledgeGraphBuilder",  # Not yet implemented
    # "DatasetBuilder",  # Not yet implemented
]