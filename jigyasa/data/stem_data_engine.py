"""
STEM-focused Data Engine
Generates training data without web scraping
"""

import json
import random
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import logging


@dataclass
class STEMTrainingData:
    """Container for STEM training data"""
    content: str
    category: str
    difficulty: str
    metadata: Dict[str, Any]
    timestamp: datetime
    quality_score: float = 1.0


class STEMDataEngine:
    """
    Simplified data engine for STEM training
    Generates data internally without web scraping
    """
    
    def __init__(self, config=None):
        self.logger = logging.getLogger(__name__)
        self.config = config
        
    def acquire_data_for_topic(self, topic: str, max_sources: int = 10) -> List[STEMTrainingData]:
        """
        Generate training data for a given topic
        For STEM training, we generate problems instead of scraping
        """
        from ..cognitive.stem_training import STEMTrainingGenerator, ConversationalTrainer
        
        data_items = []
        
        # Check if this is a STEM-related request
        if any(keyword in topic.lower() for keyword in ['math', 'code', 'science', 'stem', 'problem']):
            # Generate STEM problems
            generator = STEMTrainingGenerator()
            
            for i in range(max_sources):
                # Randomly choose category and difficulty
                category = random.choice(['math', 'coding', 'science'])
                difficulty = random.choice(['basic', 'intermediate', 'advanced'])
                
                if category == 'math':
                    example = generator.generate_math_problem(difficulty)
                elif category == 'coding':
                    example = generator.generate_coding_problem(difficulty)
                else:
                    example = generator.generate_science_problem(difficulty)
                
                # Convert to training data format
                content = {
                    'question': example.question,
                    'answer': example.answer,
                    'reasoning': example.reasoning_steps,
                    'category': example.category,
                    'difficulty': example.difficulty
                }
                
                data_item = STEMTrainingData(
                    content=json.dumps(content),
                    category=example.category,
                    difficulty=example.difficulty,
                    metadata={
                        'source': 'generated',
                        'topic': topic,
                        'index': i
                    },
                    timestamp=datetime.now(),
                    quality_score=0.9
                )
                
                data_items.append(data_item)
        
        elif 'conversation' in topic.lower():
            # Generate conversational examples
            conv_trainer = ConversationalTrainer()
            examples = conv_trainer.generate_conversational_examples(count=max_sources)
            
            for i, example in enumerate(examples):
                content = {
                    'input': example['input'],
                    'response': example['response'],
                    'style': example['style']
                }
                
                data_item = STEMTrainingData(
                    content=json.dumps(content),
                    category='conversation',
                    difficulty='basic',
                    metadata={
                        'source': 'generated',
                        'topic': topic,
                        'index': i
                    },
                    timestamp=datetime.now(),
                    quality_score=0.95
                )
                
                data_items.append(data_item)
        
        else:
            # For non-STEM topics, return empty or minimal data
            self.logger.info(f"Topic '{topic}' not recognized as STEM. Returning minimal data.")
            
            # Create a simple placeholder
            data_item = STEMTrainingData(
                content=json.dumps({
                    'topic': topic,
                    'note': 'This system focuses on STEM and conversational training'
                }),
                category='other',
                difficulty='basic',
                metadata={
                    'source': 'placeholder',
                    'topic': topic
                },
                timestamp=datetime.now(),
                quality_score=0.5
            )
            
            data_items.append(data_item)
        
        self.logger.info(f"Generated {len(data_items)} training items for topic: {topic}")
        return data_items
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Return statistics about generated data"""
        return {
            'engine_type': 'STEM Data Generator',
            'capabilities': ['mathematics', 'coding', 'science', 'conversation'],
            'web_scraping': False,
            'dynamic_generation': True
        }


# Make it compatible with the existing DataEngine interface
class DataEngine(STEMDataEngine):
    """Alias for compatibility"""
    pass