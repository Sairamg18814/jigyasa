"""
Simple preprocessing for STEM training data
"""

import json
from typing import Dict, List, Any
from dataclasses import dataclass


@dataclass
class ProcessingResult:
    """Result of data preprocessing"""
    original_content: str
    processed_content: str
    quality_score: float
    detected_issues: List[str]
    removed_pii: List[str]
    metadata: Dict[str, Any]
    should_include: bool


class SimplePreprocessor:
    """Simple preprocessor for STEM data"""
    
    def __init__(self, config=None):
        self.config = config
        self.stats = {
            'total_processed': 0,
            'quality_passed': 0
        }
    
    def process_batch(self, contents: List[Any]) -> List[ProcessingResult]:
        """Process a batch of STEM content"""
        results = []
        
        for content in contents:
            self.stats['total_processed'] += 1
            
            # Handle STEM data format
            if hasattr(content, 'content'):
                try:
                    # Parse JSON content
                    data = json.loads(content.content)
                    
                    # Format as training text
                    formatted_text = self._format_stem_data(data)
                    
                    # Always include STEM training data
                    result = ProcessingResult(
                        original_content=content.content,
                        processed_content=formatted_text,
                        quality_score=getattr(content, 'quality_score', 0.9),
                        detected_issues=[],
                        removed_pii=[],
                        metadata=getattr(content, 'metadata', {}),
                        should_include=True
                    )
                    
                    self.stats['quality_passed'] += 1
                    results.append(result)
                    
                except Exception as e:
                    # If parsing fails, still include with lower quality
                    result = ProcessingResult(
                        original_content=str(content),
                        processed_content=str(content),
                        quality_score=0.5,
                        detected_issues=[str(e)],
                        removed_pii=[],
                        metadata={},
                        should_include=True
                    )
                    results.append(result)
        
        return results
    
    def _format_stem_data(self, data: Dict[str, Any]) -> str:
        """Format STEM data into training text"""
        parts = []
        
        # Handle different data formats
        if 'question' in data:
            parts.append(f"Question: {data['question']}")
            if 'answer' in data:
                parts.append(f"Answer: {data['answer']}")
            if 'reasoning' in data:
                parts.append("Step-by-step solution:")
                for step in data['reasoning']:
                    parts.append(f"- {step}")
        
        elif 'input' in data:
            parts.append(f"User: {data['input']}")
            if 'response' in data:
                parts.append(f"Assistant: {data['response']}")
        
        else:
            # Generic format
            parts.append(json.dumps(data, indent=2))
        
        return "\n".join(parts)
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.stats.copy()


# Alias for compatibility
DataPreprocessor = SimplePreprocessor