"""
Data Processing Module
Handles data preprocessing and transformation
"""

import torch
from typing import Dict, List, Optional, Any
import re
import logging
from datetime import datetime


class DataProcessor:
    """
    Processes raw data for model consumption
    """
    
    def __init__(self):
        self.processing_stats = {
            'total_processed': 0,
            'total_tokens': 0,
            'avg_length': 0
        }
        
        logging.info("Data processor initialized")
    
    def process_batch(
        self,
        data_batch: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Process a batch of data"""
        processed = []
        
        for item in data_batch:
            # Extract text content
            if isinstance(item, dict):
                text = item.get('text', item.get('content', ''))
            else:
                text = str(item)
            
            # Clean text
            cleaned = self._clean_text(text)
            
            # Create processed item
            processed_item = {
                'original': text,
                'cleaned': cleaned,
                'length': len(cleaned),
                'metadata': item.get('metadata', {}) if isinstance(item, dict) else {},
                'timestamp': datetime.now().isoformat()
            }
            
            processed.append(processed_item)
            
            # Update stats
            self.processing_stats['total_processed'] += 1
            self.processing_stats['total_tokens'] += len(cleaned.split())
        
        # Update average length
        if self.processing_stats['total_processed'] > 0:
            self.processing_stats['avg_length'] = (
                self.processing_stats['total_tokens'] / 
                self.processing_stats['total_processed']
            )
        
        return processed
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Normalize case
        text = text.lower()
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """Simple word tokenization"""
        return text.split()
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        return self.processing_stats.copy()