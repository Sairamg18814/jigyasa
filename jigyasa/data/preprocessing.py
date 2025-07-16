"""
Data Preprocessing and Quality Control Pipeline
Handles cleaning, filtering, and bias detection for collected data
"""

import re
import string
import unicodedata
from typing import Dict, List, Optional, Tuple, Any, Set
import json
import logging
from dataclasses import dataclass
from pathlib import Path
import hashlib
# import spacy  # Optional dependency
from collections import Counter, defaultdict
import numpy as np
# from presidio_analyzer import AnalyzerEngine  # Optional dependency
# from presidio_anonymizer import AnonymizerEngine  # Optional dependency
# import langdetect  # Optional dependency
# from textstat import flesch_reading_ease, flesch_kincaid_grade  # Optional dependency

from .data_engine import ScrapedContent
from ..config import DataConfig


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


@dataclass
class BiasDetectionResult:
    """Result of bias detection analysis"""
    bias_score: float
    detected_biases: List[str]
    demographic_mentions: Dict[str, int]
    sensitive_topics: List[str]
    recommendations: List[str]


class TextNormalizer:
    """Handles text normalization and cleaning"""
    
    def __init__(self):
        # Common cleaning patterns
        self.cleaning_patterns = [
            (r'\s+', ' '),  # Multiple whitespace to single space
            (r'\n\s*\n', '\n\n'),  # Multiple newlines to double newline
            (r'[^\w\s\.,!?;:\'\"-]', ''),  # Remove special characters
            (r'\.{3,}', '...'),  # Multiple dots to ellipsis
            (r'[!]{2,}', '!'),  # Multiple exclamation marks
            (r'[?]{2,}', '?'),  # Multiple question marks
        ]
        
        # Patterns to remove
        self.removal_patterns = [
            r'\[edit\]',  # Wikipedia edit links
            r'<[^>]+>',  # HTML tags
            r'\{[^}]+\}',  # Template patterns
            r'^\s*\d+\.\s*$',  # Standalone numbers (likely page numbers)
            r'^\s*References?\s*$',  # Reference sections
            r'^\s*Bibliography\s*$',  # Bibliography sections
            r'Cookie Policy|Privacy Policy|Terms of Service',  # Legal text
        ]
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize text by cleaning and standardizing format
        
        Args:
            text: Raw text to normalize
        Returns:
            Normalized text
        """
        if not text:
            return ""
        
        # Unicode normalization
        text = unicodedata.normalize('NFKC', text)
        
        # Remove unwanted patterns
        for pattern in self.removal_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
        
        # Apply cleaning patterns
        for pattern, replacement in self.cleaning_patterns:
            text = re.sub(pattern, replacement, text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Ensure proper sentence spacing
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        
        return text
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract individual sentences from text"""
        # Simple sentence splitting
        sentences = re.split(r'[.!?]+', text)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and len(sentence.split()) > 3:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def remove_duplicates(self, sentences: List[str], threshold: float = 0.8) -> List[str]:
        """Remove duplicate or near-duplicate sentences"""
        unique_sentences = []
        
        for sentence in sentences:
            is_duplicate = False
            sentence_words = set(sentence.lower().split())
            
            for existing in unique_sentences:
                existing_words = set(existing.lower().split())
                
                if sentence_words and existing_words:
                    similarity = len(sentence_words.intersection(existing_words)) / len(sentence_words.union(existing_words))
                    if similarity > threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_sentences.append(sentence)
        
        return unique_sentences


class QualityFilter:
    """Evaluates and filters content based on quality metrics"""
    
    def __init__(self):
        self.min_length = 100
        self.max_length = 50000
        self.min_words = 20
        self.min_sentences = 3
        self.min_readability = 30  # Flesch Reading Ease score
        
        # Language patterns
        self.spam_patterns = [
            r'click here',
            r'buy now',
            r'limited time',
            r'act now',
            r'free trial',
            r'earn money',
            r'make money fast',
            r'weight loss',
            r'get rich quick'
        ]
        
        # Quality indicators
        self.quality_indicators = [
            r'\b(research|study|analysis|evidence|data|findings)\b',
            r'\b(according to|based on|studies show|research indicates)\b',
            r'\b(however|therefore|furthermore|moreover|nevertheless)\b',
            r'\b(conclusion|summary|results|methodology)\b'
        ]
    
    def evaluate_quality(self, content: str, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate content quality across multiple dimensions
        
        Args:
            content: Text content to evaluate
            metadata: Additional metadata about the content
        Returns:
            Quality evaluation results
        """
        results = {
            'overall_score': 0.0,
            'length_score': 0.0,
            'structure_score': 0.0,
            'language_score': 0.0,
            'readability_score': 0.0,
            'spam_score': 0.0,
            'informativeness_score': 0.0,
            'issues': []
        }
        
        if not content:
            results['issues'].append('Empty content')
            return results
        
        # Length evaluation
        length = len(content)
        word_count = len(content.split())
        
        if self.min_length <= length <= self.max_length:
            results['length_score'] = 1.0
        elif length < self.min_length:
            results['length_score'] = length / self.min_length
            results['issues'].append(f'Content too short ({length} chars)')
        else:
            results['length_score'] = self.max_length / length
            results['issues'].append(f'Content too long ({length} chars)')
        
        # Word count check
        if word_count < self.min_words:
            results['issues'].append(f'Too few words ({word_count})')
        
        # Structure evaluation
        sentences = re.split(r'[.!?]+', content)
        paragraphs = content.split('\n\n')
        
        structure_score = 0.0
        if len(sentences) >= self.min_sentences:
            structure_score += 0.3
        if len(paragraphs) >= 2:
            structure_score += 0.3
        if any(len(p.split()) > 20 for p in paragraphs):
            structure_score += 0.2
        if re.search(r'[A-Z][^.!?]*[.!?]', content):  # Proper capitalization
            structure_score += 0.2
        
        results['structure_score'] = structure_score
        
        # Language quality
        language_score = 0.0
        
        # Check for proper grammar patterns
        if re.search(r'\b(the|and|a|an|is|are|was|were)\b', content.lower()):
            language_score += 0.3
        
        # Check for varied vocabulary
        words = content.lower().split()
        unique_words = set(words)
        if len(words) > 0:
            vocab_diversity = len(unique_words) / len(words)
            language_score += min(vocab_diversity * 2, 0.4)
        
        # Check for proper punctuation
        if re.search(r'[.!?]', content):
            language_score += 0.3
        
        results['language_score'] = min(language_score, 1.0)
        
        # Readability evaluation
        try:
            from textstat import flesch_reading_ease
            readability = flesch_reading_ease(content)
            if readability >= self.min_readability:
                results['readability_score'] = min(readability / 100, 1.0)
            else:
                results['readability_score'] = readability / self.min_readability
                results['issues'].append(f'Low readability score ({readability:.1f})')
        except:
            results['readability_score'] = 0.5  # Default if calculation fails
        
        # Spam detection
        spam_count = 0
        for pattern in self.spam_patterns:
            spam_count += len(re.findall(pattern, content, re.IGNORECASE))
        
        results['spam_score'] = max(0, 1.0 - (spam_count * 0.2))
        if spam_count > 0:
            results['issues'].append(f'Potential spam content ({spam_count} indicators)')
        
        # Informativeness evaluation
        quality_indicators = 0
        for pattern in self.quality_indicators:
            quality_indicators += len(re.findall(pattern, content, re.IGNORECASE))
        
        results['informativeness_score'] = min(quality_indicators * 0.1, 1.0)
        
        # Calculate overall score
        weights = {
            'length_score': 0.15,
            'structure_score': 0.20,
            'language_score': 0.20,
            'readability_score': 0.15,
            'spam_score': 0.15,
            'informativeness_score': 0.15
        }
        
        overall_score = sum(results[key] * weight for key, weight in weights.items())
        results['overall_score'] = overall_score
        
        return results
    
    def should_include_content(self, quality_results: Dict[str, Any], threshold: float = 0.6) -> bool:
        """Determine if content meets quality threshold for inclusion"""
        return quality_results['overall_score'] >= threshold and len(quality_results['issues']) <= 2


class PIIDetector:
    """Detects and removes Personally Identifiable Information"""
    
    def __init__(self):
        # Initialize Presidio engines
        try:
            from presidio_analyzer import AnalyzerEngine
            from presidio_anonymizer import AnonymizerEngine
            self.analyzer = AnalyzerEngine()
            self.anonymizer = AnonymizerEngine()
            self.presidio_available = True
        except:
            logging.warning("Presidio not available. Using basic PII detection.")
            self.presidio_available = False
        
        # Basic PII patterns
        self.pii_patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b',
            'ssn': r'\b\d{3}-?\d{2}-?\d{4}\b',
            'credit_card': r'\b(?:\d{4}[-\s]?){3}\d{4}\b',
            'ip_address': r'\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}\b',
            'url': r'https?://[^\s<>"{}|\\^`\[\]]+',
        }
        
        # Name patterns (basic)
        self.name_patterns = [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # First Last
            r'\b[A-Z]\. [A-Z][a-z]+\b',  # F. Last
            r'\b[A-Z][a-z]+ [A-Z]\.\b',  # First L.
        ]
    
    def detect_pii(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect PII in text
        
        Args:
            text: Text to analyze
        Returns:
            List of detected PII entities
        """
        detected_pii = []
        
        if self.presidio_available:
            # Use Presidio for comprehensive PII detection
            results = self.analyzer.analyze(text=text, language='en')
            for result in results:
                detected_pii.append({
                    'type': result.entity_type,
                    'start': result.start,
                    'end': result.end,
                    'confidence': result.score,
                    'text': text[result.start:result.end]
                })
        else:
            # Use basic pattern matching
            for pii_type, pattern in self.pii_patterns.items():
                for match in re.finditer(pattern, text):
                    detected_pii.append({
                        'type': pii_type,
                        'start': match.start(),
                        'end': match.end(),
                        'confidence': 0.8,
                        'text': match.group()
                    })
        
        return detected_pii
    
    def anonymize_text(self, text: str, pii_entities: List[Dict[str, Any]] = None) -> Tuple[str, List[str]]:
        """
        Remove or anonymize PII from text
        
        Args:
            text: Text to anonymize
            pii_entities: Pre-detected PII entities (optional)
        Returns:
            Tuple of (anonymized_text, list_of_removed_pii)
        """
        if pii_entities is None:
            pii_entities = self.detect_pii(text)
        
        removed_pii = []
        anonymized_text = text
        
        # Sort entities by start position (reverse order for proper replacement)
        pii_entities.sort(key=lambda x: x['start'], reverse=True)
        
        for entity in pii_entities:
            pii_text = entity['text']
            pii_type = entity['type'].upper()
            
            # Create anonymized replacement
            if pii_type in ['EMAIL', 'email']:
                replacement = '[EMAIL_REDACTED]'
            elif pii_type in ['PHONE_NUMBER', 'phone']:
                replacement = '[PHONE_REDACTED]'
            elif pii_type in ['PERSON', 'name']:
                replacement = '[NAME_REDACTED]'
            elif pii_type in ['CREDIT_CARD', 'credit_card']:
                replacement = '[CREDIT_CARD_REDACTED]'
            elif pii_type in ['SSN', 'ssn']:
                replacement = '[SSN_REDACTED]'
            else:
                replacement = f'[{pii_type}_REDACTED]'
            
            # Replace in text
            anonymized_text = (
                anonymized_text[:entity['start']] + 
                replacement + 
                anonymized_text[entity['end']:]
            )
            
            removed_pii.append(f"{pii_type}: {pii_text}")
        
        return anonymized_text, removed_pii


class BiasDetector:
    """Detects various forms of bias in text content"""
    
    def __init__(self):
        # Demographic terms for bias detection
        self.demographic_terms = {
            'gender': ['man', 'woman', 'male', 'female', 'boy', 'girl', 'gentleman', 'lady'],
            'race': ['black', 'white', 'asian', 'hispanic', 'latino', 'african', 'european', 'caucasian'],
            'religion': ['christian', 'muslim', 'jewish', 'hindu', 'buddhist', 'catholic', 'protestant'],
            'nationality': ['american', 'chinese', 'indian', 'british', 'german', 'french', 'japanese'],
            'age': ['young', 'old', 'elderly', 'senior', 'teenager', 'millennial', 'boomer'],
            'socioeconomic': ['rich', 'poor', 'wealthy', 'homeless', 'upper class', 'lower class']
        }
        
        # Bias indicator patterns
        self.bias_patterns = {
            'stereotyping': [
                r'all [gender|race|religion|nationality] (are|do|have)',
                r'[demographic] people (always|never|typically)',
                r'typical [demographic]',
                r'[demographic] tend to'
            ],
            'generalization': [
                r'(all|most|many) [demographic]',
                r'[demographic] (always|never|usually)',
                r'it\'s well known that [demographic]'
            ],
            'othering': [
                r'those people',
                r'them vs us',
                r'our kind',
                r'their kind'
            ]
        }
        
        # Sensitive topics that may indicate bias
        self.sensitive_topics = [
            'immigration', 'crime', 'welfare', 'affirmative action', 'discrimination',
            'terrorism', 'violence', 'intelligence', 'education', 'employment'
        ]
    
    def detect_bias(self, text: str) -> BiasDetectionResult:
        """
        Detect potential bias in text content
        
        Args:
            text: Text to analyze for bias
        Returns:
            BiasDetectionResult with bias analysis
        """
        text_lower = text.lower()
        
        # Count demographic mentions
        demographic_mentions = defaultdict(int)
        for category, terms in self.demographic_terms.items():
            for term in terms:
                count = len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower))
                if count > 0:
                    demographic_mentions[category] += count
        
        # Detect bias patterns
        detected_biases = []
        for bias_type, patterns in self.bias_patterns.items():
            for pattern in patterns:
                # Replace [demographic] with actual demographic terms
                expanded_patterns = []
                if '[demographic]' in pattern:
                    for category, terms in self.demographic_terms.items():
                        for term in terms:
                            expanded_pattern = pattern.replace('[demographic]', term)
                            expanded_patterns.append(expanded_pattern)
                else:
                    expanded_patterns.append(pattern)
                
                # Check for pattern matches
                for expanded_pattern in expanded_patterns:
                    if re.search(expanded_pattern, text_lower):
                        detected_biases.append(f"{bias_type}: {expanded_pattern}")
        
        # Check for sensitive topics
        sensitive_topics_found = []
        for topic in self.sensitive_topics:
            if re.search(r'\b' + re.escape(topic) + r'\b', text_lower):
                sensitive_topics_found.append(topic)
        
        # Calculate bias score
        bias_score = 0.0
        
        # Score based on demographic mentions
        total_demographic_mentions = sum(demographic_mentions.values())
        if total_demographic_mentions > 5:
            bias_score += 0.3
        elif total_demographic_mentions > 2:
            bias_score += 0.1
        
        # Score based on detected bias patterns
        bias_score += len(detected_biases) * 0.2
        
        # Score based on sensitive topics
        bias_score += len(sensitive_topics_found) * 0.1
        
        bias_score = min(bias_score, 1.0)
        
        # Generate recommendations
        recommendations = []
        if bias_score > 0.5:
            recommendations.append("Review content for potential bias and stereotyping")
        if total_demographic_mentions > 3:
            recommendations.append("Consider if demographic references are necessary and balanced")
        if detected_biases:
            recommendations.append("Revise language to avoid generalizations and stereotypes")
        if sensitive_topics_found:
            recommendations.append("Ensure sensitive topics are handled objectively and fairly")
        
        return BiasDetectionResult(
            bias_score=bias_score,
            detected_biases=detected_biases,
            demographic_mentions=dict(demographic_mentions),
            sensitive_topics=sensitive_topics_found,
            recommendations=recommendations
        )


class DataPreprocessor:
    """
    Main data preprocessing pipeline coordinator
    """
    
    def __init__(self, config: DataConfig):
        self.config = config
        
        # Initialize components
        self.text_normalizer = TextNormalizer()
        self.quality_filter = QualityFilter()
        self.pii_detector = PIIDetector()
        self.bias_detector = BiasDetector()
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'quality_passed': 0,
            'pii_removed': 0,
            'bias_detected': 0,
            'content_filtered': 0
        }
    
    def process_content(
        self,
        content: ScrapedContent,
        remove_pii: bool = True,
        quality_threshold: float = 0.6,
        bias_threshold: float = 0.7
    ) -> ProcessingResult:
        """
        Process a single piece of content through the full pipeline
        
        Args:
            content: Content to process
            remove_pii: Whether to remove PII
            quality_threshold: Minimum quality score for inclusion
            bias_threshold: Maximum bias score for inclusion
        Returns:
            ProcessingResult with processed content and metadata
        """
        self.stats['total_processed'] += 1
        
        original_text = content.content
        processed_text = original_text
        detected_issues = []
        removed_pii = []
        
        # Step 1: Text normalization
        processed_text = self.text_normalizer.normalize_text(processed_text)
        
        # Step 2: Quality evaluation
        quality_results = self.quality_filter.evaluate_quality(processed_text, content.metadata)
        quality_score = quality_results['overall_score']
        detected_issues.extend(quality_results['issues'])
        
        # Step 3: PII detection and removal
        if remove_pii:
            pii_entities = self.pii_detector.detect_pii(processed_text)
            if pii_entities:
                processed_text, removed_pii = self.pii_detector.anonymize_text(
                    processed_text, pii_entities
                )
                self.stats['pii_removed'] += len(removed_pii)
                detected_issues.append(f"Removed {len(removed_pii)} PII entities")
        
        # Step 4: Bias detection
        bias_results = self.bias_detector.detect_bias(processed_text)
        if bias_results.bias_score > 0.3:
            self.stats['bias_detected'] += 1
            detected_issues.append(f"Bias score: {bias_results.bias_score:.2f}")
        
        # Step 5: Final inclusion decision
        should_include = (
            self.quality_filter.should_include_content(quality_results, quality_threshold) and
            bias_results.bias_score <= bias_threshold and
            len(processed_text.strip()) > 50
        )
        
        if should_include:
            self.stats['quality_passed'] += 1
        else:
            self.stats['content_filtered'] += 1
        
        # Create processing metadata
        processing_metadata = {
            'original_length': len(original_text),
            'processed_length': len(processed_text),
            'quality_results': quality_results,
            'bias_results': {
                'bias_score': bias_results.bias_score,
                'detected_biases': bias_results.detected_biases,
                'demographic_mentions': bias_results.demographic_mentions
            },
            'processing_timestamp': str(datetime.now()),
            'pii_entities_found': len(removed_pii) if remove_pii else 0
        }
        
        return ProcessingResult(
            original_content=original_text,
            processed_content=processed_text,
            quality_score=quality_score,
            detected_issues=detected_issues,
            removed_pii=removed_pii,
            metadata=processing_metadata,
            should_include=should_include
        )
    
    def process_batch(
        self,
        contents: List[ScrapedContent],
        **processing_kwargs
    ) -> List[ProcessingResult]:
        """Process a batch of content"""
        results = []
        
        for content in contents:
            try:
                result = self.process_content(content, **processing_kwargs)
                results.append(result)
            except Exception as e:
                logging.error(f"Failed to process content from {content.url}: {str(e)}")
                
                # Create error result
                error_result = ProcessingResult(
                    original_content=content.content,
                    processed_content="",
                    quality_score=0.0,
                    detected_issues=[f"Processing error: {str(e)}"],
                    removed_pii=[],
                    metadata={'error': str(e)},
                    should_include=False
                )
                results.append(error_result)
        
        return results
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        total = max(self.stats['total_processed'], 1)  # Avoid division by zero
        
        return {
            'total_processed': self.stats['total_processed'],
            'quality_passed': self.stats['quality_passed'],
            'quality_pass_rate': self.stats['quality_passed'] / total,
            'content_filtered': self.stats['content_filtered'],
            'filter_rate': self.stats['content_filtered'] / total,
            'pii_instances_removed': self.stats['pii_removed'],
            'bias_instances_detected': self.stats['bias_detected'],
            'bias_detection_rate': self.stats['bias_detected'] / total
        }
    
    def export_processed_data(
        self,
        processing_results: List[ProcessingResult],
        output_path: str,
        include_filtered: bool = False
    ):
        """Export processed data to file"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        exported_count = 0
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in processing_results:
                if result.should_include or include_filtered:
                    data = {
                        'content': result.processed_content,
                        'quality_score': result.quality_score,
                        'should_include': result.should_include,
                        'detected_issues': result.detected_issues,
                        'metadata': result.metadata
                    }
                    f.write(json.dumps(data, ensure_ascii=False) + '\n')
                    exported_count += 1
        
        logging.info(f"Exported {exported_count} processed content pieces to {output_path}")


# Example usage
def example_processing():
    """Example of using the preprocessing pipeline"""
    from datetime import datetime
    
    # Create sample content
    sample_content = ScrapedContent(
        url="https://example.com/article",
        title="Sample Article",
        content="""
        This is a sample article about machine learning. It contains some information
        about artificial intelligence and its applications. The author, John Smith,
        can be reached at john.smith@email.com or by phone at 555-123-4567.
        
        Machine learning is a powerful technology that has many applications in
        various industries. It's important to understand that all programmers
        should learn Python because it's the most popular language for AI.
        """,
        metadata={'domain': 'example.com'},
        timestamp=datetime.now(),
        content_hash="sample_hash",
        quality_score=0.8
    )
    
    # Initialize preprocessor
    config = DataConfig()
    preprocessor = DataPreprocessor(config)
    
    # Process content
    result = preprocessor.process_content(sample_content)
    
    print(f"Processing Result:")
    print(f"Quality Score: {result.quality_score:.2f}")
    print(f"Should Include: {result.should_include}")
    print(f"Issues: {result.detected_issues}")
    print(f"PII Removed: {result.removed_pii}")
    print(f"Processed Content Length: {len(result.processed_content)}")


if __name__ == "__main__":
    example_processing()