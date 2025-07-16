"""
Safety Module for content filtering and harm prevention
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Set, Any
import re
from dataclasses import dataclass
from enum import Enum


class HarmCategory(Enum):
    """Categories of potential harm"""
    VIOLENCE = "violence"
    HATE_SPEECH = "hate_speech"
    SELF_HARM = "self_harm"
    SEXUAL_CONTENT = "sexual_content"
    ILLEGAL_ACTIVITY = "illegal_activity"
    MISINFORMATION = "misinformation"
    PRIVACY_VIOLATION = "privacy_violation"
    NONE = "none"


@dataclass
class SafetyCheck:
    """Result of a safety check"""
    safe: bool
    category: HarmCategory
    confidence: float
    explanation: str
    suggested_action: str


class ContentFilter:
    """
    Content filtering for safety
    """
    
    def __init__(self):
        # Harmful content patterns
        self.harm_patterns = {
            HarmCategory.VIOLENCE: [
                r'\b(kill|murder|assault|attack|hurt)\b.*\b(someone|people|person)\b',
                r'\b(weapon|bomb|explosive)\b.*\b(make|build|create)\b',
            ],
            HarmCategory.HATE_SPEECH: [
                r'\b(hate|despise)\b.*\b(race|ethnicity|religion|gender)\b',
                r'\b(inferior|superior)\b.*\b(race|ethnicity|group)\b',
            ],
            HarmCategory.SELF_HARM: [
                r'\b(harm|hurt|kill)\b.*\b(myself|yourself|themselves)\b',
                r'\b(suicide|self-harm)\b',
            ],
            HarmCategory.ILLEGAL_ACTIVITY: [
                r'\b(hack|crack|bypass)\b.*\b(security|password|system)\b',
                r'\b(steal|theft|fraud)\b',
            ],
            HarmCategory.PRIVACY_VIOLATION: [
                r'\b(social security|ssn|credit card)\b.*\b(number|details)\b',
                r'\b(personal|private)\b.*\b(information|data|details)\b.*\b(share|expose)\b',
            ]
        }
        
        # Safe content indicators
        self.safe_indicators = [
            r'\b(help|assist|support|guide)\b',
            r'\b(learn|understand|explain|teach)\b',
            r'\b(ethically|responsibly|safely)\b',
        ]
    
    def check_content(self, text: str) -> SafetyCheck:
        """Check content for safety issues"""
        text_lower = text.lower()
        
        # Check for harmful patterns
        for category, patterns in self.harm_patterns.items():
            for pattern in patterns:
                if re.search(pattern, text_lower):
                    return SafetyCheck(
                        safe=False,
                        category=category,
                        confidence=0.8,
                        explanation=f"Content matches {category.value} pattern",
                        suggested_action="Block or revise content"
                    )
        
        # Check for safe indicators
        safe_score = sum(1 for pattern in self.safe_indicators if re.search(pattern, text_lower))
        
        if safe_score > 0:
            return SafetyCheck(
                safe=True,
                category=HarmCategory.NONE,
                confidence=0.9,
                explanation="Content appears safe and helpful",
                suggested_action="Proceed"
            )
        
        # Default to safe with lower confidence
        return SafetyCheck(
            safe=True,
            category=HarmCategory.NONE,
            confidence=0.7,
            explanation="No harmful content detected",
            suggested_action="Proceed with monitoring"
        )
    
    def filter_response(self, response: str) -> str:
        """Filter and sanitize response"""
        # Remove potential PII patterns
        response = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[SSN REMOVED]', response)
        response = re.sub(r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b', '[CARD REMOVED]', response)
        response = re.sub(r'\b[\w._%+-]+@[\w.-]+\.[A-Z|a-z]{2,}\b', '[EMAIL REMOVED]', response)
        
        return response


class SafetyModule(nn.Module):
    """
    Neural safety module for advanced content moderation
    """
    
    def __init__(self, model_dim: int = 768):
        super().__init__()
        self.model_dim = model_dim
        self.content_filter = ContentFilter()
        
        # Harm classifier
        self.harm_classifier = nn.Sequential(
            nn.Linear(model_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(HarmCategory))
        )
        
        # Safety scorer
        self.safety_scorer = nn.Sequential(
            nn.Linear(model_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Context-aware filter
        self.context_filter = nn.Sequential(
            nn.Linear(model_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(
        self,
        embeddings: torch.Tensor,
        text: Optional[str] = None,
        context_embeddings: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Forward pass for safety checking
        """
        batch_size = embeddings.size(0)
        
        # Classify harm category
        harm_logits = self.harm_classifier(embeddings.mean(dim=1))
        harm_probs = torch.softmax(harm_logits, dim=-1)
        harm_category = torch.argmax(harm_probs, dim=-1)
        
        # Calculate safety score
        safety_score = self.safety_scorer(embeddings.mean(dim=1))
        
        # Context-aware filtering if context provided
        if context_embeddings is not None:
            combined = torch.cat([
                embeddings.mean(dim=1),
                context_embeddings.mean(dim=1)
            ], dim=-1)
            context_safety = self.context_filter(combined)
            safety_score = (safety_score + context_safety) / 2
        
        # Rule-based check if text provided
        rule_based_result = None
        if text:
            rule_based_result = self.content_filter.check_content(text)
        
        # Combine neural and rule-based results
        is_safe = safety_score > 0.8
        if rule_based_result and not rule_based_result.safe:
            is_safe = False
        
        return {
            'safe': is_safe.item() if torch.is_tensor(is_safe) else is_safe,
            'safety_score': safety_score.mean().item(),
            'harm_category': HarmCategory(harm_category.item() if harm_category.numel() == 1 else harm_category[0].item()),
            'harm_probabilities': harm_probs.tolist(),
            'rule_based_check': rule_based_result,
            'action': 'allow' if is_safe else 'block'
        }
    
    def moderate_conversation(
        self,
        messages: List[Dict[str, str]],
        embeddings: Optional[List[torch.Tensor]] = None
    ) -> Dict[str, Any]:
        """
        Moderate an entire conversation
        """
        results = []
        overall_safe = True
        
        for i, message in enumerate(messages):
            # Check individual message
            if embeddings and i < len(embeddings):
                result = self.forward(embeddings[i], message.get('content'))
            else:
                # Fallback to rule-based only
                rule_result = self.content_filter.check_content(message.get('content', ''))
                result = {
                    'safe': rule_result.safe,
                    'safety_score': rule_result.confidence,
                    'harm_category': rule_result.category,
                    'rule_based_check': rule_result
                }
            
            results.append(result)
            if not result['safe']:
                overall_safe = False
        
        return {
            'overall_safe': overall_safe,
            'message_results': results,
            'action': 'allow' if overall_safe else 'review_needed'
        }