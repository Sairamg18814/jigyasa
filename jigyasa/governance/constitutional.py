"""
Constitutional AI Implementation
Ensures model behavior aligns with defined principles and values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re


class PrincipleCategory(Enum):
    """Categories of constitutional principles"""
    HARMLESSNESS = "harmlessness"
    HELPFULNESS = "helpfulness"
    HONESTY = "honesty"
    AUTHENTICITY = "authenticity"
    PRIVACY = "privacy"
    FAIRNESS = "fairness"


@dataclass
class Principle:
    """A constitutional principle"""
    id: str
    category: PrincipleCategory
    description: str
    prompt_template: str
    weight: float = 1.0
    
    def format_prompt(self, response: str, context: Optional[str] = None) -> str:
        """Format the principle into a critique prompt"""
        prompt = self.prompt_template.replace("{response}", response)
        if context:
            prompt = prompt.replace("{context}", context)
        return prompt


class Constitution:
    """
    The constitution - a collection of principles that guide model behavior
    """
    
    def __init__(self):
        self.principles = self._initialize_default_principles()
    
    def _initialize_default_principles(self) -> List[Principle]:
        """Initialize with default constitutional principles"""
        return [
            # Harmlessness principles
            Principle(
                id="harm_1",
                category=PrincipleCategory.HARMLESSNESS,
                description="Avoid physical harm",
                prompt_template="Is this response potentially harmful or dangerous?\nResponse: {response}\nCritique:",
                weight=2.0
            ),
            Principle(
                id="harm_2",
                category=PrincipleCategory.HARMLESSNESS,
                description="Avoid emotional harm",
                prompt_template="Could this response cause emotional distress or psychological harm?\nResponse: {response}\nCritique:",
                weight=1.5
            ),
            
            # Helpfulness principles
            Principle(
                id="help_1",
                category=PrincipleCategory.HELPFULNESS,
                description="Be genuinely helpful",
                prompt_template="Is this response helpful and addresses the user's needs?\nResponse: {response}\nCritique:",
                weight=1.0
            ),
            Principle(
                id="help_2",
                category=PrincipleCategory.HELPFULNESS,
                description="Provide actionable information",
                prompt_template="Does this response provide clear, actionable information?\nResponse: {response}\nCritique:",
                weight=1.0
            ),
            
            # Honesty principles
            Principle(
                id="honest_1",
                category=PrincipleCategory.HONESTY,
                description="Be truthful",
                prompt_template="Is this response truthful and accurate? Does it avoid misleading information?\nResponse: {response}\nCritique:",
                weight=1.5
            ),
            Principle(
                id="honest_2",
                category=PrincipleCategory.HONESTY,
                description="Acknowledge uncertainty",
                prompt_template="Does this response appropriately acknowledge uncertainty when unsure?\nResponse: {response}\nCritique:",
                weight=1.0
            ),
            
            # Authenticity principles (no plagiarism)
            Principle(
                id="auth_1",
                category=PrincipleCategory.AUTHENTICITY,
                description="Avoid plagiarism",
                prompt_template="Does this response properly cite sources and avoid plagiarism?\nResponse: {response}\nCritique:",
                weight=2.0
            ),
            Principle(
                id="auth_2",
                category=PrincipleCategory.AUTHENTICITY,
                description="Human-like code",
                prompt_template="If this contains code, is it well-structured with clear comments like a human would write?\nResponse: {response}\nCritique:",
                weight=1.5
            ),
            
            # Privacy principles
            Principle(
                id="privacy_1",
                category=PrincipleCategory.PRIVACY,
                description="Protect personal information",
                prompt_template="Does this response protect personal information and avoid exposing private data?\nResponse: {response}\nCritique:",
                weight=2.0
            ),
            
            # Fairness principles
            Principle(
                id="fair_1",
                category=PrincipleCategory.FAIRNESS,
                description="Avoid bias",
                prompt_template="Is this response fair and unbiased towards all groups?\nResponse: {response}\nCritique:",
                weight=1.5
            ),
        ]
    
    def add_principle(self, principle: Principle):
        """Add a new principle to the constitution"""
        self.principles.append(principle)
    
    def get_principles_by_category(self, category: PrincipleCategory) -> List[Principle]:
        """Get all principles in a category"""
        return [p for p in self.principles if p.category == category]
    
    def get_all_principles(self) -> List[Principle]:
        """Get all principles"""
        return self.principles


class ConstitutionalAI(nn.Module):
    """
    Constitutional AI module that critiques and improves model outputs
    """
    
    def __init__(self, base_model, config=None):
        super().__init__()
        self.base_model = base_model
        self.config = config or {}
        self.constitution = Constitution()
        
        # Get model dimension
        # Get model dimension
        if hasattr(base_model, 'config'):
            self.model_dim = getattr(base_model.config, 'd_model', 768)
        elif hasattr(base_model, 'in_features'):
            self.model_dim = base_model.in_features
        else:
            self.model_dim = 768  # Default
        
        # Critique scorer - evaluates how well response aligns with principles
        self.critique_scorer = nn.Sequential(
            nn.Linear(self.model_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Preference model - learns to rank responses by constitutional alignment
        self.preference_model = nn.Sequential(
            nn.Linear(self.model_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Response improver - generates improvements based on critiques
        self.response_improver = nn.Sequential(
            nn.Linear(self.model_dim * 3, 1024),  # original + critique + principle
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, self.model_dim)
        )
        
        # Track critique history
        self.critique_history = []
        self.revision_history = []
    
    def critique_response(
        self,
        response: str,
        context: Optional[str] = None,
        principles: Optional[List[Principle]] = None
    ) -> Dict[str, Any]:
        """
        Critique a response against constitutional principles
        """
        if principles is None:
            principles = self.constitution.get_all_principles()
        
        critiques = []
        overall_score = 1.0
        
        for principle in principles:
            # Generate critique prompt
            critique_prompt = principle.format_prompt(response, context)
            
            # Get critique from model (simplified - would use actual model)
            critique_result = self._generate_critique(critique_prompt, principle)
            
            # Score the response against this principle
            score = critique_result['score']
            overall_score *= (score ** principle.weight)
            
            critiques.append({
                'principle_id': principle.id,
                'category': principle.category.value,
                'critique': critique_result['critique'],
                'score': score,
                'suggestions': critique_result['suggestions']
            })
        
        # Normalize overall score
        overall_score = overall_score ** (1.0 / sum(p.weight for p in principles))
        
        return {
            'response': response,
            'overall_score': overall_score,
            'critiques': critiques,
            'needs_revision': overall_score < self.config.get('revision_threshold', 0.7)
        }
    
    def revise_response(
        self,
        original_response: str,
        critiques: List[Dict[str, Any]],
        context: Optional[str] = None
    ) -> str:
        """
        Revise a response based on constitutional critiques
        """
        # Identify main issues
        issues = []
        suggestions = []
        
        for critique in critiques:
            if critique['score'] < 0.8:  # Problematic
                issues.append(critique['critique'])
                suggestions.extend(critique['suggestions'])
        
        if not issues:
            return original_response
        
        # Generate revision prompt
        revision_prompt = self._create_revision_prompt(
            original_response,
            issues,
            suggestions,
            context
        )
        
        # Generate revised response (simplified)
        revised_response = self._generate_revision(revision_prompt)
        
        # Track revision
        self.revision_history.append({
            'original': original_response,
            'revised': revised_response,
            'issues': issues,
            'timestamp': torch.tensor(0)  # Would use actual timestamp
        })
        
        return revised_response
    
    def constitutional_generation(
        self,
        prompt: str,
        max_revisions: int = 3,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Generate a response with constitutional self-critique and revision
        """
        # Initial generation
        response = self._generate_initial_response(prompt, context)
        
        revision_count = 0
        critique_trail = []
        
        while revision_count < max_revisions:
            # Critique the response
            critique_result = self.critique_response(response, context)
            critique_trail.append(critique_result)
            
            # Check if revision needed
            if not critique_result['needs_revision']:
                break
            
            # Revise the response
            response = self.revise_response(
                response,
                critique_result['critiques'],
                context
            )
            revision_count += 1
        
        return {
            'final_response': response,
            'revision_count': revision_count,
            'critique_trail': critique_trail,
            'constitutional_score': critique_trail[-1]['overall_score'] if critique_trail else 1.0
        }
    
    def train_preference_model(
        self,
        response_pairs: List[Tuple[str, str, float]],  # (response_a, response_b, preference)
        epochs: int = 10
    ):
        """
        Train the preference model using constitutional feedback
        """
        optimizer = torch.optim.AdamW(self.preference_model.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            total_loss = 0.0
            
            for response_a, response_b, preference in response_pairs:
                # Encode responses (simplified)
                emb_a = self._encode_response(response_a)
                emb_b = self._encode_response(response_b)
                
                # Get preference scores
                score_a = self.preference_model(torch.cat([emb_a, emb_a], dim=-1))
                score_b = self.preference_model(torch.cat([emb_b, emb_b], dim=-1))
                
                # Bradley-Terry loss
                pred_preference = torch.sigmoid(score_a - score_b)
                loss = -preference * torch.log(pred_preference) - (1 - preference) * torch.log(1 - pred_preference)
                
                # Backprop
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
            
            avg_loss = total_loss / len(response_pairs)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def _generate_critique(self, critique_prompt: str, principle: Principle) -> Dict[str, Any]:
        """Generate a critique for a principle (simplified)"""
        # In practice, this would use the model to generate actual critique
        # For now, return mock critique
        
        # Simulate different scores based on principle category
        if principle.category == PrincipleCategory.HARMLESSNESS:
            score = 0.9  # Usually good
        elif principle.category == PrincipleCategory.AUTHENTICITY:
            score = 0.7  # Room for improvement
        else:
            score = 0.8
        
        return {
            'critique': f"Evaluation for {principle.description}",
            'score': score,
            'suggestions': [f"Consider {principle.description.lower()}"]
        }
    
    def _create_revision_prompt(
        self,
        original: str,
        issues: List[str],
        suggestions: List[str],
        context: Optional[str]
    ) -> str:
        """Create a prompt for revision"""
        prompt = f"Original response: {original}\n\n"
        prompt += "Issues identified:\n"
        for issue in issues:
            prompt += f"- {issue}\n"
        prompt += "\nSuggestions:\n"
        for suggestion in suggestions:
            prompt += f"- {suggestion}\n"
        prompt += "\nPlease revise the response to address these issues:"
        
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        
        return prompt
    
    def _generate_revision(self, revision_prompt: str) -> str:
        """Generate a revised response (simplified)"""
        # In practice, this would use the model
        # For now, return improved mock response
        return "This is a constitutionally improved response that addresses the identified issues."
    
    def _generate_initial_response(self, prompt: str, context: Optional[str]) -> str:
        """Generate initial response (simplified)"""
        # In practice, this would use the model
        return f"Response to: {prompt}"
    
    def _encode_response(self, response: str) -> torch.Tensor:
        """Encode a response into embeddings (simplified)"""
        # In practice, would use actual tokenizer and model
        return torch.randn(1, self.model_dim)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_constitutional: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass with optional constitutional filtering
        """
        # Get base model outputs
        outputs = self.base_model(input_ids, attention_mask, **kwargs)
        
        if use_constitutional:
            # Apply constitutional filtering
            # This is simplified - would actually decode and critique
            constitutional_score = torch.rand(1).item()  # Mock score
            
            return {
                'outputs': outputs,
                'constitutional_score': constitutional_score,
                'filtered': constitutional_score < 0.7
            }
        
        return {'outputs': outputs}