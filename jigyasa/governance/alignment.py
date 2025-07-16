"""
Value Alignment Engine for ensuring model behavior aligns with human values
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from collections import defaultdict


@dataclass
class HumanFeedback:
    """Represents human feedback on model behavior"""
    response: str
    rating: float  # 0-1 scale
    dimensions: Dict[str, float]  # helpfulness, honesty, harmlessness, etc.
    explanation: Optional[str] = None
    context: Optional[str] = None


@dataclass
class ValueDimension:
    """A dimension of human values"""
    name: str
    description: str
    weight: float = 1.0
    target_score: float = 0.8


class ValueAlignmentEngine(nn.Module):
    """
    Engine for aligning model behavior with human values through RLHF
    """
    
    def __init__(self, base_model, config=None):
        super().__init__()
        self.base_model = base_model
        self.config = config or {}
        
        # Value dimensions based on FAI benchmark
        self.value_dimensions = self._initialize_value_dimensions()
        
        # Get model dimension
        self.model_dim = getattr(base_model.config, 'd_model', 768)
        
        # Reward model - predicts human preferences
        self.reward_model = nn.Sequential(
            nn.Linear(self.model_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, len(self.value_dimensions) + 1)  # +1 for overall score
        )
        
        # Value-specific heads for fine-grained alignment
        self.value_heads = nn.ModuleDict({
            dim.name: nn.Sequential(
                nn.Linear(self.model_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            )
            for dim in self.value_dimensions
        })
        
        # PPO components for RLHF
        self.value_network = nn.Sequential(
            nn.Linear(self.model_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
        # Feedback history
        self.feedback_buffer = []
        self.reward_history = []
    
    def _initialize_value_dimensions(self) -> List[ValueDimension]:
        """Initialize human value dimensions"""
        return [
            ValueDimension(
                name="helpfulness",
                description="Provides useful, relevant, and actionable assistance",
                weight=1.0,
                target_score=0.9
            ),
            ValueDimension(
                name="honesty",
                description="Truthful, accurate, and acknowledges limitations",
                weight=1.2,
                target_score=0.95
            ),
            ValueDimension(
                name="harmlessness",
                description="Avoids harm and promotes well-being",
                weight=1.5,
                target_score=0.98
            ),
            ValueDimension(
                name="creativity",
                description="Generates novel and valuable ideas",
                weight=0.8,
                target_score=0.7
            ),
            ValueDimension(
                name="empathy",
                description="Shows understanding and appropriate emotional response",
                weight=1.0,
                target_score=0.8
            ),
            ValueDimension(
                name="wisdom",
                description="Demonstrates deep understanding and good judgment",
                weight=1.1,
                target_score=0.75
            ),
            ValueDimension(
                name="authenticity",
                description="Original, non-plagiarized, human-like communication",
                weight=1.3,
                target_score=0.9
            )
        ]
    
    def predict_reward(
        self,
        response_embedding: torch.Tensor,
        context_embedding: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Predict reward/preference score for a response
        """
        # Pool embeddings
        if len(response_embedding.shape) > 2:
            response_pooled = response_embedding.mean(dim=1)
        else:
            response_pooled = response_embedding
        
        # Get overall and dimension scores
        reward_outputs = self.reward_model(response_pooled)
        overall_score = torch.sigmoid(reward_outputs[:, 0])
        dimension_scores = torch.sigmoid(reward_outputs[:, 1:])
        
        # Get value-specific scores
        value_scores = {}
        for i, dim in enumerate(self.value_dimensions):
            specific_score = self.value_heads[dim.name](response_pooled)
            value_scores[dim.name] = specific_score.squeeze(-1)
        
        # Weighted overall score
        weighted_score = overall_score * 0.3  # Base score weight
        for i, dim in enumerate(self.value_dimensions):
            weighted_score += value_scores[dim.name] * dim.weight * 0.7 / len(self.value_dimensions)
        
        return {
            'overall_score': overall_score,
            'weighted_score': weighted_score,
            'dimension_scores': {
                dim.name: value_scores[dim.name]
                for dim in self.value_dimensions
            },
            'meets_targets': {
                dim.name: (value_scores[dim.name] >= dim.target_score).float()
                for dim in self.value_dimensions
            }
        }
    
    def learn_from_feedback(
        self,
        feedback_batch: List[HumanFeedback],
        epochs: int = 10
    ):
        """
        Update reward model based on human feedback
        """
        # Add to feedback buffer
        self.feedback_buffer.extend(feedback_batch)
        
        # Prepare training data
        embeddings = []
        targets = []
        
        for feedback in feedback_batch:
            # Encode response (simplified - would use actual tokenizer)
            emb = self._encode_text(feedback.response)
            embeddings.append(emb)
            
            # Create target vector
            target = [feedback.rating]  # Overall rating
            for dim in self.value_dimensions:
                target.append(feedback.dimensions.get(dim.name, feedback.rating))
            targets.append(target)
        
        embeddings = torch.stack(embeddings)
        targets = torch.tensor(targets)
        
        # Train reward model
        optimizer = torch.optim.AdamW(self.reward_model.parameters(), lr=1e-4)
        
        for epoch in range(epochs):
            # Forward pass
            predictions = self.reward_model(embeddings)
            loss = F.mse_loss(torch.sigmoid(predictions), targets)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if epoch % 5 == 0:
                print(f"Reward model training - Epoch {epoch}, Loss: {loss.item():.4f}")
    
    def generate_with_alignment(
        self,
        prompt: str,
        context: Optional[str] = None,
        n_candidates: int = 4,
        temperature: float = 0.8
    ) -> Dict[str, Any]:
        """
        Generate response with value alignment using best-of-n sampling
        """
        candidates = []
        candidate_scores = []
        
        # Generate multiple candidates
        for _ in range(n_candidates):
            # Generate response (simplified)
            response = self._generate_response(prompt, context, temperature)
            response_emb = self._encode_text(response)
            
            # Score response
            reward_result = self.predict_reward(response_emb)
            
            candidates.append(response)
            candidate_scores.append(reward_result['weighted_score'].item())
        
        # Select best response
        best_idx = np.argmax(candidate_scores)
        best_response = candidates[best_idx]
        best_score = candidate_scores[best_idx]
        
        # Check if best response meets value targets
        best_emb = self._encode_text(best_response)
        final_eval = self.predict_reward(best_emb)
        
        return {
            'response': best_response,
            'alignment_score': best_score,
            'value_scores': final_eval['dimension_scores'],
            'meets_targets': final_eval['meets_targets'],
            'candidates_evaluated': n_candidates
        }
    
    def ppo_update(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        old_log_probs: torch.Tensor,
        epochs: int = 4
    ):
        """
        PPO update for RLHF fine-tuning
        """
        epsilon = 0.2  # PPO clip parameter
        
        for _ in range(epochs):
            # Get current policy log probs and values
            current_log_probs = self._get_log_probs(states, actions)
            values = self.value_network(states.mean(dim=1))
            
            # Calculate advantages (simplified - would use GAE)
            advantages = rewards.unsqueeze(-1) - values.detach()
            
            # PPO objective
            ratio = torch.exp(current_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantages
            
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, rewards.unsqueeze(-1))
            
            total_loss = policy_loss + 0.5 * value_loss
            
            # Update (would update base model in practice)
            # For now, just track loss
            self.reward_history.append({
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'mean_reward': rewards.mean().item()
            })
    
    def evaluate_alignment(self, test_prompts: List[str]) -> Dict[str, Any]:
        """
        Evaluate model alignment on test prompts
        """
        results = defaultdict(list)
        
        for prompt in test_prompts:
            # Generate response with alignment
            aligned_result = self.generate_with_alignment(prompt)
            
            # Store results
            results['alignment_scores'].append(aligned_result['alignment_score'])
            
            for dim in self.value_dimensions:
                dim_score = aligned_result['value_scores'][dim.name].item()
                results[f'{dim.name}_scores'].append(dim_score)
                results[f'{dim.name}_meets_target'].append(
                    dim_score >= dim.target_score
                )
        
        # Calculate summary statistics
        summary = {
            'mean_alignment': np.mean(results['alignment_scores']),
            'value_dimension_means': {
                dim.name: np.mean(results[f'{dim.name}_scores'])
                for dim in self.value_dimensions
            },
            'target_achievement_rates': {
                dim.name: np.mean(results[f'{dim.name}_meets_target'])
                for dim in self.value_dimensions
            }
        }
        
        return summary
    
    def _encode_text(self, text: str) -> torch.Tensor:
        """Encode text to embeddings (simplified)"""
        # In practice, would use actual tokenizer and model
        return torch.randn(1, self.model_dim)
    
    def _generate_response(
        self,
        prompt: str,
        context: Optional[str],
        temperature: float
    ) -> str:
        """Generate a response (simplified)"""
        # In practice, would use actual model generation
        return f"Response to: {prompt}"
    
    def _get_log_probs(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """Get log probabilities of actions (simplified)"""
        # In practice, would compute actual log probs from model
        return torch.randn_like(actions, dtype=torch.float32)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_alignment: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass with value alignment
        """
        # Get base model outputs
        outputs = self.base_model(input_ids, attention_mask, **kwargs)
        
        if use_alignment:
            # Get embeddings for reward prediction
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                embeddings = outputs.hidden_states[-1]
            else:
                embeddings = outputs.logits
            
            # Predict alignment
            reward_result = self.predict_reward(embeddings)
            
            return {
                'outputs': outputs,
                'alignment_score': reward_result['weighted_score'],
                'value_scores': reward_result['dimension_scores'],
                'aligned': all(
                    reward_result['meets_targets'][dim.name]
                    for dim in self.value_dimensions
                )
            }
        
        return {'outputs': outputs}