"""
Meta-Learning Engine for Jigyasa
Enables learning how to learn and adapting learning strategies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import json
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

from ..core.model import JigyasaModel
from ..config import SEALConfig, ProRLConfig


@dataclass
class LearningExperience:
    """Container for a learning experience"""
    task_type: str
    context: str
    learning_strategy: str
    adaptation_result: Dict[str, Any]
    performance_before: float
    performance_after: float
    learning_efficiency: float
    metadata: Dict[str, Any]


class LearningStrategyPredictor(nn.Module):
    """
    Neural network that predicts the best learning strategy for a given task
    """
    
    def __init__(self, input_dim: int = 768, num_strategies: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.num_strategies = num_strategies
        
        # Strategy prediction network
        self.strategy_predictor = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 4, num_strategies),
            nn.Softmax(dim=-1)
        )
        
        # Learning efficiency predictor
        self.efficiency_predictor = nn.Sequential(
            nn.Linear(input_dim + num_strategies, input_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, task_embedding: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict learning strategy and expected efficiency
        
        Args:
            task_embedding: (batch_size, input_dim) task representation
        Returns:
            strategy_probs: (batch_size, num_strategies) strategy probabilities
            efficiency_pred: (batch_size, 1) predicted learning efficiency
        """
        strategy_probs = self.strategy_predictor(task_embedding)
        
        # Combine task embedding with strategy for efficiency prediction
        combined = torch.cat([task_embedding, strategy_probs], dim=-1)
        efficiency_pred = self.efficiency_predictor(combined)
        
        return strategy_probs, efficiency_pred


class TaskEmbedder(nn.Module):
    """
    Embeds tasks into a representation space for strategy prediction
    """
    
    def __init__(self, model: JigyasaModel, embedding_dim: int = 768):
        super().__init__()
        self.model = model
        self.embedding_dim = embedding_dim
        
        # Task type embeddings
        self.task_type_embeddings = nn.Embedding(20, 64)  # Support 20 task types
        
        # Context encoder
        self.context_encoder = nn.Sequential(
            nn.Linear(getattr(model.config, 'd_model', 768), embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Task difficulty estimator
        self.difficulty_estimator = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
    def forward(
        self,
        context: str,
        task_type_id: int,
        additional_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Embed a task into the representation space
        
        Args:
            context: Task context/description
            task_type_id: Numerical ID for task type
            additional_features: Optional additional features
        Returns:
            task_embedding: Task representation
        """
        # Encode context using the base model
        with torch.no_grad():
            tokenized = self.model.tokenizer.batch_encode([context], return_tensors="pt")
            outputs = self.model(**tokenized)
            context_hidden = outputs['hidden_states'].mean(dim=1)  # Average pooling
        
        # Get task type embedding
        task_type_emb = self.task_type_embeddings(torch.tensor([task_type_id]))
        
        # Encode context
        context_emb = self.context_encoder(context_hidden)
        
        # Combine embeddings
        if additional_features is not None:
            task_embedding = torch.cat([context_emb, task_type_emb, additional_features], dim=-1)
        else:
            task_embedding = torch.cat([context_emb, task_type_emb], dim=-1)
        
        return task_embedding


class MetaLearningEngine:
    """
    Main meta-learning engine that learns optimal learning strategies
    """
    
    def __init__(
        self,
        model: JigyasaModel,
        seal_config: SEALConfig,
        prorl_config: ProRLConfig
    ):
        self.model = model
        self.seal_config = seal_config
        self.prorl_config = prorl_config
        
        # Meta-learning components
        self.task_embedder = TaskEmbedder(model)
        self.strategy_predictor = LearningStrategyPredictor()
        
        # Learning strategies registry
        self.strategies = {
            0: "seal_qa_generation",
            1: "seal_reasoning_chain", 
            2: "seal_concept_integration",
            3: "prorl_mathematical",
            4: "prorl_logical"
        }
        
        # Experience database
        self.learning_experiences = []
        self.strategy_performance = defaultdict(list)
        
        # Meta-learning optimizer
        self.meta_optimizer = torch.optim.AdamW(
            list(self.strategy_predictor.parameters()) + 
            list(self.task_embedder.parameters()),
            lr=1e-4
        )
        
        # Task type mapping
        self.task_types = {
            'qa': 0, 'reasoning': 1, 'mathematical': 2, 'coding': 3,
            'factual': 4, 'creative': 5, 'analytical': 6, 'synthesis': 7,
            'evaluation': 8, 'application': 9, 'comprehension': 10,
            'knowledge': 11, 'problem_solving': 12, 'critical_thinking': 13,
            'pattern_recognition': 14, 'language': 15, 'logic': 16,
            'classification': 17, 'prediction': 18, 'optimization': 19
        }
    
    def predict_optimal_strategy(
        self,
        context: str,
        task_type: str,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> Tuple[str, float]:
        """
        Predict the optimal learning strategy for a given task
        
        Args:
            context: Task context/description
            task_type: Type of task
            additional_context: Additional context information
        Returns:
            strategy_name: Name of predicted optimal strategy
            expected_efficiency: Expected learning efficiency
        """
        # Get task type ID
        task_type_id = self.task_types.get(task_type.lower(), 0)
        
        # Create additional features if provided
        additional_features = None
        if additional_context:
            features = []
            # Add numerical features
            if 'difficulty' in additional_context:
                features.append(additional_context['difficulty'])
            if 'urgency' in additional_context:
                features.append(additional_context['urgency'])
            if 'complexity' in additional_context:
                features.append(additional_context['complexity'])
            
            if features:
                additional_features = torch.tensor([features], dtype=torch.float32)
        
        # Embed task
        task_embedding = self.task_embedder(context, task_type_id, additional_features)
        
        # Predict strategy
        with torch.no_grad():
            strategy_probs, efficiency_pred = self.strategy_predictor(task_embedding)
            
            # Get most likely strategy
            strategy_id = torch.argmax(strategy_probs, dim=-1).item()
            strategy_name = self.strategies[strategy_id]
            expected_efficiency = efficiency_pred.item()
        
        return strategy_name, expected_efficiency
    
    def learn_from_experience(
        self,
        experience: LearningExperience
    ):
        """
        Learn from a learning experience to improve strategy prediction
        """
        # Add experience to database
        self.learning_experiences.append(experience)
        
        # Update strategy performance tracking
        self.strategy_performance[experience.learning_strategy].append(
            experience.learning_efficiency
        )
        
        # Train meta-learning model if we have enough experiences
        if len(self.learning_experiences) >= 10:
            self._update_meta_model()
    
    def _update_meta_model(self):
        """Update the meta-learning model based on experiences"""
        
        # Prepare training data from recent experiences
        recent_experiences = self.learning_experiences[-50:]  # Use last 50 experiences
        
        task_embeddings = []
        strategy_targets = []
        efficiency_targets = []
        
        for exp in recent_experiences:
            # Get task embedding
            task_type_id = self.task_types.get(exp.task_type.lower(), 0)
            task_emb = self.task_embedder(exp.context, task_type_id)
            task_embeddings.append(task_emb)
            
            # Get strategy target
            strategy_id = None
            for sid, sname in self.strategies.items():
                if sname == exp.learning_strategy:
                    strategy_id = sid
                    break
            
            if strategy_id is not None:
                strategy_targets.append(strategy_id)
                efficiency_targets.append(exp.learning_efficiency)
        
        if not strategy_targets:
            return
        
        # Convert to tensors
        task_embeddings = torch.cat(task_embeddings, dim=0)
        strategy_targets = torch.tensor(strategy_targets, dtype=torch.long)
        efficiency_targets = torch.tensor(efficiency_targets, dtype=torch.float32)
        
        # Training step
        self.meta_optimizer.zero_grad()
        
        strategy_probs, efficiency_pred = self.strategy_predictor(task_embeddings)
        
        # Strategy prediction loss (cross-entropy)
        strategy_loss = F.cross_entropy(strategy_probs, strategy_targets)
        
        # Efficiency prediction loss (MSE)
        efficiency_loss = F.mse_loss(efficiency_pred.squeeze(), efficiency_targets)
        
        # Combined loss
        total_loss = strategy_loss + 0.5 * efficiency_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.strategy_predictor.parameters()) + 
            list(self.task_embedder.parameters()),
            1.0
        )
        self.meta_optimizer.step()
    
    def adaptive_learning_session(
        self,
        contexts: List[str],
        task_types: List[str],
        evaluation_fn: callable,
        max_adaptations_per_task: int = 3
    ) -> Dict[str, Any]:
        """
        Conduct an adaptive learning session that adjusts strategies based on performance
        
        Args:
            contexts: List of learning contexts
            task_types: List of task types
            evaluation_fn: Function to evaluate learning performance
            max_adaptations_per_task: Maximum adaptations per task
        Returns:
            session_results: Results of the adaptive learning session
        """
        session_results = {
            'total_tasks': len(contexts),
            'total_adaptations': 0,
            'strategy_usage': defaultdict(int),
            'performance_improvements': [],
            'learning_experiences': []
        }
        
        for context, task_type in zip(contexts, task_types):
            # Get baseline performance
            baseline_performance = evaluation_fn(context, "baseline")
            
            best_performance = baseline_performance
            best_strategy = "baseline"
            
            # Try different strategies and adapt
            for adaptation_round in range(max_adaptations_per_task):
                # Predict optimal strategy
                predicted_strategy, expected_efficiency = self.predict_optimal_strategy(
                    context, task_type
                )
                
                # Apply the strategy (this would integrate with SEAL/ProRL)
                adaptation_result = self._apply_learning_strategy(
                    context, predicted_strategy
                )
                
                # Evaluate performance after adaptation
                post_adaptation_performance = evaluation_fn(context, predicted_strategy)
                
                # Calculate learning efficiency
                learning_efficiency = (post_adaptation_performance - baseline_performance) / (adaptation_round + 1)
                
                # Create learning experience
                experience = LearningExperience(
                    task_type=task_type,
                    context=context,
                    learning_strategy=predicted_strategy,
                    adaptation_result=adaptation_result,
                    performance_before=baseline_performance,
                    performance_after=post_adaptation_performance,
                    learning_efficiency=learning_efficiency,
                    metadata={
                        'adaptation_round': adaptation_round,
                        'expected_efficiency': expected_efficiency
                    }
                )
                
                # Learn from this experience
                self.learn_from_experience(experience)
                session_results['learning_experiences'].append(experience)
                
                # Update tracking
                session_results['total_adaptations'] += 1
                session_results['strategy_usage'][predicted_strategy] += 1
                
                # Check if this is the best strategy so far
                if post_adaptation_performance > best_performance:
                    best_performance = post_adaptation_performance
                    best_strategy = predicted_strategy
                
                # Early stopping if performance is good enough
                if learning_efficiency > 0.8:  # Efficiency threshold
                    break
            
            # Record performance improvement for this task
            improvement = best_performance - baseline_performance
            session_results['performance_improvements'].append(improvement)
        
        # Calculate session statistics
        session_results['average_improvement'] = np.mean(session_results['performance_improvements'])
        session_results['success_rate'] = sum(1 for imp in session_results['performance_improvements'] if imp > 0) / len(session_results['performance_improvements'])
        
        return session_results
    
    def _apply_learning_strategy(
        self,
        context: str,
        strategy: str
    ) -> Dict[str, Any]:
        """
        Apply a specific learning strategy
        This would integrate with SEAL and ProRL trainers
        """
        # Placeholder implementation
        # In practice, this would call the appropriate SEAL or ProRL methods
        
        if strategy.startswith("seal_"):
            # Apply SEAL strategy
            return {
                'strategy_type': 'seal',
                'context_processed': True,
                'adaptations_made': 1,
                'adaptation_loss': 0.1  # Placeholder
            }
        elif strategy.startswith("prorl_"):
            # Apply ProRL strategy
            return {
                'strategy_type': 'prorl',
                'episodes_trained': 5,
                'reward_improvement': 0.2  # Placeholder
            }
        else:
            return {
                'strategy_type': 'baseline',
                'no_adaptation': True
            }
    
    def get_strategy_performance_summary(self) -> Dict[str, Any]:
        """Get summary of strategy performance across all experiences"""
        summary = {}
        
        for strategy, efficiencies in self.strategy_performance.items():
            if efficiencies:
                summary[strategy] = {
                    'mean_efficiency': np.mean(efficiencies),
                    'std_efficiency': np.std(efficiencies),
                    'num_uses': len(efficiencies),
                    'success_rate': sum(1 for eff in efficiencies if eff > 0.5) / len(efficiencies)
                }
        
        return summary
    
    def recommend_next_learning_targets(
        self,
        current_performance: Dict[str, float],
        improvement_threshold: float = 0.1
    ) -> List[Tuple[str, str, float]]:
        """
        Recommend next learning targets based on current performance and meta-learning insights
        
        Args:
            current_performance: Current performance on different task types
            improvement_threshold: Minimum improvement potential to recommend
        Returns:
            recommendations: List of (task_type, recommended_strategy, expected_improvement)
        """
        recommendations = []
        
        # Analyze performance gaps
        for task_type, current_perf in current_performance.items():
            if current_perf < 0.8:  # Room for improvement
                # Predict best strategy for this task type
                dummy_context = f"Task of type {task_type}"
                strategy, expected_efficiency = self.predict_optimal_strategy(
                    dummy_context, task_type
                )
                
                # Estimate potential improvement
                strategy_performance = self.strategy_performance.get(strategy, [0.5])
                avg_strategy_performance = np.mean(strategy_performance)
                
                expected_improvement = avg_strategy_performance - current_perf
                
                if expected_improvement > improvement_threshold:
                    recommendations.append((task_type, strategy, expected_improvement))
        
        # Sort by expected improvement
        recommendations.sort(key=lambda x: x[2], reverse=True)
        
        return recommendations
    
    def export_learning_insights(self) -> Dict[str, Any]:
        """Export insights from meta-learning for analysis"""
        insights = {
            'total_experiences': len(self.learning_experiences),
            'strategy_performance': self.get_strategy_performance_summary(),
            'learning_trends': {},
            'task_type_insights': {}
        }
        
        # Analyze learning trends over time
        if len(self.learning_experiences) > 10:
            recent_efficiencies = [exp.learning_efficiency for exp in self.learning_experiences[-20:]]
            early_efficiencies = [exp.learning_efficiency for exp in self.learning_experiences[:20]]
            
            insights['learning_trends'] = {
                'recent_avg_efficiency': np.mean(recent_efficiencies),
                'early_avg_efficiency': np.mean(early_efficiencies),
                'improvement_over_time': np.mean(recent_efficiencies) - np.mean(early_efficiencies)
            }
        
        # Analyze task type specific insights
        task_type_performance = defaultdict(list)
        for exp in self.learning_experiences:
            task_type_performance[exp.task_type].append(exp.learning_efficiency)
        
        for task_type, efficiencies in task_type_performance.items():
            insights['task_type_insights'][task_type] = {
                'avg_efficiency': np.mean(efficiencies),
                'num_experiences': len(efficiencies),
                'best_efficiency': max(efficiencies),
                'consistency': 1.0 - np.std(efficiencies)  # Higher is more consistent
            }
        
        return insights
    
    def save_meta_state(self, save_path: str):
        """Save meta-learning state"""
        state = {
            'strategy_predictor_state_dict': self.strategy_predictor.state_dict(),
            'task_embedder_state_dict': self.task_embedder.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'learning_experiences': self.learning_experiences,
            'strategy_performance': dict(self.strategy_performance),
            'strategies': self.strategies,
            'task_types': self.task_types
        }
        torch.save(state, save_path)
    
    def load_meta_state(self, load_path: str):
        """Load meta-learning state"""
        state = torch.load(load_path)
        
        self.strategy_predictor.load_state_dict(state['strategy_predictor_state_dict'])
        self.task_embedder.load_state_dict(state['task_embedder_state_dict'])
        self.meta_optimizer.load_state_dict(state['meta_optimizer_state_dict'])
        self.learning_experiences = state['learning_experiences']
        self.strategy_performance = defaultdict(list, state['strategy_performance'])
        self.strategies = state['strategies']
        self.task_types = state['task_types']