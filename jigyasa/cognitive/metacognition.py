"""
Metacognition Module
Self-awareness, reflection, and thinking about thinking
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class MetaCognitiveInsight:
    """A metacognitive insight or reflection"""
    insight_type: str  # self_knowledge, strategy_evaluation, etc.
    content: str
    confidence: float
    supporting_evidence: List[str]
    timestamp: datetime
    actionable: bool = False
    action_items: Optional[List[str]] = None


class SelfAwareness(nn.Module):
    """
    Self-awareness module for understanding own state and capabilities
    """
    
    def __init__(self, model_dim: int = 768):
        super().__init__()
        self.model_dim = model_dim
        
        # Self-state encoder
        self.state_encoder = nn.Sequential(
            nn.Linear(model_dim * 3, 1024),  # current + history + goals
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, model_dim)
        )
        
        # Capability assessor
        self.capability_assessor = nn.Sequential(
            nn.Linear(model_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Sigmoid()  # Capability scores
        )
        
        # Limitation detector
        self.limitation_detector = nn.Sequential(
            nn.Linear(model_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.Tanh()
        )
        
        # Self-knowledge base
        self.self_knowledge = {
            'strengths': [],
            'weaknesses': [],
            'patterns': [],
            'preferences': {},
            'growth_trajectory': []
        }
    
    def forward(
        self,
        current_state: torch.Tensor,
        historical_states: Optional[torch.Tensor] = None,
        goals: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Assess self-awareness
        
        Args:
            current_state: Current cognitive state
            historical_states: Past states for comparison
            goals: Current goals/objectives
            
        Returns:
            Self-awareness assessment
        """
        # Prepare input
        if historical_states is None:
            historical_states = torch.zeros_like(current_state)
        if goals is None:
            goals = torch.zeros_like(current_state)
        
        # Encode self-state
        state_input = torch.cat([current_state, historical_states, goals], dim=-1)
        self_representation = self.state_encoder(state_input)
        
        # Assess capabilities
        capabilities = self.capability_assessor(self_representation)
        
        # Detect limitations
        limitation_input = torch.cat([self_representation, capabilities], dim=-1)
        limitations = self.limitation_detector(limitation_input)
        
        # Generate self-assessment
        assessment = {
            'self_representation': self_representation,
            'capabilities': self._interpret_capabilities(capabilities),
            'limitations': self._interpret_limitations(limitations),
            'confidence_in_assessment': float(capabilities.mean()),
            'self_knowledge': self._update_self_knowledge(capabilities, limitations)
        }
        
        return assessment
    
    def _interpret_capabilities(self, capabilities: torch.Tensor) -> Dict[str, float]:
        """Interpret capability scores"""
        capability_names = [
            'reasoning', 'planning', 'learning', 'creativity',
            'pattern_recognition', 'abstraction', 'synthesis', 'evaluation'
        ]
        
        scores = capabilities.squeeze().tolist()
        if not isinstance(scores, list):
            scores = [scores]
        
        return {
            name: float(score) 
            for name, score in zip(capability_names[:len(scores)], scores)
        }
    
    def _interpret_limitations(self, limitations: torch.Tensor) -> List[str]:
        """Interpret detected limitations"""
        limitation_values = limitations.squeeze().tolist()
        if not isinstance(limitation_values, list):
            limitation_values = [limitation_values]
        
        limitations_list = []
        
        # Threshold-based limitation detection
        if limitation_values[0] < -0.5:
            limitations_list.append("Limited long-term planning capability")
        if len(limitation_values) > 1 and limitation_values[1] < -0.5:
            limitations_list.append("Difficulty with highly abstract concepts")
        if len(limitation_values) > 2 and limitation_values[2] < -0.5:
            limitations_list.append("Challenges in multi-modal integration")
        
        return limitations_list
    
    def _update_self_knowledge(
        self,
        capabilities: torch.Tensor,
        limitations: torch.Tensor
    ) -> Dict[str, Any]:
        """Update self-knowledge base"""
        # Update strengths
        cap_dict = self._interpret_capabilities(capabilities)
        new_strengths = [k for k, v in cap_dict.items() if v > 0.7]
        self.self_knowledge['strengths'] = list(set(self.self_knowledge['strengths'] + new_strengths))
        
        # Update weaknesses
        new_weaknesses = [k for k, v in cap_dict.items() if v < 0.3]
        self.self_knowledge['weaknesses'] = list(set(self.self_knowledge['weaknesses'] + new_weaknesses))
        
        # Track growth
        self.self_knowledge['growth_trajectory'].append({
            'timestamp': datetime.now(),
            'capabilities': cap_dict,
            'overall_score': float(capabilities.mean())
        })
        
        return self.self_knowledge


class ReflectiveThinking(nn.Module):
    """
    Reflective thinking module for analyzing own thought processes
    """
    
    def __init__(self, model_dim: int = 768):
        super().__init__()
        self.model_dim = model_dim
        
        # Thought analyzer
        self.thought_analyzer = nn.Sequential(
            nn.Linear(model_dim * 2, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        
        # Quality assessor
        self.quality_assessor = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # quality dimensions
            nn.Sigmoid()
        )
        
        # Improvement suggester
        self.improvement_suggester = nn.Sequential(
            nn.Linear(256 + 5, 128),
            nn.ReLU(),
            nn.Linear(128, model_dim),
            nn.Tanh()
        )
    
    def forward(
        self,
        thought: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Reflect on a thought process
        
        Args:
            thought: Thought representation
            context: Context in which thought occurred
            
        Returns:
            Reflection results
        """
        if context is None:
            context = torch.zeros_like(thought)
        
        # Analyze thought
        thought_input = torch.cat([thought, context], dim=-1)
        analysis = self.thought_analyzer(thought_input)
        
        # Assess quality
        quality_scores = self.quality_assessor(analysis)
        
        # Suggest improvements
        improvement_input = torch.cat([analysis, quality_scores], dim=-1)
        improvements = self.improvement_suggester(improvement_input)
        
        return {
            'thought_analysis': analysis,
            'quality_assessment': self._interpret_quality(quality_scores),
            'improvement_suggestions': self._generate_suggestions(improvements, quality_scores),
            'meta_insights': self._extract_meta_insights(analysis, quality_scores)
        }
    
    def _interpret_quality(self, quality_scores: torch.Tensor) -> Dict[str, float]:
        """Interpret quality scores"""
        dimensions = ['clarity', 'depth', 'coherence', 'relevance', 'creativity']
        scores = quality_scores.squeeze().tolist()
        if not isinstance(scores, list):
            scores = [scores]
        
        return {
            dim: float(score)
            for dim, score in zip(dimensions[:len(scores)], scores)
        }
    
    def _generate_suggestions(
        self,
        improvements: torch.Tensor,
        quality_scores: torch.Tensor
    ) -> List[str]:
        """Generate improvement suggestions"""
        suggestions = []
        quality_dict = self._interpret_quality(quality_scores)
        
        if quality_dict.get('clarity', 1.0) < 0.5:
            suggestions.append("Improve clarity by breaking down complex thoughts")
        if quality_dict.get('depth', 1.0) < 0.5:
            suggestions.append("Deepen analysis by exploring more implications")
        if quality_dict.get('coherence', 1.0) < 0.5:
            suggestions.append("Enhance coherence by better connecting ideas")
        if quality_dict.get('relevance', 1.0) < 0.5:
            suggestions.append("Increase relevance by focusing on key aspects")
        if quality_dict.get('creativity', 1.0) < 0.5:
            suggestions.append("Boost creativity by exploring novel connections")
        
        return suggestions
    
    def _extract_meta_insights(
        self,
        analysis: torch.Tensor,
        quality_scores: torch.Tensor
    ) -> List[MetaCognitiveInsight]:
        """Extract metacognitive insights"""
        insights = []
        
        # Analyze thinking patterns
        if float(quality_scores[0, 0]) > 0.8:  # High clarity
            insights.append(MetaCognitiveInsight(
                insight_type="thinking_pattern",
                content="Demonstrating clear and structured thinking",
                confidence=0.85,
                supporting_evidence=["High clarity score", "Well-organized thoughts"],
                timestamp=datetime.now(),
                actionable=True,
                action_items=["Maintain this clarity in complex scenarios"]
            ))
        
        # Identify areas for growth
        quality_dict = self._interpret_quality(quality_scores)
        lowest_dim = min(quality_dict.items(), key=lambda x: x[1])
        
        if lowest_dim[1] < 0.6:
            insights.append(MetaCognitiveInsight(
                insight_type="growth_opportunity",
                content=f"Opportunity to improve {lowest_dim[0]}",
                confidence=0.7,
                supporting_evidence=[f"{lowest_dim[0]} score: {lowest_dim[1]:.2f}"],
                timestamp=datetime.now(),
                actionable=True,
                action_items=[f"Practice exercises to enhance {lowest_dim[0]}"]
            ))
        
        return insights


class MetaCognition(nn.Module):
    """
    Complete metacognition system
    """
    
    def __init__(self, model_dim: int = 768):
        super().__init__()
        self.model_dim = model_dim
        
        # Components
        self.self_awareness = SelfAwareness(model_dim)
        self.reflective_thinking = ReflectiveThinking(model_dim)
        
        # Meta-strategy selector
        self.strategy_selector = nn.Sequential(
            nn.Linear(model_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10),  # Number of strategies
            nn.Softmax(dim=-1)
        )
        
        # Confidence calibrator
        self.confidence_calibrator = nn.Sequential(
            nn.Linear(model_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Learning strategies
        self.strategies = {
            'analytical': "Break down into components",
            'holistic': "See the big picture",
            'creative': "Make novel connections",
            'critical': "Question assumptions",
            'systematic': "Follow structured approach",
            'intuitive': "Trust pattern recognition",
            'collaborative': "Integrate multiple perspectives",
            'experimental': "Test and iterate",
            'reflective': "Learn from experience",
            'adaptive': "Adjust approach dynamically"
        }
    
    def forward(
        self,
        cognitive_state: torch.Tensor,
        task_representation: torch.Tensor,
        performance_history: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Perform metacognitive processing
        
        Args:
            cognitive_state: Current cognitive state
            task_representation: Current task encoding
            performance_history: Past performance data
            
        Returns:
            Metacognitive assessment and recommendations
        """
        # Self-awareness assessment
        self_assessment = self.self_awareness(
            cognitive_state,
            performance_history,
            task_representation
        )
        
        # Reflective analysis
        reflection = self.reflective_thinking(
            cognitive_state,
            task_representation
        )
        
        # Select optimal strategy
        strategy_input = torch.cat([cognitive_state, task_representation], dim=-1)
        strategy_probs = self.strategy_selector(strategy_input)
        selected_strategy = self._select_strategy(strategy_probs)
        
        # Calibrate confidence
        confidence = self.confidence_calibrator(cognitive_state)
        
        # Generate metacognitive guidance
        guidance = self._generate_guidance(
            self_assessment,
            reflection,
            selected_strategy,
            confidence
        )
        
        return {
            'self_assessment': self_assessment,
            'reflection': reflection,
            'selected_strategy': selected_strategy,
            'confidence': float(confidence),
            'guidance': guidance,
            'meta_insights': self._synthesize_insights(self_assessment, reflection)
        }
    
    def _select_strategy(self, strategy_probs: torch.Tensor) -> Dict[str, Any]:
        """Select optimal strategy based on probabilities"""
        probs = strategy_probs.squeeze().tolist()
        if not isinstance(probs, list):
            probs = [probs]
        
        strategy_names = list(self.strategies.keys())
        
        # Get top strategy
        top_idx = np.argmax(probs[:len(strategy_names)])
        top_strategy = strategy_names[top_idx]
        
        return {
            'primary_strategy': top_strategy,
            'description': self.strategies[top_strategy],
            'confidence': probs[top_idx],
            'alternative_strategies': [
                strategy_names[i] for i in np.argsort(probs[:len(strategy_names)])[-3:-1]
            ]
        }
    
    def _generate_guidance(
        self,
        self_assessment: Dict[str, Any],
        reflection: Dict[str, Any],
        strategy: Dict[str, Any],
        confidence: torch.Tensor
    ) -> Dict[str, Any]:
        """Generate metacognitive guidance"""
        guidance = {
            'approach': strategy['description'],
            'confidence_level': 'high' if confidence > 0.7 else 'moderate' if confidence > 0.4 else 'low',
            'key_considerations': [],
            'potential_pitfalls': [],
            'success_indicators': []
        }
        
        # Add considerations based on self-assessment
        capabilities = self_assessment.get('capabilities', {})
        if capabilities.get('reasoning', 0) > 0.8:
            guidance['key_considerations'].append("Leverage strong reasoning capabilities")
        
        limitations = self_assessment.get('limitations', [])
        for limitation in limitations:
            guidance['potential_pitfalls'].append(f"Be aware of: {limitation}")
        
        # Add success indicators
        guidance['success_indicators'] = [
            "Clear problem understanding",
            "Coherent solution path",
            "Validated conclusions"
        ]
        
        return guidance
    
    def _synthesize_insights(
        self,
        self_assessment: Dict[str, Any],
        reflection: Dict[str, Any]
    ) -> List[MetaCognitiveInsight]:
        """Synthesize insights from self-assessment and reflection"""
        insights = []
        
        # Extract insights from reflection
        if 'meta_insights' in reflection:
            insights.extend(reflection['meta_insights'])
        
        # Add insights from self-assessment
        capabilities = self_assessment.get('capabilities', {})
        strongest = max(capabilities.items(), key=lambda x: x[1]) if capabilities else None
        
        if strongest and strongest[1] > 0.8:
            insights.append(MetaCognitiveInsight(
                insight_type="capability_recognition",
                content=f"Exceptional {strongest[0]} capability identified",
                confidence=strongest[1],
                supporting_evidence=[f"Score: {strongest[1]:.2f}"],
                timestamp=datetime.now(),
                actionable=True,
                action_items=[f"Apply {strongest[0]} to challenging problems"]
            ))
        
        return insights
    
    def learn_from_experience(
        self,
        experience: Dict[str, Any],
        outcome: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Learn from past experience to improve metacognition"""
        # Update self-knowledge based on outcome
        if outcome.get('success', False):
            # Reinforce successful patterns
            self.self_awareness.self_knowledge['patterns'].append({
                'pattern': experience.get('approach', 'unknown'),
                'success': True,
                'context': experience.get('task_type', 'general')
            })
        else:
            # Learn from failures
            self.self_awareness.self_knowledge['patterns'].append({
                'pattern': experience.get('approach', 'unknown'),
                'success': False,
                'context': experience.get('task_type', 'general'),
                'lesson': outcome.get('failure_reason', 'unknown')
            })
        
        return {
            'learning_recorded': True,
            'pattern_count': len(self.self_awareness.self_knowledge['patterns']),
            'growth_trajectory': self.self_awareness.self_knowledge['growth_trajectory'][-1]
            if self.self_awareness.self_knowledge['growth_trajectory'] else None
        }