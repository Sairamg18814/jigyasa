"""
Creativity Module
Implements creative thinking, concept blending, and novel idea generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Any, Tuple
try:
    from typing import Set
except ImportError:
    Set = set
from dataclasses import dataclass
from datetime import datetime
import numpy as np
from itertools import combinations
import random


@dataclass
class CreativeIdea:
    """Represents a creative idea or insight"""
    id: str
    type: str  # blend, analogy, transformation, emergence
    content: Any
    source_concepts: List[str]
    novelty_score: float
    usefulness_score: float
    surprise_factor: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class ConceptSpace:
    """Represents a conceptual space for creativity"""
    dimensions: List[str]
    concepts: Dict[str, torch.Tensor]  # concept -> embedding
    relationships: Dict[Tuple[str, str], float]  # (concept1, concept2) -> similarity
    boundaries: Dict[str, Tuple[float, float]]  # dimension -> (min, max)


class ConceptBlender(nn.Module):
    """
    Blends concepts to create novel combinations
    """
    
    def __init__(self, model_dim: int = 768):
        super().__init__()
        self.model_dim = model_dim
        
        # Concept encoder
        self.concept_encoder = nn.Sequential(
            nn.Linear(model_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, model_dim),
            nn.Tanh()
        )
        
        # Blend operator
        self.blend_network = nn.Sequential(
            nn.Linear(model_dim * 2, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, model_dim)
        )
        
        # Compatibility scorer
        self.compatibility_scorer = nn.Sequential(
            nn.Linear(model_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Blend modes
        self.blend_modes = {
            'interpolation': self._interpolate_blend,
            'union': self._union_blend,
            'intersection': self._intersection_blend,
            'transformation': self._transformation_blend,
            'emergence': self._emergence_blend
        }
    
    def forward(
        self,
        concept1: torch.Tensor,
        concept2: torch.Tensor,
        mode: str = 'interpolation',
        alpha: float = 0.5
    ) -> Tuple[torch.Tensor, float]:
        """
        Blend two concepts
        
        Args:
            concept1: First concept embedding
            concept2: Second concept embedding
            mode: Blending mode
            alpha: Blending parameter
            
        Returns:
            Blended concept and compatibility score
        """
        # Encode concepts
        enc1 = self.concept_encoder(concept1)
        enc2 = self.concept_encoder(concept2)
        
        # Check compatibility
        compat_input = torch.cat([enc1, enc2], dim=-1)
        compatibility = self.compatibility_scorer(compat_input)
        
        # Blend based on mode
        if mode in self.blend_modes:
            blended = self.blend_modes[mode](enc1, enc2, alpha)
        else:
            blended = self._interpolate_blend(enc1, enc2, alpha)
        
        return blended, float(compatibility)
    
    def _interpolate_blend(
        self,
        concept1: torch.Tensor,
        concept2: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Linear interpolation blend"""
        return alpha * concept1 + (1 - alpha) * concept2
    
    def _union_blend(
        self,
        concept1: torch.Tensor,
        concept2: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Union blend - combines features"""
        combined = torch.cat([concept1, concept2], dim=-1)
        return self.blend_network(combined)
    
    def _intersection_blend(
        self,
        concept1: torch.Tensor,
        concept2: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Intersection blend - common features"""
        # Element-wise minimum (simplified)
        intersection = torch.min(concept1, concept2)
        return self.concept_encoder(intersection)
    
    def _transformation_blend(
        self,
        concept1: torch.Tensor,
        concept2: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Transformation blend - concept1 transformed by concept2"""
        # Use concept2 as transformation operator
        transform = self.blend_network(torch.cat([concept1, concept2], dim=-1))
        return concept1 + alpha * transform
    
    def _emergence_blend(
        self,
        concept1: torch.Tensor,
        concept2: torch.Tensor,
        alpha: float
    ) -> torch.Tensor:
        """Emergence blend - create something new"""
        # Non-linear combination
        combined = torch.cat([concept1, concept2], dim=-1)
        intermediate = self.blend_network(combined)
        
        # Add noise for emergence
        noise = torch.randn_like(intermediate) * 0.1
        emergent = intermediate + noise
        
        # Final transformation
        return self.concept_encoder(emergent)
    
    def multi_blend(
        self,
        concepts: List[torch.Tensor],
        mode: str = 'interpolation'
    ) -> torch.Tensor:
        """Blend multiple concepts"""
        if len(concepts) < 2:
            return concepts[0] if concepts else torch.zeros(1, self.model_dim)
        
        # Progressive blending
        result = concepts[0]
        for concept in concepts[1:]:
            result, _ = self.forward(result, concept, mode)
        
        return result


class NoveltyDetector(nn.Module):
    """
    Detects and measures novelty in ideas
    """
    
    def __init__(self, model_dim: int = 768, memory_size: int = 1000):
        super().__init__()
        self.model_dim = model_dim
        self.memory_size = memory_size
        
        # Novelty scorer
        self.novelty_scorer = nn.Sequential(
            nn.Linear(model_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # novelty, usefulness, surprise
        )
        
        # Memory of past ideas
        self.idea_memory = []
        self.memory_embeddings = None
    
    def forward(
        self,
        idea: torch.Tensor,
        context: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """
        Assess novelty of an idea
        
        Args:
            idea: Idea embedding
            context: Context embedding
            
        Returns:
            Novelty assessment
        """
        if context is None:
            context = torch.zeros_like(idea)
        
        # Compare with memory
        if self.memory_embeddings is not None and len(self.memory_embeddings) > 0:
            similarities = F.cosine_similarity(
                idea.unsqueeze(1),
                self.memory_embeddings.unsqueeze(0),
                dim=2
            )
            max_similarity = similarities.max().item()
            avg_similarity = similarities.mean().item()
        else:
            max_similarity = 0.0
            avg_similarity = 0.0
        
        # Score novelty
        score_input = torch.cat([idea, context], dim=-1)
        scores = self.novelty_scorer(score_input)
        scores = torch.sigmoid(scores).squeeze()
        
        # Adjust novelty based on similarity
        novelty = float(scores[0]) * (1 - max_similarity)
        
        return {
            'novelty': novelty,
            'usefulness': float(scores[1]),
            'surprise': float(scores[2]),
            'max_similarity': max_similarity,
            'avg_similarity': avg_similarity,
            'is_novel': novelty > 0.7
        }
    
    def add_to_memory(self, idea: torch.Tensor):
        """Add idea to memory"""
        self.idea_memory.append(idea)
        
        # Maintain memory size
        if len(self.idea_memory) > self.memory_size:
            self.idea_memory.pop(0)
        
        # Update embeddings
        if self.idea_memory:
            self.memory_embeddings = torch.stack(self.idea_memory)


class CreativeEngine(nn.Module):
    """
    Complete creative thinking system
    """
    
    def __init__(self, model_dim: int = 768):
        super().__init__()
        self.model_dim = model_dim
        
        # Components
        self.concept_blender = ConceptBlender(model_dim)
        self.novelty_detector = NoveltyDetector(model_dim)
        
        # Inspiration generator
        self.inspiration_generator = nn.Sequential(
            nn.Linear(model_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 768),
            nn.ReLU(),
            nn.Linear(768, model_dim),
            nn.Tanh()
        )
        
        # Metaphor creator
        self.metaphor_network = nn.Sequential(
            nn.Linear(model_dim * 2, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Linear(512, model_dim)
        )
        
        # Creative strategies
        self.strategies = {
            'brainstorm': self._brainstorm_strategy,
            'analogical': self._analogical_strategy,
            'transformational': self._transformational_strategy,
            'combinatorial': self._combinatorial_strategy,
            'emergent': self._emergent_strategy
        }
        
        # Creative idea history
        self.idea_history = []
        
        # Concept space
        self.concept_space = ConceptSpace(
            dimensions=['abstract', 'concrete', 'emotional', 'logical', 'temporal'],
            concepts={},
            relationships={},
            boundaries={}
        )
    
    def forward(
        self,
        prompt: torch.Tensor,
        strategy: str = 'brainstorm',
        num_ideas: int = 5,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[CreativeIdea]:
        """
        Generate creative ideas
        
        Args:
            prompt: Input prompt embedding
            strategy: Creative strategy to use
            num_ideas: Number of ideas to generate
            constraints: Optional constraints
            
        Returns:
            List of creative ideas
        """
        if strategy in self.strategies:
            ideas = self.strategies[strategy](prompt, num_ideas, constraints)
        else:
            ideas = self._brainstorm_strategy(prompt, num_ideas, constraints)
        
        # Filter and rank ideas
        ideas = self._filter_ideas(ideas, constraints)
        ideas = self._rank_ideas(ideas)
        
        # Store in history
        self.idea_history.extend(ideas[:num_ideas])
        
        return ideas[:num_ideas]
    
    def _brainstorm_strategy(
        self,
        prompt: torch.Tensor,
        num_ideas: int,
        constraints: Optional[Dict[str, Any]]
    ) -> List[CreativeIdea]:
        """Brainstorming strategy - generate diverse ideas"""
        ideas = []
        
        for i in range(num_ideas * 2):  # Generate extra for filtering
            # Add random inspiration
            noise = torch.randn_like(prompt) * 0.3
            inspired = self.inspiration_generator(prompt + noise)
            
            # Assess novelty
            novelty_scores = self.novelty_detector(inspired, prompt)
            
            # Create idea
            idea = CreativeIdea(
                id=f"idea_{len(self.idea_history) + i}",
                type="brainstorm",
                content=inspired,
                source_concepts=["prompt", "random_inspiration"],
                novelty_score=novelty_scores['novelty'],
                usefulness_score=novelty_scores['usefulness'],
                surprise_factor=novelty_scores['surprise'],
                timestamp=datetime.now(),
                metadata={'strategy': 'brainstorm'}
            )
            
            ideas.append(idea)
            
            # Add to novelty detector memory
            self.novelty_detector.add_to_memory(inspired)
        
        return ideas
    
    def _analogical_strategy(
        self,
        prompt: torch.Tensor,
        num_ideas: int,
        constraints: Optional[Dict[str, Any]]
    ) -> List[CreativeIdea]:
        """Analogical thinking - find analogies"""
        ideas = []
        
        # Generate source domains
        source_domains = []
        for _ in range(5):
            domain = self.inspiration_generator(torch.randn(1, self.model_dim))
            source_domains.append(domain)
        
        for i in range(num_ideas):
            # Select random source domain
            source = random.choice(source_domains)
            
            # Create metaphor
            metaphor = self.metaphor_network(torch.cat([prompt, source], dim=-1))
            
            # Assess novelty
            novelty_scores = self.novelty_detector(metaphor, prompt)
            
            idea = CreativeIdea(
                id=f"idea_{len(self.idea_history) + i}",
                type="analogy",
                content=metaphor,
                source_concepts=["prompt", "analogy_source"],
                novelty_score=novelty_scores['novelty'],
                usefulness_score=novelty_scores['usefulness'],
                surprise_factor=novelty_scores['surprise'],
                timestamp=datetime.now(),
                metadata={
                    'strategy': 'analogical',
                    'source_domain': 'abstract'  # Would identify actual domain
                }
            )
            
            ideas.append(idea)
        
        return ideas
    
    def _transformational_strategy(
        self,
        prompt: torch.Tensor,
        num_ideas: int,
        constraints: Optional[Dict[str, Any]]
    ) -> List[CreativeIdea]:
        """Transformational creativity - break constraints"""
        ideas = []
        
        # Define transformations
        transformations = [
            lambda x: -x,  # Inversion
            lambda x: x * 2.0,  # Amplification
            lambda x: x * 0.1,  # Reduction
            lambda x: torch.roll(x, shifts=10, dims=-1),  # Shift
            lambda x: x + torch.randn_like(x),  # Randomization
        ]
        
        for i in range(num_ideas):
            # Apply random transformation
            transform = random.choice(transformations)
            transformed = transform(prompt)
            
            # Blend with original
            blended, compatibility = self.concept_blender(
                prompt,
                transformed,
                mode='transformation'
            )
            
            # Assess novelty
            novelty_scores = self.novelty_detector(blended, prompt)
            
            idea = CreativeIdea(
                id=f"idea_{len(self.idea_history) + i}",
                type="transformation",
                content=blended,
                source_concepts=["prompt", "transformed"],
                novelty_score=novelty_scores['novelty'],
                usefulness_score=novelty_scores['usefulness'] * compatibility,
                surprise_factor=novelty_scores['surprise'],
                timestamp=datetime.now(),
                metadata={
                    'strategy': 'transformational',
                    'transformation': transform.__name__
                }
            )
            
            ideas.append(idea)
        
        return ideas
    
    def _combinatorial_strategy(
        self,
        prompt: torch.Tensor,
        num_ideas: int,
        constraints: Optional[Dict[str, Any]]
    ) -> List[CreativeIdea]:
        """Combinatorial creativity - combine existing concepts"""
        ideas = []
        
        # Get concepts from space
        if len(self.concept_space.concepts) < 3:
            # Generate some concepts
            for i in range(10):
                concept = self.inspiration_generator(torch.randn(1, self.model_dim))
                self.concept_space.concepts[f"concept_{i}"] = concept
        
        concept_list = list(self.concept_space.concepts.values())
        
        for i in range(num_ideas):
            # Select random concepts to combine
            num_concepts = random.randint(2, 4)
            selected = random.sample(concept_list, min(num_concepts, len(concept_list)))
            
            # Add prompt
            selected.append(prompt)
            
            # Multi-blend
            blended = self.concept_blender.multi_blend(selected, mode='union')
            
            # Assess novelty
            novelty_scores = self.novelty_detector(blended, prompt)
            
            idea = CreativeIdea(
                id=f"idea_{len(self.idea_history) + i}",
                type="blend",
                content=blended,
                source_concepts=[f"concept_{j}" for j in range(len(selected))],
                novelty_score=novelty_scores['novelty'],
                usefulness_score=novelty_scores['usefulness'],
                surprise_factor=novelty_scores['surprise'],
                timestamp=datetime.now(),
                metadata={
                    'strategy': 'combinatorial',
                    'num_combined': len(selected)
                }
            )
            
            ideas.append(idea)
        
        return ideas
    
    def _emergent_strategy(
        self,
        prompt: torch.Tensor,
        num_ideas: int,
        constraints: Optional[Dict[str, Any]]
    ) -> List[CreativeIdea]:
        """Emergent creativity - let ideas emerge"""
        ideas = []
        
        for i in range(num_ideas):
            # Start with prompt
            current = prompt
            
            # Iterative emergence
            for step in range(random.randint(3, 7)):
                # Generate variation
                variation = self.inspiration_generator(current)
                
                # Blend with emergence
                current, _ = self.concept_blender(
                    current,
                    variation,
                    mode='emergence',
                    alpha=0.7
                )
            
            # Assess novelty
            novelty_scores = self.novelty_detector(current, prompt)
            
            idea = CreativeIdea(
                id=f"idea_{len(self.idea_history) + i}",
                type="emergence",
                content=current,
                source_concepts=["prompt", "emergent_process"],
                novelty_score=novelty_scores['novelty'],
                usefulness_score=novelty_scores['usefulness'],
                surprise_factor=novelty_scores['surprise'],
                timestamp=datetime.now(),
                metadata={
                    'strategy': 'emergent',
                    'emergence_steps': step + 1
                }
            )
            
            ideas.append(idea)
        
        return ideas
    
    def _filter_ideas(
        self,
        ideas: List[CreativeIdea],
        constraints: Optional[Dict[str, Any]]
    ) -> List[CreativeIdea]:
        """Filter ideas based on constraints"""
        if not constraints:
            return ideas
        
        filtered = []
        for idea in ideas:
            # Check constraints
            if 'min_novelty' in constraints and idea.novelty_score < constraints['min_novelty']:
                continue
            if 'min_usefulness' in constraints and idea.usefulness_score < constraints['min_usefulness']:
                continue
            if 'required_type' in constraints and idea.type != constraints['required_type']:
                continue
            
            filtered.append(idea)
        
        return filtered
    
    def _rank_ideas(self, ideas: List[CreativeIdea]) -> List[CreativeIdea]:
        """Rank ideas by combined score"""
        for idea in ideas:
            # Combined score
            idea.combined_score = (
                0.4 * idea.novelty_score +
                0.4 * idea.usefulness_score +
                0.2 * idea.surprise_factor
            )
        
        return sorted(ideas, key=lambda x: x.combined_score, reverse=True)
    
    def create_metaphor(
        self,
        source: torch.Tensor,
        target: torch.Tensor
    ) -> Dict[str, Any]:
        """Create a metaphor between source and target"""
        # Generate metaphorical mapping
        metaphor = self.metaphor_network(torch.cat([source, target], dim=-1))
        
        # Assess quality
        novelty_scores = self.novelty_detector(metaphor)
        
        return {
            'metaphor': metaphor,
            'source': source,
            'target': target,
            'quality': novelty_scores['novelty'] * novelty_scores['usefulness'],
            'mapping_strength': F.cosine_similarity(metaphor, target, dim=-1).item()
        }
    
    def explore_concept_space(
        self,
        starting_point: torch.Tensor,
        steps: int = 10,
        step_size: float = 0.1
    ) -> List[torch.Tensor]:
        """Explore the concept space creatively"""
        trajectory = [starting_point]
        current = starting_point
        
        for _ in range(steps):
            # Generate direction
            direction = self.inspiration_generator(current)
            direction = F.normalize(direction, dim=-1)
            
            # Take step
            current = current + step_size * direction
            
            # Ensure within bounds (simplified)
            current = torch.tanh(current)
            
            trajectory.append(current)
        
        return trajectory