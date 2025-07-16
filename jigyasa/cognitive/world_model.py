"""
World Model
Internal representation of the world for reasoning and prediction
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
try:
    from typing import Set
except ImportError:
    Set = set
from dataclasses import dataclass
from datetime import datetime
import networkx as nx
import numpy as np


@dataclass
class Entity:
    """Represents an entity in the world model"""
    id: str
    type: str  # person, object, concept, etc.
    properties: Dict[str, Any]
    relationships: Dict[str, List[str]]  # relation_type -> [entity_ids]
    embedding: Optional[torch.Tensor] = None
    last_updated: datetime = None
    confidence: float = 1.0


@dataclass
class Belief:
    """Represents a belief about the world"""
    id: str
    content: str
    entities: List[str]  # Entity IDs involved
    confidence: float
    evidence: List[str]
    contradictions: List[str]
    timestamp: datetime


@dataclass
class CausalRelation:
    """Represents a causal relationship"""
    cause: str
    effect: str
    strength: float
    conditions: List[str]
    exceptions: List[str]
    evidence_count: int


class BeliefSystem(nn.Module):
    """
    Manages beliefs about the world
    """
    
    def __init__(self, model_dim: int = 768):
        super().__init__()
        self.model_dim = model_dim
        
        # Belief encoder
        self.belief_encoder = nn.Sequential(
            nn.Linear(model_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, model_dim)
        )
        
        # Consistency checker
        self.consistency_checker = nn.Sequential(
            nn.Linear(model_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Belief store
        self.beliefs: Dict[str, Belief] = {}
        self.belief_graph = nx.DiGraph()
    
    def add_belief(
        self,
        content: str,
        entities: List[str],
        evidence: List[str],
        confidence: float = 0.8
    ) -> Belief:
        """Add a new belief"""
        belief = Belief(
            id=f"belief_{len(self.beliefs)}",
            content=content,
            entities=entities,
            confidence=confidence,
            evidence=evidence,
            contradictions=[],
            timestamp=datetime.now()
        )
        
        # Check for contradictions
        contradictions = self._find_contradictions(belief)
        belief.contradictions = [c.id for c in contradictions]
        
        # Add to store
        self.beliefs[belief.id] = belief
        self.belief_graph.add_node(belief.id, belief=belief)
        
        # Add edges for entity relationships
        for entity in entities:
            for other_id, other_belief in self.beliefs.items():
                if other_id != belief.id and entity in other_belief.entities:
                    self.belief_graph.add_edge(belief.id, other_id, shared_entity=entity)
        
        return belief
    
    def update_belief(
        self,
        belief_id: str,
        new_evidence: List[str],
        confidence_delta: float
    ):
        """Update an existing belief"""
        if belief_id not in self.beliefs:
            return
        
        belief = self.beliefs[belief_id]
        belief.evidence.extend(new_evidence)
        belief.confidence = max(0, min(1, belief.confidence + confidence_delta))
        belief.last_updated = datetime.now()
    
    def query_beliefs(
        self,
        entities: Optional[List[str]] = None,
        min_confidence: float = 0.5
    ) -> List[Belief]:
        """Query beliefs by entities and confidence"""
        results = []
        
        for belief in self.beliefs.values():
            if belief.confidence < min_confidence:
                continue
            
            if entities:
                if any(entity in belief.entities for entity in entities):
                    results.append(belief)
            else:
                results.append(belief)
        
        return sorted(results, key=lambda b: b.confidence, reverse=True)
    
    def _find_contradictions(self, new_belief: Belief) -> List[Belief]:
        """Find beliefs that contradict the new belief"""
        contradictions = []
        
        # Simple contradiction detection (would be more sophisticated in practice)
        for existing in self.beliefs.values():
            # Check if they involve same entities but opposite claims
            shared_entities = set(new_belief.entities) & set(existing.entities)
            if shared_entities:
                # Encode beliefs and check consistency
                new_encoding = torch.randn(1, self.model_dim)  # Would encode properly
                existing_encoding = torch.randn(1, self.model_dim)
                
                consistency_input = torch.cat([new_encoding, existing_encoding], dim=-1)
                consistency_score = self.consistency_checker(consistency_input)
                
                if consistency_score < 0.3:  # Low consistency = contradiction
                    contradictions.append(existing)
        
        return contradictions
    
    def resolve_contradictions(self) -> Dict[str, Any]:
        """Attempt to resolve contradictions in belief system"""
        resolutions = []
        
        for belief_id, belief in self.beliefs.items():
            if belief.contradictions:
                # Find the most confident belief in contradiction set
                all_beliefs = [belief] + [self.beliefs[c_id] for c_id in belief.contradictions if c_id in self.beliefs]
                most_confident = max(all_beliefs, key=lambda b: b.confidence)
                
                # Reduce confidence of contradicting beliefs
                for b in all_beliefs:
                    if b.id != most_confident.id:
                        b.confidence *= 0.8
                
                resolutions.append({
                    'belief_set': [b.id for b in all_beliefs],
                    'resolved_to': most_confident.id,
                    'method': 'confidence_based'
                })
        
        return {
            'resolutions': resolutions,
            'unresolved_count': sum(1 for b in self.beliefs.values() if b.contradictions and b.confidence > 0.5)
        }


class CausalModel(nn.Module):
    """
    Models causal relationships in the world
    """
    
    def __init__(self, model_dim: int = 768):
        super().__init__()
        self.model_dim = model_dim
        
        # Causal relation encoder
        self.causal_encoder = nn.Sequential(
            nn.Linear(model_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )
        
        # Causal strength predictor
        self.strength_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Causal graph
        self.causal_graph = nx.DiGraph()
        self.causal_relations: Dict[Tuple[str, str], CausalRelation] = {}
    
    def add_causal_relation(
        self,
        cause: str,
        effect: str,
        strength: float = 0.7,
        conditions: List[str] = None,
        evidence: List[str] = None
    ) -> CausalRelation:
        """Add a causal relationship"""
        relation = CausalRelation(
            cause=cause,
            effect=effect,
            strength=strength,
            conditions=conditions or [],
            exceptions=[],
            evidence_count=len(evidence) if evidence else 1
        )
        
        # Add to graph
        self.causal_graph.add_edge(cause, effect, weight=strength, relation=relation)
        self.causal_relations[(cause, effect)] = relation
        
        return relation
    
    def predict_effects(
        self,
        cause: str,
        conditions: List[str] = None,
        depth: int = 3
    ) -> List[Tuple[str, float]]:
        """Predict effects of a cause"""
        effects = []
        visited = set()
        
        def traverse(node: str, accumulated_strength: float, current_depth: int):
            if current_depth > depth or node in visited:
                return
            
            visited.add(node)
            
            # Get direct effects
            for successor in self.causal_graph.successors(node):
                edge_data = self.causal_graph[node][successor]
                relation = edge_data.get('relation')
                
                if relation:
                    # Check conditions
                    if conditions and relation.conditions:
                        if not all(c in conditions for c in relation.conditions):
                            continue
                    
                    effect_strength = accumulated_strength * relation.strength
                    effects.append((successor, effect_strength))
                    
                    # Recursive traversal
                    traverse(successor, effect_strength, current_depth + 1)
        
        traverse(cause, 1.0, 0)
        
        # Aggregate effects
        effect_dict = {}
        for effect, strength in effects:
            if effect in effect_dict:
                # Combine strengths (max)
                effect_dict[effect] = max(effect_dict[effect], strength)
            else:
                effect_dict[effect] = strength
        
        return sorted(effect_dict.items(), key=lambda x: x[1], reverse=True)
    
    def explain_causation(
        self,
        cause: str,
        effect: str
    ) -> Optional[List[List[str]]]:
        """Explain causal paths between cause and effect"""
        try:
            # Find all paths
            paths = list(nx.all_simple_paths(self.causal_graph, cause, effect, cutoff=5))
            
            # Sort by total strength
            path_strengths = []
            for path in paths:
                strength = 1.0
                for i in range(len(path) - 1):
                    relation = self.causal_relations.get((path[i], path[i+1]))
                    if relation:
                        strength *= relation.strength
                path_strengths.append((path, strength))
            
            # Return paths sorted by strength
            path_strengths.sort(key=lambda x: x[1], reverse=True)
            return [path for path, _ in path_strengths]
            
        except nx.NetworkXNoPath:
            return None
    
    def update_from_observation(
        self,
        cause: str,
        effect: str,
        observed: bool,
        conditions: List[str] = None
    ):
        """Update causal model from observation"""
        relation = self.causal_relations.get((cause, effect))
        
        if relation:
            # Update strength based on observation
            if observed:
                relation.strength = min(1.0, relation.strength * 1.1)
                relation.evidence_count += 1
            else:
                # This might be an exception
                if conditions:
                    relation.exceptions.extend(conditions)
                relation.strength = max(0.1, relation.strength * 0.9)
        elif observed:
            # Add new causal relation
            self.add_causal_relation(cause, effect, strength=0.5, conditions=conditions)


class WorldModel(nn.Module):
    """
    Complete world model integrating entities, beliefs, and causation
    """
    
    def __init__(self, model_dim: int = 768):
        super().__init__()
        self.model_dim = model_dim
        
        # Components
        self.belief_system = BeliefSystem(model_dim)
        self.causal_model = CausalModel(model_dim)
        
        # Entity store
        self.entities: Dict[str, Entity] = {}
        self.entity_embeddings = {}
        
        # Entity encoder
        self.entity_encoder = nn.Sequential(
            nn.Linear(model_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, model_dim),
            nn.Tanh()
        )
        
        # Relation predictor
        self.relation_predictor = nn.Sequential(
            nn.Linear(model_dim * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 20)  # Number of relation types
        )
        
        # World state tracker
        self.world_state = {
            'time': datetime.now(),
            'focus_entities': [],
            'active_beliefs': [],
            'pending_updates': []
        }
    
    def add_entity(
        self,
        entity_id: str,
        entity_type: str,
        properties: Dict[str, Any],
        embedding: Optional[torch.Tensor] = None
    ) -> Entity:
        """Add an entity to the world model"""
        entity = Entity(
            id=entity_id,
            type=entity_type,
            properties=properties,
            relationships={},
            embedding=embedding,
            last_updated=datetime.now(),
            confidence=1.0
        )
        
        # Generate embedding if not provided
        if embedding is None:
            # Simple random embedding (would be more sophisticated)
            embedding = torch.randn(1, self.model_dim)
        
        entity.embedding = self.entity_encoder(embedding)
        
        # Store entity
        self.entities[entity_id] = entity
        self.entity_embeddings[entity_id] = entity.embedding
        
        return entity
    
    def update_entity(
        self,
        entity_id: str,
        properties: Optional[Dict[str, Any]] = None,
        relationships: Optional[Dict[str, List[str]]] = None
    ):
        """Update an entity's properties or relationships"""
        if entity_id not in self.entities:
            return
        
        entity = self.entities[entity_id]
        
        if properties:
            entity.properties.update(properties)
        
        if relationships:
            for rel_type, related_ids in relationships.items():
                if rel_type not in entity.relationships:
                    entity.relationships[rel_type] = []
                entity.relationships[rel_type].extend(related_ids)
                
                # Update causal model if applicable
                if rel_type in ['causes', 'affects', 'influences']:
                    for related_id in related_ids:
                        self.causal_model.add_causal_relation(
                            entity_id,
                            related_id,
                            strength=0.6
                        )
        
        entity.last_updated = datetime.now()
    
    def predict_entity_relations(
        self,
        entity1_id: str,
        entity2_id: str
    ) -> Dict[str, float]:
        """Predict relationships between entities"""
        if entity1_id not in self.entities or entity2_id not in self.entities:
            return {}
        
        # Get embeddings
        emb1 = self.entity_embeddings[entity1_id]
        emb2 = self.entity_embeddings[entity2_id]
        
        # Predict relations
        relation_input = torch.cat([emb1, emb2], dim=-1)
        relation_logits = self.relation_predictor(relation_input)
        relation_probs = torch.softmax(relation_logits, dim=-1)
        
        # Map to relation types
        relation_types = [
            'is_a', 'part_of', 'causes', 'prevents', 'similar_to',
            'opposite_of', 'located_in', 'owns', 'knows', 'created_by',
            'used_for', 'made_of', 'contains', 'connected_to', 'derives_from',
            'depends_on', 'conflicts_with', 'supports', 'replaces', 'precedes'
        ]
        
        predictions = {}
        probs = relation_probs.squeeze().tolist()
        for i, rel_type in enumerate(relation_types[:len(probs)]):
            if probs[i] > 0.3:  # Threshold
                predictions[rel_type] = float(probs[i])
        
        return predictions
    
    def query(
        self,
        query_type: str,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query the world model"""
        if query_type == "entity":
            # Query entities
            entity_id = parameters.get('id')
            if entity_id and entity_id in self.entities:
                entity = self.entities[entity_id]
                return {
                    'entity': entity,
                    'related_entities': self._get_related_entities(entity_id),
                    'beliefs': self.belief_system.query_beliefs([entity_id])
                }
        
        elif query_type == "causal":
            # Query causal relationships
            cause = parameters.get('cause')
            effect = parameters.get('effect')
            
            if cause and effect:
                paths = self.causal_model.explain_causation(cause, effect)
                return {'causal_paths': paths}
            elif cause:
                effects = self.causal_model.predict_effects(cause)
                return {'predicted_effects': effects}
        
        elif query_type == "belief":
            # Query beliefs
            entities = parameters.get('entities', [])
            min_confidence = parameters.get('min_confidence', 0.5)
            beliefs = self.belief_system.query_beliefs(entities, min_confidence)
            return {'beliefs': beliefs}
        
        return {}
    
    def _get_related_entities(self, entity_id: str) -> Dict[str, List[str]]:
        """Get all entities related to a given entity"""
        if entity_id not in self.entities:
            return {}
        
        entity = self.entities[entity_id]
        related = {}
        
        # Direct relationships
        for rel_type, related_ids in entity.relationships.items():
            related[rel_type] = related_ids
        
        # Inverse relationships
        for other_id, other_entity in self.entities.items():
            if other_id == entity_id:
                continue
            
            for rel_type, related_ids in other_entity.relationships.items():
                if entity_id in related_ids:
                    inverse_rel = f"inverse_{rel_type}"
                    if inverse_rel not in related:
                        related[inverse_rel] = []
                    related[inverse_rel].append(other_id)
        
        return related
    
    def simulate_scenario(
        self,
        initial_state: Dict[str, Any],
        actions: List[Dict[str, Any]],
        steps: int = 10
    ) -> List[Dict[str, Any]]:
        """Simulate a scenario and predict outcomes"""
        states = [initial_state]
        current_state = initial_state.copy()
        
        for step in range(steps):
            # Apply actions for this step
            step_actions = [a for a in actions if a.get('step', 0) == step]
            
            for action in step_actions:
                # Predict effects of action
                effects = self.causal_model.predict_effects(
                    action['cause'],
                    conditions=action.get('conditions', [])
                )
                
                # Update state
                for effect, strength in effects:
                    if strength > 0.5:  # Threshold
                        current_state[effect] = True
                
                # Update beliefs based on action
                self.belief_system.add_belief(
                    f"Action {action['cause']} performed",
                    entities=[action['cause']],
                    evidence=[f"Simulation step {step}"],
                    confidence=0.9
                )
            
            states.append(current_state.copy())
        
        return states
    
    def explain_state(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Explain why the world is in a particular state"""
        explanations = {}
        
        for key, value in state.items():
            if value and key in self.entities:
                # Find causal explanations
                causes = []
                for cause, effect in self.causal_model.causal_relations.keys():
                    if effect == key:
                        relation = self.causal_model.causal_relations[(cause, effect)]
                        if relation.strength > 0.5:
                            causes.append({
                                'cause': cause,
                                'strength': relation.strength,
                                'conditions': relation.conditions
                            })
                
                # Find supporting beliefs
                beliefs = self.belief_system.query_beliefs([key], min_confidence=0.6)
                
                explanations[key] = {
                    'causal_factors': causes,
                    'supporting_beliefs': [b.content for b in beliefs],
                    'confidence': np.mean([c['strength'] for c in causes]) if causes else 0.5
                }
        
        return explanations