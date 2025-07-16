"""
Causal Reasoning Module
Implements causal inference and reasoning capabilities
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Set, Any
import networkx as nx
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class CausalRelation:
    """Represents a causal relationship"""
    cause: str
    effect: str
    strength: float
    confidence: float
    evidence: List[str]


class CausalGraph:
    """
    Represents a causal graph for reasoning about cause-effect relationships
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.node_attributes = {}
        self.edge_attributes = defaultdict(dict)
    
    def add_causal_relation(self, relation: CausalRelation):
        """Add a causal relationship to the graph"""
        self.graph.add_edge(
            relation.cause,
            relation.effect,
            weight=relation.strength,
            confidence=relation.confidence
        )
        self.edge_attributes[(relation.cause, relation.effect)]['evidence'] = relation.evidence
    
    def add_node(self, node: str, attributes: Dict[str, Any]):
        """Add a node with attributes"""
        self.graph.add_node(node)
        self.node_attributes[node] = attributes
    
    def get_causal_chain(self, start: str, end: str) -> List[List[str]]:
        """Find all causal chains from start to end"""
        try:
            paths = list(nx.all_simple_paths(self.graph, start, end))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def get_direct_causes(self, effect: str) -> List[str]:
        """Get direct causes of an effect"""
        return list(self.graph.predecessors(effect))
    
    def get_direct_effects(self, cause: str) -> List[str]:
        """Get direct effects of a cause"""
        return list(self.graph.successors(cause))
    
    def compute_causal_strength(self, cause: str, effect: str) -> float:
        """Compute the total causal strength between cause and effect"""
        paths = self.get_causal_chain(cause, effect)
        if not paths:
            return 0.0
        
        total_strength = 0.0
        for path in paths:
            path_strength = 1.0
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                path_strength *= edge_data.get('weight', 0.0)
            total_strength += path_strength
        
        return min(total_strength, 1.0)  # Cap at 1.0
    
    def identify_confounders(self, cause: str, effect: str) -> Set[str]:
        """Identify potential confounders between cause and effect"""
        confounders = set()
        
        # Find common causes (confounders)
        cause_ancestors = nx.ancestors(self.graph, cause)
        effect_ancestors = nx.ancestors(self.graph, effect)
        
        # Common ancestors are potential confounders
        confounders = cause_ancestors.intersection(effect_ancestors)
        
        # Also check for colliders (common effects)
        cause_descendants = nx.descendants(self.graph, cause)
        effect_descendants = nx.descendants(self.graph, effect)
        
        # Remove colliders as they're not confounders
        colliders = cause_descendants.intersection(effect_descendants)
        confounders = confounders - colliders
        
        return confounders
    
    def perform_intervention(self, node: str, value: Any) -> Dict[str, Any]:
        """
        Perform a causal intervention (do-calculus)
        Sets a node to a specific value and propagates effects
        """
        # Create a copy of the graph for intervention
        intervened_graph = self.graph.copy()
        
        # Remove incoming edges to the intervened node (cut off causes)
        incoming_edges = list(intervened_graph.in_edges(node))
        intervened_graph.remove_edges_from(incoming_edges)
        
        # Set the node value
        results = {node: value}
        
        # Propagate effects using topological sort
        try:
            topo_order = list(nx.topological_sort(intervened_graph))
            start_idx = topo_order.index(node)
            
            for affected_node in topo_order[start_idx + 1:]:
                # Simple propagation (would be more complex in practice)
                causes = list(intervened_graph.predecessors(affected_node))
                if causes:
                    # Aggregate causal effects
                    effect_value = 0.0
                    for cause in causes:
                        if cause in results:
                            edge_data = intervened_graph.get_edge_data(cause, affected_node)
                            effect_value += results[cause] * edge_data.get('weight', 0.0)
                    results[affected_node] = effect_value
        except nx.NetworkXError:
            pass  # Graph might have cycles
        
        return results


class CausalReasoner(nn.Module):
    """
    Neural module for causal reasoning
    """
    
    def __init__(self, model_dim: int = 768):
        super().__init__()
        self.model_dim = model_dim
        
        # Causal relation extractor
        self.relation_extractor = nn.Sequential(
            nn.Linear(model_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # [is_causal, strength, confidence]
        )
        
        # Causal chain scorer
        self.chain_scorer = nn.Sequential(
            nn.Linear(model_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        
        # Intervention predictor
        self.intervention_predictor = nn.Sequential(
            nn.Linear(model_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, model_dim)
        )
        
        self.causal_graph = CausalGraph()
    
    def extract_causal_relations(
        self,
        text_embeddings: torch.Tensor,
        entity_pairs: List[Tuple[int, int, int, int]]  # [(cause_start, cause_end, effect_start, effect_end)]
    ) -> List[CausalRelation]:
        """
        Extract causal relations from text embeddings
        """
        relations = []
        
        for cause_start, cause_end, effect_start, effect_end in entity_pairs:
            # Extract entity embeddings
            cause_emb = text_embeddings[:, cause_start:cause_end].mean(dim=1)
            effect_emb = text_embeddings[:, effect_start:effect_end].mean(dim=1)
            
            # Concatenate for relation extraction
            pair_emb = torch.cat([cause_emb, effect_emb], dim=-1)
            
            # Predict causal relation
            relation_output = self.relation_extractor(pair_emb)
            is_causal = torch.sigmoid(relation_output[:, 0]) > 0.5
            
            if is_causal.item():
                strength = torch.sigmoid(relation_output[:, 1]).item()
                confidence = torch.sigmoid(relation_output[:, 2]).item()
                
                relation = CausalRelation(
                    cause=f"entity_{cause_start}_{cause_end}",
                    effect=f"entity_{effect_start}_{effect_end}",
                    strength=strength,
                    confidence=confidence,
                    evidence=[]
                )
                relations.append(relation)
        
        return relations
    
    def build_causal_graph(self, relations: List[CausalRelation]) -> CausalGraph:
        """Build a causal graph from extracted relations"""
        graph = CausalGraph()
        
        for relation in relations:
            graph.add_causal_relation(relation)
        
        return graph
    
    def reason_about_causality(
        self,
        query_embedding: torch.Tensor,
        causal_graph: CausalGraph,
        query_type: str = "effect_of_cause"
    ) -> Dict[str, Any]:
        """
        Perform causal reasoning based on query
        """
        if query_type == "effect_of_cause":
            # Find most likely effects
            return self._find_effects(query_embedding, causal_graph)
        elif query_type == "cause_of_effect":
            # Find most likely causes
            return self._find_causes(query_embedding, causal_graph)
        elif query_type == "causal_chain":
            # Find causal chains
            return self._find_causal_chains(query_embedding, causal_graph)
        elif query_type == "intervention":
            # Predict intervention effects
            return self._predict_intervention(query_embedding, causal_graph)
        else:
            return {"error": "Unknown query type"}
    
    def _find_effects(self, cause_embedding: torch.Tensor, graph: CausalGraph) -> Dict[str, Any]:
        """Find likely effects of a cause"""
        # This would match cause_embedding to nodes in graph
        # For now, return a placeholder
        return {
            "likely_effects": [],
            "confidence": 0.0,
            "reasoning": "Causal effect analysis"
        }
    
    def _find_causes(self, effect_embedding: torch.Tensor, graph: CausalGraph) -> Dict[str, Any]:
        """Find likely causes of an effect"""
        return {
            "likely_causes": [],
            "confidence": 0.0,
            "reasoning": "Causal attribution analysis"
        }
    
    def _find_causal_chains(self, query_embedding: torch.Tensor, graph: CausalGraph) -> Dict[str, Any]:
        """Find causal chains relevant to query"""
        return {
            "causal_chains": [],
            "confidence": 0.0,
            "reasoning": "Causal chain analysis"
        }
    
    def _predict_intervention(self, intervention_embedding: torch.Tensor, graph: CausalGraph) -> Dict[str, Any]:
        """Predict effects of causal intervention"""
        # This would use do-calculus principles
        return {
            "intervention_effects": {},
            "confidence": 0.0,
            "reasoning": "Causal intervention analysis"
        }
    
    def forward(
        self,
        embeddings: torch.Tensor,
        causal_query: Optional[str] = None,
        entity_pairs: Optional[List[Tuple[int, int, int, int]]] = None
    ) -> Dict[str, Any]:
        """
        Forward pass for causal reasoning
        """
        results = {}
        
        # Extract causal relations if entity pairs provided
        if entity_pairs:
            relations = self.extract_causal_relations(embeddings, entity_pairs)
            self.causal_graph = self.build_causal_graph(relations)
            results['extracted_relations'] = len(relations)
        
        # Perform causal reasoning if query provided
        if causal_query:
            query_emb = embeddings.mean(dim=1)  # Simple pooling
            reasoning_results = self.reason_about_causality(
                query_emb,
                self.causal_graph,
                self._classify_query_type(causal_query)
            )
            results.update(reasoning_results)
        
        results['causal_graph_size'] = len(self.causal_graph.graph.nodes())
        return results
    
    def _classify_query_type(self, query: str) -> str:
        """Classify the type of causal query"""
        query_lower = query.lower()
        
        if "what causes" in query_lower or "why" in query_lower:
            return "cause_of_effect"
        elif "what happens if" in query_lower or "effect of" in query_lower:
            return "effect_of_cause"
        elif "how does" in query_lower and "lead to" in query_lower:
            return "causal_chain"
        elif "if we" in query_lower or "intervention" in query_lower:
            return "intervention"
        else:
            return "effect_of_cause"  # default