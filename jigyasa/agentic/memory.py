"""
Memory System for Agents
Provides short-term and long-term memory capabilities
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime
import json
import pickle
from collections import deque
import numpy as np


@dataclass
class MemoryEntry:
    """A single memory entry"""
    id: str
    timestamp: datetime
    content: Any
    embedding: Optional[torch.Tensor] = None
    metadata: Optional[Dict[str, Any]] = None
    importance: float = 0.5
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class MemoryStore:
    """
    Base memory store interface
    """
    
    def store(self, entry: MemoryEntry):
        """Store a memory entry"""
        raise NotImplementedError
    
    def retrieve(self, query: Any, k: int = 5) -> List[MemoryEntry]:
        """Retrieve relevant memories"""
        raise NotImplementedError
    
    def update(self, entry_id: str, updates: Dict[str, Any]):
        """Update a memory entry"""
        raise NotImplementedError
    
    def delete(self, entry_id: str):
        """Delete a memory entry"""
        raise NotImplementedError


class ShortTermMemory(MemoryStore):
    """
    Short-term memory with limited capacity (like working memory)
    """
    
    def __init__(self, capacity: int = 100):
        self.capacity = capacity
        self.memories = deque(maxlen=capacity)
        self.index = {}
    
    def store(self, entry: MemoryEntry):
        """Store in short-term memory"""
        # If at capacity, oldest memory is automatically removed
        if len(self.memories) == self.capacity and self.memories:
            oldest = self.memories[0]
            if oldest.id in self.index:
                del self.index[oldest.id]
        
        self.memories.append(entry)
        self.index[entry.id] = entry
    
    def retrieve(self, query: Any, k: int = 5) -> List[MemoryEntry]:
        """Retrieve recent memories"""
        # Simple recency-based retrieval
        return list(self.memories)[-k:]
    
    def update(self, entry_id: str, updates: Dict[str, Any]):
        """Update a memory entry"""
        if entry_id in self.index:
            entry = self.index[entry_id]
            for key, value in updates.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)
    
    def delete(self, entry_id: str):
        """Delete a memory entry"""
        if entry_id in self.index:
            entry = self.index[entry_id]
            self.memories.remove(entry)
            del self.index[entry_id]
    
    def get_all(self) -> List[MemoryEntry]:
        """Get all memories"""
        return list(self.memories)


class LongTermMemory(MemoryStore):
    """
    Long-term memory with semantic retrieval
    """
    
    def __init__(self, embedding_dim: int = 768):
        self.embedding_dim = embedding_dim
        self.memories = {}
        self.embeddings = []
        self.entry_ids = []
        
        # Importance decay factor
        self.decay_factor = 0.95
    
    def store(self, entry: MemoryEntry):
        """Store in long-term memory"""
        self.memories[entry.id] = entry
        
        if entry.embedding is not None:
            self.embeddings.append(entry.embedding)
            self.entry_ids.append(entry.id)
    
    def retrieve(self, query: Any, k: int = 5) -> List[MemoryEntry]:
        """Retrieve semantically similar memories"""
        if not self.embeddings:
            return []
        
        # If query is an embedding
        if isinstance(query, torch.Tensor):
            query_embedding = query
        else:
            # Would need to encode query - simplified for now
            query_embedding = torch.randn(self.embedding_dim)
        
        # Compute similarities
        embeddings_tensor = torch.stack(self.embeddings)
        similarities = torch.cosine_similarity(
            query_embedding.unsqueeze(0),
            embeddings_tensor
        )
        
        # Get top-k
        top_k_indices = torch.topk(similarities, min(k, len(similarities))).indices
        
        # Retrieve entries
        results = []
        for idx in top_k_indices:
            entry_id = self.entry_ids[idx]
            entry = self.memories[entry_id]
            
            # Update access info
            entry.access_count += 1
            entry.last_accessed = datetime.now()
            
            results.append(entry)
        
        return results
    
    def update(self, entry_id: str, updates: Dict[str, Any]):
        """Update a memory entry"""
        if entry_id in self.memories:
            entry = self.memories[entry_id]
            for key, value in updates.items():
                if hasattr(entry, key):
                    setattr(entry, key, value)
            
            # Update embedding if provided
            if 'embedding' in updates:
                idx = self.entry_ids.index(entry_id)
                self.embeddings[idx] = updates['embedding']
    
    def delete(self, entry_id: str):
        """Delete a memory entry"""
        if entry_id in self.memories:
            del self.memories[entry_id]
            
            # Remove from embeddings
            if entry_id in self.entry_ids:
                idx = self.entry_ids.index(entry_id)
                self.embeddings.pop(idx)
                self.entry_ids.pop(idx)
    
    def consolidate(self, threshold: float = 0.3):
        """Consolidate memories by removing low-importance ones"""
        entries_to_remove = []
        
        for entry_id, entry in self.memories.items():
            # Decay importance over time
            time_since_access = datetime.now() - (entry.last_accessed or entry.timestamp)
            decay = self.decay_factor ** (time_since_access.days)
            current_importance = entry.importance * decay
            
            if current_importance < threshold:
                entries_to_remove.append(entry_id)
        
        for entry_id in entries_to_remove:
            self.delete(entry_id)
        
        return len(entries_to_remove)


class AgentMemory:
    """
    Complete memory system for an agent
    """
    
    def __init__(self, capacity: int = 1000, embedding_dim: int = 768):
        self.short_term = ShortTermMemory(capacity=min(capacity // 10, 100))
        self.long_term = LongTermMemory(embedding_dim=embedding_dim)
        
        # Memory encoder for creating embeddings
        self.memory_encoder = self._build_memory_encoder(embedding_dim)
        
        # Interaction history
        self.interaction_history = []
        
        # Memory statistics
        self.stats = {
            'total_stored': 0,
            'total_retrieved': 0,
            'consolidations': 0
        }
    
    def _build_memory_encoder(self, embedding_dim: int) -> nn.Module:
        """Build encoder for memory embeddings"""
        return nn.Sequential(
            nn.Linear(embedding_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, embedding_dim),
            nn.Tanh()
        )
    
    def add_interaction(self, request: str, context: Optional[Dict[str, Any]] = None):
        """Add an interaction to memory"""
        # Create memory entry
        entry = MemoryEntry(
            id=f"interaction_{len(self.interaction_history)}",
            timestamp=datetime.now(),
            content={
                'request': request,
                'context': context
            },
            metadata={'type': 'interaction'},
            importance=0.7
        )
        
        # Generate embedding (simplified)
        text_embedding = torch.randn(self.memory_encoder[0].in_features)
        entry.embedding = self.memory_encoder(text_embedding)
        
        # Store in both memories
        self.short_term.store(entry)
        self.long_term.store(entry)
        
        # Update history
        self.interaction_history.append({
            'timestamp': entry.timestamp,
            'request': request,
            'context': context
        })
        
        self.stats['total_stored'] += 1
    
    def add_knowledge(
        self,
        knowledge: str,
        source: Optional[str] = None,
        importance: float = 0.8
    ):
        """Add knowledge to long-term memory"""
        entry = MemoryEntry(
            id=f"knowledge_{self.stats['total_stored']}",
            timestamp=datetime.now(),
            content=knowledge,
            metadata={
                'type': 'knowledge',
                'source': source
            },
            importance=importance
        )
        
        # Generate embedding
        text_embedding = torch.randn(self.memory_encoder[0].in_features)
        entry.embedding = self.memory_encoder(text_embedding)
        
        # Store only in long-term
        self.long_term.store(entry)
        self.stats['total_stored'] += 1
    
    def retrieve_relevant(
        self,
        query: str,
        k: int = 5,
        include_short_term: bool = True
    ) -> List[MemoryEntry]:
        """Retrieve relevant memories for a query"""
        # Generate query embedding (simplified)
        query_embedding = torch.randn(self.memory_encoder[0].in_features)
        query_embedding = self.memory_encoder(query_embedding)
        
        # Retrieve from long-term
        long_term_results = self.long_term.retrieve(query_embedding, k)
        
        # Optionally include short-term
        if include_short_term:
            short_term_results = self.short_term.retrieve(query, min(k, 3))
            
            # Combine and deduplicate
            all_results = long_term_results + short_term_results
            seen_ids = set()
            unique_results = []
            
            for entry in all_results:
                if entry.id not in seen_ids:
                    seen_ids.add(entry.id)
                    unique_results.append(entry)
            
            results = unique_results[:k]
        else:
            results = long_term_results
        
        self.stats['total_retrieved'] += len(results)
        return results
    
    def get_recent(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent interactions"""
        return self.interaction_history[-n:]
    
    def consolidate_memories(self, importance_threshold: float = 0.3):
        """Consolidate memories to manage storage"""
        # Move important short-term memories to long-term
        for entry in self.short_term.get_all():
            if entry.importance > 0.7 and entry.id not in self.long_term.memories:
                self.long_term.store(entry)
        
        # Consolidate long-term memory
        removed = self.long_term.consolidate(importance_threshold)
        
        self.stats['consolidations'] += 1
        return removed
    
    def save_to_disk(self, filepath: str):
        """Save memory to disk"""
        data = {
            'long_term_memories': self.long_term.memories,
            'interaction_history': self.interaction_history,
            'stats': self.stats
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_from_disk(self, filepath: str):
        """Load memory from disk"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        # Restore long-term memories
        for entry_id, entry in data['long_term_memories'].items():
            self.long_term.store(entry)
        
        self.interaction_history = data['interaction_history']
        self.stats = data['stats']
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics"""
        return {
            'short_term_size': len(self.short_term.memories),
            'long_term_size': len(self.long_term.memories),
            'total_interactions': len(self.interaction_history),
            **self.stats
        }