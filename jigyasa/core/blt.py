"""
Byte Latent Transformer (B.L.T.) implementation
Based on Meta's research for tokenizer-free language modeling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict
from einops import rearrange, repeat
import numpy as np

from .transformer import TransformerBlock, TransformerConfig
from .embeddings import ByteEmbedding, PatchEmbedding


class EntropyBasedPatcher(nn.Module):
    """
    Dynamic patch creation based on local entropy
    Core innovation of B.L.T. - adaptive computation allocation
    """
    
    def __init__(
        self,
        min_patch_size: int = 4,
        max_patch_size: int = 32,
        entropy_window: int = 8,
        entropy_threshold: float = 2.0
    ):
        super().__init__()
        self.min_patch_size = min_patch_size
        self.max_patch_size = max_patch_size
        self.entropy_window = entropy_window
        self.entropy_threshold = entropy_threshold
        
        # Learnable parameters for entropy-based segmentation
        self.entropy_weighting = nn.Parameter(torch.ones(1))
        self.boundary_detector = nn.Linear(entropy_window, 1)
        
    def calculate_local_entropy(self, byte_sequence: torch.Tensor) -> torch.Tensor:
        """
        Calculate local entropy for each position in the byte sequence
        
        Args:
            byte_sequence: (batch_size, seq_len) tensor of byte values
        Returns:
            entropy_scores: (batch_size, seq_len) tensor of entropy values
        """
        batch_size, seq_len = byte_sequence.shape
        
        if seq_len < self.entropy_window:
            return torch.zeros_like(byte_sequence, dtype=torch.float)
        
        # Sliding window entropy calculation
        entropies = []
        
        for i in range(seq_len - self.entropy_window + 1):
            window = byte_sequence[:, i:i + self.entropy_window]  # (batch_size, window_size)
            
            # Calculate entropy for each sequence in batch
            batch_entropies = []
            for b in range(batch_size):
                byte_counts = torch.bincount(window[b], minlength=256).float()
                probs = byte_counts / self.entropy_window
                # Add small epsilon to avoid log(0)
                probs = probs + 1e-10
                entropy = -torch.sum(probs * torch.log2(probs))
                batch_entropies.append(entropy)
            
            entropies.append(torch.stack(batch_entropies))
        
        # Pad to match sequence length
        entropy_tensor = torch.stack(entropies, dim=1)  # (batch_size, seq_len - window + 1)
        
        # Pad the end
        padding = seq_len - entropy_tensor.size(1)
        if padding > 0:
            last_entropy = entropy_tensor[:, -1:].expand(-1, padding)
            entropy_tensor = torch.cat([entropy_tensor, last_entropy], dim=1)
        
        return entropy_tensor
    
    def create_patches(
        self, 
        byte_sequence: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Create dynamic patches based on entropy
        
        Args:
            byte_sequence: (batch_size, seq_len) tensor of byte values
            attention_mask: (batch_size, seq_len) attention mask
        Returns:
            patches: (batch_size, num_patches, max_patch_size) padded patches
            patch_lengths: (batch_size, num_patches) actual patch lengths
            patch_positions: (batch_size, num_patches, 2) start/end positions
        """
        batch_size, seq_len = byte_sequence.shape
        device = byte_sequence.device
        
        # Calculate local entropy
        entropy_scores = self.calculate_local_entropy(byte_sequence)
        
        # Find patch boundaries for each sequence in batch
        all_patches = []
        all_patch_lengths = []
        all_patch_positions = []
        max_num_patches = 0
        
        for b in range(batch_size):
            seq = byte_sequence[b]
            entropies = entropy_scores[b]
            
            if attention_mask is not None:
                mask = attention_mask[b]
                valid_length = mask.sum().item()
                seq = seq[:valid_length]
                entropies = entropies[:valid_length]
            
            patches, patch_lengths, positions = self._segment_sequence(seq, entropies)
            
            all_patches.append(patches)
            all_patch_lengths.append(patch_lengths)
            all_patch_positions.append(positions)
            max_num_patches = max(max_num_patches, len(patches))
        
        # Pad patches to same dimensions
        padded_patches = []
        padded_lengths = []
        padded_positions = []
        
        for patches, lengths, positions in zip(all_patches, all_patch_lengths, all_patch_positions):
            # Pad patches
            num_patches = len(patches)
            max_patch_len = max(len(p) for p in patches) if patches else 1
            max_patch_len = min(max_patch_len, self.max_patch_size)
            
            patch_tensor = torch.zeros(max_num_patches, max_patch_len, device=device, dtype=torch.long)
            length_tensor = torch.zeros(max_num_patches, device=device, dtype=torch.long)
            position_tensor = torch.zeros(max_num_patches, 2, device=device, dtype=torch.long)
            
            for i, (patch, length, pos) in enumerate(zip(patches, lengths, positions)):
                if i >= max_num_patches:
                    break
                    
                patch_len = min(len(patch), max_patch_len)
                patch_tensor[i, :patch_len] = patch[:patch_len]
                length_tensor[i] = patch_len
                position_tensor[i] = torch.tensor(pos, device=device)
            
            padded_patches.append(patch_tensor)
            padded_lengths.append(length_tensor)
            padded_positions.append(position_tensor)
        
        # Stack into batch tensors
        patches = torch.stack(padded_patches)  # (batch_size, max_num_patches, max_patch_len)
        patch_lengths = torch.stack(padded_lengths)  # (batch_size, max_num_patches)
        patch_positions = torch.stack(padded_positions)  # (batch_size, max_num_patches, 2)
        
        return patches, patch_lengths, patch_positions
    
    def _segment_sequence(
        self, 
        sequence: torch.Tensor, 
        entropies: torch.Tensor
    ) -> Tuple[List[torch.Tensor], List[int], List[Tuple[int, int]]]:
        """
        Segment a single sequence into patches based on entropy
        """
        seq_len = sequence.size(0)
        if seq_len == 0:
            return [], [], []
        
        patches = []
        patch_lengths = []
        patch_positions = []
        
        start = 0
        while start < seq_len:
            # Determine patch end based on entropy
            end = min(start + self.max_patch_size, seq_len)
            
            # Look for natural boundary within window
            if end < seq_len and end - start > self.min_patch_size:
                # Find low-entropy region for boundary
                search_start = start + self.min_patch_size
                search_end = min(start + self.max_patch_size, seq_len)
                
                if search_start < search_end:
                    window_entropies = entropies[search_start:search_end]
                    
                    # Find minimum entropy position
                    min_entropy_idx = torch.argmin(window_entropies).item()
                    candidate_end = search_start + min_entropy_idx
                    
                    # Use boundary if entropy is below threshold
                    if window_entropies[min_entropy_idx] < self.entropy_threshold:
                        end = candidate_end + 1  # Include the boundary byte
            
            # Extract patch
            patch = sequence[start:end]
            patches.append(patch)
            patch_lengths.append(len(patch))
            patch_positions.append((start, end))
            
            # Move to next patch
            start = end
        
        return patches, patch_lengths, patch_positions


class ByteTransformerLayer(nn.Module):
    """
    Byte-level transformer layer for local processing
    Processes raw bytes before patch aggregation
    """
    
    def __init__(self, d_model: int, nhead: int = 8, dim_feedforward: int = None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
            
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Linear(dim_feedforward, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
            src_mask: (seq_len, seq_len) or (batch_size, seq_len)
        """
        # Self-attention with residual connection
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm, key_padding_mask=src_mask)
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        x_norm = self.norm2(x)
        ff_output = self.feed_forward(x_norm)
        x = x + self.dropout(ff_output)
        
        return x


class PatchCrossAttention(nn.Module):
    """
    Cross-attention layer for aligning patch embeddings with byte-level representations
    Ensures fine-grained details are preserved during decoding
    """
    
    def __init__(self, d_model: int, nhead: int = 8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        patch_embeddings: torch.Tensor,
        byte_hidden_states: torch.Tensor,
        patch_positions: torch.Tensor,
        byte_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            patch_embeddings: (batch_size, num_patches, d_model)
            byte_hidden_states: (batch_size, seq_len, d_model)
            patch_positions: (batch_size, num_patches, 2) start/end positions
            byte_mask: (batch_size, seq_len) byte-level mask
        """
        batch_size, num_patches, d_model = patch_embeddings.shape
        seq_len = byte_hidden_states.size(1)
        
        # Create patch-to-byte attention mask
        # Each patch can attend to its corresponding byte positions
        patch_byte_mask = torch.zeros(batch_size, num_patches, seq_len, device=patch_embeddings.device)
        
        for b in range(batch_size):
            for p in range(num_patches):
                start, end = patch_positions[b, p]
                if start < seq_len and end <= seq_len and start < end:
                    patch_byte_mask[b, p, start:end] = 1
        
        # Apply byte mask if provided
        if byte_mask is not None:
            patch_byte_mask = patch_byte_mask * byte_mask.unsqueeze(1)
        
        # Cross-attention: patches attend to their corresponding bytes
        patch_norm = self.norm(patch_embeddings)
        
        # Reshape for cross-attention
        patch_norm_flat = patch_norm.view(batch_size * num_patches, 1, d_model)
        byte_states_expanded = byte_hidden_states.unsqueeze(1).expand(-1, num_patches, -1, -1)
        byte_states_flat = byte_states_expanded.contiguous().view(batch_size * num_patches, seq_len, d_model)
        
        # Create flattened mask
        mask_flat = patch_byte_mask.view(batch_size * num_patches, seq_len)
        mask_flat = (mask_flat == 0)  # Invert for key_padding_mask
        
        # Cross-attention
        cross_output, _ = self.cross_attn(
            query=patch_norm_flat,
            key=byte_states_flat,
            value=byte_states_flat,
            key_padding_mask=mask_flat
        )
        
        # Reshape back
        cross_output = cross_output.view(batch_size, num_patches, d_model)
        
        # Residual connection
        output = patch_embeddings + self.dropout(cross_output)
        
        return output


class ByteLatentTransformer(nn.Module):
    """
    Complete Byte Latent Transformer implementation
    Combines byte-level processing with patch-based latent representations
    """
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        
        # Entropy-based patcher
        self.patcher = EntropyBasedPatcher(
            min_patch_size=4,
            max_patch_size=config.max_seq_length // 32,  # Adaptive based on seq length
            entropy_window=8,
            entropy_threshold=2.0
        )
        
        # Byte-level components
        self.byte_embedding = ByteEmbedding(config.d_model, max_byte_value=256)
        
        # Byte transformer layers (local processing)
        self.byte_layers = nn.ModuleList([
            ByteTransformerLayer(config.d_model, nhead=8)
            for _ in range(2)  # Fewer layers for byte processing
        ])
        
        # Patch embedding and aggregation
        self.patch_embedding = PatchEmbedding(
            d_model=config.d_model,
            patch_size=16,
            max_patch_length=32,
            byte_vocab_size=256
        )
        
        # Latent transformer (main processing)
        self.latent_layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Cross-attention for decoding
        self.patch_cross_attention = PatchCrossAttention(config.d_model, nhead=8)
        
        # Output layers
        self.output_norm = nn.LayerNorm(config.d_model)
        self.output_projection = nn.Linear(config.d_model, 256)  # Byte vocabulary
        
        # Positional encoding for patches
        self.patch_pos_encoding = nn.Parameter(
            torch.randn(1, 512, config.d_model) * 0.02  # Support up to 512 patches
        )
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through B.L.T. architecture
        
        Args:
            input_ids: (batch_size, seq_len) byte sequences
            attention_mask: (batch_size, seq_len) attention mask
            past_key_values: Cached key-value pairs
            use_cache: Whether to return cache
            labels: Target labels for training
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Step 1: Byte-level embedding
        byte_embeddings = self.byte_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Step 2: Local byte processing
        byte_hidden = byte_embeddings
        byte_mask = ~attention_mask.bool() if attention_mask is not None else None
        
        for layer in self.byte_layers:
            byte_hidden = layer(byte_hidden, src_mask=byte_mask)
        
        # Step 3: Dynamic patch creation
        patches, patch_lengths, patch_positions = self.patcher.create_patches(
            input_ids, attention_mask
        )
        
        # Step 4: Patch embedding
        patch_embeddings = self.patch_embedding(patches, patch_lengths)
        num_patches = patch_embeddings.size(1)
        
        # Add positional encoding to patches
        if num_patches <= self.patch_pos_encoding.size(1):
            patch_embeddings = patch_embeddings + self.patch_pos_encoding[:, :num_patches, :]
        
        # Step 5: Latent transformer processing
        hidden_states = patch_embeddings
        
        # Create patch attention mask
        patch_mask = (patch_lengths > 0).float()  # (batch_size, num_patches)
        if patch_mask.dim() == 2:
            # Expand for attention computation
            patch_attention_mask = patch_mask[:, None, None, :]
            patch_attention_mask = (1.0 - patch_attention_mask) * torch.finfo(hidden_states.dtype).min
        else:
            patch_attention_mask = None
        
        # Process through latent transformer
        present_key_values = [] if use_cache else None
        
        for i, layer in enumerate(self.latent_layers):
            past_key_value = past_key_values[i] if past_key_values else None
            
            hidden_states, present_key_value = layer(
                hidden_states,
                mask=patch_attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Step 6: Cross-attention with byte representations
        aligned_patches = self.patch_cross_attention(
            hidden_states,
            byte_hidden,
            patch_positions,
            attention_mask
        )
        
        # Step 7: Decode to byte level
        # Expand patch representations back to byte level
        byte_level_output = self._expand_patches_to_bytes(
            aligned_patches, patch_positions, seq_len, device
        )
        
        # Final normalization and projection
        byte_level_output = self.output_norm(byte_level_output)
        logits = self.output_projection(byte_level_output)  # (batch_size, seq_len, 256)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            # Shift for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fn(shift_logits.view(-1, 256), shift_labels.view(-1))
        
        return {
            'logits': logits,
            'loss': loss,
            'past_key_values': present_key_values if use_cache else None,
            'hidden_states': byte_level_output,
            'patch_embeddings': aligned_patches,
            'patch_positions': patch_positions,
            'byte_hidden_states': byte_hidden
        }
    
    def _expand_patches_to_bytes(
        self,
        patch_embeddings: torch.Tensor,
        patch_positions: torch.Tensor,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Expand patch-level representations back to byte level
        """
        batch_size, num_patches, d_model = patch_embeddings.shape
        
        # Initialize byte-level output
        byte_output = torch.zeros(batch_size, seq_len, d_model, device=device)
        
        for b in range(batch_size):
            for p in range(num_patches):
                start, end = patch_positions[b, p]
                if start < seq_len and end <= seq_len and start < end:
                    # Distribute patch embedding across its byte positions
                    patch_emb = patch_embeddings[b, p]  # (d_model,)
                    byte_output[b, start:end] = patch_emb.unsqueeze(0).expand(end - start, -1)
        
        return byte_output


class BLTTokenizer:
    """
    Simple byte-level tokenizer for BLT
    Converts text to bytes directly without traditional tokenization
    """
    
    def __init__(self):
        self.byte_to_idx = {i: i for i in range(256)}
        self.idx_to_byte = {i: i for i in range(256)}
    
    def encode(self, text: str) -> List[int]:
        """Convert text to byte indices"""
        return list(text.encode('utf-8'))
    
    def decode(self, indices: List[int]) -> str:
        """Convert byte indices back to text"""
        try:
            return bytes(indices).decode('utf-8', errors='ignore')
        except:
            return ""