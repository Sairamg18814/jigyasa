"""
Specialized embedding layers for Jigyasa
Including byte-level embeddings and rotary positional encodings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from einops import rearrange


class ByteEmbedding(nn.Module):
    """Byte-level embedding for processing raw byte sequences"""
    
    def __init__(self, d_model: int, max_byte_value: int = 256):
        super().__init__()
        self.d_model = d_model
        self.max_byte_value = max_byte_value
        
        # Embedding for raw bytes (0-255)
        self.byte_embedding = nn.Embedding(max_byte_value, d_model)
        
        # Layer norm for embedding stabilization
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            byte_ids: Tensor of byte values (batch_size, seq_len)
        Returns:
            Embedded representation (batch_size, seq_len, d_model)
        """
        # Ensure byte values are in valid range
        byte_ids = torch.clamp(byte_ids, 0, self.max_byte_value - 1)
        
        # Embed bytes
        embeddings = self.byte_embedding(byte_ids)
        
        # Apply layer normalization
        embeddings = self.layer_norm(embeddings)
        
        return embeddings


class PatchEmbedding(nn.Module):
    """
    Patch-based embedding for B.L.T. architecture
    Converts variable-length byte patches into fixed-size embeddings
    """
    
    def __init__(
        self, 
        d_model: int, 
        patch_size: int = 16,
        max_patch_length: int = 128,
        byte_vocab_size: int = 256
    ):
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.max_patch_length = max_patch_length
        
        # Byte-level encoder
        self.byte_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=8,
                dim_feedforward=d_model * 4,
                dropout=0.1,
                batch_first=True
            ),
            num_layers=2
        )
        
        # Byte embedding
        self.byte_embedding = ByteEmbedding(d_model, byte_vocab_size)
        
        # Patch aggregation
        self.patch_aggregator = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=8,
            batch_first=True
        )
        
        # Learnable query for patch aggregation
        self.patch_query = nn.Parameter(torch.randn(1, 1, d_model))
        
    def forward(self, byte_patches: torch.Tensor, patch_lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            byte_patches: Padded byte patches (batch_size, num_patches, max_patch_length)
            patch_lengths: Actual length of each patch (batch_size, num_patches)
        Returns:
            Patch embeddings (batch_size, num_patches, d_model)
        """
        batch_size, num_patches, max_patch_len = byte_patches.shape
        
        # Reshape for processing
        byte_patches = byte_patches.view(batch_size * num_patches, max_patch_len)
        
        # Create attention mask for variable-length patches
        patch_mask = torch.arange(max_patch_len, device=byte_patches.device)[None, :] < patch_lengths.view(-1, 1)
        
        # Embed bytes
        byte_embeddings = self.byte_embedding(byte_patches)  # (batch*patches, max_patch_len, d_model)
        
        # Encode byte sequence within each patch
        encoded_bytes = self.byte_encoder(byte_embeddings, src_key_padding_mask=~patch_mask)
        
        # Aggregate bytes into patch representation using attention
        query = self.patch_query.expand(batch_size * num_patches, -1, -1)
        patch_embeddings, _ = self.patch_aggregator(
            query=query,
            key=encoded_bytes,
            value=encoded_bytes,
            key_padding_mask=~patch_mask
        )
        
        # Reshape back to original dimensions
        patch_embeddings = patch_embeddings.view(batch_size, num_patches, self.d_model)
        
        return patch_embeddings


class RotaryEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE)
    More effective than traditional sinusoidal embeddings for long sequences
    """
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        
        # Precompute rotation matrices
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Cache for efficiency
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
        
    def _update_cache(self, seq_len: int, device: torch.device, dtype: torch.dtype):
        """Update cached rotation matrices"""
        if seq_len > self._cached_seq_len:
            self._cached_seq_len = seq_len
            
            # Generate position indices
            t = torch.arange(seq_len, device=device, dtype=dtype)
            
            # Compute frequencies
            freqs = torch.outer(t, self.inv_freq)
            
            # Create rotation matrices
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cached_cos = emb.cos()
            self._cached_sin = emb.sin()
    
    def rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Rotate half of the hidden dimensions"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    
    def apply_rotary_pos_emb(self, q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply rotary position embedding to query and key tensors"""
        # Apply rotation
        q_rot = (q * cos) + (self.rotate_half(q) * sin)
        k_rot = (k * cos) + (self.rotate_half(k) * sin)
        return q_rot, k_rot
    
    def forward(self, q: torch.Tensor, k: torch.Tensor, seq_len: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            q: Query tensor (..., seq_len, dim)
            k: Key tensor (..., seq_len, dim)
            seq_len: Sequence length (inferred if None)
        Returns:
            Rotary position embedded query and key tensors
        """
        if seq_len is None:
            seq_len = q.shape[-2]
            
        # Update cache if necessary
        self._update_cache(seq_len, q.device, q.dtype)
        
        # Get cached rotation matrices
        cos = self._cached_cos[:seq_len]
        sin = self._cached_sin[:seq_len]
        
        # Expand dimensions for broadcasting
        cos = cos[None, None, :, :]  # (1, 1, seq_len, dim)
        sin = sin[None, None, :, :]  # (1, 1, seq_len, dim)
        
        # Apply rotary embedding
        q_rot, k_rot = self.apply_rotary_pos_emb(q, k, cos, sin)
        
        return q_rot, k_rot


class LearnedPositionalEmbedding(nn.Module):
    """Learned positional embeddings as an alternative to sinusoidal"""
    
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Learnable position embeddings
        self.pos_embedding = nn.Parameter(torch.randn(max_seq_len, d_model))
        
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            seq_len: Sequence length
            device: Device to place tensor on
        Returns:
            Position embeddings (seq_len, d_model)
        """
        assert seq_len <= self.max_seq_len, f"Sequence length {seq_len} exceeds maximum {self.max_seq_len}"
        
        return self.pos_embedding[:seq_len].to(device)


class ALiBiPositionalEmbedding(nn.Module):
    """
    Attention with Linear Biases (ALiBi)
    Alternative to positional embeddings that modifies attention directly
    """
    
    def __init__(self, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        
        # Compute slopes for each attention head
        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', slopes)
        
    def _get_slopes(self, n_heads: int) -> torch.Tensor:
        """Generate slopes for ALiBi attention bias"""
        def get_slopes_power_of_2(n_heads):
            start = (2**(-2**-(math.log2(n_heads)-3)))
            ratio = start
            return [start*ratio**i for i in range(n_heads)]
        
        if math.log2(n_heads).is_integer():
            return torch.tensor(get_slopes_power_of_2(n_heads))
        else:
            # Handle non-power-of-2 heads
            closest_power_of_2 = 2**math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            
            # Add extra slopes
            extra_slopes = self._get_slopes(2 * closest_power_of_2)[0::2][:n_heads - closest_power_of_2]
            slopes.extend(extra_slopes)
            
            return torch.tensor(slopes)
    
    def forward(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        Generate ALiBi bias matrix
        
        Args:
            seq_len: Sequence length
            device: Device to place tensor on
        Returns:
            ALiBi bias matrix (n_heads, seq_len, seq_len)
        """
        # Create distance matrix
        distances = torch.arange(seq_len, device=device)[None, :] - torch.arange(seq_len, device=device)[:, None]
        distances = distances.abs()
        
        # Apply slopes
        alibi_bias = distances[None, :, :] * self.slopes[:, None, None]
        
        return -alibi_bias  # Negative because we subtract from attention scores