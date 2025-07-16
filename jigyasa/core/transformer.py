"""
Core Transformer implementation from scratch
Based on "Attention Is All You Need" with modern optimizations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 50000
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_length: int = 2048
    dropout: float = 0.1
    layer_norm_eps: float = 1e-5
    use_bias: bool = True
    use_flash_attention: bool = True
    
    def get(self, key, default=None):
        """Make config compatible with dict-like access for PEFT"""
        return getattr(self, key, default)


class MultiHeadAttention(nn.Module):
    """Multi-Head Self-Attention mechanism"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.d_k = config.d_model // config.n_heads
        
        assert config.d_model % config.n_heads == 0, "d_model must be divisible by n_heads"
        
        # Linear projections for Q, K, V
        self.w_q = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.w_k = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.w_v = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        self.w_o = nn.Linear(config.d_model, config.d_model, bias=config.use_bias)
        
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask of shape (batch_size, seq_len, seq_len)
            past_key_value: Cached key-value pairs for generation
            use_cache: Whether to return cached key-value pairs
        """
        batch_size, seq_len, d_model = x.size()
        
        # Linear projections
        Q = self.w_q(x)  # (batch_size, seq_len, d_model)
        K = self.w_k(x)  # (batch_size, seq_len, d_model)
        V = self.w_v(x)  # (batch_size, seq_len, d_model)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        K = K.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        V = V.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)  # (batch_size, n_heads, seq_len, d_k)
        
        # Handle cached key-value pairs for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            K = torch.cat([past_key, K], dim=-2)
            V = torch.cat([past_value, V], dim=-2)
        
        # Cache key-value pairs if requested
        present_key_value = (K, V) if use_cache else None
        
        # Scaled dot-product attention
        if self.config.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's optimized flash attention if available
            attn_output = F.scaled_dot_product_attention(
                Q, K, V, 
                attn_mask=mask,
                dropout_p=self.config.dropout if self.training else 0.0,
                is_causal=mask is None  # Assume causal if no mask provided
            )
        else:
            # Manual attention computation
            # Compute attention scores
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # (batch_size, n_heads, seq_len, seq_len)
            
            # Apply mask if provided
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            # Apply causal mask for autoregressive generation
            if mask is None:
                causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
                scores = scores.masked_fill(~causal_mask, float('-inf'))
            
            # Softmax to get attention weights
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_weights, V)  # (batch_size, n_heads, seq_len, d_k)
        
        # Concatenate heads and put through final linear layer
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.w_o(attn_output)
        
        return output, present_key_value


class FeedForward(nn.Module):
    """Position-wise Feed-Forward Network"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.w_1 = nn.Linear(config.d_model, config.d_ff, bias=config.use_bias)
        self.w_2 = nn.Linear(config.d_ff, config.d_model, bias=config.use_bias)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.w_2(self.dropout(F.gelu(self.w_1(x))))


class TransformerBlock(nn.Module):
    """Single Transformer Block with Multi-Head Attention and Feed-Forward"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.feed_forward = FeedForward(config)
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Attention mask
            past_key_value: Cached key-value pairs
            use_cache: Whether to return cached key-value pairs
        """
        # Pre-LayerNorm architecture for better training stability
        # Self-attention with residual connection
        attn_input = self.ln_1(x)
        attn_output, present_key_value = self.attention(
            attn_input, mask=mask, past_key_value=past_key_value, use_cache=use_cache
        )
        x = x + self.dropout(attn_output)
        
        # Feed-forward with residual connection
        ff_input = self.ln_2(x)
        ff_output = self.feed_forward(ff_input)
        x = x + self.dropout(ff_output)
        
        return x, present_key_value


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.d_model = config.d_model
        self.max_seq_length = config.max_seq_length
        
        # Create positional encoding matrix
        pe = torch.zeros(config.max_seq_length, config.d_model)
        position = torch.arange(0, config.max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, config.d_model, 2).float() * 
                           (-math.log(10000.0) / config.d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Positionally encoded tensor
        """
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len, :]


class JigyasaTransformer(nn.Module):
    """Complete Transformer model implementation"""
    
    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.n_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(config.d_model, eps=config.layer_norm_eps)
        
        # Output projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Tie weights between embedding and output projection
        self.lm_head.weight = self.token_embedding.weight
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize parameters
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model parameters"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
            
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            input_ids: Token indices of shape (batch_size, seq_len)
            attention_mask: Mask for attention computation
            past_key_values: Cached key-value pairs for generation
            use_cache: Whether to return cached key-value pairs
            labels: Target labels for training
        """
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        x = self.token_embedding(input_ids)  # (batch_size, seq_len, d_model)
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Prepare attention mask
        if attention_mask is not None:
            # Convert to 4D mask for multi-head attention
            extended_attention_mask = attention_mask[:, None, None, :]
            extended_attention_mask = extended_attention_mask.to(dtype=x.dtype)
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(x.dtype).min
        else:
            extended_attention_mask = None
        
        # Process through transformer blocks
        present_key_values = [] if use_cache else None
        
        for i, block in enumerate(self.blocks):
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            x, present_key_value = block(
                x, 
                mask=extended_attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache
            )
            
            if use_cache:
                present_key_values.append(present_key_value)
        
        # Final layer norm
        x = self.ln_f(x)
        
        # Language modeling head
        logits = self.lm_head(x)  # (batch_size, seq_len, vocab_size)
        
        # Calculate loss if labels provided
        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return {
            'logits': logits,
            'loss': loss,
            'past_key_values': present_key_values if use_cache else None,
            'hidden_states': x
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None
    ) -> torch.Tensor:
        """
        Generate text using the model
        """
        self.eval()
        
        batch_size = input_ids.size(0)
        device = input_ids.device
        
        # Initialize generation
        past_key_values = None
        generated = input_ids.clone()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Forward pass
                outputs = self.forward(
                    input_ids=input_ids if past_key_values is None else input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
                
                logits = outputs['logits'][:, -1, :]  # Get last token logits
                past_key_values = outputs['past_key_values']
                
                # Apply temperature
                logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                    logits[indices_to_remove] = float('-inf')
                
                # Apply top-p (nucleus) filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    logits[indices_to_remove] = float('-inf')
                
                # Sample next token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_tokens = torch.multinomial(probs, num_samples=1)
                else:
                    next_tokens = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_tokens], dim=1)
                input_ids = next_tokens
                
                # Check for EOS token
                if eos_token_id is not None and (next_tokens == eos_token_id).all():
                    break
        
        return generated


# Alias for compatibility
ByteLatentTransformer = JigyasaTransformer