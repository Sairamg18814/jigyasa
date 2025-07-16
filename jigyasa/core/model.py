"""
Main Jigyasa model implementation
Combines transformer architecture with byte-level processing
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List, Union, Tuple
from pathlib import Path
import json

from .transformer import JigyasaTransformer, TransformerConfig
from .tokenizer import ByteTokenizer
from .embeddings import ByteEmbedding, RotaryEmbedding
from ..config import JigyasaConfig


class JigyasaModel(nn.Module):
    """
    Main Jigyasa model combining transformer architecture with byte-level processing
    """
    
    def __init__(self, config: Union[JigyasaConfig, TransformerConfig]):
        super().__init__()
        
        # Handle config types
        if isinstance(config, JigyasaConfig):
            self.jigyasa_config = config
            self.transformer_config = TransformerConfig(
                vocab_size=config.model.vocab_size,
                d_model=config.model.d_model,
                n_heads=config.model.n_heads,
                n_layers=config.model.n_layers,
                d_ff=config.model.d_ff,
                max_seq_length=config.model.max_seq_length,
                dropout=config.model.dropout
            )
        else:
            self.transformer_config = config
            self.jigyasa_config = None
        
        # Core transformer
        self.transformer = JigyasaTransformer(self.transformer_config)
        
        # Byte-level components
        self.byte_embedding = ByteEmbedding(
            d_model=self.transformer_config.d_model,
            max_byte_value=256
        )
        
        # Rotary position embedding (optional enhancement)
        self.rope = RotaryEmbedding(
            dim=self.transformer_config.d_model // self.transformer_config.n_heads,
            max_seq_len=self.transformer_config.max_seq_length
        )
        
        # Tokenizer
        self.tokenizer = ByteTokenizer(max_length=self.transformer_config.max_seq_length)
        
        # Model metadata
        self.model_type = "jigyasa"
        self.config = self.transformer_config
        
    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        use_cache: bool = False,
        labels: Optional[torch.Tensor] = None,
        byte_input: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the model
        
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
            past_key_values: Cached key-value pairs for generation
            use_cache: Whether to return cached key-value pairs
            labels: Target labels for training
            byte_input: Direct byte input (alternative to input_ids)
        """
        # Handle byte input if provided
        if byte_input is not None:
            input_ids = byte_input
            
        # Ensure input_ids are provided
        if input_ids is None:
            raise ValueError("Either input_ids or byte_input must be provided")
        
        # Forward through transformer
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            labels=labels
        )
        
        return outputs
    
    def generate(
        self,
        input_ids: Optional[torch.Tensor] = None,
        input_text: Optional[str] = None,
        max_length: int = 100,
        max_new_tokens: Optional[int] = None,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        num_return_sequences: int = 1,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        repetition_penalty: float = 1.2,
        **kwargs
    ) -> Union[torch.Tensor, List[str]]:
        """
        Generate text using the model
        
        Args:
            input_ids: Input token IDs
            input_text: Input text (alternative to input_ids)
            max_length: Maximum total length
            max_new_tokens: Maximum new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to use sampling
            num_return_sequences: Number of sequences to return
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
        """
        # Handle text input
        if input_text is not None:
            tokenized = self.tokenizer.batch_encode([input_text], return_tensors="pt")
            input_ids = tokenized['input_ids']
            
        if input_ids is None:
            raise ValueError("Either input_ids or input_text must be provided")
        
        # Set default token IDs
        if pad_token_id is None:
            pad_token_id = self.tokenizer.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.tokenizer.special_tokens['eos_token']
        
        # Calculate max_new_tokens if not provided
        if max_new_tokens is not None:
            max_length = input_ids.size(1) + max_new_tokens
        
        # Expand input for multiple sequences
        if num_return_sequences > 1:
            input_ids = input_ids.repeat(num_return_sequences, 1)
        
        # Generate using transformer
        generated_ids = self.transformer.generate(
            input_ids=input_ids,
            max_length=max_length,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            repetition_penalty=repetition_penalty
        )
        
        # Decode if text input was provided
        if input_text is not None:
            generated_texts = []
            for seq in generated_ids:
                text = self.tokenizer.decode(seq.tolist(), skip_special_tokens=True)
                generated_texts.append(text)
            return generated_texts if num_return_sequences > 1 else generated_texts[0]
        
        return generated_ids
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Optional[Tuple] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Prepare inputs for generation (required for HuggingFace compatibility)"""
        # If past_key_values are provided, only use the last token
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]
            
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
        }
        
        # Add attention mask if provided
        if attention_mask is not None:
            model_inputs["attention_mask"] = attention_mask
            
        return model_inputs
    
    def get_input_embeddings(self) -> nn.Module:
        """Get input embedding layer"""
        return self.transformer.token_embedding
    
    def set_input_embeddings(self, new_embeddings: nn.Module):
        """Set input embedding layer"""
        self.transformer.token_embedding = new_embeddings
    
    def get_output_embeddings(self) -> nn.Module:
        """Get output embedding layer"""
        return self.transformer.lm_head
    
    def set_output_embeddings(self, new_embeddings: nn.Module):
        """Set output embedding layer"""
        self.transformer.lm_head = new_embeddings
    
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Module:
        """Resize token embeddings (for compatibility)"""
        old_embeddings = self.get_input_embeddings()
        new_embeddings = nn.Embedding(new_num_tokens, old_embeddings.embedding_dim)
        
        # Copy old weights
        num_tokens_to_copy = min(old_embeddings.num_embeddings, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]
        
        self.set_input_embeddings(new_embeddings)
        
        # Update config
        self.config.vocab_size = new_num_tokens
        
        return new_embeddings
    
    def save_pretrained(
        self,
        save_directory: str,
        save_config: bool = True,
        save_tokenizer: bool = True,
        **kwargs
    ):
        """Save model, config, and tokenizer"""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        
        # Save config
        if save_config:
            config_dict = {
                'model_type': self.model_type,
                'transformer_config': {
                    'vocab_size': self.transformer_config.vocab_size,
                    'd_model': self.transformer_config.d_model,
                    'n_heads': self.transformer_config.n_heads,
                    'n_layers': self.transformer_config.n_layers,
                    'd_ff': self.transformer_config.d_ff,
                    'max_seq_length': self.transformer_config.max_seq_length,
                    'dropout': self.transformer_config.dropout,
                    'layer_norm_eps': self.transformer_config.layer_norm_eps,
                    'use_bias': self.transformer_config.use_bias,
                    'use_flash_attention': self.transformer_config.use_flash_attention
                }
            }
            
            with open(save_path / 'config.json', 'w') as f:
                json.dump(config_dict, f, indent=2)
        
        # Save tokenizer
        if save_tokenizer:
            self.tokenizer.save_pretrained(save_directory)
    
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        config: Optional[Union[JigyasaConfig, TransformerConfig]] = None,
        **kwargs
    ) -> 'JigyasaModel':
        """Load pre-trained model"""
        load_path = Path(pretrained_model_name_or_path)
        
        # Load config if not provided
        if config is None:
            config_path = load_path / 'config.json'
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = TransformerConfig(**config_dict['transformer_config'])
            else:
                raise ValueError(f"Config file not found at {config_path}")
        
        # Create model
        model = cls(config)
        
        # Load state dict
        state_dict_path = load_path / 'pytorch_model.bin'
        if state_dict_path.exists():
            state_dict = torch.load(state_dict_path, map_location='cpu')
            model.load_state_dict(state_dict)
        else:
            print(f"Warning: Model weights not found at {state_dict_path}")
        
        # Load tokenizer
        tokenizer_config_path = load_path / 'tokenizer_config.json'
        if tokenizer_config_path.exists():
            model.tokenizer = ByteTokenizer.from_pretrained(pretrained_model_name_or_path)
        
        return model
    
    def get_memory_footprint(self) -> Dict[str, int]:
        """Calculate model memory footprint"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory in bytes (assuming float32)
        model_memory = total_params * 4
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_memory_bytes': model_memory,
            'model_memory_mb': model_memory / (1024 * 1024),
            'model_memory_gb': model_memory / (1024 * 1024 * 1024)
        }
    
    def print_model_info(self):
        """Print model information"""
        memory_info = self.get_memory_footprint()
        
        print(f"Jigyasa Model Information:")
        print(f"  Model Type: {self.model_type}")
        print(f"  Total Parameters: {memory_info['total_parameters']:,}")
        print(f"  Trainable Parameters: {memory_info['trainable_parameters']:,}")
        print(f"  Memory Footprint: {memory_info['model_memory_mb']:.1f} MB")
        print(f"  Transformer Config:")
        print(f"    - Vocabulary Size: {self.transformer_config.vocab_size:,}")
        print(f"    - Model Dimension: {self.transformer_config.d_model}")
        print(f"    - Number of Heads: {self.transformer_config.n_heads}")
        print(f"    - Number of Layers: {self.transformer_config.n_layers}")
        print(f"    - Feed-Forward Dimension: {self.transformer_config.d_ff}")
        print(f"    - Max Sequence Length: {self.transformer_config.max_seq_length}")
        print(f"    - Dropout: {self.transformer_config.dropout}")
        print(f"    - Flash Attention: {self.transformer_config.use_flash_attention}")


# Convenience function for model creation
def create_jigyasa_model(
    vocab_size: int = 256,  # Byte-level vocabulary
    d_model: int = 768,
    n_heads: int = 12,
    n_layers: int = 12,
    d_ff: int = 3072,
    max_seq_length: int = 2048,
    dropout: float = 0.1,
    use_gpt2: bool = True,  # Use GPT-2 by default
    **kwargs
) -> JigyasaModel:
    """Create a Jigyasa model with specified parameters"""
    
    # Use GPT-2 for now to avoid garbled output
    if use_gpt2:
        import sys
        sys.path.append("/Volumes/asus ssd/jigyasa")
        from jigyasa.models.gpt2_wrapper import GPT2Wrapper
        return GPT2Wrapper(model_name="gpt2", device="cpu")
    
    # Original Jigyasa model creation
    config = TransformerConfig(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_seq_length=max_seq_length,
        dropout=dropout,
        **kwargs
    )
    
    return JigyasaModel(config)