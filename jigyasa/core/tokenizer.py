"""
Byte-level tokenizer for Jigyasa
Processes raw bytes without traditional tokenization
"""

import torch
import numpy as np
from typing import List, Dict, Union, Optional, Tuple
import json
import re
from pathlib import Path


class ByteTokenizer:
    """
    Tokenizer-free byte-level processor for universal text handling
    Operates directly on UTF-8 byte sequences
    """
    
    def __init__(self, max_length: int = 2048, pad_token_id: int = 0):
        self.max_length = max_length
        self.pad_token_id = pad_token_id
        self.vocab_size = 256  # All possible byte values
        
        # Special tokens
        self.special_tokens = {
            'pad_token': pad_token_id,
            'eos_token': 255,  # Use max byte value as EOS
            'bos_token': 254,  # Beginning of sequence
            'unk_token': 253,  # Unknown (though shouldn't be needed for bytes)
        }
        
    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """
        Encode text to byte sequence
        
        Args:
            text: Input text string
            add_special_tokens: Whether to add BOS/EOS tokens
        Returns:
            List of byte values (0-255)
        """
        # Convert string to UTF-8 bytes
        byte_sequence = text.encode('utf-8')
        
        # Convert bytes to list of integers
        byte_ids = list(byte_sequence)
        
        # Add special tokens if requested
        if add_special_tokens:
            byte_ids = [self.special_tokens['bos_token']] + byte_ids + [self.special_tokens['eos_token']]
        
        return byte_ids
    
    def decode(self, byte_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        Decode byte sequence to text
        
        Args:
            byte_ids: List of byte values
            skip_special_tokens: Whether to skip special tokens
        Returns:
            Decoded text string
        """
        # Filter out special tokens if requested
        if skip_special_tokens:
            filtered_ids = []
            for byte_id in byte_ids:
                if byte_id not in self.special_tokens.values():
                    filtered_ids.append(byte_id)
            byte_ids = filtered_ids
        
        # Ensure valid byte range
        byte_ids = [max(0, min(255, b)) for b in byte_ids]
        
        # Convert to bytes and decode
        try:
            byte_sequence = bytes(byte_ids)
            text = byte_sequence.decode('utf-8', errors='replace')
            return text
        except Exception as e:
            # Fallback for corrupted sequences
            return f"[DECODE_ERROR: {str(e)}]"
    
    def batch_encode(
        self, 
        texts: List[str], 
        padding: bool = True, 
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt"
    ) -> Dict[str, Union[List[List[int]], torch.Tensor]]:
        """
        Batch encode multiple texts
        
        Args:
            texts: List of input texts
            padding: Whether to pad sequences
            truncation: Whether to truncate long sequences
            max_length: Maximum sequence length
            return_tensors: Format of returned tensors ("pt" for PyTorch)
        Returns:
            Dictionary with input_ids and attention_mask
        """
        if max_length is None:
            max_length = self.max_length
        
        # Encode all texts
        encoded_texts = [self.encode(text) for text in texts]
        
        # Truncate if necessary
        if truncation:
            encoded_texts = [seq[:max_length] for seq in encoded_texts]
        
        # Pad sequences
        if padding:
            max_len = max(len(seq) for seq in encoded_texts) if encoded_texts else 0
            max_len = min(max_len, max_length)
            
            padded_sequences = []
            attention_masks = []
            
            for seq in encoded_texts:
                # Pad sequence
                padding_length = max_len - len(seq)
                padded_seq = seq + [self.pad_token_id] * padding_length
                padded_sequences.append(padded_seq)
                
                # Create attention mask
                attention_mask = [1] * len(seq) + [0] * padding_length
                attention_masks.append(attention_mask)
            
            encoded_texts = padded_sequences
        else:
            # Create attention masks without padding
            attention_masks = [[1] * len(seq) for seq in encoded_texts]
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            import torch
            result = {
                'input_ids': torch.tensor(encoded_texts, dtype=torch.long),
                'attention_mask': torch.tensor(attention_masks, dtype=torch.long)
            }
        else:
            result = {
                'input_ids': encoded_texts,
                'attention_mask': attention_masks
            }
        
        return result
    
    def create_patches(
        self, 
        byte_sequence: List[int], 
        patch_size: int = 16,
        overlap: int = 4
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Create overlapping byte patches for B.L.T. architecture
        
        Args:
            byte_sequence: Input byte sequence
            patch_size: Target patch size
            overlap: Overlap between patches
        Returns:
            Tuple of (patches, patch_lengths)
        """
        if len(byte_sequence) == 0:
            return [[]], [0]
        
        patches = []
        patch_lengths = []
        
        # Calculate entropy to determine patch boundaries
        entropy_scores = self._calculate_entropy(byte_sequence)
        
        # Create adaptive patches based on entropy
        start = 0
        while start < len(byte_sequence):
            # Determine patch end based on entropy and constraints
            end = min(start + patch_size, len(byte_sequence))
            
            # Adjust end based on entropy (try to break at low-entropy regions)
            if end < len(byte_sequence):
                # Look for low entropy region within window
                window_start = max(start + patch_size // 2, end - patch_size // 2)
                window_end = min(end + patch_size // 2, len(byte_sequence))
                
                if window_start < window_end:
                    window_entropies = entropy_scores[window_start:window_end]
                    min_entropy_idx = np.argmin(window_entropies)
                    end = window_start + min_entropy_idx
            
            # Extract patch
            patch = byte_sequence[start:end]
            patches.append(patch)
            patch_lengths.append(len(patch))
            
            # Move to next patch with overlap
            start = max(start + 1, end - overlap)
        
        return patches, patch_lengths
    
    def _calculate_entropy(self, byte_sequence: List[int], window_size: int = 8) -> np.ndarray:
        """
        Calculate local entropy for adaptive patching
        
        Args:
            byte_sequence: Input byte sequence
            window_size: Window size for entropy calculation
        Returns:
            Array of entropy scores
        """
        if len(byte_sequence) < window_size:
            return np.zeros(len(byte_sequence))
        
        entropies = []
        
        for i in range(len(byte_sequence) - window_size + 1):
            window = byte_sequence[i:i + window_size]
            
            # Calculate byte frequency
            unique, counts = np.unique(window, return_counts=True)
            probabilities = counts / len(window)
            
            # Calculate entropy
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            entropies.append(entropy)
        
        # Pad to match sequence length
        entropies.extend([entropies[-1]] * (window_size - 1))
        
        return np.array(entropies)
    
    def save_pretrained(self, save_directory: str):
        """Save tokenizer configuration"""
        save_path = Path(save_directory)
        save_path.mkdir(parents=True, exist_ok=True)
        
        config = {
            'max_length': self.max_length,
            'pad_token_id': self.pad_token_id,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens,
            'tokenizer_class': 'ByteTokenizer'
        }
        
        with open(save_path / 'tokenizer_config.json', 'w') as f:
            json.dump(config, f, indent=2)
    
    @classmethod
    def from_pretrained(cls, load_directory: str) -> 'ByteTokenizer':
        """Load tokenizer from saved configuration"""
        load_path = Path(load_directory)
        
        with open(load_path / 'tokenizer_config.json', 'r') as f:
            config = json.load(f)
        
        tokenizer = cls(
            max_length=config['max_length'],
            pad_token_id=config['pad_token_id']
        )
        tokenizer.special_tokens = config['special_tokens']
        
        return tokenizer
    
    def get_vocab(self) -> Dict[int, str]:
        """Get vocabulary mapping (byte values to their representations)"""
        vocab = {}
        for i in range(256):
            if i in self.special_tokens.values():
                # Find special token name
                for name, token_id in self.special_tokens.items():
                    if token_id == i:
                        vocab[i] = f"<{name}>"
                        break
            else:
                # Regular byte value
                try:
                    vocab[i] = bytes([i]).decode('utf-8', errors='replace')
                except:
                    vocab[i] = f"<byte_{i}>"
        
        return vocab
    
    def convert_tokens_to_ids(self, tokens: List[str]) -> List[int]:
        """Convert token strings to byte IDs (for compatibility)"""
        # This is primarily for compatibility with HuggingFace interfaces
        result = []
        for token in tokens:
            if token.startswith('<') and token.endswith('>'):
                # Special token
                token_name = token[1:-1]
                if token_name in self.special_tokens:
                    result.append(self.special_tokens[token_name])
                else:
                    result.append(self.special_tokens['unk_token'])
            else:
                # Regular text - encode and take first byte
                byte_ids = self.encode(token, add_special_tokens=False)
                result.extend(byte_ids)
        
        return result
    
    def convert_ids_to_tokens(self, ids: List[int]) -> List[str]:
        """Convert byte IDs to token strings (for compatibility)"""
        vocab = self.get_vocab()
        return [vocab.get(id, f"<byte_{id}>") for id in ids]