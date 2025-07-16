"""
Configuration management for Jigyasa
"""

import os
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import json


@dataclass
class ModelConfig:
    """Configuration for the core model architecture"""
    
    # B.L.T. Architecture
    vocab_size: int = 50000
    d_model: int = 768
    n_heads: int = 12
    n_layers: int = 12
    d_ff: int = 3072
    max_seq_length: int = 2048
    dropout: float = 0.1
    
    # Byte-level processing
    byte_vocab_size: int = 256
    patch_size: int = 16
    max_patch_length: int = 128
    
    # Attention mechanisms
    attention_type: str = "multi_head"
    use_rotary_embeddings: bool = True
    use_flash_attention: bool = True


@dataclass
class SEALConfig:
    """Configuration for Self-Adapting Language Models"""
    
    # Learning parameters
    learning_rate: float = 1e-4
    adaptation_steps: int = 100
    inner_loop_steps: int = 5
    outer_loop_steps: int = 20
    
    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: ["w_q", "w_v"])
    
    # Self-edit generation
    max_self_edit_length: int = 512
    self_edit_temperature: float = 0.7


@dataclass
class ProRLConfig:
    """Configuration for Prolonged Reinforcement Learning"""
    
    # Training parameters
    num_episodes: int = 10000
    max_episode_length: int = 1000
    batch_size: int = 32
    learning_rate: float = 3e-4
    
    # KL divergence control
    kl_penalty: float = 0.1
    target_kl: float = 0.01
    
    # Reference policy resetting
    reset_frequency: int = 1000
    
    # Reward model
    reward_model_path: Optional[str] = None
    use_verifiable_rewards: bool = True


@dataclass
class DataConfig:
    """Configuration for autonomous data acquisition"""
    
    # Web scraping
    max_pages_per_query: int = 100
    scraping_delay: float = 1.0
    user_agent: str = "Jigyasa-Bot/1.0"
    
    # Data quality
    min_text_length: int = 100
    max_text_length: int = 10000
    remove_pii: bool = True
    bias_detection: bool = True
    
    # Storage
    data_cache_dir: str = "./data/cache"
    processed_data_dir: str = "./data/processed"


@dataclass
class AgenticConfig:
    """Configuration for agentic framework"""
    
    # Tool management
    max_tool_calls: int = 10
    tool_timeout: float = 30.0
    
    # RAG parameters
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 5
    
    # Agent behavior
    proactivity_threshold: float = 0.7
    personalization_enabled: bool = True
    memory_window: int = 1000


@dataclass
class CompressionConfig:
    """Configuration for model compression"""
    
    # Knowledge distillation
    teacher_model_path: Optional[str] = None
    student_compression_ratio: float = 0.25
    distillation_temperature: float = 4.0
    
    # Pruning
    pruning_ratio: float = 0.5
    structured_pruning: bool = True
    
    # Quantization
    quantization_bits: int = 4
    use_gguf: bool = True


@dataclass
class JigyasaConfig:
    """Main configuration class for Jigyasa"""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    seal: SEALConfig = field(default_factory=SEALConfig)
    prorl: ProRLConfig = field(default_factory=ProRLConfig)
    data: DataConfig = field(default_factory=DataConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    
    # Global settings
    device: str = "auto"
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    
    # Paths
    model_save_path: str = "./models"
    log_dir: str = "./logs"
    cache_dir: str = "./cache"
    
    def save(self, path: str):
        """Save configuration to JSON file"""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(self.__dict__, f, indent=2, default=str)
    
    @classmethod
    def load(cls, path: str) -> 'JigyasaConfig':
        """Load configuration from JSON file"""
        with open(path, 'r') as f:
            data = json.load(f)
        return cls(**data)
    
    @classmethod
    def from_env(cls) -> 'JigyasaConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # Override with environment variables
        if os.getenv("JIGYASA_DEVICE"):
            config.device = os.getenv("JIGYASA_DEVICE")
        if os.getenv("JIGYASA_MODEL_PATH"):
            config.model_save_path = os.getenv("JIGYASA_MODEL_PATH")
        if os.getenv("JIGYASA_LOG_DIR"):
            config.log_dir = os.getenv("JIGYASA_LOG_DIR")
            
        return config


# Default configuration instance
default_config = JigyasaConfig()