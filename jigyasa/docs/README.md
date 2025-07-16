# Jigyasa Documentation

Welcome to the comprehensive documentation for Jigyasa, a self-improving, agentic language model framework.

## Table of Contents

1. [Quick Start](quickstart.md)
2. [Architecture Overview](architecture.md)
3. [Installation Guide](installation.md)
4. [Usage Examples](examples.md)
5. [API Reference](api.md)
6. [Training Guide](training.md)
7. [Deployment](deployment.md)
8. [Contributing](contributing.md)

## What is Jigyasa?

Jigyasa (Sanskrit: जिज्ञासा, meaning "curiosity" or "thirst for knowledge") is a comprehensive framework for building self-improving language models that combine:

- **Byte Latent Transformer (B.L.T.)**: Tokenizer-free architecture for universal data processing
- **Self-Adapting Language Models (SEAL)**: Continuous learning capabilities
- **Prolonged Reinforcement Learning (ProRL)**: Advanced reasoning through extended RL training
- **Autonomous Data Engine**: Web-scale data acquisition and preprocessing
- **Self-Correction**: Introspective "think before answering" capabilities
- **Constitutional AI**: Embedded ethical governance
- **On-Device Optimization**: Compressed models for laptop deployment

## Key Features

### 🧠 **True Self-Improvement**
- Continuous learning through SEAL framework
- Meta-learning for strategy optimization
- Self-generated training data

### 🚀 **Advanced Reasoning**
- ProRL training for complex problem solving
- Neuro-symbolic reasoning integration
- Chain-of-verification self-correction

### 🌐 **Autonomous Operation**
- Web data acquisition without human intervention
- Quality control and bias detection
- Adaptive learning strategies

### 💻 **On-Device Deployment**
- Teacher-student compression pipeline
- 4-bit quantization with GGUF format
- Optimized for laptop-class hardware

### 🛡️ **Safety & Ethics**
- Constitutional AI governance
- PII detection and removal
- Bias monitoring and mitigation

## Quick Example

```python
from jigyasa import JigyasaSystem

# Initialize system
system = JigyasaSystem()
system.initialize()

# Interactive mode with self-correction
result = system.self_correction.think_before_answer(
    query="Explain quantum computing",
    query_type="analytical"
)

print("Thinking Process:")
print(result['thinking_process'])
print("\nFinal Answer:")
print(result['final_response'])
print(f"\nConfidence: {result['confidence_score']:.2f}")
```

## Architecture Highlights

### Byte Latent Transformer (B.L.T.)
- Processes raw bytes instead of tokens
- Dynamic patch creation based on entropy
- Universal language and format support

### Cognitive Core
- **SEAL**: Self-adapting through generated training data
- **ProRL**: Extended RL for novel reasoning patterns
- **Self-Correction**: Multiple verification strategies

### Data Engine
- Autonomous web scraping and navigation
- Intelligent content extraction
- Quality filtering and bias detection

## Performance Targets

| Benchmark | Target Score | Status |
|-----------|-------------|---------|
| MMLU | >92% | 🎯 Target |
| RBench | >65% | 🎯 Target |
| HumanEval | >95% | 🎯 Target |
| FAI Benchmark | >85% | 🎯 Target |
| TruthfulQA | Top Tier | 🎯 Target |

## Getting Started

1. **Installation**: `pip install jigyasa`
2. **Quick Demo**: `python scripts/quick_start.py`
3. **Interactive Mode**: `jigyasa interactive`
4. **Full Training**: `jigyasa train --full-pipeline`

## System Requirements

### Minimum
- Python 3.8+
- 8GB RAM
- 50GB disk space
- CPU-only operation supported

### Recommended
- Python 3.10+
- 16GB+ RAM
- 100GB+ SSD storage
- NVIDIA GPU with 8GB+ VRAM

### For Full Training
- 32GB+ RAM
- 500GB+ storage
- High-end GPU (A100/H100 preferred)

## Community & Support

- **GitHub**: [jigyasa-ai/jigyasa](https://github.com/jigyasa-ai/jigyasa)
- **Documentation**: [jigyasa-ai.github.io](https://jigyasa-ai.github.io/jigyasa)
- **Issues**: [GitHub Issues](https://github.com/jigyasa-ai/jigyasa/issues)
- **Discussions**: [GitHub Discussions](https://github.com/jigyasa-ai/jigyasa/discussions)

## Citation

```bibtex
@software{jigyasa2025,
  title={Jigyasa: A Self-Improving, Agentic Language Model Framework},
  author={Jigyasa Development Team},
  year={2025},
  url={https://github.com/jigyasa-ai/jigyasa},
  note={Generated with Claude Code}
}
```

## License

MIT License - see [LICENSE](../LICENSE) for details.

---

**🚀 Generated with [Claude Code](https://claude.ai/code)**

Co-Authored-By: Claude <noreply@anthropic.com>