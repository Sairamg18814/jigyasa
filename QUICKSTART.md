# 🚀 Jigyasa AGI - Quick Start Guide

## Step-by-Step Instructions to Run Jigyasa

### 1️⃣ Check System Requirements
```bash
python3 run_jigyasa.py --mode check
```

This will:
- ✓ Check all dependencies
- ✓ Create necessary directories
- ✓ Verify your system is ready

### 2️⃣ Run Basic Test
```bash
python3 run_jigyasa.py --mode test
```

This will:
- ✓ Test core imports
- ✓ Create a small model
- ✓ Verify forward pass works
- ✓ Test tokenizer

### 3️⃣ Try Interactive Mode
```bash
python3 run_jigyasa.py --mode interactive
```

This starts the AGI in interactive mode where you can:
- Ask questions
- See self-correction in action
- Test reasoning capabilities
- Experience the "thinking before answering" feature

### 4️⃣ See Full Guide
```bash
python3 run_jigyasa.py --mode guide
```

Shows comprehensive documentation including:
- Installation instructions
- Usage examples
- Component guides
- Training pipeline

## 📋 Quick Commands

### Basic Usage
```python
from jigyasa.config import JigyasaConfig
from jigyasa.main import JigyasaSystem

# Create and initialize
config = JigyasaConfig()
system = JigyasaSystem(config)
system.initialize()

# Interactive mode
system.interactive_mode()
```

### Test Specific Components

**Cognitive Architecture:**
```python
from jigyasa.cognitive.architecture import CognitiveArchitecture

cog = CognitiveArchitecture(model_dim=256)
state = cog.introspect()
print(state['capabilities'])
```

**Reasoning System:**
```python
from jigyasa.reasoning import NeuroSymbolicReasoner
import torch.nn as nn

model = nn.Linear(256, 256)  # Dummy model
reasoner = NeuroSymbolicReasoner(model)
result = reasoner.mathematical_reasoning.solve_equation("x + 5 = 10")
```

## 🐛 Troubleshooting

### Missing Dependencies
```bash
pip install torch numpy transformers einops peft sympy networkx beautifulsoup4 requests
```

### Import Errors
Make sure you're in the jigyasa directory:
```bash
cd /Volumes/asus\ ssd/jigyasa
python3 run_jigyasa.py --mode test
```

### Memory Issues
Use smaller model configuration:
```python
config = JigyasaConfig()
config.model.d_model = 128  # Smaller
config.model.n_layers = 2   # Fewer layers
```

## 🎯 What's Implemented

✅ **Core Architecture**
- Byte Latent Transformer (B.L.T.)
- Tokenizer-free processing
- Adaptive computation

✅ **Cognitive Components**
- SEAL continuous learning
- ProRL reasoning improvement
- Self-correction mechanisms
- Metacognition & self-awareness

✅ **Reasoning Systems**
- Neuro-symbolic reasoning
- Mathematical solver
- Logical inference
- Causal reasoning

✅ **Safety & Governance**
- Constitutional AI
- Harm detection
- Value alignment
- Ethical constraints

✅ **Agentic Capabilities**
- Task planning
- Tool use
- Memory systems
- Autonomous execution

✅ **Infrastructure**
- Distributed training
- Model compression
- CI/CD pipeline
- Deployment tools

## 📊 System Architecture

```
Jigyasa AGI
├── Core (B.L.T.)
│   ├── Byte-level processing
│   ├── Dynamic patching
│   └── Adaptive computation
├── Cognitive
│   ├── SEAL (continuous learning)
│   ├── ProRL (reasoning improvement)
│   ├── Self-correction
│   └── Metacognition
├── Reasoning
│   ├── Neuro-symbolic
│   ├── Mathematical
│   ├── Logical
│   └── Causal
├── Governance
│   ├── Constitutional AI
│   ├── Safety checks
│   └── Value alignment
└── Agentic
    ├── Planning
    ├── Tool use
    ├── Memory
    └── Execution
```

## 🎉 Next Steps

1. **Explore Components**: Try different modules individually
2. **Train Small Model**: Run mini training pipeline
3. **Customize**: Modify configs for your use case
4. **Deploy**: Compress and deploy on laptop

Happy exploring with Jigyasa AGI! 🚀