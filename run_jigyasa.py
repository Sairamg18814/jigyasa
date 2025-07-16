#!/usr/bin/env python3
"""
Jigyasa AGI System Runner
Complete guide to running the system step by step
"""

import sys
import os
import subprocess
import argparse
from pathlib import Path

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def check_dependencies():
    """Check if all required dependencies are installed"""
    print("\n📦 Checking Dependencies...")
    
    required_packages = [
        'torch',
        'numpy',
        'transformers',
        'einops',
        'peft',
        'sympy',
        'networkx',
        'bs4',  # beautifulsoup4 imports as bs4
        'requests',
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - MISSING")
            missing.append(package)
    
    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("\nInstall them with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    print("\n✅ All dependencies installed!")
    return True


def create_model_directory():
    """Create necessary directories"""
    print("\n📁 Creating Directories...")
    
    dirs = [
        'models',
        'logs',
        'data',
        'checkpoints',
        'deployment'
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        print(f"✓ Created {dir_name}/")
    
    print("\n✅ Directories ready!")


def run_minimal_test():
    """Run a minimal test to verify basic functionality"""
    print("\n🧪 Running Minimal Test...")
    
    try:
        # Test imports
        from jigyasa.config import JigyasaConfig
        from jigyasa.core.transformer import JigyasaTransformer, TransformerConfig
        from jigyasa.core.blt import BLTTokenizer
        print("✓ Core imports successful")
        
        # Create minimal config
        config = TransformerConfig(
            d_model=256,
            n_heads=8,
            n_layers=2,
            vocab_size=256,
            max_seq_length=512
        )
        
        # Create model
        model = JigyasaTransformer(config)
        print(f"✓ Created model with {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
        
        # Test tokenizer
        tokenizer = BLTTokenizer()
        text = "Hello AGI!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"✓ Tokenizer test: '{text}' -> {len(tokens)} bytes -> '{decoded}'")
        
        # Test forward pass
        import torch
        input_ids = torch.randint(0, 256, (1, 50))
        with torch.no_grad():
            output = model(input_ids)
        print(f"✓ Forward pass successful: {output['logits'].shape}")
        
        print("\n✅ Basic functionality verified!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_interactive_demo():
    """Run interactive demo"""
    print("\n🚀 Starting Interactive Demo...")
    
    try:
        from jigyasa.config import JigyasaConfig
        from jigyasa.main import JigyasaSystem
        
        # Create config
        config = JigyasaConfig()
        config.model.d_model = 256  # Smaller for demo
        config.model.n_heads = 8
        config.model.n_layers = 4
        
        # Initialize system
        print("\n🧠 Initializing Jigyasa AGI System...")
        system = JigyasaSystem(config)
        system.initialize()
        
        print("\n✅ System Ready!")
        print("\n" + "="*60)
        print("JIGYASA AGI - INTERACTIVE MODE")
        print("="*60)
        print("\nCapabilities:")
        print("- 🤔 Self-correcting reasoning")
        print("- 📚 Continuous learning")
        print("- 🧮 Mathematical reasoning")
        print("- 🎯 Goal-oriented planning")
        print("- 🛡️ Constitutional AI safety")
        print("\nType 'help' for commands, 'quit' to exit")
        print("="*60)
        
        # Run interactive mode
        system.interactive_mode()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


def show_step_by_step_guide():
    """Show detailed step-by-step guide"""
    print("\n" + "="*60)
    print("📖 JIGYASA AGI - STEP BY STEP GUIDE")
    print("="*60)
    
    guide = """
1️⃣ INSTALLATION
   ```bash
   # Clone the repository (if not already done)
   git clone https://github.com/your-repo/jigyasa.git
   cd jigyasa
   
   # Install dependencies
   pip install -r jigyasa/requirements.txt
   ```

2️⃣ BASIC USAGE - Interactive Mode
   ```python
   from jigyasa.config import JigyasaConfig
   from jigyasa.main import JigyasaSystem
   
   # Create system
   config = JigyasaConfig()
   system = JigyasaSystem(config)
   system.initialize()
   
   # Run interactive mode
   system.interactive_mode()
   ```

3️⃣ TRAINING PIPELINE - Full AGI Training
   ```python
   # Phase 1: Foundational Training
   results = system.phase1_foundational_training()
   
   # Phase 2: Continuous Learning
   topics = ["AI", "science", "philosophy"]
   system.phase2_continuous_learning(topics)
   
   # Phase 3: Compression for Deployment
   system.phase3_compression_deployment()
   ```

4️⃣ USING SPECIFIC COMPONENTS

   🧠 Cognitive Architecture:
   ```python
   from jigyasa.cognitive.architecture import CognitiveArchitecture
   
   cog = CognitiveArchitecture()
   result = await cog.think("What is consciousness?", depth=3)
   ```

   🔬 Neuro-Symbolic Reasoning:
   ```python
   from jigyasa.reasoning import NeuroSymbolicReasoner
   
   reasoner = NeuroSymbolicReasoner(model)
   result = reasoner.reason("Solve x^2 + 5x + 6 = 0")
   ```

   🛡️ Constitutional AI:
   ```python
   from jigyasa.governance import ConstitutionalAI
   
   gov = ConstitutionalAI(model)
   critique = gov.critique("response", "assistant")
   ```

   🤖 Agentic Framework:
   ```python
   from jigyasa.agentic import AgentCore
   
   agent = AgentCore()
   plan = await agent.plan_task("Research AGI safety")
   ```

5️⃣ DEPLOYMENT
   ```bash
   # After training, deploy the compressed model
   python run_jigyasa.py --mode deploy
   
   # The compressed model will be in deployment/
   # Size: ~500MB (fits on laptop!)
   ```

6️⃣ ADVANCED FEATURES
   - SEAL: Self-improving through generated data
   - ProRL: Discovers new reasoning strategies  
   - B.L.T: Tokenizer-free byte processing
   - Meta-learning: Adapts to new domains
   - Self-correction: Thinks before answering

💡 TIPS:
   - Start with interactive mode to test
   - Use smaller models for development
   - Monitor logs/ directory for training progress
   - Check deployment/ for compressed models
    """
    
    print(guide)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Jigyasa AGI System Runner")
    parser.add_argument(
        '--mode',
        choices=['test', 'interactive', 'train', 'guide', 'check'],
        default='guide',
        help='Execution mode'
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("🧠 JIGYASA AGI SYSTEM")
    print("A Self-Improving Artificial General Intelligence")
    print("="*60)
    
    if args.mode == 'check':
        # Just check dependencies
        if check_dependencies():
            create_model_directory()
            print("\n✅ System ready to run!")
            print("\nNext steps:")
            print("1. Run minimal test: python run_jigyasa.py --mode test")
            print("2. Try interactive mode: python run_jigyasa.py --mode interactive")
            print("3. See full guide: python run_jigyasa.py --mode guide")
    
    elif args.mode == 'test':
        # Run minimal test
        if check_dependencies():
            create_model_directory()
            if run_minimal_test():
                print("\n✅ All tests passed!")
                print("\nTry interactive mode: python run_jigyasa.py --mode interactive")
    
    elif args.mode == 'interactive':
        # Run interactive demo
        if check_dependencies():
            create_model_directory()
            run_interactive_demo()
    
    elif args.mode == 'train':
        # Full training pipeline
        print("\n🚀 Starting full training pipeline...")
        print("This will train Jigyasa through all 3 phases:")
        print("1. Foundational training (ProRL)")
        print("2. Continuous learning (SEAL)")
        print("3. Compression for deployment")
        print("\n⚠️  This will take significant time and compute!")
        
        response = input("\nContinue? (y/n): ")
        if response.lower() == 'y':
            os.system(f"{sys.executable} -m jigyasa.main --mode train")
    
    elif args.mode == 'guide':
        # Show guide
        show_step_by_step_guide()
        print("\n🎯 Quick Start:")
        print("1. Check system: python run_jigyasa.py --mode check")
        print("2. Run test: python run_jigyasa.py --mode test")
        print("3. Try it: python run_jigyasa.py --mode interactive")


if __name__ == "__main__":
    main()