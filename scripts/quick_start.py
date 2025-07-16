#!/usr/bin/env python3
"""
Quick Start Script for Jigyasa
Demonstrates the complete system in action
"""

import os
import sys
import torch
from pathlib import Path

# Add the parent directory to path so we can import jigyasa
sys.path.insert(0, str(Path(__file__).parent.parent))

from jigyasa import JigyasaSystem, JigyasaConfig
from jigyasa.core.model import create_jigyasa_model


def quick_demo():
    """Quick demonstration of Jigyasa capabilities"""
    
    print("🚀 Welcome to Jigyasa Quick Start Demo!")
    print("=" * 50)
    
    # Create a minimal config for demo
    config = JigyasaConfig()
    
    # Use smaller model for quick demo
    config.model.d_model = 256
    config.model.n_heads = 8
    config.model.n_layers = 6
    config.model.max_seq_length = 512
    
    print("📋 Configuration:")
    print(f"  Model dimension: {config.model.d_model}")
    print(f"  Attention heads: {config.model.n_heads}")
    print(f"  Layers: {config.model.n_layers}")
    print(f"  Max sequence length: {config.model.max_seq_length}")
    
    # Initialize system
    print("\n🔧 Initializing Jigyasa system...")
    system = JigyasaSystem(config)
    system.initialize()
    
    model_params = sum(p.numel() for p in system.model.parameters())
    print(f"✅ Model created with {model_params:,} parameters")
    
    # Demo 1: Basic text generation
    print("\n🎯 Demo 1: Basic Text Generation")
    print("-" * 30)
    
    sample_prompt = "The future of artificial intelligence is"
    print(f"Prompt: '{sample_prompt}'")
    
    try:
        generated_text = system.model.generate(
            input_text=sample_prompt,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True
        )
        print(f"Generated: {generated_text}")
    except Exception as e:
        print(f"Generation error (expected for untrained model): {e}")
    
    # Demo 2: Self-correction capabilities
    print("\n🧠 Demo 2: Self-Correction (Think Before Answering)")
    print("-" * 50)
    
    test_question = "What is 15 + 27?"
    print(f"Question: {test_question}")
    
    try:
        correction_result = system.self_correction.think_before_answer(
            query=test_question,
            query_type='mathematical'
        )
        
        print(f"\n💭 Thinking Process:")
        print(correction_result['thinking_process'])
        
        print(f"\n✅ Final Answer:")
        print(correction_result['final_response'])
        
        print(f"\n📊 Confidence: {correction_result['confidence_score']:.2f}")
        
        if correction_result['corrections_made']:
            print(f"\n🔧 Corrections: {correction_result['corrections_made']}")
            
    except Exception as e:
        print(f"Self-correction demo error: {e}")
    
    # Demo 3: Data acquisition
    print("\n📊 Demo 3: Autonomous Data Acquisition")
    print("-" * 40)
    
    try:
        print("Attempting to collect data for 'machine learning'...")
        
        # This would normally collect real data from the web
        # For demo, we'll simulate it
        scraped_contents = system.data_engine.acquire_data_for_topic(
            topic="machine learning",
            max_sources=3  # Just a few for demo
        )
        
        print(f"📈 Collected {len(scraped_contents)} sources")
        
        for i, content in enumerate(scraped_contents[:2]):
            print(f"\nSource {i+1}:")
            print(f"  URL: {content.url}")
            print(f"  Title: {content.title}")
            print(f"  Quality Score: {content.quality_score:.2f}")
            print(f"  Content Length: {len(content.content)} characters")
            print(f"  Preview: {content.content[:100]}...")
            
    except Exception as e:
        print(f"Data acquisition demo error: {e}")
        print("(This is expected without internet or proper web scraping setup)")
    
    # Demo 4: System status
    print("\n📊 Demo 4: System Status")
    print("-" * 25)
    
    status = system.get_system_status()
    
    print("System Information:")
    for key, value in status.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for sub_key, sub_value in value.items():
                print(f"    {sub_key}: {sub_value}")
        else:
            print(f"  {key}: {value}")
    
    # Demo 5: Interactive prompt
    print("\n🎮 Demo 5: Mini Interactive Session")
    print("-" * 35)
    print("Ask me a question (or type 'skip' to skip):")
    
    try:
        user_input = input("You: ").strip()
        
        if user_input.lower() not in ['skip', 'quit', '']:
            print("\n🤔 Processing your question...")
            
            result = system.self_correction.think_before_answer(
                query=user_input,
                query_type='general'
            )
            
            print(f"\n💭 My thinking:")
            print(result['thinking_process'][:200] + "..." if len(result['thinking_process']) > 200 else result['thinking_process'])
            
            print(f"\n🤖 My answer:")
            print(result['final_response'])
            
    except KeyboardInterrupt:
        print("\nSkipping interactive demo...")
    except Exception as e:
        print(f"Interactive demo error: {e}")
    
    print("\n🎉 Jigyasa Quick Start Demo Completed!")
    print("=" * 50)
    print("\n📚 Next Steps:")
    print("1. Run 'python -m jigyasa.cli interactive' for full interactive mode")
    print("2. Run 'python -m jigyasa.cli train --full-pipeline' for complete training")
    print("3. Run 'python -m jigyasa.cli config --create' to create custom config")
    print("4. Check the documentation in docs/ for detailed usage")
    
    print("\n🌟 Thank you for trying Jigyasa!")


def check_requirements():
    """Check if basic requirements are met"""
    print("🔍 Checking requirements...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('numpy', 'NumPy'),
    ]
    
    missing_packages = []
    
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name}")
        except ImportError:
            print(f"  ❌ {name} (missing)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All basic requirements met!")
    return True


def main():
    """Main function"""
    print("🧠 Jigyasa Quick Start")
    print("=" * 20)
    
    # Check requirements
    if not check_requirements():
        print("\n❌ Please install missing requirements first.")
        return
    
    # Check CUDA
    if torch.cuda.is_available():
        print(f"🚀 CUDA available: {torch.cuda.get_device_name()}")
    else:
        print("💻 Running on CPU (CUDA not available)")
    
    print("\nStarting demo in 3 seconds...")
    import time
    time.sleep(3)
    
    # Run the demo
    try:
        quick_demo()
    except KeyboardInterrupt:
        print("\n\n👋 Demo interrupted. Goodbye!")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        print("\nThis is likely due to missing dependencies or configuration issues.")
        print("Please check the README.md for setup instructions.")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()