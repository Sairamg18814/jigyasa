#!/usr/bin/env python3
"""
Complete setup script for JIGYASA as an LLM model
"""

import os
import sys
import subprocess
from pathlib import Path

def setup_jigyasa_llm():
    """Complete setup process for JIGYASA LLM"""
    print("🧠 JIGYASA LLM Setup")
    print("=" * 50)
    print("\nThis will set up JIGYASA as:")
    print("1. An Ollama model for local use")
    print("2. A Hugging Face model for cloud deployment")
    print()
    
    steps = [
        {
            "name": "Build Ollama Model",
            "script": "build_jigyasa_model.py",
            "description": "Creates JIGYASA as an Ollama model"
        },
        {
            "name": "Package for Hugging Face",
            "script": "package_for_huggingface.py",
            "description": "Prepares model for HF deployment"
        },
        {
            "name": "Deploy Models",
            "script": "deploy_jigyasa.py",
            "description": "Deploys to both platforms"
        }
    ]
    
    for i, step in enumerate(steps, 1):
        print(f"\n{'='*50}")
        print(f"Step {i}/{len(steps)}: {step['name']}")
        print(f"Description: {step['description']}")
        print(f"{'='*50}")
        
        if Path(step['script']).exists():
            result = subprocess.run([sys.executable, step['script']])
            if result.returncode != 0:
                print(f"❌ Step failed: {step['name']}")
                return False
        else:
            print(f"❌ Script not found: {step['script']}")
            return False
    
    print("\n" + "="*50)
    print("✅ JIGYASA LLM Setup Complete!")
    print("="*50)
    
    print("\n📋 Quick Start Guide:")
    print("\n1. Local Usage (Ollama):")
    print("   ollama run jigyasa \"Optimize this code: [your code here]\"")
    
    print("\n2. API Usage (Hugging Face):")
    print("   Set token: export HUGGINGFACE_TOKEN=your_token")
    print("   Upload: python upload_to_huggingface.py")
    
    print("\n3. Python Usage:")
    print("   from transformers import pipeline")
    print("   jigyasa = pipeline('text-generation', model='Sairamg18814/jigyasa-agi')")
    print("   result = jigyasa('Optimize this loop: for i in range(len(items)): print(items[i])')")
    
    print("\n4. Test the Model:")
    print("   python3 -c \"import subprocess; subprocess.run(['ollama', 'run', 'jigyasa', 'What is JIGYASA?'])\"")
    
    print("\n📚 Documentation:")
    print("   - Model capabilities: jigyasa_model_docs.json")
    print("   - Deployment info: deployment_summary.json")
    print("   - GitHub: https://github.com/Sairamg18814/jigyasa")
    
    return True

if __name__ == "__main__":
    success = setup_jigyasa_llm()
    sys.exit(0 if success else 1)