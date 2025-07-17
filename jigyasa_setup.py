#!/usr/bin/env python3
"""
JIGYASA AGI Setup Script
Complete setup and configuration for the autonomous AI system
"""

import os
import sys
import subprocess
import json
from pathlib import Path

def check_python_version():
    """Ensure Python 3.8+"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        sys.exit(1)
    print(f"✅ Python {sys.version.split()[0]} detected")

def check_ollama():
    """Check if Ollama is installed and running"""
    try:
        # Check if ollama command exists
        result = subprocess.run(['which', 'ollama'], capture_output=True)
        if result.returncode != 0:
            print("❌ Ollama not found")
            print("\nPlease install Ollama:")
            print("1. Visit https://ollama.com")
            print("2. Download and install for your system")
            return False
            
        # Check if service is running
        import requests
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=2)
            if response.status_code == 200:
                print("✅ Ollama service is running")
                return True
        except:
            pass
            
        print("⚠️  Ollama installed but service not running")
        print("Run 'ollama serve' in another terminal")
        return False
        
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def pull_llama_model():
    """Pull Llama 3.1:8b model"""
    print("\n📥 Pulling Llama 3.1:8b model (this may take a while)...")
    
    try:
        # Check if model already exists
        result = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
        if 'llama3.1:8b' in result.stdout:
            print("✅ Llama 3.1:8b already available")
            return True
            
        # Pull the model
        result = subprocess.run(['ollama', 'pull', 'llama3.1:8b'])
        if result.returncode == 0:
            print("✅ Successfully pulled Llama 3.1:8b")
            return True
        else:
            print("❌ Failed to pull Llama 3.1:8b")
            return False
            
    except Exception as e:
        print(f"❌ Error pulling model: {e}")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("\n📦 Installing Python dependencies...")
    
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "ollama",  # If available as package
        "psutil>=5.8.0",
        "GitPython>=3.1.0",
        "pytest>=7.0.0",
        "flask>=2.0.0",
        "requests>=2.31.0",
        "numpy>=1.24.0",
        "beautifulsoup4>=4.12.0"
    ]
    
    for req in requirements:
        print(f"Installing {req}...")
        subprocess.run([sys.executable, '-m', 'pip', 'install', req], 
                      stdout=subprocess.DEVNULL)
    
    print("✅ Dependencies installed")

def create_jigyasa_model():
    """Create custom Jigyasa model in Ollama"""
    print("\n🏗️  Creating Jigyasa model in Ollama...")
    
    modelfile = Path("Modelfile.jigyasa")
    if not modelfile.exists():
        print("❌ Modelfile.jigyasa not found")
        return False
        
    try:
        result = subprocess.run(['ollama', 'create', 'jigyasa', '-f', 'Modelfile.jigyasa'])
        if result.returncode == 0:
            print("✅ Jigyasa model created successfully")
            print("\nYou can now use: ollama run jigyasa")
            return True
        else:
            print("❌ Failed to create Jigyasa model")
            return False
            
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        return False

def setup_directories():
    """Create necessary directories"""
    print("\n📁 Setting up directories...")
    
    dirs = [
        ".jigyasa",
        ".jigyasa/backups",
        ".jigyasa/benchmarks",
        ".jigyasa/logs"
    ]
    
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
        
    print("✅ Directories created")

def test_installation():
    """Test that everything works"""
    print("\n🧪 Testing installation...")
    
    try:
        # Test import
        from jigyasa.core.jigyasa_agi import JigyasaAGI
        print("✅ Imports working")
        
        # Test Ollama connection
        from jigyasa.models.ollama_wrapper import OllamaWrapper
        ollama = OllamaWrapper()
        if ollama.check_ollama_running():
            print("✅ Ollama connection working")
            
            # Test generation
            response = ollama.generate("Hello, this is a test", temperature=0.1)
            if response.text:
                print("✅ Model generation working")
                return True
        else:
            print("⚠️  Ollama not accessible")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main setup process"""
    print("🧠 JIGYASA AGI Setup")
    print("=" * 50)
    
    # 1. Check Python version
    check_python_version()
    
    # 2. Install dependencies
    install_dependencies()
    
    # 3. Check Ollama
    if not check_ollama():
        print("\n⚠️  Please install and start Ollama before continuing")
        return
        
    # 4. Pull Llama model
    if not pull_llama_model():
        print("\n⚠️  Failed to pull Llama 3.1:8b model")
        return
        
    # 5. Setup directories
    setup_directories()
    
    # 6. Create Jigyasa model
    create_jigyasa_model()
    
    # 7. Test installation
    if test_installation():
        print("\n✅ JIGYASA AGI setup complete!")
        print("\nYou can now:")
        print("1. Run the demo: python demo_jigyasa_agi.py")
        print("2. Use the CLI: python main_agi.py --help")
        print("3. Chat with Jigyasa: python main_agi.py chat")
        print("4. Improve code: python main_agi.py improve --path your_code.py")
        print("5. Use as Ollama model: ollama run jigyasa")
    else:
        print("\n⚠️  Setup completed with warnings")
        print("Please check the errors above")
        
    print("\n📚 See README_REAL.md for full documentation")

if __name__ == "__main__":
    main()