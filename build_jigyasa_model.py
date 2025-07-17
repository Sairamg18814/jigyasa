#!/usr/bin/env python3
"""
Build and deploy JIGYASA as an Ollama model
"""

import os
import sys
import json
import subprocess
import shutil
from pathlib import Path
import tempfile
import zipfile

class JigyasaModelBuilder:
    def __init__(self):
        self.model_name = "jigyasa"
        self.model_version = "1.0.0"
        self.base_model = "llama3.1:8b"
        
    def check_ollama(self):
        """Check if Ollama is installed and running"""
        try:
            result = subprocess.run(["ollama", "--version"], capture_output=True, text=True)
            if result.returncode == 0:
                print("✅ Ollama is installed")
                return True
            else:
                print("❌ Ollama is not installed properly")
                return False
        except FileNotFoundError:
            print("❌ Ollama not found. Please install from https://ollama.com")
            return False
    
    def check_base_model(self):
        """Check if base model is available"""
        try:
            result = subprocess.run(["ollama", "list"], capture_output=True, text=True)
            if self.base_model in result.stdout:
                print(f"✅ Base model {self.base_model} is available")
                return True
            else:
                print(f"⚠️  Base model {self.base_model} not found. Pulling...")
                subprocess.run(["ollama", "pull", self.base_model])
                return True
        except Exception as e:
            print(f"❌ Error checking base model: {e}")
            return False
    
    def create_enhanced_modelfile(self):
        """Create an enhanced Modelfile with all JIGYASA capabilities"""
        modelfile_content = '''# JIGYASA - Autonomous General Intelligence Model
# Version: 1.0.0
# Based on Llama 3.1:8b with specialized AGI capabilities

FROM llama3.1:8b

# Optimized parameters for code analysis
PARAMETER temperature 0.7
PARAMETER top_p 0.9
PARAMETER repeat_penalty 1.1
PARAMETER seed 42
PARAMETER num_ctx 8192
PARAMETER mirostat 2
PARAMETER mirostat_eta 0.1
PARAMETER mirostat_tau 5.0

# System prompt with full JIGYASA capabilities
SYSTEM """You are JIGYASA, an Autonomous General Intelligence system specialized in code analysis, improvement, and continuous learning.

CORE CAPABILITIES:

1. CODE ANALYSIS & IMPROVEMENT
   - Identify performance bottlenecks, bugs, and inefficiencies
   - Suggest algorithmic improvements with complexity analysis
   - Refactor for readability and maintainability
   - Apply design patterns and best practices
   - Measure real performance gains (not estimates)

2. CONTINUOUS LEARNING
   - Learn from each code interaction
   - Recognize and apply patterns across different codebases
   - Build knowledge of common optimizations
   - Adapt suggestions based on context and constraints

3. PERFORMANCE OPTIMIZATION
   - Loop optimization (vectorization, early termination)
   - Algorithm complexity reduction (O(n²) → O(n log n))
   - Memory optimization (generators, caching)
   - String operation improvements
   - Parallel processing opportunities

4. AUTONOMOUS OPERATION
   - Analyze entire codebases systematically
   - Prioritize improvements by impact
   - Generate comprehensive test suites
   - Create detailed documentation
   - Ensure backward compatibility

ANALYSIS METHODOLOGY:
1. Parse and understand code structure using AST concepts
2. Identify specific improvement opportunities with metrics
3. Generate optimized versions with explanations
4. Provide before/after performance comparisons
5. Create tests to validate improvements
6. Document changes and rationale

EXAMPLE IMPROVEMENTS:

# Loop Optimization
Before: for i in range(len(items)): process(items[i])
After: for item in items: process(item)
Gain: 15-20% faster, more Pythonic

# Algorithm Improvement  
Before: nested loops for duplicate detection O(n²)
After: set-based approach O(n)
Gain: 100x faster for large datasets

# Memory Optimization
Before: data = [transform(x) for x in huge_list]
After: data = (transform(x) for x in huge_list)
Gain: Constant memory vs linear memory

RESPONSE FORMAT:
1. Analysis: Understanding of current code
2. Issues: Specific problems identified
3. Solutions: Concrete improvements with code
4. Metrics: Expected performance gains
5. Tests: Validation approach
6. Learning: Patterns to remember

Always be honest about limitations and provide real, measurable improvements."""

# Enhanced template for interactions
TEMPLATE """{{ if .System }}System: {{ .System }}
{{ end }}{{ if .Prompt }}Human: {{ .Prompt }}

JIGYASA: {{ end }}{{ .Response }}"""

# Additional metadata
LICENSE """MIT License - JIGYASA AGI System
Created by the JIGYASA Contributors
Powered by Llama 3.1 and continuous learning algorithms"""

PARAMETER stop "<|im_end|>"
PARAMETER stop "<|im_start|>"
PARAMETER stop "Human:"
PARAMETER stop "JIGYASA:"'''
        
        with open("Modelfile", "w") as f:
            f.write(modelfile_content)
        print("✅ Enhanced Modelfile created")
    
    def build_ollama_model(self):
        """Build the Ollama model"""
        print(f"\n🔨 Building JIGYASA model...")
        try:
            # Remove old model if exists
            subprocess.run(["ollama", "rm", self.model_name], capture_output=True)
            
            # Create new model
            result = subprocess.run(
                ["ollama", "create", self.model_name, "-f", "Modelfile"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ Model '{self.model_name}' created successfully!")
                return True
            else:
                print(f"❌ Error creating model: {result.stderr}")
                return False
        except Exception as e:
            print(f"❌ Error building model: {e}")
            return False
    
    def test_model(self):
        """Test the created model"""
        print("\n🧪 Testing JIGYASA model...")
        test_prompts = [
            "Optimize this Python function: def factorial(n): return 1 if n <= 1 else n * factorial(n-1)",
            "Analyze this code for performance: for i in range(len(items)): if items[i] == target: return i",
            "What patterns have you learned about Python optimization?"
        ]
        
        for i, prompt in enumerate(test_prompts, 1):
            print(f"\nTest {i}: {prompt[:50]}...")
            try:
                result = subprocess.run(
                    ["ollama", "run", self.model_name, prompt],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                if result.returncode == 0:
                    print("✅ Response received")
                    print(f"Preview: {result.stdout[:200]}...")
                else:
                    print(f"❌ Test failed: {result.stderr}")
            except subprocess.TimeoutExpired:
                print("⚠️  Test timed out")
            except Exception as e:
                print(f"❌ Test error: {e}")
    
    def create_model_documentation(self):
        """Create comprehensive model documentation"""
        docs = {
            "name": "jigyasa",
            "version": "1.0.0",
            "base_model": "llama3.1:8b",
            "description": "Autonomous General Intelligence system for code analysis and improvement",
            "capabilities": [
                "Code analysis and optimization",
                "Performance measurement and improvement",
                "Continuous learning from interactions",
                "Autonomous code improvement",
                "Pattern recognition and application"
            ],
            "usage_examples": [
                {
                    "description": "Optimize a function",
                    "command": "ollama run jigyasa \"Optimize this function: def sum_list(items): total = 0; for item in items: total += item; return total\""
                },
                {
                    "description": "Analyze code performance",
                    "command": "ollama run jigyasa \"Analyze the performance of this nested loop and suggest improvements\""
                },
                {
                    "description": "Learn from patterns",
                    "command": "ollama run jigyasa \"What optimization patterns work best for data processing in Python?\""
                }
            ],
            "parameters": {
                "temperature": 0.7,
                "top_p": 0.9,
                "context_length": 8192,
                "repeat_penalty": 1.1
            }
        }
        
        with open("jigyasa_model_docs.json", "w") as f:
            json.dump(docs, f, indent=2)
        print("✅ Model documentation created")
    
    def build(self):
        """Main build process"""
        print("🚀 JIGYASA Model Builder")
        print("=" * 50)
        
        # Check prerequisites
        if not self.check_ollama():
            return False
        
        if not self.check_base_model():
            return False
        
        # Build model
        self.create_enhanced_modelfile()
        
        if not self.build_ollama_model():
            return False
        
        # Test and document
        self.test_model()
        self.create_model_documentation()
        
        print("\n✅ JIGYASA model built successfully!")
        print(f"\n📝 Usage:")
        print(f"   ollama run {self.model_name} \"Your code or question here\"")
        print(f"\n🔧 Advanced usage:")
        print(f"   ollama run {self.model_name} < your_script.py")
        print(f"\n📚 For more examples, see jigyasa_model_docs.json")
        
        return True

if __name__ == "__main__":
    builder = JigyasaModelBuilder()
    success = builder.build()
    sys.exit(0 if success else 1)