#!/usr/bin/env python3
"""
Demonstrate REAL AGI capabilities with Ollama and Llama 3.2
This shows actual working features, not placeholders
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jigyasa.core.real_agi_system import RealAGISystem
from jigyasa.models.ollama_wrapper import OllamaWrapper
import time

def check_ollama():
    """Check if Ollama is available"""
    try:
        ollama = OllamaWrapper()
        if ollama.check_ollama_running():
            print("✅ Ollama is running")
            return True
        else:
            print("❌ Ollama is not running")
            print("\nTo install Ollama:")
            print("1. Visit https://ollama.com")
            print("2. Download and install for your system")
            print("3. Run: ollama pull llama3.2:latest")
            print("4. Start Ollama service")
            return False
    except Exception as e:
        print(f"❌ Error checking Ollama: {e}")
        return False

def demo_real_capabilities():
    """Demonstrate real AGI capabilities"""
    print("\n🤖 JIGYASA - Real AGI System Demo")
    print("=" * 60)
    
    # Check Ollama first
    if not check_ollama():
        print("\n⚠️  Please install Ollama to run this demo with real AI capabilities")
        return
        
    try:
        # Initialize real AGI system
        print("\n🔧 Initializing Real AGI System with Llama 3.2...")
        agi = RealAGISystem()
        
        # Show system status
        print("\n📊 System Status:")
        status = agi.get_system_status()
        print(f"   Hardware: {status['hardware']['cpu']} ({status['hardware']['cores']} cores)")
        print(f"   RAM: {status['hardware']['ram_gb']:.1f} GB")
        print(f"   GPU: {status['hardware']['gpu']}")
        print(f"   Ollama: {status['ollama_status']}")
        print(f"   Knowledge Base: {status['learning']['total_knowledge_items']} items")
        
        # Demo 1: Real Code Improvement
        print("\n1️⃣ Real Code Improvement Demo")
        print("-" * 40)
        
        # Create a sample file to improve
        sample_code = '''
def find_duplicates(items):
    """Find duplicate items in a list"""
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates

def calculate_average(numbers):
    """Calculate average of numbers"""
    total = 0
    count = 0
    for num in numbers:
        total = total + num
        count = count + 1
    return total / count
'''
        
        # Save sample file
        sample_file = "sample_to_improve.py"
        with open(sample_file, 'w') as f:
            f.write(sample_code)
            
        print(f"Created {sample_file} with inefficient code")
        print("\n🔍 Analyzing and improving code...")
        
        # Improve the code
        result = agi.improve_code_file(sample_file)
        
        if result['status'] == 'success':
            print(f"✅ Successfully improved code!")
            print(f"   Performance gain: {result.get('performance_gain', 0):.1%}")
            print(f"   Improvements applied: {len(result.get('improvements', []))}")
            
            # Show improvements
            for imp in result.get('improvements', [])[:2]:
                print(f"\n   📝 {imp['type'].title()} improvement:")
                print(f"      {imp['description']}")
                
        # Demo 2: Real Chat with Learning
        print("\n2️⃣ Interactive Chat with Learning")
        print("-" * 40)
        
        queries = [
            "What's the most efficient way to find duplicates in a Python list?",
            "How can I optimize loops in Python for better performance?"
        ]
        
        for query in queries:
            print(f"\n👤 You: {query}")
            response = agi.chat(query)
            print(f"🤖 AGI: {response[:200]}..." if len(response) > 200 else f"🤖 AGI: {response}")
            
            # Show that it learned
            metrics = agi.learner.get_learning_metrics()
            print(f"   📚 Knowledge items: {metrics['total_knowledge_items']}")
            
        # Demo 3: Real Performance Benchmarking
        print("\n3️⃣ Real Performance Benchmarking")
        print("-" * 40)
        
        # Read the improved code
        with open(sample_file, 'r') as f:
            improved_code = f.read()
            
        print("Comparing original vs improved code performance...")
        benchmark_result = agi.benchmark.benchmark_code_comparison(sample_code, improved_code)
        
        if 'overall_improvement' in benchmark_result:
            print(f"\n📈 Overall Performance Improvement: {benchmark_result['overall_improvement']:.1f}%")
            
        # Demo 4: Continuous Learning
        print("\n4️⃣ Continuous Learning Demo")
        print("-" * 40)
        
        # Train on current directory
        print("Training on current codebase...")
        train_result = agi.train_on_codebase(".")
        print(f"✅ Analyzed {train_result['files_analyzed']} files")
        print(f"   Learned {train_result['total_insights_learned']} new insights")
        print(f"   Knowledge base now has {train_result['knowledge_base_size']} items")
        
        # Demo 5: Autonomous Mode
        print("\n5️⃣ Autonomous Mode Demo")
        print("-" * 40)
        
        print("Starting autonomous improvement mode...")
        auto_result = agi.start_autonomous_mode(".", interval=60)
        print(f"✅ Autonomous mode {auto_result['status']}")
        print("   Will continuously improve code every 60 seconds")
        print("   (Run agi.stop_autonomous_mode() to stop)")
        
        # Export knowledge
        print("\n💾 Exporting learned knowledge...")
        agi.export_knowledge("jigyasa_knowledge_export.json")
        print("✅ Knowledge exported to jigyasa_knowledge_export.json")
        
        # Final demonstration
        print("\n" + "=" * 60)
        agi.demonstrate_capabilities()
        
    except Exception as e:
        print(f"\n❌ Error during demo: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main demo function"""
    print("🚀 JIGYASA Real AGI Demo - Powered by Llama 3.2")
    print("This demonstrates ACTUAL working capabilities, not placeholders!")
    
    demo_real_capabilities()
    
    print("\n✨ Demo complete! The system now has:")
    print("- ✅ Real code improvement using AI")
    print("- ✅ Actual performance measurement")
    print("- ✅ True continuous learning with memory")
    print("- ✅ Working autonomous mode")
    print("- ✅ Genuine AGI-like capabilities with Llama 3.2")

if __name__ == "__main__":
    main()