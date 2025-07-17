#!/usr/bin/env python3
"""
JIGYASA AGI Demo - Showcasing Real Autonomous Capabilities
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from jigyasa.core.jigyasa_agi import JigyasaAGI
import time

def demo_header(title: str):
    """Print demo section header"""
    print(f"\n{'='*60}")
    print(f"🔷 {title}")
    print(f"{'='*60}")

def create_sample_code():
    """Create sample code files for demonstration"""
    # Inefficient code sample 1
    sample1 = '''
def calculate_fibonacci(n):
    """Calculate nth Fibonacci number"""
    if n <= 1:
        return n
    else:
        return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

def find_prime_numbers(limit):
    """Find all prime numbers up to limit"""
    primes = []
    for num in range(2, limit + 1):
        is_prime = True
        for i in range(2, num):
            if num % i == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(num)
    return primes

def process_data(data_list):
    """Process data inefficiently"""
    result = []
    for i in range(len(data_list)):
        item = data_list[i]
        # Inefficient string concatenation
        output = ""
        for j in range(len(item)):
            output = output + item[j].upper()
        result.append(output)
    return result
'''
    
    # Inefficient code sample 2
    sample2 = '''
def find_duplicates(items):
    """Find duplicate items in a list"""
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j]:
                if items[i] not in duplicates:
                    duplicates.append(items[i])
    return duplicates

def merge_lists(list1, list2):
    """Merge two lists inefficiently"""
    merged = []
    for item in list1:
        merged.append(item)
    for item in list2:
        merged.append(item)
    return merged

def count_occurrences(text, word):
    """Count word occurrences inefficiently"""
    count = 0
    words = text.split()
    for i in range(len(words)):
        if words[i] == word:
            count = count + 1
    return count
'''
    
    # Save samples
    with open('sample_inefficient_1.py', 'w') as f:
        f.write(sample1)
    with open('sample_inefficient_2.py', 'w') as f:
        f.write(sample2)
        
    print("✅ Created sample files: sample_inefficient_1.py, sample_inefficient_2.py")

def main():
    """Run the comprehensive demo"""
    print("🧠 JIGYASA AGI - Real Autonomous General Intelligence Demo")
    print("Powered by Llama 3.1:8b")
    print("="*60)
    
    # Check if Ollama is available
    try:
        print("\n🔍 Checking Ollama connection...")
        agi = JigyasaAGI()
        print("✅ Connected to Ollama with Llama 3.1:8b")
    except RuntimeError as e:
        print(f"❌ {e}")
        print("\nTo run this demo:")
        print("1. Install Ollama: https://ollama.com")
        print("2. Pull model: ollama pull llama3.1:8b")
        print("3. Start Ollama and run this demo again")
        return
        
    # Show initial status
    demo_header("Initial System Status")
    metrics = agi.get_system_metrics()
    print(f"Knowledge patterns: {metrics['knowledge']['patterns']}")
    print(f"Learning insights: {metrics['knowledge']['insights']}")
    print(f"Code improvements: {metrics['knowledge']['improvements']}")
    
    # Demo 1: Code Analysis and Improvement
    demo_header("Demo 1: Autonomous Code Improvement")
    create_sample_code()
    
    print("\n🔧 Analyzing and improving sample_inefficient_1.py...")
    improvement1 = agi.analyze_and_improve_code("sample_inefficient_1.py")
    
    if improvement1.improvements:
        print(f"\n✅ Applied {len(improvement1.improvements)} improvements:")
        for imp in improvement1.improvements:
            print(f"  • {imp['type'].upper()}: {imp['description']}")
        print(f"\n📈 Performance gain: {improvement1.performance_gain:.1%}")
    else:
        print("ℹ️  No improvements applied")
        
    # Demo 2: Continuous Learning
    demo_header("Demo 2: Continuous Learning Chat")
    
    test_queries = [
        "What's the most efficient way to find duplicates in a Python list?",
        "How can I optimize recursive Fibonacci calculation?",
        "What are best practices for string manipulation in Python?"
    ]
    
    for query in test_queries:
        print(f"\n👤 Question: {query}")
        response = agi.continuous_learning_chat(query)
        print(f"🤖 JIGYASA: {response[:200]}..." if len(response) > 200 else f"🤖 JIGYASA: {response}")
        time.sleep(1)  # Brief pause for readability
        
    # Demo 3: Show Learning Progress
    demo_header("Demo 3: Learning Progress")
    
    metrics_after = agi.get_system_metrics()
    print(f"Knowledge patterns: {metrics['knowledge']['patterns']} → {metrics_after['knowledge']['patterns']}")
    print(f"Learning insights: {metrics['knowledge']['insights']} → {metrics_after['knowledge']['insights']}")
    print(f"Code improvements: {metrics['knowledge']['improvements']} → {metrics_after['knowledge']['improvements']}")
    
    if metrics_after['knowledge']['avg_performance_gain'] > 0:
        print(f"\n📊 Average performance gain across all improvements: "
              f"{metrics_after['knowledge']['avg_performance_gain']:.1%}")
        
    # Demo 4: Batch Code Improvement
    demo_header("Demo 4: Batch Code Analysis")
    
    print("\n🔧 Analyzing and improving sample_inefficient_2.py...")
    improvement2 = agi.analyze_and_improve_code("sample_inefficient_2.py")
    
    if improvement2.improvements:
        print(f"\n✅ Applied {len(improvement2.improvements)} improvements")
        print(f"📈 Performance gain: {improvement2.performance_gain:.1%}")
        
    # Demo 5: Knowledge Export
    demo_header("Demo 5: Knowledge Export")
    
    export_file = "jigyasa_demo_knowledge.json"
    print(f"\n📚 Exporting learned knowledge to {export_file}...")
    agi.export_knowledge(export_file)
    print("✅ Knowledge exported successfully")
    
    # Demo 6: Create Ollama Model
    demo_header("Demo 6: Create Custom Ollama Model")
    
    print("\n🏗️  Creating specialized Jigyasa model for Ollama...")
    modelfile, instructions = agi.create_specialized_model()
    print(f"✅ Created {modelfile}")
    print("\nTo install as Ollama model:")
    print("  ollama create jigyasa -f Modelfile.jigyasa")
    
    # Demo 7: Autonomous Mode Preview
    demo_header("Demo 7: Autonomous Mode Preview")
    
    print("\n🤖 Autonomous mode can continuously improve your codebase")
    print("Start with: python main_agi.py autonomous --path ./src --interval 300")
    print("This will check and improve code every 5 minutes")
    
    # Final Summary
    demo_header("Demo Summary")
    
    final_metrics = agi.get_system_metrics()
    
    print("\n✅ JIGYASA AGI demonstrated:")
    print("  • Real code analysis and improvement using Llama 3.1:8b")
    print("  • Continuous learning with persistent knowledge base")
    print("  • Actual performance measurement and optimization")
    print("  • Knowledge export and model creation capabilities")
    print(f"\n📊 Final stats:")
    print(f"  • Total improvements made: {final_metrics['knowledge']['improvements']}")
    print(f"  • Knowledge items learned: {final_metrics['knowledge']['insights']}")
    print(f"  • Patterns identified: {final_metrics['knowledge']['patterns']}")
    
    if final_metrics['knowledge']['avg_performance_gain'] > 0:
        print(f"  • Average performance gain: {final_metrics['knowledge']['avg_performance_gain']:.1%}")
        
    print("\n🎉 Demo complete! JIGYASA AGI is ready for autonomous operation.")
    
    # Cleanup
    print("\n🧹 Cleaning up demo files...")
    for f in ['sample_inefficient_1.py', 'sample_inefficient_2.py']:
        if os.path.exists(f):
            os.remove(f)
    print("✅ Cleanup complete")

if __name__ == "__main__":
    main()