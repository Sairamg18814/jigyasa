#!/usr/bin/env python3
"""
Test JIGYASA's self-editing capabilities
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from jigyasa.autonomous.real_self_editor import RealSelfEditor
from jigyasa.models.ollama_wrapper import OllamaWrapper

def test_self_editing():
    """Test if JIGYASA can actually edit its own code"""
    print("🧪 Testing JIGYASA's self-editing capabilities...")
    
    # Initialize the self-editor
    ollama = OllamaWrapper(model_name="llama3.1:8b")
    editor = RealSelfEditor(ollama)
    
    # Create a test file with intentionally inefficient code
    test_file = "test_code_to_improve.py"
    test_code = '''
def find_duplicates(items):
    """Find duplicate items in a list - intentionally inefficient"""
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates

def concatenate_strings(strings):
    """Concatenate strings - intentionally inefficient"""
    result = ""
    for s in strings:
        result = result + str(s)
    return result
'''
    
    with open(test_file, 'w') as f:
        f.write(test_code)
    
    print(f"\n📝 Created test file: {test_file}")
    print("Original code has intentional inefficiencies:")
    print("- Nested loops for duplicate detection (O(n²))")
    print("- String concatenation in loop")
    
    # Test self-modification
    print("\n🤖 Running self-modification...")
    result = editor.modify_code_autonomously(test_file)
    
    print(f"\n📊 Result: {result['status']}")
    
    if result['status'] == 'success':
        print(f"✅ Successfully modified code!")
        print(f"📈 Performance gain: {result['performance_gain']:.1%}")
        print(f"💾 Backup saved at: {result['backup']}")
        print(f"\n🔧 Improvements made:")
        for imp in result['improvements']:
            print(f"  - {imp['description']}")
            print(f"    Type: {imp['type']}")
            if 'performance_gain' in imp:
                print(f"    Gain: {imp['performance_gain']:.1%}")
        
        # Show the improved code
        print("\n📝 Improved code:")
        with open(test_file, 'r') as f:
            improved_code = f.read()
        print(improved_code)
        
        # Test rollback
        print("\n🔄 Testing rollback functionality...")
        if editor.rollback_modification(test_file):
            print("✅ Successfully rolled back to original version")
        
    else:
        print(f"❌ Modification failed: {result.get('message', 'Unknown error')}")
    
    # Cleanup
    if os.path.exists(test_file):
        os.remove(test_file)
    
    # Test on actual JIGYASA code
    print("\n🧬 Testing on JIGYASA's own code...")
    jigyasa_file = "jigyasa/autonomous/real_self_editor.py"
    
    if os.path.exists(jigyasa_file):
        print(f"📂 Analyzing: {jigyasa_file}")
        result = editor.modify_code_autonomously(jigyasa_file)
        
        if result['status'] == 'success':
            print(f"✅ JIGYASA successfully improved its own code!")
            print(f"📈 Performance gain: {result['performance_gain']:.1%}")
            # Rollback for safety
            editor.rollback_modification(jigyasa_file)
            print("🔄 Rolled back changes for safety")
        elif result['status'] == 'no_improvements':
            print("✨ JIGYASA's code is already optimized!")
        else:
            print(f"⚠️  Could not improve: {result.get('message', 'Unknown')}")
    
    print("\n✅ Self-editing test complete!")

if __name__ == "__main__":
    test_self_editing()