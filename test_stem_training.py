#!/usr/bin/env python3
"""
Test the STEM training generator
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jigyasa.cognitive.stem_training import STEMTrainingGenerator, ConversationalTrainer


def test_stem_generator():
    """Test STEM problem generation"""
    print("🧮 Testing STEM Training Generator")
    print("=" * 50)
    
    generator = STEMTrainingGenerator()
    
    # Test math problems
    print("\n📐 Mathematical Problems:")
    print("-" * 30)
    for difficulty in ['basic', 'intermediate', 'advanced']:
        problem = generator.generate_math_problem(difficulty)
        print(f"\n{difficulty.upper()} Math:")
        print(f"Q: {problem.question}")
        print(f"A: {problem.answer}")
        if problem.reasoning_steps:
            print("Reasoning:")
            for step in problem.reasoning_steps:
                print(f"  - {step}")
    
    # Test coding problems
    print("\n\n💻 Coding Problems:")
    print("-" * 30)
    for difficulty in ['basic', 'intermediate', 'advanced']:
        problem = generator.generate_coding_problem(difficulty)
        print(f"\n{difficulty.upper()} Coding:")
        print(f"Q: {problem.question}")
        print(f"A:\n{problem.answer}")
        if problem.reasoning_steps:
            print("Approach:")
            for step in problem.reasoning_steps:
                print(f"  - {step}")
    
    # Test science problems
    print("\n\n🔬 Science Problems:")
    print("-" * 30)
    for difficulty in ['basic', 'intermediate']:
        problem = generator.generate_science_problem(difficulty)
        print(f"\n{difficulty.upper()} Science:")
        print(f"Q: {problem.question}")
        print(f"A: {problem.answer}")
        if problem.reasoning_steps:
            print("Explanation:")
            for step in problem.reasoning_steps:
                print(f"  - {step}")
    
    # Test batch generation
    print("\n\n📦 Batch Generation:")
    print("-" * 30)
    batch = generator.generate_training_batch(batch_size=10)
    
    categories = {}
    for example in batch:
        cat = example.category
        categories[cat] = categories.get(cat, 0) + 1
    
    print(f"Generated {len(batch)} examples:")
    for cat, count in categories.items():
        print(f"  - {cat}: {count} examples")


def test_conversational_trainer():
    """Test conversational training"""
    print("\n\n💬 Testing Conversational Trainer")
    print("=" * 50)
    
    trainer = ConversationalTrainer()
    examples = trainer.generate_conversational_examples(count=10)
    
    print(f"Generated {len(examples)} conversational examples:\n")
    
    for i, example in enumerate(examples[:5], 1):
        print(f"{i}. User: {example['input']}")
        print(f"   Bot: {example['response']}")
        print()


if __name__ == "__main__":
    test_stem_generator()
    test_conversational_trainer()
    
    print("\n✅ Testing complete!")