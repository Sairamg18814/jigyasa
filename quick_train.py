#!/usr/bin/env python3
"""
Quick training demo for JIGYASA
Trains for just 1 epoch with 20 examples for quick testing
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import logging
from tqdm import tqdm
from jigyasa.config import JigyasaConfig
from jigyasa.core.model import create_jigyasa_model
from jigyasa.cognitive.stem_training import STEMTrainingGenerator, ConversationalTrainer


def quick_train():
    """Quick training demo"""
    
    print("\n🚀 JIGYASA Quick Training Demo")
    print("=" * 50)
    
    # Initialize model
    print("🔧 Initializing model...")
    model = create_jigyasa_model(
        d_model=256,
        n_heads=8, 
        n_layers=4,
        max_seq_length=512
    )
    print(f"✅ Model ready!")
    
    # Initialize generators
    stem_gen = STEMTrainingGenerator()
    conv_gen = ConversationalTrainer()
    
    # Quick training - 1 epoch, 20 examples
    print("\n📚 Training for 1 epoch with 20 examples...")
    
    losses = []
    
    for i in tqdm(range(20), desc="Training"):
        # Generate example
        if i % 4 == 0:  # Conversational
            conv_example = conv_gen.generate_conversational_examples(1)[0]
            input_text = conv_example['input']
            target_text = conv_example['response']
        else:  # STEM
            problem_type = ['math', 'coding', 'science'][i % 3]
            difficulty = ['basic', 'intermediate'][i % 2]
            
            if problem_type == 'math':
                example = stem_gen.generate_math_problem(difficulty)
            elif problem_type == 'coding':
                example = stem_gen.generate_coding_problem(difficulty)
            else:
                example = stem_gen.generate_science_problem(difficulty)
            
            input_text = example.question
            target_text = example.answer
        
        # Generate response
        try:
            generated = model.generate(
                input_text=input_text,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
            
            # Simple loss
            loss = abs(len(generated) - len(target_text)) / 100.0
            losses.append(loss)
            
        except Exception as e:
            print(f"\nError: {e}")
            continue
    
    avg_loss = sum(losses) / len(losses) if losses else 0
    print(f"\n✅ Training complete! Average loss: {avg_loss:.4f}")
    
    # Test the model
    print("\n🧪 Testing model...")
    test_cases = [
        ("What is 15 + 27?", "math"),
        ("Write a Python hello world", "coding"),
        ("Hello, how are you?", "conversation")
    ]
    
    for question, category in test_cases:
        print(f"\n[{category.upper()}] Q: {question}")
        try:
            response = model.generate(
                input_text=question,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True
            )
            print(f"A: {response}")
        except:
            print("A: [Generation failed]")
    
    print("\n✅ Quick training demo complete!")
    
    # Show some training examples
    print("\n📋 Sample Training Data:")
    print("-" * 50)
    
    # Show one of each type
    math_ex = stem_gen.generate_math_problem('basic')
    print(f"\nMATH: {math_ex.question}")
    print(f"Answer: {math_ex.answer}")
    
    code_ex = stem_gen.generate_coding_problem('basic')
    print(f"\nCODING: {code_ex.question}")
    print(f"Answer: {code_ex.answer[:100]}...")
    
    conv_ex = conv_gen.generate_conversational_examples(1)[0]
    print(f"\nCONVERSATION: {conv_ex['input']}")
    print(f"Response: {conv_ex['response']}")


if __name__ == "__main__":
    quick_train()