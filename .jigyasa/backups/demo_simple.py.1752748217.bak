#!/usr/bin/env python3
"""
Simple Jigyasa Demo
Demonstrates core functionality without full system initialization
"""

import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jigyasa.core.transformer import JigyasaTransformer, TransformerConfig
from jigyasa.core.blt import BLTTokenizer
from jigyasa.reasoning.neuro_symbolic import NeuroSymbolicReasoner, MathematicalReasoner

def main():
    print("\n" + "="*60)
    print("🧠 JIGYASA AGI - SIMPLE DEMO")
    print("="*60)
    
    # 1. Core Transformer Demo
    print("\n1️⃣ Core Transformer (B.L.T.)")
    print("-" * 30)
    
    # Create small model
    config = TransformerConfig(
        d_model=128,
        n_heads=4,
        n_layers=2,
        vocab_size=256,
        max_seq_length=512
    )
    model = JigyasaTransformer(config)
    print(f"✓ Created model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M parameters")
    
    # Test tokenizer
    tokenizer = BLTTokenizer()
    text = "Hello, I am Jigyasa AGI!"
    tokens = tokenizer.encode(text)
    decoded = tokenizer.decode(tokens)
    print(f"✓ Tokenizer: '{text}' -> {len(tokens)} bytes")
    print(f"  Decoded: '{decoded}'")
    
    # Test forward pass
    input_ids = torch.randint(0, 256, (1, 20))
    with torch.no_grad():
        output = model(input_ids)
    print(f"✓ Forward pass: input {input_ids.shape} -> output {output['logits'].shape}")
    
    # 2. Mathematical Reasoning Demo
    print("\n2️⃣ Mathematical Reasoning")
    print("-" * 30)
    
    math_engine = MathematicalReasoner()
    
    # Create symbolic query for equation solving
    from jigyasa.reasoning.neuro_symbolic import SymbolicQuery
    
    equation = "x + 5 = 10"
    query = SymbolicQuery(
        query_type='mathematical',
        query_text=f'solve {equation}',
        variables={'x': None},
        constraints=[],
        expected_output_type='number'
    )
    result = math_engine.reason(query)
    print(f"✓ Equation: {equation}")
    if result.success:
        print(f"  Solution: x = {result.result}")
        print(f"  Steps: {result.steps[:2]}...")  # Show first 2 steps
    
    # Simplify expression
    expr = "2*x + 3*x - x"
    query = SymbolicQuery(
        query_type='mathematical',
        query_text=f'simplify {expr}',
        variables={'x': 1},
        constraints=[],
        expected_output_type='expression'
    )
    result = math_engine.reason(query)
    print(f"\n✓ Simplify: {expr}")
    if result.success:
        print(f"  Result: {result.result}")
    
    # 3. Reasoning Integration Demo
    print("\n3️⃣ Neuro-Symbolic Reasoning")
    print("-" * 30)
    
    # Create simple neural model
    neural_model = torch.nn.Sequential(
        torch.nn.Linear(128, 256),
        torch.nn.ReLU(),
        torch.nn.Linear(256, 128)
    )
    
    reasoner = NeuroSymbolicReasoner(neural_model)
    
    # Test reasoning
    queries = [
        "Solve: 2*x + 3 = 11",
        "What is the derivative of x^2?",
        "If all cats are animals and Tom is a cat, what is Tom?"
    ]
    
    for query in queries:
        print(f"\n✓ Query: {query}")
        result = reasoner.reason(query)
        print(f"  Type: {result['reasoning_type']}")
        if result['symbolic_result']:
            print(f"  Result: {result['symbolic_result'].result}")
            print(f"  Confidence: {result['confidence']:.2f}")
    
    # 4. Byte-Level Processing Demo
    print("\n4️⃣ Byte-Level Processing")
    print("-" * 30)
    
    # Show byte encoding
    texts = ["Hello", "你好", "🚀", "αβγ"]
    for text in texts:
        bytes_list = tokenizer.encode(text)
        print(f"✓ '{text}' -> {bytes_list} ({len(bytes_list)} bytes)")
    
    print("\n" + "="*60)
    print("✅ Demo Complete!")
    print("\nThis demonstrates:")
    print("- Byte-level tokenization (no vocabulary needed)")
    print("- Mathematical reasoning capabilities")
    print("- Neuro-symbolic integration")
    print("- Multi-lingual support through bytes")
    print("="*60)


if __name__ == "__main__":
    main()