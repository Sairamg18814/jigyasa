#!/usr/bin/env python3
"""
Simple Interactive Demo for Jigyasa
Shows the core capabilities without full system initialization
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from jigyasa.core.transformer import JigyasaTransformer, TransformerConfig
from jigyasa.core.blt import BLTTokenizer
from jigyasa.reasoning.neuro_symbolic import MathematicalReasoner, SymbolicQuery
from jigyasa.reasoning.causal import CausalReasoner

class SimpleJigyasa:
    def __init__(self):
        print("🧠 Initializing Jigyasa AGI...")
        
        # Core components
        config = TransformerConfig(
            d_model=256,
            n_heads=8,
            n_layers=4,
            vocab_size=256,
            max_seq_length=1024
        )
        self.model = JigyasaTransformer(config)
        self.tokenizer = BLTTokenizer()
        self.math_engine = MathematicalReasoner()
        self.causal_engine = CausalReasoner()
        
        print(f"✓ Model initialized: {sum(p.numel() for p in self.model.parameters())/1e6:.1f}M parameters")
        print("✓ Mathematical reasoning engine ready")
        print("✓ Causal reasoning engine ready")
    
    def process_query(self, query: str) -> str:
        """Process user query and return response"""
        query_lower = query.lower()
        
        # Mathematical queries
        if any(word in query_lower for word in ['solve', 'calculate', 'derivative', 'integral', 'simplify']):
            return self.handle_math(query)
        
        # Causal queries
        elif any(word in query_lower for word in ['cause', 'effect', 'why', 'because']):
            return self.handle_causal(query)
        
        # Coding queries
        elif any(word in query_lower for word in ['code', 'implement', 'algorithm', 'function']):
            return self.handle_coding(query)
        
        # General response
        else:
            return self.handle_general(query)
    
    def handle_math(self, query: str) -> str:
        """Handle mathematical queries"""
        response = "🧮 Mathematical Reasoning:\n"
        
        # Create symbolic query
        sym_query = SymbolicQuery(
            query_type='mathematical',
            query_text=query,
            variables={'x': None, 'y': None},
            constraints=[],
            expected_output_type='solution'
        )
        
        try:
            result = self.math_engine.reason(sym_query)
            if result.success:
                response += f"Result: {result.result}\n"
                if result.steps:
                    response += f"Steps: {result.steps[:3]}..."
            else:
                response += f"Could not solve: {result.explanation}"
        except Exception as e:
            response += f"Error: {str(e)}"
        
        return response
    
    def handle_causal(self, query: str) -> str:
        """Handle causal reasoning queries"""
        response = "🔗 Causal Reasoning:\n"
        
        # Example causal relationships
        self.causal_engine.add_cause_effect("studying", "good grades", 0.8)
        self.causal_engine.add_cause_effect("good grades", "better opportunities", 0.7)
        self.causal_engine.add_cause_effect("rain", "wet ground", 0.95)
        self.causal_engine.add_cause_effect("wet ground", "slippery", 0.8)
        
        if "studying" in query.lower():
            effects = self.causal_engine.predict_effects("studying")
            response += "If you study:\n"
            for effect, prob in effects:
                response += f"  → {effect} (probability: {prob:.2f})\n"
        
        elif "rain" in query.lower():
            effects = self.causal_engine.predict_effects("rain")
            response += "If it rains:\n"
            for effect, prob in effects:
                response += f"  → {effect} (probability: {prob:.2f})\n"
        
        else:
            response += "I can analyze causal relationships. Try asking about studying or rain."
        
        return response
    
    def handle_coding(self, query: str) -> str:
        """Handle coding queries"""
        response = "💻 Code Generation:\n"
        
        if "fibonacci" in query.lower():
            response += """
```python
def fibonacci(n):
    \"\"\"Generate nth Fibonacci number - O(n) time, O(1) space\"\"\"
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr
```
Explanation: Uses dynamic programming with space optimization."""
        
        elif "sort" in query.lower():
            response += """
```python
def quicksort(arr):
    \"\"\"Quicksort implementation - O(n log n) average case\"\"\"
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quicksort(left) + middle + quicksort(right)
```
Explanation: Divide-and-conquer algorithm with pivot selection."""
        
        else:
            response += "I can generate code for various algorithms. Try asking about Fibonacci or sorting!"
        
        return response
    
    def handle_general(self, query: str) -> str:
        """Handle general queries"""
        # Tokenize to show byte-level processing
        tokens = self.tokenizer.encode(query)
        
        response = f"🤔 Processing your query...\n"
        response += f"Byte encoding: {len(tokens)} bytes\n"
        response += f"Universal support: ✓ (works with any language/symbol)\n\n"
        
        response += "I'm Jigyasa, an AGI with capabilities in:\n"
        response += "• Mathematical reasoning (try: 'solve x^2 - 5x + 6 = 0')\n"
        response += "• Causal analysis (try: 'what happens if it rains?')\n"
        response += "• Code generation (try: 'implement fibonacci')\n"
        response += "• STEM problem solving\n"
        response += "• Self-improvement through SEAL and ProRL"
        
        return response
    
    def run_interactive(self):
        """Run interactive session"""
        print("\n" + "="*60)
        print("🧠 JIGYASA AGI - INTERACTIVE MODE")
        print("="*60)
        print("\nCapabilities:")
        print("• 🧮 Mathematical reasoning")
        print("• 🔗 Causal analysis")
        print("• 💻 Code generation")
        print("• 🔬 STEM problem solving")
        print("• 🌐 Universal language support (via bytes)")
        print("\nType 'help' for examples, 'quit' to exit")
        print("="*60)
        
        while True:
            try:
                user_input = input("\n🧑 You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("\n👋 Goodbye! Thanks for chatting with Jigyasa.")
                    break
                
                elif user_input.lower() == 'help':
                    print("\n📚 Example queries:")
                    print("• solve x + 5 = 10")
                    print("• calculate the derivative of x^2")
                    print("• what happens if it rains?")
                    print("• implement fibonacci algorithm")
                    print("• simplify 2*x + 3*x - x")
                    continue
                
                elif not user_input:
                    continue
                
                # Process query
                response = self.process_query(user_input)
                print(f"\n🤖 Jigyasa: {response}")
                
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


def main():
    print("\n🚀 Starting Jigyasa AGI Simple Interactive Demo...")
    
    # Initialize
    jigyasa = SimpleJigyasa()
    
    # Run interactive mode
    jigyasa.run_interactive()


if __name__ == "__main__":
    main()