#!/usr/bin/env python3
"""
Simple demo of JIGYASA without complex self-correction
For testing basic functionality
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jigyasa.config import JigyasaConfig
from jigyasa.core.model import create_jigyasa_model


def simple_chat():
    """Simple chat interface without self-correction"""
    print("\n🧠 JIGYASA Simple Demo")
    print("=" * 50)
    print("Chat with JIGYASA (without self-correction)")
    print("Type 'quit' to exit")
    print("=" * 50)
    
    # Initialize model
    print("\n🔧 Initializing model...")
    model = create_jigyasa_model(
        d_model=256,
        n_heads=8,
        n_layers=4,
        max_seq_length=512
    )
    print("✅ Model ready!")
    
    # Chat loop
    while True:
        try:
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("Goodbye! 👋")
                break
            
            if not user_input:
                continue
            
            # Generate response
            print("\nJigyasa: ", end='', flush=True)
            
            response = model.generate(
                input_text=user_input,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.2
            )
            
            print(response)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            print(f"\n❌ Error: {str(e)}")


if __name__ == "__main__":
    simple_chat()