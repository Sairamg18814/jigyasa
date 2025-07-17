#!/usr/bin/env python3
"""
Example usage of JIGYASA AGI model from Hugging Face
"""

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
model_name = "Sairamg18814/jigyasa-agi"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="auto"
)

def optimize_code(code: str) -> str:
    """Use JIGYASA to optimize Python code"""
    prompt = f"""<|im_start|>system
You are JIGYASA, an AGI specialized in code optimization.
<|im_end|>
<|im_start|>user
Optimize this Python code:

{code}
<|im_end|>
<|im_start|>assistant
"""
    
    inputs = tokenizer(prompt, return_tensors="pt")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=1000,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("<|im_start|>assistant")[-1].strip()

# Example usage
if __name__ == "__main__":
    code = """
def find_duplicates(items):
    duplicates = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if items[i] == items[j] and items[i] not in duplicates:
                duplicates.append(items[i])
    return duplicates
"""
    
    print("Original code:")
    print(code)
    print("
JIGYASA optimization:")
    print(optimize_code(code))
