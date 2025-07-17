#!/bin/bash
# JIGYASA CLI Usage Examples

echo "🧠 JIGYASA AGI - Code Optimization Examples"
echo "=========================================="

# Example 1: Optimize a function
curl -X POST "https://api-inference.huggingface.co/models/Sairamg18814/jigyasa-agi"   -H "Authorization: Bearer YOUR_HF_TOKEN"   -H "Content-Type: application/json"   -d '{"inputs": "Optimize this: def sum_list(items): total = 0; for item in items: total += item; return total"}'

# Example 2: Analyze performance
curl -X POST "https://api-inference.huggingface.co/models/Sairamg18814/jigyasa-agi"   -H "Authorization: Bearer YOUR_HF_TOKEN"   -H "Content-Type: application/json"   -d '{"inputs": "Analyze performance bottlenecks in nested loops"}'

# Example 3: Learn patterns
curl -X POST "https://api-inference.huggingface.co/models/Sairamg18814/jigyasa-agi"   -H "Authorization: Bearer YOUR_HF_TOKEN"   -H "Content-Type: application/json"   -d '{"inputs": "What optimization patterns work best for data processing?"}'
