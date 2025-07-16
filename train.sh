#!/bin/bash

# JIGYASA Training Script
# Runs the full training pipeline with dynamic topics

echo "🧠 JIGYASA Full Training Pipeline"
echo "================================="
echo "This will train JIGYASA through all phases:"
echo "1. Foundational ProRL training"
echo "2. STEM, coding, and conversational training"
echo "3. Model compression for deployment"
echo ""
echo "Training includes:"
echo "- Mathematical problems (basic to advanced)"
echo "- Coding challenges (algorithms, data structures)"
echo "- Science problems (physics, chemistry, biology)"
echo "- Natural conversation patterns"
echo ""

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not found."
    echo "Please install Python 3.8 or higher."
    exit 1
fi

# Check if we're in the right directory
if [ ! -f "run_jigyasa.py" ]; then
    echo "❌ Please run this script from the jigyasa directory."
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Check dependencies
echo "📦 Checking dependencies..."
python3 -c "import torch, transformers, einops, peft" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip3 install -r jigyasa/requirements.txt
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export JIGYASA_DEVICE="${JIGYASA_DEVICE:-cpu}"

echo ""
read -p "Start training? (y/n): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "🚀 Starting training pipeline..."
    python3 -m jigyasa.main --mode train
else
    echo "Training cancelled."
fi