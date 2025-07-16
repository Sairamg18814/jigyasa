#!/bin/bash

# JIGYASA GUI Launcher Script

echo "🧠 JIGYASA - Autonomous AGI Dashboard"
echo "===================================="

# Check if we're in the right directory
if [ ! -f "launch_gui.py" ]; then
    echo "❌ Please run this script from the jigyasa directory."
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "🔧 Activating virtual environment..."
    source venv/bin/activate
fi

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

echo "🚀 Launching GUI..."
python3 launch_gui.py