#!/bin/bash

# Setup script for Ollama and Llama 3.2

echo "🚀 JIGYASA - Ollama Setup Script"
echo "================================="

# Check if Ollama is installed
if command -v ollama &> /dev/null; then
    echo "✅ Ollama is already installed"
else
    echo "❌ Ollama not found. Installing..."
    
    # Detect OS
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        echo "Detected macOS"
        if command -v brew &> /dev/null; then
            brew install ollama
        else
            echo "Please install Ollama from: https://ollama.com/download"
            exit 1
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        echo "Detected Linux"
        curl -fsSL https://ollama.com/install.sh | sh
    else
        echo "Unsupported OS. Please install manually from: https://ollama.com/download"
        exit 1
    fi
fi

# Start Ollama service
echo "Starting Ollama service..."
ollama serve &
OLLAMA_PID=$!
sleep 5

# Check if service is running
if curl -s http://localhost:11434/api/tags > /dev/null; then
    echo "✅ Ollama service is running"
else
    echo "❌ Failed to start Ollama service"
    echo "Please run 'ollama serve' manually in another terminal"
fi

# Pull Llama 3.1:8b
echo "Pulling Llama 3.1:8b model (this may take a while)..."
ollama pull llama3.1:8b

# Verify installation
echo "Verifying installation..."
if ollama list | grep -q "llama3.1:8b"; then
    echo "✅ Llama 3.1:8b successfully installed!"
else
    echo "❌ Failed to install Llama 3.1:8b"
    exit 1
fi

echo ""
echo "🎉 Setup complete! You can now run:"
echo "   python3 demo_real_agi.py"
echo ""
echo "To stop Ollama service: kill $OLLAMA_PID"