#!/bin/bash
# Export JIGYASA model from Ollama
OLLAMA_MODELS_PATH="${HOME}/.ollama/models"
MODEL_MANIFEST="${OLLAMA_MODELS_PATH}/manifests/registry.ollama.ai/library/jigyasa/latest"

if [ -f "$MODEL_MANIFEST" ]; then
    echo "Found JIGYASA model manifest"
    # Copy model files
    cp -r "${OLLAMA_MODELS_PATH}/blobs/" ./ollama_blobs/
    echo "✅ Exported model blobs"
else
    echo "❌ JIGYASA model not found in Ollama"
    exit 1
fi
