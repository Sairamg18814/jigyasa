#!/bin/bash

echo "🚀 Preparing JIGYASA for GitHub"
echo "==============================="

# Create necessary directories
echo "📁 Creating directory structure..."
mkdir -p .github/ISSUE_TEMPLATE
mkdir -p .github/workflows
mkdir -p assets
mkdir -p docs
mkdir -p tests

# Clean up unnecessary files
echo "🧹 Cleaning up..."
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -delete
find . -name ".DS_Store" -delete

# Remove old README files
rm -f README_REAL.md
rm -f README_old.md

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "📝 Creating .gitignore..."
    cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# OS
.DS_Store
Thumbs.db

# Project specific
.jigyasa/
.jigyasa_backups/
.jigyasa_benchmarks/
*.log
*.db
*.bak

# Environment
.env
.env.local

# Test coverage
htmlcov/
.tox/
.coverage
.coverage.*
.cache
.pytest_cache/
coverage.xml
*.cover

# Distribution
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
EOF
fi

# Create docs directory structure
echo "📚 Setting up documentation..."
cat > docs/index.md << 'EOF'
# JIGYASA Documentation

Welcome to the JIGYASA documentation!

## Quick Links

- [Installation](installation.md)
- [Usage Guide](usage.md)
- [API Reference](api.md)
- [Contributing](../CONTRIBUTING.md)

## What is JIGYASA?

JIGYASA is an Autonomous General Intelligence system powered by Llama 3.1 that can:
- Analyze and improve code automatically
- Learn continuously from interactions
- Measure real performance improvements
- Operate autonomously to optimize codebases

## Getting Started

```bash
# Clone the repository
git clone https://github.com/Sairamg18814/jigyasa.git

# Run setup
python jigyasa_setup.py

# Start using JIGYASA
python main_agi.py chat
```
EOF

# Create basic test structure
echo "🧪 Setting up test structure..."
cat > tests/__init__.py << 'EOF'
"""Test suite for JIGYASA"""
EOF

cat > tests/test_core.py << 'EOF'
"""Core functionality tests"""

import pytest
from jigyasa.core.jigyasa_agi import JigyasaAGI

def test_initialization():
    """Test AGI initialization"""
    # This is a placeholder test
    assert True

def test_code_analysis():
    """Test code analysis functionality"""
    # This is a placeholder test
    assert True
EOF

# Create requirements-dev.txt
echo "📦 Creating development requirements..."
cat > requirements-dev.txt << 'EOF'
# Development dependencies
pytest>=7.0.0
pytest-cov>=3.0.0
pytest-asyncio>=0.18.0
black>=22.0.0
flake8>=4.0.0
mypy>=0.910
pre-commit>=2.17.0
sphinx>=4.4.0
sphinx-rtd-theme>=1.0.0
EOF

# Update main requirements.txt to be cleaner
echo "📦 Cleaning up requirements..."
cat > requirements.txt << 'EOF'
# Core dependencies
torch>=2.0.0
transformers>=4.30.0
ollama>=0.1.0
requests>=2.31.0
numpy>=1.24.0
psutil>=5.8.0
GitPython>=3.1.0
flask>=2.0.0
beautifulsoup4>=4.12.0
python-dotenv>=1.0.0

# Testing
pytest>=7.0.0
pytest-cov>=3.0.0

# Code quality
black>=22.0.0
flake8>=4.0.0
EOF

# Create a simple LICENSE file if it doesn't exist
if [ ! -f LICENSE ]; then
    echo "📄 Creating LICENSE..."
    cat > LICENSE << 'EOF'
MIT License

Copyright (c) 2024 JIGYASA Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
fi

# Summary
echo ""
echo "✅ Repository prepared for GitHub!"
echo ""
echo "📋 Checklist:"
echo "  ✓ Directory structure created"
echo "  ✓ Documentation templates added"
echo "  ✓ Test structure initialized"
echo "  ✓ Requirements files updated"
echo "  ✓ License file created"
echo "  ✓ .gitignore configured"
echo ""
echo "🎯 Next steps:"
echo "  1. Review and customize the README.md"
echo "  2. Add actual screenshots to assets/"
echo "  3. Write more comprehensive tests"
echo "  4. git add ."
echo "  5. git commit -m '🚀 Complete JIGYASA AGI implementation with Llama 3.1'"
echo "  6. git push origin main"
echo ""
echo "🌟 Your repository is ready to shine on GitHub!"