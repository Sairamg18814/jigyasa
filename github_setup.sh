#!/bin/bash

# JIGYASA GitHub Setup Script
# This script will initialize git, create a GitHub repo, and push the project

echo "🚀 JIGYASA GitHub Setup"
echo "======================="

# Check if gh CLI is installed
if ! command -v gh &> /dev/null; then
    echo "❌ GitHub CLI (gh) is not installed."
    echo "Install it from: https://cli.github.com/"
    exit 1
fi

# Check if user is authenticated
if ! gh auth status &> /dev/null; then
    echo "❌ Not authenticated with GitHub CLI."
    echo "Run: gh auth login"
    exit 1
fi

# Get GitHub username
GITHUB_USER=$(gh api user --jq .login)
echo "✓ Authenticated as: $GITHUB_USER"

# Repository name
REPO_NAME="jigyasa"
echo "📦 Repository name: $REPO_NAME"

# Check if repo already exists
if gh repo view "$GITHUB_USER/$REPO_NAME" &> /dev/null; then
    echo "⚠️  Repository already exists!"
    read -p "Do you want to use the existing repository? (y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    # Create repository
    echo "📝 Creating GitHub repository..."
    gh repo create "$REPO_NAME" \
        --public \
        --description "🧠 JIGYASA - A Self-Improving Artificial General Intelligence System" \
        --homepage "https://$GITHUB_USER.github.io/$REPO_NAME" \
        --license MIT \
        --add-readme=false
fi

# Initialize git if needed
if [ ! -d .git ]; then
    echo "🔧 Initializing git repository..."
    git init
fi

# Add remote
echo "🔗 Setting up remote..."
git remote remove origin 2>/dev/null || true
git remote add origin "https://github.com/$GITHUB_USER/$REPO_NAME.git"

# Create .gitattributes
echo "📄 Creating .gitattributes..."
cat > .gitattributes << 'EOF'
# Auto detect text files and perform LF normalization
* text=auto

# Python files
*.py text
*.pyw text
*.pyx text
*.pyi text

# Jupyter notebooks
*.ipynb filter=nbstripout

# Documentation
*.md text
*.rst text
*.txt text

# Config files
*.json text
*.yaml text
*.yml text
*.toml text
*.ini text
*.cfg text

# Scripts
*.sh text eol=lf
*.bash text eol=lf

# Data files
*.csv text
*.tsv text

# Binary files
*.png binary
*.jpg binary
*.jpeg binary
*.gif binary
*.ico binary
*.pdf binary
*.pt binary
*.pth binary
*.pkl binary
*.pickle binary
*.npy binary
*.npz binary
*.gguf binary
*.bin binary
EOF

# Stage all files
echo "📦 Staging files..."
git add .

# Create initial commit
echo "💾 Creating initial commit..."
git commit -m "🎉 Initial commit: JIGYASA AGI System

- Self-correcting reasoning with Chain-of-Verification
- Continuous learning through SEAL
- Advanced reasoning with ProRL
- Byte-level transformer architecture
- Model compression for deployment
- Constitutional AI safety measures

Built with ❤️ using cutting-edge AI research"

# Push to GitHub
echo "⬆️  Pushing to GitHub..."
git branch -M main
git push -u origin main

# Enable GitHub Pages
echo "📄 Enabling GitHub Pages..."
gh api repos/$GITHUB_USER/$REPO_NAME/pages \
    --method POST \
    --field source='{"branch":"main","path":"/docs"}' \
    2>/dev/null || echo "GitHub Pages might already be enabled"

# Set up topics
echo "🏷️  Adding repository topics..."
gh api repos/$GITHUB_USER/$REPO_NAME/topics \
    --method PUT \
    --field names='["artificial-intelligence","agi","machine-learning","self-improvement","deep-learning","transformer","continuous-learning","pytorch","nlp","reasoning"]' \
    2>/dev/null || true

echo ""
echo "✅ GitHub setup complete!"
echo ""
echo "📍 Repository URL: https://github.com/$GITHUB_USER/$REPO_NAME"
echo "📄 GitHub Pages: https://$GITHUB_USER.github.io/$REPO_NAME"
echo ""
echo "Next steps:"
echo "1. Update README.md with your GitHub username"
echo "2. Set up secrets for GitHub Actions:"
echo "   - Go to Settings > Secrets and variables > Actions"
echo "   - Add HUGGINGFACE_TOKEN if needed"
echo "3. Create documentation in the docs/ folder"
echo "4. Star ⭐ your own repository!"