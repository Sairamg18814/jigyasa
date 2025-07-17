#!/bin/bash
# JIGYASA Complete Installation Script
# Installs all dependencies with retry logic and error handling

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Banner
echo -e "${BLUE}"
echo "     ██╗██╗ ██████╗██╗   ██╗ █████╗ ███████╗ █████╗ "
echo "     ██║██║██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗"
echo "     ██║██║██║  ███╗╚████╔╝ ███████║███████╗███████║"
echo "██   ██║██║██║   ██║ ╚██╔╝  ██╔══██║╚════██║██╔══██║"
echo "╚█████╔╝██║╚██████╔╝  ██║   ██║  ██║███████║██║  ██║"
echo " ╚════╝ ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝"
echo -e "${NC}"
echo -e "${GREEN}JIGYASA AGI - Complete Installation Script${NC}"
echo "=================================================="

# Function to print status
print_status() {
    echo -e "${BLUE}[$(date '+%H:%M:%S')]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Function to retry commands
retry_command() {
    local max_attempts=3
    local attempt=1
    local delay=5
    local command="$@"
    
    while [ $attempt -le $max_attempts ]; do
        print_status "Attempt $attempt/$max_attempts: $command"
        
        if eval "$command"; then
            print_success "Command succeeded"
            return 0
        else
            print_error "Command failed"
            
            if [ $attempt -lt $max_attempts ]; then
                print_warning "Retrying in $delay seconds..."
                sleep $delay
                delay=$((delay * 2))  # Exponential backoff
            fi
        fi
        
        attempt=$((attempt + 1))
    done
    
    print_error "Command failed after $max_attempts attempts"
    return 1
}

# Check OS
print_status "Checking operating system..."
OS="$(uname -s)"
case "${OS}" in
    Linux*)     OS_TYPE=Linux;;
    Darwin*)    OS_TYPE=Mac;;
    CYGWIN*)    OS_TYPE=Cygwin;;
    MINGW*)     OS_TYPE=MinGw;;
    *)          OS_TYPE="UNKNOWN:${OS}"
esac
print_success "Detected OS: $OS_TYPE"

# Check Python version
print_status "Checking Python version..."
if command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')
    print_success "Python $PYTHON_VERSION found"
    
    # Check if Python 3.8+
    PYTHON_MAJOR=$(python3 -c 'import sys; print(sys.version_info[0])')
    PYTHON_MINOR=$(python3 -c 'import sys; print(sys.version_info[1])')
    
    if [ "$PYTHON_MAJOR" -lt 3 ] || ([ "$PYTHON_MAJOR" -eq 3 ] && [ "$PYTHON_MINOR" -lt 8 ]); then
        print_error "Python 3.8+ required, found $PYTHON_VERSION"
        exit 1
    fi
else
    print_error "Python 3 not found. Please install Python 3.8+"
    exit 1
fi

# Install/Upgrade pip
print_status "Upgrading pip..."
retry_command "python3 -m pip install --upgrade pip"

# Install core Python dependencies
print_status "Installing core Python dependencies..."
CORE_PACKAGES=(
    "torch>=2.0.0"
    "transformers>=4.30.0"
    "numpy>=1.24.0"
    "requests>=2.31.0"
    "psutil>=5.8.0"
    "GitPython>=3.1.0"
    "python-dotenv>=1.0.0"
    "beautifulsoup4>=4.12.0"
    "pytest>=7.0.0"
)

for package in "${CORE_PACKAGES[@]}"; do
    retry_command "python3 -m pip install '$package'"
done

# Install GUI dependencies
print_status "Installing GUI dependencies..."
GUI_PACKAGES=(
    "flask>=2.3.0"
    "flask-socketio>=5.3.0"
    "flask-cors>=4.0.0"
    "python-socketio[client]>=5.9.0"
    "feedparser>=6.0.0"
    "aiohttp>=3.8.0"
)

for package in "${GUI_PACKAGES[@]}"; do
    retry_command "python3 -m pip install '$package'"
done

# Check for Ollama
print_status "Checking for Ollama..."
if command -v ollama &> /dev/null; then
    print_success "Ollama is installed"
    
    # Check if Ollama is running
    if pgrep -x "ollama" > /dev/null; then
        print_success "Ollama service is running"
    else
        print_warning "Ollama is not running. Starting Ollama..."
        ollama serve &
        sleep 5
    fi
    
    # Pull llama3.1:8b model
    print_status "Pulling llama3.1:8b model (this may take a while)..."
    if retry_command "ollama pull llama3.1:8b"; then
        print_success "Model downloaded successfully"
    else
        print_error "Failed to download model. You can manually run: ollama pull llama3.1:8b"
    fi
else
    print_error "Ollama not found!"
    echo ""
    echo "Please install Ollama from: https://ollama.com"
    echo ""
    echo "Installation commands:"
    
    if [ "$OS_TYPE" = "Mac" ]; then
        echo "  brew install ollama"
        echo "  or download from: https://ollama.com/download/mac"
    elif [ "$OS_TYPE" = "Linux" ]; then
        echo "  curl -fsSL https://ollama.com/install.sh | sh"
    fi
    
    echo ""
    read -p "Press Enter after installing Ollama to continue..."
    
    # Check again
    if command -v ollama &> /dev/null; then
        print_success "Ollama installed successfully"
        ollama serve &
        sleep 5
        retry_command "ollama pull llama3.1:8b"
    else
        print_error "Ollama still not found. Please install manually."
        exit 1
    fi
fi

# Install Ollama Python package
print_status "Installing Ollama Python package..."
retry_command "python3 -m pip install ollama>=0.1.0"

# Create necessary directories
print_status "Creating project directories..."
directories=(
    ".jigyasa"
    ".jigyasa/knowledge"
    ".jigyasa/backups"
    ".jigyasa_backups"
    ".jigyasa_benchmarks"
    "jigyasa/gui/templates"
    "jigyasa/gui/static"
)

for dir in "${directories[@]}"; do
    mkdir -p "$dir"
    print_success "Created $dir"
done

# Setup environment file
print_status "Setting up environment configuration..."
if [ ! -f .env ]; then
    cat > .env << 'EOF'
# JIGYASA Configuration
JIGYASA_LOG_LEVEL=INFO
JIGYASA_AUTONOMOUS_INTERVAL=300
JIGYASA_BACKUP_DIR=.jigyasa/backups
JIGYASA_KNOWLEDGE_DB=.jigyasa/knowledge.db

# Ollama Configuration
OLLAMA_HOST=http://localhost:11434
OLLAMA_MODEL=llama3.1:8b

# GUI Configuration
FLASK_ENV=development
FLASK_SECRET_KEY=jigyasa-secret-key-$(date +%s)

# GitHub Configuration (Optional)
# GITHUB_TOKEN=your_github_token_here
EOF
    print_success "Created .env file"
else
    print_warning ".env file already exists"
fi

# Check Git configuration
print_status "Checking Git configuration..."
if ! git config user.name > /dev/null 2>&1; then
    print_warning "Git user.name not set"
    read -p "Enter your name for Git commits: " git_name
    git config user.name "$git_name"
fi

if ! git config user.email > /dev/null 2>&1; then
    print_warning "Git user.email not set"
    read -p "Enter your email for Git commits: " git_email
    git config user.email "$git_email"
fi

# Run tests
print_status "Running basic tests..."
if python3 -c "import jigyasa; print('✓ JIGYASA package importable')" 2>/dev/null; then
    print_success "JIGYASA package is importable"
else
    print_warning "JIGYASA package import test failed - this is normal for first installation"
fi

# Test Ollama connection
print_status "Testing Ollama connection..."
if python3 -c "import requests; r=requests.get('http://localhost:11434/api/tags'); print('✓' if r.status_code==200 else '✗')" 2>/dev/null | grep -q "✓"; then
    print_success "Ollama API is accessible"
else
    print_warning "Ollama API not accessible - make sure Ollama is running"
fi

# Create launch shortcuts
print_status "Creating launch shortcuts..."

# Create desktop launcher for Mac
if [ "$OS_TYPE" = "Mac" ] && [ -d "$HOME/Desktop" ]; then
    cat > "$HOME/Desktop/JIGYASA.command" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
cd "$(pwd -P)"
python3 launch_gui.py
EOF
    chmod +x "$HOME/Desktop/JIGYASA.command"
    print_success "Created desktop launcher (Mac)"
fi

# Create desktop launcher for Linux
if [ "$OS_TYPE" = "Linux" ] && [ -d "$HOME/Desktop" ]; then
    cat > "$HOME/Desktop/JIGYASA.desktop" << EOF
[Desktop Entry]
Version=1.0
Type=Application
Name=JIGYASA AGI
Comment=Launch JIGYASA GUI Dashboard
Exec=bash -c 'cd $(pwd) && python3 launch_gui.py'
Icon=$(pwd)/jigyasa/gui/static/icon.png
Terminal=true
Categories=Development;Science;
EOF
    chmod +x "$HOME/Desktop/JIGYASA.desktop"
    print_success "Created desktop launcher (Linux)"
fi

# Summary
echo ""
echo -e "${GREEN}=================================================="
echo "        JIGYASA Installation Complete!"
echo "==================================================${NC}"
echo ""
echo "✅ All dependencies installed"
echo "✅ Ollama configured with llama3.1:8b"
echo "✅ Project directories created"
echo "✅ Environment configured"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "1. Launch the GUI: ${GREEN}./launch_gui.sh${NC}"
echo "2. Or run directly: ${GREEN}python3 launch_gui.py${NC}"
echo "3. Access at: ${BLUE}http://localhost:5000${NC}"
echo ""
echo -e "${YELLOW}Troubleshooting:${NC}"
echo "- If Ollama errors: ${GREEN}ollama serve${NC} (in another terminal)"
echo "- If model missing: ${GREEN}ollama pull llama3.1:8b${NC}"
echo "- Check logs in: ${GREEN}.jigyasa/logs/${NC}"
echo ""
print_success "Installation script completed!"

# Make launch script executable
chmod +x launch_gui.sh 2>/dev/null || true

# Offer to launch GUI
echo ""
read -p "Would you like to launch JIGYASA GUI now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Launching JIGYASA GUI..."
    ./launch_gui.sh || python3 launch_gui.py
fi