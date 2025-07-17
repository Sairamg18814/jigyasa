#!/bin/bash
# JIGYASA GUI Launcher Script
# Ensures all services are running and launches the GUI

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Banner
clear
echo -e "${PURPLE}"
echo "     ██╗██╗ ██████╗██╗   ██╗ █████╗ ███████╗ █████╗ "
echo "     ██║██║██╔════╝╚██╗ ██╔╝██╔══██╗██╔════╝██╔══██╗"
echo "     ██║██║██║  ███╗╚████╔╝ ███████║███████╗███████║"
echo "██   ██║██║██║   ██║ ╚██╔╝  ██╔══██║╚════██║██╔══██║"
echo "╚█████╔╝██║╚██████╔╝  ██║   ██║  ██║███████║██║  ██║"
echo " ╚════╝ ╚═╝ ╚═════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝"
echo -e "${NC}"
echo -e "${CYAN}        🧠 Autonomous General Intelligence 🧠${NC}"
echo -e "${GREEN}=================================================="
echo "              GUI Dashboard Launcher"
echo "==================================================${NC}"
echo ""

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

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if a process is running
process_running() {
    pgrep -x "$1" > /dev/null
}

# Function to check if port is in use
port_in_use() {
    lsof -i :$1 > /dev/null 2>&1 || netstat -an | grep -q ":$1.*LISTEN"
}

# Change to script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Pre-flight checks
print_status "Running pre-flight checks..."

# Check Python
if command_exists python3; then
    PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
    print_success "Python $PYTHON_VERSION found"
else
    print_error "Python 3 not found!"
    echo "Please install Python 3.8+ first"
    exit 1
fi

# Check if installation is complete
if [ ! -f ".env" ] || [ ! -d ".jigyasa" ]; then
    print_warning "JIGYASA not fully installed"
    echo ""
    read -p "Would you like to run the installation script? (y/n) " -n 1 -r
    echo ""
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        if [ -f "install_jigyasa.sh" ]; then
            chmod +x install_jigyasa.sh
            ./install_jigyasa.sh
        else
            print_error "install_jigyasa.sh not found!"
            exit 1
        fi
    else
        print_error "Please run ./install_jigyasa.sh first"
        exit 1
    fi
fi

# Check and start Ollama
print_status "Checking Ollama service..."
if command_exists ollama; then
    if process_running "ollama" || ollama list >/dev/null 2>&1; then
        print_success "Ollama is running"
    else
        print_warning "Ollama not running, starting it..."
        # Start Ollama in background
        nohup ollama serve > .jigyasa/ollama.log 2>&1 &
        sleep 3
        
        # Check if it started
        if ollama list >/dev/null 2>&1; then
            print_success "Ollama started successfully"
        else
            print_error "Failed to start Ollama"
            echo "Please start Ollama manually: ollama serve"
            echo "Check logs at: .jigyasa/ollama.log"
            exit 1
        fi
    fi
    
    # Check for model
    print_status "Checking for llama3.1:8b model..."
    if ollama list | grep -q "llama3.1:8b"; then
        print_success "Model llama3.1:8b is available"
    else
        print_warning "Model not found, downloading..."
        echo "This may take several minutes on first run..."
        if ollama pull llama3.1:8b; then
            print_success "Model downloaded successfully"
        else
            print_error "Failed to download model"
            echo "Please run manually: ollama pull llama3.1:8b"
            exit 1
        fi
    fi
else
    print_error "Ollama not installed!"
    echo "Please install from: https://ollama.com"
    exit 1
fi

# Check port availability
print_status "Checking port 5000..."
if port_in_use 5000; then
    print_warning "Port 5000 is already in use"
    echo ""
    echo "Options:"
    echo "1. Kill the process using port 5000"
    echo "2. Use a different port"
    echo "3. Exit"
    echo ""
    read -p "Choose option (1-3): " choice
    
    case $choice in
        1)
            print_status "Killing process on port 5000..."
            if [[ "$OSTYPE" == "darwin"* ]]; then
                lsof -ti:5000 | xargs kill -9 2>/dev/null || true
            else
                fuser -k 5000/tcp 2>/dev/null || true
            fi
            sleep 2
            print_success "Port 5000 cleared"
            ;;
        2)
            read -p "Enter alternative port (e.g., 5001): " ALT_PORT
            export JIGYASA_PORT=$ALT_PORT
            print_success "Will use port $ALT_PORT"
            ;;
        3)
            print_warning "Exiting..."
            exit 0
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
else
    print_success "Port 5000 is available"
fi

# Create necessary directories if missing
print_status "Ensuring directories exist..."
mkdir -p .jigyasa/logs
mkdir -p .jigyasa/knowledge
mkdir -p .jigyasa_backups
mkdir -p jigyasa/gui/templates
mkdir -p jigyasa/gui/static

# Load environment variables
if [ -f .env ]; then
    print_status "Loading environment variables..."
    set -a
    source .env
    set +a
    print_success "Environment loaded"
fi

# GUI Features animation
echo ""
echo -e "${CYAN}🚀 Launching JIGYASA GUI with:${NC}"
sleep 0.5
echo -e "  ${GREEN}📊${NC} Real-time learning curves"
sleep 0.3
echo -e "  ${GREEN}🤖${NC} Self-editing code visualization"
sleep 0.3
echo -e "  ${GREEN}🔍${NC} Beyond RAG for live updates"
sleep 0.3
echo -e "  ${GREEN}🔄${NC} Automatic GitHub sync"
sleep 0.3
echo -e "  ${GREEN}📈${NC} Live performance metrics"
sleep 0.3
echo -e "  ${GREEN}💬${NC} Interactive chat interface"
sleep 0.5
echo ""

# Launch GUI
print_status "Starting JIGYASA GUI..."
echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

# Set Python path
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH}"

# Handle Ctrl+C gracefully
trap 'echo -e "\n\n${YELLOW}Shutting down JIGYASA GUI...${NC}"; exit 0' INT

# Launch with custom port if set
if [ -n "$JIGYASA_PORT" ]; then
    print_status "Launching on port $JIGYASA_PORT..."
    python3 launch_gui.py --port $JIGYASA_PORT 2>&1 | tee .jigyasa/logs/gui_$(date +%Y%m%d_%H%M%S).log
else
    # Launch GUI with logging
    python3 launch_gui.py 2>&1 | tee .jigyasa/logs/gui_$(date +%Y%m%d_%H%M%S).log
fi

# If GUI exits
echo ""
print_warning "JIGYASA GUI has stopped"
echo ""
echo "Check logs in: .jigyasa/logs/"
echo "To restart: ./launch_gui.sh"