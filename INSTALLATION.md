# JIGYASA Installation Guide

## 🚀 Quick Install (Recommended)

### One-line installation:
```bash
curl -fsSL https://raw.githubusercontent.com/Sairamg18814/jigyasa/main/install_jigyasa.sh | bash
```

Or if you've already cloned the repository:
```bash
./install_jigyasa.sh
```

## 📋 What Gets Installed

The installation script will:

1. **Check System Requirements**
   - Python 3.8+ verification
   - OS detection (Mac/Linux)
   - Git configuration

2. **Install Python Dependencies**
   - Core packages (torch, transformers, numpy)
   - GUI packages (flask, socketio, etc.)
   - Development tools (pytest, black)
   - With automatic retry on failure

3. **Install & Configure Ollama**
   - Download Ollama if not present
   - Start Ollama service
   - Pull llama3.1:8b model (~4GB)
   - Test API connectivity

4. **Setup Project Structure**
   - Create necessary directories
   - Initialize .env configuration
   - Setup logging directories
   - Create desktop shortcuts

## 🖥️ Launching the GUI

After installation, launch JIGYASA with:
```bash
./launch_gui.sh
```

The launcher will:
- ✅ Check all services are running
- ✅ Start Ollama if needed
- ✅ Verify model availability
- ✅ Check port availability
- ✅ Open browser automatically
- ✅ Show real-time logs

## 🔧 Manual Installation

If you prefer manual setup:

### 1. Install Python Dependencies
```bash
pip install -r jigyasa/requirements.txt
pip install -r jigyasa/gui/requirements.txt
```

### 2. Install Ollama
**Mac:**
```bash
brew install ollama
# or download from https://ollama.com/download/mac
```

**Linux:**
```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### 3. Start Ollama & Get Model
```bash
ollama serve  # In one terminal
ollama pull llama3.1:8b  # In another terminal
```

### 4. Setup Environment
```bash
cp .env.example .env  # Edit as needed
mkdir -p .jigyasa/knowledge .jigyasa/backups
```

### 5. Launch
```bash
python launch_gui.py
```

## 🛠️ Installation Script Features

### Retry Logic
- Automatically retries failed commands up to 3 times
- Exponential backoff between attempts
- Clear error messages

### Error Handling
```bash
# The script handles:
- Missing dependencies
- Network failures  
- Permission issues
- Port conflicts
```

### Progress Tracking
- Color-coded output
- Timestamp for each operation
- Success/failure indicators
- Detailed error messages

## 🚨 Troubleshooting

### Common Issues

**1. Ollama Connection Failed**
```bash
# Start Ollama manually
ollama serve

# Check if running
curl http://localhost:11434/api/tags
```

**2. Port 5000 Already in Use**
```bash
# The launcher offers options:
- Kill existing process
- Use alternative port
- Manual resolution
```

**3. Model Download Fails**
```bash
# Manual download
ollama pull llama3.1:8b

# Check available models
ollama list
```

**4. Permission Denied**
```bash
# Make scripts executable
chmod +x install_jigyasa.sh launch_gui.sh

# Run with proper permissions
sudo ./install_jigyasa.sh  # If needed
```

## 📁 Directory Structure

After installation:
```
jigyasa/
├── .jigyasa/               # Runtime data
│   ├── knowledge/         # Knowledge base
│   ├── backups/          # Code backups
│   └── logs/            # Application logs
├── .jigyasa_backups/     # Self-editing backups
├── .env                  # Configuration
├── install_jigyasa.sh    # Installation script
├── launch_gui.sh         # GUI launcher
└── jigyasa/             # Main application
    └── gui/            # GUI components
```

## 🔄 Updating JIGYASA

To update to the latest version:
```bash
git pull origin main
./install_jigyasa.sh  # Re-run installer
```

## 🎯 Post-Installation

After successful installation:

1. **Access the GUI**: http://localhost:5000
2. **View real-time metrics**: Learning curves, performance graphs
3. **Try code optimization**: Paste Python code to optimize
4. **Test Beyond RAG**: Search for latest information
5. **Watch self-editing**: See JIGYASA improve itself

## 📊 System Requirements

### Minimum:
- **OS**: macOS 10.15+ or Linux (Ubuntu 20.04+)
- **Python**: 3.8+
- **RAM**: 8GB
- **Storage**: 10GB free space
- **CPU**: 4 cores

### Recommended:
- **RAM**: 16GB+
- **GPU**: NVIDIA with CUDA (optional)
- **Storage**: 20GB+ for models

## 🆘 Getting Help

If installation fails:

1. Check the log files in `.jigyasa/logs/`
2. Run diagnostics: `python -m jigyasa.diagnostics`
3. Open an issue: https://github.com/Sairamg18814/jigyasa/issues
4. Include:
   - Error messages
   - OS version
   - Python version
   - Log files

## ✅ Verification

After installation, verify everything works:
```bash
# Test Ollama
ollama run llama3.1:8b "Hello"

# Test JIGYASA import
python -c "import jigyasa; print('✓ Import successful')"

# Test GUI
curl http://localhost:5000/api/status
```

Happy coding with JIGYASA! 🚀