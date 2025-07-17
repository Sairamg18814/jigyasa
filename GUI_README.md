# JIGYASA GUI - Real-Time Dashboard

## 🚀 Features

The JIGYASA GUI provides a comprehensive real-time dashboard with:

### 📊 Real-Time Learning Curves
- Live visualization of knowledge acquisition
- Learning rate tracking
- Pattern recognition progress
- Accuracy metrics over time

### 🤖 Self-Editing Code Visualization
- Watch JIGYASA modify its own code in real-time
- Performance gain indicators
- Before/after code comparisons
- Automatic rollback capabilities

### 🔍 Beyond RAG Integration
- Real-time information retrieval from multiple sources:
  - Web search
  - GitHub repositories
  - ArXiv papers
  - Stack Overflow
  - Documentation sites
- Continuous knowledge updates
- Intelligent caching system

### 🔄 Automatic GitHub Sync
- Auto-commits changes every 5 minutes
- Descriptive commit messages
- Change tracking
- Manual sync button

### 📈 Live Performance Metrics
- CPU and memory usage graphs
- Response time tracking
- Throughput monitoring
- Real-time updates via WebSockets

## 🖥️ How to Launch

```bash
# Simple launch
python launch_gui.py

# The GUI will:
# 1. Install missing dependencies automatically
# 2. Start the Flask server
# 3. Open your browser to http://localhost:5000
```

## 📱 Dashboard Components

### 1. **Metrics Cards**
- Learning Rate
- Code Improvements Count
- Average Performance Gain
- Knowledge Base Size

### 2. **Real-Time Charts**
- Learning curve (knowledge count + learning rate)
- Performance metrics (CPU, Memory, Response Time)
- Live updates every second

### 3. **Code Optimizer**
- Paste Python code
- Get AI-powered optimizations
- See performance improvements
- Syntax highlighting

### 4. **Self-Editing Activity Log**
- Live feed of code modifications
- File paths and improvements
- Performance gain badges
- Timestamp tracking

### 5. **Beyond RAG Search**
- Multi-source search
- Synthesized summaries
- Key points extraction
- Source attribution

### 6. **Chat Interface**
- Interactive chat with JIGYASA
- Context-aware responses
- Learning from conversations

## 🛠️ Technical Details

### Backend
- Flask + Flask-SocketIO for real-time updates
- WebSocket connections for live data
- RESTful API endpoints
- Automatic component initialization

### Frontend
- Bootstrap 5 for responsive design
- Chart.js with streaming plugin
- Prism.js for code highlighting
- Socket.IO client for real-time updates

### Data Flow
1. Backend monitors system metrics
2. WebSocket emits updates to frontend
3. Charts update in real-time
4. User actions trigger API calls
5. GitHub sync runs automatically

## 🔧 Configuration

The GUI automatically:
- Detects and installs missing packages
- Initializes Ollama connection
- Sets up GitHub sync
- Starts monitoring threads

## 🎨 Features in Action

### Real-Time Learning Visualization
- Watch knowledge count increase
- See learning rate fluctuations
- Track pattern recognition

### Code Self-Editing
- Green highlights for improvements
- Performance percentage badges
- Automatic GitHub commits

### Beyond RAG Results
- Blue-bordered result cards
- Multiple source integration
- Relevance scoring

## 🚦 Status Indicators

- 🟢 **Green**: System online and connected
- 🔵 **Blue**: Active monitoring
- 🟡 **Yellow**: Processing
- 🔴 **Red**: Error or offline

## 📌 Tips

1. **Performance**: The dashboard is optimized for modern browsers
2. **Updates**: Real-time data flows automatically - no refresh needed
3. **GitHub Sync**: Happens every 5 minutes or on-demand
4. **Search**: Beyond RAG caches results for 1 hour

## 🐛 Troubleshooting

If the GUI doesn't start:
1. Ensure Ollama is running: `ollama serve`
2. Check port 5000 is available
3. Verify llama3.1:8b model: `ollama pull llama3.1:8b`
4. Check console for specific errors

## 🎯 Next Steps

The GUI will continue to evolve with:
- More visualization options
- Enhanced Beyond RAG sources
- Collaborative features
- Export capabilities

Enjoy watching JIGYASA learn and improve in real-time! 🚀