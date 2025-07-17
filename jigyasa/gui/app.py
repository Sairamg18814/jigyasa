#!/usr/bin/env python3
"""
JIGYASA GUI Application - Real-time Dashboard
Features: Learning curves, self-editing visualization, Beyond RAG
"""

from flask import Flask, render_template, jsonify, request, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
import json
import os
import sys
import time
import threading
import queue
from datetime import datetime, timedelta
import git
import subprocess
from pathlib import Path

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jigyasa.core.jigyasa_agi import JigyasaAGI
from jigyasa.autonomous.real_self_editor import RealSelfEditor
from jigyasa.learning.real_continuous_learning import RealContinuousLearning
from jigyasa.performance.real_benchmarks import RealBenchmarks
from jigyasa.models.ollama_wrapper import OllamaWrapper
from jigyasa.gui.beyond_rag import BeyondRAG
from jigyasa.gui.github_sync import GitHubAutoSync

app = Flask(__name__)
app.config['SECRET_KEY'] = 'jigyasa-secret-key'
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Global instances
jigyasa = None
self_editor = None
continuous_learner = None
benchmarks = None
beyond_rag = None
github_sync = None

# Real-time data queues
learning_data_queue = queue.Queue()
editing_data_queue = queue.Queue()
performance_data_queue = queue.Queue()

class RealTimeMonitor:
    """Monitor system metrics in real-time"""
    
    def __init__(self):
        self.running = False
        self.threads = []
        
    def start(self):
        """Start all monitoring threads"""
        self.running = True
        
        # Learning curve monitor
        learning_thread = threading.Thread(target=self.monitor_learning)
        learning_thread.daemon = True
        learning_thread.start()
        self.threads.append(learning_thread)
        
        # Self-editing monitor
        editing_thread = threading.Thread(target=self.monitor_editing)
        editing_thread.daemon = True
        editing_thread.start()
        self.threads.append(editing_thread)
        
        # Performance monitor
        performance_thread = threading.Thread(target=self.monitor_performance)
        performance_thread.daemon = True
        performance_thread.start()
        self.threads.append(performance_thread)
        
    def stop(self):
        """Stop all monitoring threads"""
        self.running = False
        for thread in self.threads:
            thread.join()
            
    def monitor_learning(self):
        """Monitor learning progress"""
        while self.running:
            if continuous_learner:
                # Get learning metrics
                metrics = continuous_learner.get_learning_metrics()
                data = {
                    'timestamp': datetime.now().isoformat(),
                    'knowledge_count': metrics.get('total_knowledge', 0),
                    'learning_rate': metrics.get('learning_rate', 0),
                    'accuracy': metrics.get('accuracy', 0),
                    'patterns_learned': metrics.get('patterns_learned', 0)
                }
                socketio.emit('learning_update', data)
            time.sleep(1)
            
    def monitor_editing(self):
        """Monitor code self-editing"""
        while self.running:
            if self_editor and hasattr(self_editor, 'modifications_log'):
                # Get latest modifications
                if self_editor.modifications_log:
                    latest = self_editor.modifications_log[-1]
                    data = {
                        'timestamp': latest['timestamp'],
                        'file': latest['file'],
                        'improvements': latest['improvements'],
                        'performance_gain': latest['performance_gain']
                    }
                    socketio.emit('editing_update', data)
            time.sleep(2)
            
    def monitor_performance(self):
        """Monitor system performance"""
        while self.running:
            if benchmarks:
                # Get performance metrics
                metrics = benchmarks.get_current_metrics()
                data = {
                    'timestamp': datetime.now().isoformat(),
                    'cpu_usage': metrics.get('cpu_percent', 0),
                    'memory_usage': metrics.get('memory_percent', 0),
                    'response_time': metrics.get('avg_response_time', 0),
                    'throughput': metrics.get('throughput', 0)
                }
                socketio.emit('performance_update', data)
            time.sleep(1)

# Initialize monitor
monitor = RealTimeMonitor()

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get system status"""
    return jsonify({
        'status': 'online',
        'jigyasa': jigyasa is not None,
        'self_editor': self_editor is not None,
        'beyond_rag': beyond_rag is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/learning/history')
def get_learning_history():
    """Get learning history"""
    if continuous_learner:
        history = continuous_learner.get_learning_history(limit=100)
        return jsonify(history)
    return jsonify([])

@app.route('/api/editing/history')
def get_editing_history():
    """Get code editing history"""
    if self_editor:
        return jsonify(self_editor.modifications_log)
    return jsonify([])

@app.route('/api/performance/metrics')
def get_performance_metrics():
    """Get performance metrics"""
    if benchmarks:
        return jsonify(benchmarks.get_all_metrics())
    return jsonify({})

@app.route('/api/chat', methods=['POST'])
def chat():
    """Chat with JIGYASA"""
    data = request.json
    query = data.get('query', '')
    
    if jigyasa:
        response = jigyasa.chat(query)
        
        # Update Beyond RAG if needed
        if beyond_rag:
            beyond_rag.update_knowledge(query, response)
            
        return jsonify({
            'response': response,
            'timestamp': datetime.now().isoformat()
        })
    
    return jsonify({'error': 'JIGYASA not initialized'}), 500

@app.route('/api/optimize', methods=['POST'])
def optimize_code():
    """Optimize code using JIGYASA"""
    data = request.json
    code = data.get('code', '')
    
    if jigyasa:
        result = jigyasa.analyze_and_improve_code_from_string(code)
        
        # Trigger GitHub sync if enabled
        if github_sync and result.get('improved'):
            github_sync.sync_changes("Code optimization via GUI")
            
        return jsonify(result)
    
    return jsonify({'error': 'JIGYASA not initialized'}), 500

@app.route('/api/self-edit', methods=['POST'])
def trigger_self_edit():
    """Trigger self-editing on a file"""
    data = request.json
    file_path = data.get('file_path', '')
    
    if self_editor and file_path:
        result = self_editor.modify_code_autonomously(file_path)
        
        # Auto-sync to GitHub
        if github_sync and result['status'] == 'success':
            github_sync.sync_changes(f"Self-edit: {Path(file_path).name}")
            
        return jsonify(result)
    
    return jsonify({'error': 'Self-editor not initialized or no file specified'}), 500

@app.route('/api/beyond-rag/search', methods=['POST'])
def beyond_rag_search():
    """Search using Beyond RAG"""
    data = request.json
    query = data.get('query', '')
    
    if beyond_rag:
        results = beyond_rag.search(query)
        return jsonify(results)
    
    return jsonify({'error': 'Beyond RAG not initialized'}), 500

@app.route('/api/github/sync', methods=['POST'])
def sync_to_github():
    """Manually trigger GitHub sync"""
    data = request.json
    message = data.get('message', 'Manual sync from GUI')
    
    if github_sync:
        result = github_sync.sync_changes(message)
        return jsonify(result)
    
    return jsonify({'error': 'GitHub sync not initialized'}), 500

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    emit('connected', {'data': 'Connected to JIGYASA'})

@socketio.on('start_monitoring')
def handle_start_monitoring():
    """Start real-time monitoring"""
    monitor.start()
    emit('monitoring_started', {'status': 'active'})

@socketio.on('stop_monitoring')
def handle_stop_monitoring():
    """Stop real-time monitoring"""
    monitor.stop()
    emit('monitoring_stopped', {'status': 'inactive'})

def initialize_system():
    """Initialize JIGYASA components"""
    global jigyasa, self_editor, continuous_learner, benchmarks, beyond_rag, github_sync
    
    print("🚀 Initializing JIGYASA GUI components...")
    
    # Initialize Ollama
    ollama = OllamaWrapper(model_name="llama3.1:8b")
    
    # Initialize core components
    jigyasa = JigyasaAGI(ollama_wrapper=ollama)
    self_editor = RealSelfEditor(ollama)
    continuous_learner = RealContinuousLearning(ollama)
    benchmarks = RealBenchmarks()
    
    # Initialize Beyond RAG
    beyond_rag = BeyondRAG(ollama)
    
    # Initialize GitHub sync
    github_sync = GitHubAutoSync(
        repo_path=os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        auto_sync=True
    )
    
    print("✅ All components initialized!")

if __name__ == '__main__':
    initialize_system()
    socketio.run(app, debug=True, port=5000)