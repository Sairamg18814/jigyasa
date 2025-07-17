#!/usr/bin/env python3
"""
JIGYASA GUI Launcher
Launch the autonomous AGI dashboard with real-time monitoring
"""

import sys
import os
import subprocess
import webbrowser
import time
import threading
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_requirements():
    """Check and install required packages"""
    required_packages = [
        'flask',
        'flask-socketio',
        'flask-cors',
        'python-socketio[client]',
        'gitpython',
        'beautifulsoup4',
        'feedparser',
        'aiohttp'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            pkg_name = package.split('[')[0].replace('-', '_')
            __import__(pkg_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"📦 Installing missing packages: {', '.join(missing_packages)}")
        subprocess.run([sys.executable, '-m', 'pip', 'install'] + missing_packages)

def main():
    """Launch JIGYASA GUI"""
    print("🧠 JIGYASA - Autonomous AGI System")
    print("=" * 50)
    print("🚀 Advanced GUI Dashboard with:")
    print("  📊 Real-time learning curves")
    print("  🤖 Self-editing visualization")
    print("  🔍 Beyond RAG integration")
    print("  🔄 Automatic GitHub sync")
    print("  📈 Live performance metrics")
    print("=" * 50)
    
    # Check requirements
    check_requirements()
    
    print("\n🌐 Launching Enhanced Web Dashboard...")
    
    # Open browser after a delay
    def open_browser():
        time.sleep(3)
        webbrowser.open('http://localhost:5000')
    
    browser_thread = threading.Thread(target=open_browser)
    browser_thread.daemon = True
    browser_thread.start()
    
    try:
        # Try new GUI first
        from jigyasa.gui.app import app, socketio, initialize_system
        
        print("✅ Initializing JIGYASA components...")
        initialize_system()
        
        print("📱 Open your browser to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 50)
        
        # Run Flask app with SocketIO
        socketio.run(app, debug=False, port=5000, host='0.0.0.0')
        
    except ImportError as e:
        print(f"\n⚠️  New GUI not available, trying legacy interface...")
        
        try:
            # Fallback to old web interface
            from jigyasa.web.app import app
            print("✅ Starting legacy JIGYASA Web Server...")
            print("📱 Open your browser to: http://localhost:5000")
            print("🛑 Press Ctrl+C to stop")
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        except Exception as e2:
            print(f"❌ Failed to launch web GUI: {e2}")
            print("🔧 Falling back to command line mode...")
            
            # Fallback to CLI
            from jigyasa.main import main as cli_main
            sys.argv = [sys.argv[0], '--mode', 'interactive']
            cli_main()
    
    except KeyboardInterrupt:
        print("\n\n👋 Shutting down JIGYASA GUI...")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Make sure Ollama is running: ollama serve")
        print("2. Ensure llama3.1:8b model is available: ollama pull llama3.1:8b")
        print("3. Check if port 5000 is available")
        print(f"4. Check the error: {e}")

if __name__ == "__main__":
    main()