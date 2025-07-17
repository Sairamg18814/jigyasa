#!/usr/bin/env python3
"""
JIGYASA GUI Launcher - Port 5001
Launch the autonomous AGI dashboard on alternative port
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    """Launch JIGYASA GUI"""
    print("🧠 JIGYASA - Autonomous AGI System")
    print("=" * 50)
    print("🌐 Launching Web Dashboard...")
    
    try:
        # Launch web interface
        from jigyasa.web.app import app
        
        print("✅ Starting JIGYASA Web Server...")
        print("📱 Open your browser to: http://localhost:5001")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 50)
        
        # Run Flask app on port 5001
        app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
        
    except ImportError as e:
        print(f"📦 Installing missing dependencies: {e}")
        
        # Auto-install missing packages
        import subprocess
        packages = ['flask']
        
        for package in packages:
            try:
                print(f"Installing {package}...")
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], 
                              check=True, capture_output=True)
            except subprocess.CalledProcessError:
                print(f"⚠️ Could not install {package}")
        
        # Try again
        try:
            from jigyasa.web.app import app
            print("✅ Starting JIGYASA Web Server...")
            print("📱 Open your browser to: http://localhost:5001")
            print("🛑 Press Ctrl+C to stop")
            app.run(host='0.0.0.0', port=5001, debug=False, threaded=True)
        except Exception as e2:
            print(f"❌ Failed to launch web GUI: {e2}")
            print("🔧 Falling back to command line mode...")
            
            # Fallback to CLI
            from jigyasa.main import main as cli_main
            sys.argv = [sys.argv[0], '--mode', 'interactive']
            cli_main()

if __name__ == "__main__":
    main()