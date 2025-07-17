#!/usr/bin/env python3
"""
JIGYASA - Autonomous General Intelligence
Main entry point for the AGI system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import argparse
import logging
from pathlib import Path

from jigyasa.core.jigyasa_agi import JigyasaAGI

def setup_logging(verbose: bool = False):
    """Setup logging configuration"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="JIGYASA - Autonomous General Intelligence"
    )
    
    parser.add_argument(
        "command",
        choices=["chat", "improve", "autonomous", "export", "create-model", "status"],
        help="Command to execute"
    )
    
    parser.add_argument(
        "--path",
        type=str,
        default=".",
        help="Path for code improvement or autonomous mode"
    )
    
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Interval in seconds for autonomous mode"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="jigyasa_knowledge.json",
        help="Output file for export"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    setup_logging(args.verbose)
    
    # Initialize AGI
    print("🧠 Initializing JIGYASA AGI with Llama 3.1:8b...")
    try:
        agi = JigyasaAGI()
    except RuntimeError as e:
        print(f"❌ {e}")
        print("\nPlease ensure Ollama is running:")
        print("1. Install Ollama from https://ollama.com")
        print("2. Run: ollama pull llama3.1:8b")
        print("3. Start Ollama service")
        return
    
    # Execute command
    if args.command == "chat":
        print("💬 JIGYASA Chat (type 'exit' to quit)")
        print("-" * 50)
        
        while True:
            try:
                user_input = input("\n👤 You: ")
                if user_input.lower() in ['exit', 'quit']:
                    break
                    
                response = agi.continuous_learning_chat(user_input)
                print(f"\n🤖 JIGYASA: {response}")
                
            except KeyboardInterrupt:
                break
                
    elif args.command == "improve":
        # Improve a single file or directory
        path = Path(args.path)
        
        if path.is_file():
            print(f"🔧 Improving {path}")
            improvement = agi.analyze_and_improve_code(str(path))
            
            if improvement.improvements:
                print(f"✅ Applied {len(improvement.improvements)} improvements")
                print(f"📈 Performance gain: {improvement.performance_gain:.1%}")
                for imp in improvement.improvements:
                    print(f"  - {imp['type']}: {imp['description']}")
            else:
                print("ℹ️  No improvements needed")
                
        elif path.is_dir():
            print(f"🔧 Improving all Python files in {path}")
            py_files = list(path.rglob("*.py"))
            
            total_improvements = 0
            total_gain = 0.0
            
            for py_file in py_files:
                if any(skip in str(py_file) for skip in ['test', '__pycache__', '.git']):
                    continue
                    
                try:
                    improvement = agi.analyze_and_improve_code(str(py_file))
                    if improvement.improvements:
                        total_improvements += len(improvement.improvements)
                        total_gain += improvement.performance_gain
                        print(f"✅ {py_file.name}: {len(improvement.improvements)} improvements")
                except Exception as e:
                    print(f"❌ {py_file.name}: {e}")
                    
            print(f"\n📊 Total: {total_improvements} improvements")
            print(f"📈 Average performance gain: {total_gain/max(len(py_files), 1):.1%}")
            
    elif args.command == "autonomous":
        print(f"🚀 Starting autonomous mode for {args.path}")
        print(f"⏱️  Checking every {args.interval} seconds")
        print("Press Ctrl+C to stop")
        
        agi.start_autonomous_mode(args.path, args.interval)
        
        try:
            # Keep running until interrupted
            import time
            while True:
                time.sleep(60)
                metrics = agi.get_system_metrics()
                print(f"\r📊 Improvements: {metrics['knowledge']['improvements']} | "
                      f"Avg gain: {metrics['knowledge']['avg_performance_gain']:.1%} | "
                      f"CPU: {metrics['system']['cpu_usage']:.1f}%", end='')
        except KeyboardInterrupt:
            agi.stop_autonomous_mode()
            print("\n✋ Autonomous mode stopped")
            
    elif args.command == "export":
        print(f"📚 Exporting knowledge to {args.output}")
        agi.export_knowledge(args.output)
        print("✅ Export complete")
        
    elif args.command == "create-model":
        print("🏗️  Creating Jigyasa model for Ollama")
        modelfile, instructions = agi.create_specialized_model()
        print(f"✅ Created {modelfile}")
        print("\n" + instructions)
        
    elif args.command == "status":
        print("📊 JIGYASA System Status")
        print("-" * 50)
        
        metrics = agi.get_system_metrics()
        
        print(f"\n🧠 Knowledge Base:")
        print(f"  - Patterns learned: {metrics['knowledge']['patterns']}")
        print(f"  - Insights gained: {metrics['knowledge']['insights']}")
        print(f"  - Code improvements: {metrics['knowledge']['improvements']}")
        print(f"  - Avg performance gain: {metrics['knowledge']['avg_performance_gain']:.1%}")
        
        print(f"\n💻 System Resources:")
        print(f"  - CPU usage: {metrics['system']['cpu_usage']:.1f}%")
        print(f"  - Memory used: {metrics['system']['memory_used_gb']:.1f} GB")
        print(f"  - Memory available: {metrics['system']['memory_available_gb']:.1f} GB")
        
        print(f"\n🤖 Autonomous mode: {'Active' if metrics['autonomous_mode'] else 'Inactive'}")

if __name__ == "__main__":
    main()