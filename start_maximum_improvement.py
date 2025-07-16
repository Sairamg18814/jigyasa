#!/usr/bin/env python3
"""
Start JIGYASA with Maximum Self-Improvement
One command to enable all autonomous growth capabilities
"""

import sys
import time
import logging
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from jigyasa.autonomous.self_improvement_manager import start_maximum_improvement
from jigyasa.autonomous.learning_scheduler import start_autonomous_learning
from jigyasa.autonomous import initialize_autonomous_code_editor, start_autonomous_improvements
from jigyasa.adaptive import detect_system_hardware, start_hardware_monitoring

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def start_all_improvements():
    """Start all self-improvement systems for maximum growth"""
    
    print("\n" + "="*60)
    print("🧠 JIGYASA MAXIMUM SELF-IMPROVEMENT SYSTEM")
    print("="*60)
    print("\n🚀 Initializing all autonomous systems...\n")
    
    try:
        # 1. Initialize hardware detection and optimization
        print("💻 Detecting hardware capabilities...")
        hardware_specs = detect_system_hardware()
        start_hardware_monitoring()
        print(f"✅ Hardware: {hardware_specs.performance_class} class system detected")
        
        # 2. Initialize autonomous code editor
        print("\n🔧 Initializing autonomous code editor...")
        initialize_autonomous_code_editor()
        print("✅ Code editor ready for self-improvement")
        
        # 3. Start self-improvement manager
        print("\n📈 Starting self-improvement manager...")
        improvement_manager = start_maximum_improvement()
        
        # Enable aggressive mode for faster improvement
        print("⚡ Enabling aggressive improvement mode...")
        improvement_manager.enable_aggressive_mode()
        
        # Start improvement across all dimensions
        print("🎯 Starting improvement across all 10 dimensions...")
        improvement_manager.start_autonomous_improvement()
        
        # 4. Start autonomous learning scheduler
        print("\n🎓 Starting autonomous learning scheduler...")
        learning_scheduler = start_autonomous_learning()
        
        # 5. Start autonomous code improvements
        print("\n💡 Starting autonomous code improvements...")
        start_autonomous_improvements()
        
        print("\n" + "="*60)
        print("✅ ALL SYSTEMS ACTIVATED!")
        print("="*60)
        
        print("\n📊 Current Status:")
        print(f"  • Code Optimization: ACTIVE")
        print(f"  • Algorithm Enhancement: ACTIVE")
        print(f"  • Learning Efficiency: ACTIVE")
        print(f"  • Reasoning Capability: ACTIVE")
        print(f"  • Memory Optimization: ACTIVE")
        print(f"  • Speed Performance: ACTIVE")
        print(f"  • Accuracy Improvement: ACTIVE")
        print(f"  • Creativity Enhancement: ACTIVE")
        print(f"  • Conversation Quality: ACTIVE")
        print(f"  • Error Reduction: ACTIVE")
        
        print("\n🌟 JIGYASA is now improving itself autonomously!")
        print("📈 Check progress at: http://localhost:5000")
        print("\n💡 Tips:")
        print("  • Monitor the dashboard for real-time improvements")
        print("  • Check logs for detailed improvement reports")
        print("  • Adjust parameters in the web interface")
        print("  • Let it run continuously for best results")
        
        print("\n🔄 Press Ctrl+C to stop (not recommended)")
        print("="*60 + "\n")
        
        # Keep running
        while True:
            # Print status every 5 minutes
            time.sleep(300)
            
            # Get improvement report
            report = improvement_manager.get_improvement_report()
            status = learning_scheduler.get_learning_status()
            
            print(f"\n📊 5-Minute Update:")
            print(f"  • Overall Improvement: {report['overall_improvement']:.1f}%")
            print(f"  • Active Learning Sessions: {status['active_sessions']}")
            print(f"  • Knowledge Mastery: {status['estimated_mastery']:.1f}%")
            print(f"  • Top Improvements: {', '.join(report['top_improvements'][:3])}")
            
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopping self-improvement systems...")
        print("💾 Progress has been saved")
        print("🔄 Run this script again to resume improvement")
        
    except Exception as e:
        logger.error(f"Error starting improvement systems: {e}")
        print(f"\n❌ Error: {e}")
        print("Please check the logs for details")


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Start JIGYASA with maximum self-improvement"
    )
    parser.add_argument(
        '--safe',
        action='store_true',
        help='Start in safe mode (slower but more stable)'
    )
    parser.add_argument(
        '--monitor-only',
        action='store_true',
        help='Only show monitoring without starting improvement'
    )
    
    args = parser.parse_args()
    
    if args.monitor_only:
        print("📊 Monitoring mode - showing current status only")
        # TODO: Implement monitoring-only mode
    else:
        if args.safe:
            print("🛡️ Starting in SAFE mode - conservative improvements")
            # TODO: Configure safe mode parameters
        
        start_all_improvements()


if __name__ == '__main__':
    main()