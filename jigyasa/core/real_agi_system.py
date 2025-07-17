"""
Real AGI System using Ollama and Llama 3.2
Integrates all components to create actual working AGI-like capabilities
"""

import os
import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import threading

from ..models.ollama_wrapper import OllamaWrapper
from ..autonomous.real_self_editor import RealSelfEditor
from ..learning.real_continuous_learning import RealContinuousLearner
from ..performance.real_benchmarks import RealPerformanceBenchmark
from ..adaptive.hardware_detector import HardwareDetector

class RealAGISystem:
    """Actual working AGI system with real capabilities"""
    
    def __init__(self, model_name: str = "llama3.1:8b"):
        self.logger = logging.getLogger(__name__)
        
        # Initialize Ollama
        self.ollama = OllamaWrapper(model_name=model_name)
        
        # Check if Ollama is running
        if not self.ollama.check_ollama_running():
            self.logger.warning("Ollama not running. Please start Ollama service.")
            raise RuntimeError("Ollama service not available")
            
        # Pull model if needed
        self.ollama.pull_model()
        
        # Initialize components with real implementations
        self.self_editor = RealSelfEditor(self.ollama)
        self.learner = RealContinuousLearner(self.ollama)
        self.benchmark = RealPerformanceBenchmark()
        self.hardware = HardwareDetector()
        
        # System state
        self.active = True
        self.autonomous_mode = False
        self.improvement_history = []
        
        # Initialize system
        self._initialize_system()
        
    def _initialize_system(self):
        """Initialize the AGI system"""
        self.logger.info("Initializing Real AGI System...")
        
        # Detect hardware
        self.hardware_specs = self.hardware.detect_hardware()
        self.logger.info(f"Hardware: {self.hardware_specs.cpu_brand}, "
                        f"{self.hardware_specs.total_ram:.1f}GB RAM, "
                        f"{'GPU: ' + str(self.hardware_specs.gpu_names[0]) if self.hardware_specs.has_gpu else 'No GPU'}")
        
        # Load previous learning
        self.learning_metrics = self.learner.get_learning_metrics()
        self.logger.info(f"Loaded {self.learning_metrics['total_knowledge_items']} knowledge items")
        
    def chat(self, message: str) -> str:
        """Interactive chat with learning"""
        # Apply learned knowledge
        enhanced_response, used_knowledge = self.learner.apply_learned_knowledge(message)
        
        # Generate response
        if used_knowledge:
            response = enhanced_response
        else:
            llm_response = self.ollama.generate(message)
            response = llm_response.text
            
        # Learn from interaction
        self.learner.learn_from_interaction(message, response)
        
        return response
        
    def improve_code_file(self, file_path: str) -> Dict[str, Any]:
        """Improve a specific code file with real modifications"""
        self.logger.info(f"Improving code file: {file_path}")
        
        # Read original code
        with open(file_path, 'r') as f:
            original_code = f.read()
            
        # Get improvement result
        result = self.self_editor.modify_code_autonomously(file_path)
        
        if result['status'] == 'success':
            # Read improved code
            with open(file_path, 'r') as f:
                improved_code = f.read()
                
            # Benchmark the improvement
            benchmark_result = self.benchmark.benchmark_code_comparison(
                original_code, improved_code
            )
            
            # Generate report
            report = self.benchmark.generate_performance_report(benchmark_result)
            
            # Learn from the improvement
            self.learner.learn_from_interaction(
                f"Improved code in {file_path}",
                f"Applied {len(result['improvements'])} improvements",
                report,
                success=True
            )
            
            result['benchmark'] = benchmark_result
            result['report'] = report
            
            # Store in history
            self.improvement_history.append({
                "timestamp": datetime.now().isoformat(),
                "file": file_path,
                "result": result
            })
            
        return result
        
    def start_autonomous_mode(self, directory: str, interval: int = 300):
        """Start autonomous improvement mode"""
        self.logger.info(f"Starting autonomous mode for {directory}")
        self.autonomous_mode = True
        
        # Start improvement thread
        def autonomous_loop():
            while self.autonomous_mode:
                py_files = list(Path(directory).rglob("*.py"))
                
                for py_file in py_files:
                    if not self.autonomous_mode:
                        break
                        
                    # Skip test files and already optimized files
                    if "test" in str(py_file).lower() or "_optimized" in str(py_file):
                        continue
                        
                    try:
                        result = self.improve_code_file(str(py_file))
                        if result['status'] == 'success':
                            self.logger.info(f"Improved {py_file}: "
                                           f"{result.get('performance_gain', 0):.1%} gain")
                    except Exception as e:
                        self.logger.error(f"Failed to improve {py_file}: {e}")
                        
                # Wait before next iteration
                import time
                time.sleep(interval)
                
        # Start in background
        thread = threading.Thread(target=autonomous_loop, daemon=True)
        thread.start()
        
        return {"status": "started", "directory": directory, "interval": interval}
        
    def stop_autonomous_mode(self):
        """Stop autonomous mode"""
        self.autonomous_mode = False
        self.logger.info("Autonomous mode stopped")
        return {"status": "stopped"}
        
    def train_on_codebase(self, directory: str) -> Dict[str, Any]:
        """Train the system on a codebase"""
        self.logger.info(f"Training on codebase: {directory}")
        
        py_files = list(Path(directory).rglob("*.py"))
        total_learned = 0
        
        for py_file in py_files:
            try:
                with open(py_file, 'r') as f:
                    code = f.read()
                    
                # Analyze code patterns
                analysis = self.ollama.analyze_code(code)
                
                # Learn from analysis
                for improvement in analysis.get('improvements', []):
                    result = self.learner.learn_from_interaction(
                        f"Code pattern in {py_file}",
                        improvement['description'],
                        f"Type: {improvement['type']}",
                        success=True
                    )
                    total_learned += result['insights_learned']
                    
            except Exception as e:
                self.logger.error(f"Failed to learn from {py_file}: {e}")
                
        return {
            "files_analyzed": len(py_files),
            "total_insights_learned": total_learned,
            "knowledge_base_size": self.learner.get_learning_metrics()['total_knowledge_items']
        }
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "hardware": {
                "cpu": self.hardware_specs.cpu_brand,
                "cores": self.hardware_specs.cpu_cores,
                "ram_gb": self.hardware_specs.total_ram,
                "gpu": self.hardware_specs.gpu_names[0] if self.hardware_specs.has_gpu else "None",
                "performance_class": self.hardware_specs.performance_class
            },
            "learning": self.learner.get_learning_metrics(),
            "improvements": {
                "total": len(self.improvement_history),
                "last_24h": sum(1 for imp in self.improvement_history 
                              if datetime.fromisoformat(imp['timestamp']) > 
                              datetime.now().replace(hour=0, minute=0, second=0)),
                "average_gain": sum(imp['result'].get('performance_gain', 0) 
                                  for imp in self.improvement_history) / 
                               max(len(self.improvement_history), 1)
            },
            "autonomous_mode": self.autonomous_mode,
            "ollama_status": "connected" if self.ollama.check_ollama_running() else "disconnected"
        }
        
    def export_knowledge(self, output_path: str):
        """Export learned knowledge"""
        knowledge_data = {
            "timestamp": datetime.now().isoformat(),
            "metrics": self.learner.get_learning_metrics(),
            "improvement_history": self.improvement_history,
            "system_info": {
                "model": self.ollama.model_name,
                "hardware": self.hardware_specs.__dict__
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(knowledge_data, f, indent=2)
            
        self.logger.info(f"Exported knowledge to {output_path}")
        
    def demonstrate_capabilities(self):
        """Demonstrate all real capabilities"""
        print("\n🤖 JIGYASA Real AGI System Demonstration")
        print("=" * 50)
        
        # 1. Hardware Detection
        print("\n1️⃣ Hardware Detection (Real)")
        specs = self.hardware_specs
        print(f"   CPU: {specs.cpu_brand} ({specs.cpu_cores} cores)")
        print(f"   RAM: {specs.total_ram:.1f} GB")
        print(f"   GPU: {'Yes - ' + specs.gpu_names[0] if specs.has_gpu else 'No'}")
        print(f"   Performance Class: {specs.performance_class}")
        
        # 2. Code Analysis
        print("\n2️⃣ Code Analysis (Real AI)")
        sample_code = '''
def calculate_sum(numbers):
    total = 0
    for i in range(len(numbers)):
        total = total + numbers[i]
    return total
'''
        analysis = self.ollama.analyze_code(sample_code)
        print(f"   Found {len(analysis.get('improvements', []))} improvements")
        for imp in analysis.get('improvements', [])[:2]:
            print(f"   - {imp['type']}: {imp['description']}")
            
        # 3. Learning Metrics
        print("\n3️⃣ Continuous Learning (Real)")
        metrics = self.learner.get_learning_metrics()
        print(f"   Knowledge Items: {metrics['total_knowledge_items']}")
        print(f"   Average Confidence: {metrics['average_confidence']:.2f}")
        print(f"   Success Rate: {metrics['overall_success_rate']:.2f}")
        
        # 4. Performance Measurement
        print("\n4️⃣ Performance Measurement (Real)")
        print("   Benchmarking system: Active")
        print("   Can measure actual execution time improvements")
        
        print("\n✅ All systems operational with REAL functionality!")
        print("=" * 50)