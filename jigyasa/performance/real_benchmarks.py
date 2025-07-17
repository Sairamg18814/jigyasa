"""
Real Performance Measurement and Benchmarking System
Actually measures code performance improvements
"""

import time
import psutil
import subprocess
import tempfile
import statistics
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable
import ast
import cProfile
import pstats
import io
import logging
import json

class RealPerformanceBenchmark:
    """Actually measures performance improvements"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.benchmark_results = []
        self.results_dir = Path(".jigyasa_benchmarks")
        self.results_dir.mkdir(exist_ok=True)
        
    def benchmark_function(self, func: Callable, *args, **kwargs) -> Dict[str, Any]:
        """Comprehensive function benchmarking"""
        # CPU usage before
        cpu_before = psutil.cpu_percent(interval=0.1)
        
        # Memory usage before
        process = psutil.Process()
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        # Profile the function
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Time execution (multiple runs for accuracy)
        execution_times = []
        results = []
        
        for _ in range(5):  # 5 runs
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            execution_times.append(end - start)
            results.append(result)
            
        profiler.disable()
        
        # CPU usage after
        cpu_after = psutil.cpu_percent(interval=0.1)
        
        # Memory usage after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        
        # Analyze profile
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(10)  # Top 10 functions
        profile_output = s.getvalue()
        
        # Calculate statistics
        avg_time = statistics.mean(execution_times)
        std_time = statistics.stdev(execution_times) if len(execution_times) > 1 else 0
        
        return {
            "execution_time": {
                "average": avg_time,
                "min": min(execution_times),
                "max": max(execution_times),
                "std_dev": std_time,
                "all_runs": execution_times
            },
            "cpu_usage": {
                "before": cpu_before,
                "after": cpu_after,
                "delta": cpu_after - cpu_before
            },
            "memory_usage": {
                "before_mb": mem_before,
                "after_mb": mem_after,
                "delta_mb": mem_after - mem_before
            },
            "profile_top_10": profile_output,
            "result_sample": results[0] if results else None
        }
        
    def benchmark_code_comparison(self, original_code: str, improved_code: str, 
                                test_inputs: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Compare performance between original and improved code"""
        
        # Create temporary modules
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(original_code)
            orig_file = f.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(improved_code)
            imp_file = f.name
            
        try:
            # Parse to find main function
            orig_tree = ast.parse(original_code)
            imp_tree = ast.parse(improved_code)
            
            # Find functions to benchmark
            orig_funcs = [node.name for node in ast.walk(orig_tree) 
                         if isinstance(node, ast.FunctionDef)]
            imp_funcs = [node.name for node in ast.walk(imp_tree) 
                        if isinstance(node, ast.FunctionDef)]
            
            common_funcs = set(orig_funcs) & set(imp_funcs)
            
            if not common_funcs:
                # Benchmark entire scripts
                return self._benchmark_scripts(orig_file, imp_file, test_inputs)
                
            # Benchmark each function
            results = {}
            for func_name in common_funcs:
                orig_result = self._benchmark_function_in_file(orig_file, func_name, test_inputs)
                imp_result = self._benchmark_function_in_file(imp_file, func_name, test_inputs)
                
                # Calculate improvement
                orig_time = orig_result['execution_time']['average']
                imp_time = imp_result['execution_time']['average']
                
                improvement = (orig_time - imp_time) / orig_time if orig_time > 0 else 0
                
                results[func_name] = {
                    "original": orig_result,
                    "improved": imp_result,
                    "improvement_percentage": improvement * 100,
                    "speedup_factor": orig_time / imp_time if imp_time > 0 else 1
                }
                
            # Overall improvement
            overall_improvement = statistics.mean([r["improvement_percentage"] 
                                                 for r in results.values()])
            
            return {
                "function_results": results,
                "overall_improvement": overall_improvement,
                "status": "success"
            }
            
        except Exception as e:
            self.logger.error(f"Benchmark comparison failed: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
        finally:
            # Cleanup
            Path(orig_file).unlink()
            Path(imp_file).unlink()
            
    def _benchmark_scripts(self, orig_file: str, imp_file: str, 
                          test_inputs: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Benchmark entire scripts"""
        # Prepare test command
        test_args = json.dumps(test_inputs) if test_inputs else ""
        
        # Benchmark original
        orig_times = []
        for _ in range(5):
            start = time.perf_counter()
            result = subprocess.run(
                ['python', orig_file, test_args],
                capture_output=True,
                text=True
            )
            end = time.perf_counter()
            if result.returncode == 0:
                orig_times.append(end - start)
                
        # Benchmark improved
        imp_times = []
        for _ in range(5):
            start = time.perf_counter()
            result = subprocess.run(
                ['python', imp_file, test_args],
                capture_output=True,
                text=True
            )
            end = time.perf_counter()
            if result.returncode == 0:
                imp_times.append(end - start)
                
        if not orig_times or not imp_times:
            return {"status": "error", "message": "Script execution failed"}
            
        # Calculate improvement
        avg_orig = statistics.mean(orig_times)
        avg_imp = statistics.mean(imp_times)
        improvement = (avg_orig - avg_imp) / avg_orig if avg_orig > 0 else 0
        
        return {
            "script_results": {
                "original_avg_time": avg_orig,
                "improved_avg_time": avg_imp,
                "improvement_percentage": improvement * 100,
                "speedup_factor": avg_orig / avg_imp if avg_imp > 0 else 1
            },
            "status": "success"
        }
        
    def _benchmark_function_in_file(self, file_path: str, func_name: str,
                                   test_inputs: Optional[List[Any]] = None) -> Dict[str, Any]:
        """Benchmark specific function in a file"""
        # Import the module
        import importlib.util
        spec = importlib.util.spec_from_file_location("temp_module", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Get the function
        func = getattr(module, func_name)
        
        # Use default test inputs if none provided
        if test_inputs is None:
            # Try to infer from function signature
            import inspect
            sig = inspect.signature(func)
            param_count = len(sig.parameters)
            
            # Generate appropriate test inputs
            if param_count == 0:
                test_inputs = []
            elif param_count == 1:
                test_inputs = [[1, 2, 3, 4, 5]]  # List for single param
            else:
                test_inputs = [[i for i in range(param_count)]]
                
        # Benchmark the function
        return self.benchmark_function(func, *test_inputs[0] if test_inputs else [])
        
    def generate_performance_report(self, results: Dict[str, Any]) -> str:
        """Generate detailed performance report"""
        report = ["# Performance Benchmark Report", ""]
        
        if "function_results" in results:
            report.append("## Function-by-Function Analysis")
            
            for func_name, func_results in results["function_results"].items():
                report.append(f"\n### Function: `{func_name}`")
                
                orig = func_results["original"]["execution_time"]
                imp = func_results["improved"]["execution_time"]
                
                report.append(f"- Original: {orig['average']*1000:.2f}ms (±{orig['std_dev']*1000:.2f}ms)")
                report.append(f"- Improved: {imp['average']*1000:.2f}ms (±{imp['std_dev']*1000:.2f}ms)")
                report.append(f"- **Improvement: {func_results['improvement_percentage']:.1f}%**")
                report.append(f"- Speedup: {func_results['speedup_factor']:.2f}x faster")
                
                # Memory comparison
                orig_mem = func_results["original"]["memory_usage"]["delta_mb"]
                imp_mem = func_results["improved"]["memory_usage"]["delta_mb"]
                report.append(f"- Memory: {orig_mem:.1f}MB → {imp_mem:.1f}MB")
                
        report.append(f"\n## Overall Performance Improvement: {results.get('overall_improvement', 0):.1f}%")
        
        # Save report
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"benchmark_{timestamp}.md"
        report_path.write_text("\n".join(report))
        
        return "\n".join(report)
        
    def continuous_benchmarking(self, code_dir: str, interval: int = 3600):
        """Continuously benchmark code performance"""
        self.logger.info(f"Starting continuous benchmarking for {code_dir}")
        
        baseline_results = {}
        
        while True:
            py_files = list(Path(code_dir).rglob("*.py"))
            
            for py_file in py_files:
                if "test" not in str(py_file).lower():
                    try:
                        with open(py_file, 'r') as f:
                            current_code = f.read()
                            
                        # Get baseline if not exists
                        if str(py_file) not in baseline_results:
                            self.logger.info(f"Establishing baseline for {py_file}")
                            baseline_results[str(py_file)] = self._benchmark_file(current_code)
                        else:
                            # Compare with baseline
                            current_results = self._benchmark_file(current_code)
                            self._compare_with_baseline(py_file, baseline_results[str(py_file)], 
                                                      current_results)
                                                      
                    except Exception as e:
                        self.logger.error(f"Failed to benchmark {py_file}: {e}")
                        
            time.sleep(interval)
            
    def _benchmark_file(self, code: str) -> Dict[str, Any]:
        """Benchmark entire file"""
        # Simple execution benchmark
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name
            
        times = []
        for _ in range(3):
            start = time.perf_counter()
            subprocess.run(['python', temp_file], capture_output=True)
            end = time.perf_counter()
            times.append(end - start)
            
        Path(temp_file).unlink()
        
        return {
            "avg_time": statistics.mean(times),
            "timestamp": time.time()
        }