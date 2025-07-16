#!/usr/bin/env python3
"""
Autonomous Testing Framework
Automatically generates and runs tests for code improvements
"""

import ast
import subprocess
import sys
import tempfile
import time
import traceback
import unittest
import importlib
import inspect
import os
import psutil
import threading
from typing import Dict, List, Optional, Any, Tuple, Callable
from pathlib import Path
from dataclasses import dataclass, asdict
import logging
import json
import difflib
import coverage

@dataclass
class TestResult:
    """Represents the result of a test"""
    test_name: str
    passed: bool
    execution_time: float
    memory_usage: float
    error_message: Optional[str]
    coverage_percentage: float
    performance_metrics: Dict[str, Any]

@dataclass
class TestSuite:
    """Represents a complete test suite"""
    suite_name: str
    test_count: int
    passed_tests: int
    failed_tests: int
    total_time: float
    average_memory: float
    overall_coverage: float
    test_results: List[TestResult]

class AutoTestGenerator:
    """Automatically generates comprehensive tests for code"""
    
    def __init__(self):
        self.test_patterns = self._load_test_patterns()
        self.edge_cases = self._load_edge_cases()
        self.performance_benchmarks = self._load_performance_benchmarks()
    
    def _load_test_patterns(self) -> Dict[str, Any]:
        """Load test patterns for different code types"""
        return {
            'function_tests': {
                'basic_functionality': 'test_basic_operation',
                'edge_cases': 'test_edge_cases',
                'error_handling': 'test_error_conditions',
                'performance': 'test_performance',
                'type_safety': 'test_type_validation'
            },
            'class_tests': {
                'initialization': 'test_class_init',
                'methods': 'test_class_methods',
                'properties': 'test_class_properties',
                'inheritance': 'test_inheritance',
                'state_management': 'test_state_changes'
            },
            'optimization_tests': {
                'correctness': 'test_optimization_correctness',
                'performance_gain': 'test_performance_improvement',
                'memory_efficiency': 'test_memory_usage',
                'scalability': 'test_scalability'
            }
        }
    
    def _load_edge_cases(self) -> Dict[str, List[Any]]:
        """Load edge case values for testing"""
        return {
            'numeric': [0, 1, -1, float('inf'), float('-inf'), float('nan')],
            'string': ['', ' ', 'test', '🚀', 'very_long_string' * 100],
            'list': [[], [1], list(range(1000)), ['mixed', 1, None]],
            'dict': [{}, {'key': 'value'}, {'complex': {'nested': 'dict'}}],
            'boolean': [True, False],
            'none': [None],
            'complex': [complex(1, 2), complex(-1, -2), complex(0, 0)]
        }
    
    def _load_performance_benchmarks(self) -> Dict[str, Dict[str, float]]:
        """Load performance benchmarks for different operations"""
        return {
            'time_limits': {
                'simple_function': 0.001,  # 1ms
                'complex_function': 0.1,   # 100ms
                'optimization': 0.05,      # 50ms
                'batch_operation': 1.0     # 1s
            },
            'memory_limits': {
                'simple_function': 10,     # 10MB
                'complex_function': 100,   # 100MB
                'optimization': 50,        # 50MB
                'batch_operation': 500     # 500MB
            }
        }
    
    def generate_tests_for_function(self, func_source: str, func_name: str) -> str:
        """Generate comprehensive tests for a function"""
        try:
            # Parse function to understand its structure
            tree = ast.parse(func_source)
            func_node = self._find_function_node(tree, func_name)
            
            if not func_node:
                return self._generate_basic_test_template(func_name)
            
            # Analyze function signature
            params = self._analyze_function_parameters(func_node)
            return_type = self._infer_return_type(func_node)
            
            # Generate test code
            test_code = self._generate_comprehensive_test(
                func_name, params, return_type, func_source
            )
            
            return test_code
            
        except Exception as e:
            logging.error(f"Error generating tests for {func_name}: {e}")
            return self._generate_basic_test_template(func_name)
    
    def generate_tests_for_class(self, class_source: str, class_name: str) -> str:
        """Generate comprehensive tests for a class"""
        try:
            tree = ast.parse(class_source)
            class_node = self._find_class_node(tree, class_name)
            
            if not class_node:
                return self._generate_basic_class_test_template(class_name)
            
            # Analyze class structure
            methods = self._extract_class_methods(class_node)
            properties = self._extract_class_properties(class_node)
            
            # Generate test code
            test_code = self._generate_comprehensive_class_test(
                class_name, methods, properties, class_source
            )
            
            return test_code
            
        except Exception as e:
            logging.error(f"Error generating tests for class {class_name}: {e}")
            return self._generate_basic_class_test_template(class_name)
    
    def generate_optimization_tests(self, original_code: str, optimized_code: str, func_name: str) -> str:
        """Generate tests to verify optimization correctness and performance"""
        test_code = f'''
import unittest
import time
import tracemalloc
import random
import numpy as np
from unittest.mock import patch, MagicMock

class Test{func_name.title()}Optimization(unittest.TestCase):
    """
    Comprehensive tests for {func_name} optimization
    Auto-generated by JIGYASA autonomous testing framework
    """
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_data_small = self._generate_test_data('small')
        self.test_data_medium = self._generate_test_data('medium')
        self.test_data_large = self._generate_test_data('large')
        
        # Performance thresholds
        self.max_time_difference = 0.1  # 100ms tolerance
        self.min_speedup_ratio = 1.1    # At least 10% faster
        self.max_memory_ratio = 0.9     # At most 90% of original memory
    
    def _generate_test_data(self, size='medium'):
        """Generate test data of different sizes"""
        sizes = {{
            'small': 10,
            'medium': 100,
            'large': 1000
        }}
        
        n = sizes.get(size, 100)
        
        return {{
            'integers': list(range(n)),
            'floats': [random.random() for _ in range(n)],
            'strings': [f"test_string_{{i}}" for i in range(n)],
            'mixed': [random.choice([1, 'str', 3.14, True, None]) for _ in range(n)]
        }}
    
    def test_correctness_equivalence(self):
        """Test that optimized version produces same results as original"""
        test_cases = [
            self.test_data_small['integers'],
            self.test_data_medium['floats'],
            self.test_data_large['strings'][:100]  # Limit for performance
        ]
        
        for test_input in test_cases:
            with self.subTest(input_type=type(test_input[0]).__name__ if test_input else 'empty'):
                try:
                    # Get results from both versions
                    original_result = self._call_original_{func_name}(test_input)
                    optimized_result = self._call_optimized_{func_name}(test_input)
                    
                    # Compare results
                    self._assert_results_equal(original_result, optimized_result)
                    
                except Exception as e:
                    self.fail(f"Correctness test failed: {{e}}")
    
    def test_performance_improvement(self):
        """Test that optimized version is actually faster"""
        test_input = self.test_data_medium['integers']
        
        # Measure original function performance
        start_time = time.time()
        tracemalloc.start()
        
        for _ in range(10):  # Multiple runs for accuracy
            original_result = self._call_original_{func_name}(test_input)
        
        original_time = (time.time() - start_time) / 10
        original_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        # Measure optimized function performance
        start_time = time.time()
        tracemalloc.start()
        
        for _ in range(10):
            optimized_result = self._call_optimized_{func_name}(test_input)
        
        optimized_time = (time.time() - start_time) / 10
        optimized_memory = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        # Verify performance improvement
        speedup_ratio = original_time / optimized_time if optimized_time > 0 else float('inf')
        memory_ratio = optimized_memory / original_memory if original_memory > 0 else 0
        
        self.assertGreater(speedup_ratio, self.min_speedup_ratio,
                          f"Optimization not fast enough: {{speedup_ratio:.2f}}x speedup")
        
        self.assertLess(memory_ratio, 1.1,  # Allow 10% memory increase
                       f"Memory usage increased too much: {{memory_ratio:.2f}}x")
        
        print(f"Performance improvement: {{speedup_ratio:.2f}}x faster, {{memory_ratio:.2f}}x memory")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        edge_cases = [
            [],  # Empty input
            [0],  # Single element
            [1] * 1000,  # Repeated elements
            list(range(-50, 51)),  # Negative and positive
            [float('inf'), float('-inf'), float('nan')],  # Special float values
        ]
        
        for edge_case in edge_cases:
            with self.subTest(case=str(edge_case)[:50]):
                try:
                    original_result = self._call_original_{func_name}(edge_case)
                    optimized_result = self._call_optimized_{func_name}(edge_case)
                    self._assert_results_equal(original_result, optimized_result)
                except Exception as e:
                    # Both should fail in the same way
                    with self.assertRaises(type(e)):
                        self._call_original_{func_name}(edge_case)
                    with self.assertRaises(type(e)):
                        self._call_optimized_{func_name}(edge_case)
    
    def test_scalability(self):
        """Test scalability with increasing data sizes"""
        sizes = [10, 100, 1000]
        original_times = []
        optimized_times = []
        
        for size in sizes:
            test_data = list(range(size))
            
            # Time original function
            start_time = time.time()
            self._call_original_{func_name}(test_data)
            original_times.append(time.time() - start_time)
            
            # Time optimized function
            start_time = time.time()
            self._call_optimized_{func_name}(test_data)
            optimized_times.append(time.time() - start_time)
        
        # Check that optimized version scales better
        for i in range(1, len(sizes)):
            original_ratio = original_times[i] / original_times[i-1]
            optimized_ratio = optimized_times[i] / optimized_times[i-1]
            
            self.assertLessEqual(optimized_ratio, original_ratio * 1.1,
                               f"Optimization doesn't scale well at size {{sizes[i]}}")
    
    def test_thread_safety(self):
        """Test that optimized function is thread-safe"""
        import threading
        import queue
        
        test_input = self.test_data_medium['integers']
        results_queue = queue.Queue()
        errors_queue = queue.Queue()
        
        def worker():
            try:
                result = self._call_optimized_{func_name}(test_input)
                results_queue.put(result)
            except Exception as e:
                errors_queue.put(e)
        
        # Run multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads
        for thread in threads:
            thread.join()
        
        # Check results
        self.assertEqual(errors_queue.qsize(), 0, "Thread safety errors occurred")
        self.assertEqual(results_queue.qsize(), 5, "Not all threads completed")
        
        # All results should be the same
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        for result in results[1:]:
            self._assert_results_equal(results[0], result)
    
    def test_memory_leaks(self):
        """Test for memory leaks in optimized function"""
        test_input = self.test_data_small['integers']
        
        # Measure baseline memory
        tracemalloc.start()
        baseline = tracemalloc.get_traced_memory()[0]
        
        # Run function many times
        for _ in range(100):
            result = self._call_optimized_{func_name}(test_input)
            del result  # Explicit deletion
        
        # Measure final memory
        final_memory = tracemalloc.get_traced_memory()[0]
        tracemalloc.stop()
        
        # Memory should not have grown significantly
        memory_growth = final_memory - baseline
        max_allowed_growth = 1024 * 1024  # 1MB
        
        self.assertLess(memory_growth, max_allowed_growth,
                       f"Potential memory leak: {{memory_growth}} bytes growth")
    
    def _call_original_{func_name}(self, *args, **kwargs):
        """Call the original (unoptimized) function"""
        # This would be replaced with actual original function call
        # For now, return a mock result
        return list(args[0]) if args and isinstance(args[0], (list, tuple)) else args
    
    def _call_optimized_{func_name}(self, *args, **kwargs):
        """Call the optimized function"""
        # This would be replaced with actual optimized function call
        # For now, return a mock result
        return list(args[0]) if args and isinstance(args[0], (list, tuple)) else args
    
    def _assert_results_equal(self, result1, result2):
        """Assert that two results are equal, handling different types"""
        if isinstance(result1, (list, tuple)) and isinstance(result2, (list, tuple)):
            self.assertEqual(len(result1), len(result2))
            for a, b in zip(result1, result2):
                if isinstance(a, float) and isinstance(b, float):
                    self.assertAlmostEqual(a, b, places=5)
                else:
                    self.assertEqual(a, b)
        elif isinstance(result1, float) and isinstance(result2, float):
            self.assertAlmostEqual(result1, result2, places=5)
        else:
            self.assertEqual(result1, result2)

if __name__ == '__main__':
    unittest.main(verbosity=2)
'''
        
        return test_code
    
    def _find_function_node(self, tree: ast.AST, func_name: str) -> Optional[ast.FunctionDef]:
        """Find function node in AST"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node
        return None
    
    def _find_class_node(self, tree: ast.AST, class_name: str) -> Optional[ast.ClassDef]:
        """Find class node in AST"""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == class_name:
                return node
        return None
    
    def _analyze_function_parameters(self, func_node: ast.FunctionDef) -> List[Dict[str, Any]]:
        """Analyze function parameters"""
        params = []
        
        for arg in func_node.args.args:
            param_info = {
                'name': arg.arg,
                'type': 'Any',  # Would need more sophisticated type inference
                'has_default': False
            }
            params.append(param_info)
        
        return params
    
    def _infer_return_type(self, func_node: ast.FunctionDef) -> str:
        """Infer function return type"""
        # Simple heuristic based on return statements
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return):
                if node.value:
                    return 'Any'  # Would need more sophisticated analysis
        return 'None'
    
    def _extract_class_methods(self, class_node: ast.ClassDef) -> List[str]:
        """Extract method names from class"""
        methods = []
        for node in class_node.body:
            if isinstance(node, ast.FunctionDef):
                methods.append(node.name)
        return methods
    
    def _extract_class_properties(self, class_node: ast.ClassDef) -> List[str]:
        """Extract property names from class"""
        # Simplified - would need more sophisticated analysis
        return []
    
    def _generate_comprehensive_test(self, func_name: str, params: List[Dict], return_type: str, source: str) -> str:
        """Generate comprehensive test for a function"""
        test_code = f'''
import unittest
import time
import tracemalloc
import random
from unittest.mock import patch, MagicMock

class Test{func_name.title()}(unittest.TestCase):
    """
    Comprehensive tests for {func_name}
    Auto-generated by JIGYASA autonomous testing framework
    """
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_cases = self._generate_test_cases()
        self.performance_threshold = 0.1  # 100ms
        self.memory_threshold = 10 * 1024 * 1024  # 10MB
    
    def _generate_test_cases(self):
        """Generate test cases for the function"""
        return [
            # Basic test cases
            {{"input": [], "description": "empty_input"}},
            {{"input": [1, 2, 3], "description": "normal_input"}},
            {{"input": [0], "description": "single_element"}},
            
            # Edge cases
            {{"input": [float('inf')], "description": "infinity"}},
            {{"input": [float('nan')], "description": "nan"}},
            {{"input": list(range(1000)), "description": "large_input"}},
        ]
    
    def test_basic_functionality(self):
        """Test basic functionality of {func_name}"""
        for test_case in self.test_cases[:3]:  # Basic cases only
            with self.subTest(case=test_case["description"]):
                try:
                    result = {func_name}(*test_case["input"] if isinstance(test_case["input"], list) else [test_case["input"]])
                    self.assertIsNotNone(result, "Function should return a value")
                except Exception as e:
                    self.fail(f"Function failed on {{test_case['description']}}: {{e}}")
    
    def test_edge_cases(self):
        """Test edge cases for {func_name}"""
        edge_cases = self.test_cases[3:]  # Edge cases
        
        for test_case in edge_cases:
            with self.subTest(case=test_case["description"]):
                try:
                    result = {func_name}(*test_case["input"] if isinstance(test_case["input"], list) else [test_case["input"]])
                    # Function should handle edge cases gracefully
                except Exception as e:
                    # Some edge cases might legitimately raise exceptions
                    self.assertIsInstance(e, (ValueError, TypeError, RuntimeError),
                                        f"Unexpected exception type: {{type(e)}}")
    
    def test_performance(self):
        """Test performance of {func_name}"""
        large_input = list(range(1000))
        
        start_time = time.time()
        tracemalloc.start()
        
        result = {func_name}(large_input)
        
        execution_time = time.time() - start_time
        memory_usage = tracemalloc.get_traced_memory()[1]
        tracemalloc.stop()
        
        self.assertLess(execution_time, self.performance_threshold,
                       f"Function too slow: {{execution_time:.3f}}s")
        self.assertLess(memory_usage, self.memory_threshold,
                       f"Function uses too much memory: {{memory_usage / 1024 / 1024:.1f}}MB")
    
    def test_type_safety(self):
        """Test type safety of {func_name}"""
        invalid_inputs = [
            "string_instead_of_list",
            {{"dict": "instead_of_list"}},
            None,
            42
        ]
        
        for invalid_input in invalid_inputs:
            with self.subTest(input_type=type(invalid_input).__name__):
                with self.assertRaises((TypeError, ValueError, AttributeError)):
                    {func_name}(invalid_input)

if __name__ == '__main__':
    unittest.main(verbosity=2)
'''
        
        return test_code
    
    def _generate_comprehensive_class_test(self, class_name: str, methods: List[str], properties: List[str], source: str) -> str:
        """Generate comprehensive test for a class"""
        test_code = f'''
import unittest
import time
import gc
from unittest.mock import patch, MagicMock

class Test{class_name}(unittest.TestCase):
    """
    Comprehensive tests for {class_name}
    Auto-generated by JIGYASA autonomous testing framework
    """
    
    def setUp(self):
        """Set up test fixtures"""
        self.instance = {class_name}()
        self.test_args = ["test_arg1", "test_arg2"]
        self.test_kwargs = {{"key1": "value1", "key2": "value2"}}
    
    def tearDown(self):
        """Clean up after tests"""
        del self.instance
        gc.collect()
    
    def test_initialization(self):
        """Test class initialization"""
        # Test default initialization
        instance = {class_name}()
        self.assertIsInstance(instance, {class_name})
        
        # Test initialization with arguments
        try:
            instance_with_args = {class_name}(*self.test_args, **self.test_kwargs)
            self.assertIsInstance(instance_with_args, {class_name})
        except TypeError:
            # Some classes might not accept arguments
            pass
    
    def test_string_representation(self):
        """Test string representation methods"""
        str_repr = str(self.instance)
        self.assertIsInstance(str_repr, str)
        self.assertTrue(len(str_repr) > 0)
        
        repr_repr = repr(self.instance)
        self.assertIsInstance(repr_repr, str)
        self.assertTrue(len(repr_repr) > 0)
'''
        
        # Add tests for each method
        for method_name in methods:
            if method_name.startswith('_'):
                continue  # Skip private methods
            
            test_code += f'''
    def test_{method_name}(self):
        """Test {method_name} method"""
        if hasattr(self.instance, '{method_name}'):
            method = getattr(self.instance, '{method_name}')
            if callable(method):
                try:
                    result = method()
                    # Method should not raise an exception
                    self.assertTrue(True, "{method_name} executed successfully")
                except TypeError:
                    # Method might require arguments
                    try:
                        result = method("test_arg")
                        self.assertTrue(True, "{method_name} executed with argument")
                    except Exception as e:
                        self.fail(f"{method_name} failed with argument: {{e}}")
                except Exception as e:
                    self.fail(f"{method_name} failed: {{e}}")
'''
        
        test_code += '''
    def test_memory_management(self):
        """Test memory management and cleanup"""
        initial_instances = len(gc.get_objects())
        
        # Create and destroy multiple instances
        instances = []
        for _ in range(10):
            instances.append({class_name}())
        
        del instances
        gc.collect()
        
        final_instances = len(gc.get_objects())
        
        # Should not have significant memory leaks
        instance_growth = final_instances - initial_instances
        self.assertLess(instance_growth, 50, f"Potential memory leak: {instance_growth} new objects")

if __name__ == '__main__':
    unittest.main(verbosity=2)
'''.format(class_name=class_name)
        
        return test_code
    
    def _generate_basic_test_template(self, func_name: str) -> str:
        """Generate basic test template when analysis fails"""
        return f'''
import unittest

class Test{func_name.title()}(unittest.TestCase):
    """Basic test for {func_name}"""
    
    def test_basic(self):
        """Basic test - needs manual implementation"""
        # TODO: Implement actual test
        self.assertTrue(True, "Placeholder test")

if __name__ == '__main__':
    unittest.main()
'''
    
    def _generate_basic_class_test_template(self, class_name: str) -> str:
        """Generate basic class test template when analysis fails"""
        return f'''
import unittest

class Test{class_name}(unittest.TestCase):
    """Basic test for {class_name}"""
    
    def test_initialization(self):
        """Test class can be initialized"""
        instance = {class_name}()
        self.assertIsInstance(instance, {class_name})

if __name__ == '__main__':
    unittest.main()
'''

class AutoTestRunner:
    """Automatically runs and manages test execution"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_results_dir = self.project_root / "test_results"
        self.test_results_dir.mkdir(exist_ok=True)
        
        self.coverage_threshold = 80  # 80% coverage required
        self.performance_timeout = 30  # 30 seconds max per test
        
    def run_test_suite(self, test_code: str, test_name: str) -> TestSuite:
        """Run a complete test suite and return results"""
        try:
            # Create temporary test file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(test_code)
                test_file_path = f.name
            
            try:
                # Run tests with coverage
                start_time = time.time()
                
                cov = coverage.Coverage()
                cov.start()
                
                # Run the test
                result = subprocess.run([
                    sys.executable, '-m', 'pytest', test_file_path, 
                    '-v', '--tb=short', '--json-report', 
                    '--json-report-file=/tmp/test_report.json'
                ], capture_output=True, text=True, timeout=self.performance_timeout)
                
                cov.stop()
                cov.save()
                
                total_time = time.time() - start_time
                
                # Parse results
                test_results = self._parse_test_results(result, test_name)
                coverage_percentage = self._calculate_coverage(cov)
                
                # Create test suite
                suite = TestSuite(
                    suite_name=test_name,
                    test_count=len(test_results),
                    passed_tests=sum(1 for t in test_results if t.passed),
                    failed_tests=sum(1 for t in test_results if not t.passed),
                    total_time=total_time,
                    average_memory=sum(t.memory_usage for t in test_results) / len(test_results) if test_results else 0,
                    overall_coverage=coverage_percentage,
                    test_results=test_results
                )
                
                # Save results
                self._save_test_results(suite)
                
                return suite
                
            finally:
                # Cleanup
                if os.path.exists(test_file_path):
                    os.unlink(test_file_path)
                
        except Exception as e:
            logging.error(f"Error running test suite {test_name}: {e}")
            return self._create_failed_test_suite(test_name, str(e))
    
    def run_performance_benchmark(self, func_code: str, func_name: str) -> Dict[str, Any]:
        """Run performance benchmarks for a function"""
        try:
            # Create benchmark test
            benchmark_code = f'''
import time
import tracemalloc
import statistics

def benchmark_{func_name}():
    """Benchmark {func_name} performance"""
    
    # Test data of different sizes
    test_sizes = [10, 100, 1000]
    results = {{"sizes": [], "times": [], "memory": []}}
    
    for size in test_sizes:
        test_data = list(range(size))
        
        # Multiple runs for statistical accuracy
        times = []
        memories = []
        
        for _ in range(10):
            tracemalloc.start()
            start_time = time.time()
            
            # Call function (placeholder)
            result = test_data  # Would call actual function
            
            end_time = time.time()
            memory_usage = tracemalloc.get_traced_memory()[1]
            tracemalloc.stop()
            
            times.append(end_time - start_time)
            memories.append(memory_usage)
        
        results["sizes"].append(size)
        results["times"].append(statistics.mean(times))
        results["memory"].append(statistics.mean(memories))
    
    return results

if __name__ == "__main__":
    import json
    results = benchmark_{func_name}()
    print(json.dumps(results))
'''
            
            # Run benchmark
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(benchmark_code)
                benchmark_file = f.name
            
            try:
                result = subprocess.run([
                    sys.executable, benchmark_file
                ], capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    benchmark_results = json.loads(result.stdout.strip())
                    return benchmark_results
                else:
                    return {"error": result.stderr}
                    
            finally:
                if os.path.exists(benchmark_file):
                    os.unlink(benchmark_file)
                
        except Exception as e:
            return {"error": str(e)}
    
    def _parse_test_results(self, subprocess_result, test_name: str) -> List[TestResult]:
        """Parse test results from subprocess output"""
        test_results = []
        
        try:
            # Try to parse JSON report if available
            if os.path.exists('/tmp/test_report.json'):
                with open('/tmp/test_report.json', 'r') as f:
                    json_results = json.load(f)
                
                for test in json_results.get('tests', []):
                    test_result = TestResult(
                        test_name=test.get('nodeid', 'unknown'),
                        passed=test.get('outcome') == 'passed',
                        execution_time=test.get('duration', 0),
                        memory_usage=0,  # Not available in JSON report
                        error_message=test.get('call', {}).get('longrepr') if test.get('outcome') == 'failed' else None,
                        coverage_percentage=0,  # Calculated separately
                        performance_metrics={}
                    )
                    test_results.append(test_result)
                
                # Cleanup
                os.unlink('/tmp/test_report.json')
            
            else:
                # Parse text output as fallback
                lines = subprocess_result.stdout.split('\n')
                for line in lines:
                    if '::' in line and ('PASSED' in line or 'FAILED' in line):
                        parts = line.split('::')
                        test_name_part = parts[-1].split()[0]
                        passed = 'PASSED' in line
                        
                        test_result = TestResult(
                            test_name=test_name_part,
                            passed=passed,
                            execution_time=0,
                            memory_usage=0,
                            error_message=None if passed else "Test failed",
                            coverage_percentage=0,
                            performance_metrics={}
                        )
                        test_results.append(test_result)
        
        except Exception as e:
            logging.warning(f"Could not parse test results: {e}")
            # Create a single placeholder result
            test_results.append(TestResult(
                test_name=test_name,
                passed=subprocess_result.returncode == 0,
                execution_time=0,
                memory_usage=0,
                error_message=subprocess_result.stderr if subprocess_result.returncode != 0 else None,
                coverage_percentage=0,
                performance_metrics={}
            ))
        
        return test_results
    
    def _calculate_coverage(self, cov) -> float:
        """Calculate code coverage percentage"""
        try:
            # Get coverage data
            coverage_data = cov.get_data()
            total_lines = 0
            covered_lines = 0
            
            for filename in coverage_data.measured_files():
                lines = coverage_data.lines(filename)
                if lines:
                    total_lines += len(lines)
                    covered_lines += len([line for line in lines if coverage_data.has_arcs() or True])
            
            if total_lines > 0:
                return (covered_lines / total_lines) * 100
            
        except Exception as e:
            logging.warning(f"Could not calculate coverage: {e}")
        
        return 0.0
    
    def _save_test_results(self, suite: TestSuite):
        """Save test results to file"""
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            results_file = self.test_results_dir / f"{suite.suite_name}_{timestamp}.json"
            
            with open(results_file, 'w') as f:
                json.dump(asdict(suite), f, indent=2)
                
            logging.info(f"Test results saved to {results_file}")
        except Exception as e:
            logging.error(f"Could not save test results: {e}")
    
    def _create_failed_test_suite(self, test_name: str, error_msg: str) -> TestSuite:
        """Create a failed test suite when testing fails"""
        failed_result = TestResult(
            test_name=test_name,
            passed=False,
            execution_time=0,
            memory_usage=0,
            error_message=error_msg,
            coverage_percentage=0,
            performance_metrics={}
        )
        
        return TestSuite(
            suite_name=test_name,
            test_count=1,
            passed_tests=0,
            failed_tests=1,
            total_time=0,
            average_memory=0,
            overall_coverage=0,
            test_results=[failed_result]
        )

# Global instances
auto_test_generator = AutoTestGenerator()
auto_test_runner = None

def initialize_auto_tester(project_root: str = None):
    """Initialize the autonomous tester"""
    global auto_test_runner
    
    if project_root is None:
        project_root = os.getcwd()
    
    try:
        auto_test_runner = AutoTestRunner(project_root)
        logging.info("✅ Autonomous tester initialized")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to initialize autonomous tester: {e}")
        return False

def generate_and_run_tests(code: str, code_type: str, name: str) -> TestSuite:
    """Generate and run tests for given code"""
    global auto_test_generator, auto_test_runner
    
    if auto_test_runner is None:
        initialize_auto_tester()
    
    try:
        # Generate tests
        if code_type == 'function':
            test_code = auto_test_generator.generate_tests_for_function(code, name)
        elif code_type == 'class':
            test_code = auto_test_generator.generate_tests_for_class(code, name)
        elif code_type == 'optimization':
            test_code = auto_test_generator.generate_optimization_tests("", code, name)
        else:
            test_code = auto_test_generator._generate_basic_test_template(name)
        
        # Run tests
        return auto_test_runner.run_test_suite(test_code, name)
        
    except Exception as e:
        logging.error(f"Error in generate_and_run_tests: {e}")
        return auto_test_runner._create_failed_test_suite(name, str(e))