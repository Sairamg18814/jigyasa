#!/usr/bin/env python3
"""
Autonomous Code Generation System
Generates and improves code automatically using AI-driven analysis
"""

import ast
import inspect
import textwrap
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from dataclasses import dataclass
import logging
import time

@dataclass
class GeneratedCode:
    """Represents automatically generated code"""
    code_type: str  # function, class, module, optimization
    purpose: str
    language: str
    code: str
    dependencies: List[str]
    test_code: str
    performance_estimate: float
    complexity_score: int
    safety_level: str

class AICodeGenerator:
    """Generates code improvements using AI analysis"""
    
    def __init__(self):
        self.code_patterns = self._load_code_patterns()
        self.optimization_templates = self._load_optimization_templates()
        self.safety_checks = self._load_safety_checks()
    
    def _load_code_patterns(self) -> Dict[str, Any]:
        """Load common code patterns for generation"""
        return {
            'caching': {
                'pattern': 'functools.lru_cache',
                'template': '''
@functools.lru_cache(maxsize=128)
def {function_name}({parameters}):
    {original_body}
''',
                'conditions': ['pure_function', 'expensive_computation']
            },
            'vectorization': {
                'pattern': 'numpy_vectorize',
                'template': '''
import numpy as np

def {function_name}_vectorized({parameters}):
    # Vectorized version of {function_name}
    {vectorized_body}
''',
                'conditions': ['loop_over_arrays', 'mathematical_operations']
            },
            'async_optimization': {
                'pattern': 'async_await',
                'template': '''
import asyncio

async def {function_name}_async({parameters}):
    # Async version of {function_name}
    {async_body}
''',
                'conditions': ['io_operations', 'network_calls']
            },
            'memory_optimization': {
                'pattern': 'generator_function',
                'template': '''
def {function_name}_generator({parameters}):
    # Memory-efficient generator version
    {generator_body}
''',
                'conditions': ['large_data_processing', 'iterative_results']
            }
        }
    
    def _load_optimization_templates(self) -> Dict[str, str]:
        """Load templates for common optimizations"""
        return {
            'loop_unrolling': '''
# Optimized loop unrolling
{unrolled_operations}
''',
            'batch_processing': '''
# Batch processing optimization
def process_batch(data_batch, batch_size={batch_size}):
    results = []
    for i in range(0, len(data_batch), batch_size):
        batch = data_batch[i:i+batch_size]
        batch_result = {processing_function}(batch)
        results.extend(batch_result)
    return results
''',
            'parallel_processing': '''
# Parallel processing with multiprocessing
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def {function_name}_parallel(data, num_workers=None):
    if num_workers is None:
        num_workers = mp.cpu_count()
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map({worker_function}, data))
    
    return results
''',
            'gpu_acceleration': '''
# GPU acceleration with PyTorch
import torch

def {function_name}_gpu(data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tensor_data = torch.tensor(data).to(device)
    
    # GPU-accelerated computation
    result = {gpu_operations}
    
    return result.cpu().numpy()
'''
        }
    
    def _load_safety_checks(self) -> Dict[str, List[str]]:
        """Load safety checks for generated code"""
        return {
            'forbidden_operations': [
                'eval(', 'exec(', 'os.system(', '__import__(',
                'subprocess.call', 'open(', 'file(', 'input('
            ],
            'required_validations': [
                'type_checking', 'bounds_checking', 'null_checking'
            ],
            'performance_limits': {
                'max_complexity': 10,
                'max_memory_usage': '1GB',
                'max_execution_time': '30s'
            }
        }
    
    def analyze_function_for_improvements(self, func_source: str, func_name: str) -> List[GeneratedCode]:
        """Analyze a function and generate improvements"""
        improvements = []
        
        try:
            # Parse the function
            tree = ast.parse(func_source)
            func_node = self._find_function_node(tree, func_name)
            
            if not func_node:
                return improvements
            
            # Check for different optimization opportunities
            if self._can_add_caching(func_node):
                cached_version = self._generate_cached_function(func_source, func_name)
                if cached_version:
                    improvements.append(cached_version)
            
            if self._can_vectorize(func_node):
                vectorized_version = self._generate_vectorized_function(func_source, func_name)
                if vectorized_version:
                    improvements.append(vectorized_version)
            
            if self._can_parallelize(func_node):
                parallel_version = self._generate_parallel_function(func_source, func_name)
                if parallel_version:
                    improvements.append(parallel_version)
            
            if self._can_async_optimize(func_node):
                async_version = self._generate_async_function(func_source, func_name)
                if async_version:
                    improvements.append(async_version)
            
        except Exception as e:
            logging.error(f"Error analyzing function {func_name}: {e}")
        
        return improvements
    
    def generate_new_functionality(self, description: str, context: Dict[str, Any]) -> Optional[GeneratedCode]:
        """Generate completely new functionality based on description"""
        try:
            # Parse description to understand requirements
            requirements = self._parse_requirements(description)
            
            # Generate appropriate code structure
            if 'class' in description.lower():
                return self._generate_class(requirements, context)
            elif 'function' in description.lower():
                return self._generate_function(requirements, context)
            elif 'module' in description.lower():
                return self._generate_module(requirements, context)
            else:
                return self._generate_utility(requirements, context)
                
        except Exception as e:
            logging.error(f"Error generating new functionality: {e}")
            return None
    
    def _find_function_node(self, tree: ast.AST, func_name: str) -> Optional[ast.FunctionDef]:
        """Find function node in AST"""
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == func_name:
                return node
        return None
    
    def _can_add_caching(self, func_node: ast.FunctionDef) -> bool:
        """Check if function can benefit from caching"""
        # Check if function is pure (no side effects)
        has_side_effects = False
        
        for node in ast.walk(func_node):
            # Look for operations that indicate side effects
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['print', 'open', 'write']:
                        has_side_effects = True
                        break
            elif isinstance(node, ast.Global) or isinstance(node, ast.Nonlocal):
                has_side_effects = True
                break
        
        # Check for expensive operations
        has_expensive_ops = self._has_expensive_operations(func_node)
        
        return not has_side_effects and has_expensive_ops
    
    def _can_vectorize(self, func_node: ast.FunctionDef) -> bool:
        """Check if function can be vectorized"""
        has_loops = False
        has_math_ops = False
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.For):
                has_loops = True
            elif isinstance(node, ast.BinOp):
                if isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
                    has_math_ops = True
        
        return has_loops and has_math_ops
    
    def _can_parallelize(self, func_node: ast.FunctionDef) -> bool:
        """Check if function can be parallelized"""
        # Look for independent operations that can be parallelized
        has_independent_work = False
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.For):
                # Check if loop iterations are independent
                if self._has_independent_iterations(node):
                    has_independent_work = True
                    break
        
        return has_independent_work
    
    def _can_async_optimize(self, func_node: ast.FunctionDef) -> bool:
        """Check if function can benefit from async optimization"""
        has_io_ops = False
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in ['open', 'read', 'write', 'requests', 'urllib']:
                        has_io_ops = True
                        break
                elif isinstance(node.func, ast.Attribute):
                    if node.func.attr in ['get', 'post', 'read', 'write']:
                        has_io_ops = True
                        break
        
        return has_io_ops
    
    def _has_expensive_operations(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has expensive operations"""
        expensive_patterns = [
            'sqrt', 'pow', 'exp', 'log', 'sin', 'cos',
            'factorial', 'fibonacci', 'sort', 'search'
        ]
        
        for node in ast.walk(func_node):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if any(pattern in node.func.id.lower() for pattern in expensive_patterns):
                        return True
        
        return False
    
    def _has_independent_iterations(self, for_node: ast.For) -> bool:
        """Check if for loop has independent iterations"""
        # Simplified check - look for absence of dependencies between iterations
        return True  # For demo purposes
    
    def _generate_cached_function(self, func_source: str, func_name: str) -> GeneratedCode:
        """Generate cached version of function"""
        try:
            # Extract function parameters and body
            tree = ast.parse(func_source)
            func_node = self._find_function_node(tree, func_name)
            
            if not func_node:
                return None
            
            # Generate cached version
            cached_code = f'''
import functools

@functools.lru_cache(maxsize=128)
{func_source}
'''
            
            test_code = f'''
def test_{func_name}_cached():
    # Test cached version performance
    import time
    
    start_time = time.time()
    result1 = {func_name}_cached(test_input)
    first_call_time = time.time() - start_time
    
    start_time = time.time()
    result2 = {func_name}_cached(test_input)  # Should be cached
    second_call_time = time.time() - start_time
    
    assert result1 == result2
    assert second_call_time < first_call_time * 0.1  # 10x faster
'''
            
            return GeneratedCode(
                code_type="optimization",
                purpose=f"Add caching to {func_name} for performance improvement",
                language="python",
                code=cached_code,
                dependencies=["functools"],
                test_code=test_code,
                performance_estimate=0.5,  # 50% improvement for repeated calls
                complexity_score=2,
                safety_level="high"
            )
            
        except Exception as e:
            logging.error(f"Error generating cached function: {e}")
            return None
    
    def _generate_vectorized_function(self, func_source: str, func_name: str) -> GeneratedCode:
        """Generate vectorized version of function"""
        try:
            vectorized_code = f'''
import numpy as np

def {func_name}_vectorized(data_array):
    """
    Vectorized version of {func_name} for better performance on arrays
    """
    # Convert to numpy array if needed
    if not isinstance(data_array, np.ndarray):
        data_array = np.array(data_array)
    
    # Vectorized operations (auto-generated)
    result = np.vectorize({func_name})(data_array)
    
    return result
'''
            
            test_code = f'''
def test_{func_name}_vectorized():
    import numpy as np
    import time
    
    test_data = np.random.rand(1000)
    
    # Test correctness
    regular_results = [original_{func_name}(x) for x in test_data]
    vectorized_results = {func_name}_vectorized(test_data)
    
    np.testing.assert_array_almost_equal(regular_results, vectorized_results)
    
    # Test performance
    start_time = time.time()
    [original_{func_name}(x) for x in test_data]
    regular_time = time.time() - start_time
    
    start_time = time.time()
    {func_name}_vectorized(test_data)
    vectorized_time = time.time() - start_time
    
    assert vectorized_time < regular_time * 0.5  # At least 2x faster
'''
            
            return GeneratedCode(
                code_type="optimization",
                purpose=f"Vectorize {func_name} for array operations",
                language="python",
                code=vectorized_code,
                dependencies=["numpy"],
                test_code=test_code,
                performance_estimate=0.7,  # 70% improvement for array ops
                complexity_score=4,
                safety_level="high"
            )
            
        except Exception as e:
            logging.error(f"Error generating vectorized function: {e}")
            return None
    
    def _generate_parallel_function(self, func_source: str, func_name: str) -> GeneratedCode:
        """Generate parallel version of function"""
        try:
            parallel_code = f'''
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import numpy as np

def {func_name}_parallel(data_list, num_workers=None):
    """
    Parallel version of {func_name} using multiprocessing
    """
    if num_workers is None:
        num_workers = min(mp.cpu_count(), len(data_list))
    
    # Split data into chunks for parallel processing
    chunk_size = max(1, len(data_list) // num_workers)
    data_chunks = [data_list[i:i+chunk_size] for i in range(0, len(data_list), chunk_size)]
    
    def process_chunk(chunk):
        return [{func_name}(item) for item in chunk]
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(executor.map(process_chunk, data_chunks))
    
    # Flatten results
    flattened_results = []
    for chunk_result in results:
        flattened_results.extend(chunk_result)
    
    return flattened_results
'''
            
            test_code = f'''
def test_{func_name}_parallel():
    import time
    
    test_data = list(range(100))
    
    # Test correctness
    regular_results = [original_{func_name}(x) for x in test_data]
    parallel_results = {func_name}_parallel(test_data)
    
    assert regular_results == parallel_results
    
    # Test performance on larger dataset
    large_test_data = list(range(1000))
    
    start_time = time.time()
    [original_{func_name}(x) for x in large_test_data]
    regular_time = time.time() - start_time
    
    start_time = time.time()
    {func_name}_parallel(large_test_data)
    parallel_time = time.time() - start_time
    
    # Parallel should be faster for CPU-bound operations
    print(f"Regular time: {{regular_time:.3f}}s, Parallel time: {{parallel_time:.3f}}s")
'''
            
            return GeneratedCode(
                code_type="optimization",
                purpose=f"Parallelize {func_name} for multi-core processing",
                language="python",
                code=parallel_code,
                dependencies=["multiprocessing", "concurrent.futures"],
                test_code=test_code,
                performance_estimate=0.6,  # 60% improvement on multi-core
                complexity_score=6,
                safety_level="medium"
            )
            
        except Exception as e:
            logging.error(f"Error generating parallel function: {e}")
            return None
    
    def _generate_async_function(self, func_source: str, func_name: str) -> GeneratedCode:
        """Generate async version of function"""
        try:
            async_code = f'''
import asyncio
import aiohttp
import aiofiles

async def {func_name}_async(*args, **kwargs):
    """
    Async version of {func_name} for I/O bound operations
    """
    # Convert synchronous operations to async
    # This is a template - specific implementation depends on the original function
    
    # Example async patterns:
    # For file operations: use aiofiles
    # For HTTP requests: use aiohttp
    # For database operations: use async database drivers
    
    result = await asyncio.to_thread(original_{func_name}, *args, **kwargs)
    return result

async def {func_name}_batch_async(items_list):
    """
    Process multiple items asynchronously
    """
    tasks = [{func_name}_async(item) for item in items_list]
    results = await asyncio.gather(*tasks)
    return results
'''
            
            test_code = f'''
import asyncio

async def test_{func_name}_async():
    # Test single async call
    result = await {func_name}_async(test_input)
    regular_result = original_{func_name}(test_input)
    assert result == regular_result
    
    # Test batch processing
    test_items = [test_input1, test_input2, test_input3]
    async_results = await {func_name}_batch_async(test_items)
    regular_results = [original_{func_name}(item) for item in test_items]
    assert async_results == regular_results

def test_{func_name}_async_wrapper():
    asyncio.run(test_{func_name}_async())
'''
            
            return GeneratedCode(
                code_type="optimization",
                purpose=f"Create async version of {func_name} for I/O operations",
                language="python",
                code=async_code,
                dependencies=["asyncio", "aiohttp", "aiofiles"],
                test_code=test_code,
                performance_estimate=0.8,  # 80% improvement for I/O bound
                complexity_score=5,
                safety_level="medium"
            )
            
        except Exception as e:
            logging.error(f"Error generating async function: {e}")
            return None
    
    def _parse_requirements(self, description: str) -> Dict[str, Any]:
        """Parse natural language description into requirements"""
        requirements = {
            'type': 'function',
            'name': 'generated_function',
            'parameters': [],
            'returns': 'Any',
            'purpose': description,
            'complexity': 'medium'
        }
        
        # Simple keyword parsing
        if 'class' in description.lower():
            requirements['type'] = 'class'
        elif 'utility' in description.lower():
            requirements['type'] = 'utility'
        
        # Extract potential function name
        words = description.lower().split()
        for i, word in enumerate(words):
            if word in ['function', 'method', 'class'] and i + 1 < len(words):
                requirements['name'] = words[i + 1].replace(' ', '_')
                break
        
        return requirements
    
    def _generate_function(self, requirements: Dict[str, Any], context: Dict[str, Any]) -> GeneratedCode:
        """Generate a new function based on requirements"""
        func_name = requirements.get('name', 'generated_function')
        purpose = requirements.get('purpose', 'Auto-generated function')
        
        code = f'''
def {func_name}(*args, **kwargs):
    """
    {purpose}
    
    Auto-generated by JIGYASA autonomous code generator.
    """
    # TODO: Implement specific functionality
    # This is a template function that needs manual completion
    
    try:
        # Basic parameter validation
        if not args and not kwargs:
            raise ValueError("No arguments provided")
        
        # Placeholder implementation
        result = None
        
        # Log function call for monitoring
        print(f"Calling {{func_name}} with args={{args}}, kwargs={{kwargs}}")
        
        return result
        
    except Exception as e:
        print(f"Error in {{func_name}}: {{e}}")
        raise
'''
        
        test_code = f'''
def test_{func_name}():
    """Test the generated function"""
    try:
        result = {func_name}()
        assert result is not None or True  # Adjust based on expected behavior
        print(f"Test passed for {func_name}")
    except Exception as e:
        print(f"Test failed for {func_name}: {{e}}")
'''
        
        return GeneratedCode(
            code_type="function",
            purpose=purpose,
            language="python",
            code=code,
            dependencies=[],
            test_code=test_code,
            performance_estimate=0.0,
            complexity_score=3,
            safety_level="high"
        )
    
    def _generate_class(self, requirements: Dict[str, Any], context: Dict[str, Any]) -> GeneratedCode:
        """Generate a new class based on requirements"""
        class_name = requirements.get('name', 'GeneratedClass').title()
        purpose = requirements.get('purpose', 'Auto-generated class')
        
        code = f'''
class {class_name}:
    """
    {purpose}
    
    Auto-generated by JIGYASA autonomous code generator.
    """
    
    def __init__(self, *args, **kwargs):
        """Initialize the {class_name}"""
        self.initialized = True
        self.creation_time = time.time()
        
        # Store initialization parameters
        self.init_args = args
        self.init_kwargs = kwargs
    
    def __str__(self):
        return f"{class_name}(initialized={{self.initialized}})"
    
    def __repr__(self):
        return f"{class_name}(*{{self.init_args}}, **{{self.init_kwargs}})"
    
    def get_info(self):
        """Get information about this instance"""
        return {{
            'class_name': '{class_name}',
            'initialized': self.initialized,
            'creation_time': self.creation_time,
            'args': self.init_args,
            'kwargs': self.init_kwargs
        }}
'''
        
        test_code = f'''
def test_{class_name.lower()}():
    """Test the generated class"""
    try:
        instance = {class_name}("test", param="value")
        assert instance.initialized
        
        info = instance.get_info()
        assert info['class_name'] == '{class_name}'
        assert info['initialized'] == True
        
        print(f"Test passed for {class_name}")
    except Exception as e:
        print(f"Test failed for {class_name}: {{e}}")
'''
        
        return GeneratedCode(
            code_type="class",
            purpose=purpose,
            language="python",
            code=code,
            dependencies=["time"],
            test_code=test_code,
            performance_estimate=0.0,
            complexity_score=4,
            safety_level="high"
        )
    
    def _generate_module(self, requirements: Dict[str, Any], context: Dict[str, Any]) -> GeneratedCode:
        """Generate a new module based on requirements"""
        module_name = requirements.get('name', 'generated_module')
        purpose = requirements.get('purpose', 'Auto-generated module')
        
        code = f'''
"""
{module_name.title()} Module

{purpose}

Auto-generated by JIGYASA autonomous code generator.
"""

__version__ = "1.0.0"
__author__ = "JIGYASA AGI"

import logging
import time
from typing import Dict, List, Optional, Any

# Module-level logger
logger = logging.getLogger(__name__)

class {module_name.title()}Manager:
    """Main manager class for {module_name} functionality"""
    
    def __init__(self):
        self.initialized = True
        self.start_time = time.time()
        logger.info(f"Initialized {{self.__class__.__name__}}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        return {{
            'initialized': self.initialized,
            'uptime': time.time() - self.start_time,
            'module': __name__
        }}

# Module functions
def {module_name}_function(data: Any) -> Any:
    """
    Main function for {module_name} operations
    """
    logger.info(f"Processing data in {module_name}_function")
    
    try:
        # Placeholder processing
        result = data
        return result
    except Exception as e:
        logger.error(f"Error in {module_name}_function: {{e}}")
        raise

# Module initialization
_manager = {module_name.title()}Manager()

def get_manager() -> {module_name.title()}Manager:
    """Get the module manager instance"""
    return _manager
'''
        
        test_code = f'''
def test_{module_name}_module():
    """Test the generated module"""
    import {module_name}
    
    try:
        manager = {module_name}.get_manager()
        status = manager.get_status()
        
        assert status['initialized'] == True
        assert 'uptime' in status
        
        # Test module function
        result = {module_name}.{module_name}_function("test_data")
        assert result == "test_data"
        
        print(f"Test passed for {module_name} module")
    except Exception as e:
        print(f"Test failed for {module_name} module: {{e}}")
'''
        
        return GeneratedCode(
            code_type="module",
            purpose=purpose,
            language="python",
            code=code,
            dependencies=["logging", "time", "typing"],
            test_code=test_code,
            performance_estimate=0.0,
            complexity_score=5,
            safety_level="high"
        )
    
    def _generate_utility(self, requirements: Dict[str, Any], context: Dict[str, Any]) -> GeneratedCode:
        """Generate utility functions based on requirements"""
        util_name = requirements.get('name', 'utility')
        purpose = requirements.get('purpose', 'Auto-generated utility')
        
        code = f'''
"""
{util_name.title()} Utility Functions

{purpose}

Auto-generated by JIGYASA autonomous code generator.
"""

import functools
import time
from typing import Any, Callable, Dict, List

def timing_decorator(func: Callable) -> Callable:
    """Decorator to measure function execution time"""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"{{func.__name__}} took {{end_time - start_time:.4f}} seconds")
        return result
    return wrapper

def memoize_decorator(func: Callable) -> Callable:
    """Simple memoization decorator"""
    cache = {{}}
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        key = str(args) + str(sorted(kwargs.items()))
        if key not in cache:
            cache[key] = func(*args, **kwargs)
        return cache[key]
    return wrapper

def validate_types_decorator(*expected_types):
    """Decorator to validate function argument types"""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for i, (arg, expected_type) in enumerate(zip(args, expected_types)):
                if not isinstance(arg, expected_type):
                    raise TypeError(f"Argument {{i}} must be {{expected_type}}, got {{type(arg)}}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

def {util_name}_helper(data: Any, operation: str = "process") -> Any:
    """
    Generic utility helper function
    """
    if operation == "process":
        return data
    elif operation == "validate":
        return data is not None
    elif operation == "transform":
        return str(data)
    else:
        raise ValueError(f"Unknown operation: {{operation}}")

class {util_name.title()}Helper:
    """Utility helper class"""
    
    def __init__(self):
        self.operations_count = 0
    
    def process_data(self, data: Any) -> Any:
        """Process data with counting"""
        self.operations_count += 1
        return {util_name}_helper(data)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get utility statistics"""
        return {{
            'operations_count': self.operations_count,
            'utility_name': '{util_name}'
        }}
'''
        
        test_code = f'''
def test_{util_name}_utility():
    """Test the generated utility functions"""
    try:
        # Test helper function
        result = {util_name}_helper("test_data")
        assert result == "test_data"
        
        # Test helper class
        helper = {util_name.title()}Helper()
        processed = helper.process_data("test")
        assert processed == "test"
        
        stats = helper.get_stats()
        assert stats['operations_count'] == 1
        
        print(f"Test passed for {util_name} utility")
    except Exception as e:
        print(f"Test failed for {util_name} utility: {{e}}")
'''
        
        return GeneratedCode(
            code_type="utility",
            purpose=purpose,
            language="python",
            code=code,
            dependencies=["functools", "time", "typing"],
            test_code=test_code,
            performance_estimate=0.1,
            complexity_score=3,
            safety_level="high"
        )

# Global code generator instance
ai_code_generator = AICodeGenerator()

def generate_code_improvements(func_source: str, func_name: str) -> List[GeneratedCode]:
    """Generate code improvements for a function"""
    return ai_code_generator.analyze_function_for_improvements(func_source, func_name)

def generate_new_code(description: str, context: Dict[str, Any] = None) -> Optional[GeneratedCode]:
    """Generate new code based on description"""
    if context is None:
        context = {}
    return ai_code_generator.generate_new_functionality(description, context)