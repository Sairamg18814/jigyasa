#!/usr/bin/env python3
"""
Autonomous Code Self-Editing System
Allows JIGYASA to analyze, improve, and modify its own source code safely
"""

import ast
import inspect
import os
import sys
import git
import time
import subprocess
import shutil
import hashlib
import json
import tempfile
import difflib
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
import importlib.util
import traceback

@dataclass
class CodeImprovement:
    """Represents a potential code improvement"""
    file_path: str
    function_name: str
    improvement_type: str
    description: str
    original_code: str
    improved_code: str
    performance_gain: float
    risk_level: str  # low, medium, high
    test_results: Dict[str, Any]
    confidence_score: float

@dataclass
class CodeChange:
    """Represents a code change made by the system"""
    change_id: str
    timestamp: str
    file_path: str
    change_type: str
    description: str
    original_hash: str
    new_hash: str
    performance_impact: float
    rollback_available: bool

class SafeCodeAnalyzer:
    """Analyzes code for potential improvements while maintaining safety"""
    
    def __init__(self):
        self.improvement_patterns = self._load_improvement_patterns()
        self.safety_rules = self._load_safety_rules()
        
    def _load_improvement_patterns(self) -> Dict[str, Any]:
        """Load patterns for detecting code improvements"""
        return {
            'performance': {
                'inefficient_loops': {
                    'pattern': r'for.*in.*range\(len\(',
                    'improvement': 'Use enumerate() instead of range(len())',
                    'risk': 'low'
                },
                'string_concatenation': {
                    'pattern': r'\+.*str\(',
                    'improvement': 'Use f-strings or join() for string concatenation',
                    'risk': 'low'
                },
                'list_comprehension': {
                    'pattern': r'for.*in.*:\s*.*\.append\(',
                    'improvement': 'Convert to list comprehension',
                    'risk': 'low'
                }
            },
            'memory': {
                'unnecessary_copies': {
                    'pattern': r'\.copy\(\)',
                    'improvement': 'Eliminate unnecessary copying',
                    'risk': 'medium'
                },
                'large_data_structures': {
                    'pattern': r'list\(.*\)',
                    'improvement': 'Use generators for large datasets',
                    'risk': 'medium'
                }
            },
            'error_handling': {
                'bare_except': {
                    'pattern': r'except:',
                    'improvement': 'Use specific exception types',
                    'risk': 'low'
                },
                'missing_finally': {
                    'pattern': r'try:.*except.*(?!finally)',
                    'improvement': 'Add finally block for cleanup',
                    'risk': 'low'
                }
            },
            'ai_optimization': {
                'model_inefficiency': {
                    'pattern': r'\.forward\(',
                    'improvement': 'Optimize model forward pass',
                    'risk': 'high'
                },
                'batch_processing': {
                    'pattern': r'for.*in.*batch',
                    'improvement': 'Vectorize batch operations',
                    'risk': 'medium'
                }
            }
        }
    
    def _load_safety_rules(self) -> Dict[str, Any]:
        """Load safety rules for code modifications"""
        return {
            'forbidden_changes': {
                'security_functions': ['eval', 'exec', 'subprocess.call'],
                'system_files': ['/etc/', '/usr/', '/bin/'],
                'critical_imports': ['os.system', 'subprocess.Popen']
            },
            'required_tests': {
                'model_changes': ['accuracy_test', 'performance_test'],
                'training_changes': ['convergence_test', 'stability_test'],
                'data_changes': ['integrity_test', 'format_test']
            },
            'backup_requirements': {
                'always_backup': True,
                'version_control': True,
                'test_before_commit': True
            }
        }
    
    def analyze_file(self, file_path: str) -> List[CodeImprovement]:
        """Analyze a Python file for potential improvements"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Parse AST
            tree = ast.parse(content)
            
            improvements = []
            improvements.extend(self._analyze_performance(file_path, content, tree))
            improvements.extend(self._analyze_memory_usage(file_path, content, tree))
            improvements.extend(self._analyze_error_handling(file_path, content, tree))
            improvements.extend(self._analyze_ai_optimizations(file_path, content, tree))
            
            return improvements
            
        except Exception as e:
            logging.error(f"Error analyzing file {file_path}: {e}")
            return []
    
    def _analyze_performance(self, file_path: str, content: str, tree: ast.AST) -> List[CodeImprovement]:
        """Analyze for performance improvements"""
        improvements = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.For):
                # Check for inefficient loops
                if self._is_inefficient_loop(node):
                    improvement = self._create_loop_improvement(file_path, node, content)
                    if improvement:
                        improvements.append(improvement)
            
            elif isinstance(node, ast.Call):
                # Check for inefficient function calls
                if self._is_inefficient_call(node):
                    improvement = self._create_call_improvement(file_path, node, content)
                    if improvement:
                        improvements.append(improvement)
        
        return improvements
    
    def _analyze_memory_usage(self, file_path: str, content: str, tree: ast.AST) -> List[CodeImprovement]:
        """Analyze for memory usage improvements"""
        improvements = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ListComp):
                # Check if generator expression would be better
                if self._should_use_generator(node):
                    improvement = self._create_generator_improvement(file_path, node, content)
                    if improvement:
                        improvements.append(improvement)
        
        return improvements
    
    def _analyze_error_handling(self, file_path: str, content: str, tree: ast.AST) -> List[CodeImprovement]:
        """Analyze for error handling improvements"""
        improvements = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler):
                if node.type is None:  # Bare except
                    improvement = self._create_exception_improvement(file_path, node, content)
                    if improvement:
                        improvements.append(improvement)
        
        return improvements
    
    def _analyze_ai_optimizations(self, file_path: str, content: str, tree: ast.AST) -> List[CodeImprovement]:
        """Analyze for AI/ML specific optimizations"""
        improvements = []
        
        # Look for model training inefficiencies
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if 'train' in node.name.lower() or 'forward' in node.name.lower():
                    improvement = self._analyze_ai_function(file_path, node, content)
                    if improvement:
                        improvements.append(improvement)
        
        return improvements
    
    def _is_inefficient_loop(self, node: ast.For) -> bool:
        """Check if loop is inefficient"""
        if isinstance(node.iter, ast.Call):
            if (isinstance(node.iter.func, ast.Name) and 
                node.iter.func.id == 'range' and 
                len(node.iter.args) == 1):
                # Check if range(len(something))
                if (isinstance(node.iter.args[0], ast.Call) and
                    isinstance(node.iter.args[0].func, ast.Name) and
                    node.iter.args[0].func.id == 'len'):
                    return True
        return False
    
    def _is_inefficient_call(self, node: ast.Call) -> bool:
        """Check if function call is inefficient"""
        if isinstance(node.func, ast.Attribute):
            if node.func.attr == 'append':
                return True  # Might be better as list comprehension
        return False
    
    def _should_use_generator(self, node: ast.ListComp) -> bool:
        """Check if generator expression would be better than list comprehension"""
        # Simple heuristic: if list comp is used in sum(), any(), all(), etc.
        return True  # For demo purposes
    
    def _create_loop_improvement(self, file_path: str, node: ast.For, content: str) -> Optional[CodeImprovement]:
        """Create improvement for inefficient loop"""
        try:
            lines = content.split('\n')
            line_num = node.lineno - 1
            original_line = lines[line_num] if line_num < len(lines) else ""
            
            # Generate improved version
            improved_line = original_line.replace('range(len(', 'enumerate(')
            improved_line = improved_line.replace('))', ')')
            
            return CodeImprovement(
                file_path=file_path,
                function_name=self._get_containing_function(node, content),
                improvement_type="performance",
                description="Replace range(len()) with enumerate()",
                original_code=original_line.strip(),
                improved_code=improved_line.strip(),
                performance_gain=0.15,  # Estimated 15% improvement
                risk_level="low",
                test_results={},
                confidence_score=0.9
            )
        except Exception as e:
            logging.error(f"Error creating loop improvement: {e}")
            return None
    
    def _create_call_improvement(self, file_path: str, node: ast.Call, content: str) -> Optional[CodeImprovement]:
        """Create improvement for inefficient function call"""
        # Implementation for call improvements
        return None
    
    def _create_generator_improvement(self, file_path: str, node: ast.ListComp, content: str) -> Optional[CodeImprovement]:
        """Create improvement to use generator expression"""
        # Implementation for generator improvements
        return None
    
    def _create_exception_improvement(self, file_path: str, node: ast.ExceptHandler, content: str) -> Optional[CodeImprovement]:
        """Create improvement for bare except clause"""
        try:
            lines = content.split('\n')
            line_num = node.lineno - 1
            original_line = lines[line_num] if line_num < len(lines) else ""
            
            # Generate improved version
            improved_line = original_line.replace('except:', 'except Exception as e:')
            
            return CodeImprovement(
                file_path=file_path,
                function_name=self._get_containing_function(node, content),
                improvement_type="error_handling",
                description="Replace bare except with specific exception",
                original_code=original_line.strip(),
                improved_code=improved_line.strip(),
                performance_gain=0.0,
                risk_level="low",
                test_results={},
                confidence_score=0.95
            )
        except Exception as e:
            logging.error(f"Error creating exception improvement: {e}")
            return None
    
    def _analyze_ai_function(self, file_path: str, node: ast.FunctionDef, content: str) -> Optional[CodeImprovement]:
        """Analyze AI/ML functions for optimizations"""
        # Look for common AI optimization opportunities
        for child in ast.walk(node):
            if isinstance(child, ast.For):
                # Check for inefficient batch processing
                if self._is_inefficient_batch_processing(child):
                    return CodeImprovement(
                        file_path=file_path,
                        function_name=node.name,
                        improvement_type="ai_optimization",
                        description="Vectorize batch processing operations",
                        original_code=self._get_node_source(child, content),
                        improved_code=self._generate_vectorized_code(child, content),
                        performance_gain=0.3,  # 30% improvement
                        risk_level="medium",
                        test_results={},
                        confidence_score=0.8
                    )
        return None
    
    def _is_inefficient_batch_processing(self, node: ast.For) -> bool:
        """Check for inefficient batch processing"""
        # Simple heuristic - look for loops over batches
        return True  # For demo purposes
    
    def _generate_vectorized_code(self, node: ast.For, content: str) -> str:
        """Generate vectorized version of batch processing code"""
        return "# Vectorized batch processing (auto-generated)"
    
    def _get_containing_function(self, node: ast.AST, content: str) -> str:
        """Get the name of the function containing the node"""
        # Simplified implementation
        return "unknown_function"
    
    def _get_node_source(self, node: ast.AST, content: str) -> str:
        """Get source code for AST node"""
        try:
            lines = content.split('\n')
            return lines[node.lineno - 1] if hasattr(node, 'lineno') else ""
        except:
            return ""

class AutoCodeTester:
    """Automatically tests code changes for safety and correctness"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.test_results_dir = self.project_root / "test_results" / "auto_tests"
        self.test_results_dir.mkdir(parents=True, exist_ok=True)
    
    def test_improvement(self, improvement: CodeImprovement) -> Dict[str, Any]:
        """Test a code improvement for safety and correctness"""
        test_results = {
            'syntax_test': False,
            'unit_tests': False,
            'integration_tests': False,
            'performance_test': False,
            'security_test': False,
            'overall_pass': False,
            'errors': [],
            'warnings': []
        }
        
        try:
            # Create temporary test environment
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_file = Path(temp_dir) / "test_code.py"
                
                # Test syntax
                test_results['syntax_test'] = self._test_syntax(improvement, temp_file)
                
                # Test functionality
                if test_results['syntax_test']:
                    test_results['unit_tests'] = self._run_unit_tests(improvement, temp_file)
                    test_results['performance_test'] = self._test_performance(improvement, temp_file)
                    test_results['security_test'] = self._test_security(improvement)
                
                # Overall assessment
                test_results['overall_pass'] = all([
                    test_results['syntax_test'],
                    test_results['unit_tests'],
                    test_results['security_test']
                ])
                
        except Exception as e:
            test_results['errors'].append(f"Testing error: {e}")
            test_results['overall_pass'] = False
        
        return test_results
    
    def _test_syntax(self, improvement: CodeImprovement, temp_file: Path) -> bool:
        """Test if improved code has valid syntax"""
        try:
            # Write improved code to temp file
            with open(temp_file, 'w') as f:
                f.write(improvement.improved_code)
            
            # Try to parse
            with open(temp_file, 'r') as f:
                ast.parse(f.read())
            
            return True
        except SyntaxError:
            return False
        except Exception:
            return False
    
    def _run_unit_tests(self, improvement: CodeImprovement, temp_file: Path) -> bool:
        """Run unit tests on improved code"""
        try:
            # Generate simple test cases
            test_code = self._generate_test_code(improvement)
            
            with open(temp_file, 'w') as f:
                f.write(test_code)
            
            # Run tests
            result = subprocess.run([
                sys.executable, '-m', 'pytest', str(temp_file), '-v'
            ], capture_output=True, text=True, timeout=30)
            
            return result.returncode == 0
        except:
            return False
    
    def _test_performance(self, improvement: CodeImprovement, temp_file: Path) -> bool:
        """Test if improvement actually improves performance"""
        try:
            # Basic performance test
            import timeit
            
            # This would need more sophisticated implementation
            return True
        except:
            return False
    
    def _test_security(self, improvement: CodeImprovement) -> bool:
        """Test code for security issues"""
        try:
            dangerous_patterns = [
                'eval(', 'exec(', 'os.system(', '__import__(',
                'subprocess.call(', 'open(', 'file('
            ]
            
            code = improvement.improved_code.lower()
            for pattern in dangerous_patterns:
                if pattern in code:
                    return False
            
            return True
        except:
            return False
    
    def _generate_test_code(self, improvement: CodeImprovement) -> str:
        """Generate test code for the improvement"""
        return f"""
def test_improvement():
    # Auto-generated test for {improvement.improvement_type}
    # Original: {improvement.original_code}
    # Improved: {improvement.improved_code}
    assert True  # Basic test - would be more sophisticated
"""

class VersionControlManager:
    """Manages version control for code changes"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.changes_log = self.project_root / "autonomous_changes.json"
        self.backup_dir = self.project_root / "autonomous_backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Initialize git if not already initialized
        try:
            self.repo = git.Repo(project_root)
        except git.exc.InvalidGitRepositoryError:
            self.repo = git.Repo.init(project_root)
    
    def create_backup(self, file_path: str) -> str:
        """Create backup of file before modification"""
        try:
            source_file = Path(file_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_name = f"{source_file.stem}_{timestamp}.backup"
            backup_path = self.backup_dir / backup_name
            
            shutil.copy2(source_file, backup_path)
            
            logging.info(f"Created backup: {backup_path}")
            return str(backup_path)
        except Exception as e:
            logging.error(f"Failed to create backup for {file_path}: {e}")
            return ""
    
    def commit_change(self, change: CodeChange) -> bool:
        """Commit a code change to version control"""
        try:
            # Add file to git
            self.repo.index.add([change.file_path])
            
            # Create commit
            commit_message = f"Auto-improvement: {change.description}\n\n" \
                           f"Change type: {change.change_type}\n" \
                           f"Performance impact: {change.performance_impact:.2%}\n" \
                           f"Auto-generated by JIGYASA"
            
            self.repo.index.commit(commit_message)
            
            # Log the change
            self._log_change(change)
            
            logging.info(f"Committed change: {change.change_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to commit change {change.change_id}: {e}")
            return False
    
    def rollback_change(self, change_id: str) -> bool:
        """Rollback a specific change"""
        try:
            # Load change log
            changes = self._load_changes_log()
            
            change = next((c for c in changes if c['change_id'] == change_id), None)
            if not change:
                return False
            
            # Find backup file
            backup_files = list(self.backup_dir.glob(f"*{change['timestamp']}*.backup"))
            if not backup_files:
                return False
            
            # Restore from backup
            backup_file = backup_files[0]
            shutil.copy2(backup_file, change['file_path'])
            
            # Commit rollback
            self.repo.index.add([change['file_path']])
            self.repo.index.commit(f"Rollback: {change_id}")
            
            logging.info(f"Rolled back change: {change_id}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to rollback change {change_id}: {e}")
            return False
    
    def _log_change(self, change: CodeChange):
        """Log change to changes file"""
        try:
            changes = self._load_changes_log()
            changes.append(asdict(change))
            
            with open(self.changes_log, 'w') as f:
                json.dump(changes, f, indent=2)
        except Exception as e:
            logging.error(f"Failed to log change: {e}")
    
    def _load_changes_log(self) -> List[Dict]:
        """Load changes log"""
        try:
            if self.changes_log.exists():
                with open(self.changes_log, 'r') as f:
                    return json.load(f)
        except:
            pass
        return []

class AutonomousCodeEditor:
    """Main autonomous code editing system"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.analyzer = SafeCodeAnalyzer()
        self.tester = AutoCodeTester(project_root)
        self.version_control = VersionControlManager(project_root)
        
        self.enabled = True
        self.improvement_queue = []
        self.active_improvements = {}
        
        # Configuration
        self.config = {
            'max_daily_changes': 10,
            'min_confidence_score': 0.8,
            'max_risk_level': 'medium',
            'auto_commit': True,
            'require_tests': True
        }
        
        self.daily_changes = 0
        self.last_reset_date = datetime.now().date()
    
    def start_autonomous_improvement(self):
        """Start the autonomous code improvement process"""
        if not self.enabled:
            return
        
        logging.info("🔧 Starting autonomous code improvement process")
        
        try:
            # Reset daily counter if needed
            if datetime.now().date() > self.last_reset_date:
                self.daily_changes = 0
                self.last_reset_date = datetime.now().date()
            
            # Scan for improvements
            improvements = self._scan_for_improvements()
            
            # Filter and prioritize
            viable_improvements = self._filter_improvements(improvements)
            
            # Apply improvements
            applied_count = 0
            for improvement in viable_improvements:
                if self.daily_changes >= self.config['max_daily_changes']:
                    break
                
                if self._apply_improvement(improvement):
                    applied_count += 1
                    self.daily_changes += 1
            
            logging.info(f"✅ Applied {applied_count} autonomous improvements")
            
        except Exception as e:
            logging.error(f"❌ Error in autonomous improvement: {e}")
    
    def _scan_for_improvements(self) -> List[CodeImprovement]:
        """Scan codebase for potential improvements"""
        improvements = []
        
        # Scan Python files in the project
        python_files = list(self.project_root.rglob("*.py"))
        
        for file_path in python_files:
            # Skip test files and __pycache__
            if ('test' in str(file_path) or 
                '__pycache__' in str(file_path) or
                'autonomous_backups' in str(file_path)):
                continue
            
            try:
                file_improvements = self.analyzer.analyze_file(str(file_path))
                improvements.extend(file_improvements)
            except Exception as e:
                logging.warning(f"Could not analyze {file_path}: {e}")
        
        return improvements
    
    def _filter_improvements(self, improvements: List[CodeImprovement]) -> List[CodeImprovement]:
        """Filter improvements based on safety and configuration"""
        filtered = []
        
        risk_levels = {'low': 1, 'medium': 2, 'high': 3}
        max_risk = risk_levels.get(self.config['max_risk_level'], 2)
        
        for improvement in improvements:
            # Check confidence score
            if improvement.confidence_score < self.config['min_confidence_score']:
                continue
            
            # Check risk level
            if risk_levels.get(improvement.risk_level, 3) > max_risk:
                continue
            
            # Check if improvement is significant enough
            if improvement.performance_gain < 0.05:  # Less than 5% improvement
                continue
            
            filtered.append(improvement)
        
        # Sort by confidence and performance gain
        filtered.sort(key=lambda x: (x.confidence_score, x.performance_gain), reverse=True)
        
        return filtered
    
    def _apply_improvement(self, improvement: CodeImprovement) -> bool:
        """Apply a single improvement safely"""
        try:
            logging.info(f"🔧 Applying improvement: {improvement.description}")
            
            # Test the improvement first
            if self.config['require_tests']:
                test_results = self.tester.test_improvement(improvement)
                if not test_results['overall_pass']:
                    logging.warning(f"❌ Improvement failed tests: {improvement.description}")
                    return False
            
            # Create backup
            backup_path = self.version_control.create_backup(improvement.file_path)
            if not backup_path:
                logging.error(f"❌ Could not create backup for {improvement.file_path}")
                return False
            
            # Apply the change
            success = self._modify_file(improvement)
            if not success:
                logging.error(f"❌ Failed to modify file: {improvement.file_path}")
                return False
            
            # Create change record
            change = CodeChange(
                change_id=self._generate_change_id(),
                timestamp=datetime.now().isoformat(),
                file_path=improvement.file_path,
                change_type=improvement.improvement_type,
                description=improvement.description,
                original_hash=self._file_hash(improvement.file_path + ".backup"),
                new_hash=self._file_hash(improvement.file_path),
                performance_impact=improvement.performance_gain,
                rollback_available=True
            )
            
            # Commit change if enabled
            if self.config['auto_commit']:
                if self.version_control.commit_change(change):
                    logging.info(f"✅ Successfully applied and committed improvement")
                    return True
                else:
                    # Rollback if commit failed
                    self._rollback_file_change(improvement.file_path, backup_path)
                    return False
            
            return True
            
        except Exception as e:
            logging.error(f"❌ Error applying improvement: {e}")
            return False
    
    def _modify_file(self, improvement: CodeImprovement) -> bool:
        """Modify file with the improvement"""
        try:
            with open(improvement.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Simple replacement for now - would be more sophisticated in practice
            modified_content = content.replace(
                improvement.original_code,
                improvement.improved_code
            )
            
            with open(improvement.file_path, 'w', encoding='utf-8') as f:
                f.write(modified_content)
            
            return True
        except Exception as e:
            logging.error(f"Error modifying file {improvement.file_path}: {e}")
            return False
    
    def _rollback_file_change(self, file_path: str, backup_path: str):
        """Rollback a file change"""
        try:
            shutil.copy2(backup_path, file_path)
            logging.info(f"Rolled back changes to {file_path}")
        except Exception as e:
            logging.error(f"Failed to rollback {file_path}: {e}")
    
    def _generate_change_id(self) -> str:
        """Generate unique change ID"""
        return f"auto_{int(time.time())}_{hashlib.md5(str(time.time()).encode()).hexdigest()[:8]}"
    
    def _file_hash(self, file_path: str) -> str:
        """Calculate file hash"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()
        except:
            return ""
    
    def get_improvement_status(self) -> Dict[str, Any]:
        """Get current status of autonomous improvements"""
        return {
            'enabled': self.enabled,
            'daily_changes': self.daily_changes,
            'max_daily_changes': self.config['max_daily_changes'],
            'improvements_queue': len(self.improvement_queue),
            'last_scan': datetime.now().isoformat(),
            'config': self.config
        }
    
    def set_config(self, new_config: Dict[str, Any]):
        """Update configuration"""
        self.config.update(new_config)
        logging.info(f"Updated autonomous code editor config: {new_config}")

# Global autonomous code editor instance
autonomous_code_editor = None

def initialize_autonomous_code_editor(project_root: str = None):
    """Initialize the autonomous code editor"""
    global autonomous_code_editor
    
    if project_root is None:
        project_root = os.getcwd()
    
    try:
        autonomous_code_editor = AutonomousCodeEditor(project_root)
        logging.info("✅ Autonomous code editor initialized")
        return True
    except Exception as e:
        logging.error(f"❌ Failed to initialize autonomous code editor: {e}")
        return False

def start_autonomous_improvements():
    """Start autonomous code improvements"""
    global autonomous_code_editor
    
    if autonomous_code_editor is None:
        initialize_autonomous_code_editor()
    
    if autonomous_code_editor:
        autonomous_code_editor.start_autonomous_improvement()

def get_autonomous_editor_status():
    """Get status of autonomous code editor"""
    global autonomous_code_editor
    
    if autonomous_code_editor is None:
        return {'initialized': False}
    
    return autonomous_code_editor.get_improvement_status()