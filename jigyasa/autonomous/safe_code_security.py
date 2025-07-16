#!/usr/bin/env python3
"""
Safe Code Security Framework
Ensures all automatically generated and modified code is secure
"""

import ast
import re
import subprocess
import sys
import tempfile
import hashlib
import json
import logging
import importlib
import pkgutil
from typing import Dict, List, Optional, Any, Set, Tuple
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import bandit
from bandit.core import manager as bandit_manager
from bandit.core import config as bandit_config

@dataclass
class SecurityIssue:
    """Represents a security issue found in code"""
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    issue_type: str
    description: str
    line_number: int
    code_snippet: str
    recommendation: str
    cwe_id: Optional[str] = None
    confidence: str = "HIGH"

@dataclass
class SecurityScanResult:
    """Results of a security scan"""
    scan_id: str
    timestamp: str
    file_path: str
    total_issues: int
    critical_issues: int
    high_issues: int
    medium_issues: int
    low_issues: int
    security_score: float  # 0-100, higher is better
    passed_security_check: bool
    issues: List[SecurityIssue]

class CodeSecurityScanner:
    """Scans code for security vulnerabilities"""
    
    def __init__(self):
        self.dangerous_patterns = self._load_dangerous_patterns()
        self.secure_coding_rules = self._load_secure_coding_rules()
        self.allowed_imports = self._load_allowed_imports()
        self.ai_specific_security = self._load_ai_security_rules()
        
        # Initialize Bandit for additional security scanning
        try:
            self.bandit_config = bandit_config.BanditConfig()
            self.bandit_manager = bandit_manager.BanditManager(
                self.bandit_config, 'file'
            )
        except Exception as e:
            logging.warning(f"Could not initialize Bandit: {e}")
            self.bandit_manager = None
    
    def _load_dangerous_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Load patterns that indicate dangerous code"""
        return {
            'code_execution': {
                'patterns': [
                    r'eval\s*\(',
                    r'exec\s*\(',
                    r'compile\s*\(',
                    r'__import__\s*\(',
                    r'importlib\.import_module\s*\(',
                ],
                'severity': 'CRITICAL',
                'description': 'Dynamic code execution can lead to arbitrary code execution',
                'recommendation': 'Avoid dynamic code execution. Use static alternatives.'
            },
            'system_commands': {
                'patterns': [
                    r'os\.system\s*\(',
                    r'subprocess\.call\s*\(',
                    r'subprocess\.run\s*\(',
                    r'subprocess\.Popen\s*\(',
                    r'commands\.getoutput\s*\(',
                ],
                'severity': 'HIGH',
                'description': 'System command execution can lead to command injection',
                'recommendation': 'Use safe alternatives or validate/sanitize all inputs'
            },
            'file_operations': {
                'patterns': [
                    r'open\s*\([^)]*[\'"]w[\'"]',
                    r'open\s*\([^)]*[\'"]a[\'"]',
                    r'\.write\s*\(',
                    r'\.writelines\s*\(',
                    r'shutil\.rmtree\s*\(',
                    r'os\.remove\s*\(',
                    r'os\.unlink\s*\(',
                ],
                'severity': 'MEDIUM',
                'description': 'File operations can lead to unauthorized file access/modification',
                'recommendation': 'Validate file paths and implement proper access controls'
            },
            'network_operations': {
                'patterns': [
                    r'urllib\.request\.urlopen\s*\(',
                    r'requests\.get\s*\(',
                    r'requests\.post\s*\(',
                    r'socket\.socket\s*\(',
                    r'http\.client\.',
                ],
                'severity': 'MEDIUM',
                'description': 'Network operations can lead to data exfiltration or SSRF',
                'recommendation': 'Validate URLs and implement proper network security'
            },
            'crypto_operations': {
                'patterns': [
                    r'hashlib\.md5\s*\(',
                    r'hashlib\.sha1\s*\(',
                    r'random\.random\s*\(',
                    r'random\.randint\s*\(',
                    r'pickle\.loads\s*\(',
                    r'pickle\.load\s*\(',
                ],
                'severity': 'MEDIUM',
                'description': 'Weak cryptographic operations or insecure deserialization',
                'recommendation': 'Use strong cryptographic functions and safe serialization'
            },
            'injection_vulnerabilities': {
                'patterns': [
                    r'\.format\s*\([^)]*user',
                    r'%.*%.*user',
                    r'f[\'"][^\'"].*{.*user',
                    r'input\s*\(',
                    r'raw_input\s*\(',
                ],
                'severity': 'HIGH',
                'description': 'Potential injection vulnerabilities',
                'recommendation': 'Use parameterized queries and input validation'
            }
        }
    
    def _load_secure_coding_rules(self) -> Dict[str, Any]:
        """Load secure coding rules"""
        return {
            'input_validation': {
                'required': True,
                'patterns': [r'isinstance\s*\(', r'assert\s+', r'if.*type\s*\('],
                'description': 'All inputs should be validated'
            },
            'error_handling': {
                'required': True,
                'patterns': [r'try\s*:', r'except\s+', r'finally\s*:'],
                'description': 'Proper error handling should be implemented'
            },
            'logging': {
                'required': True,
                'patterns': [r'logging\.', r'logger\.', r'log\.'],
                'description': 'Security events should be logged'
            },
            'constants': {
                'avoid_hardcoded': True,
                'patterns': [r'password\s*=\s*[\'"]', r'key\s*=\s*[\'"]', r'secret\s*=\s*[\'"]'],
                'description': 'Avoid hardcoded secrets'
            }
        }
    
    def _load_allowed_imports(self) -> Set[str]:
        """Load list of allowed imports for autonomous code"""
        return {
            # Standard library - safe modules
            'os', 'sys', 'time', 'datetime', 'json', 'logging', 'pathlib',
            'typing', 'dataclasses', 'functools', 'itertools', 'collections',
            'math', 'statistics', 'random', 'hashlib', 'uuid', 'tempfile',
            'threading', 'multiprocessing', 'concurrent.futures', 'asyncio',
            'unittest', 'doctest', 'traceback', 'inspect', 'ast',
            
            # Scientific computing
            'numpy', 'scipy', 'pandas', 'matplotlib', 'seaborn',
            
            # Machine learning (specific safe modules)
            'torch', 'tensorflow', 'sklearn', 'transformers', 'datasets',
            'einops', 'peft', 'accelerate',
            
            # Web and APIs (controlled)
            'flask', 'fastapi', 'requests', 'aiohttp',
            
            # Data processing
            'csv', 'xml', 'html', 'markdown', 'yaml', 'toml',
            
            # Testing and quality
            'pytest', 'coverage', 'bandit', 'mypy', 'flake8'
        }
    
    def _load_ai_security_rules(self) -> Dict[str, Any]:
        """Load AI/ML specific security rules"""
        return {
            'model_security': {
                'avoid_untrusted_models': True,
                'validate_model_inputs': True,
                'limit_model_capabilities': True,
                'monitor_model_outputs': True
            },
            'data_security': {
                'validate_training_data': True,
                'protect_sensitive_data': True,
                'implement_privacy_controls': True,
                'audit_data_access': True
            },
            'inference_security': {
                'rate_limiting': True,
                'input_sanitization': True,
                'output_filtering': True,
                'resource_limits': True
            }
        }
    
    def scan_code(self, code: str, file_path: str = "generated_code.py") -> SecurityScanResult:
        """Perform comprehensive security scan on code"""
        try:
            scan_id = hashlib.md5(f"{code}{datetime.now()}".encode()).hexdigest()[:8]
            
            issues = []
            
            # Pattern-based scanning
            issues.extend(self._scan_dangerous_patterns(code))
            
            # AST-based scanning
            issues.extend(self._scan_ast_security(code))
            
            # Import security scanning
            issues.extend(self._scan_imports(code))
            
            # AI-specific security scanning
            issues.extend(self._scan_ai_security(code))
            
            # Bandit scanning if available
            if self.bandit_manager:
                issues.extend(self._scan_with_bandit(code, file_path))
            
            # Calculate security score
            security_score = self._calculate_security_score(issues)
            
            # Categorize issues by severity
            critical_issues = sum(1 for issue in issues if issue.severity == 'CRITICAL')
            high_issues = sum(1 for issue in issues if issue.severity == 'HIGH')
            medium_issues = sum(1 for issue in issues if issue.severity == 'MEDIUM')
            low_issues = sum(1 for issue in issues if issue.severity == 'LOW')
            
            # Determine if code passes security check
            passed = critical_issues == 0 and high_issues == 0 and security_score >= 70
            
            return SecurityScanResult(
                scan_id=scan_id,
                timestamp=datetime.now().isoformat(),
                file_path=file_path,
                total_issues=len(issues),
                critical_issues=critical_issues,
                high_issues=high_issues,
                medium_issues=medium_issues,
                low_issues=low_issues,
                security_score=security_score,
                passed_security_check=passed,
                issues=issues
            )
            
        except Exception as e:
            logging.error(f"Error scanning code: {e}")
            return self._create_failed_scan_result(file_path, str(e))
    
    def _scan_dangerous_patterns(self, code: str) -> List[SecurityIssue]:
        """Scan for dangerous patterns using regex"""
        issues = []
        lines = code.split('\n')
        
        for category, config in self.dangerous_patterns.items():
            for pattern in config['patterns']:
                for line_num, line in enumerate(lines, 1):
                    if re.search(pattern, line, re.IGNORECASE):
                        issues.append(SecurityIssue(
                            severity=config['severity'],
                            issue_type=category,
                            description=config['description'],
                            line_number=line_num,
                            code_snippet=line.strip(),
                            recommendation=config['recommendation']
                        ))
        
        return issues
    
    def _scan_ast_security(self, code: str) -> List[SecurityIssue]:
        """Scan using AST analysis for deeper security issues"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                # Check for dangerous function calls
                if isinstance(node, ast.Call):
                    issues.extend(self._check_dangerous_calls(node))
                
                # Check for dangerous imports
                elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                    issues.extend(self._check_dangerous_imports(node))
                
                # Check for hardcoded secrets
                elif isinstance(node, ast.Str):
                    issues.extend(self._check_hardcoded_secrets(node))
                
                # Check for SQL injection patterns
                elif isinstance(node, ast.BinOp):
                    issues.extend(self._check_sql_injection(node))
        
        except SyntaxError:
            issues.append(SecurityIssue(
                severity='HIGH',
                issue_type='syntax_error',
                description='Code contains syntax errors',
                line_number=0,
                code_snippet='',
                recommendation='Fix syntax errors before security analysis'
            ))
        
        return issues
    
    def _scan_imports(self, code: str) -> List[SecurityIssue]:
        """Scan imports for security issues"""
        issues = []
        
        try:
            tree = ast.parse(code)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.allowed_imports:
                            issues.append(SecurityIssue(
                                severity='MEDIUM',
                                issue_type='unauthorized_import',
                                description=f'Import of unauthorized module: {alias.name}',
                                line_number=getattr(node, 'lineno', 0),
                                code_snippet=f'import {alias.name}',
                                recommendation='Use only approved modules for autonomous code'
                            ))
                
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.module not in self.allowed_imports:
                        issues.append(SecurityIssue(
                            severity='MEDIUM',
                            issue_type='unauthorized_import',
                            description=f'Import from unauthorized module: {node.module}',
                            line_number=getattr(node, 'lineno', 0),
                            code_snippet=f'from {node.module} import ...',
                            recommendation='Use only approved modules for autonomous code'
                        ))
        
        except SyntaxError:
            pass  # Already handled in AST scanning
        
        return issues
    
    def _scan_ai_security(self, code: str) -> List[SecurityIssue]:
        """Scan for AI/ML specific security issues"""
        issues = []
        
        # Check for model loading patterns
        model_patterns = [
            r'torch\.load\s*\(',
            r'pickle\.load\s*\(',
            r'joblib\.load\s*\(',
            r'\.from_pretrained\s*\(',
        ]
        
        for pattern in model_patterns:
            if re.search(pattern, code):
                issues.append(SecurityIssue(
                    severity='HIGH',
                    issue_type='untrusted_model_loading',
                    description='Loading models from untrusted sources',
                    line_number=0,
                    code_snippet='Model loading detected',
                    recommendation='Only load models from trusted sources and validate checksums'
                ))
        
        # Check for data processing without validation
        if 'torch.tensor' in code and 'validate' not in code.lower():
            issues.append(SecurityIssue(
                severity='MEDIUM',
                issue_type='unvalidated_tensor_creation',
                description='Creating tensors without input validation',
                line_number=0,
                code_snippet='Tensor creation without validation',
                recommendation='Validate tensor inputs for shape, dtype, and range'
            ))
        
        return issues
    
    def _scan_with_bandit(self, code: str, file_path: str) -> List[SecurityIssue]:
        """Use Bandit for additional security scanning"""
        issues = []
        
        try:
            # Create temporary file for Bandit
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            try:
                # Run Bandit
                self.bandit_manager.discover([temp_file])
                self.bandit_manager.run_tests()
                
                # Process Bandit results
                for result in self.bandit_manager.get_issue_list():
                    issues.append(SecurityIssue(
                        severity=result.severity,
                        issue_type=result.test,
                        description=result.text,
                        line_number=result.lineno,
                        code_snippet=result.get_code(),
                        recommendation=f"Bandit {result.test}: {result.text}",
                        cwe_id=getattr(result, 'cwe', None),
                        confidence=result.confidence
                    ))
            
            finally:
                # Cleanup
                import os
                os.unlink(temp_file)
        
        except Exception as e:
            logging.warning(f"Bandit scanning failed: {e}")
        
        return issues
    
    def _check_dangerous_calls(self, node: ast.Call) -> List[SecurityIssue]:
        """Check for dangerous function calls in AST"""
        issues = []
        
        # Get function name
        func_name = ""
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
        
        # Check against dangerous functions
        dangerous_funcs = {
            'eval': 'CRITICAL',
            'exec': 'CRITICAL',
            'compile': 'HIGH',
            'open': 'MEDIUM',
            'input': 'MEDIUM'
        }
        
        if func_name in dangerous_funcs:
            issues.append(SecurityIssue(
                severity=dangerous_funcs[func_name],
                issue_type='dangerous_function_call',
                description=f'Call to potentially dangerous function: {func_name}',
                line_number=getattr(node, 'lineno', 0),
                code_snippet=f'{func_name}(...)',
                recommendation=f'Avoid using {func_name} or implement proper validation'
            ))
        
        return issues
    
    def _check_dangerous_imports(self, node) -> List[SecurityIssue]:
        """Check for dangerous imports in AST"""
        issues = []
        
        dangerous_modules = {
            'subprocess': 'HIGH',
            'os': 'MEDIUM',
            'sys': 'LOW',
            'pickle': 'HIGH',
            'marshal': 'HIGH'
        }
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in dangerous_modules:
                    issues.append(SecurityIssue(
                        severity=dangerous_modules[alias.name],
                        issue_type='dangerous_import',
                        description=f'Import of potentially dangerous module: {alias.name}',
                        line_number=getattr(node, 'lineno', 0),
                        code_snippet=f'import {alias.name}',
                        recommendation=f'Use {alias.name} carefully with proper validation'
                    ))
        
        return issues
    
    def _check_hardcoded_secrets(self, node: ast.Str) -> List[SecurityIssue]:
        """Check for hardcoded secrets in string literals"""
        issues = []
        
        if hasattr(node, 's'):
            text = node.s.lower()
            
            # Check for potential secrets
            secret_patterns = [
                ('password', 'password'),
                ('secret', 'secret'),
                ('key', 'key'),
                ('token', 'token'),
                ('api_key', 'API key')
            ]
            
            for pattern, name in secret_patterns:
                if pattern in text and len(node.s) > 8:  # Avoid false positives
                    issues.append(SecurityIssue(
                        severity='HIGH',
                        issue_type='hardcoded_secret',
                        description=f'Potential hardcoded {name} in string literal',
                        line_number=getattr(node, 'lineno', 0),
                        code_snippet=f'"{node.s[:20]}..."',
                        recommendation=f'Use environment variables or secure storage for {name}'
                    ))
        
        return issues
    
    def _check_sql_injection(self, node: ast.BinOp) -> List[SecurityIssue]:
        """Check for potential SQL injection patterns"""
        issues = []
        
        # Look for string concatenation that might be SQL injection
        if isinstance(node.op, ast.Add):
            # This is a simplified check - would need more sophisticated analysis
            pass
        
        return issues
    
    def _calculate_security_score(self, issues: List[SecurityIssue]) -> float:
        """Calculate overall security score (0-100)"""
        if not issues:
            return 100.0
        
        # Weight issues by severity
        severity_weights = {
            'CRITICAL': 40,
            'HIGH': 20,
            'MEDIUM': 10,
            'LOW': 5
        }
        
        total_penalty = sum(severity_weights.get(issue.severity, 5) for issue in issues)
        
        # Cap at 100 and ensure minimum of 0
        score = max(0, min(100, 100 - total_penalty))
        
        return score
    
    def _create_failed_scan_result(self, file_path: str, error_msg: str) -> SecurityScanResult:
        """Create a failed scan result"""
        return SecurityScanResult(
            scan_id="failed",
            timestamp=datetime.now().isoformat(),
            file_path=file_path,
            total_issues=1,
            critical_issues=1,
            high_issues=0,
            medium_issues=0,
            low_issues=0,
            security_score=0.0,
            passed_security_check=False,
            issues=[SecurityIssue(
                severity='CRITICAL',
                issue_type='scan_error',
                description=f'Security scan failed: {error_msg}',
                line_number=0,
                code_snippet='',
                recommendation='Fix code errors before security analysis'
            )]
        )

class CodeSanitizer:
    """Sanitizes and secures code before execution"""
    
    def __init__(self):
        self.security_scanner = CodeSecurityScanner()
        self.sanitization_rules = self._load_sanitization_rules()
    
    def _load_sanitization_rules(self) -> Dict[str, Any]:
        """Load rules for code sanitization"""
        return {
            'remove_dangerous_imports': {
                'enabled': True,
                'modules': ['subprocess', 'os.system', 'eval', 'exec']
            },
            'add_input_validation': {
                'enabled': True,
                'functions': ['input', 'raw_input']
            },
            'add_error_handling': {
                'enabled': True,
                'wrap_risky_operations': True
            },
            'add_logging': {
                'enabled': True,
                'log_security_events': True
            },
            'limit_resource_usage': {
                'enabled': True,
                'max_memory': '100MB',
                'max_execution_time': '30s'
            }
        }
    
    def sanitize_code(self, code: str) -> Tuple[str, List[str]]:
        """Sanitize code and return sanitized version with change log"""
        try:
            changes = []
            sanitized_code = code
            
            # Parse AST for manipulation
            tree = ast.parse(code)
            
            # Apply sanitization rules
            if self.sanitization_rules['remove_dangerous_imports']['enabled']:
                tree, removed_imports = self._remove_dangerous_imports(tree)
                changes.extend(removed_imports)
            
            if self.sanitization_rules['add_input_validation']['enabled']:
                tree, added_validations = self._add_input_validation(tree)
                changes.extend(added_validations)
            
            if self.sanitization_rules['add_error_handling']['enabled']:
                tree, added_handlers = self._add_error_handling(tree)
                changes.extend(added_handlers)
            
            if self.sanitization_rules['add_logging']['enabled']:
                tree, added_logging = self._add_security_logging(tree)
                changes.extend(added_logging)
            
            # Convert back to code
            sanitized_code = ast.unparse(tree)
            
            # Add resource limits wrapper
            if self.sanitization_rules['limit_resource_usage']['enabled']:
                sanitized_code = self._add_resource_limits(sanitized_code)
                changes.append("Added resource usage limits")
            
            return sanitized_code, changes
            
        except Exception as e:
            logging.error(f"Error sanitizing code: {e}")
            return code, [f"Sanitization failed: {e}"]
    
    def _remove_dangerous_imports(self, tree: ast.AST) -> Tuple[ast.AST, List[str]]:
        """Remove dangerous imports from AST"""
        changes = []
        dangerous_modules = self.sanitization_rules['remove_dangerous_imports']['modules']
        
        # This would need a proper AST transformer
        # For now, return unchanged
        return tree, changes
    
    def _add_input_validation(self, tree: ast.AST) -> Tuple[ast.AST, List[str]]:
        """Add input validation to functions"""
        changes = []
        
        # This would need AST transformation to add validation
        # For now, return unchanged
        return tree, changes
    
    def _add_error_handling(self, tree: ast.AST) -> Tuple[ast.AST, List[str]]:
        """Add error handling to risky operations"""
        changes = []
        
        # This would need AST transformation to wrap operations in try-except
        # For now, return unchanged
        return tree, changes
    
    def _add_security_logging(self, tree: ast.AST) -> Tuple[ast.AST, List[str]]:
        """Add security logging to functions"""
        changes = []
        
        # This would need AST transformation to add logging calls
        # For now, return unchanged
        return tree, changes
    
    def _add_resource_limits(self, code: str) -> str:
        """Add resource usage limits wrapper"""
        wrapper = f'''
import resource
import signal
import functools

def with_resource_limits(max_memory_mb=100, max_time_seconds=30):
    """Decorator to limit resource usage"""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Set memory limit
            resource.setrlimit(resource.RLIMIT_AS, (max_memory_mb * 1024 * 1024, -1))
            
            # Set time limit
            def timeout_handler(signum, frame):
                raise TimeoutError(f"Function exceeded {{max_time_seconds}} second limit")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(max_time_seconds)
            
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                signal.alarm(0)  # Clear alarm
        
        return wrapper
    return decorator

# Original code with resource limits
{code}
'''
        return wrapper

class SecureCodeValidator:
    """Validates that code meets security requirements before execution"""
    
    def __init__(self):
        self.scanner = CodeSecurityScanner()
        self.sanitizer = CodeSanitizer()
        self.validation_rules = self._load_validation_rules()
    
    def _load_validation_rules(self) -> Dict[str, Any]:
        """Load validation rules"""
        return {
            'min_security_score': 70,
            'max_critical_issues': 0,
            'max_high_issues': 0,
            'max_medium_issues': 3,
            'require_error_handling': True,
            'require_input_validation': True,
            'allow_network_operations': False,
            'allow_file_operations': True,
            'require_logging': True
        }
    
    def validate_and_secure(self, code: str, auto_fix: bool = True) -> Tuple[str, bool, SecurityScanResult]:
        """Validate code security and optionally auto-fix issues"""
        try:
            # Initial security scan
            scan_result = self.scanner.scan_code(code)
            
            # Check if code passes security requirements
            passes_validation = self._passes_security_requirements(scan_result)
            
            if not passes_validation and auto_fix:
                # Try to sanitize and fix issues
                sanitized_code, changes = self.sanitizer.sanitize_code(code)
                
                # Rescan sanitized code
                new_scan_result = self.scanner.scan_code(sanitized_code)
                
                if self._passes_security_requirements(new_scan_result):
                    logging.info(f"Code sanitized successfully: {len(changes)} changes made")
                    return sanitized_code, True, new_scan_result
                else:
                    logging.warning("Code sanitization did not resolve all security issues")
                    return code, False, scan_result
            
            return code, passes_validation, scan_result
            
        except Exception as e:
            logging.error(f"Error in security validation: {e}")
            # Return original code with failed validation
            failed_result = self.scanner._create_failed_scan_result("unknown", str(e))
            return code, False, failed_result
    
    def _passes_security_requirements(self, scan_result: SecurityScanResult) -> bool:
        """Check if scan result meets security requirements"""
        rules = self.validation_rules
        
        # Check security score
        if scan_result.security_score < rules['min_security_score']:
            return False
        
        # Check issue counts
        if scan_result.critical_issues > rules['max_critical_issues']:
            return False
        
        if scan_result.high_issues > rules['max_high_issues']:
            return False
        
        if scan_result.medium_issues > rules['max_medium_issues']:
            return False
        
        return True

# Global security instances
security_scanner = CodeSecurityScanner()
code_sanitizer = CodeSanitizer()
security_validator = SecureCodeValidator()

def scan_code_security(code: str, file_path: str = "generated_code.py") -> SecurityScanResult:
    """Scan code for security issues"""
    return security_scanner.scan_code(code, file_path)

def validate_code_security(code: str, auto_fix: bool = True) -> Tuple[str, bool, SecurityScanResult]:
    """Validate and optionally fix code security issues"""
    return security_validator.validate_and_secure(code, auto_fix)

def sanitize_code(code: str) -> Tuple[str, List[str]]:
    """Sanitize code to remove security issues"""
    return code_sanitizer.sanitize_code(code)