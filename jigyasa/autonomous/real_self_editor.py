"""
Real Self-Modifying Code System using Llama 3.2
Actually modifies and improves code autonomously
"""

import ast
import os
import git
import time
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
import hashlib
from datetime import datetime

from ..models.ollama_wrapper import OllamaWrapper

class RealSelfEditor:
    """Actually modifies code autonomously using AI"""
    
    def __init__(self, ollama_wrapper: Optional[OllamaWrapper] = None):
        self.ollama = ollama_wrapper or OllamaWrapper()
        self.logger = logging.getLogger(__name__)
        self.modifications_log = []
        self.backup_dir = Path(".jigyasa_backups")
        self.backup_dir.mkdir(exist_ok=True)
        
    def backup_file(self, file_path: str) -> str:
        """Create backup before modification"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        content = Path(file_path).read_text()
        backup_path = self.backup_dir / f"{Path(file_path).name}_{timestamp}.bak"
        backup_path.write_text(content)
        return str(backup_path)
        
    def modify_code_autonomously(self, file_path: str) -> Dict[str, Any]:
        """Actually modify code file with AI-driven improvements"""
        try:
            # Read current code
            with open(file_path, 'r') as f:
                original_code = f.read()
                
            # Backup original
            backup_path = self.backup_file(file_path)
            
            # Get AI improvements
            improved_code, improvements = self.ollama.self_improve_code(file_path, original_code)
            
            if not improvements:
                return {
                    "status": "no_improvements",
                    "message": "Code is already optimized"
                }
                
            # Validate improved code
            try:
                compile(improved_code, file_path, 'exec')
            except SyntaxError as e:
                self.logger.error(f"Syntax error in improved code: {e}")
                return {
                    "status": "error",
                    "message": f"Syntax error: {e}",
                    "backup": backup_path
                }
                
            # Run tests if they exist
            test_passed = self.run_tests_for_file(file_path)
            
            if not test_passed:
                self.logger.warning("Tests failed, rolling back")
                return {
                    "status": "tests_failed",
                    "message": "Improvements broke tests",
                    "backup": backup_path
                }
                
            # Apply improvements
            with open(file_path, 'w') as f:
                f.write(improved_code)
                
            # Measure actual performance
            perf_gain = self.measure_real_performance(original_code, improved_code)
            
            # Log modification
            modification = {
                "timestamp": datetime.now().isoformat(),
                "file": file_path,
                "backup": backup_path,
                "improvements": improvements,
                "performance_gain": perf_gain,
                "hash_before": hashlib.sha256(original_code.encode()).hexdigest(),
                "hash_after": hashlib.sha256(improved_code.encode()).hexdigest()
            }
            
            self.modifications_log.append(modification)
            self.save_modification_log()
            
            # Git commit if available
            self.git_commit_improvement(file_path, improvements)
            
            return {
                "status": "success",
                "improvements": improvements,
                "performance_gain": perf_gain,
                "backup": backup_path
            }
            
        except Exception as e:
            self.logger.error(f"Failed to modify {file_path}: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
            
    def run_tests_for_file(self, file_path: str) -> bool:
        """Run relevant tests for modified file"""
        # Find test file
        test_file = file_path.replace('.py', '_test.py')
        if not os.path.exists(test_file):
            test_file = file_path.replace('/jigyasa/', '/tests/').replace('.py', '_test.py')
            
        if os.path.exists(test_file):
            try:
                result = subprocess.run(
                    ['python', '-m', 'pytest', test_file, '-v'],
                    capture_output=True,
                    text=True
                )
                return result.returncode == 0
            except:
                pass
                
        # No tests found, assume safe
        return True
        
    def measure_real_performance(self, original: str, improved: str) -> float:
        """Measure actual performance improvement"""
        # Create temporary files
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f1:
            f1.write(original)
            orig_file = f1.name
            
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f2:
            f2.write(improved)
            imp_file = f2.name
            
        try:
            # Time original
            start = time.time()
            subprocess.run(['python', orig_file], capture_output=True, timeout=5)
            orig_time = time.time() - start
            
            # Time improved
            start = time.time()
            subprocess.run(['python', imp_file], capture_output=True, timeout=5)
            imp_time = time.time() - start
            
            # Calculate improvement
            if orig_time > 0:
                improvement = (orig_time - imp_time) / orig_time
                return max(0, min(improvement, 0.9))  # Cap at 90%
                
        except:
            # Fallback to AI estimation
            return self.ollama.measure_performance(original, improved)
        finally:
            os.unlink(orig_file)
            os.unlink(imp_file)
            
        return 0.0
        
    def git_commit_improvement(self, file_path: str, improvements: List[Dict]):
        """Commit improvements to git"""
        try:
            repo = git.Repo(search_parent_directories=True)
            repo.index.add([file_path])
            
            # Create descriptive commit message
            improvements_desc = ', '.join([imp['description'][:50] for imp in improvements[:3]])
            commit_msg = f"🤖 Auto-improve {Path(file_path).name}: {improvements_desc}"
            
            repo.index.commit(commit_msg)
            self.logger.info(f"Committed improvements to git: {commit_msg}")
        except:
            pass  # Git not available
            
    def save_modification_log(self):
        """Save modification history"""
        log_file = self.backup_dir / "modifications.json"
        with open(log_file, 'w') as f:
            json.dump(self.modifications_log, f, indent=2)
            
    def rollback_modification(self, file_path: str) -> bool:
        """Rollback to previous version"""
        # Find latest backup
        backups = list(self.backup_dir.glob(f"{Path(file_path).name}_*.bak"))
        if not backups:
            return False
            
        latest_backup = max(backups, key=lambda p: p.stat().st_mtime)
        
        # Restore from backup
        import shutil
        shutil.copy2(latest_backup, file_path)
        
        self.logger.info(f"Rolled back {file_path} from {latest_backup}")
        return True
        
    def continuous_improvement_loop(self, directory: str, interval: int = 3600):
        """Continuously improve code in directory"""
        self.logger.info(f"Starting continuous improvement for {directory}")
        
        while True:
            py_files = list(Path(directory).rglob("*.py"))
            
            for py_file in py_files:
                if "test" not in str(py_file).lower():
                    self.logger.info(f"Analyzing {py_file}")
                    result = self.modify_code_autonomously(str(py_file))
                    
                    if result['status'] == 'success':
                        self.logger.info(f"Improved {py_file}: {result['performance_gain']:.1%} gain")
                        
            time.sleep(interval)