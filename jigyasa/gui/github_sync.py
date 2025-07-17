#!/usr/bin/env python3
"""
GitHub Auto-Sync - Automatically sync changes to GitHub
"""

import git
import os
import time
import threading
import queue
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import hashlib
import json

class GitHubAutoSync:
    """Automatically sync code changes to GitHub"""
    
    def __init__(self, repo_path: str, auto_sync: bool = True, sync_interval: int = 300):
        self.repo_path = Path(repo_path)
        self.auto_sync = auto_sync
        self.sync_interval = sync_interval
        self.sync_queue = queue.Queue()
        self.running = False
        
        # Initialize git repo
        try:
            self.repo = git.Repo(self.repo_path)
        except git.InvalidGitRepositoryError:
            print(f"⚠️  {repo_path} is not a git repository")
            self.repo = None
            
        # Track file changes
        self.file_hashes = {}
        self.change_log = []
        
        if auto_sync and self.repo:
            self._start_sync_loop()
            
    def _start_sync_loop(self):
        """Start automatic sync loop"""
        self.running = True
        
        # File watcher thread
        watcher_thread = threading.Thread(target=self._watch_files)
        watcher_thread.daemon = True
        watcher_thread.start()
        
        # Sync thread
        sync_thread = threading.Thread(target=self._sync_loop)
        sync_thread.daemon = True
        sync_thread.start()
        
    def _watch_files(self):
        """Watch for file changes"""
        while self.running:
            changes = self._detect_changes()
            
            if changes:
                # Queue sync with descriptive message
                change_summary = self._summarize_changes(changes)
                self.sync_queue.put({
                    'changes': changes,
                    'summary': change_summary,
                    'timestamp': datetime.now()
                })
                
            time.sleep(10)  # Check every 10 seconds
            
    def _detect_changes(self) -> List[Dict[str, Any]]:
        """Detect file changes"""
        changes = []
        
        # Python files to watch
        py_files = list(self.repo_path.rglob("*.py"))
        
        for file_path in py_files:
            # Skip __pycache__ and test files
            if '__pycache__' in str(file_path) or '.git' in str(file_path):
                continue
                
            try:
                # Calculate file hash
                with open(file_path, 'rb') as f:
                    file_hash = hashlib.sha256(f.read()).hexdigest()
                    
                # Check if changed
                rel_path = str(file_path.relative_to(self.repo_path))
                
                if rel_path in self.file_hashes:
                    if self.file_hashes[rel_path] != file_hash:
                        changes.append({
                            'file': rel_path,
                            'type': 'modified',
                            'old_hash': self.file_hashes[rel_path],
                            'new_hash': file_hash
                        })
                else:
                    changes.append({
                        'file': rel_path,
                        'type': 'added',
                        'new_hash': file_hash
                    })
                    
                self.file_hashes[rel_path] = file_hash
                
            except Exception as e:
                print(f"Error checking {file_path}: {e}")
                
        # Check for deleted files
        for rel_path in list(self.file_hashes.keys()):
            full_path = self.repo_path / rel_path
            if not full_path.exists():
                changes.append({
                    'file': rel_path,
                    'type': 'deleted',
                    'old_hash': self.file_hashes[rel_path]
                })
                del self.file_hashes[rel_path]
                
        return changes
        
    def _summarize_changes(self, changes: List[Dict[str, Any]]) -> str:
        """Create a summary of changes"""
        if not changes:
            return "No changes"
            
        # Count change types
        added = sum(1 for c in changes if c['type'] == 'added')
        modified = sum(1 for c in changes if c['type'] == 'modified')
        deleted = sum(1 for c in changes if c['type'] == 'deleted')
        
        parts = []
        if added:
            parts.append(f"Added {added} file(s)")
        if modified:
            parts.append(f"Modified {modified} file(s)")
        if deleted:
            parts.append(f"Deleted {deleted} file(s)")
            
        return ", ".join(parts)
        
    def _sync_loop(self):
        """Main sync loop"""
        while self.running:
            # Process sync queue
            synced = False
            pending_changes = []
            
            # Collect all pending changes
            while not self.sync_queue.empty():
                try:
                    sync_data = self.sync_queue.get(timeout=1)
                    pending_changes.extend(sync_data['changes'])
                except queue.Empty:
                    break
                    
            # Sync if we have changes
            if pending_changes:
                summary = self._summarize_changes(pending_changes)
                result = self.sync_changes(f"🤖 Auto-sync: {summary}")
                if result['success']:
                    synced = True
                    
            # Wait before next sync
            time.sleep(self.sync_interval if not synced else 60)
            
    def sync_changes(self, message: str = "Auto-sync from JIGYASA GUI") -> Dict[str, Any]:
        """Sync changes to GitHub"""
        if not self.repo:
            return {'success': False, 'error': 'No git repository'}
            
        try:
            # Check for changes
            if self.repo.is_dirty() or self.repo.untracked_files:
                # Add all changes
                self.repo.git.add(A=True)
                
                # Create detailed commit message
                changed_files = [item.a_path for item in self.repo.index.diff("HEAD")]
                
                commit_message = f"{message}\n\n"
                if changed_files:
                    commit_message += "Changed files:\n"
                    for file in changed_files[:10]:  # Limit to 10 files
                        commit_message += f"  - {file}\n"
                    if len(changed_files) > 10:
                        commit_message += f"  ... and {len(changed_files) - 10} more\n"
                        
                commit_message += f"\nTimestamp: {datetime.now().isoformat()}"
                
                # Commit
                self.repo.index.commit(commit_message)
                
                # Push to remote
                origin = self.repo.remote('origin')
                push_info = origin.push()
                
                # Log the sync
                self.change_log.append({
                    'timestamp': datetime.now().isoformat(),
                    'message': message,
                    'files_changed': len(changed_files),
                    'commit_hash': self.repo.head.commit.hexsha
                })
                
                return {
                    'success': True,
                    'commit': self.repo.head.commit.hexsha,
                    'files_changed': len(changed_files),
                    'message': message
                }
            else:
                return {
                    'success': True,
                    'message': 'No changes to sync'
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
            
    def get_sync_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get sync history"""
        return self.change_log[-limit:]
        
    def enable_auto_sync(self):
        """Enable automatic syncing"""
        self.auto_sync = True
        if not self.running:
            self._start_sync_loop()
            
    def disable_auto_sync(self):
        """Disable automatic syncing"""
        self.auto_sync = False
        self.running = False
        
    def get_status(self) -> Dict[str, Any]:
        """Get sync status"""
        return {
            'enabled': self.auto_sync,
            'running': self.running,
            'repo_path': str(self.repo_path),
            'has_changes': self.repo.is_dirty() if self.repo else False,
            'last_sync': self.change_log[-1] if self.change_log else None,
            'total_syncs': len(self.change_log)
        }
        
    def configure_remote(self, remote_url: str, token: Optional[str] = None):
        """Configure git remote with authentication"""
        if not self.repo:
            return False
            
        try:
            # Update remote URL with token if provided
            if token:
                # Parse URL to insert token
                if 'github.com' in remote_url:
                    parts = remote_url.split('github.com')
                    remote_url = f"{parts[0]}{token}@github.com{parts[1]}"
                    
            # Set remote URL
            origin = self.repo.remote('origin')
            origin.set_url(remote_url)
            
            return True
        except Exception as e:
            print(f"Error configuring remote: {e}")
            return False