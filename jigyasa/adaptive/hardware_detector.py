#!/usr/bin/env python3
"""
Hardware Detection and Adaptation System
Automatically detects system capabilities and adapts training parameters
"""

import psutil
import platform
import subprocess
import time
import json
import threading
import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import torch
import numpy as np

@dataclass
class HardwareSpecs:
    """System hardware specifications"""
    # CPU Information
    cpu_cores: int
    cpu_threads: int
    cpu_frequency: float  # GHz
    cpu_architecture: str
    cpu_brand: str
    
    # Memory Information
    total_ram: float  # GB
    available_ram: float  # GB
    ram_speed: Optional[float]  # MHz
    
    # GPU Information
    has_gpu: bool
    gpu_count: int
    gpu_memory: List[float]  # GB per GPU
    gpu_compute_capability: List[str]
    gpu_names: List[str]
    
    # Storage Information
    storage_type: str  # SSD, HDD, NVMe
    storage_speed: float  # MB/s
    available_storage: float  # GB
    
    # System Information
    os_type: str
    os_version: str
    python_version: str
    torch_version: str
    
    # Performance Metrics
    performance_class: str  # low, medium, high, extreme
    training_capability_score: float  # 0-100

@dataclass
class PerformanceMetrics:
    """Real-time performance metrics"""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: List[float]
    gpu_memory_usage: List[float]
    temperature: Dict[str, float]
    power_consumption: Optional[float]
    training_speed: float  # samples/second
    memory_efficiency: float  # MB/sample

class HardwareDetector:
    """Detects and monitors system hardware capabilities"""
    
    def __init__(self):
        self.specs = None
        self.performance_history = []
        self.monitoring_active = False
        self.monitor_thread = None
        
    def detect_hardware(self) -> HardwareSpecs:
        """Comprehensive hardware detection"""
        try:
            logging.info("🔍 Detecting system hardware...")
            
            # CPU Detection
            cpu_info = self._detect_cpu()
            
            # Memory Detection
            memory_info = self._detect_memory()
            
            # GPU Detection
            gpu_info = self._detect_gpu()
            
            # Storage Detection
            storage_info = self._detect_storage()
            
            # System Information
            system_info = self._detect_system()
            
            # Calculate performance metrics
            performance_class, capability_score = self._calculate_performance_class(
                cpu_info, memory_info, gpu_info, storage_info
            )
            
            specs = HardwareSpecs(
                # CPU
                cpu_cores=cpu_info['cores'],
                cpu_threads=cpu_info['threads'],
                cpu_frequency=cpu_info['frequency'],
                cpu_architecture=cpu_info['architecture'],
                cpu_brand=cpu_info['brand'],
                
                # Memory
                total_ram=memory_info['total'],
                available_ram=memory_info['available'],
                ram_speed=memory_info['speed'],
                
                # GPU
                has_gpu=gpu_info['has_gpu'],
                gpu_count=gpu_info['count'],
                gpu_memory=gpu_info['memory'],
                gpu_compute_capability=gpu_info['compute_capability'],
                gpu_names=gpu_info['names'],
                
                # Storage
                storage_type=storage_info['type'],
                storage_speed=storage_info['speed'],
                available_storage=storage_info['available'],
                
                # System
                os_type=system_info['os_type'],
                os_version=system_info['os_version'],
                python_version=system_info['python_version'],
                torch_version=system_info['torch_version'],
                
                # Performance
                performance_class=performance_class,
                training_capability_score=capability_score
            )
            
            self.specs = specs
            logging.info(f"✅ Hardware detection complete: {performance_class} class system")
            return specs
            
        except Exception as e:
            logging.error(f"❌ Hardware detection failed: {e}")
            return self._create_fallback_specs()
    
    def _detect_cpu(self) -> Dict[str, Any]:
        """Detect CPU specifications"""
        cpu_info = {
            'cores': psutil.cpu_count(logical=False),
            'threads': psutil.cpu_count(logical=True),
            'frequency': psutil.cpu_freq().max / 1000 if psutil.cpu_freq() else 2.0,
            'architecture': platform.machine(),
            'brand': platform.processor() or 'Unknown'
        }
        
        # Try to get more detailed CPU info on different platforms
        try:
            if platform.system() == "Darwin":  # macOS
                result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    cpu_info['brand'] = result.stdout.strip()
            elif platform.system() == "Linux":
                with open('/proc/cpuinfo', 'r') as f:
                    for line in f:
                        if 'model name' in line:
                            cpu_info['brand'] = line.split(':')[1].strip()
                            break
        except:
            pass
        
        return cpu_info
    
    def _detect_memory(self) -> Dict[str, Any]:
        """Detect memory specifications"""
        memory = psutil.virtual_memory()
        
        memory_info = {
            'total': memory.total / (1024**3),  # GB
            'available': memory.available / (1024**3),  # GB
            'speed': None
        }
        
        # Try to detect RAM speed
        try:
            if platform.system() == "Linux":
                result = subprocess.run(['dmidecode', '--type', 'memory'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    for line in result.stdout.split('\n'):
                        if 'Speed:' in line and 'MHz' in line:
                            speed_str = line.split(':')[1].strip()
                            memory_info['speed'] = float(speed_str.split()[0])
                            break
        except:
            pass
        
        return memory_info
    
    def _detect_gpu(self) -> Dict[str, Any]:
        """Detect GPU specifications"""
        gpu_info = {
            'has_gpu': False,
            'count': 0,
            'memory': [],
            'compute_capability': [],
            'names': []
        }
        
        try:
            # Check CUDA availability
            if torch.cuda.is_available():
                gpu_info['has_gpu'] = True
                gpu_info['count'] = torch.cuda.device_count()
                
                for i in range(gpu_info['count']):
                    # GPU memory
                    gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                    gpu_info['memory'].append(gpu_memory)
                    
                    # GPU name
                    gpu_name = torch.cuda.get_device_name(i)
                    gpu_info['names'].append(gpu_name)
                    
                    # Compute capability
                    props = torch.cuda.get_device_properties(i)
                    compute_cap = f"{props.major}.{props.minor}"
                    gpu_info['compute_capability'].append(compute_cap)
            
            # Check for MPS (Apple Silicon)
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                gpu_info['has_gpu'] = True
                gpu_info['count'] = 1
                gpu_info['memory'] = [8.0]  # Unified memory approximation
                gpu_info['names'] = ['Apple Silicon GPU']
                gpu_info['compute_capability'] = ['mps']
        
        except Exception as e:
            logging.warning(f"GPU detection failed: {e}")
        
        return gpu_info
    
    def _detect_storage(self) -> Dict[str, Any]:
        """Detect storage specifications"""
        storage_info = {
            'type': 'Unknown',
            'speed': 100.0,  # Default MB/s
            'available': 0.0
        }
        
        try:
            # Get available storage
            disk_usage = psutil.disk_usage('/')
            storage_info['available'] = disk_usage.free / (1024**3)  # GB
            
            # Try to determine storage type
            try:
                if platform.system() == "Linux":
                    # Check if SSD
                    result = subprocess.run(['lsblk', '-d', '-o', 'name,rota'], 
                                          capture_output=True, text=True)
                    if result.returncode == 0:
                        for line in result.stdout.split('\n')[1:]:
                            if line.strip():
                                parts = line.split()
                                if len(parts) >= 2 and parts[1] == '0':
                                    storage_info['type'] = 'SSD'
                                    storage_info['speed'] = 500.0  # Typical SSD speed
                                    break
                                else:
                                    storage_info['type'] = 'HDD'
                                    storage_info['speed'] = 100.0  # Typical HDD speed
                
                elif platform.system() == "Darwin":  # macOS
                    # Most modern Macs have SSDs
                    result = subprocess.run(['system_profiler', 'SPStorageDataType'], 
                                          capture_output=True, text=True)
                    if 'SSD' in result.stdout or 'Flash' in result.stdout:
                        storage_info['type'] = 'SSD'
                        storage_info['speed'] = 1000.0  # Fast NVMe on modern Macs
                    
            except:
                pass
            
        except Exception as e:
            logging.warning(f"Storage detection failed: {e}")
        
        return storage_info
    
    def _detect_system(self) -> Dict[str, Any]:
        """Detect system information"""
        return {
            'os_type': platform.system(),
            'os_version': platform.release(),
            'python_version': platform.python_version(),
            'torch_version': torch.__version__
        }
    
    def _calculate_performance_class(self, cpu_info, memory_info, gpu_info, storage_info) -> Tuple[str, float]:
        """Calculate system performance class and capability score"""
        score = 0.0
        
        # CPU Score (0-25 points)
        cpu_score = min(25, 
            (cpu_info['cores'] * 2) + 
            (cpu_info['frequency'] * 3) + 
            (cpu_info['threads'] * 0.5)
        )
        score += cpu_score
        
        # Memory Score (0-20 points)
        memory_score = min(20, memory_info['total'] * 2)
        score += memory_score
        
        # GPU Score (0-40 points)
        if gpu_info['has_gpu']:
            gpu_score = min(40, 
                (gpu_info['count'] * 10) + 
                (sum(gpu_info['memory']) * 3)
            )
        else:
            gpu_score = 0
        score += gpu_score
        
        # Storage Score (0-15 points)
        storage_multiplier = {
            'NVMe': 3.0,
            'SSD': 2.0,
            'HDD': 1.0,
            'Unknown': 1.5
        }
        storage_score = min(15, 
            (storage_info['speed'] / 100) * storage_multiplier.get(storage_info['type'], 1.5)
        )
        score += storage_score
        
        # Determine performance class
        if score >= 80:
            performance_class = "extreme"
        elif score >= 60:
            performance_class = "high"
        elif score >= 40:
            performance_class = "medium"
        else:
            performance_class = "low"
        
        return performance_class, score
    
    def _create_fallback_specs(self) -> HardwareSpecs:
        """Create fallback specs when detection fails"""
        return HardwareSpecs(
            cpu_cores=4,
            cpu_threads=8,
            cpu_frequency=2.5,
            cpu_architecture="x86_64",
            cpu_brand="Unknown CPU",
            total_ram=8.0,
            available_ram=4.0,
            ram_speed=None,
            has_gpu=False,
            gpu_count=0,
            gpu_memory=[],
            gpu_compute_capability=[],
            gpu_names=[],
            storage_type="Unknown",
            storage_speed=100.0,
            available_storage=50.0,
            os_type=platform.system(),
            os_version=platform.release(),
            python_version=platform.python_version(),
            torch_version=torch.__version__,
            performance_class="medium",
            training_capability_score=50.0
        )
    
    def start_monitoring(self, interval: float = 5.0):
        """Start real-time hardware monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop, 
            args=(interval,), 
            daemon=True
        )
        self.monitor_thread.start()
        logging.info("🔄 Started hardware monitoring")
    
    def stop_monitoring(self):
        """Stop hardware monitoring"""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logging.info("⏹️ Stopped hardware monitoring")
    
    def _monitoring_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                metrics = self._collect_performance_metrics()
                self.performance_history.append(metrics)
                
                # Keep only last 1000 entries
                if len(self.performance_history) > 1000:
                    self.performance_history = self.performance_history[-1000:]
                
                time.sleep(interval)
                
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(interval)
    
    def _collect_performance_metrics(self) -> PerformanceMetrics:
        """Collect current performance metrics"""
        # CPU and Memory
        cpu_usage = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        memory_usage = memory.percent
        
        # GPU metrics
        gpu_usage = []
        gpu_memory_usage = []
        
        if self.specs and self.specs.has_gpu:
            try:
                if torch.cuda.is_available():
                    for i in range(self.specs.gpu_count):
                        # GPU utilization (approximation)
                        gpu_usage.append(0.0)  # Would need nvidia-ml-py for real metrics
                        
                        # GPU memory usage
                        gpu_memory = torch.cuda.memory_allocated(i) / torch.cuda.max_memory_allocated(i) * 100
                        gpu_memory_usage.append(gpu_memory)
                        
            except Exception:
                pass
        
        # Temperature (if available)
        temperatures = {}
        try:
            temp_sensors = psutil.sensors_temperatures()
            for name, entries in temp_sensors.items():
                for entry in entries:
                    temperatures[f"{name}_{entry.label or 'temp'}"] = entry.current
        except:
            pass
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_usage,
            memory_usage=memory_usage,
            gpu_usage=gpu_usage,
            gpu_memory_usage=gpu_memory_usage,
            temperature=temperatures,
            power_consumption=None,  # Would need specialized hardware
            training_speed=0.0,  # Updated during training
            memory_efficiency=0.0  # Updated during training
        )
    
    def get_latest_metrics(self) -> Optional[PerformanceMetrics]:
        """Get the most recent performance metrics"""
        return self.performance_history[-1] if self.performance_history else None
    
    def get_average_metrics(self, duration_minutes: int = 10) -> Optional[PerformanceMetrics]:
        """Get average metrics over specified duration"""
        if not self.performance_history:
            return None
        
        cutoff_time = time.time() - (duration_minutes * 60)
        recent_metrics = [m for m in self.performance_history if m.timestamp >= cutoff_time]
        
        if not recent_metrics:
            return recent_metrics[-1] if self.performance_history else None
        
        # Calculate averages
        avg_cpu = sum(m.cpu_usage for m in recent_metrics) / len(recent_metrics)
        avg_memory = sum(m.memory_usage for m in recent_metrics) / len(recent_metrics)
        
        avg_gpu_usage = []
        avg_gpu_memory = []
        if recent_metrics[0].gpu_usage:
            for i in range(len(recent_metrics[0].gpu_usage)):
                avg_gpu_usage.append(
                    sum(m.gpu_usage[i] for m in recent_metrics) / len(recent_metrics)
                )
            for i in range(len(recent_metrics[0].gpu_memory_usage)):
                avg_gpu_memory.append(
                    sum(m.gpu_memory_usage[i] for m in recent_metrics) / len(recent_metrics)
                )
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_usage=avg_cpu,
            memory_usage=avg_memory,
            gpu_usage=avg_gpu_usage,
            gpu_memory_usage=avg_gpu_memory,
            temperature={},
            power_consumption=None,
            training_speed=sum(m.training_speed for m in recent_metrics) / len(recent_metrics),
            memory_efficiency=sum(m.memory_efficiency for m in recent_metrics) / len(recent_metrics)
        )
    
    def save_specs(self, file_path: str):
        """Save hardware specs to file"""
        if self.specs:
            with open(file_path, 'w') as f:
                json.dump(asdict(self.specs), f, indent=2)
    
    def load_specs(self, file_path: str) -> Optional[HardwareSpecs]:
        """Load hardware specs from file"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                self.specs = HardwareSpecs(**data)
                return self.specs
        except Exception as e:
            logging.warning(f"Could not load specs from {file_path}: {e}")
            return None

# Global hardware detector instance
hardware_detector = HardwareDetector()

def detect_system_hardware() -> HardwareSpecs:
    """Detect system hardware specifications"""
    return hardware_detector.detect_hardware()

def start_hardware_monitoring():
    """Start monitoring system performance"""
    hardware_detector.start_monitoring()

def get_hardware_specs() -> Optional[HardwareSpecs]:
    """Get current hardware specifications"""
    return hardware_detector.specs

def get_performance_metrics() -> Optional[PerformanceMetrics]:
    """Get latest performance metrics"""
    return hardware_detector.get_latest_metrics()