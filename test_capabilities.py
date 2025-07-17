#!/usr/bin/env python3
"""Test Jigyasa's actual capabilities"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from jigyasa.adaptive.hardware_detector import HardwareDetector
from jigyasa.autonomous.safe_code_security import CodeSecurityScanner

def test_hardware_detection():
    """Test hardware detection capabilities"""
    print("\n🔧 Testing Hardware Detection...")
    detector = HardwareDetector()
    specs = detector.detect_hardware()
    
    print(f"CPU: {specs.cpu_brand} ({specs.cpu_cores} cores)")
    print(f"RAM: {specs.total_ram:.1f} GB")
    print(f"GPU: {'Yes' if specs.has_gpu else 'No'}")
    print(f"Performance Class: {specs.performance_class}")
    return specs

def test_code_security():
    """Test code security scanning"""
    print("\n🔒 Testing Code Security Scanner...")
    scanner = CodeSecurityScanner()
    
    # Test with unsafe code
    unsafe_code = """
import os
password = "secret123"
os.system(f"echo {password}")
eval(input("Enter code: "))
"""
    
    issues = scanner.scan_code(unsafe_code, "test.py")
    print(f"Found {len(issues)} security issues")
    for issue in issues[:3]:  # Show first 3
        print(f"  - {issue.severity}: {issue.description}")
    return issues

def test_model_capabilities():
    """Test basic model functionality"""
    print("\n🧠 Testing Model Capabilities...")
    try:
        from jigyasa.core.model import create_jigyasa_model
        model = create_jigyasa_model(d_model=256, n_heads=8, n_layers=4)
        print("✅ Model created successfully")
        print(f"Model type: {type(model).__name__}")
        return True
    except Exception as e:
        print(f"❌ Model creation failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 Testing Jigyasa Capabilities")
    print("=" * 50)
    
    # Test each component
    hardware = test_hardware_detection()
    security = test_code_security()
    model = test_model_capabilities()
    
    print("\n📊 Summary:")
    print(f"- Hardware Detection: {'✅ Working' if hardware else '❌ Failed'}")
    print(f"- Security Scanner: {'✅ Working' if security else '❌ Failed'}")
    print(f"- Model Creation: {'✅ Working' if model else '❌ Failed'}")