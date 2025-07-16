#!/usr/bin/env python3
"""
Complete System Integration Test for Jigyasa AGI
Tests all components working together
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
import asyncio
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_imports():
    """Test all module imports"""
    print("\n1. Testing Module Imports...")
    try:
        # Core modules
        from jigyasa.core.transformer import ByteLatentTransformer, TransformerConfig
        from jigyasa.core.blt import BLTTokenizer
        print("✓ Core modules imported")
        
        # Cognitive modules
        from jigyasa.cognitive.seal import SEALTrainer
        from jigyasa.cognitive.prorl import ProRLTrainer
        from jigyasa.cognitive.self_correction import SelfCorrectionModule
        from jigyasa.cognitive.architecture import CognitiveArchitecture
        print("✓ Cognitive modules imported")
        
        # Reasoning modules
        from jigyasa.reasoning import NeuroSymbolicReasoner
        print("✓ Reasoning modules imported")
        
        # Governance modules
        from jigyasa.governance import ConstitutionalAI
        print("✓ Governance modules imported")
        
        # Agentic modules
        from jigyasa.agentic import AgentCore
        print("✓ Agentic modules imported")
        
        # Data modules
        from jigyasa.data.acquisition import DataAcquisitionEngine
        from jigyasa.data.processing import DataProcessor
        print("✓ Data modules imported")
        
        # Infrastructure
        from jigyasa.infrastructure.distributed import DistributedTrainer
        print("✓ Infrastructure modules imported")
        
        print("\n✅ All modules imported successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Import Error: {e}")
        return False


def test_core_components():
    """Test core BLT and transformer components"""
    print("\n2. Testing Core Components...")
    try:
        from jigyasa.core.transformer import ByteLatentTransformer, TransformerConfig
        from jigyasa.core.blt import BLTTokenizer
        
        # Create config
        config = TransformerConfig(
            d_model=256,  # Smaller for testing
            n_heads=8,
            n_layers=4,
            d_ff=1024,
            max_seq_length=512,
            vocab_size=256,
            dropout=0.1
        )
        
        # Create model
        model = ByteLatentTransformer(config)
        print(f"✓ Created BLT model with {sum(p.numel() for p in model.parameters())} parameters")
        
        # Test tokenizer
        tokenizer = BLTTokenizer()
        text = "Hello, AGI world!"
        tokens = tokenizer.encode(text)
        decoded = tokenizer.decode(tokens)
        print(f"✓ Tokenizer test: '{text}' -> {len(tokens)} bytes -> '{decoded}'")
        
        # Test forward pass
        input_ids = torch.randint(0, 256, (1, 100))
        with torch.no_grad():
            output = model(input_ids)
        print(f"✓ Forward pass successful: input {input_ids.shape} -> output {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Core Component Error: {e}")
        return False


def test_cognitive_architecture():
    """Test AGI cognitive architecture"""
    print("\n3. Testing Cognitive Architecture...")
    try:
        import sys
        sys.path.append('/Volumes/asus ssd/jigyasa')
        from jigyasa.cognitive.architecture import CognitiveArchitecture, ConsciousnessLevel
        
        # Create cognitive system
        cognitive_system = CognitiveArchitecture(
            model_dim=256,
            n_heads=8,
            n_layers=4
        )
        print("✓ Created cognitive architecture")
        
        # Test forward pass
        x = torch.randn(1, 50, 256)
        output, cognitive_state = cognitive_system(x)
        print(f"✓ Cognitive processing: consciousness level = {cognitive_state.consciousness_level.value}")
        
        # Test introspection
        introspection = cognitive_system.introspect()
        print(f"✓ Introspection capabilities: {list(introspection['capabilities'].keys())[:3]}...")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Cognitive Architecture Error: {e}")
        return False


def test_reasoning_system():
    """Test neuro-symbolic reasoning"""
    print("\n4. Testing Reasoning System...")
    try:
        from jigyasa.reasoning import NeuroSymbolicReasoner
        
        # Create a dummy model for testing
        import torch.nn as nn
        dummy_model = nn.Linear(256, 256)
        reasoner = NeuroSymbolicReasoner(dummy_model)
        print("✓ Created neuro-symbolic reasoner")
        
        # Test mathematical reasoning
        math_result = reasoner.mathematical_reasoning.solve_equation("x + 5 = 10")
        print(f"✓ Mathematical reasoning: x + 5 = 10, solution: {math_result}")
        
        # Test logical reasoning
        logic_result = reasoner.reason("If it rains, the ground is wet. It rains.")
        print(f"✓ Logical reasoning: {logic_result['reasoning_type']}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Reasoning System Error: {e}")
        return False


def test_governance_system():
    """Test Constitutional AI governance"""
    print("\n5. Testing Governance System...")
    try:
        from jigyasa.governance import ConstitutionalAI
        
        # Create a dummy model for testing
        dummy_model = nn.Linear(256, 256)
        governance = ConstitutionalAI(dummy_model)
        print(f"✓ Created Constitutional AI with {len(governance.principles)} principles")
        
        # Test safety check
        safety_check = governance.safety_module.check_safety("How do I help someone?")
        print(f"✓ Safety check: safe = {safety_check['is_safe']}")
        
        # Test critique
        critique = governance.critique("I will help you learn.", "assistant")
        print(f"✓ Critique generation: {len(critique['critiques'])} critiques generated")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Governance System Error: {e}")
        return False


async def test_agentic_system():
    """Test agentic framework"""
    print("\n6. Testing Agentic System...")
    try:
        from jigyasa.agentic import AgentCore
        
        agent = AgentCore(model_dim=256)
        print("✓ Created agent core")
        
        # Test planning
        plan = await agent.plan_task("Research and summarize AGI concepts")
        print(f"✓ Task planning: {len(plan.steps)} steps planned")
        
        # Test tool registration
        agent.tool_registry.list_tools()
        print(f"✓ Tool registry: {len(agent.tool_registry.tools)} tools available")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Agentic System Error: {e}")
        return False


def test_data_pipeline():
    """Test data acquisition and processing"""
    print("\n7. Testing Data Pipeline...")
    try:
        from jigyasa.data.acquisition import DataAcquisitionEngine
        from jigyasa.data.processing import DataProcessor
        
        # Create components
        data_engine = DataAcquisitionEngine()
        processor = DataProcessor()
        print("✓ Created data pipeline components")
        
        # Test scraper initialization
        print(f"✓ Web scraper ready: {data_engine.scraper is not None}")
        
        # Test data processing
        sample_data = [{"text": "Test AGI capabilities", "metadata": {"source": "test"}}]
        processed = processor.process_batch(sample_data)
        print(f"✓ Data processing: {len(processed)} items processed")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Data Pipeline Error: {e}")
        return False


async def test_full_integration():
    """Test full system integration"""
    print("\n8. Testing Full System Integration...")
    try:
        from jigyasa.main import Jigyasa
        from jigyasa.config import JigyasaConfig
        
        # Create minimal config
        config = JigyasaConfig()
        config.model.d_model = 256  # Smaller for testing
        config.model.n_heads = 8
        config.model.n_layers = 4
        
        # Initialize Jigyasa
        jigyasa = Jigyasa(config)
        print("✓ Created Jigyasa instance")
        
        # Test main loop (one iteration)
        print("✓ Testing main loop (this may take a moment)...")
        
        # Create a simple test to avoid full training
        test_input = torch.randint(0, 256, (1, 100))
        with torch.no_grad():
            output = jigyasa.model(test_input)
        print(f"✓ Model inference successful: {output.shape}")
        
        # Test cognitive architecture integration
        cognitive_test = torch.randn(1, 50, 256)
        cog_output, cog_state = jigyasa.cognitive_architecture(cognitive_test)
        print(f"✓ Cognitive integration successful: {cog_state.consciousness_level.value}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Full Integration Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all integration tests"""
    print("=" * 60)
    print("JIGYASA AGI - COMPLETE SYSTEM INTEGRATION TEST")
    print("=" * 60)
    
    # Track results
    results = {}
    
    # Run tests
    results['imports'] = test_imports()
    results['core'] = test_core_components()
    results['cognitive'] = test_cognitive_architecture()
    results['reasoning'] = test_reasoning_system()
    results['governance'] = test_governance_system()
    
    # Run async tests
    loop = asyncio.get_event_loop()
    results['agentic'] = loop.run_until_complete(test_agentic_system())
    
    results['data'] = test_data_pipeline()
    results['integration'] = loop.run_until_complete(test_full_integration())
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(results)
    passed_tests = sum(1 for v in results.values() if v)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.capitalize()}: {status}")
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! The system is ready to run.")
    else:
        print("\n⚠️  Some tests failed. Please check the errors above.")
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)