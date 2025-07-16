"""
Main Jigyasa System Integration
Coordinates all components for end-to-end AGI functionality
"""

import torch
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
from datetime import datetime
import random

from .config import JigyasaConfig
from .core.model import JigyasaModel, create_jigyasa_model
from .cognitive.seal import SEALTrainer
from .cognitive.prorl import ProRLTrainer
from .cognitive.self_correction import SelfCorrectionModule
from .cognitive.meta_learning import MetaLearningEngine
try:
    from .cognitive.architecture import CognitiveArchitecture
except ImportError:
    CognitiveArchitecture = None
try:
    from .data.data_engine import DataEngine
except ImportError:
    # Use STEM data engine if web scraping dependencies are missing
    from .data.stem_data_engine import DataEngine
try:
    from .data.preprocessing import DataPreprocessor
except ImportError:
    # Use simple preprocessor for STEM data
    from .data.simple_preprocessing import DataPreprocessor
from .compression.distillation import distill_model
from .compression.quantization import quantize_model_ptq
from .autonomous import make_autonomous, autonomous_wrapper, autonomous_system


class JigyasaSystem:
    """
    Main Jigyasa system that integrates all components
    """
    
    def __init__(self, config: JigyasaConfig = None):
        self.config = config or JigyasaConfig()
        
        # Initialize autonomous capabilities first
        make_autonomous()
        
        # Initialize logging
        self._setup_logging()
        
        # Core components
        self.model = None
        self.data_engine = None
        self.preprocessor = None
        self.seal_trainer = None
        self.prorl_trainer = None
        self.self_correction = None
        self.meta_learning = None
        self.cognitive_architecture = None
        
        # System state
        self.is_initialized = False
        self.training_phase = "none"  # none, prorl, seal, deployed
        
        logging.info("Jigyasa system initialized with autonomous capabilities")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(Path(self.config.log_dir) / 'jigyasa.log'),
                logging.StreamHandler()
            ]
        )
    
    @autonomous_wrapper
    def initialize(self, load_pretrained: Optional[str] = None):
        """
        Initialize all system components
        
        Args:
            load_pretrained: Path to pretrained model (optional)
        """
        logging.info("Initializing Jigyasa system components...")
        
        # Create or load model
        if load_pretrained:
            self.model = JigyasaModel.from_pretrained(load_pretrained)
            logging.info(f"Loaded pretrained model from {load_pretrained}")
        else:
            self.model = create_jigyasa_model(
                d_model=self.config.model.d_model,
                n_heads=self.config.model.n_heads,
                n_layers=self.config.model.n_layers,
                max_seq_length=self.config.model.max_seq_length
            )
            logging.info("Created new Jigyasa model from scratch")
        
        # Initialize data components
        self.data_engine = DataEngine(self.config.data)
        self.preprocessor = DataPreprocessor(self.config.data)
        
        # Initialize cognitive components
        # Create a separate model instance for ProRL to avoid PEFT conflicts
        prorl_model = create_jigyasa_model(
            d_model=self.config.model.d_model,
            n_heads=self.config.model.n_heads,
            n_layers=self.config.model.n_layers,
            max_seq_length=self.config.model.max_seq_length
        )
        # Don't load state dict if models are different types
        if not hasattr(self.model, 'tokenizer'):  # GPT2Wrapper has tokenizer
            prorl_model.load_state_dict(self.model.state_dict())
        
        self.seal_trainer = SEALTrainer(self.model, self.config.seal)
        self.prorl_trainer = ProRLTrainer(prorl_model, self.config.prorl)
        self.self_correction = SelfCorrectionModule(self.model)
        self.meta_learning = MetaLearningEngine(
            self.model, self.config.seal, self.config.prorl
        )
        
        # Initialize cognitive architecture if available
        if CognitiveArchitecture is not None:
            self.cognitive_architecture = CognitiveArchitecture(
                model_dim=self.config.model.d_model,
                n_heads=self.config.model.n_heads,
                n_layers=self.config.model.n_layers,
                max_seq_length=self.config.model.max_seq_length
            )
            logging.info("Cognitive architecture initialized")
        
        self.is_initialized = True
        logging.info("Jigyasa system initialization completed")
    
    @autonomous_wrapper
    def phase1_foundational_training(
        self,
        save_checkpoint_every: int = 1000,
        checkpoint_dir: str = "./checkpoints/phase1"
    ):
        """
        Phase 1: Foundational ProRL training for general reasoning
        This creates the "teacher" model with advanced reasoning capabilities
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        logging.info("Starting Phase 1: Foundational ProRL Training")
        self.training_phase = "prorl"
        
        # Train with ProRL for advanced reasoning
        training_results = self.prorl_trainer.train()
        
        # Save checkpoint
        checkpoint_path = Path(checkpoint_dir) / "prorl_teacher_model"
        self.prorl_trainer.save_checkpoint(str(checkpoint_path))
        
        # Evaluate on multiple domains
        evaluation_results = {}
        domains = ["mathematical", "logical", "coding", "physics"]
        
        for domain in domains:
            domain_results = self.prorl_trainer.evaluate_on_domain(domain)
            evaluation_results[domain] = domain_results
            logging.info(f"Phase 1 {domain} evaluation: {domain_results}")
        
        logging.info("Phase 1: Foundational training completed")
        return {
            'training_results': training_results,
            'evaluation_results': evaluation_results,
            'checkpoint_path': str(checkpoint_path)
        }
    
    @autonomous_wrapper
    def phase2_continuous_learning(
        self,
        learning_topics: Optional[List[str]] = None,
        learning_cycles: int = 10,
        checkpoint_dir: str = "./checkpoints/phase2",
        dynamic_topics: bool = True
    ):
        """
        Phase 2: Continuous learning with SEAL
        Enables the model to adapt to new information continuously
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        logging.info("Starting Phase 2: Continuous Learning with SEAL")
        self.training_phase = "seal"
        
        # Use STEM and coding training instead of topics
        from .cognitive.stem_training import STEMTrainingGenerator, ConversationalTrainer
        stem_generator = STEMTrainingGenerator()
        conv_trainer = ConversationalTrainer()
        logging.info("Using STEM, coding, and conversational training")
        
        learning_results = []
        
        for cycle in range(learning_cycles):
            logging.info(f"Learning cycle {cycle + 1}/{learning_cycles}")
            
            # Generate STEM and coding problems for this cycle
            batch_size = random.randint(20, 30)
            
            # Mix of problem types
            problem_mix = {
                'math': 0.4,      # 40% math problems
                'coding': 0.4,    # 40% coding problems  
                'science': 0.2    # 20% science problems
            }
            
            # Generate training examples
            training_examples = stem_generator.generate_training_batch(
                batch_size=batch_size,
                mix=problem_mix
            )
            
            # Add conversational examples
            conv_examples = conv_trainer.generate_conversational_examples(count=10)
            
            logging.info(f"Generated {len(training_examples)} STEM problems and {len(conv_examples)} conversational examples for cycle {cycle + 1}")
            
            cycle_results = {}
            
            # Train on STEM examples
            for example in training_examples:
                # Create training data from the example
                training_data = {
                    'input': example.question,
                    'output': example.answer,
                    'reasoning': example.reasoning_steps,
                    'category': example.category,
                    'difficulty': example.difficulty
                }
                
                logging.info(f"Training on {example.category} problem ({example.difficulty})")
                
                # Create evaluation task
                eval_task = {
                    'type': 'problem_solving',
                    'question': example.question,
                    'expected_answer': example.answer,
                    'reasoning_steps': example.reasoning_steps
                }
                
                # Train SEAL episode on this example
                episode_results = self.seal_trainer.train_episode(
                    new_contexts=[json.dumps(training_data)],
                    evaluation_tasks=[eval_task],
                    episode_id=f"{cycle}_{example.category}_{example.difficulty}"
                )
                
                cycle_results[f"{example.category}_{example.difficulty}"] = episode_results
            
            # Train on conversational examples
            for conv_example in conv_examples:
                conv_task = {
                    'type': 'conversation',
                    'input': conv_example['input'],
                    'expected_response': conv_example['response'],
                    'style': conv_example['style']
                }
                
                # Quick conversational training
                conv_results = self.seal_trainer.train_episode(
                    new_contexts=[json.dumps(conv_example)],
                    evaluation_tasks=[conv_task],
                    episode_id=f"{cycle}_conversation"
                )
                
                cycle_results['conversational'] = conv_results
            
            learning_results.append(cycle_results)
            
            # Save checkpoint every few cycles
            if (cycle + 1) % 3 == 0:
                checkpoint_path = Path(checkpoint_dir) / f"seal_cycle_{cycle + 1}"
                self.seal_trainer.save_checkpoint(str(checkpoint_path))
        
        logging.info("Phase 2: Continuous learning completed")
        return learning_results
    
    @autonomous_wrapper
    def phase3_compression_deployment(
        self,
        compression_ratio: float = 0.25,
        output_dir: str = "./deployment"
    ):
        """
        Phase 3: Model compression for on-device deployment
        Creates the compressed "student" model for laptop deployment
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        logging.info("Starting Phase 3: Model Compression and Deployment")
        self.training_phase = "compression"
        
        # Create synthetic training data for distillation
        distillation_data = self._create_distillation_dataset()
        
        # Perform knowledge distillation
        logging.info("Performing knowledge distillation...")
        student_model = distill_model(
            teacher_model=self.model,
            train_dataloader=distillation_data,
            compression_ratio=compression_ratio,
            save_dir=str(Path(output_dir) / "distilled_model")
        )
        
        # Quantize for final deployment
        logging.info("Quantizing model for deployment...")
        quantized_path = str(Path(output_dir) / "jigyasa_quantized.gguf")
        quantize_model_ptq(
            model=student_model,
            output_path=quantized_path,
            calibration_dataloader=distillation_data
        )
        
        # Save deployment package
        deployment_info = {
            'model_path': quantized_path,
            'compression_ratio': compression_ratio,
            'deployment_timestamp': datetime.now().isoformat(),
            'model_size_mb': Path(quantized_path).stat().st_size / (1024 * 1024),
            'config': self.config.__dict__
        }
        
        with open(Path(output_dir) / "deployment_info.json", 'w') as f:
            json.dump(deployment_info, f, indent=2)
        
        logging.info(f"Phase 3: Compression completed. Model saved to {quantized_path}")
        return deployment_info
    
    def interactive_mode(self):
        """
        Interactive mode for testing and demonstration
        Showcases the "think before answering" capability
        """
        if not self.is_initialized:
            raise RuntimeError("System not initialized. Call initialize() first.")
        
        print("\n🧠 Jigyasa Interactive Mode")
        print("=" * 50)
        print("Ask me anything! Type 'quit' to exit, 'help' for commands.")
        print("I will think before answering using self-correction.")
        print("=" * 50)
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye! 👋")
                    break
                
                if user_input.lower() == 'help':
                    self._show_help()
                    continue
                
                if not user_input:
                    continue
                
                # Check for simple greetings
                if user_input.lower() in ['hello', 'hi', 'hey', 'good morning', 'good afternoon', 'good evening']:
                    # Simple response for greetings
                    print("\n💭 Response:")
                    print("-" * 30)
                    print("Hello! I'm Jigyasa, your AI assistant. How can I help you today?")
                    print(f"\n📊 Confidence: 1.00")
                    continue
                
                # Process with self-correction
                print("\n🤔 Thinking...")
                
                result = self.self_correction.think_before_answer(
                    query=user_input,
                    query_type=self._classify_query_type(user_input)
                )
                
                # Display results
                print(f"\n💭 Thinking Process:")
                print("-" * 30)
                print(result['thinking_process'])
                
                print(f"\n✅ Final Answer:")
                print("-" * 30)
                print(result['final_response'])
                
                if result['corrections_made']:
                    print(f"\n🔧 Corrections Made:")
                    for correction in result['corrections_made']:
                        print(f"  • {correction}")
                
                print(f"\n📊 Confidence: {result['confidence_score']:.2f}")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                logging.error(f"Interactive mode error: {str(e)}")
    
    def _create_evaluation_tasks(self, content_list: List[Any]) -> List[Dict[str, Any]]:
        """Create evaluation tasks from processed content"""
        tasks = []
        
        for content in content_list[:5]:  # Limit to 5 tasks
            # Create a simple Q&A task
            text = content.processed_content[:500]  # First 500 chars
            
            task = {
                'type': 'qa',
                'question': f"What is the main topic discussed in this text: {text[:100]}...?",
                'answer': self._extract_main_topic(text),
                'context': text
            }
            tasks.append(task)
        
        return tasks
    
    def _extract_main_topic(self, text: str) -> str:
        """Simple topic extraction"""
        # This is a placeholder - in practice would use NLP techniques
        words = text.lower().split()
        # Remove common words and take first few meaningful words
        meaningful_words = [w for w in words if len(w) > 4 and w.isalpha()]
        return " ".join(meaningful_words[:3])
    
    def _create_distillation_dataset(self):
        """Create synthetic dataset for distillation"""
        # This is a placeholder - would create a proper dataset
        # For now, return None (distillation would handle this)
        return None
    
    def _classify_query_type(self, query: str) -> str:
        """Classify query type for self-correction strategy selection"""
        query_lower = query.lower()
        
        if any(word in query_lower for word in ['calculate', 'solve', 'math', 'equation']):
            return 'mathematical'
        elif any(word in query_lower for word in ['why', 'how', 'explain', 'reason']):
            return 'analytical'
        elif any(word in query_lower for word in ['create', 'write', 'design', 'imagine']):
            return 'creative'
        else:
            return 'factual'
    
    def _show_help(self):
        """Show help information"""
        print("\n📚 Jigyasa Commands:")
        print("  help     - Show this help message")
        print("  quit     - Exit interactive mode")
        print("\n🎯 Query Types:")
        print("  Mathematical - I'll use verification for math problems")
        print("  Analytical   - I'll use reverse reasoning for explanations")
        print("  Creative     - I'll use iterative refinement")
        print("  Factual      - I'll use chain-of-verification")
        print("\nJust ask me anything naturally! 🚀")
    
    def _check_for_existing_checkpoints(self, checkpoint_dir: str) -> Optional[Dict[str, Any]]:
        """
        Check for existing checkpoints and determine where to resume from
        
        Args:
            checkpoint_dir: Directory to check for checkpoints
            
        Returns:
            Dict with resume information or None if no checkpoints found
        """
        import os
        
        checkpoint_path = Path(checkpoint_dir)
        
        # Check for Phase 3 (deployment) completion
        deployment_dir = checkpoint_path / "deployment"
        if deployment_dir.exists() and (deployment_dir / "deployment_info.json").exists():
            return {
                'phase': 'completed',
                'checkpoint_path': str(deployment_dir),
                'description': 'All phases completed'
            }
        
        # Check for Phase 2 (SEAL) checkpoints
        phase2_dir = checkpoint_path / "phase2"
        if phase2_dir.exists():
            # Look for SEAL cycle checkpoints
            seal_checkpoints = list(phase2_dir.glob("seal_cycle_*"))
            if seal_checkpoints:
                latest_seal = max(seal_checkpoints, key=lambda x: int(x.name.split('_')[-1]))
                return {
                    'phase': 'phase3',
                    'checkpoint_path': str(latest_seal),
                    'description': f'Phase 2 completed, found {len(seal_checkpoints)} SEAL cycles'
                }
        
        # Check for Phase 1 (ProRL) checkpoint
        phase1_dir = checkpoint_path / "phase1"
        prorl_checkpoint = phase1_dir / "prorl_teacher_model"
        if prorl_checkpoint.exists():
            return {
                'phase': 'phase2', 
                'checkpoint_path': str(prorl_checkpoint),
                'description': 'Phase 1 (ProRL) completed'
            }
        
        # No checkpoints found
        return None
    
    def _load_checkpoint_if_exists(self, checkpoint_path: str, component: str):
        """Load checkpoint for specific component if it exists"""
        try:
            if component == 'prorl' and self.prorl_trainer:
                self.prorl_trainer.load_checkpoint(checkpoint_path)
                logging.info(f"Loaded ProRL checkpoint from {checkpoint_path}")
            elif component == 'seal' and self.seal_trainer:
                self.seal_trainer.load_checkpoint(checkpoint_path)
                logging.info(f"Loaded SEAL checkpoint from {checkpoint_path}")
            else:
                logging.warning(f"Cannot load checkpoint for component: {component}")
        except Exception as e:
            logging.error(f"Failed to load checkpoint {checkpoint_path}: {e}")

    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and statistics"""
        status = {
            'initialized': self.is_initialized,
            'training_phase': self.training_phase,
            'model_parameters': sum(p.numel() for p in self.model.parameters()) if self.model else 0,
            'timestamp': datetime.now().isoformat()
        }
        
        if self.data_engine:
            status['data_statistics'] = self.data_engine.get_data_statistics()
        
        if self.seal_trainer:
            status['seal_metrics'] = self.seal_trainer.get_training_metrics()
        
        if self.preprocessor:
            status['preprocessing_stats'] = self.preprocessor.get_processing_statistics()
        
        return status


# Main execution function
def main():
    """Main function to run Jigyasa system"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Jigyasa AGI System")
    parser.add_argument('--mode', choices=['train', 'interactive', 'status', 'gui'], 
                       default='interactive', help='Operation mode')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--load-model', type=str, help='Path to pretrained model')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints', 
                       help='Directory containing checkpoints')
    
    args = parser.parse_args()
    
    # Load config
    if args.config:
        config = JigyasaConfig.load(args.config)
    else:
        config = JigyasaConfig()
    
    # Initialize system
    system = JigyasaSystem(config)
    system.initialize(load_pretrained=args.load_model)
    
    if args.mode == 'train':
        print("🚀 Starting full Jigyasa training pipeline...")
        
        # Check for resume functionality
        resume_state = None
        if args.resume:
            resume_state = system._check_for_existing_checkpoints(args.checkpoint_dir)
            if resume_state:
                print(f"📂 Found existing checkpoints! Resuming from: {resume_state['phase']}")
            else:
                print("📂 No existing checkpoints found. Starting fresh training.")
        
        # Phase 1: Foundational training
        if not resume_state or resume_state['phase'] == 'phase1':
            phase1_results = system.phase1_foundational_training(
                checkpoint_dir=f"{args.checkpoint_dir}/phase1"
            )
            print(f"✅ Phase 1 completed: {phase1_results}")
        else:
            print("⏭️ Skipping Phase 1 (already completed)")
        
        # Phase 2: STEM, coding, and conversational training  
        if not resume_state or resume_state['phase'] in ['phase1', 'phase2']:
            phase2_results = system.phase2_continuous_learning(
                learning_topics=None,  # Not using topics anymore
                learning_cycles=5,
                checkpoint_dir=f"{args.checkpoint_dir}/phase2",
                dynamic_topics=False  # Using STEM training instead
            )
            print(f"✅ Phase 2 completed")
        else:
            print("⏭️ Skipping Phase 2 (already completed)")
        
        # Phase 3: Compression
        if not resume_state or resume_state['phase'] in ['phase1', 'phase2', 'phase3']:
            deployment_info = system.phase3_compression_deployment(
                output_dir=f"{args.checkpoint_dir}/deployment"
            )
            print(f"✅ Phase 3 completed: {deployment_info}")
        else:
            print("⏭️ Skipping Phase 3 (already completed)")
        
        print("\n🎉 Jigyasa training pipeline completed successfully!")
        
    elif args.mode == 'interactive':
        system.interactive_mode()
        
    elif args.mode == 'gui':
        print("🚀 Launching JIGYASA Web Dashboard...")
        try:
            from .web.app import app
            print("✅ Starting JIGYASA Web Server...")
            print("📱 Open your browser to: http://localhost:5000")
            print("🛑 Press Ctrl+C to stop")
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        except ImportError as e:
            print(f"❌ Web dependencies missing: {e}")
            print("📦 Installing web dependencies...")
            import subprocess
            subprocess.run([sys.executable, '-m', 'pip', 'install', 'flask'])
            from .web.app import app
            print("📱 Open your browser to: http://localhost:5000")
            app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    elif args.mode == 'status':
        status = system.get_system_status()
        print("\n📊 Jigyasa System Status:")
        print("=" * 40)
        for key, value in status.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()


# Alias for compatibility
Jigyasa = JigyasaSystem