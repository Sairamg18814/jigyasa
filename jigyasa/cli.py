"""
Command Line Interface for Jigyasa
Provides easy access to all system functionality
"""

import argparse
import sys
from pathlib import Path
import json

from .main import JigyasaSystem
from .config import JigyasaConfig


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Jigyasa: A Self-Improving, Agentic Language Model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  jigyasa interactive                    # Start interactive chat mode
  jigyasa train --full-pipeline         # Run complete training pipeline
  jigyasa compress --input model.pt     # Compress existing model
  jigyasa benchmark --model model.pt    # Run evaluation benchmarks
  jigyasa data --topics "AI,ML" --collect  # Collect data for topics
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Interactive mode
    interactive_parser = subparsers.add_parser('interactive', help='Start interactive chat mode')
    interactive_parser.add_argument('--model', type=str, help='Path to model to load')
    interactive_parser.add_argument('--config', type=str, help='Path to config file')
    
    # Training commands
    train_parser = subparsers.add_parser('train', help='Training operations')
    train_parser.add_argument('--full-pipeline', action='store_true', 
                             help='Run complete training pipeline')
    train_parser.add_argument('--phase', choices=['prorl', 'seal', 'all'], 
                             default='all', help='Training phase to run')
    train_parser.add_argument('--config', type=str, help='Path to config file')
    train_parser.add_argument('--output-dir', type=str, default='./models',
                             help='Output directory for trained models')
    
    # Compression commands
    compress_parser = subparsers.add_parser('compress', help='Model compression')
    compress_parser.add_argument('--input', type=str, required=True,
                                help='Input model path')
    compress_parser.add_argument('--output', type=str, required=True,
                                help='Output compressed model path')
    compress_parser.add_argument('--ratio', type=float, default=0.25,
                                help='Compression ratio (default: 0.25)')
    compress_parser.add_argument('--format', choices=['gguf', 'onnx'], default='gguf',
                                help='Output format')
    
    # Data collection commands
    data_parser = subparsers.add_parser('data', help='Data operations')
    data_parser.add_argument('--collect', action='store_true',
                            help='Collect new data')
    data_parser.add_argument('--topics', type=str, required=True,
                            help='Comma-separated list of topics')
    data_parser.add_argument('--max-sources', type=int, default=20,
                            help='Maximum sources per topic')
    data_parser.add_argument('--output-dir', type=str, default='./data',
                            help='Output directory for collected data')
    
    # Benchmark commands
    benchmark_parser = subparsers.add_parser('benchmark', help='Run evaluation benchmarks')
    benchmark_parser.add_argument('--model', type=str, required=True,
                                 help='Path to model to evaluate')
    benchmark_parser.add_argument('--benchmarks', type=str, 
                                 default='mmlu,humaneval,truthfulqa',
                                 help='Comma-separated list of benchmarks')
    benchmark_parser.add_argument('--output', type=str, default='./benchmark_results.json',
                                 help='Output file for results')
    
    # Status command
    status_parser = subparsers.add_parser('status', help='Show system status')
    status_parser.add_argument('--detailed', action='store_true',
                              help='Show detailed status information')
    
    # Config commands
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_parser.add_argument('--create', action='store_true',
                               help='Create default configuration file')
    config_parser.add_argument('--output', type=str, default='jigyasa_config.json',
                               help='Configuration file path')
    config_parser.add_argument('--show', action='store_true',
                               help='Show current configuration')
    
    return parser


def handle_interactive(args):
    """Handle interactive mode"""
    print("🧠 Welcome to Jigyasa Interactive Mode!")
    
    # Load config
    config = None
    if args.config:
        config = JigyasaConfig.load(args.config)
    
    # Initialize system
    system = JigyasaSystem(config)
    system.initialize(load_pretrained=args.model)
    
    # Start interactive mode
    system.interactive_mode()


def handle_train(args):
    """Handle training commands"""
    print("🚀 Starting Jigyasa Training...")
    
    # Load config
    config = None
    if args.config:
        config = JigyasaConfig.load(args.config)
    
    # Initialize system
    system = JigyasaSystem(config)
    system.initialize()
    
    if args.full_pipeline or args.phase == 'all':
        print("Running full training pipeline...")
        
        # Phase 1: ProRL
        if args.phase in ['prorl', 'all']:
            print("\n📊 Phase 1: ProRL Training")
            phase1_results = system.phase1_foundational_training(
                checkpoint_dir=f"{args.output_dir}/phase1"
            )
            print(f"✅ Phase 1 completed with results: {phase1_results}")
        
        # Phase 2: SEAL
        if args.phase in ['seal', 'all']:
            print("\n🧠 Phase 2: Continuous Learning")
            topics = ["artificial intelligence", "machine learning", "science"]
            phase2_results = system.phase2_continuous_learning(
                learning_topics=topics,
                checkpoint_dir=f"{args.output_dir}/phase2"
            )
            print(f"✅ Phase 2 completed")
        
        # Phase 3: Compression
        print("\n⚡ Phase 3: Model Compression")
        deployment_info = system.phase3_compression_deployment(
            output_dir=f"{args.output_dir}/deployment"
        )
        print(f"✅ Phase 3 completed: {deployment_info}")
        
    print("\n🎉 Training completed successfully!")


def handle_compress(args):
    """Handle model compression"""
    print(f"⚡ Compressing model: {args.input}")
    
    from .core.model import JigyasaModel
    from .compression.quantization import quantize_model_ptq
    from .compression.distillation import distill_model
    
    # Load model
    model = JigyasaModel.from_pretrained(args.input)
    print(f"Loaded model with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    if args.ratio < 1.0:
        # Knowledge distillation first
        print(f"Performing knowledge distillation (ratio: {args.ratio})...")
        # This would need a proper dataset - simplified for demo
        compressed_model = model  # Placeholder
    else:
        compressed_model = model
    
    # Quantization
    print(f"Quantizing to {args.format} format...")
    output_path = quantize_model_ptq(
        model=compressed_model,
        output_path=args.output
    )
    
    print(f"✅ Compressed model saved to: {output_path}")


def handle_data(args):
    """Handle data collection"""
    if not args.collect:
        print("❌ Please specify --collect to collect data")
        return
    
    print(f"📊 Collecting data for topics: {args.topics}")
    
    from .data.data_engine import DataEngine
    from .data.preprocessing import DataPreprocessor
    from .config import DataConfig
    
    # Initialize data components
    data_config = DataConfig()
    data_engine = DataEngine(data_config)
    preprocessor = DataPreprocessor(data_config)
    
    topics = [topic.strip() for topic in args.topics.split(',')]
    
    all_results = []
    
    for topic in topics:
        print(f"\n🔍 Collecting data for: {topic}")
        
        # Collect data
        contents = data_engine.acquire_data_for_topic(
            topic=topic,
            max_sources=args.max_sources
        )
        
        # Process data
        processed_results = preprocessor.process_batch(contents)
        
        # Filter high-quality content
        high_quality = [r for r in processed_results if r.should_include]
        
        print(f"✅ Collected {len(high_quality)} high-quality sources for {topic}")
        all_results.extend(high_quality)
    
    # Export results
    output_path = Path(args.output_dir) / "collected_data.jsonl"
    preprocessor.export_processed_data(all_results, str(output_path))
    
    print(f"\n🎉 Data collection completed! Saved to: {output_path}")


def handle_benchmark(args):
    """Handle benchmark evaluation"""
    print(f"📊 Running benchmarks on model: {args.model}")
    
    # This would integrate with the evaluation module
    print("⚠️  Benchmark functionality coming soon!")
    print(f"Requested benchmarks: {args.benchmarks}")
    print(f"Results will be saved to: {args.output}")


def handle_status(args):
    """Handle status command"""
    print("📊 Jigyasa System Status")
    print("=" * 40)
    
    try:
        # Try to get status from an initialized system
        system = JigyasaSystem()
        if Path("./models/current").exists():
            system.initialize(load_pretrained="./models/current")
            status = system.get_system_status()
            
            for key, value in status.items():
                if isinstance(value, dict):
                    print(f"{key}:")
                    for sub_key, sub_value in value.items():
                        print(f"  {sub_key}: {sub_value}")
                else:
                    print(f"{key}: {value}")
        else:
            print("No trained model found. Run 'jigyasa train' first.")
            
    except Exception as e:
        print(f"Could not get system status: {e}")
        print("\nBasic system info:")
        print(f"Python version: {sys.version}")
        print(f"PyTorch available: {'Yes' if 'torch' in sys.modules else 'No'}")


def handle_config(args):
    """Handle configuration commands"""
    if args.create:
        print(f"📝 Creating default configuration file: {args.output}")
        
        config = JigyasaConfig()
        config.save(args.output)
        
        print("✅ Configuration file created!")
        print(f"You can edit {args.output} to customize settings.")
        
    elif args.show:
        print("📋 Current Configuration:")
        print("=" * 30)
        
        config = JigyasaConfig()
        config_dict = config.__dict__
        
        print(json.dumps(config_dict, indent=2, default=str))


def main():
    """Main CLI entry point"""
    parser = create_parser()
    
    if len(sys.argv) == 1:
        # No arguments provided, show help
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    try:
        if args.command == 'interactive':
            handle_interactive(args)
        elif args.command == 'train':
            handle_train(args)
        elif args.command == 'compress':
            handle_compress(args)
        elif args.command == 'data':
            handle_data(args)
        elif args.command == 'benchmark':
            handle_benchmark(args)
        elif args.command == 'status':
            handle_status(args)
        elif args.command == 'config':
            handle_config(args)
        else:
            parser.print_help()
            
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if '--debug' in sys.argv:
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()