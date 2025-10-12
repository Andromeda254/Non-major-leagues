#!/usr/bin/env python3
"""
Simple Pipeline Runner
=====================

This script provides an easy way to run the Non-Major League ML Pipeline
with common configurations and options.

Usage:
    python run_pipeline.py --quick          # Quick test run
    python run_pipeline.py --full           # Full pipeline run
    python run_pipeline.py --phase 1        # Run only phase 1
    python run_pipeline.py --league E1      # Run for specific league
    python run_pipeline.py --deploy prod    # Deploy to production
"""

import argparse
import os
import sys
from pathlib import Path
from master_pipeline import MasterPipeline

def run_quick_test():
    """Run a quick test of the pipeline with minimal configuration"""
    print("🚀 Running Quick Test Pipeline...")
    
    # Create minimal config for testing
    config = {
        'pipeline': {
            'output_dir': './test_output',
            'temp_dir': './test_temp',
            'log_level': 'INFO'
        },
        'phase1': {
            'enabled': True,
            'leagues': ['E1'],
            'seasons': 1,
            'data_sources': ['football-data']
        },
        'phase2': {
            'enabled': True,
            'models': ['xgboost', 'lightgbm'],
            'ensemble_method': 'weighted',
            'transfer_learning': False,
            'hyperparameter_tuning': False
        },
        'phase3': {
            'enabled': True,
            'backtesting_period': '6_months',
            'initial_capital': 1000,
            'kelly_fraction': 0.01
        },
        'phase4': {
            'enabled': False  # Skip deployment for quick test
        }
    }
    
    # Save config temporarily
    import yaml
    config_path = './test_config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        pipeline = MasterPipeline(config_path=config_path)
        result = pipeline.run_all_phases(league='E1')
        print("✅ Quick test completed successfully!")
        return result
    finally:
        # Clean up
        if os.path.exists(config_path):
            os.remove(config_path)

def run_full_pipeline(league='E1', environment='development'):
    """Run the full pipeline with complete configuration"""
    print(f"🚀 Running Full Pipeline for {league}...")
    
    config_path = './config.yaml'
    if not os.path.exists(config_path):
        print("❌ Configuration file not found. Please create config.yaml")
        return None
    
    # Update config for deployment environment
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['phase4']['deployment_environment'] = environment
    
    # Save updated config
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    try:
        pipeline = MasterPipeline(config_path=config_path)
        result = pipeline.run_all_phases(league=league)
        print("✅ Full pipeline completed successfully!")
        return result
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        return None

def run_single_phase(phase, league='E1'):
    """Run a single phase of the pipeline"""
    print(f"🚀 Running Phase {phase} for {league}...")
    
    config_path = './config.yaml'
    if not os.path.exists(config_path):
        print("❌ Configuration file not found. Please create config.yaml")
        return None
    
    try:
        pipeline = MasterPipeline(config_path=config_path)
        
        if phase == 1:
            result = pipeline.run_phase1(league=league)
        elif phase == 2:
            result = pipeline.run_phase2()
        elif phase == 3:
            result = pipeline.run_phase3()
        elif phase == 4:
            result = pipeline.run_phase4()
        else:
            print(f"❌ Invalid phase: {phase}")
            return None
            
        print(f"✅ Phase {phase} completed successfully!")
        return result
    except Exception as e:
        print(f"❌ Phase {phase} failed: {e}")
        return None

def setup_environment():
    """Setup the environment for running the pipeline"""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    dirs = [
        './pipeline_output',
        './temp',
        './logs',
        './models',
        './data',
        './reports',
        './deployments'
    ]
    
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    # Check if config file exists
    if not os.path.exists('./config.yaml'):
        print("⚠️  Configuration file not found. Creating template...")
        # The config.yaml should already be created
        print("📝 Please edit config.yaml with your API keys and settings")
    
    print("✅ Environment setup complete!")

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'pandas', 'numpy', 'scikit-learn', 'xgboost', 'lightgbm',
        'optuna', 'joblib', 'requests', 'schedule', 'yaml', 'fastapi',
        'uvicorn', 'matplotlib', 'seaborn', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("📦 Please install missing packages:")
        print(f"   pip install {' '.join(missing_packages)}")
        return False
    else:
        print("✅ All dependencies are installed!")
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Non-Major League ML Pipeline Runner')
    parser.add_argument('--quick', action='store_true',
                       help='Run quick test pipeline')
    parser.add_argument('--full', action='store_true',
                       help='Run full pipeline')
    parser.add_argument('--phase', type=int, choices=[1, 2, 3, 4],
                       help='Run single phase')
    parser.add_argument('--league', type=str, default='E1',
                       help='League code (default: E1)')
    parser.add_argument('--deploy', choices=['dev', 'staging', 'prod'],
                       help='Deployment environment')
    parser.add_argument('--setup', action='store_true',
                       help='Setup environment')
    parser.add_argument('--check-deps', action='store_true',
                       help='Check dependencies')
    
    args = parser.parse_args()
    
    # Check dependencies first
    if args.check_deps:
        check_dependencies()
        return
    
    # Setup environment
    if args.setup:
        setup_environment()
        return
    
    # Check dependencies before running
    if not check_dependencies():
        sys.exit(1)
    
    # Run pipeline based on arguments
    if args.quick:
        run_quick_test()
    elif args.full:
        environment = args.deploy or 'development'
        run_full_pipeline(league=args.league, environment=environment)
    elif args.phase:
        run_single_phase(phase=args.phase, league=args.league)
    else:
        print("❓ No action specified. Use --help for options.")
        print("\nQuick start:")
        print("  python run_pipeline.py --setup     # Setup environment")
        print("  python run_pipeline.py --quick     # Quick test run")
        print("  python run_pipeline.py --full      # Full pipeline run")

if __name__ == "__main__":
    main()







