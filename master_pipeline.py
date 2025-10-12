#!/usr/bin/env python3
"""
Master Pipeline Script for Non-Major League ML System
====================================================

This script orchestrates the complete 4-phase ML pipeline:
- Phase 1: Data Collection, Preprocessing, Validation, Feature Engineering
- Phase 2: Model Architecture, Ensemble, Transfer Learning, Hyperparameter Tuning, Model Validation
- Phase 3: Backtesting, Betting Strategy, Performance Metrics, Risk Management, Live Testing
- Phase 4: Production Deployment, Monitoring, Data Pipeline, Model Serving, Performance Tracking

Usage:
    python master_pipeline.py --phase all --league E1 --config config.yaml
    python master_pipeline.py --phase 1 --league E1
    python master_pipeline.py --phase 2 --input-dir ./phase1_output
    python master_pipeline.py --phase 3 --model-dir ./phase2_output
    python master_pipeline.py --phase 4 --strategy-dir ./phase3_output
"""

import argparse
import os
import sys
import json
import yaml
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')

# Import all pipeline components
from phase1_integration import Phase1Integration
from phase2_integration import Phase2Integration
from phase3_integration import Phase3Integration
from phase4_integration import Phase4Integration

# Import individual components for direct access if needed
from non_major_league_data_collector import NonMajorLeagueDataCollector
from non_major_league_preprocessor import NonMajorLeaguePreprocessor
from non_major_league_validator import NonMajorLeagueValidator
from non_major_league_feature_engineer import NonMajorLeagueFeatureEngineer
from non_major_league_model_architecture import NonMajorLeagueModelArchitecture
from non_major_league_ensemble import NonMajorLeagueEnsemble
from non_major_league_transfer_learning import NonMajorLeagueTransferLearning
from non_major_league_hyperparameter_tuning import NonMajorLeagueHyperparameterTuning
from non_major_league_model_validation import NonMajorLeagueModelValidation
from non_major_league_backtesting import NonMajorLeagueBacktesting
from non_major_league_betting_strategy import NonMajorLeagueBettingStrategy
from non_major_league_performance_metrics import NonMajorLeaguePerformanceMetrics
from non_major_league_risk_management import NonMajorLeagueRiskManagement
from non_major_league_live_testing import NonMajorLeagueLiveTesting
from non_major_league_deployment import NonMajorLeagueDeployment
from non_major_league_monitoring import NonMajorLeagueMonitoring
from non_major_league_data_pipeline import NonMajorLeagueDataPipeline
from non_major_league_model_serving import NonMajorLeagueModelServing
from non_major_league_performance_tracking import NonMajorLeaguePerformanceTracking

class MasterPipeline:
    """
    Master pipeline orchestrator that coordinates all phases and components
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize the master pipeline
        
        Args:
            config_path: Path to configuration file
        """
        self.setup_logging()
        self.load_config(config_path)
        self.setup_directories()
        self.pipeline_state = {
            'start_time': datetime.now(),
            'phases_completed': [],
            'current_phase': None,
            'errors': [],
            'warnings': [],
            'results': {}
        }
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        log_dir = Path('./logs')
        log_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = log_dir / f'master_pipeline_{timestamp}.log'
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info("Master Pipeline initialized")
        
    def load_config(self, config_path: str):
        """Load configuration from file or use defaults"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            self.logger.info(f"Configuration loaded from {config_path}")
        else:
            self.config = self._get_default_config()
            self.logger.info("Using default configuration")
            
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'pipeline': {
                'output_dir': './pipeline_output',
                'temp_dir': './temp',
                'log_level': 'INFO',
                'parallel_processing': False,
                'max_workers': 4
            },
            'phase1': {
                'enabled': True,
                'leagues': ['E1', 'E2', 'E3'],  # Championship, League 1, League 2
                'seasons': 3,
                'data_sources': ['football-data', 'odds-api', 'api-football'],
                'output_format': 'parquet'
            },
            'phase2': {
                'enabled': True,
                'models': ['xgboost', 'lightgbm', 'random_forest', 'logistic_regression'],
                'ensemble_method': 'weighted',
                'transfer_learning': True,
                'hyperparameter_tuning': True,
                'validation_method': 'time_series_split'
            },
            'phase3': {
                'enabled': True,
                'backtesting_period': '2_years',
                'initial_capital': 10000,
                'kelly_fraction': 0.02,
                'risk_management': True,
                'live_testing': True
            },
            'phase4': {
                'enabled': True,
                'deployment_environment': 'development',
                'monitoring': True,
                'data_pipeline': True,
                'model_serving': True,
                'performance_tracking': True
            },
            'data_sources': {
                'football_data': {
                    'api_key': 'your_api_key_here',
                    'base_url': 'https://api.football-data.org/v4'
                },
                'odds_api': {
                    'api_key': 'your_api_key_here',
                    'base_url': 'https://api.the-odds-api.com/v4'
                },
                'api_football': {
                    'api_key': 'your_api_key_here',
                    'base_url': 'https://v3.football.api-sports.io'
                }
            },
            'model_config': {
                'xgboost': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8
                },
                'lightgbm': {
                    'n_estimators': 100,
                    'max_depth': 6,
                    'learning_rate': 0.1,
                    'subsample': 0.8
                },
                'random_forest': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5
                },
                'logistic_regression': {
                    'C': 1.0,
                    'max_iter': 1000
                }
            },
            'risk_management': {
                'max_drawdown': 0.2,
                'daily_loss_limit': 0.05,
                'position_size_limit': 0.1,
                'emergency_stop': 0.15
            },
            'deployment': {
                'environments': {
                    'development': {
                        'base_url': 'http://localhost:8000',
                        'database_url': 'sqlite:///dev.db'
                    },
                    'staging': {
                        'base_url': 'http://staging.example.com',
                        'database_url': 'postgresql://staging:password@staging-db:5432/staging'
                    },
                    'production': {
                        'base_url': 'https://api.example.com',
                        'database_url': 'postgresql://prod:password@prod-db:5432/production'
                    }
                }
            }
        }
        
    def setup_directories(self):
        """Create necessary directories"""
        dirs = [
            self.config['pipeline']['output_dir'],
            self.config['pipeline']['temp_dir'],
            './logs',
            './models',
            './data',
            './reports',
            './deployments'
        ]
        
        for dir_path in dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
            
    def run_phase1(self, league: str = None, input_dir: str = None) -> Dict[str, Any]:
        """Run Phase 1: Data Collection, Preprocessing, Validation, Feature Engineering"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING PHASE 1: DATA COLLECTION & PREPROCESSING")
        self.logger.info("=" * 80)
        
        try:
            self.pipeline_state['current_phase'] = 'phase1'
            start_time = time.time()
            
            # Initialize Phase 1 integration
            phase1 = Phase1Integration()
            
            # Run Phase 1 pipeline
            result = phase1.run_phase1_pipeline(
                league=league or self.config['phase1']['leagues'][0],
                output_dir=self.config['pipeline']['output_dir'],
                config=self.config
            )
            
            # Store results
            phase1_output_dir = Path(self.config['pipeline']['output_dir']) / 'phase1_output'
            phase1_output_dir.mkdir(exist_ok=True)
            
            with open(phase1_output_dir / 'phase1_results.json', 'w') as f:
                json.dump(result, f, indent=2, default=str)
                
            duration = time.time() - start_time
            self.pipeline_state['phases_completed'].append('phase1')
            self.pipeline_state['results']['phase1'] = {
                'status': 'completed',
                'duration': duration,
                'output_dir': str(phase1_output_dir),
                'result': result
            }
            
            self.logger.info(f"Phase 1 completed successfully in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Phase 1 failed: {e}")
            self.pipeline_state['errors'].append(f"Phase 1: {str(e)}")
            self.pipeline_state['results']['phase1'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
            
    def run_phase2(self, input_dir: str = None) -> Dict[str, Any]:
        """Run Phase 2: Model Architecture, Ensemble, Transfer Learning, Hyperparameter Tuning, Model Validation"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING PHASE 2: MODEL DEVELOPMENT & VALIDATION")
        self.logger.info("=" * 80)
        
        try:
            self.pipeline_state['current_phase'] = 'phase2'
            start_time = time.time()
            
            # Determine input directory
            if input_dir is None:
                input_dir = Path(self.config['pipeline']['output_dir']) / 'phase1_output'
            else:
                input_dir = Path(input_dir)
                
            if not input_dir.exists():
                raise FileNotFoundError(f"Phase 1 output directory not found: {input_dir}")
                
            # Initialize Phase 2 integration
            phase2 = Phase2Integration()
            
            # Run Phase 2 pipeline
            result = phase2.run_phase2_pipeline(
                feature_data_path=str(input_dir / 'processed_features.parquet'),
                output_dir=self.config['pipeline']['output_dir'],
                config=self.config
            )
            
            # Store results
            phase2_output_dir = Path(self.config['pipeline']['output_dir']) / 'phase2_output'
            phase2_output_dir.mkdir(exist_ok=True)
            
            with open(phase2_output_dir / 'phase2_results.json', 'w') as f:
                json.dump(result, f, indent=2, default=str)
                
            duration = time.time() - start_time
            self.pipeline_state['phases_completed'].append('phase2')
            self.pipeline_state['results']['phase2'] = {
                'status': 'completed',
                'duration': duration,
                'output_dir': str(phase2_output_dir),
                'result': result
            }
            
            self.logger.info(f"Phase 2 completed successfully in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Phase 2 failed: {e}")
            self.pipeline_state['errors'].append(f"Phase 2: {str(e)}")
            self.pipeline_state['results']['phase2'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
            
    def run_phase3(self, model_dir: str = None) -> Dict[str, Any]:
        """Run Phase 3: Backtesting, Betting Strategy, Performance Metrics, Risk Management, Live Testing"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING PHASE 3: BACKTESTING & STRATEGY VALIDATION")
        self.logger.info("=" * 80)
        
        try:
            self.pipeline_state['current_phase'] = 'phase3'
            start_time = time.time()
            
            # Determine model directory
            if model_dir is None:
                model_dir = Path(self.config['pipeline']['output_dir']) / 'phase2_output'
            else:
                model_dir = Path(model_dir)
                
            if not model_dir.exists():
                raise FileNotFoundError(f"Phase 2 output directory not found: {model_dir}")
                
            # Initialize Phase 3 integration
            phase3 = Phase3Integration()
            
            # Run Phase 3 pipeline
            result = phase3.run_phase3_pipeline(
                feature_data_path=str(Path(self.config['pipeline']['output_dir']) / 'phase1_output' / 'processed_features.parquet'),
                model_path=str(model_dir / 'ensemble_model.pkl'),
                output_dir=self.config['pipeline']['output_dir'],
                config=self.config
            )
            
            # Store results
            phase3_output_dir = Path(self.config['pipeline']['output_dir']) / 'phase3_output'
            phase3_output_dir.mkdir(exist_ok=True)
            
            with open(phase3_output_dir / 'phase3_results.json', 'w') as f:
                json.dump(result, f, indent=2, default=str)
                
            duration = time.time() - start_time
            self.pipeline_state['phases_completed'].append('phase3')
            self.pipeline_state['results']['phase3'] = {
                'status': 'completed',
                'duration': duration,
                'output_dir': str(phase3_output_dir),
                'result': result
            }
            
            self.logger.info(f"Phase 3 completed successfully in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Phase 3 failed: {e}")
            self.pipeline_state['errors'].append(f"Phase 3: {str(e)}")
            self.pipeline_state['results']['phase3'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
            
    def run_phase4(self, strategy_dir: str = None) -> Dict[str, Any]:
        """Run Phase 4: Production Deployment, Monitoring, Data Pipeline, Model Serving, Performance Tracking"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING PHASE 4: PRODUCTION DEPLOYMENT & MONITORING")
        self.logger.info("=" * 80)
        
        try:
            self.pipeline_state['current_phase'] = 'phase4'
            start_time = time.time()
            
            # Determine strategy directory
            if strategy_dir is None:
                strategy_dir = Path(self.config['pipeline']['output_dir']) / 'phase3_output'
            else:
                strategy_dir = Path(strategy_dir)
                
            if not strategy_dir.exists():
                raise FileNotFoundError(f"Phase 3 output directory not found: {strategy_dir}")
                
            # Initialize Phase 4 integration
            phase4 = Phase4Integration()
            
            # Run Phase 4 pipeline
            result = phase4.run_phase4_pipeline(
                strategy_data_path=str(strategy_dir / 'betting_strategy.pkl'),
                output_dir=self.config['pipeline']['output_dir'],
                config=self.config
            )
            
            # Store results
            phase4_output_dir = Path(self.config['pipeline']['output_dir']) / 'phase4_output'
            phase4_output_dir.mkdir(exist_ok=True)
            
            with open(phase4_output_dir / 'phase4_results.json', 'w') as f:
                json.dump(result, f, indent=2, default=str)
                
            duration = time.time() - start_time
            self.pipeline_state['phases_completed'].append('phase4')
            self.pipeline_state['results']['phase4'] = {
                'status': 'completed',
                'duration': duration,
                'output_dir': str(phase4_output_dir),
                'result': result
            }
            
            self.logger.info(f"Phase 4 completed successfully in {duration:.2f} seconds")
            return result
            
        except Exception as e:
            self.logger.error(f"Phase 4 failed: {e}")
            self.pipeline_state['errors'].append(f"Phase 4: {str(e)}")
            self.pipeline_state['results']['phase4'] = {
                'status': 'failed',
                'error': str(e)
            }
            raise
            
    def run_all_phases(self, league: str = None) -> Dict[str, Any]:
        """Run all phases in sequence"""
        self.logger.info("=" * 80)
        self.logger.info("STARTING COMPLETE PIPELINE EXECUTION")
        self.logger.info("=" * 80)
        
        try:
            # Phase 1: Data Collection & Preprocessing
            if self.config['phase1']['enabled']:
                phase1_result = self.run_phase1(league=league)
            else:
                self.logger.info("Phase 1 disabled in configuration")
                
            # Phase 2: Model Development & Validation
            if self.config['phase2']['enabled']:
                phase2_result = self.run_phase2()
            else:
                self.logger.info("Phase 2 disabled in configuration")
                
            # Phase 3: Backtesting & Strategy Validation
            if self.config['phase3']['enabled']:
                phase3_result = self.run_phase3()
            else:
                self.logger.info("Phase 3 disabled in configuration")
                
            # Phase 4: Production Deployment & Monitoring
            if self.config['phase4']['enabled']:
                phase4_result = self.run_phase4()
            else:
                self.logger.info("Phase 4 disabled in configuration")
                
            # Generate final report
            final_report = self.generate_final_report()
            
            self.logger.info("=" * 80)
            self.logger.info("PIPELINE EXECUTION COMPLETED SUCCESSFULLY")
            self.logger.info("=" * 80)
            
            return final_report
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            self.pipeline_state['errors'].append(f"Pipeline: {str(e)}")
            raise
            
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report"""
        self.pipeline_state['end_time'] = datetime.now()
        total_duration = (self.pipeline_state['end_time'] - self.pipeline_state['start_time']).total_seconds()
        
        report = {
            'pipeline_summary': {
                'start_time': self.pipeline_state['start_time'].isoformat(),
                'end_time': self.pipeline_state['end_time'].isoformat(),
                'total_duration': total_duration,
                'phases_completed': self.pipeline_state['phases_completed'],
                'success_rate': len(self.pipeline_state['phases_completed']) / 4.0
            },
            'phase_results': self.pipeline_state['results'],
            'errors': self.pipeline_state['errors'],
            'warnings': self.pipeline_state['warnings'],
            'recommendations': self._generate_recommendations()
        }
        
        # Save final report
        report_path = Path(self.config['pipeline']['output_dir']) / 'final_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
            
        # Generate human-readable report
        self._generate_human_readable_report(report)
        
        return report
        
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on pipeline results"""
        recommendations = []
        
        # Check for errors
        if self.pipeline_state['errors']:
            recommendations.append("Address the following errors before production deployment:")
            for error in self.pipeline_state['errors']:
                recommendations.append(f"  - {error}")
                
        # Check phase completion
        if len(self.pipeline_state['phases_completed']) < 4:
            recommendations.append("Complete all phases before production deployment")
            
        # Check performance metrics if Phase 3 completed
        if 'phase3' in self.pipeline_state['results']:
            phase3_result = self.pipeline_state['results']['phase3']['result']
            if 'performance_metrics' in phase3_result:
                metrics = phase3_result['performance_metrics']
                if metrics.get('roi', 0) < 0.05:
                    recommendations.append("Consider improving model performance - ROI below 5%")
                if metrics.get('max_drawdown', 0) > 0.2:
                    recommendations.append("High drawdown detected - review risk management")
                    
        return recommendations
        
    def _generate_human_readable_report(self, report: Dict[str, Any]):
        """Generate human-readable report"""
        report_path = Path(self.config['pipeline']['output_dir']) / 'final_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("NON-MAJOR LEAGUE ML PIPELINE - FINAL REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            # Pipeline Summary
            f.write("PIPELINE SUMMARY:\n")
            f.write("-" * 40 + "\n")
            f.write(f"Start Time: {report['pipeline_summary']['start_time']}\n")
            f.write(f"End Time: {report['pipeline_summary']['end_time']}\n")
            f.write(f"Total Duration: {report['pipeline_summary']['total_duration']:.2f} seconds\n")
            f.write(f"Phases Completed: {', '.join(report['pipeline_summary']['phases_completed'])}\n")
            f.write(f"Success Rate: {report['pipeline_summary']['success_rate']:.2%}\n\n")
            
            # Phase Results
            f.write("PHASE RESULTS:\n")
            f.write("-" * 40 + "\n")
            for phase, result in report['phase_results'].items():
                f.write(f"{phase.upper()}:\n")
                f.write(f"  Status: {result['status']}\n")
                if 'duration' in result:
                    f.write(f"  Duration: {result['duration']:.2f} seconds\n")
                if 'error' in result:
                    f.write(f"  Error: {result['error']}\n")
                f.write("\n")
                
            # Errors and Warnings
            if report['errors']:
                f.write("ERRORS:\n")
                f.write("-" * 40 + "\n")
                for error in report['errors']:
                    f.write(f"  - {error}\n")
                f.write("\n")
                
            if report['warnings']:
                f.write("WARNINGS:\n")
                f.write("-" * 40 + "\n")
                for warning in report['warnings']:
                    f.write(f"  - {warning}\n")
                f.write("\n")
                
            # Recommendations
            if report['recommendations']:
                f.write("RECOMMENDATIONS:\n")
                f.write("-" * 40 + "\n")
                for rec in report['recommendations']:
                    f.write(f"  - {rec}\n")
                f.write("\n")
                
            f.write("=" * 80 + "\n")
            
        self.logger.info(f"Human-readable report saved to {report_path}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Non-Major League ML Pipeline Master Script')
    parser.add_argument('--phase', choices=['1', '2', '3', '4', 'all'], default='all',
                       help='Phase to run (default: all)')
    parser.add_argument('--league', type=str, default='E1',
                       help='League code to process (default: E1)')
    parser.add_argument('--config', type=str, default=None,
                       help='Configuration file path')
    parser.add_argument('--input-dir', type=str, default=None,
                       help='Input directory for phase 2')
    parser.add_argument('--model-dir', type=str, default=None,
                       help='Model directory for phase 3')
    parser.add_argument('--strategy-dir', type=str, default=None,
                       help='Strategy directory for phase 4')
    parser.add_argument('--output-dir', type=str, default='./pipeline_output',
                       help='Output directory (default: ./pipeline_output)')
    
    args = parser.parse_args()
    
    try:
        # Initialize master pipeline
        pipeline = MasterPipeline(config_path=args.config)
        
        # Override output directory if specified
        if args.output_dir:
            pipeline.config['pipeline']['output_dir'] = args.output_dir
            pipeline.setup_directories()
            
        # Run specified phase(s)
        if args.phase == 'all':
            result = pipeline.run_all_phases(league=args.league)
        elif args.phase == '1':
            result = pipeline.run_phase1(league=args.league)
        elif args.phase == '2':
            result = pipeline.run_phase2(input_dir=args.input_dir)
        elif args.phase == '3':
            result = pipeline.run_phase3(model_dir=args.model_dir)
        elif args.phase == '4':
            result = pipeline.run_phase4(strategy_dir=args.strategy_dir)
            
        print(f"\nPipeline execution completed successfully!")
        print(f"Results saved to: {pipeline.config['pipeline']['output_dir']}")
        
    except Exception as e:
        print(f"Pipeline execution failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()

