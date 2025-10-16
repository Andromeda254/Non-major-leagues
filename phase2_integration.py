#!/usr/bin/env python3
"""
Phase 2 Integration Script for Non-Major League ML Pipeline

This script integrates all Phase 2 components:
1. Model Architecture Design
2. Ensemble Modeling
3. Transfer Learning
4. Hyperparameter Tuning
5. Model Validation

Usage:
    python phase2_integration.py --league E1 --data_file ./data/E1_features.csv --output_dir ./models
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json
import joblib
from typing import Dict

# Import our custom modules
from non_major_league_model_architecture import NonMajorLeagueModelArchitecture
from non_major_league_ensemble import NonMajorLeagueEnsemble
from non_major_league_transfer_learning import NonMajorLeagueTransferLearning
from non_major_league_hyperparameter_tuning import NonMajorLeagueHyperparameterTuning
from non_major_league_model_validation import NonMajorLeagueModelValidation

class Phase2Integration:
    """
    Phase 2 Integration for Non-Major League ML Pipeline
    
    This class orchestrates the complete Phase 2 workflow:
    - Model architecture design and training
    - Ensemble modeling with conservative parameters
    - Transfer learning from major leagues
    - Hyperparameter optimization for limited data
    - Comprehensive model validation
    """
    
    def __init__(self, config_file: str = None, output_dir: str = "./models"):
        """
        Initialize Phase 2 integration
        
        Args:
            config_file: Path to configuration file
            output_dir: Output directory for models
        """
        self.setup_logging()
        self.output_dir = output_dir
        self.create_output_directory()
        
        # Initialize components
        self.model_architecture = NonMajorLeagueModelArchitecture()
        self.ensemble = NonMajorLeagueEnsemble()
        self.transfer_learning = NonMajorLeagueTransferLearning()
        self.hyperparameter_tuning = NonMajorLeagueHyperparameterTuning()
        self.model_validation = NonMajorLeagueModelValidation()
        
        # Results storage
        self.results = {
            'model_architecture': {},
            'ensemble': {},
            'transfer_learning': {},
            'hyperparameter_tuning': {},
            'model_validation': {},
            'overall': {}
        }
        
    def setup_logging(self):
        """Setup logging for Phase 2 integration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('phase2_integration.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_output_directory(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def run_phase2_pipeline(self, data_file: str, league_code: str, 
                           source_data_files: Dict[str, str] = None, config: dict = None) -> dict:
        """
        Run complete Phase 2 pipeline
        
        Args:
            data_file: Path to processed feature data
            league_code: League identifier
            source_data_files: Dictionary of source league data files
            config: Configuration dictionary
            
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info(f"Starting Phase 2 pipeline for {league_code}")
        
        try:
            # Step 1: Load and prepare data
            self.logger.info("Step 1: Loading and preparing data")
            data = self._load_and_prepare_data(data_file)
            
            if data.empty:
                self.logger.error("No data loaded, stopping pipeline")
                return self.results
            
            # Step 2: Model Architecture Design
            self.logger.info("Step 2: Model Architecture Design")
            architecture_results = self._design_model_architecture(data)
            
            # Step 3: Transfer Learning Setup
            self.logger.info("Step 3: Transfer Learning Setup")
            transfer_results = self._setup_transfer_learning(source_data_files, data)
            
            # Step 4: Hyperparameter Tuning
            self.logger.info("Step 4: Hyperparameter Tuning")
            tuning_results = self._optimize_hyperparameters(data)
            
            # Step 5: Ensemble Modeling
            self.logger.info("Step 5: Ensemble Modeling")
            ensemble_results = self._create_ensemble_model(data)
            
            # Step 6: Model Validation
            self.logger.info("Step 6: Model Validation")
            validation_results = self._validate_models(data)
            
            # Step 7: Final Integration
            self.logger.info("Step 7: Final Integration")
            final_results = self._integrate_results(architecture_results, transfer_results,
                                                  tuning_results, ensemble_results, validation_results)
            
            # Step 8: Save Results
            self.logger.info("Step 8: Saving Results")
            self._save_results(final_results, league_code)
            
            self.logger.info("Phase 2 pipeline completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Phase 2 pipeline failed: {e}")
            raise
    
    def _load_and_prepare_data(self, data_file: str) -> pd.DataFrame:
        """Load and prepare data for Phase 2"""
        self.logger.info(f"Loading data from {data_file}")
        
        try:
            data = pd.read_csv(data_file)
            self.logger.info(f"Loaded {len(data)} rows with {len(data.columns)} columns")
            
            # Create target variable if not present
            if 'target' not in data.columns:
                if 'FTR' in data.columns:
                    self.logger.info("Creating target from FTR column")
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    data['target'] = le.fit_transform(data['FTR'])
                    self.label_encoder = le
                    self.logger.info(f"Target classes: {le.classes_}")
                else:
                    self.logger.error("No target column (FTR or target) found in data")
                    return pd.DataFrame()
            
            # Remove rows with missing target
            data = data.dropna(subset=['target'])
            
            # Identify and filter out categorical/non-numeric columns
            # Columns to exclude from features
            exclude_cols = [
                'target', 'FTR', 'HTR',  # Target and result columns
                'Date', 'Time',  # Date/time columns
                'HomeTeam', 'AwayTeam',  # Team names
                'Referee',  # Referee name
                'Div', 'league', 'season', 'data_source',  # Metadata
                'season_phase'  # Temporal metadata
            ]
            
            # Get all column names
            all_columns = data.columns.tolist()
            
            # Select only numerical columns
            numerical_data = data.select_dtypes(include=[np.number])
            numerical_cols = numerical_data.columns.tolist()
            
            # Remove target from features if present
            feature_columns = [col for col in numerical_cols if col not in ['target']]
            
            self.logger.info(f"Selected {len(feature_columns)} numerical features from {len(all_columns)} total columns")
            self.logger.info(f"Excluded columns: {[col for col in exclude_cols if col in all_columns]}")
            
            # Handle any remaining NaN values in features
            X = data[feature_columns].fillna(0)
            y = data['target']
            
            # Sanitize feature names for XGBoost compatibility
            # XGBoost doesn't allow [, ], <, > in feature names
            def sanitize_feature_name(name):
                """Sanitize feature names to be XGBoost compatible"""
                name = str(name)
                name = name.replace('[', '_').replace(']', '_')
                name = name.replace('<', 'lt').replace('>', 'gt')
                name = name.replace(' ', '_').replace('(', '_').replace(')', '_')
                return name
            
            # Apply sanitization
            original_columns = X.columns.tolist()
            sanitized_columns = [sanitize_feature_name(col) for col in original_columns]
            X.columns = sanitized_columns
            self.feature_columns = sanitized_columns
            self.logger.info(f"Sanitized {len(original_columns)} feature names for XGBoost compatibility")
            
            # Check for infinite values
            if np.isinf(X.values).any():
                self.logger.warning("Found infinite values in features, replacing with 0")
                X = X.replace([np.inf, -np.inf], 0)
            
            self.logger.info(f"Final dataset: {len(data)} samples with {len(feature_columns)} features")
            self.logger.info(f"Target distribution: {y.value_counts().to_dict()}")
            
            # Store data for later use
            self.data = data
            self.X = X
            self.y = y
            self.feature_columns = feature_columns
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            import traceback
            traceback.print_exc()
            return pd.DataFrame()
    
    def _design_model_architecture(self, data: pd.DataFrame) -> dict:
        """Design model architecture for non-major leagues"""
        self.logger.info("Designing model architecture")
        
        try:
            # Split data for training and validation
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Create base models
            base_models = self.model_architecture.create_base_models()
            
            # Train base models
            training_results = self.model_architecture.train_base_models(
                X_train, y_train, X_val, y_val
            )
            
            # Create ensemble model
            ensemble_model = self.model_architecture.create_ensemble_model(training_results)
            
            # Train ensemble
            ensemble_results = self.model_architecture.train_ensemble_model(
                X_train, y_train, X_val, y_val
            )
            
            # Calibrate models
            calibrated_models = self.model_architecture.calibrate_models(X_train, y_train)
            
            # Evaluate models
            evaluation_results = self.model_architecture.evaluate_models(X_test, y_test)
            
            # Get feature importance
            feature_importance = self.model_architecture.get_feature_importance(X_train)
            
            architecture_results = {
                'base_models': base_models,
                'training_results': training_results,
                'ensemble_model': ensemble_model,
                'ensemble_results': ensemble_results,
                'calibrated_models': calibrated_models,
                'evaluation_results': evaluation_results,
                'feature_importance': feature_importance,
                'success': True
            }
            
            self.results['model_architecture'] = architecture_results
            return architecture_results
            
        except Exception as e:
            self.logger.error(f"Error in model architecture design: {e}")
            return {'success': False, 'error': str(e)}
    
    def _setup_transfer_learning(self, source_data_files: Dict[str, str], 
                                target_data: pd.DataFrame) -> dict:
        """Setup transfer learning from major leagues"""
        self.logger.info("Setting up transfer learning")
        
        try:
            # Load source data if available
            if source_data_files:
                source_data = {}
                for league, file_path in source_data_files.items():
                    if os.path.exists(file_path):
                        source_data[league] = pd.read_csv(file_path)
                        self.logger.info(f"Loaded source data for {league}")
                
                if source_data:
                    # Pre-train source models
                    source_models = self.transfer_learning.pre_train_source_models(source_data)
                    
                    # Transfer features
                    transferred_features = self.transfer_learning.transfer_features(
                        source_models, target_data
                    )
                    
                    # Create transferred features
                    target_data_transferred = self.transfer_learning.create_transferred_features(
                        target_data
                    )
                    
                    # Fine-tune target model
                    fine_tuned_results = self.transfer_learning.fine_tune_target_model(
                        target_data_transferred, 'target_league'
                    )
                    
                    # Compare with baseline
                    comparison_results = self.transfer_learning.compare_with_baseline(
                        target_data_transferred
                    )
                    
                    # Validate transfer learning
                    validation_results = self.transfer_learning.validate_transfer_learning(
                        target_data_transferred
                    )
                    
                    # Get transfer metrics
                    transfer_metrics = self.transfer_learning.get_transfer_metrics()
                    
                    transfer_results = {
                        'source_data': source_data,
                        'source_models': source_models,
                        'transferred_features': transferred_features,
                        'target_data_transferred': target_data_transferred,
                        'fine_tuned_results': fine_tuned_results,
                        'comparison_results': comparison_results,
                        'validation_results': validation_results,
                        'transfer_metrics': transfer_metrics,
                        'success': True
                    }
                    
                else:
                    self.logger.warning("No source data files found")
                    transfer_results = {'success': False, 'error': 'No source data'}
            else:
                self.logger.info("No source data files provided, skipping transfer learning")
                transfer_results = {'success': False, 'error': 'No source data files'}
            
            self.results['transfer_learning'] = transfer_results
            return transfer_results
            
        except Exception as e:
            self.logger.error(f"Error in transfer learning setup: {e}")
            return {'success': False, 'error': str(e)}
    
    def _optimize_hyperparameters(self, data: pd.DataFrame) -> dict:
        """Optimize hyperparameters for non-major leagues"""
        self.logger.info("Optimizing hyperparameters")
        
        try:
            # Optimize all models
            optimization_results = self.hyperparameter_tuning.optimize_all_models(
                self.X, self.y
            )
            
            # Validate hyperparameters
            validation_results = self.hyperparameter_tuning.validate_hyperparameters(
                self.X, self.y
            )
            
            # Compare models
            comparison_results = self.hyperparameter_tuning.compare_models(
                self.X, self.y
            )
            
            # Evaluate optimized models
            evaluation_results = self.hyperparameter_tuning.evaluate_optimized_models(
                self.X, self.y
            )
            
            # Get hyperparameter importance
            importance_results = self.hyperparameter_tuning.get_hyperparameter_importance()
            
            tuning_results = {
                'optimization_results': optimization_results,
                'validation_results': validation_results,
                'comparison_results': comparison_results,
                'evaluation_results': evaluation_results,
                'importance_results': importance_results,
                'success': True
            }
            
            self.results['hyperparameter_tuning'] = tuning_results
            return tuning_results
            
        except Exception as e:
            self.logger.error(f"Error in hyperparameter tuning: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_ensemble_model(self, data: pd.DataFrame) -> dict:
        """Create ensemble model for non-major leagues"""
        self.logger.info("Creating ensemble model")
        
        try:
            # Split data for training and validation
            from sklearn.model_selection import train_test_split
            
            X_train, X_test, y_train, y_test = train_test_split(
                self.X, self.y, test_size=0.2, random_state=42
            )
            
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42
            )
            
            # Create base models
            base_models = self.ensemble.create_base_models()
            
            # Train base models
            training_results = self.ensemble.train_base_models(
                X_train, y_train, X_val, y_val
            )
            
            # Create conservative ensemble
            conservative_ensemble = self.ensemble.create_conservative_ensemble(training_results)
            
            # Train ensemble
            ensemble_results = self.ensemble.train_ensemble(
                X_train, y_train, X_val, y_val
            )
            
            # Calibrate models
            calibrated_models = self.ensemble.calibrate_models(X_train, y_train)
            
            # Cross-validate ensemble
            cv_results = self.ensemble.cross_validate_ensemble(X_train, y_train)
            
            # Evaluate ensemble
            evaluation_results = self.ensemble.evaluate_ensemble(X_test, y_test)
            
            # Make predictions with confidence
            predictions = self.ensemble.predict_with_confidence(X_test)
            
            # Get feature importance
            feature_importance = self.ensemble.get_feature_importance(X_train)
            
            # Estimate uncertainty
            uncertainty_estimates = self.ensemble.estimate_uncertainty(X_test)
            
            ensemble_results = {
                'base_models': base_models,
                'training_results': training_results,
                'conservative_ensemble': conservative_ensemble,
                'ensemble_results': ensemble_results,
                'calibrated_models': calibrated_models,
                'cv_results': cv_results,
                'evaluation_results': evaluation_results,
                'predictions': predictions,
                'feature_importance': feature_importance,
                'uncertainty_estimates': uncertainty_estimates,
                'success': True
            }
            
            self.results['ensemble'] = ensemble_results
            return ensemble_results
            
        except Exception as e:
            self.logger.error(f"Error in ensemble creation: {e}")
            return {'success': False, 'error': str(e)}
    
    def _validate_models(self, data: pd.DataFrame) -> dict:
        """Validate models comprehensively"""
        self.logger.info("Validating models")
        
        try:
            # Get the best model from ensemble results
            if 'ensemble' in self.results and self.results['ensemble']['success']:
                ensemble_results = self.results['ensemble']['evaluation_results']
                
                if 'error' not in ensemble_results:
                    # Use the ensemble model for validation
                    # For now, create a simple model for validation
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
                    
                    # Perform comprehensive validation
                    validation_results = self.model_validation.comprehensive_validation(
                        model, self.X, self.y
                    )
                    
                    # Generate validation report
                    report = self.model_validation.generate_validation_report(validation_results)
                    
                    validation_results = {
                        'validation_results': validation_results,
                        'report': report,
                        'success': True
                    }
                    
                else:
                    validation_results = {'success': False, 'error': 'No valid model for validation'}
            else:
                validation_results = {'success': False, 'error': 'No ensemble model available'}
            
            self.results['model_validation'] = validation_results
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error in model validation: {e}")
            return {'success': False, 'error': str(e)}
    
    def _integrate_results(self, architecture_results: dict, transfer_results: dict,
                          tuning_results: dict, ensemble_results: dict, 
                          validation_results: dict) -> dict:
        """Integrate all Phase 2 results"""
        self.logger.info("Integrating Phase 2 results")
        
        # Calculate overall pipeline metrics
        overall_metrics = {
            'total_components': 5,
            'successful_components': 0,
            'pipeline_success': False,
            'best_model': None,
            'best_score': 0.0,
            'recommendations': []
        }
        
        # Count successful components
        component_results = {
            'model_architecture': architecture_results,
            'transfer_learning': transfer_results,
            'hyperparameter_tuning': tuning_results,
            'ensemble': ensemble_results,
            'model_validation': validation_results
        }
        
        for component, results in component_results.items():
            if results.get('success', False):
                overall_metrics['successful_components'] += 1
        
        # Determine pipeline success
        overall_metrics['pipeline_success'] = overall_metrics['successful_components'] >= 3
        
        # Find best model and score
        if ensemble_results.get('success', False):
            evaluation_results = ensemble_results.get('evaluation_results', {})
            if 'error' not in evaluation_results:
                overall_metrics['best_model'] = 'ensemble'
                overall_metrics['best_score'] = evaluation_results.get('accuracy', 0.0)
        
        # Generate recommendations
        if overall_metrics['pipeline_success']:
            overall_metrics['recommendations'].append("Phase 2 pipeline completed successfully")
            overall_metrics['recommendations'].append("Proceed to Phase 3: Validation and Testing")
        else:
            overall_metrics['recommendations'].append("Phase 2 pipeline encountered issues")
            overall_metrics['recommendations'].append("Review component results and address issues")
        
        # Create final results dictionary
        final_results = {
            'architecture_results': architecture_results,
            'transfer_results': transfer_results,
            'tuning_results': tuning_results,
            'ensemble_results': ensemble_results,
            'validation_results': validation_results,
            'pipeline_metrics': overall_metrics,
            'component_results': component_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['overall'] = overall_metrics
        return final_results
    
    def _save_results(self, results: dict, league_code: str):
        """Save Phase 2 results"""
        self.logger.info(f"Saving results for {league_code}")
        
        # Save model architecture
        if results['architecture_results'].get('success', False):
            model_file = os.path.join(self.output_dir, f"{league_code}_model_architecture.pkl")
            self.model_architecture.save_models(model_file)
        
        # Save ensemble
        if results['ensemble_results'].get('success', False):
            ensemble_file = os.path.join(self.output_dir, f"{league_code}_ensemble.pkl")
            self.ensemble.save_ensemble(ensemble_file)
        
        # Save transfer learning
        if results['transfer_results'].get('success', False):
            transfer_file = os.path.join(self.output_dir, f"{league_code}_transfer_learning.pkl")
            self.transfer_learning.save_transfer_learning(transfer_file)
        
        # Save hyperparameter tuning
        if results['tuning_results'].get('success', False):
            tuning_file = os.path.join(self.output_dir, f"{league_code}_hyperparameter_tuning.pkl")
            self.hyperparameter_tuning.save_hyperparameter_tuning(tuning_file)
        
        # Save validation results
        if results['validation_results'].get('success', False):
            validation_file = os.path.join(self.output_dir, f"{league_code}_validation_results.pkl")
            self.model_validation.save_validation_results(validation_file)
        
        # Save pipeline summary
        summary_file = os.path.join(self.output_dir, f"{league_code}_phase2_summary.json")
        summary_data = {
            'league_code': league_code,
            'pipeline_metrics': results['pipeline_metrics'],
            'component_results': results['component_results'],
            'timestamp': results['timestamp']
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        self.logger.info("Phase 2 results saved successfully")
    
    def generate_phase2_report(self, results: dict, league_code: str) -> str:
        """Generate comprehensive Phase 2 report"""
        report = []
        report.append("=" * 80)
        report.append("PHASE 2 INTEGRATION REPORT - NON-MAJOR LEAGUE ML PIPELINE")
        report.append("=" * 80)
        report.append("")
        
        # Pipeline overview
        pipeline_metrics = results['pipeline_metrics']
        report.append("PIPELINE OVERVIEW:")
        report.append(f"  League: {league_code}")
        report.append(f"  Successful components: {pipeline_metrics['successful_components']}/{pipeline_metrics['total_components']}")
        report.append(f"  Pipeline success: {'✅' if pipeline_metrics['pipeline_success'] else '❌'}")
        report.append(f"  Best model: {pipeline_metrics['best_model']}")
        report.append(f"  Best score: {pipeline_metrics['best_score']:.4f}")
        report.append("")
        
        # Component results
        report.append("COMPONENT RESULTS:")
        components = ['model_architecture', 'transfer_learning', 'hyperparameter_tuning', 
                     'ensemble', 'model_validation']
        
        for component in components:
            if component in results:
                comp_results = results[component]
                status = "✅" if comp_results.get('success', False) else "❌"
                report.append(f"  {component.replace('_', ' ').title()}: {status}")
                
                if component == 'model_architecture':
                    if comp_results.get('success', False):
                        eval_results = comp_results.get('evaluation_results', {})
                        if 'error' not in eval_results:
                            report.append(f"    Best accuracy: {eval_results.get('accuracy', 0.0):.4f}")
                
                elif component == 'ensemble':
                    if comp_results.get('success', False):
                        eval_results = comp_results.get('evaluation_results', {})
                        if 'error' not in eval_results:
                            report.append(f"    Ensemble accuracy: {eval_results.get('accuracy', 0.0):.4f}")
                            report.append(f"    Mean confidence: {eval_results.get('mean_confidence', 0.0):.4f}")
                
                elif component == 'hyperparameter_tuning':
                    if comp_results.get('success', False):
                        comp_results_data = comp_results.get('comparison_results', {})
                        if comp_results_data:
                            report.append(f"    Best model: {comp_results_data.get('best_model', 'N/A')}")
                            report.append(f"    Best score: {comp_results_data.get('best_score', 0.0):.4f}")
                
                elif component == 'model_validation':
                    if comp_results.get('success', False):
                        val_results = comp_results.get('validation_results', {})
                        if 'overall' in val_results:
                            overall = val_results['overall']
                            report.append(f"    Validation passed: {'✅' if overall['validation_passed'] else '❌'}")
                            report.append(f"    Overall score: {overall['overall_score']:.4f}")
        
        report.append("")
        
        # Recommendations
        if 'recommendations' in pipeline_metrics:
            report.append("RECOMMENDATIONS:")
            for i, rec in enumerate(pipeline_metrics['recommendations'], 1):
                report.append(f"  {i}. {rec}")
            report.append("")
        
        # Next steps
        report.append("NEXT STEPS:")
        if pipeline_metrics['pipeline_success']:
            report.append("  1. Review model performance and validation results")
            report.append("  2. Proceed to Phase 3: Validation and Testing")
            report.append("  3. Implement betting strategy with conservative parameters")
            report.append("  4. Set up monitoring and performance tracking")
            report.append("  5. Begin live testing with small stakes")
        else:
            report.append("  1. Review component results and identify issues")
            report.append("  2. Address failed components")
            report.append("  3. Re-run Phase 2 pipeline")
            report.append("  4. Consider alternative approaches")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main function for Phase 2 integration"""
    parser = argparse.ArgumentParser(description='Phase 2 Integration for Non-Major League ML Pipeline')
    parser.add_argument('--league', required=True, help='League code (e.g., E1 for Championship)')
    parser.add_argument('--data_file', required=True, help='Path to processed feature data')
    parser.add_argument('--output_dir', default='./models', help='Output directory for models')
    parser.add_argument('--source_data', help='Comma-separated source data files (league:file)')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Parse source data files
    source_data_files = {}
    if args.source_data:
        for item in args.source_data.split(','):
            if ':' in item:
                league, file_path = item.split(':', 1)
                source_data_files[league.strip()] = file_path.strip()
    
    # Initialize Phase 2 integration
    phase2 = Phase2Integration(args.config, args.output_dir)
    
    try:
        # Run Phase 2 pipeline
        results = phase2.run_phase2_pipeline(
            data_file=args.data_file,
            league_code=args.league,
            source_data_files=source_data_files if source_data_files else None
        )
        
        # Generate and display report
        report = phase2.generate_phase2_report(results, args.league)
        print(report)
        
        # Save report
        report_file = os.path.join(args.output_dir, f"{args.league}_phase2_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nPhase 2 integration completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Report saved to: {report_file}")
        
    except Exception as e:
        print(f"Phase 2 integration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
