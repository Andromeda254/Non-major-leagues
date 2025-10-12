#!/usr/bin/env python3
"""
Phase 1 Integration Script for Non-Major League ML Pipeline

This script integrates all Phase 1 components:
1. Data Collection
2. Data Preprocessing
3. Data Validation
4. Feature Engineering

Usage:
    python phase1_integration.py --league E1 --seasons 2324,2223 --output_dir ./data
"""

import argparse
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import json

# Import our custom modules
from non_major_league_data_collector import NonMajorLeagueDataCollector
from non_major_league_preprocessor import NonMajorLeaguePreprocessor
from non_major_league_validator import NonMajorLeagueValidator
from non_major_league_feature_engineer import NonMajorLeagueFeatureEngineer

class Phase1Integration:
    """
    Phase 1 Integration for Non-Major League ML Pipeline
    
    This class orchestrates the complete Phase 1 workflow:
    - Data collection from multiple sources
    - Data preprocessing and cleaning
    - Data validation and quality assessment
    - Feature engineering for non-major leagues
    """
    
    def __init__(self, config_file: str = None, output_dir: str = "./data"):
        """
        Initialize Phase 1 integration
        
        Args:
            config_file: Path to configuration file
            output_dir: Output directory for processed data
        """
        self.setup_logging()
        self.output_dir = output_dir
        self.create_output_directory()
        
        # Initialize components
        self.data_collector = NonMajorLeagueDataCollector(config_file)
        self.preprocessor = NonMajorLeaguePreprocessor()
        self.validator = NonMajorLeagueValidator()
        self.feature_engineer = NonMajorLeagueFeatureEngineer()
        
        # Results storage
        self.results = {
            'data_collection': {},
            'preprocessing': {},
            'validation': {},
            'feature_engineering': {},
            'overall': {}
        }
        
    def setup_logging(self):
        """Setup logging for Phase 1 integration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('phase1_integration.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_output_directory(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def run_phase1_pipeline(self, league_code: str, seasons: list, 
                           collect_live_odds: bool = True, config: dict = None) -> dict:
        """
        Run complete Phase 1 pipeline
        
        Args:
            league_code: League identifier (e.g., 'E1' for Championship)
            seasons: List of seasons to process (e.g., ['2324', '2223'])
            collect_live_odds: Whether to collect live odds data
            config: Configuration dictionary
            
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info(f"Starting Phase 1 pipeline for {league_code}, seasons: {seasons}")
        
        try:
            # Step 1: Data Collection
            self.logger.info("Step 1: Data Collection")
            raw_data = self._collect_data(league_code, seasons, collect_live_odds)
            
            if raw_data.empty:
                self.logger.error("No data collected, stopping pipeline")
                return self.results
            
            # Step 2: Data Preprocessing
            self.logger.info("Step 2: Data Preprocessing")
            preprocessed_data = self._preprocess_data(raw_data)
            
            # Step 3: Data Validation
            self.logger.info("Step 3: Data Validation")
            validation_results = self._validate_data(preprocessed_data, league_code)
            
            # Step 4: Feature Engineering
            self.logger.info("Step 4: Feature Engineering")
            features_data = self._engineer_features(preprocessed_data, league_code)
            
            # Step 5: Final Integration
            self.logger.info("Step 5: Final Integration")
            final_results = self._integrate_results(raw_data, preprocessed_data, 
                                                   features_data, validation_results)
            
            # Step 6: Save Results
            self.logger.info("Step 6: Saving Results")
            self._save_results(final_results, league_code)
            
            self.logger.info("Phase 1 pipeline completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Phase 1 pipeline failed: {e}")
            raise
    
    def _collect_data(self, league_code: str, seasons: list, collect_live_odds: bool) -> pd.DataFrame:
        """Collect data from multiple sources"""
        self.logger.info(f"Collecting data for {league_code}, seasons: {seasons}")
        
        # Collect historical data
        historical_data = self.data_collector.collect_historical_data(league_code, seasons)
        
        if historical_data.empty:
            self.logger.warning("No historical data collected")
            return pd.DataFrame()
        
        # Collect live odds if requested
        live_odds = None
        if collect_live_odds:
            try:
                live_odds = self.data_collector.collect_live_odds(league_code)
                if live_odds is not None and not live_odds.empty:
                    self.logger.info(f"Collected live odds for {len(live_odds)} matches")
            except Exception as e:
                self.logger.warning(f"Failed to collect live odds: {e}")
        
        # Collect additional data
        additional_data = self.data_collector.collect_additional_data(league_code)
        
        # Store results
        self.results['data_collection'] = {
            'historical_data_shape': historical_data.shape,
            'live_odds_shape': live_odds.shape if live_odds is not None else (0, 0),
            'additional_data_keys': list(additional_data.keys()),
            'success': True
        }
        
        return historical_data
    
    def _preprocess_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """Preprocess raw data"""
        self.logger.info(f"Preprocessing {len(raw_data)} rows of data")
        
        try:
            preprocessed_data = self.preprocessor.preprocess_pipeline(raw_data)
            
            self.results['preprocessing'] = {
                'input_shape': raw_data.shape,
                'output_shape': preprocessed_data.shape,
                'columns_added': len(preprocessed_data.columns) - len(raw_data.columns),
                'success': True
            }
            
            return preprocessed_data
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            self.results['preprocessing'] = {
                'success': False,
                'error': str(e)
            }
            raise
    
    def _validate_data(self, data: pd.DataFrame, league_code: str) -> dict:
        """Validate preprocessed data"""
        self.logger.info(f"Validating {len(data)} rows of data")
        
        try:
            validation_results = self.validator.validate_dataset(data, league_code)
            
            self.results['validation'] = {
                'overall_score': validation_results.get('overall_score', 0.0),
                'data_quality_score': validation_results.get('data_quality', {}).get('score', 0.0),
                'statistical_score': validation_results.get('statistical_validation', {}).get('score', 0.0),
                'recommendations_count': len(validation_results.get('recommendations', [])),
                'success': True
            }
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Validation failed: {e}")
            self.results['validation'] = {
                'success': False,
                'error': str(e)
            }
            raise
    
    def _engineer_features(self, data: pd.DataFrame, league_code: str) -> pd.DataFrame:
        """Engineer features for non-major leagues"""
        self.logger.info(f"Engineering features for {len(data)} rows of data")
        
        try:
            features_data = self.feature_engineer.create_all_features(data, league_code)
            
            # Get feature importance
            feature_importance = self.feature_engineer.get_feature_importance(features_data)
            
            # Select top features
            top_features = self.feature_engineer.select_top_features(features_data, n_features=50)
            
            self.results['feature_engineering'] = {
                'input_shape': data.shape,
                'output_shape': features_data.shape,
                'features_created': len(features_data.columns) - len(data.columns),
                'top_features_count': len(top_features),
                'feature_importance_available': not feature_importance.empty,
                'success': True
            }
            
            return features_data
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            self.results['feature_engineering'] = {
                'success': False,
                'error': str(e)
            }
            raise
    
    def _integrate_results(self, raw_data: pd.DataFrame, preprocessed_data: pd.DataFrame,
                          features_data: pd.DataFrame, validation_results: dict) -> dict:
        """Integrate all Phase 1 results"""
        self.logger.info("Integrating Phase 1 results")
        
        # Calculate overall pipeline metrics
        overall_metrics = {
            'total_rows_processed': len(features_data),
            'total_features_created': len(features_data.columns),
            'data_quality_score': validation_results.get('overall_score', 0.0),
            'pipeline_success': all([
                self.results['data_collection'].get('success', False),
                self.results['preprocessing'].get('success', False),
                self.results['validation'].get('success', False),
                self.results['feature_engineering'].get('success', False)
            ])
        }
        
        # Create final results dictionary
        final_results = {
            'raw_data': raw_data,
            'preprocessed_data': preprocessed_data,
            'features_data': features_data,
            'validation_results': validation_results,
            'pipeline_metrics': overall_metrics,
            'component_results': self.results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['overall'] = overall_metrics
        
        return final_results
    
    def _save_results(self, results: dict, league_code: str):
        """Save Phase 1 results"""
        self.logger.info(f"Saving results for {league_code}")
        
        # Save processed data
        if 'features_data' in results:
            features_file = os.path.join(self.output_dir, f"{league_code}_features.csv")
            results['features_data'].to_csv(features_file, index=False)
            self.logger.info(f"Features data saved to {features_file}")
        
        # Save validation report
        if 'validation_results' in results:
            report_file = os.path.join(self.output_dir, f"{league_code}_validation_report.txt")
            self.validator.generate_validation_report(results['validation_results'], report_file)
            self.logger.info(f"Validation report saved to {report_file}")
        
        # Save pipeline summary
        summary_file = os.path.join(self.output_dir, f"{league_code}_phase1_summary.json")
        summary_data = {
            'league_code': league_code,
            'pipeline_metrics': results['pipeline_metrics'],
            'component_results': results['component_results'],
            'timestamp': results['timestamp']
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        self.logger.info(f"Pipeline summary saved to {summary_file}")
        
        # Save preprocessors and feature engineer state
        preprocessor_file = os.path.join(self.output_dir, f"{league_code}_preprocessor.pkl")
        self.preprocessor.save_preprocessors(preprocessor_file)
        
        feature_engineer_file = os.path.join(self.output_dir, f"{league_code}_feature_engineer.pkl")
        self.feature_engineer.save_feature_engineer(feature_engineer_file)
        
        self.logger.info("Phase 1 results saved successfully")
    
    def generate_phase1_report(self, results: dict, league_code: str) -> str:
        """Generate comprehensive Phase 1 report"""
        report = []
        report.append("=" * 80)
        report.append("PHASE 1 INTEGRATION REPORT - NON-MAJOR LEAGUE ML PIPELINE")
        report.append("=" * 80)
        report.append("")
        
        # Pipeline overview
        pipeline_metrics = results['pipeline_metrics']
        report.append("PIPELINE OVERVIEW:")
        report.append(f"  League: {league_code}")
        report.append(f"  Total rows processed: {pipeline_metrics['total_rows_processed']:,}")
        report.append(f"  Total features created: {pipeline_metrics['total_features_created']}")
        report.append(f"  Data quality score: {pipeline_metrics['data_quality_score']:.2f}/1.00")
        report.append(f"  Pipeline success: {'✅' if pipeline_metrics['pipeline_success'] else '❌'}")
        report.append("")
        
        # Component results
        report.append("COMPONENT RESULTS:")
        components = ['data_collection', 'preprocessing', 'validation', 'feature_engineering']
        
        for component in components:
            if component in results['component_results']:
                comp_results = results['component_results'][component]
                status = "✅" if comp_results.get('success', False) else "❌"
                report.append(f"  {component.replace('_', ' ').title()}: {status}")
                
                if component == 'data_collection':
                    report.append(f"    Historical data: {comp_results.get('historical_data_shape', (0, 0))}")
                    report.append(f"    Live odds: {comp_results.get('live_odds_shape', (0, 0))}")
                elif component == 'preprocessing':
                    report.append(f"    Input shape: {comp_results.get('input_shape', (0, 0))}")
                    report.append(f"    Output shape: {comp_results.get('output_shape', (0, 0))}")
                    report.append(f"    Columns added: {comp_results.get('columns_added', 0)}")
                elif component == 'validation':
                    report.append(f"    Overall score: {comp_results.get('overall_score', 0.0):.2f}")
                    report.append(f"    Data quality: {comp_results.get('data_quality_score', 0.0):.2f}")
                    report.append(f"    Recommendations: {comp_results.get('recommendations_count', 0)}")
                elif component == 'feature_engineering':
                    report.append(f"    Features created: {comp_results.get('features_created', 0)}")
                    report.append(f"    Top features: {comp_results.get('top_features_count', 0)}")
        
        report.append("")
        
        # Validation details
        if 'validation_results' in results:
            validation = results['validation_results']
            report.append("VALIDATION DETAILS:")
            report.append(f"  Overall score: {validation.get('overall_score', 0.0):.2f}/1.00")
            
            if 'recommendations' in validation and validation['recommendations']:
                report.append("  Recommendations:")
                for i, rec in enumerate(validation['recommendations'], 1):
                    report.append(f"    {i}. {rec}")
            
            report.append("")
        
        # Next steps
        report.append("NEXT STEPS:")
        report.append("  1. Review validation results and recommendations")
        report.append("  2. Proceed to Phase 2: Model Development")
        report.append("  3. Implement transfer learning from major leagues")
        report.append("  4. Set up ensemble modeling approach")
        report.append("  5. Configure conservative betting strategy")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main function for Phase 1 integration"""
    parser = argparse.ArgumentParser(description='Phase 1 Integration for Non-Major League ML Pipeline')
    parser.add_argument('--league', required=True, help='League code (e.g., E1 for Championship)')
    parser.add_argument('--seasons', required=True, help='Comma-separated list of seasons (e.g., 2324,2223)')
    parser.add_argument('--output_dir', default='./data', help='Output directory for processed data')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--no_live_odds', action='store_true', help='Skip live odds collection')
    
    args = parser.parse_args()
    
    # Parse seasons
    seasons = [s.strip() for s in args.seasons.split(',')]
    
    # Initialize Phase 1 integration
    phase1 = Phase1Integration(args.config, args.output_dir)
    
    try:
        # Run Phase 1 pipeline
        results = phase1.run_phase1_pipeline(
            league_code=args.league,
            seasons=seasons,
            collect_live_odds=not args.no_live_odds
        )
        
        # Generate and display report
        report = phase1.generate_phase1_report(results, args.league)
        print(report)
        
        # Save report
        report_file = os.path.join(args.output_dir, f"{args.league}_phase1_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nPhase 1 integration completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Report saved to: {report_file}")
        
    except Exception as e:
        print(f"Phase 1 integration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
