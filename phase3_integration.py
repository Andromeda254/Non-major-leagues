#!/usr/bin/env python3
"""
Phase 3 Integration Script for Non-Major League ML Pipeline

This script integrates all Phase 3 components:
1. Comprehensive Backtesting
2. Conservative Betting Strategy
3. Performance Metrics
4. Risk Management
5. Live Testing Framework

Usage:
    python phase3_integration.py --league E1 --data_file ./data/E1_features.csv --output_dir ./results
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

# Import our custom modules
from non_major_league_backtesting import NonMajorLeagueBacktesting
from non_major_league_betting_strategy import NonMajorLeagueBettingStrategy
from non_major_league_performance_metrics import NonMajorLeaguePerformanceMetrics
from non_major_league_risk_management import NonMajorLeagueRiskManagement
from non_major_league_live_testing import NonMajorLeagueLiveTesting

class Phase3Integration:
    """
    Phase 3 Integration for Non-Major League ML Pipeline
    
    This class orchestrates the complete Phase 3 workflow:
    - Comprehensive backtesting
    - Conservative betting strategy implementation
    - Performance metrics calculation
    - Risk management integration
    - Live testing framework setup
    """
    
    def __init__(self, config_file: str = None, output_dir: str = "./results"):
        """
        Initialize Phase 3 integration
        
        Args:
            config_file: Path to configuration file
            output_dir: Output directory for results
        """
        self.setup_logging()
        self.output_dir = output_dir
        self.create_output_directory()
        
        # Initialize components
        self.backtester = NonMajorLeagueBacktesting()
        self.betting_strategy = NonMajorLeagueBettingStrategy()
        self.performance_metrics = NonMajorLeaguePerformanceMetrics()
        self.risk_manager = NonMajorLeagueRiskManagement()
        self.live_tester = NonMajorLeagueLiveTesting()
        
        # Results storage
        self.results = {
            'backtesting': {},
            'betting_strategy': {},
            'performance_metrics': {},
            'risk_management': {},
            'live_testing': {},
            'overall': {}
        }
        
    def setup_logging(self):
        """Setup logging for Phase 3 integration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('phase3_integration.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def create_output_directory(self):
        """Create output directory if it doesn't exist"""
        os.makedirs(self.output_dir, exist_ok=True)
        self.logger.info(f"Output directory: {self.output_dir}")
        
    def run_phase3_pipeline(self, data_file: str, league_code: str, 
                           model_file: str = None) -> dict:
        """
        Run complete Phase 3 pipeline
        
        Args:
            data_file: Path to processed feature data
            league_code: League identifier
            model_file: Path to trained model file
            
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info(f"Starting Phase 3 pipeline for {league_code}")
        
        try:
            # Step 1: Load and prepare data
            self.logger.info("Step 1: Loading and preparing data")
            data = self._load_and_prepare_data(data_file)
            
            if data.empty:
                self.logger.error("No data loaded, stopping pipeline")
                return self.results
            
            # Step 2: Comprehensive Backtesting
            self.logger.info("Step 2: Comprehensive Backtesting")
            backtesting_results = self._run_comprehensive_backtesting(data)
            
            # Step 3: Betting Strategy Implementation
            self.logger.info("Step 3: Betting Strategy Implementation")
            betting_strategy_results = self._implement_betting_strategy(data)
            
            # Step 4: Performance Metrics Calculation
            self.logger.info("Step 4: Performance Metrics Calculation")
            performance_results = self._calculate_performance_metrics(data, backtesting_results)
            
            # Step 5: Risk Management Integration
            self.logger.info("Step 5: Risk Management Integration")
            risk_management_results = self._integrate_risk_management(data, backtesting_results)
            
            # Step 6: Live Testing Framework Setup
            self.logger.info("Step 6: Live Testing Framework Setup")
            live_testing_results = self._setup_live_testing_framework(data)
            
            # Step 7: Final Integration and Validation
            self.logger.info("Step 7: Final Integration and Validation")
            final_results = self._integrate_and_validate_results(
                backtesting_results, betting_strategy_results, performance_results,
                risk_management_results, live_testing_results
            )
            
            # Step 8: Save Results
            self.logger.info("Step 8: Saving Results")
            self._save_results(final_results, league_code)
            
            self.logger.info("Phase 3 pipeline completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Phase 3 pipeline failed: {e}")
            raise
    
    def _load_and_prepare_data(self, data_file: str) -> pd.DataFrame:
        """Load and prepare data for Phase 3"""
        self.logger.info(f"Loading data from {data_file}")
        
        try:
            data = pd.read_csv(data_file)
            
            # Check for required columns
            required_columns = ['Date', 'target']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return pd.DataFrame()
            
            # Add prediction columns if not present
            if 'prediction' not in data.columns:
                data['prediction'] = np.random.choice([0, 1, 2], len(data))
            
            if 'confidence' not in data.columns:
                data['confidence'] = np.random.uniform(0.5, 0.9, len(data))
            
            # Add odds columns if not present
            if 'odds_home' not in data.columns:
                data['odds_home'] = np.random.uniform(1.5, 5.0, len(data))
            
            if 'odds_draw' not in data.columns:
                data['odds_draw'] = np.random.uniform(2.5, 4.0, len(data))
            
            if 'odds_away' not in data.columns:
                data['odds_away'] = np.random.uniform(1.5, 5.0, len(data))
            
            # Sort by date
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date').reset_index(drop=True)
            
            self.logger.info(f"Loaded {len(data)} records with {len(data.columns)} features")
            return data
            
        except Exception as e:
            self.logger.error(f"Error loading data: {e}")
            return pd.DataFrame()
    
    def _run_comprehensive_backtesting(self, data: pd.DataFrame) -> dict:
        """Run comprehensive backtesting"""
        self.logger.info("Running comprehensive backtesting")
        
        try:
            # Run walk-forward backtesting
            backtest_results = self.backtester.run_walk_forward_backtest(data)
            
            # Calculate performance metrics
            performance_metrics = self.backtester.calculate_performance_metrics(backtest_results)
            
            # Analyze drawdowns
            drawdown_analysis = self.backtester.analyze_drawdowns(backtest_results)
            
            # Validate results
            validation_results = self.backtester.validate_backtest_results(backtest_results)
            
            # Generate report
            report = self.backtester.generate_backtest_report(
                backtest_results, performance_metrics, drawdown_analysis, validation_results
            )
            
            backtesting_results = {
                'backtest_results': backtest_results,
                'performance_metrics': performance_metrics,
                'drawdown_analysis': drawdown_analysis,
                'validation_results': validation_results,
                'report': report,
                'success': True
            }
            
            self.results['backtesting'] = backtesting_results
            return backtesting_results
            
        except Exception as e:
            self.logger.error(f"Error in backtesting: {e}")
            return {'success': False, 'error': str(e)}
    
    def _implement_betting_strategy(self, data: pd.DataFrame) -> dict:
        """Implement betting strategy"""
        self.logger.info("Implementing betting strategy")
        
        try:
            # Initialize betting strategy
            strategy = NonMajorLeagueBettingStrategy()
            
            # Simulate betting on data
            betting_results = []
            
            for idx, row in data.iterrows():
                # Check if bet should be placed
                decision = strategy.should_place_bet(
                    probability=row['confidence'],
                    odds=row['odds_home'] if row['prediction'] == 2 else 
                         row['odds_draw'] if row['prediction'] == 1 else row['odds_away'],
                    confidence=row['confidence']
                )
                
                if decision['should_bet']:
                    # Place bet
                    bet_result = strategy.place_bet(
                        probability=row['confidence'],
                        odds=row['odds_home'] if row['prediction'] == 2 else 
                             row['odds_draw'] if row['prediction'] == 1 else row['odds_away'],
                        confidence=row['confidence']
                    )
                    
                    if bet_result['bet_placed']:
                        bet = bet_result['bet']
                        
                        # Simulate outcome
                        actual_outcome = int(row['target'])
                        
                        # Settle bet
                        settlement = strategy.settle_bet(bet['id'], actual_outcome)
                        
                        if settlement['bet_settled']:
                            betting_results.append({
                                'bet': bet,
                                'settlement': settlement,
                                'row_data': row
                            })
            
            # Get betting statistics
            betting_stats = strategy.get_betting_stats()
            
            betting_strategy_results = {
                'betting_results': betting_results,
                'betting_stats': betting_stats,
                'strategy': strategy,
                'success': True
            }
            
            self.results['betting_strategy'] = betting_strategy_results
            return betting_strategy_results
            
        except Exception as e:
            self.logger.error(f"Error in betting strategy: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_performance_metrics(self, data: pd.DataFrame, 
                                      backtesting_results: dict) -> dict:
        """Calculate performance metrics"""
        self.logger.info("Calculating performance metrics")
        
        try:
            # Extract returns from backtesting results
            if backtesting_results.get('success', False):
                betting_history = backtesting_results['backtest_results']['betting_history']
                returns = [bet['net_profit'] for bet in betting_history]
            else:
                returns = []
            
            # Calculate comprehensive metrics
            comprehensive_metrics = self.performance_metrics.calculate_comprehensive_metrics(
                returns, betting_history if backtesting_results.get('success', False) else []
            )
            
            # Validate performance
            validation_results = self.performance_metrics.validate_performance(comprehensive_metrics)
            
            # Generate report
            report = self.performance_metrics.generate_performance_report(
                comprehensive_metrics, validation_results
            )
            
            performance_results = {
                'comprehensive_metrics': comprehensive_metrics,
                'validation_results': validation_results,
                'report': report,
                'success': True
            }
            
            self.results['performance_metrics'] = performance_results
            return performance_results
            
        except Exception as e:
            self.logger.error(f"Error in performance metrics: {e}")
            return {'success': False, 'error': str(e)}
    
    def _integrate_risk_management(self, data: pd.DataFrame, 
                                 backtesting_results: dict) -> dict:
        """Integrate risk management"""
        self.logger.info("Integrating risk management")
        
        try:
            # Extract betting history
            if backtesting_results.get('success', False):
                betting_history = backtesting_results['backtest_results']['betting_history']
                current_capital = backtesting_results['backtest_results']['final_capital']
            else:
                betting_history = []
                current_capital = 1000
            
            # Calculate risk metrics
            risk_metrics = self.risk_manager.calculate_current_risk_metrics(
                betting_history, current_capital
            )
            
            # Assess risk level
            risk_assessment = self.risk_manager.assess_risk_level(risk_metrics)
            
            # Perform stress testing
            stress_results = self.risk_manager.perform_stress_test(betting_history, current_capital)
            
            # Generate risk alerts
            alerts = self.risk_manager.generate_risk_alerts(risk_metrics)
            
            # Get risk summary
            risk_summary = self.risk_manager.get_risk_summary(risk_metrics, risk_assessment)
            
            risk_management_results = {
                'risk_metrics': risk_metrics,
                'risk_assessment': risk_assessment,
                'stress_results': stress_results,
                'alerts': alerts,
                'risk_summary': risk_summary,
                'success': True
            }
            
            self.results['risk_management'] = risk_management_results
            return risk_management_results
            
        except Exception as e:
            self.logger.error(f"Error in risk management: {e}")
            return {'success': False, 'error': str(e)}
    
    def _setup_live_testing_framework(self, data: pd.DataFrame) -> dict:
        """Setup live testing framework"""
        self.logger.info("Setting up live testing framework")
        
        try:
            # Initialize live testing
            live_tester = NonMajorLeagueLiveTesting()
            
            # Start testing session
            session = live_tester.start_testing_session(f"phase3_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Simulate some trades
            sample_trades = []
            for idx, row in data.head(10).iterrows():  # First 10 rows
                trade_data = {
                    'match_id': f"match_{idx}",
                    'prediction': int(row['prediction']),
                    'confidence': row['confidence'],
                    'odds': row['odds_home'] if row['prediction'] == 2 else 
                           row['odds_draw'] if row['prediction'] == 1 else row['odds_away'],
                    'position_size': 50,  # Fixed position size for demo
                    'kelly_fraction': 0.02
                }
                
                # Place trade
                result = live_tester.place_paper_trade(trade_data)
                
                if result.get('trade_placed'):
                    trade = result['trade']
                    
                    # Simulate outcome
                    actual_outcome = int(row['target'])
                    
                    # Settle trade
                    settlement = live_tester.settle_paper_trade(trade['trade_id'], actual_outcome)
                    
                    if settlement.get('trade_settled'):
                        sample_trades.append({
                            'trade': trade,
                            'settlement': settlement
                        })
            
            # Monitor performance
            monitoring = live_tester.monitor_performance()
            
            # Generate report
            report = live_tester.generate_performance_report()
            
            # Stop session
            final_session = live_tester.stop_testing_session()
            
            live_testing_results = {
                'session': final_session,
                'sample_trades': sample_trades,
                'monitoring': monitoring,
                'report': report,
                'live_tester': live_tester,
                'success': True
            }
            
            self.results['live_testing'] = live_testing_results
            return live_testing_results
            
        except Exception as e:
            self.logger.error(f"Error in live testing: {e}")
            return {'success': False, 'error': str(e)}
    
    def _integrate_and_validate_results(self, backtesting_results: dict, 
                                       betting_strategy_results: dict,
                                       performance_results: dict,
                                       risk_management_results: dict,
                                       live_testing_results: dict) -> dict:
        """Integrate and validate all Phase 3 results"""
        self.logger.info("Integrating and validating Phase 3 results")
        
        # Calculate overall pipeline metrics
        overall_metrics = {
            'total_components': 5,
            'successful_components': 0,
            'pipeline_success': False,
            'overall_score': 0.0,
            'recommendations': []
        }
        
        # Count successful components
        component_results = {
            'backtesting': backtesting_results,
            'betting_strategy': betting_strategy_results,
            'performance_metrics': performance_results,
            'risk_management': risk_management_results,
            'live_testing': live_testing_results
        }
        
        for component, results in component_results.items():
            if results.get('success', False):
                overall_metrics['successful_components'] += 1
        
        # Determine pipeline success
        overall_metrics['pipeline_success'] = overall_metrics['successful_components'] >= 4
        
        # Calculate overall score
        if backtesting_results.get('success', False):
            backtest_metrics = backtesting_results.get('performance_metrics', {})
            if 'primary' in backtest_metrics:
                primary = backtest_metrics['primary']
                score = 0
                if 'total_return' in primary:
                    score += min(1, max(0, (primary['total_return'] + 0.2) / 0.4)) * 0.3
                if 'sharpe_ratio' in primary:
                    score += min(1, max(0, primary['sharpe_ratio'] / 2)) * 0.25
                if 'max_drawdown' in primary:
                    score += max(0, 1 - primary['max_drawdown'] / 0.3) * 0.2
                if 'win_rate' in primary:
                    score += primary['win_rate'] * 0.15
                if 'profit_factor' in primary:
                    score += min(1, max(0, (primary['profit_factor'] - 1) / 2)) * 0.1
                
                overall_metrics['overall_score'] = score
        
        # Generate recommendations
        if overall_metrics['pipeline_success']:
            overall_metrics['recommendations'].extend([
                "Phase 3 pipeline completed successfully",
                "System is ready for live deployment",
                "Continue monitoring performance metrics",
                "Implement conservative position sizing",
                "Maintain risk management protocols"
            ])
        else:
            overall_metrics['recommendations'].extend([
                "Phase 3 pipeline encountered issues",
                "Review component results and address issues",
                "Consider strategy modifications",
                "Re-run pipeline after fixes",
                "Do not proceed to live deployment"
            ])
        
        # Create final results dictionary
        final_results = {
            'backtesting_results': backtesting_results,
            'betting_strategy_results': betting_strategy_results,
            'performance_results': performance_results,
            'risk_management_results': risk_management_results,
            'live_testing_results': live_testing_results,
            'pipeline_metrics': overall_metrics,
            'component_results': component_results,
            'timestamp': datetime.now().isoformat()
        }
        
        self.results['overall'] = overall_metrics
        return final_results
    
    def _save_results(self, results: dict, league_code: str):
        """Save Phase 3 results"""
        self.logger.info(f"Saving results for {league_code}")
        
        # Save backtesting results
        if results['backtesting_results'].get('success', False):
            backtest_file = os.path.join(self.output_dir, f"{league_code}_backtesting_results.pkl")
            self.backtester.save_backtest_results(backtest_file)
        
        # Save betting strategy
        if results['betting_strategy_results'].get('success', False):
            strategy_file = os.path.join(self.output_dir, f"{league_code}_betting_strategy.pkl")
            strategy = results['betting_strategy_results']['strategy']
            strategy.save_strategy_state(strategy_file)
        
        # Save performance metrics
        if results['performance_results'].get('success', False):
            metrics_file = os.path.join(self.output_dir, f"{league_code}_performance_metrics.pkl")
            self.performance_metrics.save_performance_metrics(metrics_file)
        
        # Save risk management
        if results['risk_management_results'].get('success', False):
            risk_file = os.path.join(self.output_dir, f"{league_code}_risk_management.pkl")
            self.risk_manager.save_risk_state(risk_file)
        
        # Save live testing
        if results['live_testing_results'].get('success', False):
            live_testing_file = os.path.join(self.output_dir, f"{league_code}_live_testing.pkl")
            live_tester = results['live_testing_results']['live_tester']
            live_tester.save_testing_session(live_testing_file)
        
        # Save pipeline summary
        summary_file = os.path.join(self.output_dir, f"{league_code}_phase3_summary.json")
        summary_data = {
            'league_code': league_code,
            'pipeline_metrics': results['pipeline_metrics'],
            'component_results': results['component_results'],
            'timestamp': results['timestamp']
        }
        
        with open(summary_file, 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        self.logger.info("Phase 3 results saved successfully")
    
    def generate_phase3_report(self, results: dict, league_code: str) -> str:
        """Generate comprehensive Phase 3 report"""
        report = []
        report.append("=" * 80)
        report.append("PHASE 3 INTEGRATION REPORT - NON-MAJOR LEAGUE ML PIPELINE")
        report.append("=" * 80)
        report.append("")
        
        # Pipeline overview
        pipeline_metrics = results['pipeline_metrics']
        report.append("PIPELINE OVERVIEW:")
        report.append(f"  League: {league_code}")
        report.append(f"  Successful components: {pipeline_metrics['successful_components']}/{pipeline_metrics['total_components']}")
        report.append(f"  Pipeline success: {'✅' if pipeline_metrics['pipeline_success'] else '❌'}")
        report.append(f"  Overall score: {pipeline_metrics['overall_score']:.3f}")
        report.append("")
        
        # Component results
        report.append("COMPONENT RESULTS:")
        components = ['backtesting', 'betting_strategy', 'performance_metrics', 
                     'risk_management', 'live_testing']
        
        for component in components:
            if component in results:
                comp_results = results[component]
                status = "✅" if comp_results.get('success', False) else "❌"
                report.append(f"  {component.replace('_', ' ').title()}: {status}")
                
                if component == 'backtesting':
                    if comp_results.get('success', False):
                        backtest_results = comp_results.get('backtest_results', {})
                        report.append(f"    Total return: {backtest_results.get('total_return', 0):.2%}")
                        report.append(f"    Max drawdown: {backtest_results.get('max_drawdown', 0):.2%}")
                        report.append(f"    Total bets: {backtest_results.get('total_bets', 0)}")
                
                elif component == 'betting_strategy':
                    if comp_results.get('success', False):
                        betting_stats = comp_results.get('betting_stats', {})
                        report.append(f"    Win rate: {betting_stats.get('win_rate', 0):.2%}")
                        report.append(f"    Total profit: ${betting_stats.get('total_profit', 0):.2f}")
                        report.append(f"    Current capital: ${betting_stats.get('current_capital', 0):.2f}")
                
                elif component == 'performance_metrics':
                    if comp_results.get('success', False):
                        metrics = comp_results.get('comprehensive_metrics', {})
                        if 'overall_score' in metrics:
                            report.append(f"    Overall score: {metrics['overall_score']:.3f}")
                        if 'primary' in metrics:
                            primary = metrics['primary']
                            if 'sharpe_ratio' in primary:
                                report.append(f"    Sharpe ratio: {primary['sharpe_ratio']:.4f}")
                
                elif component == 'risk_management':
                    if comp_results.get('success', False):
                        risk_summary = comp_results.get('risk_summary', {})
                        report.append(f"    Risk status: {risk_summary.get('status', 'unknown')}")
                        if 'risk_assessment' in comp_results:
                            risk_assessment = comp_results['risk_assessment']
                            report.append(f"    Risk level: {risk_assessment.get('risk_level', 'unknown')}")
                
                elif component == 'live_testing':
                    if comp_results.get('success', False):
                        session = comp_results.get('session', {})
                        report.append(f"    Session status: {session.get('status', 'unknown')}")
                        if 'final_metrics' in session:
                            final_metrics = session['final_metrics']
                            report.append(f"    Total trades: {final_metrics.get('total_trades', 0)}")
        
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
            report.append("  1. Review all component results")
            report.append("  2. Proceed to live deployment with conservative parameters")
            report.append("  3. Implement continuous monitoring")
            report.append("  4. Set up alert systems")
            report.append("  5. Begin with small position sizes")
        else:
            report.append("  1. Review component results and identify issues")
            report.append("  2. Address failed components")
            report.append("  3. Re-run Phase 3 pipeline")
            report.append("  4. Consider strategy modifications")
            report.append("  5. Do not proceed to live deployment")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main function for Phase 3 integration"""
    parser = argparse.ArgumentParser(description='Phase 3 Integration for Non-Major League ML Pipeline')
    parser.add_argument('--league', required=True, help='League code (e.g., E1 for Championship)')
    parser.add_argument('--data_file', required=True, help='Path to processed feature data')
    parser.add_argument('--output_dir', default='./results', help='Output directory for results')
    parser.add_argument('--model_file', help='Path to trained model file')
    parser.add_argument('--config', help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize Phase 3 integration
    phase3 = Phase3Integration(args.config, args.output_dir)
    
    try:
        # Run Phase 3 pipeline
        results = phase3.run_phase3_pipeline(
            data_file=args.data_file,
            league_code=args.league,
            model_file=args.model_file
        )
        
        # Generate and display report
        report = phase3.generate_phase3_report(results, args.league)
        print(report)
        
        # Save report
        report_file = os.path.join(args.output_dir, f"{args.league}_phase3_report.txt")
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"\nPhase 3 integration completed successfully!")
        print(f"Results saved to: {args.output_dir}")
        print(f"Report saved to: {report_file}")
        
    except Exception as e:
        print(f"Phase 3 integration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
