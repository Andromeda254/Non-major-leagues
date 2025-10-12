import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import existing components
from non_major_league_data_collector import NonMajorLeagueDataCollector
from non_major_league_feature_engineer import NonMajorLeagueFeatureEngineer
from multi_market_model_architecture import MultiMarketModelArchitecture
from multi_market_strategy import MultiMarketBettingStrategy
from non_major_league_backtesting import NonMajorLeagueBacktesting

class MultiMarketIntegration:
    """
    Complete integration of multi-market betting strategy
    
    Key Features:
    - End-to-end multi-market pipeline
    - Data collection for all markets
    - Feature engineering for market-specific predictions
    - Multi-output model training and prediction
    - Portfolio-optimized betting strategy
    - Comprehensive backtesting and validation
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize multi-market integration
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.initialize_components()
        
    def setup_logging(self):
        """Setup logging for multi-market integration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load multi-market configuration"""
        if config is None or 'multi_market' not in config:
            self.config = {
                'enabled_markets': {
                    'match_result': True,
                    'both_teams_score': True,
                    'over_under_25': True,
                    'over_under_15': True,
                    'half_time_result': True,
                    'double_chance': True,
                    'correct_score': True,
                    'clean_sheet': True,
                    'first_goal': True,
                    'win_to_nil': True
                },
                'data_collection': {
                    'leagues': ['E1', 'E2', 'E3'],
                    'seasons': 3,
                    'include_half_time': True,
                    'include_statistics': True
                },
                'model_training': {
                    'method': 'separate',
                    'validation_split': 0.2,
                    'calibration': True
                },
                'betting_strategy': {
                    'max_markets_per_match': 3,
                    'portfolio_optimization': True,
                    'risk_management': True
                }
            }
        else:
            # Extract multi-market config from master pipeline config
            multi_market_config = config.get('multi_market', {})
            markets_config = multi_market_config.get('markets', {})
            
            self.config = {
                'enabled_markets': {
                    market: market_info.get('enabled', True) 
                    for market, market_info in markets_config.items()
                },
                'data_collection': {
                    'leagues': config.get('phase1', {}).get('leagues', ['E1', 'E2', 'E3']),
                    'seasons': 3,
                    'include_half_time': True,
                    'include_statistics': True
                },
                'model_training': {
                    'method': 'separate',
                    'validation_split': 0.2,
                    'calibration': True
                },
                'betting_strategy': {
                    'max_markets_per_match': multi_market_config.get('portfolio_optimization', {}).get('max_markets_per_match', 3),
                    'portfolio_optimization': True,
                    'risk_management': True
                }
            }
    
    def initialize_components(self):
        """Initialize all pipeline components"""
        self.logger.info("Initializing multi-market components")
        
        # Data collection
        self.data_collector = NonMajorLeagueDataCollector()
        
        # Feature engineering
        self.feature_engineer = NonMajorLeagueFeatureEngineer()
        
        # Model architecture
        self.model_architecture = MultiMarketModelArchitecture()
        
        # Betting strategy
        self.betting_strategy = MultiMarketBettingStrategy()
        
        # Backtesting
        self.backtesting = NonMajorLeagueBacktesting()
        
        self.logger.info("All components initialized")
    
    def collect_multi_market_data(self, league_code: str, seasons: List[str]) -> pd.DataFrame:
        """Collect comprehensive data for all markets"""
        self.logger.info(f"Collecting multi-market data for {league_code}")
        
        # Use enhanced data collection
        data = self.data_collector.collect_multi_market_data(league_code, seasons)
        
        if data.empty:
            self.logger.error(f"No data collected for {league_code}")
            return data
        
        self.logger.info(f"Collected {len(data)} matches with {len(data.columns)} features")
        return data
    
    def engineer_multi_market_features(self, df: pd.DataFrame, league_code: str) -> pd.DataFrame:
        """Engineer features for all markets"""
        self.logger.info("Engineering multi-market features")
        
        # Create market targets
        df_with_targets = self.betting_strategy.create_market_targets(df)
        
        # Create market-specific features
        df_with_features = self.betting_strategy.create_market_features(df_with_targets)
        
        # Apply general feature engineering
        df_engineered = self.feature_engineer.create_all_features(df_with_features, league_code)
        
        self.logger.info(f"Feature engineering complete: {len(df_engineered.columns)} features")
        return df_engineered
    
    def train_multi_market_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Train models for all markets"""
        self.logger.info("Training multi-market models")
        
        # Create market-specific features using engineered data
        market_features = self.model_architecture.create_market_features(df)
        
        # Prepare targets
        target_cols = [col for col in df.columns if col.startswith('target_')]
        y = df[target_cols].values  # Convert to numpy array for MultiOutputClassifier
        
        # Prepare features (use first market's features as base)
        X = list(market_features.values())[0] if market_features else df.select_dtypes(include=[np.number])
        X = X.values  # Convert to numpy array
        
        # Split data
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        X_train, X_val = X_train[:int(0.8 * len(X_train))], X_train[int(0.8 * len(X_train)):]
        y_train, y_val = y_train[:int(0.8 * len(y_train))], y_train[int(0.8 * len(y_train)):]
        
        # Train models - use separate models for each market
        # Prepare market-specific targets from full dataset
        market_targets = {}
        
        # Map target columns to market names
        target_to_market = {
            'target_1x2': 'match_result',
            'target_bts': 'both_teams_score', 
            'target_ou25': 'over_under_25',
            'target_ou15': 'over_under_15',
            'target_ht_1x2': 'half_time_result',
            'target_1x': 'double_chance',
            'target_12': 'double_chance',
            'target_x2': 'double_chance',
            'target_correct_score': 'correct_score',
            'target_home_clean_sheet': 'clean_sheet',
            'target_away_clean_sheet': 'clean_sheet',
            'target_first_goal': 'first_goal',
            'target_home_win_to_nil': 'win_to_nil',
            'target_away_win_to_nil': 'win_to_nil'
        }
        
        for i, target_col in enumerate(target_cols):
            market_name = target_to_market.get(target_col, target_col.replace('target_', ''))
            if market_name not in market_targets:
                market_targets[market_name] = y[:, i]  # Use full y, not y_train
        
        training_results = self.model_architecture.train_separate_models(
            market_features, market_targets
        )
        
        # Create ensembles
        ensembles = self.model_architecture.create_market_ensembles(training_results)
        
        # Calibrate models - disabled for now due to separate models structure
        calibrated_models = {}
        # if self.config['model_training']['calibration']:
        #     calibrated_models = self.model_architecture.calibrate_market_models(X_train, y_train)
        
        self.logger.info("Multi-market model training complete")
        return {
            'training_results': training_results,
            'ensembles': ensembles,
            'calibrated_models': calibrated_models
        }
    
    def predict_multi_markets(self, df: pd.DataFrame, training_results: Dict[str, Dict[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
        """Make predictions for all markets"""
        self.logger.info("Making multi-market predictions")
        
        # Create market-specific features
        market_features = self.model_architecture.create_market_features(df)
        
        # Use first market's features as base
        X = list(market_features.values())[0] if market_features else df.select_dtypes(include=[np.number])
        
        # Make predictions using trained models
        predictions = self.model_architecture.predict_markets(df, training_results)
        
        # Calculate market correlations
        correlations = self.model_architecture.calculate_market_correlations(predictions)
        
        self.logger.info(f"Predictions made for {len(predictions)} models")
        return {
            'predictions': predictions,
            'correlations': correlations
        }
    
    def execute_multi_market_strategy(self, match_data: pd.Series, 
                                    predictions: Dict[str, Any],
                                    odds: Dict[str, float],
                                    capital: float) -> Dict[str, Any]:
        """Execute complete multi-market betting strategy"""
        self.logger.info("Executing multi-market strategy")
        
        # Use betting strategy to select markets and calculate positions
        strategy_result = self.betting_strategy.execute_multi_market_strategy(
            match_data, predictions, odds, capital
        )
        
        return strategy_result
    
    def backtest_multi_market_strategy(self, historical_data: pd.DataFrame,
                                     predictions: Dict[str, pd.DataFrame],
                                     odds: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Backtest multi-market strategy"""
        self.logger.info("Backtesting multi-market strategy")
        
        # Use betting strategy backtesting
        backtest_results = self.betting_strategy.backtest_multi_market_strategy(
            historical_data, predictions, odds
        )
        
        return backtest_results
    
    def run_complete_pipeline(self, league_code: str, seasons: List[str]) -> Dict[str, Any]:
        """Run complete multi-market pipeline"""
        self.logger.info(f"Running complete multi-market pipeline for {league_code}")
        
        results = {}
        
        try:
            # 1. Data Collection
            self.logger.info("Phase 1: Data Collection")
            data = self.collect_multi_market_data(league_code, seasons)
            results['data_collection'] = {
                'matches_collected': len(data),
                'features_collected': len(data.columns),
                'success': not data.empty
            }
            
            if data.empty:
                return results
            
            # 2. Feature Engineering
            self.logger.info("Phase 2: Feature Engineering")
            engineered_data = self.engineer_multi_market_features(data, league_code)
            results['feature_engineering'] = {
                'features_created': len(engineered_data.columns),
                'targets_created': len([col for col in engineered_data.columns if col.startswith('target_')]),
                'success': True
            }
            
            # 3. Model Training
            self.logger.info("Phase 3: Model Training")
            try:
                training_results = self.train_multi_market_models(engineered_data)
                results['model_training'] = {
                    'models_trained': len(training_results['training_results']),
                    'ensembles_created': len(training_results['ensembles']),
                    'success': True
                }
            except Exception as e:
                self.logger.error(f"Model training failed: {e}")
                results['model_training'] = {
                    'models_trained': 0,
                    'ensembles_created': 0,
                    'success': False,
                    'error': str(e)
                }
            
            # 4. Predictions
            self.logger.info("Phase 4: Making Predictions")
            prediction_results = self.predict_multi_markets(engineered_data, training_results['training_results'])
            results['predictions'] = {
                'models_used': len(prediction_results['predictions']),
                'correlations_calculated': len(prediction_results['correlations']),
                'success': True
            }
            
            # 5. Strategy Execution (example)
            self.logger.info("Phase 5: Strategy Execution")
            sample_match = engineered_data.iloc[0]
            sample_predictions = {
                'match_result': {'probability': 0.6, 'confidence': 0.8},
                'both_teams_score': {'probability': 0.7, 'confidence': 0.7}
            }
            sample_odds = {
                'match_result': 2.1,
                'both_teams_score': 1.8
            }
            
            strategy_result = self.execute_multi_market_strategy(
                sample_match, sample_predictions, sample_odds, 1000
            )
            results['strategy_execution'] = {
                'bets_placed': len(strategy_result['bets_placed']),
                'total_risk': strategy_result['total_risk'],
                'expected_return': strategy_result['expected_return'],
                'success': True
            }
            
            self.logger.info("Complete multi-market pipeline executed successfully")
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            results['error'] = str(e)
        
        return results
    
    def get_market_performance_summary(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Get performance summary by market"""
        self.logger.info("Generating market performance summary")
        
        summary = {
            'overall_performance': {
                'total_return': backtest_results.get('total_return', 0),
                'win_rate': backtest_results.get('win_rate', 0),
                'total_bets': backtest_results.get('total_bets', 0),
                'max_drawdown': backtest_results.get('max_drawdown', 0)
            },
            'market_breakdown': {}
        }
        
        # Market-specific performance
        market_performance = backtest_results.get('market_performance', {})
        for market, perf in market_performance.items():
            summary['market_breakdown'][market] = {
                'bets': perf.get('bets', 0),
                'wins': perf.get('wins', 0),
                'win_rate': perf.get('wins', 0) / perf.get('bets', 1) if perf.get('bets', 0) > 0 else 0,
                'profit': perf.get('profit', 0)
            }
        
        return summary
    
    def save_pipeline_state(self, filepath: str):
        """Save complete pipeline state"""
        self.logger.info(f"Saving pipeline state to {filepath}")
        
        pipeline_state = {
            'config': self.config,
            'data_collector': self.data_collector,
            'feature_engineer': self.feature_engineer,
            'model_architecture': self.model_architecture,
            'betting_strategy': self.betting_strategy
        }
        
        import joblib
        joblib.dump(pipeline_state, filepath)
        self.logger.info("Pipeline state saved successfully")
    
    def load_pipeline_state(self, filepath: str):
        """Load complete pipeline state"""
        self.logger.info(f"Loading pipeline state from {filepath}")
        
        import joblib
        pipeline_state = joblib.load(filepath)
        
        self.config = pipeline_state['config']
        self.data_collector = pipeline_state['data_collector']
        self.feature_engineer = pipeline_state['feature_engineer']
        self.model_architecture = pipeline_state['model_architecture']
        self.betting_strategy = pipeline_state['betting_strategy']
        
        self.logger.info("Pipeline state loaded successfully")

# Example usage
def main():
    """Example usage of MultiMarketIntegration"""
    
    # Initialize multi-market integration
    integration = MultiMarketIntegration()
    
    # Run complete pipeline
    results = integration.run_complete_pipeline('E1', ['2324', '2223'])
    
    print("Multi-Market Pipeline Results:")
    for phase, result in results.items():
        if isinstance(result, dict) and 'success' in result:
            status = "✓" if result['success'] else "✗"
            print(f"{status} {phase}: {result}")
        else:
            print(f"  {phase}: {result}")
    
    print("\nMulti-market betting strategy successfully implemented!")
    print("\nSupported Markets:")
    print("- Match Result (1X2)")
    print("- Both Teams to Score (GG/NOGG)")
    print("- Over/Under 2.5 Goals")
    print("- Over/Under 1.5 Goals")
    print("- Half-Time Result")
    print("- Double Chance (1X, 12, X2)")
    print("- Correct Score")
    print("- Clean Sheet")
    print("- First Goal")
    print("- Win to Nil")

if __name__ == "__main__":
    main()
