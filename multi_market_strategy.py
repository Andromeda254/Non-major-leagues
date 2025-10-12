import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MultiMarketBettingStrategy:
    """
    Multi-market betting strategy for non-major soccer leagues
    
    Key Features:
    - Multiple betting market predictions
    - Portfolio optimization across markets
    - Market correlation analysis
    - Risk-adjusted position sizing
    - Diversification benefits
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize multi-market betting strategy
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.market_models = {}
        self.market_correlations = {}
        self.betting_portfolio = []
        self.performance_by_market = {}
        
    def setup_logging(self):
        """Setup logging for multi-market strategy"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load multi-market configuration"""
        if config is None:
            self.config = {
                'enabled_markets': {
                    'match_result': True,  # 1X2
                    'both_teams_score': True,  # GG/NOGG
                    'over_under_25': True,  # O/U 2.5
                    'over_under_15': True,  # O/U 1.5
                    'half_time_result': True,  # HT 1X2
                    'double_chance': True,  # 1X, 12, X2
                    'correct_score': True,  # Most likely scores
                    'clean_sheet': True,  # Home/Away clean sheet
                    'first_goal': True,  # First goal scorer
                    'win_to_nil': True  # Win without conceding
                },
                'market_weights': {
                    'match_result': 0.25,
                    'both_teams_score': 0.15,
                    'over_under_25': 0.15,
                    'over_under_15': 0.10,
                    'half_time_result': 0.10,
                    'double_chance': 0.10,
                    'correct_score': 0.05,
                    'clean_sheet': 0.05,
                    'first_goal': 0.03,
                    'win_to_nil': 0.02
                },
                'risk_management': {
                    'max_markets_per_match': 3,
                    'correlation_threshold': 0.7,
                    'max_portfolio_risk': 0.15,
                    'market_specific_limits': {
                        'match_result': 0.05,
                        'both_teams_score': 0.03,
                        'over_under_25': 0.03,
                        'correct_score': 0.01
                    }
                },
                'data_requirements': {
                    'half_time_scores': True,
                    'goal_timings': True,
                    'team_statistics': True,
                    'player_data': False  # Optional for first goal scorer
                }
            }
        else:
            self.config = config
    
    def create_market_targets(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for all betting markets"""
        self.logger.info("Creating multi-market targets")
        
        df_targets = df.copy()
        
        # 1. Match Result (1X2) - already exists
        df_targets['target_1x2'] = df['FTR'].map({'A': 0, 'D': 1, 'H': 2})
        
        # 2. Both Teams to Score (GG/NOGG)
        df_targets['target_bts'] = ((df['FTHG'] > 0) & (df['FTAG'] > 0)).astype(int)
        
        # 3. Over/Under 2.5 Goals
        df_targets['target_ou25'] = (df['FTHG'] + df['FTAG'] > 2.5).astype(int)
        
        # 4. Over/Under 1.5 Goals  
        df_targets['target_ou15'] = (df['FTHG'] + df['FTAG'] > 1.5).astype(int)
        
        # 5. Half-Time Result (if available)
        if 'HTHG' in df.columns and 'HTAG' in df.columns:
            df_targets['target_ht_1x2'] = df.apply(
                lambda row: 0 if row['HTHG'] < row['HTAG'] else 1 if row['HTHG'] == row['HTAG'] else 2, axis=1
            )
        else:
            df_targets['target_ht_1x2'] = df_targets['target_1x2']  # Fallback to FT result
        
        # 6. Double Chance Markets
        df_targets['target_1x'] = ((df['FTR'] == 'H') | (df['FTR'] == 'D')).astype(int)  # Home or Draw
        df_targets['target_12'] = ((df['FTR'] == 'H') | (df['FTR'] == 'A')).astype(int)  # Home or Away
        df_targets['target_x2'] = ((df['FTR'] == 'D') | (df['FTR'] == 'A')).astype(int)  # Draw or Away
        
        # 7. Correct Score (simplified - most common scores)
        df_targets['target_correct_score'] = df.apply(self._get_most_common_score, axis=1)
        
        # 8. Clean Sheet Markets
        df_targets['target_home_clean_sheet'] = (df['FTAG'] == 0).astype(int)
        df_targets['target_away_clean_sheet'] = (df['FTHG'] == 0).astype(int)
        
        # 9. First Goal (simplified - team that scores first)
        df_targets['target_first_goal'] = df.apply(self._get_first_goal_team, axis=1)
        
        # 10. Win to Nil
        df_targets['target_home_win_to_nil'] = ((df['FTR'] == 'H') & (df['FTAG'] == 0)).astype(int)
        df_targets['target_away_win_to_nil'] = ((df['FTR'] == 'A') & (df['FTHG'] == 0)).astype(int)
        
        self.logger.info(f"Created {len([col for col in df_targets.columns if col.startswith('target_')])} market targets")
        return df_targets
    
    def _get_most_common_score(self, row: pd.Series) -> int:
        """Get most common score category"""
        home_goals = row['FTHG']
        away_goals = row['FTAG']
        total_goals = home_goals + away_goals
        
        # Map to common score categories
        if total_goals == 0:
            return 0  # 0-0
        elif total_goals == 1:
            return 1  # 1-0 or 0-1
        elif total_goals == 2:
            return 2  # 2-0, 1-1, 0-2
        elif total_goals == 3:
            return 3  # 3-0, 2-1, 1-2, 0-3
        else:
            return 4  # 4+ goals
    
    def _get_first_goal_team(self, row: pd.Series) -> int:
        """Get team that scores first (simplified)"""
        home_goals = row['FTHG']
        away_goals = row['FTAG']
        
        if home_goals > 0 and away_goals == 0:
            return 0  # Home scores first
        elif away_goals > 0 and home_goals == 0:
            return 1  # Away scores first
        elif home_goals > 0 and away_goals > 0:
            # Simplified: assume home scores first if more goals
            return 0 if home_goals >= away_goals else 1
        else:
            return 2  # No goals
    
    def create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market-specific features"""
        self.logger.info("Creating market-specific features")
        
        df_features = df.copy()
        
        # Goal-scoring features for BTS and O/U markets
        df_features['home_goals_avg'] = df.groupby('HomeTeam')['FTHG'].transform('mean')
        df_features['away_goals_avg'] = df.groupby('AwayTeam')['FTAG'].transform('mean')
        df_features['home_goals_conceded_avg'] = df.groupby('HomeTeam')['FTAG'].transform('mean')
        df_features['away_goals_conceded_avg'] = df.groupby('AwayTeam')['FTHG'].transform('mean')
        
        # Both teams scoring probability
        df_features['home_bts_prob'] = df.groupby('HomeTeam')['FTHG'].transform(
            lambda x: (x > 0).mean()
        )
        df_features['away_bts_prob'] = df.groupby('AwayTeam')['FTAG'].transform(
            lambda x: (x > 0).mean()
        )
        
        # Over/Under features
        df_features['total_goals_avg'] = (df['FTHG'] + df['FTAG']).groupby(
            df.index // 10  # Group by batches for rolling average
        ).transform('mean')
        
        # Half-time specific features (if available)
        if 'HTHG' in df.columns and 'HTAG' in df.columns:
            df_features['ht_home_goals_avg'] = df.groupby('HomeTeam')['HTHG'].transform('mean')
            df_features['ht_away_goals_avg'] = df.groupby('AwayTeam')['HTAG'].transform('mean')
            df_features['ht_total_goals_avg'] = (df['HTHG'] + df['HTAG']).groupby(
                df.index // 10
            ).transform('mean')
        
        # Clean sheet features
        df_features['home_clean_sheet_rate'] = df.groupby('HomeTeam')['FTAG'].transform(
            lambda x: (x == 0).mean()
        )
        df_features['away_clean_sheet_rate'] = df.groupby('AwayTeam')['FTHG'].transform(
            lambda x: (x == 0).mean()
        )
        
        # First goal features
        df_features['home_first_goal_rate'] = df.groupby('HomeTeam').apply(
            lambda x: (x['FTHG'] > x['FTAG']).mean() if len(x) > 0 else 0.5
        ).reset_index(level=0, drop=True)
        
        self.logger.info(f"Created market-specific features")
        return df_features
    
    def calculate_market_correlations(self, df: pd.DataFrame) -> Dict[str, float]:
        """Calculate correlations between different markets"""
        self.logger.info("Calculating market correlations")
        
        target_cols = [col for col in df.columns if col.startswith('target_')]
        correlations = {}
        
        for i, col1 in enumerate(target_cols):
            for col2 in target_cols[i+1:]:
                if col1 in df.columns and col2 in df.columns:
                    corr = df[col1].corr(df[col2])
                    correlations[f"{col1}_{col2}"] = corr
        
        self.market_correlations = correlations
        self.logger.info(f"Calculated {len(correlations)} market correlations")
        return correlations
    
    def optimize_portfolio_weights(self, predictions: Dict[str, Any], 
                                 odds: Dict[str, float]) -> Dict[str, float]:
        """Optimize portfolio weights across markets using Modern Portfolio Theory"""
        self.logger.info("Optimizing portfolio weights")
        
        # Get enabled markets
        enabled_markets = [market for market, enabled in self.config['enabled_markets'].items() if enabled]
        
        # Calculate expected returns and risks for each market
        market_stats = {}
        for market in enabled_markets:
            if market in predictions and market in odds:
                prob = predictions[market]['probability']
                odd = odds[market]
                expected_return = prob * (odd - 1) - (1 - prob)  # Kelly-style return
                risk = prob * (1 - prob)  # Variance as risk proxy
                
                market_stats[market] = {
                    'expected_return': expected_return,
                    'risk': risk,
                    'sharpe_ratio': expected_return / (risk + 1e-10)
                }
        
        # Simple optimization: weight by Sharpe ratio, adjusted for correlation
        total_sharpe = sum(stats['sharpe_ratio'] for stats in market_stats.values())
        weights = {}
        
        for market, stats in market_stats.items():
            base_weight = stats['sharpe_ratio'] / total_sharpe if total_sharpe > 0 else 1/len(market_stats)
            
            # Adjust for market-specific limits
            max_weight = self.config['risk_management']['market_specific_limits'].get(market, 0.05)
            weights[market] = min(base_weight, max_weight)
        
        # Normalize weights
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v/total_weight for k, v in weights.items()}
        
        self.logger.info(f"Optimized weights: {weights}")
        return weights
    
    def select_best_markets(self, predictions: Dict[str, Any], 
                          odds: Dict[str, float], max_markets: int = 3) -> List[str]:
        """Select best markets for betting based on value and diversification"""
        self.logger.info("Selecting best markets")
        
        # Calculate value for each market
        market_values = {}
        for market, pred in predictions.items():
            if market in odds:
                prob = pred['probability']
                odd = odds[market]
                value = prob * odd - 1  # Positive value = good bet
                confidence = pred.get('confidence', 0.5)
                
                # Combined score: value + confidence
                market_values[market] = value * confidence
        
        # Sort by value and select top markets
        sorted_markets = sorted(market_values.items(), key=lambda x: x[1], reverse=True)
        selected_markets = [market for market, value in sorted_markets[:max_markets] if value > 0]
        
        self.logger.info(f"Selected markets: {selected_markets}")
        return selected_markets
    
    def calculate_position_sizes(self, selected_markets: List[str], 
                               predictions: Dict[str, Any], 
                               odds: Dict[str, float],
                               total_capital: float) -> Dict[str, float]:
        """Calculate position sizes for selected markets"""
        self.logger.info("Calculating position sizes")
        
        # Get portfolio weights
        weights = self.optimize_portfolio_weights(predictions, odds)
        
        # Calculate position sizes
        position_sizes = {}
        for market in selected_markets:
            if market in weights and market in predictions and market in odds:
                prob = predictions[market]['probability']
                odd = odds[market]
                weight = weights[market]
                
                # Kelly fraction for this market
                kelly_fraction = (prob * odd - 1) / (odd - 1) if odd > 1 else 0
                kelly_fraction = max(0, min(kelly_fraction, 0.05))  # Conservative cap
                
                # Position size = weight * capital * kelly_fraction
                position_size = weight * total_capital * kelly_fraction
                position_sizes[market] = position_size
        
        self.logger.info(f"Position sizes: {position_sizes}")
        return position_sizes
    
    def execute_multi_market_strategy(self, match_data: pd.Series, 
                                    predictions: Dict[str, Any],
                                    odds: Dict[str, float],
                                    capital: float) -> Dict[str, Any]:
        """Execute complete multi-market betting strategy"""
        self.logger.info("Executing multi-market strategy")
        
        # Select best markets
        selected_markets = self.select_best_markets(
            predictions, odds, 
            self.config['risk_management']['max_markets_per_match']
        )
        
        if not selected_markets:
            return {'bets_placed': [], 'total_risk': 0, 'expected_return': 0}
        
        # Calculate position sizes
        position_sizes = self.calculate_position_sizes(selected_markets, predictions, odds, capital)
        
        # Create bet portfolio
        bets = []
        total_risk = 0
        expected_return = 0
        
        for market in selected_markets:
            if market in position_sizes and position_sizes[market] > 0:
                bet = {
                    'market': market,
                    'position_size': position_sizes[market],
                    'odds': odds[market],
                    'probability': predictions[market]['probability'],
                    'confidence': predictions[market].get('confidence', 0.5),
                    'expected_return': predictions[market]['probability'] * (odds[market] - 1) - (1 - predictions[market]['probability'])
                }
                bets.append(bet)
                total_risk += position_sizes[market]
                expected_return += bet['expected_return'] * position_sizes[market]
        
        result = {
            'bets_placed': bets,
            'total_risk': total_risk,
            'expected_return': expected_return,
            'selected_markets': selected_markets,
            'diversification_ratio': len(selected_markets) / len(self.config['enabled_markets'])
        }
        
        self.logger.info(f"Multi-market strategy executed: {len(bets)} bets, {total_risk:.2f} total risk")
        return result
    
    def backtest_multi_market_strategy(self, historical_data: pd.DataFrame, 
                                     predictions: Dict[str, pd.DataFrame],
                                     odds: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Backtest multi-market strategy on historical data"""
        self.logger.info("Backtesting multi-market strategy")
        
        results = {
            'total_return': 0,
            'total_bets': 0,
            'winning_bets': 0,
            'market_performance': {},
            'monthly_returns': [],
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
        capital = 10000  # Starting capital
        peak_capital = capital
        
        for idx, row in historical_data.iterrows():
            # Get predictions and odds for this match
            match_predictions = {}
            match_odds = {}
            
            for market in self.config['enabled_markets']:
                if market in predictions and idx in predictions[market].index:
                    match_predictions[market] = {
                        'probability': predictions[market].loc[idx, 'probability'],
                        'confidence': predictions[market].loc[idx, 'confidence']
                    }
                
                if market in odds and idx in odds[market].index:
                    match_odds[market] = odds[market].loc[idx, 'odds']
            
            # Execute strategy
            strategy_result = self.execute_multi_market_strategy(
                row, match_predictions, match_odds, capital
            )
            
            # Simulate bet outcomes
            for bet in strategy_result['bets_placed']:
                market = bet['market']
                position_size = bet['position_size']
                odds_value = bet['odds']
                
                # Determine actual outcome (simplified)
                actual_outcome = self._get_actual_outcome(row, market)
                predicted_outcome = 1 if bet['probability'] > 0.5 else 0
                
                if actual_outcome == predicted_outcome:
                    # Win
                    profit = position_size * (odds_value - 1)
                    capital += profit
                    results['winning_bets'] += 1
                else:
                    # Loss
                    capital -= position_size
                
                results['total_bets'] += 1
                
                # Track market performance
                if market not in results['market_performance']:
                    results['market_performance'][market] = {'bets': 0, 'wins': 0, 'profit': 0}
                
                results['market_performance'][market]['bets'] += 1
                if actual_outcome == predicted_outcome:
                    results['market_performance'][market]['wins'] += 1
                    results['market_performance'][market]['profit'] += position_size * (odds_value - 1)
                else:
                    results['market_performance'][market]['profit'] -= position_size
            
            # Update peak and drawdown
            if capital > peak_capital:
                peak_capital = capital
            
            current_drawdown = (peak_capital - capital) / peak_capital
            if current_drawdown > results['max_drawdown']:
                results['max_drawdown'] = current_drawdown
        
        # Calculate final metrics
        results['total_return'] = (capital - 10000) / 10000
        results['win_rate'] = results['winning_bets'] / results['total_bets'] if results['total_bets'] > 0 else 0
        
        self.logger.info(f"Backtest completed: {results['total_return']:.2%} return, {results['win_rate']:.2%} win rate")
        return results
    
    def _get_actual_outcome(self, row: pd.Series, market: str) -> int:
        """Get actual outcome for a specific market"""
        if market == 'match_result':
            return {'A': 0, 'D': 1, 'H': 2}[row['FTR']]
        elif market == 'both_teams_score':
            return int((row['FTHG'] > 0) & (row['FTAG'] > 0))
        elif market == 'over_under_25':
            return int((row['FTHG'] + row['FTAG']) > 2.5)
        elif market == 'over_under_15':
            return int((row['FTHG'] + row['FTAG']) > 1.5)
        elif market == 'double_chance':
            # Simplified - assume 1X (home or draw)
            return int((row['FTR'] == 'H') or (row['FTR'] == 'D'))
        else:
            return 0  # Default

# Example usage
def main():
    """Example usage of MultiMarketBettingStrategy"""
    
    # Create sample data
    np.random.seed(42)
    n_matches = 100
    
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=n_matches, freq='D'),
        'HomeTeam': ['Team A', 'Team B', 'Team C'] * (n_matches // 3 + 1),
        'AwayTeam': ['Team B', 'Team C', 'Team A'] * (n_matches // 3 + 1),
        'FTHG': np.random.randint(0, 4, n_matches),
        'FTAG': np.random.randint(0, 4, n_matches),
        'FTR': np.random.choice(['H', 'D', 'A'], n_matches),
        'HTHG': np.random.randint(0, 3, n_matches),
        'HTAG': np.random.randint(0, 3, n_matches)
    })
    
    # Initialize multi-market strategy
    strategy = MultiMarketBettingStrategy()
    
    # Create market targets
    data_with_targets = strategy.create_market_targets(sample_data)
    
    # Create market features
    data_with_features = strategy.create_market_features(data_with_targets)
    
    # Calculate market correlations
    correlations = strategy.calculate_market_correlations(data_with_features)
    
    print(f"Created {len([col for col in data_with_targets.columns if col.startswith('target_')])} market targets")
    print(f"Calculated {len(correlations)} market correlations")
    
    # Example predictions and odds
    sample_predictions = {
        'match_result': {'probability': 0.6, 'confidence': 0.8},
        'both_teams_score': {'probability': 0.7, 'confidence': 0.7},
        'over_under_25': {'probability': 0.5, 'confidence': 0.6}
    }
    
    sample_odds = {
        'match_result': 2.1,
        'both_teams_score': 1.8,
        'over_under_25': 1.9
    }
    
    # Execute strategy
    result = strategy.execute_multi_market_strategy(
        sample_data.iloc[0], sample_predictions, sample_odds, 1000
    )
    
    print(f"Strategy result: {len(result['bets_placed'])} bets, {result['total_risk']:.2f} total risk")
    print("Multi-market betting strategy successfully implemented!")

if __name__ == "__main__":
    main()






