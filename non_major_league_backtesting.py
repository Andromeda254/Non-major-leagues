import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeagueBacktesting:
    """
    Comprehensive backtesting system for non-major soccer leagues
    
    Key Features:
    - Time series aware backtesting
    - Conservative betting simulation
    - Multiple performance metrics
    - Risk management controls
    - Drawdown analysis
    - Market impact simulation
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize backtesting system
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.backtest_results = {}
        self.betting_history = []
        self.performance_metrics = {}
        
    def setup_logging(self):
        """Setup logging for backtesting"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load backtesting configuration"""
        if config is None:
            self.config = {
                'backtesting': {
                    'method': 'walk_forward',
                    'initial_capital': 1000,
                    'min_bet_size': 1,
                    'max_bet_size': 100,
                    'commission': 0.05,  # 5% commission
                    'slippage': 0.02,    # 2% slippage
                    'market_impact': True
                },
                'betting_strategy': {
                    'kelly_fraction': 0.1,  # Conservative Kelly (10% of optimal)
                    'confidence_threshold': 0.6,
                    'min_odds': 1.5,
                    'max_odds': 10.0,
                    'max_bets_per_day': 5,
                    'max_bets_per_week': 20
                },
                'risk_management': {
                    'max_drawdown': 0.2,  # 20% max drawdown
                    'stop_loss': 0.15,    # 15% stop loss
                    'take_profit': 0.3,   # 30% take profit
                    'position_sizing': 'kelly',
                    'diversification': True
                },
                'performance_metrics': {
                    'primary': ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate'],
                    'secondary': ['profit_factor', 'recovery_factor', 'calmar_ratio', 'sortino_ratio'],
                    'betting': ['roi', 'avg_odds', 'avg_bet_size', 'bets_per_day']
                },
                'validation': {
                    'min_trading_days': 30,
                    'min_total_bets': 50,
                    'min_win_rate': 0.4,
                    'max_max_drawdown': 0.3
                }
            }
        else:
            self.config = config
    
    def prepare_backtest_data(self, data: pd.DataFrame, 
                            start_date: str = None, end_date: str = None) -> pd.DataFrame:
        """Prepare data for backtesting"""
        self.logger.info("Preparing backtest data")
        
        # Ensure data is sorted by date
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data = data.sort_values('Date').reset_index(drop=True)
        else:
            self.logger.warning("No Date column found, using index as time")
            data['Date'] = pd.date_range(start='2023-01-01', periods=len(data), freq='D')
        
        # Filter by date range if specified
        if start_date:
            data = data[data['Date'] >= start_date]
        if end_date:
            data = data[data['Date'] <= end_date]
        
        # Add required columns for backtesting
        if 'odds_home' not in data.columns:
            data['odds_home'] = 2.0  # Default odds
        if 'odds_draw' not in data.columns:
            data['odds_draw'] = 3.0
        if 'odds_away' not in data.columns:
            data['odds_away'] = 2.5
        
        # Add prediction columns if not present
        if 'prediction' not in data.columns:
            data['prediction'] = np.random.choice([0, 1, 2], len(data))
        if 'confidence' not in data.columns:
            data['confidence'] = np.random.uniform(0.5, 0.9, len(data))
        
        self.logger.info(f"Prepared {len(data)} records for backtesting")
        return data
    
    def calculate_kelly_bet_size(self, probability: float, odds: float, 
                                 kelly_fraction: float = None) -> float:
        """Calculate Kelly Criterion bet size"""
        if kelly_fraction is None:
            kelly_fraction = self.config['betting_strategy']['kelly_fraction']
        
        if probability <= 0 or probability >= 1 or odds <= 1:
            return 0
        
        # Kelly formula: f = (bp - q) / b
        # where b = odds - 1, p = probability, q = 1 - p
        b = odds - 1
        p = probability
        q = 1 - p
        
        kelly_optimal = (b * p - q) / b
        
        # Apply conservative fraction
        kelly_conservative = kelly_optimal * kelly_fraction
        
        # Ensure bet size is within bounds
        min_bet = self.config['backtesting']['min_bet_size']
        max_bet = self.config['backtesting']['max_bet_size']
        
        bet_size = max(min_bet, min(max_bet, kelly_conservative))
        
        return bet_size
    
    def simulate_bet(self, row: pd.Series, capital: float) -> Dict[str, Any]:
        """Simulate a single bet"""
        prediction = int(row['prediction'])
        confidence = row['confidence']
        
        # Check confidence threshold
        if confidence < self.config['betting_strategy']['confidence_threshold']:
            return {'bet_placed': False, 'reason': 'low_confidence'}
        
        # Get odds for predicted outcome
        if prediction == 0:  # Away win
            odds = row['odds_away']
        elif prediction == 1:  # Draw
            odds = row['odds_draw']
        else:  # Home win
            odds = row['odds_home']
        
        # Check odds bounds
        min_odds = self.config['betting_strategy']['min_odds']
        max_odds = self.config['betting_strategy']['max_odds']
        
        if odds < min_odds or odds > max_odds:
            return {'bet_placed': False, 'reason': 'odds_out_of_range'}
        
        # Calculate bet size
        probability = confidence
        bet_size = self.calculate_kelly_bet_size(probability, odds)
        
        # Check if we have enough capital
        if bet_size > capital:
            return {'bet_placed': False, 'reason': 'insufficient_capital'}
        
        # Simulate bet outcome
        actual_outcome = int(row['target']) if 'target' in row else np.random.choice([0, 1, 2])
        
        # Calculate profit/loss
        if prediction == actual_outcome:
            # Win
            gross_profit = bet_size * (odds - 1)
            commission = bet_size * self.config['backtesting']['commission']
            slippage = bet_size * self.config['backtesting']['slippage']
            net_profit = gross_profit - commission - slippage
            outcome = 'win'
        else:
            # Loss
            net_profit = -bet_size
            outcome = 'loss'
        
        # Market impact simulation
        if self.config['backtesting']['market_impact']:
            # Simulate odds movement after bet
            impact_factor = min(0.1, bet_size / capital * 0.5)
            if prediction == 0:
                row['odds_away'] *= (1 - impact_factor)
            elif prediction == 1:
                row['odds_draw'] *= (1 - impact_factor)
            else:
                row['odds_home'] *= (1 - impact_factor)
        
        bet_result = {
            'bet_placed': True,
            'date': row['Date'],
            'prediction': prediction,
            'actual_outcome': actual_outcome,
            'odds': odds,
            'bet_size': bet_size,
            'confidence': confidence,
            'net_profit': net_profit,
            'outcome': outcome,
            'capital_after': capital + net_profit
        }
        
        return bet_result
    
    def run_walk_forward_backtest(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Run walk-forward backtesting"""
        self.logger.info("Running walk-forward backtesting")
        
        # Prepare data
        data = self.prepare_backtest_data(data)
        
        # Initialize tracking variables
        capital = self.config['backtesting']['initial_capital']
        betting_history = []
        daily_returns = []
        daily_capital = []
        max_capital = capital
        max_drawdown = 0
        
        # Risk management variables
        stop_loss_triggered = False
        take_profit_triggered = False
        
        # Betting limits
        bets_today = 0
        bets_this_week = 0
        current_date = None
        current_week = None
        
        for idx, row in data.iterrows():
            # Update date tracking
            if current_date != row['Date'].date():
                current_date = row['Date'].date()
                bets_today = 0
            
            if current_week != row['Date'].isocalendar()[1]:
                current_week = row['Date'].isocalendar()[1]
                bets_this_week = 0
            
            # Check betting limits
            if bets_today >= self.config['betting_strategy']['max_bets_per_day']:
                continue
            if bets_this_week >= self.config['betting_strategy']['max_bets_per_week']:
                continue
            
            # Check risk management
            if stop_loss_triggered or take_profit_triggered:
                break
            
            # Calculate current drawdown
            if capital > max_capital:
                max_capital = capital
            
            current_drawdown = (max_capital - capital) / max_capital
            if current_drawdown > max_drawdown:
                max_drawdown = current_drawdown
            
            # Check stop loss
            if current_drawdown > self.config['risk_management']['stop_loss']:
                stop_loss_triggered = True
                self.logger.warning(f"Stop loss triggered at {current_drawdown:.2%} drawdown")
                break
            
            # Check take profit
            if capital > self.config['backtesting']['initial_capital'] * (1 + self.config['risk_management']['take_profit']):
                take_profit_triggered = True
                self.logger.info(f"Take profit triggered at {capital:.2f}")
                break
            
            # Simulate bet
            bet_result = self.simulate_bet(row, capital)
            
            if bet_result['bet_placed']:
                # Update capital
                capital = bet_result['capital_after']
                
                # Update betting counters
                bets_today += 1
                bets_this_week += 1
                
                # Record bet
                betting_history.append(bet_result)
                
                # Calculate daily return
                if len(daily_capital) > 0:
                    daily_return = (capital - daily_capital[-1]) / daily_capital[-1]
                    daily_returns.append(daily_return)
                else:
                    daily_returns.append(0)
                
                daily_capital.append(capital)
                
                self.logger.info(f"Bet {len(betting_history)}: {bet_result['outcome']}, "
                               f"Profit: {bet_result['net_profit']:.2f}, "
                               f"Capital: {capital:.2f}")
        
        # Calculate final metrics
        total_return = (capital - self.config['backtesting']['initial_capital']) / self.config['backtesting']['initial_capital']
        
        backtest_results = {
            'method': 'walk_forward',
            'initial_capital': self.config['backtesting']['initial_capital'],
            'final_capital': capital,
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'total_bets': len(betting_history),
            'winning_bets': len([b for b in betting_history if b['outcome'] == 'win']),
            'losing_bets': len([b for b in betting_history if b['outcome'] == 'loss']),
            'win_rate': len([b for b in betting_history if b['outcome'] == 'win']) / max(1, len(betting_history)),
            'betting_history': betting_history,
            'daily_returns': daily_returns,
            'daily_capital': daily_capital,
            'stop_loss_triggered': stop_loss_triggered,
            'take_profit_triggered': take_profit_triggered
        }
        
        self.backtest_results = backtest_results
        self.betting_history = betting_history
        
        self.logger.info(f"Walk-forward backtest complete: {total_return:.2%} return, "
                        f"{len(betting_history)} bets, {max_drawdown:.2%} max drawdown")
        
        return backtest_results
    
    def calculate_performance_metrics(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        self.logger.info("Calculating performance metrics")
        
        if not backtest_results:
            return {}
        
        betting_history = backtest_results['betting_history']
        daily_returns = backtest_results['daily_returns']
        
        if not betting_history:
            return {'error': 'No betting history available'}
        
        # Basic metrics
        total_return = backtest_results['total_return']
        max_drawdown = backtest_results['max_drawdown']
        win_rate = backtest_results['win_rate']
        
        # Calculate returns
        returns = [bet['net_profit'] for bet in betting_history]
        winning_returns = [bet['net_profit'] for bet in betting_history if bet['outcome'] == 'win']
        losing_returns = [bet['net_profit'] for bet in betting_history if bet['outcome'] == 'loss']
        
        # Risk metrics
        if daily_returns:
            returns_array = np.array(daily_returns)
            sharpe_ratio = np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0
            sortino_ratio = np.mean(returns_array) / np.std(returns_array[returns_array < 0]) * np.sqrt(252) if np.std(returns_array[returns_array < 0]) > 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
        
        # Betting metrics
        avg_bet_size = np.mean([bet['bet_size'] for bet in betting_history])
        avg_odds = np.mean([bet['odds'] for bet in betting_history])
        avg_confidence = np.mean([bet['confidence'] for bet in betting_history])
        
        # Profit factor
        total_wins = sum(winning_returns) if winning_returns else 0
        total_losses = abs(sum(losing_returns)) if losing_returns else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Recovery factor
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        # Calmar ratio
        calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else float('inf')
        
        # ROI
        total_invested = sum([bet['bet_size'] for bet in betting_history])
        roi = total_return if total_invested > 0 else 0
        
        # Bets per day
        if backtest_results['betting_history']:
            first_date = min([bet['date'] for bet in betting_history])
            last_date = max([bet['date'] for bet in betting_history])
            trading_days = (last_date - first_date).days + 1
            bets_per_day = len(betting_history) / trading_days if trading_days > 0 else 0
        else:
            bets_per_day = 0
        
        performance_metrics = {
            'primary': {
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate
            },
            'secondary': {
                'profit_factor': profit_factor,
                'recovery_factor': recovery_factor,
                'calmar_ratio': calmar_ratio,
                'sortino_ratio': sortino_ratio
            },
            'betting': {
                'roi': roi,
                'avg_odds': avg_odds,
                'avg_bet_size': avg_bet_size,
                'avg_confidence': avg_confidence,
                'bets_per_day': bets_per_day,
                'total_bets': len(betting_history)
            },
            'risk': {
                'total_wins': total_wins,
                'total_losses': total_losses,
                'avg_win': np.mean(winning_returns) if winning_returns else 0,
                'avg_loss': np.mean(losing_returns) if losing_returns else 0,
                'largest_win': max(winning_returns) if winning_returns else 0,
                'largest_loss': min(losing_returns) if losing_returns else 0
            }
        }
        
        self.performance_metrics = performance_metrics
        return performance_metrics
    
    def analyze_drawdowns(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze drawdown periods"""
        self.logger.info("Analyzing drawdowns")
        
        if not backtest_results or not backtest_results['daily_capital']:
            return {}
        
        daily_capital = backtest_results['daily_capital']
        
        # Calculate running maximum
        running_max = np.maximum.accumulate(daily_capital)
        
        # Calculate drawdowns
        drawdowns = (running_max - daily_capital) / running_max
        
        # Find drawdown periods
        drawdown_periods = []
        in_drawdown = False
        start_idx = 0
        
        for i, dd in enumerate(drawdowns):
            # Convert to scalar for comparison
            dd_value = float(dd) if isinstance(dd, np.ndarray) else dd
            
            if dd_value > 0 and not in_drawdown:
                # Start of drawdown
                in_drawdown = True
                start_idx = i
            elif dd_value == 0 and in_drawdown:
                # End of drawdown
                in_drawdown = False
                drawdown_periods.append({
                    'start_idx': start_idx,
                    'end_idx': i,
                    'duration': i - start_idx,
                    'max_drawdown': float(max(drawdowns[start_idx:i+1])),
                    'recovery_time': i - start_idx
                })
        
        # Calculate drawdown statistics
        if drawdown_periods:
            max_drawdown = max([dd['max_drawdown'] for dd in drawdown_periods])
            avg_drawdown = np.mean([dd['max_drawdown'] for dd in drawdown_periods])
            max_duration = max([dd['duration'] for dd in drawdown_periods])
            avg_duration = np.mean([dd['duration'] for dd in drawdown_periods])
            total_drawdown_time = sum([dd['duration'] for dd in drawdown_periods])
            drawdown_frequency = len(drawdown_periods) / len(daily_capital) if daily_capital else 0
        else:
            max_drawdown = 0
            avg_drawdown = 0
            max_duration = 0
            avg_duration = 0
            total_drawdown_time = 0
            drawdown_frequency = 0
        
        drawdown_analysis = {
            'max_drawdown': max_drawdown,
            'avg_drawdown': avg_drawdown,
            'max_duration': max_duration,
            'avg_duration': avg_duration,
            'total_drawdown_time': total_drawdown_time,
            'drawdown_frequency': drawdown_frequency,
            'drawdown_periods': drawdown_periods,
            'current_drawdown': float(drawdowns[-1]) if len(drawdowns) > 0 else 0
        }
        
        return drawdown_analysis
    
    def validate_backtest_results(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Validate backtest results"""
        self.logger.info("Validating backtest results")
        
        validation_results = {
            'validation_passed': True,
            'issues': [],
            'warnings': []
        }
        
        # Check minimum requirements
        min_trading_days = self.config['validation']['min_trading_days']
        min_total_bets = self.config['validation']['min_total_bets']
        min_win_rate = self.config['validation']['min_win_rate']
        max_max_drawdown = self.config['validation']['max_max_drawdown']
        
        # Check trading days
        if backtest_results['betting_history']:
            try:
                dates = [bet['date'] for bet in backtest_results['betting_history']]
                first_date = min(dates)
                last_date = max(dates)
                trading_days = (last_date - first_date).days + 1
                
                if trading_days < min_trading_days:
                    validation_results['validation_passed'] = False
                    validation_results['issues'].append(f"Insufficient trading days: {trading_days} < {min_trading_days}")
            except (KeyError, TypeError, ValueError) as e:
                self.logger.warning(f"Could not calculate trading days: {e}")
        else:
            validation_results['validation_passed'] = False
            validation_results['issues'].append("No betting history available")
        
        # Check total bets
        total_bets = backtest_results['total_bets']
        if total_bets < min_total_bets:
            validation_results['validation_passed'] = False
            validation_results['issues'].append(f"Insufficient total bets: {total_bets} < {min_total_bets}")
        
        # Check win rate
        win_rate = backtest_results['win_rate']
        if win_rate < min_win_rate:
            validation_results['validation_passed'] = False
            validation_results['issues'].append(f"Low win rate: {win_rate:.2%} < {min_win_rate:.2%}")
        
        # Check max drawdown
        max_drawdown = backtest_results['max_drawdown']
        if max_drawdown > max_max_drawdown:
            validation_results['validation_passed'] = False
            validation_results['issues'].append(f"High max drawdown: {max_drawdown:.2%} > {max_max_drawdown:.2%}")
        
        # Check for warnings
        if backtest_results['stop_loss_triggered']:
            validation_results['warnings'].append("Stop loss was triggered during backtest")
        
        if backtest_results['take_profit_triggered']:
            validation_results['warnings'].append("Take profit was triggered during backtest")
        
        if total_bets < min_total_bets * 1.5:
            validation_results['warnings'].append("Low number of bets - results may not be statistically significant")
        
        self.logger.info(f"Backtest validation: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
        
        return validation_results
    
    def generate_backtest_report(self, backtest_results: Dict[str, Any], 
                               performance_metrics: Dict[str, Any],
                               drawdown_analysis: Dict[str, Any],
                               validation_results: Dict[str, Any]) -> str:
        """Generate comprehensive backtest report"""
        report = []
        report.append("=" * 80)
        report.append("NON-MAJOR LEAGUE BACKTESTING REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Summary
        report.append("BACKTEST SUMMARY:")
        report.append(f"  Method: {backtest_results['method']}")
        report.append(f"  Initial Capital: ${backtest_results['initial_capital']:.2f}")
        report.append(f"  Final Capital: ${backtest_results['final_capital']:.2f}")
        report.append(f"  Total Return: {backtest_results['total_return']:.2%}")
        report.append(f"  Total Bets: {backtest_results['total_bets']}")
        report.append(f"  Win Rate: {backtest_results['win_rate']:.2%}")
        report.append(f"  Max Drawdown: {backtest_results['max_drawdown']:.2%}")
        report.append("")
        
        # Performance metrics
        if performance_metrics:
            report.append("PERFORMANCE METRICS:")
            report.append("  Primary Metrics:")
            for metric, value in performance_metrics['primary'].items():
                report.append(f"    {metric.replace('_', ' ').title()}: {value:.4f}")
            
            report.append("  Secondary Metrics:")
            for metric, value in performance_metrics['secondary'].items():
                if value != float('inf'):
                    report.append(f"    {metric.replace('_', ' ').title()}: {value:.4f}")
            
            report.append("  Betting Metrics:")
            for metric, value in performance_metrics['betting'].items():
                report.append(f"    {metric.replace('_', ' ').title()}: {value:.4f}")
            report.append("")
        
        # Drawdown analysis
        if drawdown_analysis:
            report.append("DRAWDOWN ANALYSIS:")
            report.append(f"  Max Drawdown: {drawdown_analysis['max_drawdown']:.2%}")
            report.append(f"  Avg Drawdown: {drawdown_analysis['avg_drawdown']:.2%}")
            report.append(f"  Max Duration: {drawdown_analysis['max_duration']} days")
            report.append(f"  Avg Duration: {drawdown_analysis['avg_duration']:.1f} days")
            report.append(f"  Drawdown Frequency: {drawdown_analysis['drawdown_frequency']:.4f}")
            report.append("")
        
        # Validation results
        report.append("VALIDATION RESULTS:")
        report.append(f"  Validation Status: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
        
        if validation_results['issues']:
            report.append("  Issues:")
            for issue in validation_results['issues']:
                report.append(f"    - {issue}")
        
        if validation_results['warnings']:
            report.append("  Warnings:")
            for warning in validation_results['warnings']:
                report.append(f"    - {warning}")
        report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if validation_results['validation_passed']:
            report.append("  ✅ Backtest passed validation - suitable for live testing")
            report.append("  📈 Consider implementing with conservative position sizing")
            report.append("  🔍 Monitor performance closely during live testing")
        else:
            report.append("  ❌ Backtest failed validation - review strategy")
            report.append("  🔧 Address identified issues before live testing")
            report.append("  📊 Consider additional data or strategy modifications")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_backtest_results(self, filepath: str):
        """Save backtest results"""
        self.logger.info(f"Saving backtest results to {filepath}")
        
        backtest_state = {
            'backtest_results': self.backtest_results,
            'performance_metrics': self.performance_metrics,
            'betting_history': self.betting_history,
            'config': self.config
        }
        
        import joblib
        joblib.dump(backtest_state, filepath)
        self.logger.info("Backtest results saved successfully")

# Example usage
def main():
    """Example usage of NonMajorLeagueBacktesting"""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    data = pd.DataFrame({
        'Date': pd.date_range(start='2023-01-01', periods=n_samples, freq='D'),
        'prediction': np.random.choice([0, 1, 2], n_samples),
        'confidence': np.random.uniform(0.5, 0.9, n_samples),
        'target': np.random.choice([0, 1, 2], n_samples),
        'odds_home': np.random.uniform(1.5, 5.0, n_samples),
        'odds_draw': np.random.uniform(2.5, 4.0, n_samples),
        'odds_away': np.random.uniform(1.5, 5.0, n_samples)
    })
    
    # Initialize backtesting
    backtester = NonMajorLeagueBacktesting()
    
    # Run backtest
    backtest_results = backtester.run_walk_forward_backtest(data)
    
    # Calculate performance metrics
    performance_metrics = backtester.calculate_performance_metrics(backtest_results)
    
    # Analyze drawdowns
    drawdown_analysis = backtester.analyze_drawdowns(backtest_results)
    
    # Validate results
    validation_results = backtester.validate_backtest_results(backtest_results)
    
    # Generate report
    report = backtester.generate_backtest_report(
        backtest_results, performance_metrics, drawdown_analysis, validation_results
    )
    
    print(report)
    
    # Save results
    backtester.save_backtest_results('backtest_results.pkl')

if __name__ == "__main__":
    main()
