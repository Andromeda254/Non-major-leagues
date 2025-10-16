import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeaguePerformanceMetrics:
    """
    Specialized performance metrics for non-major soccer leagues
    
    Key Features:
    - Conservative performance evaluation
    - Risk-adjusted metrics
    - Market-specific metrics
    - Drawdown analysis
    - Confidence calibration
    - Statistical significance testing
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize performance metrics system
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.metrics_history = []
        self.performance_summary = {}
        
    def setup_logging(self):
        """Setup logging for performance metrics"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load performance metrics configuration"""
        if config is None:
            self.config = {
                'metrics': {
                    'primary': {
                        'total_return': {'enabled': True, 'weight': 0.3},
                        'sharpe_ratio': {'enabled': True, 'weight': 0.25},
                        'max_drawdown': {'enabled': True, 'weight': 0.2},
                        'win_rate': {'enabled': True, 'weight': 0.15},
                        'profit_factor': {'enabled': True, 'weight': 0.1}
                    },
                    'secondary': {
                        'sortino_ratio': {'enabled': True, 'weight': 0.2},
                        'calmar_ratio': {'enabled': True, 'weight': 0.2},
                        'recovery_factor': {'enabled': True, 'weight': 0.15},
                        'var_95': {'enabled': True, 'weight': 0.15},
                        'cvar_95': {'enabled': True, 'weight': 0.15},
                        'skewness': {'enabled': True, 'weight': 0.15}
                    },
                    'betting_specific': {
                        'roi': {'enabled': True, 'weight': 0.25},
                        'avg_odds': {'enabled': True, 'weight': 0.15},
                        'avg_bet_size': {'enabled': True, 'weight': 0.15},
                        'bets_per_day': {'enabled': True, 'weight': 0.15},
                        'confidence_accuracy': {'enabled': True, 'weight': 0.15},
                        'kelly_efficiency': {'enabled': True, 'weight': 0.15}
                    },
                    'risk_metrics': {
                        'volatility': {'enabled': True, 'weight': 0.2},
                        'downside_deviation': {'enabled': True, 'weight': 0.2},
                        'tail_ratio': {'enabled': True, 'weight': 0.15},
                        'ulcer_index': {'enabled': True, 'weight': 0.15},
                        'pain_index': {'enabled': True, 'weight': 0.15},
                        'stability_ratio': {'enabled': True, 'weight': 0.15}
                    }
                },
                'benchmarks': {
                    'risk_free_rate': 0.02,  # 2% annual
                    'market_return': 0.08,   # 8% annual
                    'benchmark_volatility': 0.15,  # 15% annual
                    'benchmark_sharpe': 0.4
                },
                'statistical_tests': {
                    'min_samples': 30,
                    'confidence_level': 0.95,
                    'bootstrap_samples': 1000,
                    'significance_threshold': 0.05
                },
                'conservative_thresholds': {
                    'min_win_rate': 0.45,
                    'max_max_drawdown': 0.25,
                    'min_sharpe_ratio': 0.3,
                    'min_profit_factor': 1.2,
                    'max_volatility': 0.3
                }
            }
        else:
            self.config = config
    
    def calculate_primary_metrics(self, returns: List[float], 
                                 betting_history: List[Dict] = None) -> Dict[str, float]:
        """Calculate primary performance metrics"""
        self.logger.info("Calculating primary performance metrics")
        
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        # Total return
        total_return = np.sum(returns_array)
        
        # Sharpe ratio (annualized)
        if len(returns_array) > 1 and np.std(returns_array) > 0:
            sharpe_ratio = (np.mean(returns_array) - self.config['benchmarks']['risk_free_rate']/252) / np.std(returns_array) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / (running_max + 1e-10)
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Win rate
        if betting_history:
            winning_bets = [bet for bet in betting_history if bet.get('outcome') == 'win']
            win_rate = len(winning_bets) / len(betting_history) if betting_history else 0
        else:
            win_rate = len(returns_array[returns_array > 0]) / len(returns_array) if len(returns_array) > 0 else 0
        
        # Profit factor
        if betting_history:
            winning_profits = [bet.get('net_profit', 0) for bet in betting_history if bet.get('outcome') == 'win']
            losing_losses = [abs(bet.get('net_profit', 0)) for bet in betting_history if bet.get('outcome') == 'loss']
            
            total_wins = sum(winning_profits) if winning_profits else 0
            total_losses = sum(losing_losses) if losing_losses else 0
            
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        else:
            positive_returns = returns_array[returns_array > 0]
            negative_returns = returns_array[returns_array < 0]
            
            total_wins = sum(positive_returns) if len(positive_returns) > 0 else 0
            total_losses = abs(sum(negative_returns)) if len(negative_returns) > 0 else 0
            
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        primary_metrics = {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_factor': profit_factor
        }
        
        return primary_metrics
    
    def calculate_secondary_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate secondary performance metrics"""
        self.logger.info("Calculating secondary performance metrics")
        
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        # Sortino ratio (downside deviation)
        downside_returns = returns_array[returns_array < 0]
        if len(downside_returns) > 1:
            downside_deviation = np.std(downside_returns)
            sortino_ratio = (np.mean(returns_array) - self.config['benchmarks']['risk_free_rate']/252) / downside_deviation * np.sqrt(252) if downside_deviation > 0 else 0
        else:
            sortino_ratio = 0
        
        # Calmar ratio (return / max drawdown)
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / (running_max + 1e-10)
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        calmar_ratio = np.mean(returns_array) / max_drawdown if max_drawdown > 0 else 0
        
        # Recovery factor (total return / max drawdown)
        total_return = np.sum(returns_array)
        recovery_factor = total_return / max_drawdown if max_drawdown > 0 else 0
        
        # Value at Risk (VaR) 95%
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 0 else 0
        
        # Conditional Value at Risk (CVaR) 95%
        cvar_95 = np.mean(returns_array[returns_array <= var_95]) if len(returns_array) > 0 else 0
        
        # Skewness
        skewness = self._calculate_skewness(returns_array)
        
        secondary_metrics = {
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'recovery_factor': recovery_factor,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'skewness': skewness
        }
        
        return secondary_metrics
    
    def calculate_betting_specific_metrics(self, betting_history: List[Dict]) -> Dict[str, float]:
        """Calculate betting-specific performance metrics"""
        self.logger.info("Calculating betting-specific metrics")
        
        if not betting_history:
            return {}
        
        # ROI (Return on Investment)
        total_invested = sum([bet.get('position_size', 0) for bet in betting_history])
        total_profit = sum([bet.get('net_profit', 0) for bet in betting_history])
        roi = total_profit / total_invested if total_invested > 0 else 0
        
        # Average odds
        avg_odds = np.mean([bet.get('odds', 0) for bet in betting_history])
        
        # Average bet size
        avg_bet_size = np.mean([bet.get('position_size', 0) for bet in betting_history])
        
        # Bets per day
        if betting_history:
            first_date = min([bet.get('date', datetime.now()) for bet in betting_history])
            last_date = max([bet.get('date', datetime.now()) for bet in betting_history])
            trading_days = (last_date - first_date).days + 1
            bets_per_day = len(betting_history) / trading_days if trading_days > 0 else 0
        else:
            bets_per_day = 0
        
        # Confidence accuracy
        confidence_predictions = [(bet.get('confidence', 0), bet.get('outcome') == 'win') for bet in betting_history]
        if confidence_predictions:
            confidence_accuracy = self._calculate_confidence_accuracy(confidence_predictions)
        else:
            confidence_accuracy = 0
        
        # Kelly efficiency
        kelly_fractions = [bet.get('kelly_fraction', 0) for bet in betting_history]
        if kelly_fractions:
            kelly_efficiency = self._calculate_kelly_efficiency(betting_history)
        else:
            kelly_efficiency = 0
        
        betting_metrics = {
            'roi': roi,
            'avg_odds': avg_odds,
            'avg_bet_size': avg_bet_size,
            'bets_per_day': bets_per_day,
            'confidence_accuracy': confidence_accuracy,
            'kelly_efficiency': kelly_efficiency
        }
        
        return betting_metrics
    
    def calculate_risk_metrics(self, returns: List[float]) -> Dict[str, float]:
        """Calculate risk-related performance metrics"""
        self.logger.info("Calculating risk metrics")
        
        if not returns:
            return {}
        
        returns_array = np.array(returns)
        
        # Volatility (annualized)
        volatility = np.std(returns_array) * np.sqrt(252) if len(returns_array) > 1 else 0
        
        # Downside deviation
        downside_returns = returns_array[returns_array < 0]
        downside_deviation = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 1 else 0
        
        # Tail ratio (95th percentile / 5th percentile)
        if len(returns_array) > 0:
            tail_ratio = np.percentile(returns_array, 95) / abs(np.percentile(returns_array, 5)) if np.percentile(returns_array, 5) != 0 else 0
        else:
            tail_ratio = 0
        
        # Ulcer Index
        ulcer_index = self._calculate_ulcer_index(returns_array)
        
        # Pain Index
        pain_index = self._calculate_pain_index(returns_array)
        
        # Stability ratio
        stability_ratio = self._calculate_stability_ratio(returns_array)
        
        risk_metrics = {
            'volatility': volatility,
            'downside_deviation': downside_deviation,
            'tail_ratio': tail_ratio,
            'ulcer_index': ulcer_index,
            'pain_index': pain_index,
            'stability_ratio': stability_ratio
        }
        
        return risk_metrics
    
    def _calculate_skewness(self, returns: np.ndarray) -> float:
        """Calculate skewness of returns"""
        if len(returns) < 3:
            return 0
        
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0
        
        skewness = np.mean(((returns - mean_return) / std_return) ** 3)
        return skewness
    
    def _calculate_confidence_accuracy(self, confidence_predictions: List[Tuple[float, bool]]) -> float:
        """Calculate confidence accuracy (calibration)"""
        if not confidence_predictions:
            return 0
        
        # Sort by confidence
        sorted_predictions = sorted(confidence_predictions, key=lambda x: x[0])
        
        # Calculate calibration error
        n_bins = 10
        bin_size = len(sorted_predictions) // n_bins
        
        calibration_error = 0
        
        for i in range(n_bins):
            start_idx = i * bin_size
            end_idx = (i + 1) * bin_size if i < n_bins - 1 else len(sorted_predictions)
            
            bin_predictions = sorted_predictions[start_idx:end_idx]
            
            if bin_predictions:
                avg_confidence = np.mean([pred[0] for pred in bin_predictions])
                actual_accuracy = np.mean([pred[1] for pred in bin_predictions])
                
                calibration_error += abs(avg_confidence - actual_accuracy)
        
        # Return inverse of calibration error (higher is better)
        confidence_accuracy = 1 - (calibration_error / n_bins)
        return max(0, confidence_accuracy)
    
    def _calculate_kelly_efficiency(self, betting_history: List[Dict]) -> float:
        """Calculate Kelly Criterion efficiency"""
        if not betting_history:
            return 0
        
        # Calculate optimal Kelly fractions
        optimal_kelly_fractions = []
        actual_kelly_fractions = []
        
        for bet in betting_history:
            probability = bet.get('probability', 0)
            odds = bet.get('odds', 0)
            
            if probability > 0 and odds > 1:
                # Optimal Kelly: f = (bp - q) / b
                b = odds - 1
                p = probability
                q = 1 - p
                
                optimal_kelly = (b * p - q) / b
                actual_kelly = bet.get('kelly_fraction', 0)
                
                if optimal_kelly > 0:
                    optimal_kelly_fractions.append(optimal_kelly)
                    actual_kelly_fractions.append(actual_kelly)
        
        if not optimal_kelly_fractions:
            return 0
        
        # Calculate efficiency as ratio of actual to optimal
        efficiency = np.mean(actual_kelly_fractions) / np.mean(optimal_kelly_fractions)
        return efficiency
    
    def _calculate_ulcer_index(self, returns: np.ndarray) -> float:
        """Calculate Ulcer Index"""
        if len(returns) < 2:
            return 0
        
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / (running_max + 1e-10)
        
        # Ulcer Index is the square root of the mean of squared drawdowns
        ulcer_index = np.sqrt(np.mean(drawdowns ** 2))
        return ulcer_index
    
    def _calculate_pain_index(self, returns: np.ndarray) -> float:
        """Calculate Pain Index"""
        if len(returns) < 2:
            return 0
        
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / (running_max + 1e-10)
        
        # Pain Index is the mean of drawdowns
        pain_index = np.mean(drawdowns)
        return pain_index
    
    def _calculate_stability_ratio(self, returns: np.ndarray) -> float:
        """Calculate Stability Ratio (mean return / volatility)"""
        if len(returns) < 2:
            return 0
        
        mean_return = np.mean(returns)
        volatility = np.std(returns)
        
        if volatility == 0:
            return 0
        
        stability_ratio = mean_return / volatility
        return stability_ratio
    
    def calculate_comprehensive_metrics(self, returns: List[float], 
                                      betting_history: List[Dict] = None) -> Dict[str, Any]:
        """Calculate comprehensive performance metrics"""
        self.logger.info("Calculating comprehensive performance metrics")
        
        # Calculate all metric categories
        primary_metrics = self.calculate_primary_metrics(returns, betting_history)
        secondary_metrics = self.calculate_secondary_metrics(returns)
        risk_metrics = self.calculate_risk_metrics(returns)
        
        betting_metrics = {}
        if betting_history:
            betting_metrics = self.calculate_betting_specific_metrics(betting_history)
        
        # Combine all metrics
        comprehensive_metrics = {
            'primary': primary_metrics,
            'secondary': secondary_metrics,
            'betting_specific': betting_metrics,
            'risk': risk_metrics
        }
        
        # Calculate overall score
        overall_score = self._calculate_overall_score(comprehensive_metrics)
        comprehensive_metrics['overall_score'] = overall_score
        
        # Calculate benchmark comparison
        benchmark_comparison = self._compare_with_benchmarks(comprehensive_metrics)
        comprehensive_metrics['benchmark_comparison'] = benchmark_comparison
        
        # Statistical significance
        if len(returns) >= self.config['statistical_tests']['min_samples']:
            statistical_tests = self._perform_statistical_tests(returns)
            comprehensive_metrics['statistical_tests'] = statistical_tests
        
        self.performance_summary = comprehensive_metrics
        return comprehensive_metrics
    
    def _calculate_overall_score(self, metrics: Dict[str, Any]) -> float:
        """Calculate overall performance score"""
        score = 0
        total_weight = 0
        
        # Primary metrics
        for metric_name, metric_config in self.config['metrics']['primary'].items():
            if metric_config['enabled'] and metric_name in metrics['primary']:
                value = metrics['primary'][metric_name]
                
                # Normalize value based on metric type
                if metric_name == 'total_return':
                    normalized_value = min(1, max(0, (value + 0.2) / 0.4))  # -20% to +20%
                elif metric_name == 'sharpe_ratio':
                    normalized_value = min(1, max(0, value / 2))  # 0 to 2
                elif metric_name == 'max_drawdown':
                    normalized_value = max(0, 1 - value / 0.3)  # 0% to 30%
                elif metric_name == 'win_rate':
                    normalized_value = value  # 0 to 1
                elif metric_name == 'profit_factor':
                    normalized_value = min(1, max(0, (value - 1) / 2))  # 1 to 3
                else:
                    normalized_value = min(1, max(0, value))
                
                score += normalized_value * metric_config['weight']
                total_weight += metric_config['weight']
        
        # Secondary metrics
        for metric_name, metric_config in self.config['metrics']['secondary'].items():
            if metric_config['enabled'] and metric_name in metrics['secondary']:
                value = metrics['secondary'][metric_name]
                
                # Normalize value
                if metric_name in ['sortino_ratio', 'calmar_ratio']:
                    normalized_value = min(1, max(0, value / 2))
                elif metric_name == 'recovery_factor':
                    normalized_value = min(1, max(0, value / 3))
                elif metric_name in ['var_95', 'cvar_95']:
                    normalized_value = max(0, 1 + value / 0.1)  # -10% to 0%
                else:
                    normalized_value = min(1, max(0, value))
                
                score += normalized_value * metric_config['weight']
                total_weight += metric_config['weight']
        
        # Betting-specific metrics
        for metric_name, metric_config in self.config['metrics']['betting_specific'].items():
            if metric_config['enabled'] and metric_name in metrics['betting_specific']:
                value = metrics['betting_specific'][metric_name]
                
                # Normalize value
                if metric_name == 'roi':
                    normalized_value = min(1, max(0, (value + 0.2) / 0.4))  # -20% to +20%
                elif metric_name == 'avg_odds':
                    normalized_value = min(1, max(0, (value - 1.5) / 3.5))  # 1.5 to 5.0
                elif metric_name == 'bets_per_day':
                    normalized_value = min(1, max(0, value / 5))  # 0 to 5
                elif metric_name in ['confidence_accuracy', 'kelly_efficiency']:
                    normalized_value = value  # 0 to 1
                else:
                    normalized_value = min(1, max(0, value))
                
                score += normalized_value * metric_config['weight']
                total_weight += metric_config['weight']
        
        # Risk metrics
        if 'risk' in metrics and 'risk' in self.config.get('metrics', {}):
            for metric_name, metric_config in self.config['metrics']['risk'].items():
                if metric_config['enabled'] and metric_name in metrics['risk']:
                    value = metrics['risk'][metric_name]
                
                # Normalize value (lower is better for risk metrics)
                if metric_name == 'volatility':
                    normalized_value = max(0, 1 - value / 0.3)  # 0% to 30%
                elif metric_name == 'downside_deviation':
                    normalized_value = max(0, 1 - value / 0.2)  # 0% to 20%
                elif metric_name == 'tail_ratio':
                    normalized_value = min(1, max(0, value / 3))  # 0 to 3
                elif metric_name in ['ulcer_index', 'pain_index']:
                    normalized_value = max(0, 1 - value / 0.1)  # 0 to 0.1
                else:
                    normalized_value = min(1, max(0, value))
                
                score += normalized_value * metric_config['weight']
                total_weight += metric_config['weight']
        
        overall_score = score / total_weight if total_weight > 0 else 0
        return overall_score
    
    def _compare_with_benchmarks(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Compare performance with benchmarks"""
        benchmark_comparison = {}
        
        # Sharpe ratio comparison
        if 'sharpe_ratio' in metrics['primary']:
            sharpe_ratio = metrics['primary']['sharpe_ratio']
            benchmark_sharpe = self.config['benchmarks']['benchmark_sharpe']
            benchmark_comparison['sharpe_ratio'] = {
                'value': sharpe_ratio,
                'benchmark': benchmark_sharpe,
                'outperformance': sharpe_ratio - benchmark_sharpe,
                'outperformance_pct': (sharpe_ratio - benchmark_sharpe) / benchmark_sharpe * 100
            }
        
        # Volatility comparison
        if 'risk' in metrics and 'volatility' in metrics['risk']:
            volatility = metrics['risk']['volatility']
            benchmark_volatility = self.config['benchmarks']['benchmark_volatility']
            benchmark_comparison['volatility'] = {
                'value': volatility,
                'benchmark': benchmark_volatility,
                'difference': volatility - benchmark_volatility,
                'difference_pct': (volatility - benchmark_volatility) / benchmark_volatility * 100
            }
        
        return benchmark_comparison
    
    def _perform_statistical_tests(self, returns: List[float]) -> Dict[str, Any]:
        """Perform statistical significance tests"""
        if len(returns) < self.config['statistical_tests']['min_samples']:
            return {}
        
        returns_array = np.array(returns)
        
        # Bootstrap test for mean return
        bootstrap_samples = self.config['statistical_tests']['bootstrap_samples']
        bootstrap_means = []
        
        for _ in range(bootstrap_samples):
            bootstrap_sample = np.random.choice(returns_array, size=len(returns_array), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        bootstrap_means = np.array(bootstrap_means)
        
        # Calculate confidence interval
        alpha = 1 - self.config['statistical_tests']['confidence_level']
        lower_bound = np.percentile(bootstrap_means, alpha/2 * 100)
        upper_bound = np.percentile(bootstrap_means, (1 - alpha/2) * 100)
        
        # Test if mean is significantly different from zero
        mean_return = np.mean(returns_array)
        p_value = 2 * min(
            np.mean(bootstrap_means <= 0),
            np.mean(bootstrap_means >= 0)
        )
        
        statistical_tests = {
            'bootstrap_test': {
                'mean_return': mean_return,
                'confidence_interval': [lower_bound, upper_bound],
                'p_value': p_value,
                'significant': p_value < self.config['statistical_tests']['significance_threshold']
            }
        }
        
        return statistical_tests
    
    def validate_performance(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Validate performance against conservative thresholds"""
        self.logger.info("Validating performance against conservative thresholds")
        
        validation_results = {
            'validation_passed': True,
            'issues': [],
            'warnings': []
        }
        
        thresholds = self.config['conservative_thresholds']
        
        # Check primary metrics
        if 'primary' in metrics:
            primary = metrics['primary']
            
            # Win rate
            if 'win_rate' in primary:
                if primary['win_rate'] < thresholds['min_win_rate']:
                    validation_results['validation_passed'] = False
                    validation_results['issues'].append(f"Low win rate: {primary['win_rate']:.2%} < {thresholds['min_win_rate']:.2%}")
            
            # Max drawdown
            if 'max_drawdown' in primary:
                if primary['max_drawdown'] > thresholds['max_max_drawdown']:
                    validation_results['validation_passed'] = False
                    validation_results['issues'].append(f"High max drawdown: {primary['max_drawdown']:.2%} > {thresholds['max_max_drawdown']:.2%}")
            
            # Sharpe ratio
            if 'sharpe_ratio' in primary:
                if primary['sharpe_ratio'] < thresholds['min_sharpe_ratio']:
                    validation_results['validation_passed'] = False
                    validation_results['issues'].append(f"Low Sharpe ratio: {primary['sharpe_ratio']:.2f} < {thresholds['min_sharpe_ratio']:.2f}")
            
            # Profit factor
            if 'profit_factor' in primary:
                if primary['profit_factor'] < thresholds['min_profit_factor']:
                    validation_results['validation_passed'] = False
                    validation_results['issues'].append(f"Low profit factor: {primary['profit_factor']:.2f} < {thresholds['min_profit_factor']:.2f}")
        
        # Check risk metrics
        if 'risk' in metrics:
            risk = metrics['risk']
            
            # Volatility
            if 'volatility' in risk:
                if risk['volatility'] > thresholds['max_volatility']:
                    validation_results['validation_passed'] = False
                    validation_results['issues'].append(f"High volatility: {risk['volatility']:.2%} > {thresholds['max_volatility']:.2%}")
        
        # Check overall score
        if 'overall_score' in metrics:
            if metrics['overall_score'] < 0.5:
                validation_results['warnings'].append(f"Low overall score: {metrics['overall_score']:.2f}")
        
        self.logger.info(f"Performance validation: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
        
        return validation_results
    
    def generate_performance_report(self, metrics: Dict[str, Any], 
                                  validation_results: Dict[str, Any] = None) -> str:
        """Generate comprehensive performance report"""
        report = []
        report.append("=" * 80)
        report.append("NON-MAJOR LEAGUE PERFORMANCE METRICS REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall score
        if 'overall_score' in metrics:
            report.append("OVERALL PERFORMANCE SCORE:")
            report.append(f"  Score: {metrics['overall_score']:.3f} / 1.000")
            report.append(f"  Grade: {self._get_performance_grade(metrics['overall_score'])}")
            report.append("")
        
        # Primary metrics
        if 'primary' in metrics:
            report.append("PRIMARY METRICS:")
            for metric, value in metrics['primary'].items():
                if isinstance(value, float):
                    if metric in ['total_return', 'max_drawdown']:
                        report.append(f"  {metric.replace('_', ' ').title()}: {value:.2%}")
                    elif metric == 'win_rate':
                        report.append(f"  {metric.replace('_', ' ').title()}: {value:.2%}")
                    else:
                        report.append(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            report.append("")
        
        # Secondary metrics
        if 'secondary' in metrics:
            report.append("SECONDARY METRICS:")
            for metric, value in metrics['secondary'].items():
                if isinstance(value, float):
                    if metric in ['var_95', 'cvar_95']:
                        report.append(f"  {metric.replace('_', ' ').title()}: {value:.2%}")
                    else:
                        report.append(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            report.append("")
        
        # Betting-specific metrics
        if 'betting_specific' in metrics:
            report.append("BETTING-SPECIFIC METRICS:")
            for metric, value in metrics['betting_specific'].items():
                if isinstance(value, float):
                    if metric == 'roi':
                        report.append(f"  {metric.replace('_', ' ').title()}: {value:.2%}")
                    elif metric == 'avg_odds':
                        report.append(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
                    elif metric == 'bets_per_day':
                        report.append(f"  {metric.replace('_', ' ').title()}: {value:.2f}")
                    else:
                        report.append(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            report.append("")
        
        # Risk metrics
        if 'risk' in metrics and metrics['risk']:
            report.append("RISK METRICS:")
            for metric, value in metrics['risk'].items():
                if isinstance(value, float):
                    if metric in ['volatility', 'downside_deviation']:
                        report.append(f"  {metric.replace('_', ' ').title()}: {value:.2%}")
                    else:
                        report.append(f"  {metric.replace('_', ' ').title()}: {value:.4f}")
            report.append("")
        
        # Benchmark comparison
        if 'benchmark_comparison' in metrics:
            report.append("BENCHMARK COMPARISON:")
            for metric, comparison in metrics['benchmark_comparison'].items():
                report.append(f"  {metric.replace('_', ' ').title()}:")
                if 'value' in comparison:
                    report.append(f"    Value: {comparison['value']:.4f}")
                if 'benchmark' in comparison:
                    report.append(f"    Benchmark: {comparison['benchmark']:.4f}")
                if 'outperformance' in comparison:
                    report.append(f"    Outperformance: {comparison['outperformance']:+.4f}")
                if 'outperformance_pct' in comparison:
                    report.append(f"    Outperformance %: {comparison['outperformance_pct']:+.2f}%")
                if 'difference' in comparison:
                    report.append(f"    Difference: {comparison['difference']:+.4f}")
                if 'difference_pct' in comparison:
                    report.append(f"    Difference %: {comparison['difference_pct']:+.2f}%")
            report.append("")
        
        # Statistical tests
        if 'statistical_tests' in metrics:
            report.append("STATISTICAL TESTS:")
            for test_name, test_results in metrics['statistical_tests'].items():
                report.append(f"  {test_name.replace('_', ' ').title()}:")
                report.append(f"    Mean Return: {test_results['mean_return']:.4f}")
                report.append(f"    Confidence Interval: [{test_results['confidence_interval'][0]:.4f}, {test_results['confidence_interval'][1]:.4f}]")
                report.append(f"    P-value: {test_results['p_value']:.4f}")
                report.append(f"    Significant: {'Yes' if test_results['significant'] else 'No'}")
            report.append("")
        
        # Validation results
        if validation_results:
            report.append("VALIDATION RESULTS:")
            report.append(f"  Status: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
            
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
        if validation_results and validation_results['validation_passed']:
            report.append("  ✅ Performance meets conservative thresholds")
            report.append("  📈 Suitable for live testing with proper risk management")
            report.append("  🔍 Continue monitoring performance metrics")
        else:
            report.append("  ❌ Performance does not meet conservative thresholds")
            report.append("  🔧 Review strategy and address identified issues")
            report.append("  📊 Consider additional data or model improvements")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def _get_performance_grade(self, score: float) -> str:
        """Get performance grade based on score"""
        if score >= 0.9:
            return "A+ (Excellent)"
        elif score >= 0.8:
            return "A (Very Good)"
        elif score >= 0.7:
            return "B+ (Good)"
        elif score >= 0.6:
            return "B (Satisfactory)"
        elif score >= 0.5:
            return "C (Acceptable)"
        elif score >= 0.4:
            return "D (Poor)"
        else:
            return "F (Fail)"
    
    def save_performance_metrics(self, filepath: str):
        """Save performance metrics"""
        self.logger.info(f"Saving performance metrics to {filepath}")
        
        import joblib
        
        metrics_state = {
            'performance_summary': self.performance_summary,
            'metrics_history': self.metrics_history,
            'config': self.config
        }
        
        joblib.dump(metrics_state, filepath)
        self.logger.info("Performance metrics saved successfully")

# Example usage
def main():
    """Example usage of NonMajorLeaguePerformanceMetrics"""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 100
    
    # Sample returns
    returns = np.random.normal(0.001, 0.02, n_samples).tolist()
    
    # Sample betting history
    betting_history = []
    for i in range(50):
        bet = {
            'id': i + 1,
            'date': datetime.now() - timedelta(days=n_samples-i),
            'probability': np.random.uniform(0.6, 0.9),
            'odds': np.random.uniform(2.0, 4.0),
            'position_size': np.random.uniform(10, 50),
            'confidence': np.random.uniform(0.6, 0.9),
            'kelly_fraction': np.random.uniform(0.01, 0.05),
            'outcome': np.random.choice(['win', 'loss']),
            'net_profit': np.random.uniform(-50, 100)
        }
        betting_history.append(bet)
    
    # Initialize performance metrics
    metrics_calculator = NonMajorLeaguePerformanceMetrics()
    
    # Calculate comprehensive metrics
    comprehensive_metrics = metrics_calculator.calculate_comprehensive_metrics(returns, betting_history)
    
    # Validate performance
    validation_results = metrics_calculator.validate_performance(comprehensive_metrics)
    
    # Generate report
    report = metrics_calculator.generate_performance_report(comprehensive_metrics, validation_results)
    
    print(report)
    
    # Save metrics
    metrics_calculator.save_performance_metrics('performance_metrics.pkl')

if __name__ == "__main__":
    main()
