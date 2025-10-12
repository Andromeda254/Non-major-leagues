import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeagueRiskManagement:
    """
    Comprehensive risk management system for non-major soccer leagues
    
    Key Features:
    - Multi-level risk controls
    - Dynamic position sizing
    - Drawdown protection
    - Volatility management
    - Correlation monitoring
    - Stress testing
    - Real-time risk monitoring
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize risk management system
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.risk_history = []
        self.current_risk_state = {}
        self.risk_alerts = []
        
    def setup_logging(self):
        """Setup logging for risk management"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load risk management configuration"""
        if config is None:
            self.config = {
                'capital_protection': {
                    'max_capital_at_risk': 0.2,  # 20% max capital at risk
                    'reserve_capital': 0.1,       # 10% reserve capital
                    'emergency_stop': 0.15,      # 15% emergency stop
                    'position_size_limit': 0.05, # 5% max position size
                    'daily_loss_limit': 0.03,    # 3% daily loss limit
                    'weekly_loss_limit': 0.08,   # 8% weekly loss limit
                    'monthly_loss_limit': 0.15   # 15% monthly loss limit
                },
                'drawdown_management': {
                    'max_drawdown': 0.2,         # 20% max drawdown
                    'drawdown_warning': 0.1,      # 10% drawdown warning
                    'drawdown_recovery': 0.05,   # 5% recovery threshold
                    'drawdown_penalty': 0.5,     # 50% position size reduction
                    'recovery_bonus': 1.2,       # 20% position size increase
                    'drawdown_decay': 0.95       # 95% decay factor
                },
                'volatility_management': {
                    'max_volatility': 0.3,       # 30% max volatility
                    'volatility_window': 20,      # 20-day volatility window
                    'volatility_threshold': 0.2, # 20% volatility threshold
                    'volatility_penalty': 0.7,   # 30% position size reduction
                    'volatility_reward': 1.1     # 10% position size increase
                },
                'correlation_management': {
                    'max_correlation': 0.7,      # 70% max correlation
                    'correlation_window': 30,     # 30-day correlation window
                    'correlation_penalty': 0.6,  # 40% position size reduction
                    'diversification_bonus': 1.15 # 15% position size increase
                },
                'stress_testing': {
                    'enabled': True,
                    'scenarios': ['market_crash', 'high_volatility', 'correlation_breakdown'],
                    'stress_multiplier': 0.5,    # 50% position size reduction
                    'stress_threshold': 0.1      # 10% stress threshold
                },
                'monitoring': {
                    'real_time': True,
                    'alert_thresholds': {
                        'drawdown': 0.05,        # 5% drawdown alert
                        'volatility': 0.15,      # 15% volatility alert
                        'correlation': 0.6,     # 60% correlation alert
                        'loss_rate': 0.4         # 40% loss rate alert
                    },
                    'update_frequency': 300      # 5 minutes
                },
                'recovery_protocols': {
                    'drawdown_recovery': {
                        'enabled': True,
                        'recovery_threshold': 0.05,
                        'recovery_bonus': 1.2,
                        'recovery_duration': 7   # 7 days
                    },
                    'volatility_recovery': {
                        'enabled': True,
                        'recovery_threshold': 0.15,
                        'recovery_bonus': 1.1,
                        'recovery_duration': 5   # 5 days
                    }
                }
            }
        else:
            self.config = config
    
    def calculate_current_risk_metrics(self, betting_history: List[Dict], 
                                     current_capital: float) -> Dict[str, Any]:
        """Calculate current risk metrics"""
        self.logger.info("Calculating current risk metrics")
        
        if not betting_history:
            return {}
        
        # Basic risk metrics
        total_bets = len(betting_history)
        winning_bets = [bet for bet in betting_history if bet.get('outcome') == 'win']
        losing_bets = [bet for bet in betting_history if bet.get('outcome') == 'loss']
        
        win_rate = len(winning_bets) / total_bets if total_bets > 0 else 0
        loss_rate = len(losing_bets) / total_bets if total_bets > 0 else 0
        
        # Calculate returns
        returns = [bet.get('net_profit', 0) for bet in betting_history]
        total_return = sum(returns)
        
        # Drawdown calculation
        cumulative_returns = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (running_max - cumulative_returns) / (running_max + 1e-10)
        current_drawdown = drawdowns[-1] if len(drawdowns) > 0 else 0
        max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0
        
        # Volatility calculation
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
        else:
            volatility = 0
        
        # Correlation calculation (simplified)
        correlation = self._calculate_correlation(betting_history)
        
        # Position sizing metrics
        position_sizes = [bet.get('position_size', 0) for bet in betting_history]
        avg_position_size = np.mean(position_sizes) if position_sizes else 0
        max_position_size = np.max(position_sizes) if position_sizes else 0
        
        # Time-based metrics
        recent_bets = self._get_recent_bets(betting_history, days=7)
        daily_loss = sum([bet.get('net_profit', 0) for bet in recent_bets if bet.get('net_profit', 0) < 0])
        
        risk_metrics = {
            'total_bets': total_bets,
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'total_return': total_return,
            'current_drawdown': current_drawdown,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'correlation': correlation,
            'avg_position_size': avg_position_size,
            'max_position_size': max_position_size,
            'daily_loss': daily_loss,
            'current_capital': current_capital
        }
        
        self.current_risk_state = risk_metrics
        return risk_metrics
    
    def _calculate_correlation(self, betting_history: List[Dict]) -> float:
        """Calculate correlation between recent bets"""
        if len(betting_history) < 10:
            return 0
        
        # Get recent bets
        recent_bets = betting_history[-20:]  # Last 20 bets
        
        # Calculate correlation based on outcomes
        outcomes = [1 if bet.get('outcome') == 'win' else 0 for bet in recent_bets]
        
        if len(outcomes) < 2:
            return 0
        
        # Calculate autocorrelation
        correlation = np.corrcoef(outcomes[:-1], outcomes[1:])[0, 1]
        return correlation if not np.isnan(correlation) else 0
    
    def _get_recent_bets(self, betting_history: List[Dict], days: int = 7) -> List[Dict]:
        """Get recent bets within specified days"""
        cutoff_date = datetime.now() - timedelta(days=days)
        return [bet for bet in betting_history 
                if bet.get('date', datetime.now()) >= cutoff_date]
    
    def assess_risk_level(self, risk_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall risk level"""
        self.logger.info("Assessing risk level")
        
        risk_level = 'LOW'
        risk_score = 0
        risk_factors = []
        
        # Drawdown assessment
        current_drawdown = risk_metrics.get('current_drawdown', 0)
        max_drawdown = risk_metrics.get('max_drawdown', 0)
        
        if current_drawdown > self.config['drawdown_management']['drawdown_warning']:
            risk_score += 30
            risk_factors.append(f"High current drawdown: {current_drawdown:.2%}")
        
        if max_drawdown > self.config['drawdown_management']['max_drawdown']:
            risk_score += 50
            risk_factors.append(f"Excessive max drawdown: {max_drawdown:.2%}")
        
        # Volatility assessment
        volatility = risk_metrics.get('volatility', 0)
        if volatility > self.config['volatility_management']['volatility_threshold']:
            risk_score += 25
            risk_factors.append(f"High volatility: {volatility:.2%}")
        
        # Loss rate assessment
        loss_rate = risk_metrics.get('loss_rate', 0)
        if loss_rate > self.config['monitoring']['alert_thresholds']['loss_rate']:
            risk_score += 20
            risk_factors.append(f"High loss rate: {loss_rate:.2%}")
        
        # Correlation assessment
        correlation = abs(risk_metrics.get('correlation', 0))
        if correlation > self.config['correlation_management']['max_correlation']:
            risk_score += 15
            risk_factors.append(f"High correlation: {correlation:.2%}")
        
        # Daily loss assessment
        daily_loss = risk_metrics.get('daily_loss', 0)
        current_capital = risk_metrics.get('current_capital', 1)
        daily_loss_pct = abs(daily_loss) / current_capital if current_capital > 0 else 0
        
        if daily_loss_pct > self.config['capital_protection']['daily_loss_limit']:
            risk_score += 35
            risk_factors.append(f"High daily loss: {daily_loss_pct:.2%}")
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = 'CRITICAL'
        elif risk_score >= 50:
            risk_level = 'HIGH'
        elif risk_score >= 30:
            risk_level = 'MEDIUM'
        elif risk_score >= 15:
            risk_level = 'LOW'
        else:
            risk_level = 'MINIMAL'
        
        risk_assessment = {
            'risk_level': risk_level,
            'risk_score': risk_score,
            'risk_factors': risk_factors,
            'recommendations': self._get_risk_recommendations(risk_level, risk_factors)
        }
        
        return risk_assessment
    
    def _get_risk_recommendations(self, risk_level: str, risk_factors: List[str]) -> List[str]:
        """Get risk management recommendations"""
        recommendations = []
        
        if risk_level == 'CRITICAL':
            recommendations.extend([
                "IMMEDIATE ACTION REQUIRED",
                "Stop all betting activities",
                "Review and revise strategy",
                "Consider capital injection",
                "Implement emergency protocols"
            ])
        elif risk_level == 'HIGH':
            recommendations.extend([
                "Reduce position sizes by 50%",
                "Increase confidence thresholds",
                "Implement stricter risk controls",
                "Monitor performance closely",
                "Consider strategy modification"
            ])
        elif risk_level == 'MEDIUM':
            recommendations.extend([
                "Reduce position sizes by 25%",
                "Increase monitoring frequency",
                "Review recent betting decisions",
                "Consider temporary pause",
                "Implement additional safeguards"
            ])
        elif risk_level == 'LOW':
            recommendations.extend([
                "Continue with caution",
                "Monitor key risk metrics",
                "Maintain current position sizes",
                "Review risk management protocols"
            ])
        else:  # MINIMAL
            recommendations.extend([
                "Continue normal operations",
                "Maintain risk monitoring",
                "Consider position size increases",
                "Optimize strategy parameters"
            ])
        
        return recommendations
    
    def calculate_position_size_adjustment(self, base_position_size: float, 
                                         risk_metrics: Dict[str, Any]) -> float:
        """Calculate position size adjustment based on risk metrics"""
        self.logger.info("Calculating position size adjustment")
        
        adjustment_factor = 1.0
        
        # Drawdown adjustment
        current_drawdown = risk_metrics.get('current_drawdown', 0)
        if current_drawdown > self.config['drawdown_management']['drawdown_warning']:
            drawdown_penalty = self.config['drawdown_management']['drawdown_penalty']
            adjustment_factor *= drawdown_penalty
        
        # Volatility adjustment
        volatility = risk_metrics.get('volatility', 0)
        if volatility > self.config['volatility_management']['volatility_threshold']:
            volatility_penalty = self.config['volatility_management']['volatility_penalty']
            adjustment_factor *= volatility_penalty
        
        # Correlation adjustment
        correlation = abs(risk_metrics.get('correlation', 0))
        if correlation > self.config['correlation_management']['max_correlation']:
            correlation_penalty = self.config['correlation_management']['correlation_penalty']
            adjustment_factor *= correlation_penalty
        
        # Loss rate adjustment
        loss_rate = risk_metrics.get('loss_rate', 0)
        if loss_rate > 0.4:  # 40% loss rate
            adjustment_factor *= 0.8
        
        # Daily loss adjustment
        daily_loss = risk_metrics.get('daily_loss', 0)
        current_capital = risk_metrics.get('current_capital', 1)
        daily_loss_pct = abs(daily_loss) / current_capital if current_capital > 0 else 0
        
        if daily_loss_pct > self.config['capital_protection']['daily_loss_limit']:
            adjustment_factor *= 0.5
        
        # Apply bounds
        min_adjustment = 0.1  # Minimum 10% of base position
        max_adjustment = 1.5  # Maximum 150% of base position
        
        adjustment_factor = max(min_adjustment, min(max_adjustment, adjustment_factor))
        
        adjusted_position_size = base_position_size * adjustment_factor
        
        self.logger.info(f"Position size adjustment: {adjustment_factor:.2f}x, "
                        f"Adjusted size: {adjusted_position_size:.2f}")
        
        return adjusted_position_size
    
    def check_risk_limits(self, proposed_bet: Dict[str, Any], 
                         current_capital: float) -> Dict[str, Any]:
        """Check if proposed bet violates risk limits"""
        self.logger.info("Checking risk limits")
        
        risk_check = {
            'approved': True,
            'violations': [],
            'warnings': [],
            'adjusted_bet': proposed_bet.copy()
        }
        
        # Check position size limit
        position_size = proposed_bet.get('position_size', 0)
        max_position_size = current_capital * self.config['capital_protection']['position_size_limit']
        
        if position_size > max_position_size:
            risk_check['approved'] = False
            risk_check['violations'].append(f"Position size exceeds limit: {position_size:.2f} > {max_position_size:.2f}")
            risk_check['adjusted_bet']['position_size'] = max_position_size
        
        # Check capital at risk
        total_capital_at_risk = position_size / current_capital if current_capital > 0 else 0
        max_capital_at_risk = self.config['capital_protection']['max_capital_at_risk']
        
        if total_capital_at_risk > max_capital_at_risk:
            risk_check['approved'] = False
            risk_check['violations'].append(f"Capital at risk exceeds limit: {total_capital_at_risk:.2%} > {max_capital_at_risk:.2%}")
        
        # Check reserve capital
        reserve_capital = current_capital * self.config['capital_protection']['reserve_capital']
        if position_size > (current_capital - reserve_capital):
            risk_check['approved'] = False
            risk_check['violations'].append(f"Insufficient reserve capital: {position_size:.2f} > {current_capital - reserve_capital:.2f}")
        
        # Check daily loss limit
        # This would require access to today's betting history
        # For now, we'll assume it's checked elsewhere
        
        # Check volatility limits
        # This would require access to recent returns
        # For now, we'll assume it's checked elsewhere
        
        return risk_check
    
    def perform_stress_test(self, betting_history: List[Dict], 
                           current_capital: float) -> Dict[str, Any]:
        """Perform stress testing scenarios"""
        self.logger.info("Performing stress testing")
        
        if not self.config['stress_testing']['enabled']:
            return {'enabled': False}
        
        stress_results = {}
        
        # Market crash scenario
        if 'market_crash' in self.config['stress_testing']['scenarios']:
            crash_results = self._simulate_market_crash(betting_history, current_capital)
            stress_results['market_crash'] = crash_results
        
        # High volatility scenario
        if 'high_volatility' in self.config['stress_testing']['scenarios']:
            volatility_results = self._simulate_high_volatility(betting_history, current_capital)
            stress_results['high_volatility'] = volatility_results
        
        # Correlation breakdown scenario
        if 'correlation_breakdown' in self.config['stress_testing']['scenarios']:
            correlation_results = self._simulate_correlation_breakdown(betting_history, current_capital)
            stress_results['correlation_breakdown'] = correlation_results
        
        # Overall stress assessment
        stress_results['overall'] = self._assess_stress_results(stress_results)
        
        return stress_results
    
    def _simulate_market_crash(self, betting_history: List[Dict], 
                              current_capital: float) -> Dict[str, Any]:
        """Simulate market crash scenario"""
        # Assume 50% loss on all active bets
        active_bets = [bet for bet in betting_history if bet.get('status') == 'placed']
        
        total_exposure = sum([bet.get('position_size', 0) for bet in active_bets])
        crash_loss = total_exposure * 0.5
        
        remaining_capital = current_capital - crash_loss
        
        return {
            'scenario': 'market_crash',
            'total_exposure': total_exposure,
            'crash_loss': crash_loss,
            'remaining_capital': remaining_capital,
            'capital_ratio': remaining_capital / current_capital if current_capital > 0 else 0,
            'survivable': remaining_capital > current_capital * 0.5
        }
    
    def _simulate_high_volatility(self, betting_history: List[Dict], 
                                 current_capital: float) -> Dict[str, Any]:
        """Simulate high volatility scenario"""
        # Assume 2x normal volatility
        recent_bets = betting_history[-20:] if len(betting_history) >= 20 else betting_history
        
        if not recent_bets:
            return {'scenario': 'high_volatility', 'no_data': True}
        
        returns = [bet.get('net_profit', 0) for bet in recent_bets]
        normal_volatility = np.std(returns) if len(returns) > 1 else 0
        high_volatility = normal_volatility * 2
        
        # Simulate impact on position sizing
        volatility_penalty = self.config['volatility_management']['volatility_penalty']
        adjusted_position_sizes = [bet.get('position_size', 0) * volatility_penalty for bet in recent_bets]
        
        return {
            'scenario': 'high_volatility',
            'normal_volatility': normal_volatility,
            'high_volatility': high_volatility,
            'volatility_multiplier': 2.0,
            'position_size_reduction': volatility_penalty,
            'adjusted_exposure': sum(adjusted_position_sizes)
        }
    
    def _simulate_correlation_breakdown(self, betting_history: List[Dict], 
                                      current_capital: float) -> Dict[str, Any]:
        """Simulate correlation breakdown scenario"""
        # Assume all bets become highly correlated (all win or all lose)
        recent_bets = betting_history[-20:] if len(betting_history) >= 20 else betting_history
        
        if not recent_bets:
            return {'scenario': 'correlation_breakdown', 'no_data': True}
        
        # Simulate worst case: all bets lose
        total_exposure = sum([bet.get('position_size', 0) for bet in recent_bets])
        worst_case_loss = total_exposure
        
        remaining_capital = current_capital - worst_case_loss
        
        return {
            'scenario': 'correlation_breakdown',
            'total_exposure': total_exposure,
            'worst_case_loss': worst_case_loss,
            'remaining_capital': remaining_capital,
            'capital_ratio': remaining_capital / current_capital if current_capital > 0 else 0,
            'survivable': remaining_capital > current_capital * 0.3
        }
    
    def _assess_stress_results(self, stress_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall stress test results"""
        overall_assessment = {
            'overall_survivable': True,
            'critical_scenarios': [],
            'recommendations': []
        }
        
        for scenario, results in stress_results.items():
            if scenario == 'overall':
                continue
            
            if 'survivable' in results and not results['survivable']:
                overall_assessment['overall_survivable'] = False
                overall_assessment['critical_scenarios'].append(scenario)
        
        # Generate recommendations
        if not overall_assessment['overall_survivable']:
            overall_assessment['recommendations'].extend([
                "CRITICAL: System may not survive stress scenarios",
                "Reduce position sizes significantly",
                "Implement additional risk controls",
                "Consider capital injection",
                "Review and revise strategy"
            ])
        else:
            overall_assessment['recommendations'].extend([
                "System appears resilient to stress scenarios",
                "Continue monitoring risk metrics",
                "Maintain current risk controls",
                "Consider gradual position size increases"
            ])
        
        return overall_assessment
    
    def generate_risk_alerts(self, risk_metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate risk alerts based on current metrics"""
        alerts = []
        
        # Drawdown alert
        current_drawdown = risk_metrics.get('current_drawdown', 0)
        if current_drawdown > self.config['monitoring']['alert_thresholds']['drawdown']:
            alerts.append({
                'type': 'drawdown',
                'level': 'warning',
                'message': f"Current drawdown: {current_drawdown:.2%}",
                'threshold': self.config['monitoring']['alert_thresholds']['drawdown'],
                'timestamp': datetime.now()
            })
        
        # Volatility alert
        volatility = risk_metrics.get('volatility', 0)
        if volatility > self.config['monitoring']['alert_thresholds']['volatility']:
            alerts.append({
                'type': 'volatility',
                'level': 'warning',
                'message': f"High volatility: {volatility:.2%}",
                'threshold': self.config['monitoring']['alert_thresholds']['volatility'],
                'timestamp': datetime.now()
            })
        
        # Correlation alert
        correlation = abs(risk_metrics.get('correlation', 0))
        if correlation > self.config['monitoring']['alert_thresholds']['correlation']:
            alerts.append({
                'type': 'correlation',
                'level': 'warning',
                'message': f"High correlation: {correlation:.2%}",
                'threshold': self.config['monitoring']['alert_thresholds']['correlation'],
                'timestamp': datetime.now()
            })
        
        # Loss rate alert
        loss_rate = risk_metrics.get('loss_rate', 0)
        if loss_rate > self.config['monitoring']['alert_thresholds']['loss_rate']:
            alerts.append({
                'type': 'loss_rate',
                'level': 'critical',
                'message': f"High loss rate: {loss_rate:.2%}",
                'threshold': self.config['monitoring']['alert_thresholds']['loss_rate'],
                'timestamp': datetime.now()
            })
        
        # Store alerts
        self.risk_alerts.extend(alerts)
        
        return alerts
    
    def get_risk_summary(self, risk_metrics: Dict[str, Any], 
                        risk_assessment: Dict[str, Any]) -> Dict[str, Any]:
        """Get comprehensive risk summary"""
        risk_summary = {
            'timestamp': datetime.now(),
            'risk_metrics': risk_metrics,
            'risk_assessment': risk_assessment,
            'alerts': self.risk_alerts[-10:],  # Last 10 alerts
            'recommendations': risk_assessment.get('recommendations', []),
            'status': 'monitoring'
        }
        
        # Determine overall status
        if risk_assessment.get('risk_level') == 'CRITICAL':
            risk_summary['status'] = 'critical'
        elif risk_assessment.get('risk_level') == 'HIGH':
            risk_summary['status'] = 'warning'
        elif risk_assessment.get('risk_level') == 'MEDIUM':
            risk_summary['status'] = 'caution'
        else:
            risk_summary['status'] = 'normal'
        
        return risk_summary
    
    def save_risk_state(self, filepath: str):
        """Save current risk state"""
        self.logger.info(f"Saving risk state to {filepath}")
        
        import joblib
        
        risk_state = {
            'risk_history': self.risk_history,
            'current_risk_state': self.current_risk_state,
            'risk_alerts': self.risk_alerts,
            'config': self.config
        }
        
        joblib.dump(risk_state, filepath)
        self.logger.info("Risk state saved successfully")

# Example usage
def main():
    """Example usage of NonMajorLeagueRiskManagement"""
    
    # Create sample betting history
    betting_history = []
    for i in range(50):
        bet = {
            'id': i + 1,
            'date': datetime.now() - timedelta(days=50-i),
            'position_size': np.random.uniform(10, 50),
            'outcome': np.random.choice(['win', 'loss']),
            'net_profit': np.random.uniform(-50, 100),
            'status': 'settled'
        }
        betting_history.append(bet)
    
    # Initialize risk management
    risk_manager = NonMajorLeagueRiskManagement()
    
    # Calculate risk metrics
    risk_metrics = risk_manager.calculate_current_risk_metrics(betting_history, 1000)
    
    # Assess risk level
    risk_assessment = risk_manager.assess_risk_level(risk_metrics)
    
    # Calculate position size adjustment
    base_position_size = 50
    adjusted_position_size = risk_manager.calculate_position_size_adjustment(base_position_size, risk_metrics)
    
    # Check risk limits
    proposed_bet = {'position_size': 100}
    risk_check = risk_manager.check_risk_limits(proposed_bet, 1000)
    
    # Perform stress test
    stress_results = risk_manager.perform_stress_test(betting_history, 1000)
    
    # Generate risk alerts
    alerts = risk_manager.generate_risk_alerts(risk_metrics)
    
    # Get risk summary
    risk_summary = risk_manager.get_risk_summary(risk_metrics, risk_assessment)
    
    # Print results
    print("Risk Management Results:")
    print(f"Risk Level: {risk_assessment['risk_level']}")
    print(f"Risk Score: {risk_assessment['risk_score']}")
    print(f"Position Size Adjustment: {adjusted_position_size:.2f}")
    print(f"Risk Check Approved: {risk_check['approved']}")
    print(f"Alerts Generated: {len(alerts)}")
    
    # Save risk state
    risk_manager.save_risk_state('risk_state.pkl')

if __name__ == "__main__":
    main()
