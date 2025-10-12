import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeagueLiveTesting:
    """
    Live testing framework with paper trading for non-major soccer leagues
    
    Key Features:
    - Paper trading simulation
    - Real-time monitoring
    - Performance tracking
    - Risk management integration
    - Alert system
    - Performance reporting
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize live testing system
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.paper_trades = []
        self.live_performance = {}
        self.alerts = []
        self.testing_session = None
        
    def setup_logging(self):
        """Setup logging for live testing"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load live testing configuration"""
        if config is None:
            self.config = {
                'paper_trading': {
                    'enabled': True,
                    'initial_capital': 1000,
                    'commission': 0.05,  # 5% commission
                    'slippage': 0.02,    # 2% slippage
                    'market_impact': True,
                    'realistic_execution': True
                },
                'monitoring': {
                    'real_time': True,
                    'update_frequency': 60,  # 1 minute
                    'performance_window': 24,  # 24 hours
                    'alert_frequency': 300,   # 5 minutes
                    'log_frequency': 3600    # 1 hour
                },
                'performance_tracking': {
                    'metrics': ['return', 'drawdown', 'win_rate', 'sharpe_ratio'],
                    'benchmarks': ['risk_free_rate', 'market_return'],
                    'comparison_periods': ['daily', 'weekly', 'monthly'],
                    'real_time_calculation': True
                },
                'risk_management': {
                    'max_drawdown': 0.2,      # 20% max drawdown
                    'daily_loss_limit': 0.05, # 5% daily loss limit
                    'position_size_limit': 0.1, # 10% position size limit
                    'emergency_stop': 0.15,   # 15% emergency stop
                    'auto_stop': True
                },
                'alerts': {
                    'enabled': True,
                    'channels': ['console', 'file', 'email'],
                    'levels': ['info', 'warning', 'critical'],
                    'thresholds': {
                        'drawdown': 0.1,      # 10% drawdown alert
                        'loss_rate': 0.4,     # 40% loss rate alert
                        'volatility': 0.2,   # 20% volatility alert
                        'performance': -0.05  # -5% performance alert
                    }
                },
                'reporting': {
                    'real_time': True,
                    'daily_summary': True,
                    'weekly_report': True,
                    'monthly_report': True,
                    'performance_dashboard': True
                },
                'data_sources': {
                    'odds_api': {
                        'enabled': True,
                        'update_frequency': 300,  # 5 minutes
                        'timeout': 30
                    },
                    'match_data': {
                        'enabled': True,
                        'update_frequency': 600,  # 10 minutes
                        'timeout': 60
                    },
                    'market_data': {
                        'enabled': True,
                        'update_frequency': 60,   # 1 minute
                        'timeout': 15
                    }
                }
            }
        else:
            self.config = config
    
    def start_testing_session(self, session_name: str = None) -> Dict[str, Any]:
        """Start a new testing session"""
        self.logger.info("Starting live testing session")
        
        if session_name is None:
            session_name = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.testing_session = {
            'session_name': session_name,
            'start_time': datetime.now(),
            'initial_capital': self.config['paper_trading']['initial_capital'],
            'current_capital': self.config['paper_trading']['initial_capital'],
            'peak_capital': self.config['paper_trading']['initial_capital'],
            'status': 'active',
            'trades': [],
            'performance_metrics': {},
            'alerts': [],
            'last_update': datetime.now()
        }
        
        self.logger.info(f"Testing session started: {session_name}")
        return self.testing_session
    
    def stop_testing_session(self) -> Dict[str, Any]:
        """Stop the current testing session"""
        if not self.testing_session:
            return {'error': 'No active testing session'}
        
        self.logger.info("Stopping live testing session")
        
        self.testing_session['end_time'] = datetime.now()
        self.testing_session['status'] = 'stopped'
        self.testing_session['duration'] = (
            self.testing_session['end_time'] - self.testing_session['start_time']
        ).total_seconds()
        
        # Calculate final performance metrics
        final_metrics = self.calculate_session_performance()
        self.testing_session['final_metrics'] = final_metrics
        
        self.logger.info(f"Testing session stopped: {self.testing_session['session_name']}")
        return self.testing_session
    
    def place_paper_trade(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Place a paper trade"""
        if not self.testing_session or self.testing_session['status'] != 'active':
            return {'error': 'No active testing session'}
        
        self.logger.info("Placing paper trade")
        
        # Validate trade data
        validation_result = self._validate_trade_data(trade_data)
        if not validation_result['valid']:
            return {'error': 'Invalid trade data', 'details': validation_result['errors']}
        
        # Create paper trade
        paper_trade = {
            'trade_id': len(self.testing_session['trades']) + 1,
            'timestamp': datetime.now(),
            'match_id': trade_data.get('match_id'),
            'prediction': trade_data.get('prediction'),
            'confidence': trade_data.get('confidence'),
            'odds': trade_data.get('odds'),
            'position_size': trade_data.get('position_size'),
            'kelly_fraction': trade_data.get('kelly_fraction'),
            'status': 'placed',
            'execution_price': self._calculate_execution_price(trade_data),
            'commission': self._calculate_commission(trade_data),
            'slippage': self._calculate_slippage(trade_data),
            'market_impact': self._calculate_market_impact(trade_data),
            'net_profit': 0,
            'outcome': None,
            'settlement_time': None
        }
        
        # Update capital
        self.testing_session['current_capital'] -= paper_trade['position_size']
        
        # Record trade
        self.testing_session['trades'].append(paper_trade)
        self.paper_trades.append(paper_trade)
        
        # Update peak capital
        if self.testing_session['current_capital'] > self.testing_session['peak_capital']:
            self.testing_session['peak_capital'] = self.testing_session['current_capital']
        
        # Generate alert
        self._generate_trade_alert(paper_trade, 'placed')
        
        self.logger.info(f"Paper trade placed: ID={paper_trade['trade_id']}, "
                        f"Size=${paper_trade['position_size']:.2f}")
        
        return {
            'trade_placed': True,
            'trade': paper_trade,
            'current_capital': self.testing_session['current_capital']
        }
    
    def settle_paper_trade(self, trade_id: int, actual_outcome: int) -> Dict[str, Any]:
        """Settle a paper trade"""
        if not self.testing_session or self.testing_session['status'] != 'active':
            return {'error': 'No active testing session'}
        
        # Find the trade
        trade = None
        for t in self.testing_session['trades']:
            if t['trade_id'] == trade_id:
                trade = t
                break
        
        if not trade:
            return {'error': 'Trade not found'}
        
        if trade['status'] != 'placed':
            return {'error': 'Trade already settled'}
        
        self.logger.info(f"Settling paper trade: ID={trade_id}")
        
        # Determine if trade won
        won = trade['prediction'] == actual_outcome
        
        # Calculate profit/loss
        if won:
            gross_profit = trade['position_size'] * (trade['odds'] - 1)
            net_profit = gross_profit - trade['commission'] - trade['slippage'] - trade['market_impact']
        else:
            net_profit = -trade['position_size']
        
        # Update trade
        trade['status'] = 'settled'
        trade['actual_outcome'] = actual_outcome
        trade['net_profit'] = net_profit
        trade['outcome'] = 'win' if won else 'loss'
        trade['settlement_time'] = datetime.now()
        
        # Update capital
        self.testing_session['current_capital'] += trade['position_size'] + net_profit
        
        # Update peak capital
        if self.testing_session['current_capital'] > self.testing_session['peak_capital']:
            self.testing_session['peak_capital'] = self.testing_session['current_capital']
        
        # Generate alert
        self._generate_trade_alert(trade, 'settled')
        
        self.logger.info(f"Paper trade settled: ID={trade_id}, "
                        f"Outcome={'WIN' if won else 'LOSS'}, "
                        f"Profit=${net_profit:.2f}")
        
        return {
            'trade_settled': True,
            'trade': trade,
            'current_capital': self.testing_session['current_capital']
        }
    
    def _validate_trade_data(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate trade data"""
        validation = {'valid': True, 'errors': []}
        
        required_fields = ['match_id', 'prediction', 'confidence', 'odds', 'position_size']
        for field in required_fields:
            if field not in trade_data:
                validation['valid'] = False
                validation['errors'].append(f"Missing required field: {field}")
        
        # Check confidence threshold
        if 'confidence' in trade_data:
            if trade_data['confidence'] < 0.6:
                validation['valid'] = False
                validation['errors'].append("Confidence below threshold")
        
        # Check odds range
        if 'odds' in trade_data:
            if trade_data['odds'] < 1.5 or trade_data['odds'] > 10.0:
                validation['valid'] = False
                validation['errors'].append("Odds out of range")
        
        # Check position size
        if 'position_size' in trade_data:
            max_position = self.testing_session['current_capital'] * self.config['risk_management']['position_size_limit']
            if trade_data['position_size'] > max_position:
                validation['valid'] = False
                validation['errors'].append("Position size exceeds limit")
        
        return validation
    
    def _calculate_execution_price(self, trade_data: Dict[str, Any]) -> float:
        """Calculate execution price with realistic execution"""
        base_price = trade_data['odds']
        
        if self.config['paper_trading']['realistic_execution']:
            # Simulate price movement
            price_change = np.random.normal(0, 0.01)  # 1% standard deviation
            execution_price = base_price * (1 + price_change)
        else:
            execution_price = base_price
        
        return execution_price
    
    def _calculate_commission(self, trade_data: Dict[str, Any]) -> float:
        """Calculate commission"""
        return trade_data['position_size'] * self.config['paper_trading']['commission']
    
    def _calculate_slippage(self, trade_data: Dict[str, Any]) -> float:
        """Calculate slippage"""
        return trade_data['position_size'] * self.config['paper_trading']['slippage']
    
    def _calculate_market_impact(self, trade_data: Dict[str, Any]) -> float:
        """Calculate market impact"""
        if self.config['paper_trading']['market_impact']:
            impact_factor = min(0.05, trade_data['position_size'] / self.testing_session['current_capital'] * 0.5)
            return trade_data['position_size'] * impact_factor
        else:
            return 0
    
    def _generate_trade_alert(self, trade: Dict[str, Any], action: str):
        """Generate trade alert"""
        if not self.config['alerts']['enabled']:
            return
        
        alert = {
            'timestamp': datetime.now(),
            'type': 'trade',
            'level': 'info',
            'action': action,
            'trade_id': trade['trade_id'],
            'message': f"Trade {action}: ID={trade['trade_id']}, Size=${trade['position_size']:.2f}"
        }
        
        self.alerts.append(alert)
        self.testing_session['alerts'].append(alert)
        
        self.logger.info(f"Trade alert: {alert['message']}")
    
    def calculate_session_performance(self) -> Dict[str, Any]:
        """Calculate session performance metrics"""
        if not self.testing_session:
            return {}
        
        trades = self.testing_session['trades']
        settled_trades = [t for t in trades if t['status'] == 'settled']
        
        if not settled_trades:
            return {'no_settled_trades': True}
        
        # Basic metrics
        total_trades = len(settled_trades)
        winning_trades = [t for t in settled_trades if t['outcome'] == 'win']
        losing_trades = [t for t in settled_trades if t['outcome'] == 'loss']
        
        win_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Return metrics
        total_profit = sum([t['net_profit'] for t in settled_trades])
        initial_capital = self.testing_session['initial_capital']
        current_capital = self.testing_session['current_capital']
        total_return = (current_capital - initial_capital) / initial_capital
        
        # Drawdown metrics
        peak_capital = self.testing_session['peak_capital']
        current_drawdown = (peak_capital - current_capital) / peak_capital if peak_capital > 0 else 0
        
        # Risk metrics
        returns = [t['net_profit'] / t['position_size'] for t in settled_trades]
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(252)  # Annualized
            sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            volatility = 0
            sharpe_ratio = 0
        
        # Position sizing metrics
        position_sizes = [t['position_size'] for t in settled_trades]
        avg_position_size = np.mean(position_sizes) if position_sizes else 0
        
        # Time metrics
        session_duration = (datetime.now() - self.testing_session['start_time']).total_seconds() / 3600  # hours
        trades_per_hour = total_trades / session_duration if session_duration > 0 else 0
        
        performance_metrics = {
            'total_trades': total_trades,
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_return': total_return,
            'current_capital': current_capital,
            'peak_capital': peak_capital,
            'current_drawdown': current_drawdown,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'avg_position_size': avg_position_size,
            'trades_per_hour': trades_per_hour,
            'session_duration_hours': session_duration
        }
        
        return performance_metrics
    
    def monitor_performance(self) -> Dict[str, Any]:
        """Monitor real-time performance"""
        if not self.testing_session or self.testing_session['status'] != 'active':
            return {'error': 'No active testing session'}
        
        # Calculate current performance
        current_metrics = self.calculate_session_performance()
        
        # Check for alerts
        alerts = self._check_performance_alerts(current_metrics)
        
        # Update session
        self.testing_session['performance_metrics'] = current_metrics
        self.testing_session['last_update'] = datetime.now()
        
        # Check for emergency stop
        emergency_stop = self._check_emergency_stop(current_metrics)
        
        monitoring_result = {
            'timestamp': datetime.now(),
            'performance_metrics': current_metrics,
            'alerts': alerts,
            'emergency_stop': emergency_stop,
            'session_status': self.testing_session['status']
        }
        
        return monitoring_result
    
    def _check_performance_alerts(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Check for performance alerts"""
        alerts = []
        thresholds = self.config['alerts']['thresholds']
        
        # Drawdown alert
        if 'current_drawdown' in metrics:
            if metrics['current_drawdown'] > thresholds['drawdown']:
                alerts.append({
                    'timestamp': datetime.now(),
                    'type': 'drawdown',
                    'level': 'warning',
                    'message': f"High drawdown: {metrics['current_drawdown']:.2%}"
                })
        
        # Loss rate alert
        if 'win_rate' in metrics:
            loss_rate = 1 - metrics['win_rate']
            if loss_rate > thresholds['loss_rate']:
                alerts.append({
                    'timestamp': datetime.now(),
                    'type': 'loss_rate',
                    'level': 'critical',
                    'message': f"High loss rate: {loss_rate:.2%}"
                })
        
        # Volatility alert
        if 'volatility' in metrics:
            if metrics['volatility'] > thresholds['volatility']:
                alerts.append({
                    'timestamp': datetime.now(),
                    'type': 'volatility',
                    'level': 'warning',
                    'message': f"High volatility: {metrics['volatility']:.2%}"
                })
        
        # Performance alert
        if 'total_return' in metrics:
            if metrics['total_return'] < thresholds['performance']:
                alerts.append({
                    'timestamp': datetime.now(),
                    'type': 'performance',
                    'level': 'warning',
                    'message': f"Poor performance: {metrics['total_return']:.2%}"
                })
        
        # Add alerts to session
        for alert in alerts:
            self.testing_session['alerts'].append(alert)
            self.alerts.append(alert)
        
        return alerts
    
    def _check_emergency_stop(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Check for emergency stop conditions"""
        emergency_stop = {
            'triggered': False,
            'reason': None,
            'action': None
        }
        
        if not self.config['risk_management']['auto_stop']:
            return emergency_stop
        
        # Check max drawdown
        if 'current_drawdown' in metrics:
            if metrics['current_drawdown'] > self.config['risk_management']['max_drawdown']:
                emergency_stop['triggered'] = True
                emergency_stop['reason'] = f"Max drawdown exceeded: {metrics['current_drawdown']:.2%}"
                emergency_stop['action'] = 'stop_session'
        
        # Check daily loss limit
        if 'total_return' in metrics:
            if metrics['total_return'] < -self.config['risk_management']['daily_loss_limit']:
                emergency_stop['triggered'] = True
                emergency_stop['reason'] = f"Daily loss limit exceeded: {metrics['total_return']:.2%}"
                emergency_stop['action'] = 'stop_session'
        
        # Check emergency stop
        if 'current_drawdown' in metrics:
            if metrics['current_drawdown'] > self.config['risk_management']['emergency_stop']:
                emergency_stop['triggered'] = True
                emergency_stop['reason'] = f"Emergency stop triggered: {metrics['current_drawdown']:.2%}"
                emergency_stop['action'] = 'emergency_stop'
        
        if emergency_stop['triggered']:
            self.testing_session['status'] = 'emergency_stop'
            self.logger.critical(f"Emergency stop triggered: {emergency_stop['reason']}")
        
        return emergency_stop
    
    def generate_performance_report(self, report_type: str = 'summary') -> str:
        """Generate performance report"""
        if not self.testing_session:
            return "No active testing session"
        
        report = []
        report.append("=" * 80)
        report.append("LIVE TESTING PERFORMANCE REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Session information
        report.append("SESSION INFORMATION:")
        report.append(f"  Session Name: {self.testing_session['session_name']}")
        report.append(f"  Start Time: {self.testing_session['start_time']}")
        report.append(f"  Status: {self.testing_session['status']}")
        if 'end_time' in self.testing_session:
            report.append(f"  End Time: {self.testing_session['end_time']}")
            report.append(f"  Duration: {self.testing_session['duration']:.2f} seconds")
        report.append("")
        
        # Performance metrics
        if 'performance_metrics' in self.testing_session:
            metrics = self.testing_session['performance_metrics']
            
            report.append("PERFORMANCE METRICS:")
            report.append(f"  Total Trades: {metrics.get('total_trades', 0)}")
            report.append(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            report.append(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            report.append(f"  Current Capital: ${metrics.get('current_capital', 0):.2f}")
            report.append(f"  Peak Capital: ${metrics.get('peak_capital', 0):.2f}")
            report.append(f"  Current Drawdown: {metrics.get('current_drawdown', 0):.2%}")
            report.append(f"  Volatility: {metrics.get('volatility', 0):.2%}")
            report.append(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            report.append("")
        
        # Recent trades
        if self.testing_session['trades']:
            report.append("RECENT TRADES:")
            recent_trades = self.testing_session['trades'][-5:]  # Last 5 trades
            for trade in recent_trades:
                report.append(f"  Trade {trade['trade_id']}: {trade['status']}, "
                            f"Size=${trade['position_size']:.2f}, "
                            f"Profit=${trade['net_profit']:.2f}")
            report.append("")
        
        # Alerts
        if self.testing_session['alerts']:
            report.append("RECENT ALERTS:")
            recent_alerts = self.testing_session['alerts'][-5:]  # Last 5 alerts
            for alert in recent_alerts:
                report.append(f"  {alert['timestamp']}: {alert['level'].upper()} - {alert['message']}")
            report.append("")
        
        # Recommendations
        report.append("RECOMMENDATIONS:")
        if self.testing_session['status'] == 'active':
            report.append("  ✅ Session is active - continue monitoring")
            report.append("  📊 Monitor performance metrics closely")
            report.append("  🔍 Watch for risk management alerts")
        elif self.testing_session['status'] == 'stopped':
            report.append("  ✅ Session completed successfully")
            report.append("  📈 Review performance metrics")
            report.append("  🔄 Consider starting new session")
        elif self.testing_session['status'] == 'emergency_stop':
            report.append("  ❌ Emergency stop triggered")
            report.append("  🔧 Review strategy and risk management")
            report.append("  📊 Analyze what went wrong")
        report.append("")
        
        report.append("=" * 80)
        
        return "\n".join(report)
    
    def save_testing_session(self, filepath: str):
        """Save testing session data"""
        self.logger.info(f"Saving testing session to {filepath}")
        
        import joblib
        
        session_data = {
            'testing_session': self.testing_session,
            'paper_trades': self.paper_trades,
            'alerts': self.alerts,
            'config': self.config
        }
        
        joblib.dump(session_data, filepath)
        self.logger.info("Testing session saved successfully")

# Example usage
def main():
    """Example usage of NonMajorLeagueLiveTesting"""
    
    # Initialize live testing
    live_tester = NonMajorLeagueLiveTesting()
    
    # Start testing session
    session = live_tester.start_testing_session("demo_session")
    
    # Simulate some trades
    test_trades = [
        {
            'match_id': 'match_1',
            'prediction': 2,  # Home win
            'confidence': 0.75,
            'odds': 2.5,
            'position_size': 50,
            'kelly_fraction': 0.02
        },
        {
            'match_id': 'match_2',
            'prediction': 0,  # Away win
            'confidence': 0.65,
            'odds': 3.0,
            'position_size': 30,
            'kelly_fraction': 0.015
        }
    ]
    
    # Place trades
    for i, trade_data in enumerate(test_trades):
        result = live_tester.place_paper_trade(trade_data)
        if result.get('trade_placed'):
            trade = result['trade']
            print(f"Trade {trade['trade_id']} placed: ${trade['position_size']:.2f}")
            
            # Simulate outcome
            actual_outcome = np.random.choice([0, 1, 2])
            
            # Settle trade
            settlement = live_tester.settle_paper_trade(trade['trade_id'], actual_outcome)
            if settlement.get('trade_settled'):
                settled_trade = settlement['trade']
                print(f"Trade {settled_trade['trade_id']} settled: "
                      f"{settled_trade['outcome']}, Profit: ${settled_trade['net_profit']:.2f}")
    
    # Monitor performance
    monitoring = live_tester.monitor_performance()
    print(f"Current capital: ${monitoring['performance_metrics']['current_capital']:.2f}")
    
    # Generate report
    report = live_tester.generate_performance_report()
    print(report)
    
    # Stop session
    final_session = live_tester.stop_testing_session()
    print(f"Session stopped: {final_session['session_name']}")
    
    # Save session
    live_tester.save_testing_session('testing_session.pkl')

if __name__ == "__main__":
    main()
