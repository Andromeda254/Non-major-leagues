import pandas as pd
import numpy as np
import time
import threading
import schedule
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeaguePerformanceTracking:
    """
    Real-time performance tracking and reporting system for non-major soccer leagues
    
    Key Features:
    - Real-time performance monitoring
    - Automated reporting
    - Performance analytics
    - Trend analysis
    - Alert generation
    - Dashboard integration
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize performance tracking system
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.tracking_active = False
        self.tracking_thread = None
        self.performance_data = []
        self.reporting_history = []
        self.alert_history = []
        
    def setup_logging(self):
        """Setup logging for performance tracking"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load performance tracking configuration"""
        if config is None:
            self.config = {
                'tracking': {
                    'enabled': True,
                    'interval': 300,  # 5 minutes
                    'retention_days': 90,
                    'real_time': True
                },
                'metrics': {
                    'prediction_accuracy': {
                        'enabled': True,
                        'threshold': 0.45,
                        'window': 24,  # hours
                        'alert_threshold': 0.4
                    },
                    'betting_performance': {
                        'enabled': True,
                        'win_rate_threshold': 0.4,
                        'roi_threshold': 0.05,
                        'drawdown_threshold': 0.2,
                        'window': 24  # hours
                    },
                    'model_performance': {
                        'enabled': True,
                        'response_time_threshold': 2.0,
                        'throughput_threshold': 100,
                        'error_rate_threshold': 0.05,
                        'window': 1  # hour
                    },
                    'system_performance': {
                        'enabled': True,
                        'cpu_threshold': 0.8,
                        'memory_threshold': 0.8,
                        'disk_threshold': 0.9,
                        'window': 1  # hour
                    }
                },
                'reporting': {
                    'enabled': True,
                    'schedule': {
                        'real_time': True,
                        'hourly': True,
                        'daily': True,
                        'weekly': True,
                        'monthly': True
                    },
                    'formats': ['json', 'csv', 'html'],
                    'destinations': {
                        'file': {
                            'enabled': True,
                            'path': './reports',
                            'format': 'html'
                        },
                        'email': {
                            'enabled': False,
                            'recipients': ['admin@example.com'],
                            'format': 'html'
                        },
                        'webhook': {
                            'enabled': False,
                            'url': 'https://api.example.com/reports',
                            'format': 'json'
                        }
                    }
                },
                'analytics': {
                    'enabled': True,
                    'trend_analysis': True,
                    'anomaly_detection': True,
                    'correlation_analysis': True,
                    'forecasting': True
                },
                'alerting': {
                    'enabled': True,
                    'channels': {
                        'email': {
                            'enabled': True,
                            'recipients': ['admin@example.com']
                        },
                        'slack': {
                            'enabled': True,
                            'webhook_url': 'https://hooks.slack.com/services/...'
                        },
                        'webhook': {
                            'enabled': True,
                            'url': 'https://api.example.com/alerts'
                        }
                    },
                    'rules': {
                        'critical': {
                            'conditions': ['prediction_accuracy < 0.35', 'roi < -0.1', 'drawdown > 0.25'],
                            'cooldown': 300,  # 5 minutes
                            'escalation': True
                        },
                        'warning': {
                            'conditions': ['prediction_accuracy < 0.4', 'roi < 0', 'drawdown > 0.15'],
                            'cooldown': 600,  # 10 minutes
                            'escalation': False
                        },
                        'info': {
                            'conditions': ['prediction_accuracy < 0.45', 'roi < 0.05'],
                            'cooldown': 1800,  # 30 minutes
                            'escalation': False
                        }
                    }
                },
                'dashboard': {
                    'enabled': True,
                    'port': 8080,
                    'refresh_interval': 30,
                    'widgets': [
                        'performance_overview',
                        'prediction_accuracy',
                        'betting_performance',
                        'model_performance',
                        'system_metrics',
                        'alerts',
                        'trends'
                    ]
                }
            }
        else:
            self.config = config
    
    def start_tracking(self):
        """Start performance tracking"""
        if self.tracking_active:
            self.logger.warning("Performance tracking already active")
            return
        
        self.logger.info("Starting performance tracking")
        self.tracking_active = True
        
        # Start tracking thread
        self.tracking_thread = threading.Thread(target=self._tracking_loop)
        self.tracking_thread.daemon = True
        self.tracking_thread.start()
        
        # Schedule tracking tasks
        schedule.every(self.config['tracking']['interval']).seconds.do(self._collect_performance_metrics)
        schedule.every(1).hour.do(self._generate_hourly_report)
        schedule.every(1).day.do(self._generate_daily_report)
        schedule.every(1).week.do(self._generate_weekly_report)
        schedule.every(1).month.do(self._generate_monthly_report)
        
        self.logger.info("Performance tracking started successfully")
    
    def stop_tracking(self):
        """Stop performance tracking"""
        self.logger.info("Stopping performance tracking")
        self.tracking_active = False
        
        if self.tracking_thread:
            self.tracking_thread.join(timeout=30)
        
        schedule.clear()
        self.logger.info("Performance tracking stopped")
    
    def _tracking_loop(self):
        """Main tracking loop"""
        while self.tracking_active:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in tracking loop: {e}")
                time.sleep(5)
    
    def _collect_performance_metrics(self):
        """Collect performance metrics"""
        try:
            self.logger.info("Collecting performance metrics")
            
            timestamp = datetime.now()
            metrics = {
                'timestamp': timestamp,
                'prediction_accuracy': self._calculate_prediction_accuracy(),
                'betting_performance': self._calculate_betting_performance(),
                'model_performance': self._calculate_model_performance(),
                'system_performance': self._calculate_system_performance()
            }
            
            # Store metrics
            self.performance_data.append(metrics)
            
            # Check for alerts
            self._check_performance_alerts(metrics)
            
            # Generate real-time report if enabled
            if self.config['reporting']['schedule']['real_time']:
                self._generate_real_time_report(metrics)
            
            self.logger.info("Performance metrics collected successfully")
            
        except Exception as e:
            self.logger.error(f"Error collecting performance metrics: {e}")
    
    def _calculate_prediction_accuracy(self) -> Dict[str, Any]:
        """Calculate prediction accuracy metrics"""
        try:
            # This would typically fetch from a database or API
            # For now, we'll simulate the data
            
            # Simulate prediction accuracy data
            accuracy_data = {
                'overall_accuracy': np.random.uniform(0.4, 0.6),
                'home_win_accuracy': np.random.uniform(0.35, 0.55),
                'draw_accuracy': np.random.uniform(0.2, 0.4),
                'away_win_accuracy': np.random.uniform(0.35, 0.55),
                'confidence_calibration': np.random.uniform(0.6, 0.9),
                'recent_accuracy': np.random.uniform(0.4, 0.6),
                'trend': np.random.choice(['improving', 'stable', 'declining']),
                'sample_size': np.random.randint(100, 1000)
            }
            
            return accuracy_data
            
        except Exception as e:
            self.logger.error(f"Error calculating prediction accuracy: {e}")
            return {}
    
    def _calculate_betting_performance(self) -> Dict[str, Any]:
        """Calculate betting performance metrics"""
        try:
            # Simulate betting performance data
            betting_data = {
                'total_bets': np.random.randint(50, 500),
                'winning_bets': np.random.randint(20, 250),
                'losing_bets': np.random.randint(20, 250),
                'win_rate': np.random.uniform(0.35, 0.55),
                'total_profit': np.random.uniform(-1000, 2000),
                'roi': np.random.uniform(-0.1, 0.2),
                'avg_odds': np.random.uniform(2.0, 4.0),
                'kelly_efficiency': np.random.uniform(0.6, 0.9),
                'max_drawdown': np.random.uniform(0.05, 0.25),
                'current_drawdown': np.random.uniform(0.0, 0.15),
                'sharpe_ratio': np.random.uniform(0.2, 1.0),
                'profit_factor': np.random.uniform(0.8, 1.5)
            }
            
            return betting_data
            
        except Exception as e:
            self.logger.error(f"Error calculating betting performance: {e}")
            return {}
    
    def _calculate_model_performance(self) -> Dict[str, Any]:
        """Calculate model performance metrics"""
        try:
            # Simulate model performance data
            model_data = {
                'avg_response_time': np.random.uniform(0.5, 3.0),
                'max_response_time': np.random.uniform(1.0, 5.0),
                'throughput': np.random.uniform(50, 200),
                'error_rate': np.random.uniform(0.01, 0.1),
                'uptime': np.random.uniform(0.95, 0.99),
                'memory_usage': np.random.uniform(0.3, 0.8),
                'cpu_usage': np.random.uniform(0.2, 0.7),
                'model_accuracy': np.random.uniform(0.4, 0.6),
                'prediction_confidence': np.random.uniform(0.6, 0.9)
            }
            
            return model_data
            
        except Exception as e:
            self.logger.error(f"Error calculating model performance: {e}")
            return {}
    
    def _calculate_system_performance(self) -> Dict[str, Any]:
        """Calculate system performance metrics"""
        try:
            # Simulate system performance data
            system_data = {
                'cpu_usage': np.random.uniform(0.2, 0.8),
                'memory_usage': np.random.uniform(0.3, 0.8),
                'disk_usage': np.random.uniform(0.4, 0.9),
                'network_latency': np.random.uniform(10, 100),
                'active_connections': np.random.randint(10, 100),
                'queue_length': np.random.randint(0, 50),
                'error_count': np.random.randint(0, 10),
                'warning_count': np.random.randint(0, 20)
            }
            
            return system_data
            
        except Exception as e:
            self.logger.error(f"Error calculating system performance: {e}")
            return {}
    
    def _check_performance_alerts(self, metrics: Dict[str, Any]):
        """Check for performance alerts"""
        try:
            for alert_level, alert_config in self.config['alerting']['rules'].items():
                for condition in alert_config['conditions']:
                    if self._evaluate_alert_condition(condition, metrics):
                        self._trigger_performance_alert(alert_level, condition, metrics)
                        
        except Exception as e:
            self.logger.error(f"Error checking performance alerts: {e}")
    
    def _evaluate_alert_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate alert condition"""
        try:
            # Parse condition (simplified)
            if '<' in condition:
                metric_name, threshold_str = condition.split(' < ')
                metric_name = metric_name.strip()
                threshold = float(threshold_str.strip())
                
                # Find metric value
                metric_value = self._get_metric_value(metric_name, metrics)
                if metric_value is not None:
                    return metric_value < threshold
            
            elif '>' in condition:
                metric_name, threshold_str = condition.split(' > ')
                metric_name = metric_name.strip()
                threshold = float(threshold_str.strip())
                
                # Find metric value
                metric_value = self._get_metric_value(metric_name, metrics)
                if metric_value is not None:
                    return metric_value > threshold
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating alert condition '{condition}': {e}")
            return False
    
    def _get_metric_value(self, metric_name: str, metrics: Dict[str, Any]) -> Optional[float]:
        """Get metric value from metrics data"""
        try:
            # Check prediction accuracy metrics
            if metric_name in metrics.get('prediction_accuracy', {}):
                return metrics['prediction_accuracy'][metric_name]
            
            # Check betting performance metrics
            if metric_name in metrics.get('betting_performance', {}):
                return metrics['betting_performance'][metric_name]
            
            # Check model performance metrics
            if metric_name in metrics.get('model_performance', {}):
                return metrics['model_performance'][metric_name]
            
            # Check system performance metrics
            if metric_name in metrics.get('system_performance', {}):
                return metrics['system_performance'][metric_name]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting metric value for '{metric_name}': {e}")
            return None
    
    def _trigger_performance_alert(self, level: str, condition: str, metrics: Dict[str, Any]):
        """Trigger performance alert"""
        try:
            alert = {
                'id': f"perf_alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'level': level,
                'condition': condition,
                'timestamp': datetime.now(),
                'metrics': metrics,
                'status': 'active'
            }
            
            self.alert_history.append(alert)
            
            # Send alert through configured channels
            self._send_performance_alert(alert)
            
            self.logger.warning(f"Performance alert triggered: {level} - {condition}")
            
        except Exception as e:
            self.logger.error(f"Error triggering performance alert: {e}")
    
    def _send_performance_alert(self, alert: Dict[str, Any]):
        """Send performance alert through configured channels"""
        try:
            alert_message = self._format_performance_alert_message(alert)
            
            # Email alert
            if self.config['alerting']['channels']['email']['enabled']:
                self._send_email_alert(alert_message)
            
            # Slack alert
            if self.config['alerting']['channels']['slack']['enabled']:
                self._send_slack_alert(alert_message)
            
            # Webhook alert
            if self.config['alerting']['channels']['webhook']['enabled']:
                self._send_webhook_alert(alert_message)
            
        except Exception as e:
            self.logger.error(f"Error sending performance alert: {e}")
    
    def _format_performance_alert_message(self, alert: Dict[str, Any]) -> str:
        """Format performance alert message"""
        return f"""
PERFORMANCE ALERT: {alert['level'].upper()}
Time: {alert['timestamp']}
Condition: {alert['condition']}
Status: {alert['status']}

Metrics:
{self._format_metrics_for_alert(alert['metrics'])}
"""
    
    def _format_metrics_for_alert(self, metrics: Dict[str, Any]) -> str:
        """Format metrics for alert message"""
        formatted_metrics = []
        
        for category, category_metrics in metrics.items():
            formatted_metrics.append(f"{category.upper()}:")
            for metric, value in category_metrics.items():
                if isinstance(value, float):
                    formatted_metrics.append(f"  {metric}: {value:.3f}")
                else:
                    formatted_metrics.append(f"  {metric}: {value}")
        
        return "\n".join(formatted_metrics)
    
    def _send_email_alert(self, message: str):
        """Send email alert"""
        try:
            # Simplified email sending (would use smtplib in production)
            self.logger.info(f"Email performance alert sent: {message[:100]}...")
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
    
    def _send_slack_alert(self, message: str):
        """Send Slack alert"""
        try:
            # Simplified Slack sending (would use requests in production)
            self.logger.info(f"Slack performance alert sent: {message[:100]}...")
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {e}")
    
    def _send_webhook_alert(self, message: str):
        """Send webhook alert"""
        try:
            # Simplified webhook sending (would use requests in production)
            self.logger.info(f"Webhook performance alert sent: {message[:100]}...")
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {e}")
    
    def _generate_real_time_report(self, metrics: Dict[str, Any]):
        """Generate real-time performance report"""
        try:
            report = {
                'type': 'real_time',
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'summary': self._generate_performance_summary(metrics)
            }
            
            # Store report
            self.reporting_history.append(report)
            
            # Send to configured destinations
            self._send_report(report)
            
        except Exception as e:
            self.logger.error(f"Error generating real-time report: {e}")
    
    def _generate_hourly_report(self):
        """Generate hourly performance report"""
        try:
            self.logger.info("Generating hourly performance report")
            
            # Get metrics from last hour
            cutoff_time = datetime.now() - timedelta(hours=1)
            hourly_metrics = [
                m for m in self.performance_data 
                if m['timestamp'] > cutoff_time
            ]
            
            if not hourly_metrics:
                self.logger.warning("No metrics available for hourly report")
                return
            
            # Calculate hourly summary
            hourly_summary = self._calculate_hourly_summary(hourly_metrics)
            
            report = {
                'type': 'hourly',
                'timestamp': datetime.now().isoformat(),
                'period': '1 hour',
                'metrics_count': len(hourly_metrics),
                'summary': hourly_summary,
                'trends': self._analyze_trends(hourly_metrics)
            }
            
            # Store report
            self.reporting_history.append(report)
            
            # Send to configured destinations
            self._send_report(report)
            
            self.logger.info("Hourly performance report generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating hourly report: {e}")
    
    def _generate_daily_report(self):
        """Generate daily performance report"""
        try:
            self.logger.info("Generating daily performance report")
            
            # Get metrics from last 24 hours
            cutoff_time = datetime.now() - timedelta(hours=24)
            daily_metrics = [
                m for m in self.performance_data 
                if m['timestamp'] > cutoff_time
            ]
            
            if not daily_metrics:
                self.logger.warning("No metrics available for daily report")
                return
            
            # Calculate daily summary
            daily_summary = self._calculate_daily_summary(daily_metrics)
            
            report = {
                'type': 'daily',
                'timestamp': datetime.now().isoformat(),
                'period': '24 hours',
                'metrics_count': len(daily_metrics),
                'summary': daily_summary,
                'trends': self._analyze_trends(daily_metrics),
                'alerts': self._get_period_alerts(cutoff_time)
            }
            
            # Store report
            self.reporting_history.append(report)
            
            # Send to configured destinations
            self._send_report(report)
            
            self.logger.info("Daily performance report generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating daily report: {e}")
    
    def _generate_weekly_report(self):
        """Generate weekly performance report"""
        try:
            self.logger.info("Generating weekly performance report")
            
            # Get metrics from last 7 days
            cutoff_time = datetime.now() - timedelta(days=7)
            weekly_metrics = [
                m for m in self.performance_data 
                if m['timestamp'] > cutoff_time
            ]
            
            if not weekly_metrics:
                self.logger.warning("No metrics available for weekly report")
                return
            
            # Calculate weekly summary
            weekly_summary = self._calculate_weekly_summary(weekly_metrics)
            
            report = {
                'type': 'weekly',
                'timestamp': datetime.now().isoformat(),
                'period': '7 days',
                'metrics_count': len(weekly_metrics),
                'summary': weekly_summary,
                'trends': self._analyze_trends(weekly_metrics),
                'alerts': self._get_period_alerts(cutoff_time),
                'recommendations': self._generate_recommendations(weekly_summary)
            }
            
            # Store report
            self.reporting_history.append(report)
            
            # Send to configured destinations
            self._send_report(report)
            
            self.logger.info("Weekly performance report generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating weekly report: {e}")
    
    def _generate_monthly_report(self):
        """Generate monthly performance report"""
        try:
            self.logger.info("Generating monthly performance report")
            
            # Get metrics from last 30 days
            cutoff_time = datetime.now() - timedelta(days=30)
            monthly_metrics = [
                m for m in self.performance_data 
                if m['timestamp'] > cutoff_time
            ]
            
            if not monthly_metrics:
                self.logger.warning("No metrics available for monthly report")
                return
            
            # Calculate monthly summary
            monthly_summary = self._calculate_monthly_summary(monthly_metrics)
            
            report = {
                'type': 'monthly',
                'timestamp': datetime.now().isoformat(),
                'period': '30 days',
                'metrics_count': len(monthly_metrics),
                'summary': monthly_summary,
                'trends': self._analyze_trends(monthly_metrics),
                'alerts': self._get_period_alerts(cutoff_time),
                'recommendations': self._generate_recommendations(monthly_summary),
                'forecast': self._generate_forecast(monthly_metrics)
            }
            
            # Store report
            self.reporting_history.append(report)
            
            # Send to configured destinations
            self._send_report(report)
            
            self.logger.info("Monthly performance report generated successfully")
            
        except Exception as e:
            self.logger.error(f"Error generating monthly report: {e}")
    
    def _calculate_hourly_summary(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate hourly summary"""
        try:
            if not metrics:
                return {}
            
            summary = {}
            
            # Prediction accuracy summary
            accuracy_values = [m['prediction_accuracy']['overall_accuracy'] for m in metrics if 'prediction_accuracy' in m]
            if accuracy_values:
                summary['prediction_accuracy'] = {
                    'avg': np.mean(accuracy_values),
                    'min': np.min(accuracy_values),
                    'max': np.max(accuracy_values),
                    'trend': 'stable'  # Simplified
                }
            
            # Betting performance summary
            roi_values = [m['betting_performance']['roi'] for m in metrics if 'betting_performance' in m]
            if roi_values:
                summary['betting_performance'] = {
                    'avg_roi': np.mean(roi_values),
                    'total_profit': sum([m['betting_performance']['total_profit'] for m in metrics if 'betting_performance' in m]),
                    'win_rate': np.mean([m['betting_performance']['win_rate'] for m in metrics if 'betting_performance' in m])
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error calculating hourly summary: {e}")
            return {}
    
    def _calculate_daily_summary(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate daily summary"""
        try:
            if not metrics:
                return {}
            
            summary = {}
            
            # Prediction accuracy summary
            accuracy_values = [m['prediction_accuracy']['overall_accuracy'] for m in metrics if 'prediction_accuracy' in m]
            if accuracy_values:
                summary['prediction_accuracy'] = {
                    'avg': np.mean(accuracy_values),
                    'min': np.min(accuracy_values),
                    'max': np.max(accuracy_values),
                    'std': np.std(accuracy_values),
                    'trend': 'stable'  # Simplified
                }
            
            # Betting performance summary
            roi_values = [m['betting_performance']['roi'] for m in metrics if 'betting_performance' in m]
            if roi_values:
                summary['betting_performance'] = {
                    'avg_roi': np.mean(roi_values),
                    'total_profit': sum([m['betting_performance']['total_profit'] for m in metrics if 'betting_performance' in m]),
                    'win_rate': np.mean([m['betting_performance']['win_rate'] for m in metrics if 'betting_performance' in m]),
                    'max_drawdown': max([m['betting_performance']['max_drawdown'] for m in metrics if 'betting_performance' in m])
                }
            
            # Model performance summary
            response_times = [m['model_performance']['avg_response_time'] for m in metrics if 'model_performance' in m]
            if response_times:
                summary['model_performance'] = {
                    'avg_response_time': np.mean(response_times),
                    'max_response_time': np.max(response_times),
                    'error_rate': np.mean([m['model_performance']['error_rate'] for m in metrics if 'model_performance' in m])
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error calculating daily summary: {e}")
            return {}
    
    def _calculate_weekly_summary(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate weekly summary"""
        try:
            if not metrics:
                return {}
            
            summary = {}
            
            # Prediction accuracy summary
            accuracy_values = [m['prediction_accuracy']['overall_accuracy'] for m in metrics if 'prediction_accuracy' in m]
            if accuracy_values:
                summary['prediction_accuracy'] = {
                    'avg': np.mean(accuracy_values),
                    'min': np.min(accuracy_values),
                    'max': np.max(accuracy_values),
                    'std': np.std(accuracy_values),
                    'trend': 'stable'  # Simplified
                }
            
            # Betting performance summary
            roi_values = [m['betting_performance']['roi'] for m in metrics if 'betting_performance' in m]
            if roi_values:
                summary['betting_performance'] = {
                    'avg_roi': np.mean(roi_values),
                    'total_profit': sum([m['betting_performance']['total_profit'] for m in metrics if 'betting_performance' in m]),
                    'win_rate': np.mean([m['betting_performance']['win_rate'] for m in metrics if 'betting_performance' in m]),
                    'max_drawdown': max([m['betting_performance']['max_drawdown'] for m in metrics if 'betting_performance' in m]),
                    'sharpe_ratio': np.mean([m['betting_performance']['sharpe_ratio'] for m in metrics if 'betting_performance' in m])
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error calculating weekly summary: {e}")
            return {}
    
    def _calculate_monthly_summary(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate monthly summary"""
        try:
            if not metrics:
                return {}
            
            summary = {}
            
            # Prediction accuracy summary
            accuracy_values = [m['prediction_accuracy']['overall_accuracy'] for m in metrics if 'prediction_accuracy' in m]
            if accuracy_values:
                summary['prediction_accuracy'] = {
                    'avg': np.mean(accuracy_values),
                    'min': np.min(accuracy_values),
                    'max': np.max(accuracy_values),
                    'std': np.std(accuracy_values),
                    'trend': 'stable'  # Simplified
                }
            
            # Betting performance summary
            roi_values = [m['betting_performance']['roi'] for m in metrics if 'betting_performance' in m]
            if roi_values:
                summary['betting_performance'] = {
                    'avg_roi': np.mean(roi_values),
                    'total_profit': sum([m['betting_performance']['total_profit'] for m in metrics if 'betting_performance' in m]),
                    'win_rate': np.mean([m['betting_performance']['win_rate'] for m in metrics if 'betting_performance' in m]),
                    'max_drawdown': max([m['betting_performance']['max_drawdown'] for m in metrics if 'betting_performance' in m]),
                    'sharpe_ratio': np.mean([m['betting_performance']['sharpe_ratio'] for m in metrics if 'betting_performance' in m]),
                    'profit_factor': np.mean([m['betting_performance']['profit_factor'] for m in metrics if 'betting_performance' in m])
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error calculating monthly summary: {e}")
            return {}
    
    def _analyze_trends(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Analyze performance trends"""
        try:
            if len(metrics) < 2:
                return {'trend': 'insufficient_data'}
            
            trends = {}
            
            # Analyze prediction accuracy trend
            accuracy_values = [m['prediction_accuracy']['overall_accuracy'] for m in metrics if 'prediction_accuracy' in m]
            if len(accuracy_values) >= 2:
                trend_slope = np.polyfit(range(len(accuracy_values)), accuracy_values, 1)[0]
                if trend_slope > 0.01:
                    trends['prediction_accuracy'] = 'improving'
                elif trend_slope < -0.01:
                    trends['prediction_accuracy'] = 'declining'
                else:
                    trends['prediction_accuracy'] = 'stable'
            
            # Analyze ROI trend
            roi_values = [m['betting_performance']['roi'] for m in metrics if 'betting_performance' in m]
            if len(roi_values) >= 2:
                trend_slope = np.polyfit(range(len(roi_values)), roi_values, 1)[0]
                if trend_slope > 0.001:
                    trends['roi'] = 'improving'
                elif trend_slope < -0.001:
                    trends['roi'] = 'declining'
                else:
                    trends['roi'] = 'stable'
            
            return trends
            
        except Exception as e:
            self.logger.error(f"Error analyzing trends: {e}")
            return {}
    
    def _get_period_alerts(self, cutoff_time: datetime) -> List[Dict[str, Any]]:
        """Get alerts for a specific period"""
        try:
            period_alerts = [
                alert for alert in self.alert_history 
                if alert['timestamp'] > cutoff_time
            ]
            
            return period_alerts
            
        except Exception as e:
            self.logger.error(f"Error getting period alerts: {e}")
            return []
    
    def _generate_recommendations(self, summary: Dict[str, Any]) -> List[str]:
        """Generate performance recommendations"""
        try:
            recommendations = []
            
            # Prediction accuracy recommendations
            if 'prediction_accuracy' in summary:
                accuracy = summary['prediction_accuracy']['avg']
                if accuracy < 0.4:
                    recommendations.append("Consider reviewing model features and training data")
                elif accuracy < 0.45:
                    recommendations.append("Monitor model performance closely")
            
            # Betting performance recommendations
            if 'betting_performance' in summary:
                roi = summary['betting_performance']['avg_roi']
                if roi < 0:
                    recommendations.append("Review betting strategy and risk management")
                elif roi < 0.05:
                    recommendations.append("Consider optimizing position sizing")
            
            # Model performance recommendations
            if 'model_performance' in summary:
                response_time = summary['model_performance']['avg_response_time']
                if response_time > 2.0:
                    recommendations.append("Optimize model inference performance")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _generate_forecast(self, metrics: List[Dict]) -> Dict[str, Any]:
        """Generate performance forecast"""
        try:
            if len(metrics) < 10:
                return {'forecast': 'insufficient_data'}
            
            # Simple linear forecast for prediction accuracy
            accuracy_values = [m['prediction_accuracy']['overall_accuracy'] for m in metrics if 'prediction_accuracy' in m]
            if len(accuracy_values) >= 10:
                # Fit linear trend
                x = np.arange(len(accuracy_values))
                coeffs = np.polyfit(x, accuracy_values, 1)
                
                # Forecast next 7 days
                future_x = np.arange(len(accuracy_values), len(accuracy_values) + 7)
                forecast_values = np.polyval(coeffs, future_x)
                
                return {
                    'prediction_accuracy_forecast': {
                        'next_7_days': forecast_values.tolist(),
                        'trend': 'improving' if coeffs[0] > 0 else 'declining' if coeffs[0] < 0 else 'stable'
                    }
                }
            
            return {'forecast': 'insufficient_data'}
            
        except Exception as e:
            self.logger.error(f"Error generating forecast: {e}")
            return {}
    
    def _generate_performance_summary(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary"""
        try:
            summary = {
                'overall_status': 'healthy',
                'key_metrics': {},
                'alerts': len([a for a in self.alert_history if a['status'] == 'active'])
            }
            
            # Extract key metrics
            if 'prediction_accuracy' in metrics:
                summary['key_metrics']['prediction_accuracy'] = metrics['prediction_accuracy']['overall_accuracy']
            
            if 'betting_performance' in metrics:
                summary['key_metrics']['roi'] = metrics['betting_performance']['roi']
                summary['key_metrics']['win_rate'] = metrics['betting_performance']['win_rate']
            
            if 'model_performance' in metrics:
                summary['key_metrics']['response_time'] = metrics['model_performance']['avg_response_time']
                summary['key_metrics']['error_rate'] = metrics['model_performance']['error_rate']
            
            # Determine overall status
            if summary['key_metrics'].get('prediction_accuracy', 0) < 0.4:
                summary['overall_status'] = 'critical'
            elif summary['key_metrics'].get('roi', 0) < 0:
                summary['overall_status'] = 'warning'
            elif summary['key_metrics'].get('response_time', 0) > 2.0:
                summary['overall_status'] = 'warning'
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating performance summary: {e}")
            return {}
    
    def _send_report(self, report: Dict[str, Any]):
        """Send report to configured destinations"""
        try:
            destinations = self.config['reporting']['destinations']
            
            # File destination
            if destinations['file']['enabled']:
                self._save_report_to_file(report)
            
            # Email destination
            if destinations['email']['enabled']:
                self._send_report_by_email(report)
            
            # Webhook destination
            if destinations['webhook']['enabled']:
                self._send_report_by_webhook(report)
            
        except Exception as e:
            self.logger.error(f"Error sending report: {e}")
    
    def _save_report_to_file(self, report: Dict[str, Any]):
        """Save report to file"""
        try:
            import os
            import json
            
            # Create reports directory
            reports_dir = self.config['reporting']['destinations']['file']['path']
            os.makedirs(reports_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"performance_report_{report['type']}_{timestamp}.json"
            filepath = os.path.join(reports_dir, filename)
            
            # Save report
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            self.logger.info(f"Report saved to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error saving report to file: {e}")
    
    def _send_report_by_email(self, report: Dict[str, Any]):
        """Send report by email"""
        try:
            # Simplified email sending (would use smtplib in production)
            self.logger.info(f"Email report sent: {report['type']}")
        except Exception as e:
            self.logger.error(f"Error sending report by email: {e}")
    
    def _send_report_by_webhook(self, report: Dict[str, Any]):
        """Send report by webhook"""
        try:
            # Simplified webhook sending (would use requests in production)
            self.logger.info(f"Webhook report sent: {report['type']}")
        except Exception as e:
            self.logger.error(f"Error sending report by webhook: {e}")
    
    def get_performance_data(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get performance data for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            return [
                m for m in self.performance_data 
                if m['timestamp'] > cutoff_time
            ]
        except Exception as e:
            self.logger.error(f"Error getting performance data: {e}")
            return []
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for specified time period"""
        try:
            performance_data = self.get_performance_data(hours)
            
            if not performance_data:
                return {'error': 'No performance data available'}
            
            # Calculate summary statistics
            summary = {
                'period_hours': hours,
                'data_points': len(performance_data),
                'prediction_accuracy': self._calculate_metric_summary('prediction_accuracy', performance_data),
                'betting_performance': self._calculate_metric_summary('betting_performance', performance_data),
                'model_performance': self._calculate_metric_summary('model_performance', performance_data),
                'system_performance': self._calculate_metric_summary('system_performance', performance_data)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}
    
    def _calculate_metric_summary(self, metric_category: str, performance_data: List[Dict]) -> Dict[str, Any]:
        """Calculate summary for a metric category"""
        try:
            category_data = [m[metric_category] for m in performance_data if metric_category in m]
            
            if not category_data:
                return {}
            
            summary = {}
            for metric_name in category_data[0].keys():
                values = [m[metric_name] for m in category_data if metric_name in m and isinstance(m[metric_name], (int, float))]
                if values:
                    summary[metric_name] = {
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'latest': values[-1]
                    }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error calculating metric summary: {e}")
            return {}
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active performance alerts"""
        return [
            alert for alert in self.alert_history 
            if alert['status'] == 'active'
        ]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge a performance alert"""
        try:
            for alert in self.alert_history:
                if alert['id'] == alert_id:
                    alert['status'] = 'acknowledged'
                    alert['acknowledged_at'] = datetime.now()
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve a performance alert"""
        try:
            for alert in self.alert_history:
                if alert['id'] == alert_id:
                    alert['status'] = 'resolved'
                    alert['resolved_at'] = datetime.now()
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error resolving alert: {e}")
            return False
    
    def generate_performance_report(self, report_type: str = 'summary', hours: int = 24) -> str:
        """Generate comprehensive performance report"""
        try:
            report = []
            report.append("=" * 80)
            report.append("PERFORMANCE TRACKING REPORT - NON-MAJOR LEAGUE ML PIPELINE")
            report.append("=" * 80)
            report.append("")
            
            # Performance summary
            summary = self.get_performance_summary(hours)
            if 'error' not in summary:
                report.append("PERFORMANCE SUMMARY:")
                report.append(f"  Period: {hours} hours")
                report.append(f"  Data Points: {summary['data_points']}")
                report.append("")
                
                # Prediction accuracy
                if 'prediction_accuracy' in summary:
                    report.append("PREDICTION ACCURACY:")
                    for metric, stats in summary['prediction_accuracy'].items():
                        report.append(f"  {metric}: {stats['latest']:.3f} (avg: {stats['avg']:.3f})")
                    report.append("")
                
                # Betting performance
                if 'betting_performance' in summary:
                    report.append("BETTING PERFORMANCE:")
                    for metric, stats in summary['betting_performance'].items():
                        if metric in ['roi', 'win_rate']:
                            report.append(f"  {metric}: {stats['latest']:.3f} (avg: {stats['avg']:.3f})")
                        else:
                            report.append(f"  {metric}: {stats['latest']:.2f} (avg: {stats['avg']:.2f})")
                    report.append("")
                
                # Model performance
                if 'model_performance' in summary:
                    report.append("MODEL PERFORMANCE:")
                    for metric, stats in summary['model_performance'].items():
                        report.append(f"  {metric}: {stats['latest']:.3f} (avg: {stats['avg']:.3f})")
                    report.append("")
                
                # System performance
                if 'system_performance' in summary:
                    report.append("SYSTEM PERFORMANCE:")
                    for metric, stats in summary['system_performance'].items():
                        report.append(f"  {metric}: {stats['latest']:.3f} (avg: {stats['avg']:.3f})")
                    report.append("")
            
            # Active alerts
            active_alerts = self.get_active_alerts()
            report.append("ACTIVE ALERTS:")
            if active_alerts:
                for alert in active_alerts:
                    report.append(f"  {alert['level'].upper()}: {alert['condition']}")
                    report.append(f"    Time: {alert['timestamp']}")
                    report.append(f"    Status: {alert['status']}")
            else:
                report.append("  No active alerts")
            report.append("")
            
            # Recommendations
            report.append("RECOMMENDATIONS:")
            if active_alerts:
                report.append(f"  ⚠️  Resolve {len(active_alerts)} active alerts")
            if summary.get('prediction_accuracy', {}).get('overall_accuracy', {}).get('latest', 0) < 0.45:
                report.append("  📊 Review prediction accuracy and model performance")
            if summary.get('betting_performance', {}).get('roi', {}).get('latest', 0) < 0:
                report.append("  💰 Review betting strategy and risk management")
            if not active_alerts and summary.get('prediction_accuracy', {}).get('overall_accuracy', {}).get('latest', 0) > 0.45:
                report.append("  ✅ System performing within acceptable parameters")
            report.append("")
            
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return f"Error generating report: {e}"
    
    def save_performance_state(self, filepath: str):
        """Save performance tracking state"""
        self.logger.info(f"Saving performance tracking state to {filepath}")
        
        import json
        
        performance_state = {
            'performance_data': self.performance_data,
            'reporting_history': self.reporting_history,
            'alert_history': self.alert_history,
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(performance_state, f, indent=2, default=str)
        
        self.logger.info("Performance tracking state saved successfully")

# Example usage
def main():
    """Example usage of NonMajorLeaguePerformanceTracking"""
    
    # Initialize performance tracking
    tracker = NonMajorLeaguePerformanceTracking()
    
    # Start tracking
    tracker.start_tracking()
    
    # Let it run for a bit
    import time
    time.sleep(10)
    
    # Get performance summary
    summary = tracker.get_performance_summary(1)
    print(f"Performance Summary: {len(summary.get('data_points', 0))} data points")
    
    # Get active alerts
    alerts = tracker.get_active_alerts()
    print(f"Active Alerts: {len(alerts)}")
    
    # Generate report
    report = tracker.generate_performance_report('summary', 1)
    print(report)
    
    # Stop tracking
    tracker.stop_tracking()
    
    # Save state
    tracker.save_performance_state('performance_tracking_state.json')

if __name__ == "__main__":
    main()
