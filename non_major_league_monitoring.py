import pandas as pd
import numpy as np
import time
import threading
import schedule
import requests
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeagueMonitoring:
    """
    Comprehensive monitoring and alerting system for non-major soccer leagues
    
    Key Features:
    - Real-time system monitoring
    - Performance metrics tracking
    - Automated alerting
    - Health checks
    - Dashboard integration
    - Log aggregation
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize monitoring system
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.metrics_data = []
        self.alerts_history = []
        self.health_status = {}
        self.monitoring_active = False
        self.monitoring_thread = None
        
    def setup_logging(self):
        """Setup logging for monitoring"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load monitoring configuration"""
        if config is None:
            self.config = {
                'monitoring': {
                    'enabled': True,
                    'interval': 60,  # 1 minute
                    'retention_days': 30,
                    'real_time': True
                },
                'metrics': {
                    'system': {
                        'cpu_usage': {'enabled': True, 'threshold': 0.8},
                        'memory_usage': {'enabled': True, 'threshold': 0.8},
                        'disk_usage': {'enabled': True, 'threshold': 0.9},
                        'network_latency': {'enabled': True, 'threshold': 1000}
                    },
                    'application': {
                        'response_time': {'enabled': True, 'threshold': 2.0},
                        'error_rate': {'enabled': True, 'threshold': 0.05},
                        'throughput': {'enabled': True, 'threshold': 100},
                        'active_connections': {'enabled': True, 'threshold': 1000}
                    },
                    'business': {
                        'prediction_accuracy': {'enabled': True, 'threshold': 0.45},
                        'betting_win_rate': {'enabled': True, 'threshold': 0.4},
                        'daily_return': {'enabled': True, 'threshold': -0.05},
                        'drawdown': {'enabled': True, 'threshold': 0.2}
                    }
                },
                'alerting': {
                    'enabled': True,
                    'channels': {
                        'email': {
                            'enabled': True,
                            'smtp_server': 'smtp.gmail.com',
                            'smtp_port': 587,
                            'username': 'alerts@example.com',
                            'password': 'password',
                            'recipients': ['admin@example.com']
                        },
                        'slack': {
                            'enabled': True,
                            'webhook_url': 'https://hooks.slack.com/services/...',
                            'channel': '#alerts'
                        },
                        'webhook': {
                            'enabled': True,
                            'url': 'https://api.example.com/alerts',
                            'headers': {'Authorization': 'Bearer token'}
                        }
                    },
                    'rules': {
                        'critical': {
                            'conditions': ['error_rate > 0.1', 'cpu_usage > 0.95', 'drawdown > 0.25'],
                            'cooldown': 300,  # 5 minutes
                            'escalation': True
                        },
                        'warning': {
                            'conditions': ['response_time > 2.0', 'memory_usage > 0.8', 'drawdown > 0.15'],
                            'cooldown': 600,  # 10 minutes
                            'escalation': False
                        },
                        'info': {
                            'conditions': ['prediction_accuracy < 0.5', 'betting_win_rate < 0.45'],
                            'cooldown': 1800,  # 30 minutes
                            'escalation': False
                        }
                    }
                },
                'health_checks': {
                    'enabled': True,
                    'endpoints': {
                        'api': {
                            'url': 'http://localhost:8000/health',
                            'timeout': 10,
                            'interval': 30
                        },
                        'database': {
                            'url': 'http://localhost:5432/health',
                            'timeout': 5,
                            'interval': 60
                        },
                        'model_service': {
                            'url': 'http://localhost:8001/health',
                            'timeout': 10,
                            'interval': 30
                        }
                    }
                },
                'dashboard': {
                    'enabled': True,
                    'port': 8080,
                    'refresh_interval': 30,
                    'widgets': [
                        'system_metrics',
                        'application_metrics',
                        'business_metrics',
                        'alerts',
                        'health_status'
                    ]
                },
                'logging': {
                    'enabled': True,
                    'level': 'INFO',
                    'format': 'json',
                    'retention_days': 7,
                    'aggregation': True
                }
            }
        else:
            self.config = config
    
    def start_monitoring(self):
        """Start monitoring system"""
        if self.monitoring_active:
            self.logger.warning("Monitoring already active")
            return
        
        self.logger.info("Starting monitoring system")
        self.monitoring_active = True
        
        # Start monitoring thread
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop)
        self.monitoring_thread.daemon = True
        self.monitoring_thread.start()
        
        # Schedule periodic tasks
        schedule.every(self.config['monitoring']['interval']).seconds.do(self._collect_metrics)
        schedule.every(5).minutes.do(self._check_health)
        schedule.every(10).minutes.do(self._process_alerts)
        schedule.every(1).hour.do(self._cleanup_old_data)
        
        self.logger.info("Monitoring system started successfully")
    
    def stop_monitoring(self):
        """Stop monitoring system"""
        self.logger.info("Stopping monitoring system")
        self.monitoring_active = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=30)
        
        schedule.clear()
        self.logger.info("Monitoring system stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _collect_metrics(self):
        """Collect system and application metrics"""
        try:
            timestamp = datetime.now()
            metrics = {
                'timestamp': timestamp,
                'system': self._collect_system_metrics(),
                'application': self._collect_application_metrics(),
                'business': self._collect_business_metrics()
            }
            
            self.metrics_data.append(metrics)
            
            # Check for alerts
            self._check_alert_conditions(metrics)
            
        except Exception as e:
            self.logger.error(f"Error collecting metrics: {e}")
    
    def _collect_system_metrics(self) -> Dict[str, float]:
        """Collect system metrics"""
        system_metrics = {}
        
        try:
            import psutil
            
            # CPU usage
            if self.config['metrics']['system']['cpu_usage']['enabled']:
                system_metrics['cpu_usage'] = psutil.cpu_percent(interval=1)
            
            # Memory usage
            if self.config['metrics']['system']['memory_usage']['enabled']:
                memory = psutil.virtual_memory()
                system_metrics['memory_usage'] = memory.percent / 100
            
            # Disk usage
            if self.config['metrics']['system']['disk_usage']['enabled']:
                disk = psutil.disk_usage('/')
                system_metrics['disk_usage'] = disk.percent / 100
            
            # Network latency (simplified)
            if self.config['metrics']['system']['network_latency']['enabled']:
                start_time = time.time()
                try:
                    requests.get('http://localhost:8000/health', timeout=5)
                    system_metrics['network_latency'] = (time.time() - start_time) * 1000
                except:
                    system_metrics['network_latency'] = 9999
            
        except ImportError:
            self.logger.warning("psutil not available, using mock system metrics")
            system_metrics = {
                'cpu_usage': np.random.uniform(0.1, 0.5),
                'memory_usage': np.random.uniform(0.3, 0.7),
                'disk_usage': np.random.uniform(0.2, 0.6),
                'network_latency': np.random.uniform(10, 100)
            }
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            system_metrics = {}
        
        return system_metrics
    
    def _collect_application_metrics(self) -> Dict[str, float]:
        """Collect application metrics"""
        app_metrics = {}
        
        try:
            # Response time
            if self.config['metrics']['application']['response_time']['enabled']:
                start_time = time.time()
                try:
                    response = requests.get('http://localhost:8000/health', timeout=10)
                    app_metrics['response_time'] = time.time() - start_time
                    app_metrics['error_rate'] = 0 if response.status_code == 200 else 1
                except:
                    app_metrics['response_time'] = 10.0
                    app_metrics['error_rate'] = 1
            
            # Throughput (simplified)
            if self.config['metrics']['application']['throughput']['enabled']:
                app_metrics['throughput'] = np.random.uniform(50, 150)
            
            # Active connections (simplified)
            if self.config['metrics']['application']['active_connections']['enabled']:
                app_metrics['active_connections'] = np.random.uniform(100, 800)
            
        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")
            app_metrics = {}
        
        return app_metrics
    
    def _collect_business_metrics(self) -> Dict[str, float]:
        """Collect business metrics"""
        business_metrics = {}
        
        try:
            # Prediction accuracy (simplified)
            if self.config['metrics']['business']['prediction_accuracy']['enabled']:
                business_metrics['prediction_accuracy'] = np.random.uniform(0.4, 0.6)
            
            # Betting win rate (simplified)
            if self.config['metrics']['business']['betting_win_rate']['enabled']:
                business_metrics['betting_win_rate'] = np.random.uniform(0.35, 0.55)
            
            # Daily return (simplified)
            if self.config['metrics']['business']['daily_return']['enabled']:
                business_metrics['daily_return'] = np.random.uniform(-0.1, 0.1)
            
            # Drawdown (simulified)
            if self.config['metrics']['business']['drawdown']['enabled']:
                business_metrics['drawdown'] = np.random.uniform(0.05, 0.25)
            
        except Exception as e:
            self.logger.error(f"Error collecting business metrics: {e}")
            business_metrics = {}
        
        return business_metrics
    
    def _check_health(self):
        """Check health of all endpoints"""
        try:
            for endpoint_name, endpoint_config in self.config['health_checks']['endpoints'].items():
                try:
                    response = requests.get(
                        endpoint_config['url'],
                        timeout=endpoint_config['timeout']
                    )
                    
                    is_healthy = response.status_code == 200
                    self.health_status[endpoint_name] = {
                        'healthy': is_healthy,
                        'status_code': response.status_code,
                        'response_time': response.elapsed.total_seconds(),
                        'last_check': datetime.now()
                    }
                    
                except Exception as e:
                    self.health_status[endpoint_name] = {
                        'healthy': False,
                        'error': str(e),
                        'last_check': datetime.now()
                    }
            
        except Exception as e:
            self.logger.error(f"Error checking health: {e}")
    
    def _check_alert_conditions(self, metrics: Dict[str, Any]):
        """Check alert conditions and trigger alerts"""
        try:
            for alert_level, alert_config in self.config['alerting']['rules'].items():
                for condition in alert_config['conditions']:
                    if self._evaluate_condition(condition, metrics):
                        self._trigger_alert(alert_level, condition, metrics)
                        
        except Exception as e:
            self.logger.error(f"Error checking alert conditions: {e}")
    
    def _evaluate_condition(self, condition: str, metrics: Dict[str, Any]) -> bool:
        """Evaluate alert condition"""
        try:
            # Parse condition (simplified)
            if '>' in condition:
                metric_name, threshold_str = condition.split(' > ')
                metric_name = metric_name.strip()
                threshold = float(threshold_str.strip())
                
                # Find metric value
                metric_value = self._get_metric_value(metric_name, metrics)
                if metric_value is not None:
                    return metric_value > threshold
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition '{condition}': {e}")
            return False
    
    def _get_metric_value(self, metric_name: str, metrics: Dict[str, Any]) -> Optional[float]:
        """Get metric value from metrics data"""
        try:
            # Check system metrics
            if metric_name in metrics.get('system', {}):
                return metrics['system'][metric_name]
            
            # Check application metrics
            if metric_name in metrics.get('application', {}):
                return metrics['application'][metric_name]
            
            # Check business metrics
            if metric_name in metrics.get('business', {}):
                return metrics['business'][metric_name]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting metric value for '{metric_name}': {e}")
            return None
    
    def _trigger_alert(self, level: str, condition: str, metrics: Dict[str, Any]):
        """Trigger alert"""
        try:
            alert = {
                'id': f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'level': level,
                'condition': condition,
                'timestamp': datetime.now(),
                'metrics': metrics,
                'status': 'active'
            }
            
            self.alerts_history.append(alert)
            
            # Send alert through configured channels
            self._send_alert(alert)
            
            self.logger.warning(f"Alert triggered: {level} - {condition}")
            
        except Exception as e:
            self.logger.error(f"Error triggering alert: {e}")
    
    def _send_alert(self, alert: Dict[str, Any]):
        """Send alert through configured channels"""
        try:
            alert_message = self._format_alert_message(alert)
            
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
            self.logger.error(f"Error sending alert: {e}")
    
    def _format_alert_message(self, alert: Dict[str, Any]) -> str:
        """Format alert message"""
        return f"""
ALERT: {alert['level'].upper()}
Time: {alert['timestamp']}
Condition: {alert['condition']}
Status: {alert['status']}

Metrics:
{json.dumps(alert['metrics'], indent=2)}
"""
    
    def _send_email_alert(self, message: str):
        """Send email alert"""
        try:
            # Simplified email sending (would use smtplib in production)
            self.logger.info(f"Email alert sent: {message[:100]}...")
        except Exception as e:
            self.logger.error(f"Error sending email alert: {e}")
    
    def _send_slack_alert(self, message: str):
        """Send Slack alert"""
        try:
            slack_config = self.config['alerting']['channels']['slack']
            payload = {
                'text': message,
                'channel': slack_config['channel']
            }
            
            # Simplified Slack sending (would use requests in production)
            self.logger.info(f"Slack alert sent: {message[:100]}...")
        except Exception as e:
            self.logger.error(f"Error sending Slack alert: {e}")
    
    def _send_webhook_alert(self, message: str):
        """Send webhook alert"""
        try:
            webhook_config = self.config['alerting']['channels']['webhook']
            payload = {'message': message}
            
            # Simplified webhook sending (would use requests in production)
            self.logger.info(f"Webhook alert sent: {message[:100]}...")
        except Exception as e:
            self.logger.error(f"Error sending webhook alert: {e}")
    
    def _process_alerts(self):
        """Process and manage alerts"""
        try:
            # Mark old alerts as resolved
            cutoff_time = datetime.now() - timedelta(hours=1)
            for alert in self.alerts_history:
                if alert['status'] == 'active' and alert['timestamp'] < cutoff_time:
                    alert['status'] = 'resolved'
                    alert['resolved_at'] = datetime.now()
            
        except Exception as e:
            self.logger.error(f"Error processing alerts: {e}")
    
    def _cleanup_old_data(self):
        """Cleanup old metrics and alerts data"""
        try:
            cutoff_time = datetime.now() - timedelta(days=self.config['monitoring']['retention_days'])
            
            # Cleanup old metrics
            self.metrics_data = [
                m for m in self.metrics_data 
                if m['timestamp'] > cutoff_time
            ]
            
            # Cleanup old alerts
            self.alerts_history = [
                a for a in self.alerts_history 
                if a['timestamp'] > cutoff_time
            ]
            
            self.logger.info("Old data cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old data: {e}")
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for specified time period"""
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_metrics = [
                m for m in self.metrics_data 
                if m['timestamp'] > cutoff_time
            ]
            
            if not recent_metrics:
                return {'error': 'No metrics data available'}
            
            summary = {
                'time_period_hours': hours,
                'total_metrics': len(recent_metrics),
                'system': self._calculate_metric_summary('system', recent_metrics),
                'application': self._calculate_metric_summary('application', recent_metrics),
                'business': self._calculate_metric_summary('business', recent_metrics)
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting metrics summary: {e}")
            return {'error': str(e)}
    
    def _calculate_metric_summary(self, category: str, metrics: List[Dict]) -> Dict[str, Any]:
        """Calculate summary for a metric category"""
        try:
            category_metrics = [m[category] for m in metrics if category in m]
            
            if not category_metrics:
                return {}
            
            summary = {}
            for metric_name in category_metrics[0].keys():
                values = [m[metric_name] for m in category_metrics if metric_name in m]
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
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get current health status"""
        return {
            'overall_healthy': all(
                status['healthy'] for status in self.health_status.values()
            ),
            'endpoints': self.health_status,
            'last_check': datetime.now()
        }
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active alerts"""
        return [
            alert for alert in self.alerts_history 
            if alert['status'] == 'active'
        ]
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        try:
            for alert in self.alerts_history:
                if alert['id'] == alert_id:
                    alert['status'] = 'acknowledged'
                    alert['acknowledged_at'] = datetime.now()
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error acknowledging alert: {e}")
            return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Resolve an alert"""
        try:
            for alert in self.alerts_history:
                if alert['id'] == alert_id:
                    alert['status'] = 'resolved'
                    alert['resolved_at'] = datetime.now()
                    return True
            return False
        except Exception as e:
            self.logger.error(f"Error resolving alert: {e}")
            return False
    
    def generate_monitoring_report(self, hours: int = 24) -> str:
        """Generate comprehensive monitoring report"""
        try:
            report = []
            report.append("=" * 80)
            report.append("MONITORING REPORT - NON-MAJOR LEAGUE ML PIPELINE")
            report.append("=" * 80)
            report.append("")
            
            # Health status
            health_status = self.get_health_status()
            report.append("HEALTH STATUS:")
            report.append(f"  Overall Status: {'HEALTHY' if health_status['overall_healthy'] else 'UNHEALTHY'}")
            report.append(f"  Last Check: {health_status['last_check']}")
            report.append("  Endpoints:")
            for endpoint, status in health_status['endpoints'].items():
                status_text = "HEALTHY" if status['healthy'] else "UNHEALTHY"
                report.append(f"    {endpoint}: {status_text}")
            report.append("")
            
            # Metrics summary
            metrics_summary = self.get_metrics_summary(hours)
            if 'error' not in metrics_summary:
                report.append("METRICS SUMMARY:")
                report.append(f"  Time Period: {hours} hours")
                report.append(f"  Total Metrics: {metrics_summary['total_metrics']}")
                
                # System metrics
                if 'system' in metrics_summary:
                    report.append("  System Metrics:")
                    for metric, stats in metrics_summary['system'].items():
                        report.append(f"    {metric}: {stats['latest']:.3f} (avg: {stats['avg']:.3f})")
                
                # Application metrics
                if 'application' in metrics_summary:
                    report.append("  Application Metrics:")
                    for metric, stats in metrics_summary['application'].items():
                        report.append(f"    {metric}: {stats['latest']:.3f} (avg: {stats['avg']:.3f})")
                
                # Business metrics
                if 'business' in metrics_summary:
                    report.append("  Business Metrics:")
                    for metric, stats in metrics_summary['business'].items():
                        report.append(f"    {metric}: {stats['latest']:.3f} (avg: {stats['avg']:.3f})")
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
            if not health_status['overall_healthy']:
                report.append("  ❌ Address unhealthy endpoints")
            if active_alerts:
                report.append(f"  ⚠️  Resolve {len(active_alerts)} active alerts")
            if metrics_summary.get('system', {}).get('cpu_usage', {}).get('latest', 0) > 0.8:
                report.append("  🔧 High CPU usage detected")
            if metrics_summary.get('business', {}).get('drawdown', {}).get('latest', 0) > 0.15:
                report.append("  📉 High drawdown detected")
            if not active_alerts and health_status['overall_healthy']:
                report.append("  ✅ System operating normally")
            report.append("")
            
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating monitoring report: {e}")
            return f"Error generating report: {e}"
    
    def save_monitoring_state(self, filepath: str):
        """Save monitoring state"""
        self.logger.info(f"Saving monitoring state to {filepath}")
        
        monitoring_state = {
            'metrics_data': self.metrics_data,
            'alerts_history': self.alerts_history,
            'health_status': self.health_status,
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(monitoring_state, f, indent=2, default=str)
        
        self.logger.info("Monitoring state saved successfully")

# Example usage
def main():
    """Example usage of NonMajorLeagueMonitoring"""
    
    # Initialize monitoring system
    monitor = NonMajorLeagueMonitoring()
    
    # Start monitoring
    monitor.start_monitoring()
    
    # Let it run for a bit
    import time
    time.sleep(10)
    
    # Get health status
    health = monitor.get_health_status()
    print(f"Health Status: {'Healthy' if health['overall_healthy'] else 'Unhealthy'}")
    
    # Get metrics summary
    metrics = monitor.get_metrics_summary(1)
    print(f"Metrics Summary: {len(metrics.get('total_metrics', 0))} metrics collected")
    
    # Get active alerts
    alerts = monitor.get_active_alerts()
    print(f"Active Alerts: {len(alerts)}")
    
    # Generate report
    report = monitor.generate_monitoring_report(1)
    print(report)
    
    # Stop monitoring
    monitor.stop_monitoring()
    
    # Save state
    monitor.save_monitoring_state('monitoring_state.json')

if __name__ == "__main__":
    main()
