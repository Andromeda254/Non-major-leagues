#!/usr/bin/env python3
"""
Phase 4 Integration Script for Non-Major League ML Pipeline

This script integrates all Phase 4 components:
1. Production Deployment System
2. Comprehensive Monitoring and Alerting
3. Automated Data Pipeline
4. Model Serving API
5. Real-time Performance Tracking

Usage:
    python phase4_integration.py --environment production --config ./config/phase4_config.yaml
"""

import argparse
import os
import sys
import yaml
import json
import time
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Import our custom modules
from non_major_league_deployment import NonMajorLeagueDeployment
from non_major_league_monitoring import NonMajorLeagueMonitoring
from non_major_league_data_pipeline import NonMajorLeagueDataPipeline
from non_major_league_model_serving import NonMajorLeagueModelServing
from non_major_league_performance_tracking import NonMajorLeaguePerformanceTracking

class Phase4Integration:
    """
    Phase 4 Integration for Non-Major League ML Pipeline
    
    This class orchestrates the complete Phase 4 workflow:
    - Production deployment with automated workflows
    - Comprehensive monitoring and alerting
    - Automated data pipeline for live data ingestion
    - Model serving API with prediction endpoints
    - Real-time performance tracking and reporting
    """
    
    def __init__(self, config_file: str = None, environment: str = "production"):
        """
        Initialize Phase 4 integration
        
        Args:
            config_file: Path to configuration file
            environment: Deployment environment (development, staging, production)
        """
        self.setup_logging()
        self.environment = environment
        self.load_config(config_file)
        
        # Initialize components
        self.deployment = NonMajorLeagueDeployment(self.config.get('deployment'))
        self.monitoring = NonMajorLeagueMonitoring(self.config.get('monitoring'))
        self.data_pipeline = NonMajorLeagueDataPipeline(self.config.get('data_pipeline'))
        self.model_serving = NonMajorLeagueModelServing(self.config.get('model_serving'))
        self.performance_tracking = NonMajorLeaguePerformanceTracking(self.config.get('performance_tracking'))
        
        # Integration state
        self.integration_active = False
        self.integration_thread = None
        self.component_status = {}
        
    def setup_logging(self):
        """Setup logging for Phase 4 integration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('phase4_integration.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config_file: str):
        """Load configuration from file or use defaults"""
        if config_file and os.path.exists(config_file):
            with open(config_file, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = self._get_default_config()
        
        self.logger.info(f"Configuration loaded for environment: {self.environment}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'deployment': {
                'environments': {
                    'development': {
                        'enabled': True,
                        'base_url': 'http://localhost:8000',
                        'database_url': 'sqlite:///dev.db',
                        'log_level': 'DEBUG',
                        'debug': True
                    },
                    'staging': {
                        'enabled': True,
                        'base_url': 'http://staging.example.com',
                        'database_url': 'postgresql://staging:password@staging-db:5432/staging',
                        'log_level': 'INFO',
                        'debug': False
                    },
                    'production': {
                        'enabled': True,
                        'base_url': 'https://api.example.com',
                        'database_url': 'postgresql://prod:password@prod-db:5432/production',
                        'log_level': 'WARNING',
                        'debug': False
                    }
                },
                'services': {
                    'api_server': {
                        'name': 'ml-soccer-api',
                        'port': 8000,
                        'replicas': 2,
                        'resources': {
                            'cpu': '500m',
                            'memory': '1Gi'
                        }
                    },
                    'data_pipeline': {
                        'name': 'ml-soccer-pipeline',
                        'schedule': '0 */6 * * *',
                        'resources': {
                            'cpu': '200m',
                            'memory': '512Mi'
                        }
                    },
                    'monitoring': {
                        'name': 'ml-soccer-monitoring',
                        'port': 9090,
                        'resources': {
                            'cpu': '100m',
                            'memory': '256Mi'
                        }
                    }
                }
            },
            'monitoring': {
                'enabled': True,
                'interval': 60,
                'metrics': {
                    'system': {
                        'cpu_usage': {'enabled': True, 'threshold': 0.8},
                        'memory_usage': {'enabled': True, 'threshold': 0.8},
                        'disk_usage': {'enabled': True, 'threshold': 0.9}
                    },
                    'application': {
                        'response_time': {'enabled': True, 'threshold': 2.0},
                        'error_rate': {'enabled': True, 'threshold': 0.05},
                        'throughput': {'enabled': True, 'threshold': 100}
                    },
                    'business': {
                        'prediction_accuracy': {'enabled': True, 'threshold': 0.45},
                        'betting_win_rate': {'enabled': True, 'threshold': 0.4},
                        'daily_return': {'enabled': True, 'threshold': -0.05}
                    }
                },
                'alerting': {
                    'enabled': True,
                    'channels': {
                        'email': {'enabled': True},
                        'slack': {'enabled': True},
                        'webhook': {'enabled': True}
                    }
                }
            },
            'data_pipeline': {
                'enabled': True,
                'schedule_interval': 3600,
                'data_sources': {
                    'football_data_co_uk': {'enabled': True},
                    'the_odds_api': {'enabled': True},
                    'api_football': {'enabled': True}
                },
                'target_leagues': {
                    'Championship': {'enabled': True, 'code': 'E1'},
                    'League_One': {'enabled': True, 'code': 'E2'},
                    'League_Two': {'enabled': True, 'code': 'E3'}
                }
            },
            'model_serving': {
                'enabled': True,
                'api': {
                    'host': '0.0.0.0',
                    'port': 8000,
                    'debug': False
                },
                'models': {
                    'xgboost': {'enabled': True, 'model_path': './models/xgboost_model.pkl'},
                    'lightgbm': {'enabled': True, 'model_path': './models/lightgbm_model.pkl'},
                    'ensemble': {'enabled': True, 'model_path': './models/ensemble_model.pkl'}
                },
                'authentication': {
                    'enabled': True,
                    'secret_key': 'your-secret-key-here'
                }
            },
            'performance_tracking': {
                'enabled': True,
                'interval': 300,
                'metrics': {
                    'prediction_accuracy': {'enabled': True, 'threshold': 0.45},
                    'betting_performance': {'enabled': True, 'win_rate_threshold': 0.4},
                    'model_performance': {'enabled': True, 'response_time_threshold': 2.0}
                },
                'reporting': {
                    'enabled': True,
                    'schedule': {
                        'real_time': True,
                        'hourly': True,
                        'daily': True,
                        'weekly': True
                    }
                }
            }
        }
    
    def run_phase4_pipeline(self, model_files: List[str] = None) -> Dict[str, Any]:
        """
        Run complete Phase 4 pipeline
        
        Args:
            model_files: List of model files to deploy
            
        Returns:
            Dictionary with pipeline results
        """
        self.logger.info(f"Starting Phase 4 pipeline for {self.environment}")
        
        try:
            # Step 1: Production Deployment
            self.logger.info("Step 1: Production Deployment")
            deployment_results = self._run_production_deployment(model_files)
            
            # Step 2: Monitoring Setup
            self.logger.info("Step 2: Monitoring Setup")
            monitoring_results = self._setup_monitoring()
            
            # Step 3: Data Pipeline Deployment
            self.logger.info("Step 3: Data Pipeline Deployment")
            data_pipeline_results = self._deploy_data_pipeline()
            
            # Step 4: Model Serving API
            self.logger.info("Step 4: Model Serving API")
            model_serving_results = self._deploy_model_serving()
            
            # Step 5: Performance Tracking
            self.logger.info("Step 5: Performance Tracking")
            performance_tracking_results = self._setup_performance_tracking()
            
            # Step 6: Integration and Validation
            self.logger.info("Step 6: Integration and Validation")
            integration_results = self._integrate_and_validate_components(
                deployment_results, monitoring_results, data_pipeline_results,
                model_serving_results, performance_tracking_results
            )
            
            # Step 7: Start Production Services
            self.logger.info("Step 7: Start Production Services")
            production_results = self._start_production_services()
            
            # Step 8: Final Validation
            self.logger.info("Step 8: Final Validation")
            final_results = self._final_validation()
            
            self.logger.info("Phase 4 pipeline completed successfully")
            return final_results
            
        except Exception as e:
            self.logger.error(f"Phase 4 pipeline failed: {e}")
            raise
    
    def _run_production_deployment(self, model_files: List[str]) -> Dict[str, Any]:
        """Run production deployment"""
        self.logger.info("Running production deployment")
        
        try:
            if model_files is None:
                model_files = [
                    './models/xgboost_model.pkl',
                    './models/lightgbm_model.pkl',
                    './models/ensemble_model.pkl'
                ]
            
            # Deploy to specified environment
            deployment_result = self.deployment.deploy(
                environment=self.environment,
                version="1.0.0",
                model_files=model_files
            )
            
            if deployment_result['success']:
                self.component_status['deployment'] = 'success'
                self.logger.info("Production deployment completed successfully")
            else:
                self.component_status['deployment'] = 'failed'
                self.logger.error(f"Production deployment failed: {deployment_result.get('error')}")
            
            return {
                'deployment_result': deployment_result,
                'status': self.component_status['deployment']
            }
            
        except Exception as e:
            self.logger.error(f"Error in production deployment: {e}")
            self.component_status['deployment'] = 'failed'
            return {'status': 'failed', 'error': str(e)}
    
    def _setup_monitoring(self) -> Dict[str, Any]:
        """Setup monitoring system"""
        self.logger.info("Setting up monitoring system")
        
        try:
            # Start monitoring
            self.monitoring.start_monitoring()
            
            # Wait for monitoring to initialize
            time.sleep(5)
            
            # Check monitoring status
            health_status = self.monitoring.get_health_status()
            
            if health_status['overall_healthy']:
                self.component_status['monitoring'] = 'success'
                self.logger.info("Monitoring system setup completed successfully")
            else:
                self.component_status['monitoring'] = 'warning'
                self.logger.warning("Monitoring system setup completed with warnings")
            
            return {
                'health_status': health_status,
                'status': self.component_status['monitoring']
            }
            
        except Exception as e:
            self.logger.error(f"Error setting up monitoring: {e}")
            self.component_status['monitoring'] = 'failed'
            return {'status': 'failed', 'error': str(e)}
    
    def _deploy_data_pipeline(self) -> Dict[str, Any]:
        """Deploy data pipeline"""
        self.logger.info("Deploying data pipeline")
        
        try:
            # Start data pipeline
            self.data_pipeline.start_pipeline()
            
            # Wait for pipeline to initialize
            time.sleep(5)
            
            # Check pipeline status
            pipeline_status = self.data_pipeline.get_pipeline_status()
            
            if pipeline_status['active']:
                self.component_status['data_pipeline'] = 'success'
                self.logger.info("Data pipeline deployment completed successfully")
            else:
                self.component_status['data_pipeline'] = 'failed'
                self.logger.error("Data pipeline deployment failed")
            
            return {
                'pipeline_status': pipeline_status,
                'status': self.component_status['data_pipeline']
            }
            
        except Exception as e:
            self.logger.error(f"Error deploying data pipeline: {e}")
            self.component_status['data_pipeline'] = 'failed'
            return {'status': 'failed', 'error': str(e)}
    
    def _deploy_model_serving(self) -> Dict[str, Any]:
        """Deploy model serving API"""
        self.logger.info("Deploying model serving API")
        
        try:
            # Load all models
            self.model_serving.load_all_models()
            
            # Get server info
            server_info = self.model_serving.get_server_info()
            
            if server_info['models_loaded'] > 0:
                self.component_status['model_serving'] = 'success'
                self.logger.info("Model serving API deployment completed successfully")
            else:
                self.component_status['model_serving'] = 'failed'
                self.logger.error("Model serving API deployment failed - no models loaded")
            
            return {
                'server_info': server_info,
                'status': self.component_status['model_serving']
            }
            
        except Exception as e:
            self.logger.error(f"Error deploying model serving API: {e}")
            self.component_status['model_serving'] = 'failed'
            return {'status': 'failed', 'error': str(e)}
    
    def _setup_performance_tracking(self) -> Dict[str, Any]:
        """Setup performance tracking"""
        self.logger.info("Setting up performance tracking")
        
        try:
            # Start performance tracking
            self.performance_tracking.start_tracking()
            
            # Wait for tracking to initialize
            time.sleep(5)
            
            # Get initial performance summary
            performance_summary = self.performance_tracking.get_performance_summary(1)
            
            if 'error' not in performance_summary:
                self.component_status['performance_tracking'] = 'success'
                self.logger.info("Performance tracking setup completed successfully")
            else:
                self.component_status['performance_tracking'] = 'warning'
                self.logger.warning("Performance tracking setup completed with warnings")
            
            return {
                'performance_summary': performance_summary,
                'status': self.component_status['performance_tracking']
            }
            
        except Exception as e:
            self.logger.error(f"Error setting up performance tracking: {e}")
            self.component_status['performance_tracking'] = 'failed'
            return {'status': 'failed', 'error': str(e)}
    
    def _integrate_and_validate_components(self, deployment_results: Dict, 
                                        monitoring_results: Dict,
                                        data_pipeline_results: Dict,
                                        model_serving_results: Dict,
                                        performance_tracking_results: Dict) -> Dict[str, Any]:
        """Integrate and validate all components"""
        self.logger.info("Integrating and validating components")
        
        try:
            # Calculate overall integration status
            component_results = {
                'deployment': deployment_results,
                'monitoring': monitoring_results,
                'data_pipeline': data_pipeline_results,
                'model_serving': model_serving_results,
                'performance_tracking': performance_tracking_results
            }
            
            successful_components = sum(1 for result in component_results.values() 
                                     if result.get('status') == 'success')
            total_components = len(component_results)
            
            integration_success = successful_components >= 4  # At least 4/5 components must succeed
            
            # Generate integration summary
            integration_summary = {
                'total_components': total_components,
                'successful_components': successful_components,
                'integration_success': integration_success,
                'component_status': self.component_status,
                'overall_status': 'success' if integration_success else 'failed'
            }
            
            # Generate recommendations
            recommendations = []
            if integration_success:
                recommendations.extend([
                    "Phase 4 integration completed successfully",
                    "All production services are operational",
                    "Continue monitoring system performance",
                    "Regular maintenance and updates recommended"
                ])
            else:
                recommendations.extend([
                    "Phase 4 integration encountered issues",
                    "Review failed components and address issues",
                    "Consider rolling back to previous version",
                    "Do not proceed with production traffic"
                ])
            
            integration_summary['recommendations'] = recommendations
            
            self.logger.info(f"Integration validation: {'PASSED' if integration_success else 'FAILED'}")
            
            return {
                'integration_summary': integration_summary,
                'component_results': component_results,
                'status': 'success' if integration_success else 'failed'
            }
            
        except Exception as e:
            self.logger.error(f"Error in integration validation: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _start_production_services(self) -> Dict[str, Any]:
        """Start production services"""
        self.logger.info("Starting production services")
        
        try:
            # Start integration thread
            self.integration_active = True
            self.integration_thread = threading.Thread(target=self._integration_loop)
            self.integration_thread.daemon = True
            self.integration_thread.start()
            
            # Wait for services to start
            time.sleep(10)
            
            # Check service health
            service_health = self._check_service_health()
            
            if service_health['overall_healthy']:
                self.logger.info("Production services started successfully")
                return {'status': 'success', 'service_health': service_health}
            else:
                self.logger.warning("Production services started with health issues")
                return {'status': 'warning', 'service_health': service_health}
            
        except Exception as e:
            self.logger.error(f"Error starting production services: {e}")
            return {'status': 'failed', 'error': str(e)}
    
    def _integration_loop(self):
        """Main integration loop for continuous operation"""
        while self.integration_active:
            try:
                # Check component health
                self._check_component_health()
                
                # Generate performance reports
                self._generate_performance_reports()
                
                # Check for alerts
                self._process_alerts()
                
                # Sleep for 5 minutes
                time.sleep(300)
                
            except Exception as e:
                self.logger.error(f"Error in integration loop: {e}")
                time.sleep(60)
    
    def _check_component_health(self):
        """Check health of all components"""
        try:
            # Check monitoring health
            if self.component_status.get('monitoring') == 'success':
                health_status = self.monitoring.get_health_status()
                if not health_status['overall_healthy']:
                    self.logger.warning("Monitoring system health issues detected")
            
            # Check data pipeline health
            if self.component_status.get('data_pipeline') == 'success':
                pipeline_status = self.data_pipeline.get_pipeline_status()
                if not pipeline_status['active']:
                    self.logger.warning("Data pipeline health issues detected")
            
            # Check performance tracking health
            if self.component_status.get('performance_tracking') == 'success':
                performance_summary = self.performance_tracking.get_performance_summary(1)
                if 'error' in performance_summary:
                    self.logger.warning("Performance tracking health issues detected")
            
        except Exception as e:
            self.logger.error(f"Error checking component health: {e}")
    
    def _generate_performance_reports(self):
        """Generate performance reports"""
        try:
            # Generate performance tracking report
            if self.component_status.get('performance_tracking') == 'success':
                report = self.performance_tracking.generate_performance_report('summary', 24)
                self.logger.info("Performance report generated")
            
        except Exception as e:
            self.logger.error(f"Error generating performance reports: {e}")
    
    def _process_alerts(self):
        """Process alerts from all components"""
        try:
            # Process monitoring alerts
            if self.component_status.get('monitoring') == 'success':
                alerts = self.monitoring.get_active_alerts()
                if alerts:
                    self.logger.warning(f"Active monitoring alerts: {len(alerts)}")
            
            # Process performance tracking alerts
            if self.component_status.get('performance_tracking') == 'success':
                alerts = self.performance_tracking.get_active_alerts()
                if alerts:
                    self.logger.warning(f"Active performance alerts: {len(alerts)}")
            
        except Exception as e:
            self.logger.error(f"Error processing alerts: {e}")
    
    def _check_service_health(self) -> Dict[str, Any]:
        """Check health of all services"""
        try:
            service_health = {
                'overall_healthy': True,
                'services': {}
            }
            
            # Check deployment health
            if self.component_status.get('deployment') == 'success':
                deployment_health = self.deployment.health_check(self.environment)
                service_health['services']['deployment'] = deployment_health
                if not deployment_health['healthy']:
                    service_health['overall_healthy'] = False
            
            # Check monitoring health
            if self.component_status.get('monitoring') == 'success':
                monitoring_health = self.monitoring.get_health_status()
                service_health['services']['monitoring'] = monitoring_health
                if not monitoring_health['overall_healthy']:
                    service_health['overall_healthy'] = False
            
            # Check data pipeline health
            if self.component_status.get('data_pipeline') == 'success':
                pipeline_status = self.data_pipeline.get_pipeline_status()
                service_health['services']['data_pipeline'] = {
                    'healthy': pipeline_status['active'],
                    'status': pipeline_status
                }
                if not pipeline_status['active']:
                    service_health['overall_healthy'] = False
            
            # Check model serving health
            if self.component_status.get('model_serving') == 'success':
                server_info = self.model_serving.get_server_info()
                service_health['services']['model_serving'] = {
                    'healthy': server_info['models_loaded'] > 0,
                    'status': server_info
                }
                if server_info['models_loaded'] == 0:
                    service_health['overall_healthy'] = False
            
            # Check performance tracking health
            if self.component_status.get('performance_tracking') == 'success':
                performance_summary = self.performance_tracking.get_performance_summary(1)
                service_health['services']['performance_tracking'] = {
                    'healthy': 'error' not in performance_summary,
                    'status': performance_summary
                }
                if 'error' in performance_summary:
                    service_health['overall_healthy'] = False
            
            return service_health
            
        except Exception as e:
            self.logger.error(f"Error checking service health: {e}")
            return {'overall_healthy': False, 'error': str(e)}
    
    def _final_validation(self) -> Dict[str, Any]:
        """Perform final validation of the complete system"""
        self.logger.info("Performing final validation")
        
        try:
            # Check all component status
            component_status_check = all(
                status in ['success', 'warning'] 
                for status in self.component_status.values()
            )
            
            # Check service health
            service_health = self._check_service_health()
            
            # Generate final report
            final_report = self._generate_final_report()
            
            # Determine overall success
            overall_success = (
                component_status_check and 
                service_health['overall_healthy'] and
                len(self.component_status) >= 4
            )
            
            final_results = {
                'overall_success': overall_success,
                'component_status': self.component_status,
                'service_health': service_health,
                'final_report': final_report,
                'timestamp': datetime.now().isoformat(),
                'environment': self.environment
            }
            
            self.logger.info(f"Final validation: {'PASSED' if overall_success else 'FAILED'}")
            
            return final_results
            
        except Exception as e:
            self.logger.error(f"Error in final validation: {e}")
            return {'overall_success': False, 'error': str(e)}
    
    def _generate_final_report(self) -> str:
        """Generate final integration report"""
        try:
            report = []
            report.append("=" * 80)
            report.append("PHASE 4 INTEGRATION REPORT - NON-MAJOR LEAGUE ML PIPELINE")
            report.append("=" * 80)
            report.append("")
            
            # Environment information
            report.append("ENVIRONMENT INFORMATION:")
            report.append(f"  Environment: {self.environment}")
            report.append(f"  Timestamp: {datetime.now().isoformat()}")
            report.append("")
            
            # Component status
            report.append("COMPONENT STATUS:")
            for component, status in self.component_status.items():
                status_icon = "✅" if status == 'success' else "⚠️" if status == 'warning' else "❌"
                report.append(f"  {component.replace('_', ' ').title()}: {status_icon} {status}")
            report.append("")
            
            # Service health
            service_health = self._check_service_health()
            report.append("SERVICE HEALTH:")
            report.append(f"  Overall Status: {'HEALTHY' if service_health['overall_healthy'] else 'UNHEALTHY'}")
            for service, health in service_health['services'].items():
                health_icon = "✅" if health['healthy'] else "❌"
                report.append(f"  {service.replace('_', ' ').title()}: {health_icon}")
            report.append("")
            
            # Recommendations
            report.append("RECOMMENDATIONS:")
            if service_health['overall_healthy']:
                report.append("  ✅ System is ready for production traffic")
                report.append("  📊 Continue monitoring system performance")
                report.append("  🔄 Schedule regular maintenance and updates")
                report.append("  📈 Monitor key performance metrics")
            else:
                report.append("  ❌ System not ready for production traffic")
                report.append("  🔧 Address health issues before proceeding")
                report.append("  📋 Review component logs for details")
                report.append("  🔄 Consider rolling back to previous version")
            report.append("")
            
            # Next steps
            report.append("NEXT STEPS:")
            if service_health['overall_healthy']:
                report.append("  1. Begin gradual traffic migration")
                report.append("  2. Monitor system performance closely")
                report.append("  3. Set up automated alerting")
                report.append("  4. Schedule regular health checks")
                report.append("  5. Plan for maintenance windows")
            else:
                report.append("  1. Investigate and fix health issues")
                report.append("  2. Review component logs")
                report.append("  3. Test components individually")
                report.append("  4. Consider rolling back")
                report.append("  5. Re-run Phase 4 integration")
            report.append("")
            
            report.append("=" * 80)
            
            return "\n".join(report)
            
        except Exception as e:
            self.logger.error(f"Error generating final report: {e}")
            return f"Error generating report: {e}"
    
    def stop_integration(self):
        """Stop the integration system"""
        self.logger.info("Stopping Phase 4 integration")
        
        try:
            # Stop integration loop
            self.integration_active = False
            
            if self.integration_thread:
                self.integration_thread.join(timeout=30)
            
            # Stop all components
            if self.component_status.get('monitoring') == 'success':
                self.monitoring.stop_monitoring()
            
            if self.component_status.get('data_pipeline') == 'success':
                self.data_pipeline.stop_pipeline()
            
            if self.component_status.get('performance_tracking') == 'success':
                self.performance_tracking.stop_tracking()
            
            self.logger.info("Phase 4 integration stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping integration: {e}")
    
    def save_integration_state(self, filepath: str):
        """Save integration state"""
        self.logger.info(f"Saving integration state to {filepath}")
        
        try:
            integration_state = {
                'environment': self.environment,
                'component_status': self.component_status,
                'config': self.config,
                'timestamp': datetime.now().isoformat()
            }
            
            with open(filepath, 'w') as f:
                json.dump(integration_state, f, indent=2)
            
            self.logger.info("Integration state saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving integration state: {e}")

def main():
    """Main function for Phase 4 integration"""
    parser = argparse.ArgumentParser(description='Phase 4 Integration for Non-Major League ML Pipeline')
    parser.add_argument('--environment', required=True, 
                       choices=['development', 'staging', 'production'],
                       help='Deployment environment')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--model_files', nargs='+', 
                       help='List of model files to deploy')
    parser.add_argument('--output_dir', default='./results', 
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Initialize Phase 4 integration
    phase4 = Phase4Integration(args.config, args.environment)
    
    try:
        # Run Phase 4 pipeline
        results = phase4.run_phase4_pipeline(args.model_files)
        
        # Generate and display report
        if 'final_report' in results:
            print(results['final_report'])
        
        # Save results
        os.makedirs(args.output_dir, exist_ok=True)
        results_file = os.path.join(args.output_dir, f"{args.environment}_phase4_results.json")
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save integration state
        state_file = os.path.join(args.output_dir, f"{args.environment}_phase4_state.json")
        phase4.save_integration_state(state_file)
        
        print(f"\nPhase 4 integration completed!")
        print(f"Results saved to: {results_file}")
        print(f"State saved to: {state_file}")
        
        if results['overall_success']:
            print("✅ System is ready for production!")
        else:
            print("❌ System requires attention before production deployment")
            sys.exit(1)
        
    except Exception as e:
        print(f"Phase 4 integration failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
