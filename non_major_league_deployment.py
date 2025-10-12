import pandas as pd
import numpy as np
import os
import sys
import subprocess
import json
import yaml
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeagueDeployment:
    """
    Production deployment system for non-major soccer leagues
    
    Key Features:
    - Automated deployment workflows
    - Environment management
    - Configuration management
    - Health checks and monitoring
    - Rollback capabilities
    - Security and access controls
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize deployment system
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.deployment_history = []
        self.current_deployment = None
        
    def setup_logging(self):
        """Setup logging for deployment"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load deployment configuration"""
        if config is None:
            self.config = {
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
                'deployment': {
                    'strategy': 'blue_green',  # 'blue_green', 'rolling', 'canary'
                    'health_check_timeout': 300,  # 5 minutes
                    'rollback_timeout': 600,      # 10 minutes
                    'max_deployment_time': 1800,  # 30 minutes
                    'backup_retention': 7,        # 7 days
                    'auto_rollback': True
                },
                'services': {
                    'api_server': {
                        'name': 'ml-soccer-api',
                        'port': 8000,
                        'replicas': 2,
                        'resources': {
                            'cpu': '500m',
                            'memory': '1Gi'
                        },
                        'health_check': {
                            'path': '/health',
                            'interval': 30,
                            'timeout': 10,
                            'retries': 3
                        }
                    },
                    'data_pipeline': {
                        'name': 'ml-soccer-pipeline',
                        'schedule': '0 */6 * * *',  # Every 6 hours
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
                },
                'security': {
                    'authentication': {
                        'enabled': True,
                        'method': 'jwt',
                        'token_expiry': 3600  # 1 hour
                    },
                    'authorization': {
                        'enabled': True,
                        'roles': ['admin', 'user', 'viewer'],
                        'permissions': {
                            'admin': ['read', 'write', 'delete', 'deploy'],
                            'user': ['read', 'write'],
                            'viewer': ['read']
                        }
                    },
                    'encryption': {
                        'enabled': True,
                        'algorithm': 'AES-256',
                        'key_rotation': 30  # 30 days
                    }
                },
                'monitoring': {
                    'metrics': {
                        'enabled': True,
                        'endpoint': '/metrics',
                        'interval': 60,  # 1 minute
                        'retention': 30   # 30 days
                    },
                    'logging': {
                        'enabled': True,
                        'level': 'INFO',
                        'format': 'json',
                        'retention': 7  # 7 days
                    },
                    'alerting': {
                        'enabled': True,
                        'channels': ['email', 'slack', 'webhook'],
                        'thresholds': {
                            'error_rate': 0.05,    # 5% error rate
                            'response_time': 2.0,  # 2 seconds
                            'cpu_usage': 0.8,      # 80% CPU
                            'memory_usage': 0.8    # 80% memory
                        }
                    }
                },
                'backup': {
                    'enabled': True,
                    'schedule': '0 2 * * *',  # Daily at 2 AM
                    'retention': 30,          # 30 days
                    'compression': True,
                    'encryption': True
                }
            }
        else:
            self.config = config
    
    def validate_deployment_config(self, environment: str) -> Dict[str, Any]:
        """Validate deployment configuration"""
        self.logger.info(f"Validating deployment configuration for {environment}")
        
        validation_result = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check environment exists
        if environment not in self.config['environments']:
            validation_result['valid'] = False
            validation_result['errors'].append(f"Environment '{environment}' not found")
            return validation_result
        
        env_config = self.config['environments'][environment]
        
        # Check environment is enabled
        if not env_config.get('enabled', False):
            validation_result['valid'] = False
            validation_result['errors'].append(f"Environment '{environment}' is disabled")
        
        # Check required services
        required_services = ['api_server', 'data_pipeline', 'monitoring']
        for service in required_services:
            if service not in self.config['services']:
                validation_result['valid'] = False
                validation_result['errors'].append(f"Required service '{service}' not configured")
        
        # Check security configuration
        if self.config['security']['authentication']['enabled']:
            if 'jwt' not in self.config['security']['authentication']['method']:
                validation_result['warnings'].append("JWT authentication recommended for production")
        
        # Check monitoring configuration
        if not self.config['monitoring']['metrics']['enabled']:
            validation_result['warnings'].append("Metrics collection is disabled")
        
        if not self.config['monitoring']['alerting']['enabled']:
            validation_result['warnings'].append("Alerting is disabled")
        
        self.logger.info(f"Configuration validation: {'PASSED' if validation_result['valid'] else 'FAILED'}")
        return validation_result
    
    def prepare_deployment(self, environment: str, version: str, 
                          model_files: List[str]) -> Dict[str, Any]:
        """Prepare deployment package"""
        self.logger.info(f"Preparing deployment for {environment} version {version}")
        
        try:
            # Validate configuration
            validation = self.validate_deployment_config(environment)
            if not validation['valid']:
                return {'success': False, 'errors': validation['errors']}
            
            # Create deployment package
            deployment_package = {
                'environment': environment,
                'version': version,
                'timestamp': datetime.now().isoformat(),
                'model_files': model_files,
                'config': self.config,
                'services': self.config['services'],
                'security': self.config['security'],
                'monitoring': self.config['monitoring']
            }
            
            # Create deployment directory
            deployment_dir = f"deployments/{environment}/{version}"
            os.makedirs(deployment_dir, exist_ok=True)
            
            # Save deployment package
            package_file = os.path.join(deployment_dir, 'deployment_package.json')
            with open(package_file, 'w') as f:
                json.dump(deployment_package, f, indent=2)
            
            # Create Docker files
            self._create_docker_files(deployment_dir, environment)
            
            # Create Kubernetes manifests
            self._create_kubernetes_manifests(deployment_dir, environment)
            
            # Create configuration files
            self._create_config_files(deployment_dir, environment)
            
            # Create startup scripts
            self._create_startup_scripts(deployment_dir, environment)
            
            self.logger.info(f"Deployment package created: {deployment_dir}")
            
            return {
                'success': True,
                'deployment_dir': deployment_dir,
                'package_file': package_file,
                'deployment_package': deployment_package
            }
            
        except Exception as e:
            self.logger.error(f"Error preparing deployment: {e}")
            return {'success': False, 'error': str(e)}
    
    def _create_docker_files(self, deployment_dir: str, environment: str):
        """Create Docker configuration files"""
        # Dockerfile
        dockerfile_content = """
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

# Start application
CMD ["python", "app.py"]
"""
        
        with open(os.path.join(deployment_dir, 'Dockerfile'), 'w') as f:
            f.write(dockerfile_content)
        
        # docker-compose.yml
        docker_compose_content = f"""
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - ENVIRONMENT={environment}
      - LOG_LEVEL={self.config['environments'][environment]['log_level']}
      - DEBUG={self.config['environments'][environment]['debug']}
    volumes:
      - ./models:/app/models
      - ./logs:/app/logs
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  monitoring:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  database:
    image: postgres:13
    environment:
      - POSTGRES_DB=ml_soccer
      - POSTGRES_USER=app
      - POSTGRES_PASSWORD=password
    volumes:
      - postgres_data:/var/lib/postgresql/data
    restart: unless-stopped

volumes:
  postgres_data:
"""
        
        with open(os.path.join(deployment_dir, 'docker-compose.yml'), 'w') as f:
            f.write(docker_compose_content)
    
    def _create_kubernetes_manifests(self, deployment_dir: str, environment: str):
        """Create Kubernetes deployment manifests"""
        # Create k8s directory
        k8s_dir = os.path.join(deployment_dir, 'k8s')
        os.makedirs(k8s_dir, exist_ok=True)
        
        # API deployment
        api_deployment = f"""
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ml-soccer-api
  labels:
    app: ml-soccer-api
spec:
  replicas: {self.config['services']['api_server']['replicas']}
  selector:
    matchLabels:
      app: ml-soccer-api
  template:
    metadata:
      labels:
        app: ml-soccer-api
    spec:
      containers:
      - name: api
        image: ml-soccer-api:{environment}
        ports:
        - containerPort: {self.config['services']['api_server']['port']}
        env:
        - name: ENVIRONMENT
          value: "{environment}"
        - name: LOG_LEVEL
          value: "{self.config['environments'][environment]['log_level']}"
        resources:
          requests:
            cpu: {self.config['services']['api_server']['resources']['cpu']}
            memory: {self.config['services']['api_server']['resources']['memory']}
          limits:
            cpu: {self.config['services']['api_server']['resources']['cpu']}
            memory: {self.config['services']['api_server']['resources']['memory']}
        livenessProbe:
          httpGet:
            path: /health
            port: {self.config['services']['api_server']['port']}
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /ready
            port: {self.config['services']['api_server']['port']}
          initialDelaySeconds: 5
          periodSeconds: 10
"""
        
        with open(os.path.join(k8s_dir, 'api-deployment.yaml'), 'w') as f:
            f.write(api_deployment)
        
        # API service
        api_service = f"""
apiVersion: v1
kind: Service
metadata:
  name: ml-soccer-api-service
spec:
  selector:
    app: ml-soccer-api
  ports:
  - port: 80
    targetPort: {self.config['services']['api_server']['port']}
  type: LoadBalancer
"""
        
        with open(os.path.join(k8s_dir, 'api-service.yaml'), 'w') as f:
            f.write(api_service)
    
    def _create_config_files(self, deployment_dir: str, environment: str):
        """Create configuration files"""
        # Create config directory
        config_dir = os.path.join(deployment_dir, 'config')
        os.makedirs(config_dir, exist_ok=True)
        
        # Environment configuration
        env_config = self.config['environments'][environment]
        config_content = {
            'environment': environment,
            'base_url': env_config['base_url'],
            'database_url': env_config['database_url'],
            'log_level': env_config['log_level'],
            'debug': env_config['debug'],
            'services': self.config['services'],
            'security': self.config['security'],
            'monitoring': self.config['monitoring']
        }
        
        with open(os.path.join(config_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config_content, f, default_flow_style=False)
        
        # Requirements file
        requirements_content = """
fastapi==0.104.1
uvicorn==0.24.0
pandas==2.1.3
numpy==1.24.3
scikit-learn==1.3.2
xgboost==2.0.2
lightgbm==4.1.0
joblib==1.3.2
pydantic==2.5.0
python-multipart==0.0.6
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
prometheus-client==0.19.0
"""
        
        with open(os.path.join(deployment_dir, 'requirements.txt'), 'w') as f:
            f.write(requirements_content)
    
    def _create_startup_scripts(self, deployment_dir: str, environment: str):
        """Create startup scripts"""
        # Create scripts directory
        scripts_dir = os.path.join(deployment_dir, 'scripts')
        os.makedirs(scripts_dir, exist_ok=True)
        
        # Deploy script
        deploy_script = f"""#!/bin/bash
set -e

echo "Starting deployment to {environment}..."

# Build Docker image
docker build -t ml-soccer-api:{environment} .

# Push to registry (if configured)
# docker push ml-soccer-api:{environment}

# Deploy to Kubernetes
kubectl apply -f k8s/

# Wait for deployment
kubectl rollout status deployment/ml-soccer-api

# Run health checks
kubectl get pods -l app=ml-soccer-api

echo "Deployment to {environment} completed successfully!"
"""
        
        with open(os.path.join(scripts_dir, 'deploy.sh'), 'w') as f:
            f.write(deploy_script)
        
        os.chmod(os.path.join(scripts_dir, 'deploy.sh'), 0o755)
        
        # Rollback script
        rollback_script = f"""#!/bin/bash
set -e

echo "Starting rollback for {environment}..."

# Rollback deployment
kubectl rollout undo deployment/ml-soccer-api

# Wait for rollback
kubectl rollout status deployment/ml-soccer-api

# Run health checks
kubectl get pods -l app=ml-soccer-api

echo "Rollback for {environment} completed successfully!"
"""
        
        with open(os.path.join(scripts_dir, 'rollback.sh'), 'w') as f:
            f.write(rollback_script)
        
        os.chmod(os.path.join(scripts_dir, 'rollback.sh'), 0o755)
    
    def deploy(self, environment: str, version: str, 
               model_files: List[str]) -> Dict[str, Any]:
        """Deploy to specified environment"""
        self.logger.info(f"Deploying version {version} to {environment}")
        
        try:
            # Prepare deployment
            preparation = self.prepare_deployment(environment, version, model_files)
            if not preparation['success']:
                return preparation
            
            deployment_dir = preparation['deployment_dir']
            
            # Create deployment record
            deployment_record = {
                'id': f"{environment}_{version}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'environment': environment,
                'version': version,
                'status': 'deploying',
                'start_time': datetime.now(),
                'deployment_dir': deployment_dir,
                'model_files': model_files
            }
            
            self.current_deployment = deployment_record
            self.deployment_history.append(deployment_record)
            
            # Execute deployment
            if environment == 'development':
                result = self._deploy_development(deployment_dir)
            elif environment == 'staging':
                result = self._deploy_staging(deployment_dir)
            elif environment == 'production':
                result = self._deploy_production(deployment_dir)
            else:
                return {'success': False, 'error': f'Unknown environment: {environment}'}
            
            # Update deployment record
            deployment_record['status'] = 'completed' if result['success'] else 'failed'
            deployment_record['end_time'] = datetime.now()
            deployment_record['result'] = result
            
            self.logger.info(f"Deployment {'completed' if result['success'] else 'failed'}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error during deployment: {e}")
            if self.current_deployment:
                self.current_deployment['status'] = 'failed'
                self.current_deployment['end_time'] = datetime.now()
                self.current_deployment['error'] = str(e)
            return {'success': False, 'error': str(e)}
    
    def _deploy_development(self, deployment_dir: str) -> Dict[str, Any]:
        """Deploy to development environment"""
        self.logger.info("Deploying to development environment")
        
        try:
            # Run docker-compose
            result = subprocess.run(
                ['docker-compose', 'up', '-d'],
                cwd=deployment_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {'success': False, 'error': result.stderr}
            
            # Wait for health check
            import time
            time.sleep(30)
            
            # Check health
            health_result = subprocess.run(
                ['curl', '-f', 'http://localhost:8000/health'],
                capture_output=True,
                text=True
            )
            
            if health_result.returncode != 0:
                return {'success': False, 'error': 'Health check failed'}
            
            return {'success': True, 'message': 'Development deployment successful'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _deploy_staging(self, deployment_dir: str) -> Dict[str, Any]:
        """Deploy to staging environment"""
        self.logger.info("Deploying to staging environment")
        
        try:
            # Deploy to Kubernetes
            result = subprocess.run(
                ['kubectl', 'apply', '-f', 'k8s/'],
                cwd=deployment_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {'success': False, 'error': result.stderr}
            
            # Wait for rollout
            rollout_result = subprocess.run(
                ['kubectl', 'rollout', 'status', 'deployment/ml-soccer-api'],
                capture_output=True,
                text=True
            )
            
            if rollout_result.returncode != 0:
                return {'success': False, 'error': 'Rollout failed'}
            
            return {'success': True, 'message': 'Staging deployment successful'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _deploy_production(self, deployment_dir: str) -> Dict[str, Any]:
        """Deploy to production environment"""
        self.logger.info("Deploying to production environment")
        
        try:
            # Blue-green deployment strategy
            if self.config['deployment']['strategy'] == 'blue_green':
                result = self._blue_green_deployment(deployment_dir)
            else:
                result = self._rolling_deployment(deployment_dir)
            
            return result
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _blue_green_deployment(self, deployment_dir: str) -> Dict[str, Any]:
        """Blue-green deployment strategy"""
        self.logger.info("Executing blue-green deployment")
        
        try:
            # Deploy to green environment
            result = subprocess.run(
                ['kubectl', 'apply', '-f', 'k8s/'],
                cwd=deployment_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {'success': False, 'error': result.stderr}
            
            # Wait for green deployment
            rollout_result = subprocess.run(
                ['kubectl', 'rollout', 'status', 'deployment/ml-soccer-api-green'],
                capture_output=True,
                text=True
            )
            
            if rollout_result.returncode != 0:
                return {'success': False, 'error': 'Green deployment failed'}
            
            # Switch traffic to green
            switch_result = subprocess.run(
                ['kubectl', 'patch', 'service', 'ml-soccer-api-service', 
                 '-p', '{"spec":{"selector":{"version":"green"}}}'],
                capture_output=True,
                text=True
            )
            
            if switch_result.returncode != 0:
                return {'success': False, 'error': 'Traffic switch failed'}
            
            return {'success': True, 'message': 'Blue-green deployment successful'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def _rolling_deployment(self, deployment_dir: str) -> Dict[str, Any]:
        """Rolling deployment strategy"""
        self.logger.info("Executing rolling deployment")
        
        try:
            # Deploy with rolling update
            result = subprocess.run(
                ['kubectl', 'apply', '-f', 'k8s/'],
                cwd=deployment_dir,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return {'success': False, 'error': result.stderr}
            
            # Wait for rollout
            rollout_result = subprocess.run(
                ['kubectl', 'rollout', 'status', 'deployment/ml-soccer-api'],
                capture_output=True,
                text=True
            )
            
            if rollout_result.returncode != 0:
                return {'success': False, 'error': 'Rolling deployment failed'}
            
            return {'success': True, 'message': 'Rolling deployment successful'}
            
        except Exception as e:
            return {'success': False, 'error': str(e)}
    
    def rollback(self, environment: str, version: str = None) -> Dict[str, Any]:
        """Rollback deployment"""
        self.logger.info(f"Rolling back {environment} deployment")
        
        try:
            if environment == 'development':
                result = subprocess.run(
                    ['docker-compose', 'down'],
                    capture_output=True,
                    text=True
                )
            else:
                result = subprocess.run(
                    ['kubectl', 'rollout', 'undo', 'deployment/ml-soccer-api'],
                    capture_output=True,
                    text=True
                )
            
            if result.returncode != 0:
                return {'success': False, 'error': result.stderr}
            
            # Wait for rollback
            if environment != 'development':
                rollout_result = subprocess.run(
                    ['kubectl', 'rollout', 'status', 'deployment/ml-soccer-api'],
                    capture_output=True,
                    text=True
                )
                
                if rollout_result.returncode != 0:
                    return {'success': False, 'error': 'Rollback failed'}
            
            self.logger.info(f"Rollback to {environment} completed successfully")
            return {'success': True, 'message': f'Rollback to {environment} successful'}
            
        except Exception as e:
            self.logger.error(f"Error during rollback: {e}")
            return {'success': False, 'error': str(e)}
    
    def health_check(self, environment: str) -> Dict[str, Any]:
        """Perform health check"""
        self.logger.info(f"Performing health check for {environment}")
        
        try:
            env_config = self.config['environments'][environment]
            base_url = env_config['base_url']
            
            # Check API health
            health_result = subprocess.run(
                ['curl', '-f', f'{base_url}/health'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if health_result.returncode != 0:
                return {'healthy': False, 'error': 'Health check failed'}
            
            # Check metrics endpoint
            metrics_result = subprocess.run(
                ['curl', '-f', f'{base_url}/metrics'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if metrics_result.returncode != 0:
                return {'healthy': False, 'error': 'Metrics endpoint failed'}
            
            return {'healthy': True, 'message': 'All health checks passed'}
            
        except Exception as e:
            self.logger.error(f"Error during health check: {e}")
            return {'healthy': False, 'error': str(e)}
    
    def get_deployment_status(self, environment: str = None) -> Dict[str, Any]:
        """Get deployment status"""
        if environment:
            deployments = [d for d in self.deployment_history if d['environment'] == environment]
        else:
            deployments = self.deployment_history
        
        return {
            'deployments': deployments,
            'current_deployment': self.current_deployment,
            'total_deployments': len(deployments)
        }
    
    def save_deployment_state(self, filepath: str):
        """Save deployment state"""
        self.logger.info(f"Saving deployment state to {filepath}")
        
        deployment_state = {
            'deployment_history': self.deployment_history,
            'current_deployment': self.current_deployment,
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(deployment_state, f, indent=2, default=str)
        
        self.logger.info("Deployment state saved successfully")

# Example usage
def main():
    """Example usage of NonMajorLeagueDeployment"""
    
    # Initialize deployment system
    deployer = NonMajorLeagueDeployment()
    
    # Deploy to development
    result = deployer.deploy(
        environment='development',
        version='1.0.0',
        model_files=['model.pkl', 'config.json']
    )
    
    print(f"Development deployment: {'Success' if result['success'] else 'Failed'}")
    
    # Health check
    health = deployer.health_check('development')
    print(f"Health check: {'Healthy' if health['healthy'] else 'Unhealthy'}")
    
    # Get deployment status
    status = deployer.get_deployment_status()
    print(f"Total deployments: {status['total_deployments']}")
    
    # Save deployment state
    deployer.save_deployment_state('deployment_state.json')

if __name__ == "__main__":
    main()
