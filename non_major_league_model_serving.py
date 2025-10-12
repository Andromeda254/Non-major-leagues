import pandas as pd
import numpy as np
import joblib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# FastAPI imports
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# For authentication (simplified)
import jwt
from passlib.context import CryptContext

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Single prediction request model"""
    model_name: str = Field(..., description="Name of the model to use")
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    league: str = Field(..., description="League name")
    home_team_elo: Optional[float] = Field(None, description="Home team Elo rating")
    away_team_elo: Optional[float] = Field(None, description="Away team Elo rating")
    home_form: Optional[float] = Field(None, description="Home team form")
    away_form: Optional[float] = Field(None, description="Away team form")
    h2h_home_wins: Optional[int] = Field(None, description="Head-to-head home wins")
    h2h_draws: Optional[int] = Field(None, description="Head-to-head draws")
    h2h_away_wins: Optional[int] = Field(None, description="Head-to-head away wins")
    home_odds: Optional[float] = Field(None, description="Home team odds")
    draw_odds: Optional[float] = Field(None, description="Draw odds")
    away_odds: Optional[float] = Field(None, description="Away team odds")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""
    model_name: str = Field(..., description="Name of the model to use")
    predictions: List[PredictionRequest] = Field(..., description="List of prediction requests")

class NonMajorLeagueModelServing:
    """
    Model serving API for non-major soccer leagues
    
    Key Features:
    - RESTful API endpoints
    - Model prediction services
    - Authentication and authorization
    - Request validation
    - Response formatting
    - Error handling
    - Performance monitoring
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize model serving system
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.models = {}
        self.model_metadata = {}
        self.request_history = []
        self.performance_stats = {}
        
        # Initialize FastAPI app
        self.app = FastAPI(
            title="Non-Major League ML API",
            description="Machine Learning API for Non-Major Soccer Leagues",
            version="1.0.0"
        )
        
        # Setup middleware
        self._setup_middleware()
        
        # Setup routes
        self._setup_routes()
        
        # Initialize authentication
        self._setup_authentication()
        
    def setup_logging(self):
        """Setup logging for model serving"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load model serving configuration"""
        if config is None:
            self.config = {
                'api': {
                    'host': '0.0.0.0',
                    'port': 8000,
                    'debug': False,
                    'reload': False
                },
                'models': {
                    'xgboost': {
                        'enabled': True,
                        'model_path': './models/xgboost_model.pkl',
                        'version': '1.0.0',
                        'description': 'XGBoost model for match predictions'
                    },
                    'lightgbm': {
                        'enabled': True,
                        'model_path': './models/lightgbm_model.pkl',
                        'version': '1.0.0',
                        'description': 'LightGBM model for match predictions'
                    },
                    'ensemble': {
                        'enabled': True,
                        'model_path': './models/ensemble_model.pkl',
                        'version': '1.0.0',
                        'description': 'Ensemble model combining multiple algorithms'
                    }
                },
                'authentication': {
                    'enabled': True,
                    'secret_key': 'your-secret-key-here',
                    'algorithm': 'HS256',
                    'token_expiry': 3600,  # 1 hour
                    'users': {
                        'admin': {
                            'password': 'admin123',
                            'role': 'admin',
                            'permissions': ['read', 'write', 'predict', 'admin']
                        },
                        'user': {
                            'password': 'user123',
                            'role': 'user',
                            'permissions': ['read', 'predict']
                        },
                        'viewer': {
                            'password': 'viewer123',
                            'role': 'viewer',
                            'permissions': ['read']
                        }
                    }
                },
                'prediction': {
                    'max_batch_size': 100,
                    'timeout': 30,
                    'confidence_threshold': 0.6,
                    'cache_predictions': True,
                    'cache_ttl': 3600  # 1 hour
                },
                'monitoring': {
                    'enabled': True,
                    'metrics_endpoint': '/metrics',
                    'health_endpoint': '/health',
                    'log_requests': True,
                    'performance_tracking': True
                },
                'cors': {
                    'enabled': True,
                    'origins': ['*'],
                    'methods': ['GET', 'POST'],
                    'headers': ['*']
                }
            }
        else:
            self.config = config
    
    def _setup_middleware(self):
        """Setup FastAPI middleware"""
        # CORS middleware
        if self.config['cors']['enabled']:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config['cors']['origins'],
                allow_credentials=True,
                allow_methods=self.config['cors']['methods'],
                allow_headers=self.config['cors']['headers']
            )
    
    def _setup_authentication(self):
        """Setup authentication system"""
        if self.config['authentication']['enabled']:
            self.security = HTTPBearer()
            self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
            self.secret_key = self.config['authentication']['secret_key']
            self.algorithm = self.config['authentication']['algorithm']
        else:
            self.security = None
    
    def _setup_routes(self):
        """Setup API routes"""
        
        # Health check endpoint
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "version": "1.0.0"
            }
        
        # Metrics endpoint
        @self.app.get("/metrics")
        async def get_metrics():
            """Get API metrics"""
            return {
                "total_requests": len(self.request_history),
                "active_models": len(self.models),
                "performance_stats": self.performance_stats,
                "timestamp": datetime.now().isoformat()
            }
        
        # Model info endpoint
        @self.app.get("/models")
        async def get_models():
            """Get available models"""
            return {
                "models": self.model_metadata,
                "timestamp": datetime.now().isoformat()
            }
        
        # Single prediction endpoint
        @self.app.post("/predict")
        async def predict_single(
            request: PredictionRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self._get_current_user) if self.config['authentication']['enabled'] else None
        ):
            """Single prediction endpoint"""
            try:
                # Validate request
                if not self._validate_prediction_request(request):
                    raise HTTPException(status_code=400, detail="Invalid request data")
                
                # Make prediction
                prediction_result = await self._make_prediction(request)
                
                # Log request
                if self.config['monitoring']['log_requests']:
                    background_tasks.add_task(self._log_request, request, prediction_result)
                
                return prediction_result
                
            except Exception as e:
                self.logger.error(f"Error in single prediction: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Batch prediction endpoint
        @self.app.post("/predict/batch")
        async def predict_batch(
            request: BatchPredictionRequest,
            background_tasks: BackgroundTasks,
            credentials: HTTPAuthorizationCredentials = Depends(self._get_current_user) if self.config['authentication']['enabled'] else None
        ):
            """Batch prediction endpoint"""
            try:
                # Validate request
                if not self._validate_batch_request(request):
                    raise HTTPException(status_code=400, detail="Invalid batch request")
                
                # Make batch predictions
                batch_results = await self._make_batch_predictions(request)
                
                # Log request
                if self.config['monitoring']['log_requests']:
                    background_tasks.add_task(self._log_batch_request, request, batch_results)
                
                return batch_results
                
            except Exception as e:
                self.logger.error(f"Error in batch prediction: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        # Model management endpoints
        @self.app.post("/models/{model_name}/load")
        async def load_model(
            model_name: str,
            credentials: HTTPAuthorizationCredentials = Depends(self._get_current_user) if self.config['authentication']['enabled'] else None
        ):
            """Load a model"""
            try:
                if not self._check_permission(credentials, 'admin'):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                result = await self._load_model(model_name)
                return result
                
            except Exception as e:
                self.logger.error(f"Error loading model {model_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.post("/models/{model_name}/unload")
        async def unload_model(
            model_name: str,
            credentials: HTTPAuthorizationCredentials = Depends(self._get_current_user) if self.config['authentication']['enabled'] else None
        ):
            """Unload a model"""
            try:
                if not self._check_permission(credentials, 'admin'):
                    raise HTTPException(status_code=403, detail="Insufficient permissions")
                
                result = await self._unload_model(model_name)
                return result
                
            except Exception as e:
                self.logger.error(f"Error unloading model {model_name}: {e}")
                raise HTTPException(status_code=500, detail=str(e))
    
    def _get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Get current user from JWT token"""
        try:
            token = credentials.credentials
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username = payload.get("sub")
            
            if username is None:
                raise HTTPException(status_code=401, detail="Invalid token")
            
            return username
            
        except jwt.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token")
    
    def _check_permission(self, credentials: HTTPAuthorizationCredentials, required_permission: str) -> bool:
        """Check if user has required permission"""
        if not self.config['authentication']['enabled']:
            return True
        
        try:
            token = credentials.credentials
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            username = payload.get("sub")
            
            if username not in self.config['authentication']['users']:
                return False
            
            user_permissions = self.config['authentication']['users'][username]['permissions']
            return required_permission in user_permissions
            
        except jwt.PyJWTError:
            return False
    
    def _validate_prediction_request(self, request: PredictionRequest) -> bool:
        """Validate prediction request"""
        try:
            # Check required fields
            required_fields = ['home_team', 'away_team', 'league']
            for field in required_fields:
                if not hasattr(request, field) or getattr(request, field) is None:
                    return False
            
            # Check model exists
            if request.model_name not in self.models:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating prediction request: {e}")
            return False
    
    def _validate_batch_request(self, request: BatchPredictionRequest) -> bool:
        """Validate batch prediction request"""
        try:
            # Check batch size
            if len(request.requests) > self.config['prediction']['max_batch_size']:
                return False
            
            # Validate each request
            for req in request.requests:
                if not self._validate_prediction_request(req):
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating batch request: {e}")
            return False
    
    async def _make_prediction(self, request: PredictionRequest) -> Dict[str, Any]:
        """Make single prediction"""
        try:
            start_time = datetime.now()
            
            # Get model
            model = self.models.get(request.model_name)
            if model is None:
                raise HTTPException(status_code=404, detail=f"Model {request.model_name} not found")
            
            # Prepare features
            features = self._prepare_features(request)
            
            # Make prediction
            prediction = model.predict_proba(features)
            
            # Format response
            result = {
                "prediction_id": f"pred_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "model_name": request.model_name,
                "home_team": request.home_team,
                "away_team": request.away_team,
                "league": request.league,
                "predictions": {
                    "home_win": float(prediction[0][2]),
                    "draw": float(prediction[0][1]),
                    "away_win": float(prediction[0][0])
                },
                "confidence": float(np.max(prediction[0])),
                "recommended_bet": self._get_recommended_bet(prediction[0]),
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
            # Update performance stats
            self._update_performance_stats(start_time, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            raise
    
    async def _make_batch_predictions(self, request: BatchPredictionRequest) -> Dict[str, Any]:
        """Make batch predictions"""
        try:
            start_time = datetime.now()
            results = []
            
            for req in request.requests:
                try:
                    result = await self._make_prediction(req)
                    results.append(result)
                except Exception as e:
                    self.logger.error(f"Error in batch prediction item: {e}")
                    results.append({
                        "error": str(e),
                        "home_team": req.home_team,
                        "away_team": req.away_team,
                        "league": req.league
                    })
            
            return {
                "batch_id": f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                "total_requests": len(request.requests),
                "successful_predictions": len([r for r in results if 'error' not in r]),
                "failed_predictions": len([r for r in results if 'error' in r]),
                "results": results,
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error making batch predictions: {e}")
            raise
    
    def _prepare_features(self, request: PredictionRequest) -> np.ndarray:
        """Prepare features for prediction"""
        try:
            # This is a simplified feature preparation
            # In a real system, this would use the feature engineering pipeline
            
            features = np.array([
                request.home_team_elo or 1500,
                request.away_team_elo or 1500,
                request.home_form or 0.5,
                request.away_form or 0.5,
                request.h2h_home_wins or 0,
                request.h2h_draws or 0,
                request.h2h_away_wins or 0,
                request.home_odds or 2.0,
                request.draw_odds or 3.0,
                request.away_odds or 2.0
            ]).reshape(1, -1)
            
            return features
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            raise
    
    def _get_recommended_bet(self, prediction: np.ndarray) -> Dict[str, Any]:
        """Get recommended bet based on prediction"""
        try:
            # Find highest probability outcome
            outcome_idx = np.argmax(prediction)
            outcome_names = ['away_win', 'draw', 'home_win']
            outcome = outcome_names[outcome_idx]
            probability = prediction[outcome_idx]
            
            # Check confidence threshold
            if probability < self.config['prediction']['confidence_threshold']:
                return {
                    "recommendation": "no_bet",
                    "reason": "low_confidence",
                    "confidence": probability
                }
            
            # Calculate Kelly fraction (simplified)
            kelly_fraction = max(0, (probability * 2 - 1) * 0.1)  # Conservative Kelly
            
            return {
                "recommendation": outcome,
                "confidence": probability,
                "kelly_fraction": kelly_fraction,
                "bet_size": "small" if kelly_fraction < 0.02 else "medium" if kelly_fraction < 0.05 else "large"
            }
            
        except Exception as e:
            self.logger.error(f"Error getting recommended bet: {e}")
            return {
                "recommendation": "no_bet",
                "reason": "error",
                "confidence": 0.0
            }
    
    def _update_performance_stats(self, start_time: datetime, result: Dict[str, Any]):
        """Update performance statistics"""
        try:
            processing_time = result['processing_time']
            
            if 'prediction_times' not in self.performance_stats:
                self.performance_stats['prediction_times'] = []
            
            self.performance_stats['prediction_times'].append(processing_time)
            
            # Keep only last 1000 predictions
            if len(self.performance_stats['prediction_times']) > 1000:
                self.performance_stats['prediction_times'] = self.performance_stats['prediction_times'][-1000:]
            
            # Update averages
            self.performance_stats['avg_processing_time'] = np.mean(self.performance_stats['prediction_times'])
            self.performance_stats['max_processing_time'] = np.max(self.performance_stats['prediction_times'])
            self.performance_stats['min_processing_time'] = np.min(self.performance_stats['prediction_times'])
            
        except Exception as e:
            self.logger.error(f"Error updating performance stats: {e}")
    
    def _log_request(self, request: PredictionRequest, result: Dict[str, Any]):
        """Log prediction request"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "single_prediction",
                "request": request.dict(),
                "result": result
            }
            
            self.request_history.append(log_entry)
            
            # Keep only last 10000 requests
            if len(self.request_history) > 10000:
                self.request_history = self.request_history[-10000:]
            
        except Exception as e:
            self.logger.error(f"Error logging request: {e}")
    
    def _log_batch_request(self, request: BatchPredictionRequest, result: Dict[str, Any]):
        """Log batch prediction request"""
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "type": "batch_prediction",
                "request": request.dict(),
                "result": result
            }
            
            self.request_history.append(log_entry)
            
            # Keep only last 10000 requests
            if len(self.request_history) > 10000:
                self.request_history = self.request_history[-10000:]
            
        except Exception as e:
            self.logger.error(f"Error logging batch request: {e}")
    
    async def _load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a model"""
        try:
            if model_name not in self.config['models']:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not configured")
            
            model_config = self.config['models'][model_name]
            
            if not model_config['enabled']:
                raise HTTPException(status_code=400, detail=f"Model {model_name} is disabled")
            
            # Load model
            model = joblib.load(model_config['model_path'])
            self.models[model_name] = model
            
            # Update metadata
            self.model_metadata[model_name] = {
                "loaded": True,
                "version": model_config['version'],
                "description": model_config['description'],
                "loaded_at": datetime.now().isoformat()
            }
            
            self.logger.info(f"Model {model_name} loaded successfully")
            
            return {
                "model_name": model_name,
                "status": "loaded",
                "version": model_config['version'],
                "loaded_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error loading model {model_name}: {e}")
            raise
    
    async def _unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a model"""
        try:
            if model_name not in self.models:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not loaded")
            
            # Remove model
            del self.models[model_name]
            
            # Update metadata
            if model_name in self.model_metadata:
                self.model_metadata[model_name]["loaded"] = False
                self.model_metadata[model_name]["unloaded_at"] = datetime.now().isoformat()
            
            self.logger.info(f"Model {model_name} unloaded successfully")
            
            return {
                "model_name": model_name,
                "status": "unloaded",
                "unloaded_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error unloading model {model_name}: {e}")
            raise
    
    def load_all_models(self):
        """Load all enabled models"""
        self.logger.info("Loading all enabled models")
        
        for model_name, model_config in self.config['models'].items():
            if model_config['enabled']:
                try:
                    # Load model
                    model = joblib.load(model_config['model_path'])
                    self.models[model_name] = model
                    
                    # Update metadata
                    self.model_metadata[model_name] = {
                        "loaded": True,
                        "version": model_config['version'],
                        "description": model_config['description'],
                        "loaded_at": datetime.now().isoformat()
                    }
                    
                    self.logger.info(f"Model {model_name} loaded successfully")
                    
                except Exception as e:
                    self.logger.error(f"Error loading model {model_name}: {e}")
                    self.model_metadata[model_name] = {
                        "loaded": False,
                        "error": str(e),
                        "loaded_at": datetime.now().isoformat()
                    }
    
    def start_server(self):
        """Start the FastAPI server"""
        self.logger.info("Starting model serving API")
        
        # Load all models
        self.load_all_models()
        
        # Start server
        uvicorn.run(
            self.app,
            host=self.config['api']['host'],
            port=self.config['api']['port'],
            debug=self.config['api']['debug'],
            reload=self.config['api']['reload']
        )
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return {
            "status": "running",
            "models_loaded": len(self.models),
            "total_requests": len(self.request_history),
            "performance_stats": self.performance_stats,
            "config": self.config
        }

# Pydantic models for request/response validation
class PredictionRequest(BaseModel):
    """Single prediction request model"""
    model_name: str = Field(..., description="Name of the model to use")
    home_team: str = Field(..., description="Home team name")
    away_team: str = Field(..., description="Away team name")
    league: str = Field(..., description="League name")
    home_team_elo: Optional[float] = Field(None, description="Home team Elo rating")
    away_team_elo: Optional[float] = Field(None, description="Away team Elo rating")
    home_form: Optional[float] = Field(None, description="Home team form")
    away_form: Optional[float] = Field(None, description="Away team form")
    h2h_home_wins: Optional[int] = Field(None, description="Head-to-head home wins")
    h2h_draws: Optional[int] = Field(None, description="Head-to-head draws")
    h2h_away_wins: Optional[int] = Field(None, description="Head-to-head away wins")
    home_odds: Optional[float] = Field(None, description="Home team odds")
    draw_odds: Optional[float] = Field(None, description="Draw odds")
    away_odds: Optional[float] = Field(None, description="Away team odds")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request model"""
    requests: List[PredictionRequest] = Field(..., description="List of prediction requests")

# Example usage
def main():
    """Example usage of NonMajorLeagueModelServing"""
    
    # Initialize model serving
    server = NonMajorLeagueModelServing()
    
    # Start server
    server.start_server()

if __name__ == "__main__":
    main()
