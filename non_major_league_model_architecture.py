import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
import optuna
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeagueModelArchitecture:
    """
    Advanced model architecture for non-major soccer leagues
    
    Key Features:
    - Transfer learning from major leagues
    - Conservative ensemble approach
    - Robust validation for limited data
    - Probability calibration
    - Uncertainty quantification
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize model architecture
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.models = {}
        self.ensemble_model = None
        self.calibrated_models = {}
        self.feature_importance = {}
        self.class_weights = {}
        
    def setup_logging(self):
        """Setup logging for model architecture"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load model configuration"""
        if config is None:
            self.config = {
                'models': {
                    'xgboost': {
                        'enabled': True,
                        'params': {
                            'n_estimators': 200,
                            'max_depth': 6,
                            'learning_rate': 0.05,
                            'subsample': 0.8,
                            'colsample_bytree': 0.8,
                            'min_child_weight': 3,
                            'reg_alpha': 0.1,
                            'reg_lambda': 1.0,
                            'random_state': 42
                        }
                    },
                    'lightgbm': {
                        'enabled': True,
                        'params': {
                            'n_estimators': 200,
                            'max_depth': 6,
                            'learning_rate': 0.05,
                            'num_leaves': 50,
                            'feature_fraction': 0.8,
                            'bagging_fraction': 0.8,
                            'bagging_freq': 5,
                            'lambda_l1': 0.1,
                            'lambda_l2': 1.0,
                            'min_data_in_leaf': 20,
                            'random_state': 42,
                            'verbose': -1
                        }
                    },
                    'random_forest': {
                        'enabled': True,
                        'params': {
                            'n_estimators': 100,
                            'max_depth': 10,
                            'min_samples_split': 5,
                            'min_samples_leaf': 2,
                            'random_state': 42
                        }
                    },
                    'logistic_regression': {
                        'enabled': True,
                        'params': {
                            'max_iter': 1000,
                            'random_state': 42,
                            'C': 1.0
                        }
                    }
                },
                'ensemble': {
                    'method': 'voting',  # 'voting', 'stacking', 'weighted'
                    'voting_type': 'soft',
                    'weights': None,  # Will be calculated based on performance
                    'calibration': True
                },
                'transfer_learning': {
                    'enabled': True,
                    'source_leagues': ['EPL', 'LaLiga', 'Bundesliga'],
                    'transfer_ratio': 0.3,
                    'fine_tune_epochs': 50
                },
                'validation': {
                    'method': 'time_series_split',
                    'n_splits': 5,
                    'test_size': 0.2,
                    'calibration_method': 'isotonic'
                },
                'hyperparameter_tuning': {
                    'enabled': True,
                    'n_trials': 50,
                    'timeout': 3600,
                    'direction': 'minimize'
                },
                'class_imbalance': {
                    'enabled': True,
                    'method': 'class_weights',  # 'class_weights', 'smote', 'cost_sensitive'
                    'smote_threshold': 2.0,
                    'cost_matrix': {
                        'draw_misclassification_penalty': 1.5,  # Draws harder to predict
                        'home_away_confusion_penalty': 1.2
                    }
                }
            }
        else:
            self.config = config
    
    def create_base_models(self) -> Dict[str, Any]:
        """Create base models for non-major leagues"""
        self.logger.info("Creating base models for non-major leagues")
        
        base_models = {}
        
        # XGBoost (primary model)
        if self.config['models']['xgboost']['enabled']:
            base_models['xgboost'] = xgb.XGBClassifier(
                **self.config['models']['xgboost']['params'],
                eval_metric='mlogloss'
            )
        
        # LightGBM (secondary model)
        if self.config['models']['lightgbm']['enabled']:
            base_models['lightgbm'] = lgb.LGBMClassifier(
                **self.config['models']['lightgbm']['params'],
                objective='multiclass',
                num_class=3,
                metric='multi_logloss'
            )
        
        # Random Forest (robust model)
        if self.config['models']['random_forest']['enabled']:
            base_models['random_forest'] = RandomForestClassifier(
                **self.config['models']['random_forest']['params']
            )
        
        # Logistic Regression (baseline model)
        if self.config['models']['logistic_regression']['enabled']:
            base_models['logistic_regression'] = LogisticRegression(
                **self.config['models']['logistic_regression']['params']
            )
        
        self.models = base_models
        self.logger.info(f"Created {len(base_models)} base models")
        return base_models
    
    def _calculate_class_weights(self, y):
        """Calculate balanced class weights for imbalanced soccer outcomes"""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        
        # Apply conservative adjustment (reduce extreme weights)
        weights = weights ** 0.8  # Dampen extreme weights
        
        return dict(zip(classes, weights))
    
    def _apply_smote_if_needed(self, X_train, y_train):
        """Apply SMOTE only if severe imbalance exists"""
        try:
            from imblearn.over_sampling import SMOTE
            
            class_counts = pd.Series(y_train).value_counts()
            imbalance_ratio = class_counts.max() / class_counts.min()
            
            if imbalance_ratio > self.config['class_imbalance']['smote_threshold']:
                self.logger.info(f"Applying SMOTE due to imbalance ratio: {imbalance_ratio:.2f}")
                smote = SMOTE(sampling_strategy='not majority', 
                              random_state=42,
                              k_neighbors=3)  # Conservative
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                return X_resampled, y_resampled
            
            return X_train, y_train
        except ImportError:
            self.logger.warning("imblearn not available, skipping SMOTE")
            return X_train, y_train
    
    def train_base_models(self, X_train: pd.DataFrame, y_train: pd.Series, 
                         X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train base models with validation"""
        self.logger.info("Training base models")
        
        if not self.models:
            self.create_base_models()
        
        # Handle class imbalance
        if self.config['class_imbalance']['enabled']:
            method = self.config['class_imbalance']['method']
            
            if method == 'class_weights':
                self.class_weights = self._calculate_class_weights(y_train)
                self.logger.info(f"Calculated class weights: {self.class_weights}")
            elif method == 'smote':
                X_train, y_train = self._apply_smote_if_needed(X_train, y_train)
                self.logger.info(f"Applied SMOTE, new training size: {len(X_train)}")
        
        training_results = {}
        
        for name, model in self.models.items():
            self.logger.info(f"Training {name}...")
            
            try:
                if name == 'xgboost':
                    # Apply class weights if available
                    if self.class_weights:
                        sample_weights = np.array([self.class_weights[y_val] for y_val in y_train])
                        model.fit(
                            X_train, y_train,
                            sample_weight=sample_weights,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=50,
                            verbose=False
                        )
                    else:
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            early_stopping_rounds=50,
                            verbose=False
                        )
                elif name == 'lightgbm':
                    # Apply class weights if available
                    if self.class_weights:
                        sample_weights = np.array([self.class_weights[y_val] for y_val in y_train])
                        lgb_train = lgb.Dataset(X_train, y_train, weight=sample_weights)
                        model.fit(
                            X_train, y_train,
                            sample_weight=sample_weights,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                        )
                    else:
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_val, y_val)],
                            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
                        )
                elif name == 'random_forest':
                    # Use class_weight='balanced' for Random Forest
                    if self.class_weights:
                        model.set_params(class_weight='balanced')
                    model.fit(X_train, y_train)
                elif name == 'logistic_regression':
                    # Use class_weight='balanced' for Logistic Regression
                    if self.class_weights:
                        model.set_params(class_weight='balanced')
                    model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)
                
                # Evaluate model
                train_pred = model.predict(X_train)
                val_pred = model.predict(X_val)
                train_acc = accuracy_score(y_train, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                
                # Calculate log loss
                train_proba = model.predict_proba(X_train)
                val_proba = model.predict_proba(X_val)
                train_logloss = log_loss(y_train, train_proba)
                val_logloss = log_loss(y_val, val_proba)
                
                training_results[name] = {
                    'model': model,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'train_logloss': train_logloss,
                    'val_logloss': val_logloss,
                    'success': True
                }
                
                self.logger.info(f"{name} - Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                training_results[name] = {
                    'model': None,
                    'success': False,
                    'error': str(e)
                }
        
        return training_results
    
    def create_ensemble_model(self, training_results: Dict[str, Any]) -> Any:
        """Create ensemble model for non-major leagues"""
        self.logger.info("Creating ensemble model")
        
        # Filter successful models
        successful_models = {
            name: result['model'] for name, result in training_results.items()
            if result['success'] and result['model'] is not None
        }
        
        if len(successful_models) < 2:
            self.logger.warning("Not enough successful models for ensemble")
            return None
        
        ensemble_method = self.config['ensemble']['method']
        
        if ensemble_method == 'voting':
            ensemble_model = self._create_voting_ensemble(successful_models)
        elif ensemble_method == 'stacking':
            ensemble_model = self._create_stacking_ensemble(successful_models)
        elif ensemble_method == 'weighted':
            ensemble_model = self._create_weighted_ensemble(successful_models, training_results)
        else:
            self.logger.warning(f"Unknown ensemble method: {ensemble_method}")
            return None
        
        self.ensemble_model = ensemble_model
        return ensemble_model
    
    def _create_voting_ensemble(self, models: Dict[str, Any]) -> VotingClassifier:
        """Create voting ensemble"""
        voting_type = self.config['ensemble']['voting_type']
        
        estimators = [(name, model) for name, model in models.items()]
        
        ensemble = VotingClassifier(
            estimators=estimators,
            voting=voting_type
        )
        
        return ensemble
    
    def _create_stacking_ensemble(self, models: Dict[str, Any]) -> StackingClassifier:
        """Create stacking ensemble"""
        # Use top 3 models for base estimators
        base_estimators = list(models.items())[:3]
        
        # Use logistic regression as meta-estimator
        meta_estimator = LogisticRegression(max_iter=1000, random_state=42)
        
        ensemble = StackingClassifier(
            estimators=base_estimators,
            final_estimator=meta_estimator,
            cv=3  # Conservative CV for limited data
        )
        
        return ensemble
    
    def _create_weighted_ensemble(self, models: Dict[str, Any], 
                                 training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create weighted ensemble"""
        # Calculate weights based on validation performance
        weights = {}
        total_weight = 0
        
        for name, result in training_results.items():
            if result['success'] and name in models:
                # Use inverse log loss as weight (lower log loss = higher weight)
                weight = 1 / (result['val_logloss'] + 1e-10)
                weights[name] = weight
                total_weight += weight
        
        # Normalize weights
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        return {
            'models': models,
            'weights': weights,
            'type': 'weighted'
        }
    
    def train_ensemble_model(self, X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train ensemble model"""
        self.logger.info("Training ensemble model")
        
        if self.ensemble_model is None:
            self.logger.error("No ensemble model created")
            return None
        
        try:
            if isinstance(self.ensemble_model, (VotingClassifier, StackingClassifier)):
                self.ensemble_model.fit(X_train, y_train)
                
                # Evaluate ensemble
                train_pred = self.ensemble_model.predict(X_train)
                val_pred = self.ensemble_model.predict(X_val)
                train_acc = accuracy_score(y_train, train_pred)
                val_acc = accuracy_score(y_val, val_pred)
                
                train_proba = self.ensemble_model.predict_proba(X_train)
                val_proba = self.ensemble_model.predict_proba(X_val)
                train_logloss = log_loss(y_train, train_proba)
                val_logloss = log_loss(y_val, val_proba)
                
                ensemble_results = {
                    'model': self.ensemble_model,
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'train_logloss': train_logloss,
                    'val_logloss': val_logloss,
                    'success': True
                }
                
            elif isinstance(self.ensemble_model, dict) and self.ensemble_model['type'] == 'weighted':
                # For weighted ensemble, we don't train a single model
                # Instead, we use the individual models with weights
                ensemble_results = {
                    'model': self.ensemble_model,
                    'success': True,
                    'type': 'weighted'
                }
            
            self.logger.info(f"Ensemble training complete")
            return ensemble_results
            
        except Exception as e:
            self.logger.error(f"Error training ensemble: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def calibrate_models(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        """Calibrate model probabilities"""
        self.logger.info("Calibrating model probabilities")
        
        if not self.config['ensemble']['calibration']:
            return {}
        
        calibration_method = self.config['validation']['calibration_method']
        calibrated_models = {}
        
        # Calibrate individual models
        for name, model in self.models.items():
            if model is not None:
                try:
                    calibrated_model = CalibratedClassifierCV(
                        model, 
                        method=calibration_method, 
                        cv=3
                    )
                    calibrated_model.fit(X_train, y_train)
                    calibrated_models[name] = calibrated_model
                    self.logger.info(f"Calibrated {name}")
                except Exception as e:
                    self.logger.error(f"Error calibrating {name}: {e}")
        
        # Calibrate ensemble if it's a single model
        if isinstance(self.ensemble_model, (VotingClassifier, StackingClassifier)):
            try:
                calibrated_ensemble = CalibratedClassifierCV(
                    self.ensemble_model,
                    method=calibration_method,
                    cv=3
                )
                calibrated_ensemble.fit(X_train, y_train)
                calibrated_models['ensemble'] = calibrated_ensemble
                self.logger.info("Calibrated ensemble model")
            except Exception as e:
                self.logger.error(f"Error calibrating ensemble: {e}")
        
        self.calibrated_models = calibrated_models
        return calibrated_models
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Hyperparameter tuning for non-major leagues"""
        self.logger.info("Starting hyperparameter tuning")
        
        if not self.config['hyperparameter_tuning']['enabled']:
            return {}
        
        n_trials = self.config['hyperparameter_tuning']['n_trials']
        timeout = self.config['hyperparameter_tuning']['timeout']
        
        def objective(trial):
            # XGBoost parameters
            xgb_params = {
                'n_estimators': trial.suggest_int('xgb_n_estimators', 100, 300),
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 8),
                'learning_rate': trial.suggest_float('xgb_learning_rate', 0.01, 0.1),
                'subsample': trial.suggest_float('xgb_subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('xgb_colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('xgb_min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('xgb_reg_alpha', 0.0, 1.0),
                'reg_lambda': trial.suggest_float('xgb_reg_lambda', 0.0, 2.0),
                'random_state': 42
            }
            
            # Train XGBoost model
            model = xgb.XGBClassifier(**xgb_params, eval_metric='mlogloss')
            model.fit(X_train, y_train)
            
            # Evaluate
            val_proba = model.predict_proba(X_val)
            val_logloss = log_loss(y_val, val_proba)
            
            return val_logloss
        
        try:
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=n_trials, timeout=timeout)
            
            best_params = study.best_params
            best_score = study.best_value
            
            self.logger.info(f"Best hyperparameters: {best_params}")
            self.logger.info(f"Best score: {best_score:.4f}")
            
            # Update XGBoost parameters
            if 'xgb_' in str(best_params):
                xgb_params = {k.replace('xgb_', ''): v for k, v in best_params.items() if k.startswith('xgb_')}
                self.config['models']['xgboost']['params'].update(xgb_params)
            
            return {
                'best_params': best_params,
                'best_score': best_score,
                'study': study
            }
            
        except Exception as e:
            self.logger.error(f"Hyperparameter tuning failed: {e}")
            return {'error': str(e)}
    
    def transfer_learning_setup(self, source_data: pd.DataFrame, 
                              target_data: pd.DataFrame) -> Dict[str, Any]:
        """Setup transfer learning from major leagues"""
        self.logger.info("Setting up transfer learning")
        
        if not self.config['transfer_learning']['enabled']:
            return {}
        
        transfer_ratio = self.config['transfer_learning']['transfer_ratio']
        
        # This is a simplified transfer learning setup
        # In practice, you would implement more sophisticated transfer learning
        
        transfer_results = {
            'source_data_shape': source_data.shape,
            'target_data_shape': target_data.shape,
            'transfer_ratio': transfer_ratio,
            'enabled': True
        }
        
        self.logger.info("Transfer learning setup complete")
        return transfer_results
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate all models"""
        self.logger.info("Evaluating models")
        
        evaluation_results = {}
        
        # Evaluate individual models
        for name, model in self.models.items():
            if model is not None:
                try:
                    test_pred = model.predict(X_test)
                    test_proba = model.predict_proba(X_test)
                    
                    test_acc = accuracy_score(y_test, test_pred)
                    test_logloss = log_loss(y_test, test_proba)
                    
                    evaluation_results[name] = {
                        'accuracy': test_acc,
                        'log_loss': test_logloss,
                        'predictions': test_pred,
                        'probabilities': test_proba
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating {name}: {e}")
                    evaluation_results[name] = {'error': str(e)}
        
        # Evaluate ensemble
        if self.ensemble_model is not None:
            try:
                if isinstance(self.ensemble_model, (VotingClassifier, StackingClassifier)):
                    test_pred = self.ensemble_model.predict(X_test)
                    test_proba = self.ensemble_model.predict_proba(X_test)
                    
                    test_acc = accuracy_score(y_test, test_pred)
                    test_logloss = log_loss(y_test, test_proba)
                    
                    evaluation_results['ensemble'] = {
                        'accuracy': test_acc,
                        'log_loss': test_logloss,
                        'predictions': test_pred,
                        'probabilities': test_proba
                    }
                
                elif isinstance(self.ensemble_model, dict) and self.ensemble_model['type'] == 'weighted':
                    # Evaluate weighted ensemble
                    models = self.ensemble_model['models']
                    weights = self.ensemble_model['weights']
                    
                    # Calculate weighted predictions
                    weighted_proba = np.zeros((len(X_test), 3))
                    
                    for name, model in models.items():
                        if name in weights:
                            model_proba = model.predict_proba(X_test)
                            weighted_proba += weights[name] * model_proba
                    
                    test_pred = np.argmax(weighted_proba, axis=1)
                    test_acc = accuracy_score(y_test, test_pred)
                    test_logloss = log_loss(y_test, weighted_proba)
                    
                    evaluation_results['ensemble'] = {
                        'accuracy': test_acc,
                        'log_loss': test_logloss,
                        'predictions': test_pred,
                        'probabilities': weighted_proba
                    }
                
            except Exception as e:
                self.logger.error(f"Error evaluating ensemble: {e}")
                evaluation_results['ensemble'] = {'error': str(e)}
        
        # Evaluate calibrated models
        for name, model in self.calibrated_models.items():
            try:
                test_pred = model.predict(X_test)
                test_proba = model.predict_proba(X_test)
                
                test_acc = accuracy_score(y_test, test_pred)
                test_logloss = log_loss(y_test, test_proba)
                
                evaluation_results[f'{name}_calibrated'] = {
                    'accuracy': test_acc,
                    'log_loss': test_logloss,
                    'predictions': test_pred,
                    'probabilities': test_proba
                }
                
            except Exception as e:
                self.logger.error(f"Error evaluating calibrated {name}: {e}")
                evaluation_results[f'{name}_calibrated'] = {'error': str(e)}
        
        return evaluation_results
    
    def get_feature_importance(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Get feature importance from all models"""
        self.logger.info("Calculating feature importance")
        
        feature_importance = {}
        
        for name, model in self.models.items():
            if model is not None and hasattr(model, 'feature_importances_'):
                try:
                    importance_df = pd.DataFrame({
                        'feature': X.columns,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    feature_importance[name] = importance_df
                    
                except Exception as e:
                    self.logger.error(f"Error getting feature importance for {name}: {e}")
        
        self.feature_importance = feature_importance
        return feature_importance
    
    def save_models(self, filepath: str):
        """Save all trained models"""
        self.logger.info(f"Saving models to {filepath}")
        
        model_state = {
            'models': self.models,
            'ensemble_model': self.ensemble_model,
            'calibrated_models': self.calibrated_models,
            'feature_importance': self.feature_importance,
            'config': self.config
        }
        
        joblib.dump(model_state, filepath)
        self.logger.info("Models saved successfully")
    
    def load_models(self, filepath: str):
        """Load trained models"""
        self.logger.info(f"Loading models from {filepath}")
        
        model_state = joblib.load(filepath)
        self.models = model_state['models']
        self.ensemble_model = model_state['ensemble_model']
        self.calibrated_models = model_state['calibrated_models']
        self.feature_importance = model_state['feature_importance']
        self.config = model_state['config']
        
        self.logger.info("Models loaded successfully")
    
    def predict(self, X: pd.DataFrame, use_calibrated: bool = True) -> Dict[str, Any]:
        """Make predictions using all models"""
        self.logger.info("Making predictions")
        
        predictions = {}
        
        # Individual model predictions
        for name, model in self.models.items():
            if model is not None:
                try:
                    pred = model.predict(X)
                    proba = model.predict_proba(X)
                    
                    predictions[name] = {
                        'predictions': pred,
                        'probabilities': proba
                    }
                except Exception as e:
                    self.logger.error(f"Error predicting with {name}: {e}")
        
        # Ensemble predictions
        if self.ensemble_model is not None:
            try:
                if isinstance(self.ensemble_model, (VotingClassifier, StackingClassifier)):
                    pred = self.ensemble_model.predict(X)
                    proba = self.ensemble_model.predict_proba(X)
                    
                    predictions['ensemble'] = {
                        'predictions': pred,
                        'probabilities': proba
                    }
                
                elif isinstance(self.ensemble_model, dict) and self.ensemble_model['type'] == 'weighted':
                    models = self.ensemble_model['models']
                    weights = self.ensemble_model['weights']
                    
                    weighted_proba = np.zeros((len(X), 3))
                    
                    for name, model in models.items():
                        if name in weights:
                            model_proba = model.predict_proba(X)
                            weighted_proba += weights[name] * model_proba
                    
                    pred = np.argmax(weighted_proba, axis=1)
                    
                    predictions['ensemble'] = {
                        'predictions': pred,
                        'probabilities': weighted_proba
                    }
                
            except Exception as e:
                self.logger.error(f"Error predicting with ensemble: {e}")
        
        # Calibrated predictions
        if use_calibrated:
            for name, model in self.calibrated_models.items():
                try:
                    pred = model.predict(X)
                    proba = model.predict_proba(X)
                    
                    predictions[f'{name}_calibrated'] = {
                        'predictions': pred,
                        'probabilities': proba
                    }
                except Exception as e:
                    self.logger.error(f"Error predicting with calibrated {name}: {e}")
        
        return predictions

# Example usage
def main():
    """Example usage of NonMajorLeagueModelArchitecture"""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.choice([0, 1, 2], n_samples))
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    X_train, X_val = X_train[:int(0.8 * len(X_train))], X_train[int(0.8 * len(X_train)):]
    y_train, y_val = y_train[:int(0.8 * len(y_train))], y_train[int(0.8 * len(y_train)):]
    
    # Initialize model architecture
    model_arch = NonMajorLeagueModelArchitecture()
    
    # Create and train base models
    base_models = model_arch.create_base_models()
    training_results = model_arch.train_base_models(X_train, y_train, X_val, y_val)
    
    # Create and train ensemble
    ensemble_model = model_arch.create_ensemble_model(training_results)
    ensemble_results = model_arch.train_ensemble_model(X_train, y_train, X_val, y_val)
    
    # Calibrate models
    calibrated_models = model_arch.calibrate_models(X_train, y_train)
    
    # Evaluate models
    evaluation_results = model_arch.evaluate_models(X_test, y_test)
    
    # Get feature importance
    feature_importance = model_arch.get_feature_importance(X_train)
    
    # Print results
    print("Model Evaluation Results:")
    for name, results in evaluation_results.items():
        if 'error' not in results:
            print(f"{name}: Accuracy = {results['accuracy']:.4f}, Log Loss = {results['log_loss']:.4f}")
    
    # Save models
    model_arch.save_models('non_major_league_models.pkl')

if __name__ == "__main__":
    main()
