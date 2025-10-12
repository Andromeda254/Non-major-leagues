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

class NonMajorLeagueEnsemble:
    """
    Conservative ensemble approach for non-major soccer leagues
    
    Key Features:
    - Conservative model selection and weighting
    - Robust validation for limited data
    - Uncertainty quantification
    - Probability calibration
    - Cross-validation with time series splits
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize ensemble for non-major leagues
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.base_models = {}
        self.ensemble_model = None
        self.calibrated_models = {}
        self.performance_metrics = {}
        self.uncertainty_estimates = {}
        
    def setup_logging(self):
        """Setup logging for ensemble"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load ensemble configuration"""
        if config is None:
            self.config = {
                'ensemble_strategy': {
                    'method': 'conservative_weighted',  # 'voting', 'stacking', 'conservative_weighted'
                    'min_models': 2,
                    'max_models': 4,
                    'confidence_threshold': 0.6,
                    'uncertainty_threshold': 0.3
                },
                'ensemble_diversity': {
                    'feature_subsets': True,
                    'temporal_windows': True,
                    'algorithm_mix': True,
                    'stacking': True,
                    'dynamic_weighting': True,
                    'diversity_bonus': 0.05
                },
                'base_models': {
                    'xgboost': {
                        'enabled': True,
                        'weight': 0.4,
                        'params': {
                            'n_estimators': 150,
                            'max_depth': 5,
                            'learning_rate': 0.03,
                            'subsample': 0.8,
                            'colsample_bytree': 0.8,
                            'min_child_weight': 5,
                            'reg_alpha': 0.2,
                            'reg_lambda': 1.5,
                            'random_state': 42
                        }
                    },
                    'lightgbm': {
                        'enabled': True,
                        'weight': 0.35,
                        'params': {
                            'n_estimators': 150,
                            'max_depth': 5,
                            'learning_rate': 0.03,
                            'num_leaves': 30,
                            'feature_fraction': 0.8,
                            'bagging_fraction': 0.8,
                            'bagging_freq': 5,
                            'lambda_l1': 0.2,
                            'lambda_l2': 1.5,
                            'min_data_in_leaf': 10,
                            'random_state': 42,
                            'verbose': -1
                        }
                    },
                    'random_forest': {
                        'enabled': True,
                        'weight': 0.15,
                        'params': {
                            'n_estimators': 100,
                            'max_depth': 8,
                            'min_samples_split': 10,
                            'min_samples_leaf': 5,
                            'random_state': 42
                        }
                    },
                    'logistic_regression': {
                        'enabled': True,
                        'weight': 0.1,
                        'params': {
                            'max_iter': 1000,
                            'random_state': 42,
                            'C': 0.1  # More regularization for limited data
                        }
                    },
                    'gradient_boosting': {
                        'enabled': True,
                        'weight': 0.15,
                        'params': {
                            'n_estimators': 100,
                            'learning_rate': 0.05,
                            'max_depth': 5,
                            'random_state': 42
                        }
                    },
                    'extra_trees': {
                        'enabled': True,
                        'weight': 0.10,
                        'params': {
                            'n_estimators': 100,
                            'max_depth': 8,
                            'min_samples_split': 10,
                            'random_state': 42
                        }
                    }
                },
                'validation': {
                    'method': 'time_series_split',
                    'n_splits': 3,  # Conservative for limited data
                    'test_size': 0.2,
                    'calibration_method': 'isotonic'
                },
                'uncertainty': {
                    'enabled': True,
                    'method': 'bootstrap',
                    'n_bootstrap': 50,
                    'confidence_level': 0.95
                },
                'conservative_settings': {
                    'min_training_samples': 100,
                    'max_overfitting_threshold': 0.1,
                    'min_validation_accuracy': 0.45,
                    'max_log_loss': 1.5
                }
            }
        else:
            self.config = config
    
    def create_base_models(self) -> Dict[str, Any]:
        """Create base models with conservative parameters"""
        self.logger.info("Creating base models with conservative parameters")
        
        base_models = {}
        
        # XGBoost (primary model - most weight)
        if self.config['base_models']['xgboost']['enabled']:
            base_models['xgboost'] = xgb.XGBClassifier(
                **self.config['base_models']['xgboost']['params'],
                eval_metric='mlogloss'
            )
        
        # LightGBM (secondary model)
        if self.config['base_models']['lightgbm']['enabled']:
            base_models['lightgbm'] = lgb.LGBMClassifier(
                **self.config['base_models']['lightgbm']['params'],
                objective='multiclass',
                num_class=3,
                metric='multi_logloss'
            )
        
        # Random Forest (robust model)
        if self.config['base_models']['random_forest']['enabled']:
            base_models['random_forest'] = RandomForestClassifier(
                **self.config['base_models']['random_forest']['params']
            )
        
        # Logistic Regression (baseline model)
        if self.config['base_models']['logistic_regression']['enabled']:
            base_models['logistic_regression'] = LogisticRegression(
                **self.config['base_models']['logistic_regression']['params']
            )
        
        self.base_models = base_models
        self.logger.info(f"Created {len(base_models)} base models")
        return base_models
    
    def train_base_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                         X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train base models with conservative validation"""
        self.logger.info("Training base models with conservative validation")
        
        if not self.base_models:
            self.create_base_models()
        
        training_results = {}
        conservative_settings = self.config['conservative_settings']
        
        for name, model in self.base_models.items():
            self.logger.info(f"Training {name}...")
            
            try:
                # Check minimum training samples
                if len(X_train) < conservative_settings['min_training_samples']:
                    self.logger.warning(f"Insufficient training samples for {name}: {len(X_train)}")
                    training_results[name] = {
                        'model': None,
                        'success': False,
                        'error': 'Insufficient training samples'
                    }
                    continue
                
                # Train model
                if name == 'xgboost':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        early_stopping_rounds=30,  # Conservative early stopping
                        verbose=False
                    )
                elif name == 'lightgbm':
                    model.fit(
                        X_train, y_train,
                        eval_set=[(X_val, y_val)],
                        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)]
                    )
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
                
                # Check conservative criteria
                overfitting = train_acc - val_acc
                meets_criteria = (
                    val_acc >= conservative_settings['min_validation_accuracy'] and
                    overfitting <= conservative_settings['max_overfitting_threshold'] and
                    val_logloss <= conservative_settings['max_log_loss']
                )
                
                if meets_criteria:
                    training_results[name] = {
                        'model': model,
                        'train_accuracy': train_acc,
                        'val_accuracy': val_acc,
                        'train_logloss': train_logloss,
                        'val_logloss': val_logloss,
                        'overfitting': overfitting,
                        'success': True
                    }
                    self.logger.info(f"{name} - Val Acc: {val_acc:.4f}, Val LogLoss: {val_logloss:.4f}")
                else:
                    self.logger.warning(f"{name} failed conservative criteria")
                    training_results[name] = {
                        'model': None,
                        'success': False,
                        'error': 'Failed conservative criteria',
                        'val_accuracy': val_acc,
                        'val_logloss': val_logloss,
                        'overfitting': overfitting
                    }
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                training_results[name] = {
                    'model': None,
                    'success': False,
                    'error': str(e)
                }
        
        return training_results
    
    def create_conservative_ensemble(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create conservative ensemble with dynamic weighting"""
        self.logger.info("Creating conservative ensemble")
        
        # Filter successful models
        successful_models = {
            name: result['model'] for name, result in training_results.items()
            if result['success'] and result['model'] is not None
        }
        
        if len(successful_models) < self.config['ensemble_strategy']['min_models']:
            self.logger.error("Not enough successful models for ensemble")
            return None
        
        # Calculate dynamic weights based on performance
        weights = {}
        total_weight = 0
        
        for name, result in training_results.items():
            if result['success'] and name in successful_models:
                # Weight based on validation performance and log loss
                val_acc = result['val_accuracy']
                val_logloss = result['val_logloss']
                
                # Performance score (higher is better)
                performance_score = val_acc - val_logloss
                
                # Apply base weight from config
                base_weight = self.config['base_models'][name]['weight']
                
                # Dynamic weight = base_weight * performance_score
                dynamic_weight = base_weight * (1 + performance_score)
                
                weights[name] = dynamic_weight
                total_weight += dynamic_weight
        
        # Normalize weights
        if total_weight > 0:
            weights = {name: weight / total_weight for name, weight in weights.items()}
        
        # Create ensemble
        ensemble = {
            'models': successful_models,
            'weights': weights,
            'type': 'conservative_weighted',
            'training_results': training_results
        }
        
        self.ensemble_model = ensemble
        self.logger.info(f"Created ensemble with {len(successful_models)} models")
        self.logger.info(f"Model weights: {weights}")
        
        return ensemble
    
    def train_ensemble(self, X_train: pd.DataFrame, y_train: pd.Series,
                      X_val: pd.DataFrame, y_val: pd.Series) -> Dict[str, Any]:
        """Train ensemble model"""
        self.logger.info("Training ensemble model")
        
        if self.ensemble_model is None:
            self.logger.error("No ensemble model created")
            return None
        
        try:
            # For conservative weighted ensemble, we don't train a single model
            # Instead, we evaluate the ensemble performance
            
            models = self.ensemble_model['models']
            weights = self.ensemble_model['weights']
            
            # Calculate ensemble predictions
            ensemble_val_proba = np.zeros((len(X_val), 3))
            
            for name, model in models.items():
                if name in weights:
                    model_proba = model.predict_proba(X_val)
                    ensemble_val_proba += weights[name] * model_proba
            
            # Calculate ensemble metrics
            ensemble_val_pred = np.argmax(ensemble_val_proba, axis=1)
            ensemble_val_acc = accuracy_score(y_val, ensemble_val_pred)
            ensemble_val_logloss = log_loss(y_val, ensemble_val_proba)
            
            ensemble_results = {
                'val_accuracy': ensemble_val_acc,
                'val_logloss': ensemble_val_logloss,
                'success': True,
                'type': 'conservative_weighted'
            }
            
            self.logger.info(f"Ensemble - Val Acc: {ensemble_val_acc:.4f}, Val LogLoss: {ensemble_val_logloss:.4f}")
            
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
        
        calibration_method = self.config['validation']['calibration_method']
        calibrated_models = {}
        
        # Calibrate individual models
        for name, model in self.base_models.items():
            if model is not None:
                try:
                    calibrated_model = CalibratedClassifierCV(
                        model,
                        method=calibration_method,
                        cv=3  # Conservative CV for limited data
                    )
                    calibrated_model.fit(X_train, y_train)
                    calibrated_models[name] = calibrated_model
                    self.logger.info(f"Calibrated {name}")
                except Exception as e:
                    self.logger.error(f"Error calibrating {name}: {e}")
        
        # Calibrate ensemble
        if self.ensemble_model is not None:
            try:
                # Create a wrapper for the ensemble
                class EnsembleWrapper:
                    def __init__(self, ensemble):
                        self.ensemble = ensemble
                    
                    def predict_proba(self, X):
                        models = self.ensemble['models']
                        weights = self.ensemble['weights']
                        
                        ensemble_proba = np.zeros((len(X), 3))
                        
                        for name, model in models.items():
                            if name in weights:
                                model_proba = model.predict_proba(X)
                                ensemble_proba += weights[name] * model_proba
                        
                        return ensemble_proba
                    
                    def predict(self, X):
                        proba = self.predict_proba(X)
                        return np.argmax(proba, axis=1)
                
                ensemble_wrapper = EnsembleWrapper(self.ensemble_model)
                calibrated_ensemble = CalibratedClassifierCV(
                    ensemble_wrapper,
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
    
    def estimate_uncertainty(self, X: pd.DataFrame, n_bootstrap: int = None) -> Dict[str, Any]:
        """Estimate prediction uncertainty using bootstrap"""
        self.logger.info("Estimating prediction uncertainty")
        
        if not self.config['uncertainty']['enabled']:
            return {}
        
        if n_bootstrap is None:
            n_bootstrap = self.config['uncertainty']['n_bootstrap']
        
        if self.ensemble_model is None:
            self.logger.error("No ensemble model for uncertainty estimation")
            return {}
        
        try:
            models = self.ensemble_model['models']
            weights = self.ensemble_model['weights']
            
            # Bootstrap predictions
            bootstrap_predictions = []
            
            for i in range(n_bootstrap):
                # Bootstrap sample
                bootstrap_indices = np.random.choice(len(X), size=len(X), replace=True)
                X_bootstrap = X.iloc[bootstrap_indices]
                
                # Ensemble prediction
                ensemble_proba = np.zeros((len(X_bootstrap), 3))
                
                for name, model in models.items():
                    if name in weights:
                        model_proba = model.predict_proba(X_bootstrap)
                        ensemble_proba += weights[name] * model_proba
                
                bootstrap_predictions.append(ensemble_proba)
            
            # Calculate uncertainty metrics
            bootstrap_predictions = np.array(bootstrap_predictions)
            
            # Mean prediction
            mean_prediction = np.mean(bootstrap_predictions, axis=0)
            
            # Prediction variance
            prediction_variance = np.var(bootstrap_predictions, axis=0)
            
            # Confidence intervals
            confidence_level = self.config['uncertainty']['confidence_level']
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            lower_bound = np.percentile(bootstrap_predictions, lower_percentile, axis=0)
            upper_bound = np.percentile(bootstrap_predictions, upper_percentile, axis=0)
            
            uncertainty_estimates = {
                'mean_prediction': mean_prediction,
                'prediction_variance': prediction_variance,
                'confidence_intervals': {
                    'lower': lower_bound,
                    'upper': upper_bound,
                    'level': confidence_level
                },
                'bootstrap_predictions': bootstrap_predictions
            }
            
            self.uncertainty_estimates = uncertainty_estimates
            self.logger.info("Uncertainty estimation complete")
            
            return uncertainty_estimates
            
        except Exception as e:
            self.logger.error(f"Error estimating uncertainty: {e}")
            return {'error': str(e)}
    
    def cross_validate_ensemble(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Cross-validate ensemble with time series splits"""
        self.logger.info("Cross-validating ensemble")
        
        n_splits = self.config['validation']['n_splits']
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_scores = {
            'accuracy': [],
            'log_loss': [],
            'folds': []
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"Cross-validation fold {fold + 1}/{n_splits}")
            
            X_train_fold, X_val_fold = X.iloc[train_idx], X.iloc[val_idx]
            y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]
            
            try:
                # Train base models for this fold
                training_results = self.train_base_models(X_train_fold, y_train_fold, X_val_fold, y_val_fold)
                
                # Create ensemble for this fold
                ensemble = self.create_conservative_ensemble(training_results)
                
                if ensemble is not None:
                    # Evaluate ensemble
                    models = ensemble['models']
                    weights = ensemble['weights']
                    
                    ensemble_proba = np.zeros((len(X_val_fold), 3))
                    
                    for name, model in models.items():
                        if name in weights:
                            model_proba = model.predict_proba(X_val_fold)
                            ensemble_proba += weights[name] * model_proba
                    
                    ensemble_pred = np.argmax(ensemble_proba, axis=1)
                    fold_acc = accuracy_score(y_val_fold, ensemble_pred)
                    fold_logloss = log_loss(y_val_fold, ensemble_proba)
                    
                    cv_scores['accuracy'].append(fold_acc)
                    cv_scores['log_loss'].append(fold_logloss)
                    cv_scores['folds'].append(fold)
                    
                    self.logger.info(f"Fold {fold + 1} - Acc: {fold_acc:.4f}, LogLoss: {fold_logloss:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error in fold {fold + 1}: {e}")
                continue
        
        # Calculate summary statistics
        if cv_scores['accuracy']:
            cv_summary = {
                'mean_accuracy': np.mean(cv_scores['accuracy']),
                'std_accuracy': np.std(cv_scores['accuracy']),
                'mean_log_loss': np.mean(cv_scores['log_loss']),
                'std_log_loss': np.std(cv_scores['log_loss']),
                'n_folds': len(cv_scores['accuracy']),
                'scores': cv_scores
            }
            
            self.logger.info(f"CV Summary - Mean Acc: {cv_summary['mean_accuracy']:.4f} ± {cv_summary['std_accuracy']:.4f}")
            self.logger.info(f"CV Summary - Mean LogLoss: {cv_summary['mean_log_loss']:.4f} ± {cv_summary['std_log_loss']:.4f}")
            
            return cv_summary
        else:
            self.logger.error("No successful cross-validation folds")
            return {'error': 'No successful folds'}
    
    def evaluate_ensemble(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Any]:
        """Evaluate ensemble performance"""
        self.logger.info("Evaluating ensemble performance")
        
        if self.ensemble_model is None:
            self.logger.error("No ensemble model to evaluate")
            return {}
        
        try:
            models = self.ensemble_model['models']
            weights = self.ensemble_model['weights']
            
            # Calculate ensemble predictions
            ensemble_proba = np.zeros((len(X_test), 3))
            
            for name, model in models.items():
                if name in weights:
                    model_proba = model.predict_proba(X_test)
                    ensemble_proba += weights[name] * model_proba
            
            ensemble_pred = np.argmax(ensemble_proba, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, ensemble_pred)
            logloss = log_loss(y_test, ensemble_proba)
            
            # Classification report
            class_report = classification_report(y_test, ensemble_pred, 
                                                target_names=['Away Win', 'Draw', 'Home Win'],
                                                output_dict=True)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, ensemble_pred)
            
            # Calculate confidence scores
            confidence_scores = np.max(ensemble_proba, axis=1)
            mean_confidence = np.mean(confidence_scores)
            
            evaluation_results = {
                'accuracy': accuracy,
                'log_loss': logloss,
                'mean_confidence': mean_confidence,
                'classification_report': class_report,
                'confusion_matrix': cm,
                'predictions': ensemble_pred,
                'probabilities': ensemble_proba,
                'confidence_scores': confidence_scores
            }
            
            self.logger.info(f"Ensemble Evaluation - Acc: {accuracy:.4f}, LogLoss: {logloss:.4f}")
            self.logger.info(f"Mean Confidence: {mean_confidence:.4f}")
            
            return evaluation_results
            
        except Exception as e:
            self.logger.error(f"Error evaluating ensemble: {e}")
            return {'error': str(e)}
    
    def get_feature_importance(self, X: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Get feature importance from ensemble"""
        self.logger.info("Calculating ensemble feature importance")
        
        if self.ensemble_model is None:
            return {}
        
        feature_importance = {}
        models = self.ensemble_model['models']
        weights = self.ensemble_model['weights']
        
        # Calculate weighted feature importance
        weighted_importance = np.zeros(len(X.columns))
        
        for name, model in models.items():
            if name in weights and hasattr(model, 'feature_importances_'):
                try:
                    importance = model.feature_importances_
                    weighted_importance += weights[name] * importance
                    
                    # Individual model importance
                    importance_df = pd.DataFrame({
                        'feature': X.columns,
                        'importance': importance
                    }).sort_values('importance', ascending=False)
                    
                    feature_importance[name] = importance_df
                    
                except Exception as e:
                    self.logger.error(f"Error getting feature importance for {name}: {e}")
        
        # Ensemble feature importance
        if np.sum(weighted_importance) > 0:
            ensemble_importance_df = pd.DataFrame({
                'feature': X.columns,
                'importance': weighted_importance
            }).sort_values('importance', ascending=False)
            
            feature_importance['ensemble'] = ensemble_importance_df
        
        return feature_importance
    
    def predict_with_confidence(self, X: pd.DataFrame, 
                               confidence_threshold: float = None) -> Dict[str, Any]:
        """Make predictions with confidence filtering"""
        self.logger.info("Making predictions with confidence filtering")
        
        if confidence_threshold is None:
            confidence_threshold = self.config['ensemble_strategy']['confidence_threshold']
        
        if self.ensemble_model is None:
            self.logger.error("No ensemble model for prediction")
            return {}
        
        try:
            models = self.ensemble_model['models']
            weights = self.ensemble_model['weights']
            
            # Calculate ensemble predictions
            ensemble_proba = np.zeros((len(X), 3))
            
            for name, model in models.items():
                if name in weights:
                    model_proba = model.predict_proba(X)
                    ensemble_proba += weights[name] * model_proba
            
            ensemble_pred = np.argmax(ensemble_proba, axis=1)
            confidence_scores = np.max(ensemble_proba, axis=1)
            
            # Filter by confidence threshold
            high_confidence_mask = confidence_scores >= confidence_threshold
            low_confidence_mask = confidence_scores < confidence_threshold
            
            predictions = {
                'predictions': ensemble_pred,
                'probabilities': ensemble_proba,
                'confidence_scores': confidence_scores,
                'high_confidence_mask': high_confidence_mask,
                'low_confidence_mask': low_confidence_mask,
                'high_confidence_count': np.sum(high_confidence_mask),
                'low_confidence_count': np.sum(low_confidence_mask),
                'confidence_threshold': confidence_threshold
            }
            
            self.logger.info(f"High confidence predictions: {predictions['high_confidence_count']}/{len(X)}")
            
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error making predictions: {e}")
            return {'error': str(e)}
    
    def save_ensemble(self, filepath: str):
        """Save ensemble model"""
        self.logger.info(f"Saving ensemble to {filepath}")
        
        ensemble_state = {
            'base_models': self.base_models,
            'ensemble_model': self.ensemble_model,
            'calibrated_models': self.calibrated_models,
            'uncertainty_estimates': self.uncertainty_estimates,
            'performance_metrics': self.performance_metrics,
            'config': self.config
        }
        
        joblib.dump(ensemble_state, filepath)
        self.logger.info("Ensemble saved successfully")
    
    def load_ensemble(self, filepath: str):
        """Load ensemble model"""
        self.logger.info(f"Loading ensemble from {filepath}")
        
        ensemble_state = joblib.load(filepath)
        self.base_models = ensemble_state['base_models']
        self.ensemble_model = ensemble_state['ensemble_model']
        self.calibrated_models = ensemble_state['calibrated_models']
        self.uncertainty_estimates = ensemble_state['uncertainty_estimates']
        self.performance_metrics = ensemble_state['performance_metrics']
        self.config = ensemble_state['config']
        
        self.logger.info("Ensemble loaded successfully")
    
    def _create_feature_subset_models(self, X, y):
        """Train models on different feature subsets for diversity"""
        if not self.config['ensemble_diversity']['feature_subsets']:
            return {}
        
        feature_subsets = {
            'form_focused': [col for col in X.columns if 'form' in col or 'goals' in col],
            'market_focused': [col for col in X.columns if 'odds' in col or 'prob' in col],
            'temporal_focused': [col for col in X.columns if 'season' in col or 'day' in col],
            'consistency_focused': [col for col in X.columns if 'consistency' in col or 'variance' in col]
        }
        
        subset_models = {}
        for name, features in feature_subsets.items():
            if len(features) >= 10:  # Minimum features
                try:
                    import xgboost as xgb
                    model = xgb.XGBClassifier(**self.config['base_models']['xgboost']['params'])
                    model.fit(X[features], y)
                    subset_models[name] = {'model': model, 'features': features}
                    self.logger.info(f"Created {name} subset model with {len(features)} features")
                except Exception as e:
                    self.logger.error(f"Error creating {name} subset model: {e}")
        
        return subset_models
    
    def _create_temporal_window_models(self, X, y):
        """Train models on different temporal windows"""
        if not self.config['ensemble_diversity']['temporal_windows']:
            return {}
        
        windows = {
            'short_term': 5,   # Last 5 matches
            'medium_term': 10, # Last 10 matches
            'long_term': 20    # Last 20 matches
        }
        
        window_models = {}
        for name, window in windows.items():
            # Create features using specific window
            window_features = [col for col in X.columns if f'_{window}' in col or 'form' in col]
            
            if len(window_features) >= 10:
                try:
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(**self.config['base_models']['lightgbm']['params'])
                    model.fit(X[window_features], y)
                    window_models[name] = {'model': model, 'window': window}
                    self.logger.info(f"Created {name} window model with {len(window_features)} features")
                except Exception as e:
                    self.logger.error(f"Error creating {name} window model: {e}")
        
        return window_models
    
    def _create_stacked_ensemble(self, base_models, X, y):
        """Create stacked ensemble with meta-learner"""
        if not self.config['ensemble_diversity']['stacking']:
            return None
        
        try:
            from sklearn.ensemble import StackingClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.model_selection import TimeSeriesSplit
            
            estimators = [(name, model) for name, model in base_models.items()]
            
            # Use conservative meta-learner
            meta_learner = LogisticRegression(
                C=0.1,  # Strong regularization
                max_iter=1000,
                random_state=42
            )
            
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=TimeSeriesSplit(n_splits=3),
                stack_method='predict_proba',
                n_jobs=-1
            )
            
            self.logger.info("Created stacked ensemble")
            return stacking
        except Exception as e:
            self.logger.error(f"Error creating stacked ensemble: {e}")
            return None
    
    def _calculate_dynamic_weights(self, models, X_val, y_val):
        """Calculate weights based on recent performance + diversity bonus"""
        if not self.config['ensemble_diversity']['dynamic_weighting']:
            return {}
        
        weights = {}
        predictions = {}
        
        # Get predictions from all models
        for name, model in models.items():
            try:
                pred_proba = model.predict_proba(X_val)
                predictions[name] = pred_proba
                
                # Base performance
                from sklearn.metrics import accuracy_score, log_loss
                acc = accuracy_score(y_val, pred_proba.argmax(axis=1))
                logloss = log_loss(y_val, pred_proba)
                
                # Performance score
                perf_score = acc - (logloss * 0.2)
                weights[name] = max(0.0, perf_score)
            except Exception as e:
                self.logger.error(f"Error calculating weights for {name}: {e}")
                weights[name] = 0.0
        
        # Add diversity bonus
        diversity_bonus = self.config['ensemble_diversity']['diversity_bonus']
        for name1 in models:
            diversity_score = 0.0
            for name2 in models:
                if name1 != name2 and name1 in predictions and name2 in predictions:
                    try:
                        # Calculate prediction correlation
                        corr = np.corrcoef(
                            predictions[name1].argmax(axis=1),
                            predictions[name2].argmax(axis=1)
                        )[0, 1]
                        # Reward low correlation (high diversity)
                        diversity_score += (1.0 - abs(corr)) * diversity_bonus
                    except Exception as e:
                        self.logger.error(f"Error calculating diversity for {name1}-{name2}: {e}")
            
            weights[name1] += diversity_score
        
        # Normalize weights
        total = sum(weights.values())
        if total > 0:
            weights = {k: v/total for k, v in weights.items()}
        
        self.logger.info(f"Dynamic weights calculated: {weights}")
        return weights
    
    def create_diverse_ensemble(self, X_train, y_train, X_val, y_val):
        """Create diverse ensemble with multiple strategies"""
        self.logger.info("Creating diverse ensemble")
        
        diverse_models = {}
        
        # 1. Feature subset models
        subset_models = self._create_feature_subset_models(X_train, y_train)
        diverse_models.update(subset_models)
        
        # 2. Temporal window models
        window_models = self._create_temporal_window_models(X_train, y_train)
        diverse_models.update(window_models)
        
        # 3. Add new algorithm types
        if self.config['ensemble_diversity']['algorithm_mix']:
            try:
                from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
                
                # Gradient Boosting
                if self.config['base_models']['gradient_boosting']['enabled']:
                    gb_model = GradientBoostingClassifier(**self.config['base_models']['gradient_boosting']['params'])
                    gb_model.fit(X_train, y_train)
                    diverse_models['gradient_boosting'] = gb_model
                
                # Extra Trees
                if self.config['base_models']['extra_trees']['enabled']:
                    et_model = ExtraTreesClassifier(**self.config['base_models']['extra_trees']['params'])
                    et_model.fit(X_train, y_train)
                    diverse_models['extra_trees'] = et_model
                    
            except Exception as e:
                self.logger.error(f"Error creating new algorithm models: {e}")
        
        # 4. Create stacked ensemble
        stacked_ensemble = self._create_stacked_ensemble(diverse_models, X_train, y_train)
        
        # 5. Calculate dynamic weights
        dynamic_weights = self._calculate_dynamic_weights(diverse_models, X_val, y_val)
        
        return {
            'diverse_models': diverse_models,
            'stacked_ensemble': stacked_ensemble,
            'dynamic_weights': dynamic_weights
        }

# Example usage
def main():
    """Example usage of NonMajorLeagueEnsemble"""
    
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
    
    # Initialize ensemble
    ensemble = NonMajorLeagueEnsemble()
    
    # Create and train base models
    base_models = ensemble.create_base_models()
    training_results = ensemble.train_base_models(X_train, y_train, X_val, y_val)
    
    # Create conservative ensemble
    conservative_ensemble = ensemble.create_conservative_ensemble(training_results)
    ensemble_results = ensemble.train_ensemble(X_train, y_train, X_val, y_val)
    
    # Calibrate models
    calibrated_models = ensemble.calibrate_models(X_train, y_train)
    
    # Cross-validate
    cv_results = ensemble.cross_validate_ensemble(X_train, y_train)
    
    # Evaluate ensemble
    evaluation_results = ensemble.evaluate_ensemble(X_test, y_test)
    
    # Make predictions with confidence
    predictions = ensemble.predict_with_confidence(X_test)
    
    # Get feature importance
    feature_importance = ensemble.get_feature_importance(X_train)
    
    # Print results
    print("Ensemble Evaluation Results:")
    if 'error' not in evaluation_results:
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"Log Loss: {evaluation_results['log_loss']:.4f}")
        print(f"Mean Confidence: {evaluation_results['mean_confidence']:.4f}")
    
    print(f"\nCross-Validation Results:")
    if 'error' not in cv_results:
        print(f"Mean Accuracy: {cv_results['mean_accuracy']:.4f} ± {cv_results['std_accuracy']:.4f}")
        print(f"Mean Log Loss: {cv_results['mean_log_loss']:.4f} ± {cv_results['std_log_loss']:.4f}")
    
    print(f"\nPrediction Confidence:")
    print(f"High confidence: {predictions['high_confidence_count']}/{len(X_test)}")
    print(f"Low confidence: {predictions['low_confidence_count']}/{len(X_test)}")
    
    # Save ensemble
    ensemble.save_ensemble('non_major_league_ensemble.pkl')

if __name__ == "__main__":
    main()
