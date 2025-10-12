import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, make_scorer
import optuna
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeagueHyperparameterTuning:
    """
    Hyperparameter tuning system for non-major soccer leagues
    
    Key Features:
    - Conservative tuning for limited data
    - Time series aware optimization
    - Multi-objective optimization
    - Early stopping and pruning
    - Robust validation strategies
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize hyperparameter tuning system
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.study = None
        self.best_params = {}
        self.tuning_results = {}
        
    def setup_logging(self):
        """Setup logging for hyperparameter tuning"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load hyperparameter tuning configuration"""
        if config is None:
            self.config = {
                'optimization': {
                    'n_trials': 30,  # Conservative for limited data
                    'timeout': 1800,  # 30 minutes
                    'direction': 'minimize',
                    'metric': 'log_loss',
                    'pruning': True,
                    'early_stopping': True
                },
                'models': {
                    'xgboost': {
                        'enabled': True,
                        'priority': 1,
                        'param_space': {
                            'n_estimators': [50, 200],
                            'max_depth': [3, 8],
                            'learning_rate': [0.01, 0.1],
                            'subsample': [0.6, 1.0],
                            'colsample_bytree': [0.6, 1.0],
                            'min_child_weight': [1, 10],
                            'reg_alpha': [0.0, 1.0],
                            'reg_lambda': [0.0, 2.0]
                        }
                    },
                    'lightgbm': {
                        'enabled': True,
                        'priority': 2,
                        'param_space': {
                            'n_estimators': [50, 200],
                            'max_depth': [3, 8],
                            'learning_rate': [0.01, 0.1],
                            'num_leaves': [10, 100],
                            'feature_fraction': [0.6, 1.0],
                            'bagging_fraction': [0.6, 1.0],
                            'bagging_freq': [1, 10],
                            'lambda_l1': [0.0, 1.0],
                            'lambda_l2': [0.0, 2.0],
                            'min_data_in_leaf': [5, 50]
                        }
                    },
                    'random_forest': {
                        'enabled': True,
                        'priority': 3,
                        'param_space': {
                            'n_estimators': [50, 200],
                            'max_depth': [3, 15],
                            'min_samples_split': [2, 20],
                            'min_samples_leaf': [1, 10],
                            'max_features': ['sqrt', 'log2', None]
                        }
                    },
                    'logistic_regression': {
                        'enabled': True,
                        'priority': 4,
                        'param_space': {
                            'C': [0.01, 10.0],
                            'max_iter': [100, 1000],
                            'solver': ['lbfgs', 'liblinear']
                        }
                    }
                },
                'validation': {
                    'method': 'time_series_split',
                    'n_splits': 3,  # Conservative for limited data
                    'test_size': 0.2,
                    'scoring': 'neg_log_loss',
                    'cv_folds': 3
                },
                'conservative_settings': {
                    'min_samples_per_fold': 20,
                    'max_overfitting_threshold': 0.15,
                    'min_improvement_threshold': 0.01,
                    'patience': 5
                },
                'pruning': {
                    'enabled': True,
                    'method': 'median',
                    'n_startup_trials': 5,
                    'n_warmup_steps': 10
                }
            }
        else:
            self.config = config
    
    def create_objective_function(self, X: pd.DataFrame, y: pd.Series, 
                                 model_type: str) -> callable:
        """Create objective function for hyperparameter optimization"""
        
        def objective(trial):
            try:
                # Sample hyperparameters based on model type
                if model_type == 'xgboost':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 8),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
                        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 2.0),
                        'random_state': 42,
                        'eval_metric': 'mlogloss'
                    }
                    model = xgb.XGBClassifier(**params)
                    
                elif model_type == 'lightgbm':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 8),
                        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
                        'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                        'feature_fraction': trial.suggest_float('feature_fraction', 0.6, 1.0),
                        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0),
                        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
                        'lambda_l1': trial.suggest_float('lambda_l1', 0.0, 1.0),
                        'lambda_l2': trial.suggest_float('lambda_l2', 0.0, 2.0),
                        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50),
                        'random_state': 42,
                        'verbose': -1
                    }
                    model = lgb.LGBMClassifier(**params)
                    
                elif model_type == 'random_forest':
                    params = {
                        'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                        'max_depth': trial.suggest_int('max_depth', 3, 15),
                        'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                        'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),
                        'random_state': 42
                    }
                    model = RandomForestClassifier(**params)
                    
                elif model_type == 'logistic_regression':
                    params = {
                        'C': trial.suggest_float('C', 0.01, 10.0),
                        'max_iter': trial.suggest_int('max_iter', 100, 1000),
                        'solver': trial.suggest_categorical('solver', ['lbfgs', 'liblinear']),
                        'random_state': 42
                    }
                    model = LogisticRegression(**params)
                
                else:
                    raise ValueError(f"Unknown model type: {model_type}")
                
                # Time series cross-validation
                tscv = TimeSeriesSplit(n_splits=self.config['validation']['n_splits'])
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    # Check minimum samples per fold
                    if len(X_train) < self.config['conservative_settings']['min_samples_per_fold']:
                        continue
                    
                    try:
                        # Train model
                        if model_type in ['xgboost', 'lightgbm']:
                            model.fit(X_train, y_train)
                        else:
                            model.fit(X_train, y_train)
                        
                        # Evaluate
                        val_proba = model.predict_proba(X_val)
                        val_logloss = log_loss(y_val, val_proba)
                        scores.append(val_logloss)
                        
                    except Exception as e:
                        self.logger.warning(f"Error in fold: {e}")
                        continue
                
                if not scores:
                    return float('inf')
                
                # Return mean score
                return np.mean(scores)
                
            except Exception as e:
                self.logger.error(f"Error in objective function: {e}")
                return float('inf')
        
        return objective
    
    def optimize_model(self, X: pd.DataFrame, y: pd.Series, 
                       model_type: str) -> Dict[str, Any]:
        """Optimize hyperparameters for a specific model"""
        self.logger.info(f"Optimizing hyperparameters for {model_type}")
        
        if model_type not in self.config['models'] or not self.config['models'][model_type]['enabled']:
            self.logger.error(f"Model {model_type} not enabled")
            return {}
        
        try:
            # Create objective function
            objective = self.create_objective_function(X, y, model_type)
            
            # Create study
            study = optuna.create_study(
                direction=self.config['optimization']['direction'],
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=self.config['pruning']['n_startup_trials'],
                    n_warmup_steps=self.config['pruning']['n_warmup_steps']
                ) if self.config['pruning']['enabled'] else None
            )
            
            # Optimize
            study.optimize(
                objective,
                n_trials=self.config['optimization']['n_trials'],
                timeout=self.config['optimization']['timeout']
            )
            
            # Store results
            optimization_results = {
                'best_params': study.best_params,
                'best_value': study.best_value,
                'n_trials': len(study.trials),
                'study': study,
                'model_type': model_type
            }
            
            self.best_params[model_type] = study.best_params
            self.logger.info(f"{model_type} optimization complete")
            self.logger.info(f"Best score: {study.best_value:.4f}")
            self.logger.info(f"Best params: {study.best_params}")
            
            return optimization_results
            
        except Exception as e:
            self.logger.error(f"Error optimizing {model_type}: {e}")
            return {'error': str(e)}
    
    def optimize_all_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Optimize hyperparameters for all enabled models"""
        self.logger.info("Optimizing hyperparameters for all models")
        
        all_results = {}
        
        # Sort models by priority
        models_by_priority = sorted(
            self.config['models'].items(),
            key=lambda x: x[1]['priority']
        )
        
        for model_type, model_config in models_by_priority:
            if model_config['enabled']:
                self.logger.info(f"Optimizing {model_type} (priority: {model_config['priority']})")
                
                try:
                    results = self.optimize_model(X, y, model_type)
                    all_results[model_type] = results
                    
                except Exception as e:
                    self.logger.error(f"Error optimizing {model_type}: {e}")
                    all_results[model_type] = {'error': str(e)}
        
        self.tuning_results = all_results
        return all_results
    
    def validate_hyperparameters(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Validate optimized hyperparameters"""
        self.logger.info("Validating optimized hyperparameters")
        
        validation_results = {
            'models': {},
            'overall': {
                'validation_passed': True,
                'issues': []
            }
        }
        
        conservative_settings = self.config['conservative_settings']
        
        for model_type, results in self.tuning_results.items():
            if 'error' in results:
                validation_results['models'][model_type] = {
                    'validation_passed': False,
                    'error': results['error']
                }
                validation_results['overall']['validation_passed'] = False
                validation_results['overall']['issues'].append(f"{model_type}: {results['error']}")
                continue
            
            model_validation = {
                'validation_passed': True,
                'issues': []
            }
            
            # Check best value
            best_value = results.get('best_value', float('inf'))
            if best_value == float('inf'):
                model_validation['validation_passed'] = False
                model_validation['issues'].append("No valid trials")
            
            # Check number of trials
            n_trials = results.get('n_trials', 0)
            if n_trials < 5:
                model_validation['validation_passed'] = False
                model_validation['issues'].append(f"Insufficient trials: {n_trials}")
            
            # Check parameter ranges
            best_params = results.get('best_params', {})
            if not best_params:
                model_validation['validation_passed'] = False
                model_validation['issues'].append("No best parameters found")
            
            validation_results['models'][model_type] = model_validation
            
            if not model_validation['validation_passed']:
                validation_results['overall']['validation_passed'] = False
                validation_results['overall']['issues'].extend(model_validation['issues'])
        
        self.logger.info(f"Hyperparameter validation: {'PASSED' if validation_results['overall']['validation_passed'] else 'FAILED'}")
        
        return validation_results
    
    def compare_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Compare optimized models"""
        self.logger.info("Comparing optimized models")
        
        comparison_results = {
            'models': {},
            'best_model': None,
            'best_score': float('inf')
        }
        
        for model_type, results in self.tuning_results.items():
            if 'error' in results or 'best_value' not in results:
                continue
            
            best_score = results['best_value']
            best_params = results['best_params']
            
            comparison_results['models'][model_type] = {
                'score': best_score,
                'params': best_params
            }
            
            if best_score < comparison_results['best_score']:
                comparison_results['best_score'] = best_score
                comparison_results['best_model'] = model_type
        
        self.logger.info(f"Best model: {comparison_results['best_model']}")
        self.logger.info(f"Best score: {comparison_results['best_score']:.4f}")
        
        return comparison_results
    
    def create_optimized_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Create models with optimized hyperparameters"""
        self.logger.info("Creating optimized models")
        
        optimized_models = {}
        
        for model_type, results in self.tuning_results.items():
            if 'error' in results or 'best_params' not in results:
                continue
            
            try:
                best_params = results['best_params']
                
                if model_type == 'xgboost':
                    model = xgb.XGBClassifier(**best_params, eval_metric='mlogloss')
                elif model_type == 'lightgbm':
                    model = lgb.LGBMClassifier(**best_params, verbose=-1)
                elif model_type == 'random_forest':
                    model = RandomForestClassifier(**best_params)
                elif model_type == 'logistic_regression':
                    model = LogisticRegression(**best_params)
                else:
                    continue
                
                optimized_models[model_type] = model
                self.logger.info(f"Created optimized {model_type} model")
                
            except Exception as e:
                self.logger.error(f"Error creating optimized {model_type}: {e}")
                continue
        
        return optimized_models
    
    def evaluate_optimized_models(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, Any]:
        """Evaluate optimized models"""
        self.logger.info("Evaluating optimized models")
        
        optimized_models = self.create_optimized_models(X, y)
        evaluation_results = {}
        
        # Time series split for evaluation
        tscv = TimeSeriesSplit(n_splits=self.config['validation']['n_splits'])
        
        for model_type, model in optimized_models.items():
            try:
                scores = []
                
                for train_idx, val_idx in tscv.split(X):
                    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                    
                    if len(X_train) < self.config['conservative_settings']['min_samples_per_fold']:
                        continue
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    val_proba = model.predict_proba(X_val)
                    val_logloss = log_loss(y_val, val_proba)
                    scores.append(val_logloss)
                
                if scores:
                    evaluation_results[model_type] = {
                        'mean_score': np.mean(scores),
                        'std_score': np.std(scores),
                        'scores': scores,
                        'model': model
                    }
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_type}: {e}")
                evaluation_results[model_type] = {'error': str(e)}
        
        return evaluation_results
    
    def get_hyperparameter_importance(self) -> Dict[str, Any]:
        """Get hyperparameter importance from optimization"""
        self.logger.info("Calculating hyperparameter importance")
        
        importance_results = {}
        
        for model_type, results in self.tuning_results.items():
            if 'error' in results or 'study' not in results:
                continue
            
            try:
                study = results['study']
                importance = optuna.importance.get_param_importances(study)
                
                importance_results[model_type] = {
                    'importance': importance,
                    'best_params': results['best_params'],
                    'best_value': results['best_value']
                }
                
            except Exception as e:
                self.logger.error(f"Error calculating importance for {model_type}: {e}")
                continue
        
        return importance_results
    
    def save_hyperparameter_tuning(self, filepath: str):
        """Save hyperparameter tuning results"""
        self.logger.info(f"Saving hyperparameter tuning results to {filepath}")
        
        tuning_state = {
            'best_params': self.best_params,
            'tuning_results': self.tuning_results,
            'config': self.config
        }
        
        joblib.dump(tuning_state, filepath)
        self.logger.info("Hyperparameter tuning results saved successfully")
    
    def load_hyperparameter_tuning(self, filepath: str):
        """Load hyperparameter tuning results"""
        self.logger.info(f"Loading hyperparameter tuning results from {filepath}")
        
        tuning_state = joblib.load(filepath)
        self.best_params = tuning_state['best_params']
        self.tuning_results = tuning_state['tuning_results']
        self.config = tuning_state['config']
        
        self.logger.info("Hyperparameter tuning results loaded successfully")

# Example usage
def main():
    """Example usage of NonMajorLeagueHyperparameterTuning"""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 15
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.choice([0, 1, 2], n_samples))
    
    # Initialize hyperparameter tuning
    hyperparameter_tuning = NonMajorLeagueHyperparameterTuning()
    
    # Optimize all models
    optimization_results = hyperparameter_tuning.optimize_all_models(X, y)
    
    # Validate hyperparameters
    validation_results = hyperparameter_tuning.validate_hyperparameters(X, y)
    
    # Compare models
    comparison_results = hyperparameter_tuning.compare_models(X, y)
    
    # Evaluate optimized models
    evaluation_results = hyperparameter_tuning.evaluate_optimized_models(X, y)
    
    # Get hyperparameter importance
    importance_results = hyperparameter_tuning.get_hyperparameter_importance()
    
    # Print results
    print("Hyperparameter Tuning Results:")
    for model_type, results in optimization_results.items():
        if 'error' not in results:
            print(f"{model_type}: Best score = {results['best_value']:.4f}")
    
    print(f"\nBest model: {comparison_results['best_model']}")
    print(f"Best score: {comparison_results['best_score']:.4f}")
    
    print(f"\nValidation passed: {validation_results['overall']['validation_passed']}")
    
    # Save results
    hyperparameter_tuning.save_hyperparameter_tuning('hyperparameter_tuning_results.pkl')

if __name__ == "__main__":
    main()
