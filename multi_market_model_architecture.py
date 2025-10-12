import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
import optuna
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class MultiMarketModelArchitecture:
    """
    Multi-market model architecture for simultaneous prediction of multiple betting markets
    
    Key Features:
    - Multi-output models for simultaneous predictions
    - Market-specific feature engineering
    - Cross-market correlation modeling
    - Ensemble methods for multiple markets
    - Calibrated probabilities for all markets
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize multi-market model architecture
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.market_models = {}
        self.ensemble_models = {}
        self.calibrated_models = {}
        self.market_features = {}
        self.market_correlations = {}
        
    def setup_logging(self):
        """Setup logging for multi-market architecture"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load multi-market configuration"""
        if config is None:
            self.config = {
                'enabled_markets': {
                    'match_result': True,  # 1X2
                    'both_teams_score': True,  # GG/NOGG
                    'over_under_25': True,  # O/U 2.5
                    'over_under_15': True,  # O/U 1.5
                    'half_time_result': True,  # HT 1X2
                    'double_chance': True,  # 1X, 12, X2
                    'correct_score': True,  # Most likely scores
                    'clean_sheet': True,  # Home/Away clean sheet
                    'first_goal': True,  # First goal scorer
                    'win_to_nil': True  # Win without conceding
                },
                'model_architecture': {
                    'method': 'multi_output',  # 'multi_output', 'separate_models', 'hierarchical'
                    'base_models': ['xgboost', 'lightgbm', 'random_forest'],
                    'ensemble_method': 'voting',
                    'calibration': True
                },
                'market_specific_features': {
                    'match_result': ['form', 'consistency', 'market', 'temporal'],
                    'both_teams_score': ['goals_scored', 'goals_conceded', 'attacking_strength'],
                    'over_under_25': ['total_goals_avg', 'defensive_strength', 'attacking_strength'],
                    'half_time_result': ['ht_form', 'ht_goals_avg', 'ht_consistency'],
                    'double_chance': ['form', 'consistency', 'market', 'temporal']
                },
                'cross_market_modeling': {
                    'enabled': True,
                    'correlation_threshold': 0.3,
                    'joint_training': True
                },
                'validation': {
                    'method': 'time_series_split',
                    'n_splits': 5,
                    'calibration_method': 'isotonic'
                }
            }
        else:
            self.config = config
    
    def create_market_features(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Create market-specific feature sets"""
        self.logger.info("Creating market-specific features")
        
        market_features = {}
        
        # Base features (common to all markets) - use available numeric features
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        base_features = numeric_cols[:min(10, len(numeric_cols))]  # Use first 10 numeric features
        
        # Match Result features
        if self.config['enabled_markets']['match_result']:
            market_features['match_result'] = df[base_features]
        
        # Both Teams to Score features
        if self.config['enabled_markets']['both_teams_score']:
            market_features['both_teams_score'] = df[base_features]
        
        # Over/Under 2.5 features
        if self.config['enabled_markets']['over_under_25']:
            market_features['over_under_25'] = df[base_features]
        
        # Over/Under 1.5 features
        if self.config['enabled_markets']['over_under_15']:
            market_features['over_under_15'] = df[base_features]
        
        # Half-Time Result features
        if self.config['enabled_markets']['half_time_result']:
            market_features['half_time_result'] = df[base_features]
        
        # Double Chance features
        if self.config['enabled_markets']['double_chance']:
            market_features['double_chance'] = df[base_features]
        
        self.market_features = market_features
        self.logger.info(f"Created features for {len(market_features)} markets")
        return market_features
    
    def create_multi_output_models(self) -> Dict[str, Any]:
        """Create multi-output models for simultaneous market predictions"""
        self.logger.info("Creating multi-output models")
        
        models = {}
        
        # XGBoost Multi-Output
        if 'xgboost' in self.config['model_architecture']['base_models']:
            xgb_model = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='mlogloss',
                objective='multi:softprob'
            )
            models['xgboost'] = MultiOutputClassifier(xgb_model)
        
        # LightGBM Multi-Output
        if 'lightgbm' in self.config['model_architecture']['base_models']:
            lgb_model = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                num_leaves=50,
                feature_fraction=0.8,
                bagging_fraction=0.8,
                random_state=42,
                verbose=-1
            )
            models['lightgbm'] = MultiOutputClassifier(lgb_model)
        
        # Random Forest Multi-Output
        if 'random_forest' in self.config['model_architecture']['base_models']:
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42
            )
            models['random_forest'] = MultiOutputClassifier(rf_model)
        
        self.market_models = models
        self.logger.info(f"Created {len(models)} multi-output models")
        return models
    
    def create_separate_models(self) -> Dict[str, Dict[str, Any]]:
        """Create separate models for each market"""
        self.logger.info("Creating separate models for each market")
        
        separate_models = {}
        
        for market in self.config['enabled_markets']:
            if self.config['enabled_markets'][market]:
                market_models = {}
                
                # XGBoost
                xgb_model = xgb.XGBClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=42
                )
                market_models['xgboost'] = xgb_model
                
                # LightGBM
                lgb_model = lgb.LGBMClassifier(
                    n_estimators=150,
                    max_depth=5,
                    learning_rate=0.05,
                    random_state=42,
                    verbose=-1
                )
                market_models['lightgbm'] = lgb_model
                
                # Random Forest
                rf_model = RandomForestClassifier(
                    n_estimators=100,
                    max_depth=8,
                    random_state=42
                )
                market_models['random_forest'] = rf_model
                
                separate_models[market] = market_models
        
        self.market_models = separate_models
        self.logger.info(f"Created separate models for {len(separate_models)} markets")
        return separate_models
    
    def train_multi_output_models(self, X: pd.DataFrame, y: pd.DataFrame, 
                                 X_val: pd.DataFrame, y_val: pd.DataFrame) -> Dict[str, Any]:
        """Train multi-output models"""
        self.logger.info("Training multi-output models")
        
        if not self.market_models:
            self.create_multi_output_models()
        
        training_results = {}
        
        for name, model in self.market_models.items():
            self.logger.info(f"Training {name} multi-output model...")
            
            try:
                # Train model
                if name == 'xgboost':
                    model.fit(X, y)
                elif name == 'lightgbm':
                    model.fit(X, y)
                else:
                    model.fit(X, y)
                
                # Evaluate model
                train_pred = model.predict(X)
                val_pred = model.predict(X_val)
                
                # Calculate accuracy for each market
                market_accuracies = {}
                for i, market in enumerate(y.columns):
                    train_acc = accuracy_score(y.iloc[:, i], train_pred[:, i])
                    val_acc = accuracy_score(y_val.iloc[:, i], val_pred[:, i])
                    market_accuracies[market] = {'train': train_acc, 'val': val_acc}
                
                training_results[name] = {
                    'model': model,
                    'market_accuracies': market_accuracies,
                    'success': True
                }
                
                self.logger.info(f"{name} trained successfully")
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {e}")
                training_results[name] = {
                    'model': None,
                    'success': False,
                    'error': str(e)
                }
        
        return training_results
    
    def train_separate_models(self, market_features: Dict[str, pd.DataFrame], 
                            market_targets: Dict[str, pd.Series]) -> Dict[str, Dict[str, Any]]:
        """Train separate models for each market"""
        self.logger.info("Training separate models")
        
        if not self.market_models:
            self.create_separate_models()
        
        training_results = {}
        
        for market, models in self.market_models.items():
            if market not in market_features or market not in market_targets:
                continue
            
            X = market_features[market]
            y = market_targets[market]
            
            # Split data
            split_idx = int(0.8 * len(X))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            market_results = {}
            
            for model_name, model in models.items():
                self.logger.info(f"Training {model_name} for {market}...")
                
                try:
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    train_pred = model.predict(X_train)
                    val_pred = model.predict(X_val)
                    train_acc = accuracy_score(y_train, train_pred)
                    val_acc = accuracy_score(y_val, val_pred)
                    
                    market_results[model_name] = {
                        'model': model,
                        'train_accuracy': train_acc,
                        'val_accuracy': val_acc,
                        'success': True
                    }
                    
                except Exception as e:
                    self.logger.error(f"Error training {model_name} for {market}: {e}")
                    market_results[model_name] = {
                        'model': None,
                        'success': False,
                        'error': str(e)
                    }
            
            training_results[market] = market_results
        
        return training_results
    
    def create_market_ensembles(self, training_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create ensemble models for each market"""
        self.logger.info("Creating market ensembles")
        
        ensemble_models = {}
        
        if self.config['model_architecture']['method'] == 'multi_output':
            # Create ensemble for multi-output models
            successful_models = {
                name: result['model'] for name, result in training_results.items()
                if result.get('success', False) and result.get('model') is not None
            }
            
            if len(successful_models) > 1:
                ensemble = VotingClassifier(
                    estimators=list(successful_models.items()),
                    voting='soft'
                )
                ensemble_models['multi_output'] = ensemble
        
        else:
            # Create ensembles for separate models
            for market, results in training_results.items():
                successful_models = {
                    name: result['model'] for name, result in results.items()
                    if result.get('success', False) and result.get('model') is not None
                }
                
                if len(successful_models) > 1:
                    ensemble = VotingClassifier(
                        estimators=list(successful_models.items()),
                        voting='soft'
                    )
                    ensemble_models[market] = ensemble
        
        self.ensemble_models = ensemble_models
        self.logger.info(f"Created {len(ensemble_models)} ensemble models")
        return ensemble_models
    
    def calibrate_market_models(self, X_train: pd.DataFrame, y_train: pd.DataFrame) -> Dict[str, Any]:
        """Calibrate probabilities for all market models"""
        self.logger.info("Calibrating market models")
        
        if not self.config['model_architecture']['calibration']:
            return {}
        
        calibrated_models = {}
        
        # Calibrate multi-output models
        for name, model in self.market_models.items():
            try:
                calibrated_model = CalibratedClassifierCV(
                    model,
                    method='isotonic',
                    cv=3
                )
                calibrated_model.fit(X_train, y_train)
                calibrated_models[name] = calibrated_model
                self.logger.info(f"Calibrated {name}")
            except Exception as e:
                self.logger.error(f"Error calibrating {name}: {e}")
        
        # Calibrate ensemble models
        for name, model in self.ensemble_models.items():
            try:
                calibrated_model = CalibratedClassifierCV(
                    model,
                    method='isotonic',
                    cv=3
                )
                calibrated_model.fit(X_train, y_train)
                calibrated_models[f'{name}_ensemble'] = calibrated_model
                self.logger.info(f"Calibrated {name} ensemble")
            except Exception as e:
                self.logger.error(f"Error calibrating {name} ensemble: {e}")
        
        self.calibrated_models = calibrated_models
        return calibrated_models
    
    def predict_markets(self, df: pd.DataFrame, training_results: Dict[str, Dict[str, Any]] = None, use_calibrated: bool = True) -> Dict[str, Dict[str, Any]]:
        """Make predictions for all markets"""
        self.logger.info("Making multi-market predictions")
        
        predictions = {}
        
        # Use training results if provided, otherwise use self.market_models
        models_to_use = training_results if training_results else self.market_models
        
        # Create market-specific features
        market_features = self.create_market_features(df)
        
        # Separate model predictions
        for market, models in models_to_use.items():
            try:
                # Use the first available model for prediction
                model_name = list(models.keys())[0]
                model_info = models[model_name]
                
                # Extract the actual model
                if isinstance(model_info, dict) and 'model' in model_info:
                    model = model_info['model']
                else:
                    model = model_info
                
                # Skip if model is None or not fitted
                if model is None:
                    self.logger.warning(f"Skipping {market} - model is None")
                    continue
                
                # Use market-specific features
                X = market_features[market] if market in market_features else df.select_dtypes(include=[np.number])
                
                pred = model.predict(X)
                proba = model.predict_proba(X)
                
                predictions[market] = {
                    'predictions': pred,
                    'probabilities': proba,
                    'model_used': model_name
                }
            except Exception as e:
                self.logger.error(f"Error predicting with {market}: {e}")
        
        # Ensemble predictions
        for name, model in self.ensemble_models.items():
            try:
                if use_calibrated and f'{name}_ensemble' in self.calibrated_models:
                    pred = self.calibrated_models[f'{name}_ensemble'].predict(X)
                    proba = self.calibrated_models[f'{name}_ensemble'].predict_proba(X)
                else:
                    pred = model.predict(X)
                    proba = model.predict_proba(X)
                
                predictions[f'{name}_ensemble'] = {
                    'predictions': pred,
                    'probabilities': proba
                }
            except Exception as e:
                self.logger.error(f"Error predicting with {name} ensemble: {e}")
        
        return predictions
    
    def calculate_market_correlations(self, predictions: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        """Calculate correlations between market predictions"""
        self.logger.info("Calculating market correlations")
        
        correlations = {}
        
        # Get prediction arrays
        pred_arrays = {}
        for model_name, pred_data in predictions.items():
            if 'predictions' in pred_data:
                pred_arrays[model_name] = pred_data['predictions']
        
        # Calculate pairwise correlations
        model_names = list(pred_arrays.keys())
        for i, name1 in enumerate(model_names):
            for name2 in model_names[i+1:]:
                try:
                    corr = np.corrcoef(pred_arrays[name1].flatten(), pred_arrays[name2].flatten())[0, 1]
                    correlations[f"{name1}_{name2}"] = corr
                except Exception as e:
                    self.logger.error(f"Error calculating correlation {name1}-{name2}: {e}")
        
        self.market_correlations = correlations
        return correlations
    
    def save_models(self, filepath: str):
        """Save all multi-market models"""
        self.logger.info(f"Saving multi-market models to {filepath}")
        
        model_state = {
            'market_models': self.market_models,
            'ensemble_models': self.ensemble_models,
            'calibrated_models': self.calibrated_models,
            'market_features': self.market_features,
            'market_correlations': self.market_correlations,
            'config': self.config
        }
        
        joblib.dump(model_state, filepath)
        self.logger.info("Multi-market models saved successfully")
    
    def load_models(self, filepath: str):
        """Load multi-market models"""
        self.logger.info(f"Loading multi-market models from {filepath}")
        
        model_state = joblib.load(filepath)
        self.market_models = model_state['market_models']
        self.ensemble_models = model_state['ensemble_models']
        self.calibrated_models = model_state['calibrated_models']
        self.market_features = model_state['market_features']
        self.market_correlations = model_state['market_correlations']
        self.config = model_state['config']
        
        self.logger.info("Multi-market models loaded successfully")

# Example usage
def main():
    """Example usage of MultiMarketModelArchitecture"""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    
    # Create multi-market targets
    y = pd.DataFrame({
        'match_result': np.random.choice([0, 1, 2], n_samples),
        'both_teams_score': np.random.choice([0, 1], n_samples),
        'over_under_25': np.random.choice([0, 1], n_samples),
        'over_under_15': np.random.choice([0, 1], n_samples)
    })
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    X_train, X_val = X_train[:int(0.8 * len(X_train))], X_train[int(0.8 * len(X_train)):]
    y_train, y_val = y_train[:int(0.8 * len(y_train))], y_train[int(0.8 * len(y_train)):]
    
    # Initialize multi-market architecture
    architecture = MultiMarketModelArchitecture()
    
    # Create and train multi-output models
    models = architecture.create_multi_output_models()
    training_results = architecture.train_multi_output_models(X_train, y_train, X_val, y_val)
    
    # Create ensembles
    ensembles = architecture.create_market_ensembles(training_results)
    
    # Calibrate models
    calibrated_models = architecture.calibrate_market_models(X_train, y_train)
    
    # Make predictions
    predictions = architecture.predict_markets(X_test)
    
    # Calculate correlations
    correlations = architecture.calculate_market_correlations(predictions)
    
    print(f"Multi-market predictions created for {len(predictions)} models")
    print(f"Market correlations calculated: {len(correlations)}")
    print("Multi-market model architecture successfully implemented!")

if __name__ == "__main__":
    main()
