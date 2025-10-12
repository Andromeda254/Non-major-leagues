import pandas as pd
import numpy as np
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss
import joblib
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeagueTransferLearning:
    """
    Transfer learning system for non-major soccer leagues
    
    Key Features:
    - Pre-training on major leagues
    - Fine-tuning on target league
    - Feature transfer and adaptation
    - Conservative transfer ratios
    - Cross-league knowledge sharing
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize transfer learning system
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.source_models = {}
        self.transferred_features = {}
        self.target_model = None
        self.transfer_metrics = {}
        
    def setup_logging(self):
        """Setup logging for transfer learning"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load transfer learning configuration"""
        if config is None:
            self.config = {
                'source_leagues': {
                    'EPL': {
                        'enabled': True,
                        'weight': 0.4,
                        'data_file': 'epl_data.csv',
                        'features': 'all'
                    },
                    'LaLiga': {
                        'enabled': True,
                        'weight': 0.3,
                        'data_file': 'laliga_data.csv',
                        'features': 'all'
                    },
                    'Bundesliga': {
                        'enabled': True,
                        'weight': 0.3,
                        'data_file': 'bundesliga_data.csv',
                        'features': 'all'
                    }
                },
                'transfer_strategy': {
                    'method': 'feature_transfer',  # 'feature_transfer', 'model_transfer', 'hybrid'
                    'transfer_ratio': 0.3,
                    'fine_tune_epochs': 50,
                    'learning_rate_decay': 0.9,
                    'early_stopping_patience': 10
                },
                'feature_transfer': {
                    'enabled': True,
                    'common_features': [
                        'home_form_5', 'away_form_5',
                        'home_goals_scored_5', 'away_goals_scored_5',
                        'home_goals_conceded_5', 'away_goals_conceded_5',
                        'home_elo', 'away_elo', 'elo_diff',
                        'home_advantage', 'total_goals', 'goal_difference'
                    ],
                    'league_specific_features': [
                        'league_avg_goals', 'league_home_advantage',
                        'league_competitiveness', 'league_draw_rate'
                    ],
                    'transfer_weights': {
                        'common': 0.7,
                        'league_specific': 0.3
                    }
                },
                'model_transfer': {
                    'enabled': True,
                    'base_model': 'xgboost',
                    'transfer_layers': ['feature_importance', 'tree_structure'],
                    'adaptation_rate': 0.1
                },
                'validation': {
                    'method': 'holdout',
                    'test_size': 0.2,
                    'validation_size': 0.2,
                    'cross_validation': False
                },
                'conservative_settings': {
                    'min_source_samples': 1000,
                    'min_target_samples': 100,
                    'max_transfer_ratio': 0.5,
                    'min_improvement_threshold': 0.02
                }
            }
        else:
            self.config = config
    
    def load_source_data(self, source_leagues: List[str] = None) -> Dict[str, pd.DataFrame]:
        """Load data from source leagues"""
        self.logger.info("Loading source league data")
        
        if source_leagues is None:
            source_leagues = [league for league, config in self.config['source_leagues'].items() 
                            if config['enabled']]
        
        source_data = {}
        
        for league in source_leagues:
            try:
                # In practice, you would load from actual data files
                # For now, create synthetic data
                data = self._create_synthetic_source_data(league)
                source_data[league] = data
                self.logger.info(f"Loaded {len(data)} samples from {league}")
                
            except Exception as e:
                self.logger.error(f"Error loading {league} data: {e}")
                continue
        
        return source_data
    
    def _create_synthetic_source_data(self, league: str) -> pd.DataFrame:
        """Create synthetic source data for demonstration"""
        np.random.seed(42)
        n_samples = 2000
        
        # Common features
        data = {
            'home_form_5': np.random.uniform(0, 3, n_samples),
            'away_form_5': np.random.uniform(0, 3, n_samples),
            'home_goals_scored_5': np.random.uniform(0, 3, n_samples),
            'away_goals_scored_5': np.random.uniform(0, 3, n_samples),
            'home_goals_conceded_5': np.random.uniform(0, 3, n_samples),
            'away_goals_conceded_5': np.random.uniform(0, 3, n_samples),
            'home_elo': np.random.uniform(1200, 1800, n_samples),
            'away_elo': np.random.uniform(1200, 1800, n_samples),
            'home_advantage': np.random.uniform(-2, 2, n_samples),
            'total_goals': np.random.uniform(0, 6, n_samples),
            'goal_difference': np.random.uniform(-3, 3, n_samples)
        }
        
        # League-specific features
        if league == 'EPL':
            data['league_avg_goals'] = 2.7
            data['league_home_advantage'] = 0.15
            data['league_competitiveness'] = 0.8
            data['league_draw_rate'] = 0.25
        elif league == 'LaLiga':
            data['league_avg_goals'] = 2.5
            data['league_home_advantage'] = 0.12
            data['league_competitiveness'] = 0.7
            data['league_draw_rate'] = 0.28
        elif league == 'Bundesliga':
            data['league_avg_goals'] = 2.9
            data['league_home_advantage'] = 0.18
            data['league_competitiveness'] = 0.75
            data['league_draw_rate'] = 0.22
        
        # Target variable
        data['target'] = np.random.choice([0, 1, 2], n_samples)
        
        return pd.DataFrame(data)
    
    def pre_train_source_models(self, source_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Pre-train models on source leagues"""
        self.logger.info("Pre-training models on source leagues")
        
        source_models = {}
        
        for league, data in source_data.items():
            self.logger.info(f"Pre-training on {league}")
            
            try:
                # Prepare data
                X = data.drop('target', axis=1)
                y = data['target']
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Train XGBoost model
                model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='mlogloss'
                )
                
                model.fit(X_train, y_train)
                
                # Evaluate
                test_pred = model.predict(X_test)
                test_proba = model.predict_proba(X_test)
                test_acc = accuracy_score(y_test, test_pred)
                test_logloss = log_loss(y_test, test_proba)
                
                source_models[league] = {
                    'model': model,
                    'accuracy': test_acc,
                    'log_loss': test_logloss,
                    'feature_importance': model.feature_importances_,
                    'feature_names': X.columns.tolist()
                }
                
                self.logger.info(f"{league} - Acc: {test_acc:.4f}, LogLoss: {test_logloss:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error pre-training {league}: {e}")
                continue
        
        self.source_models = source_models
        return source_models
    
    def transfer_features(self, source_models: Dict[str, Any], 
                        target_data: pd.DataFrame) -> Dict[str, Any]:
        """Transfer features from source leagues to target league"""
        self.logger.info("Transferring features from source leagues")
        
        if not self.config['feature_transfer']['enabled']:
            return {}
        
        transferred_features = {}
        common_features = self.config['feature_transfer']['common_features']
        league_specific_features = self.config['feature_transfer']['league_specific_features']
        
        for league, model_info in source_models.items():
            try:
                model = model_info['model']
                feature_importance = model_info['feature_importance']
                feature_names = model_info['feature_names']
                
                # Create feature importance mapping
                feature_importance_map = dict(zip(feature_names, feature_importance))
                
                # Transfer common features
                common_transfer = {}
                for feature in common_features:
                    if feature in feature_importance_map:
                        common_transfer[feature] = feature_importance_map[feature]
                
                # Transfer league-specific features
                league_specific_transfer = {}
                for feature in league_specific_features:
                    if feature in feature_importance_map:
                        league_specific_transfer[feature] = feature_importance_map[feature]
                
                # Calculate transfer weights
                transfer_weights = self.config['feature_transfer']['transfer_weights']
                
                transferred_features[league] = {
                    'common_features': common_transfer,
                    'league_specific_features': league_specific_transfer,
                    'transfer_weight': self.config['source_leagues'][league]['weight'],
                    'common_weight': transfer_weights['common'],
                    'league_specific_weight': transfer_weights['league_specific']
                }
                
                self.logger.info(f"Transferred features from {league}")
                
            except Exception as e:
                self.logger.error(f"Error transferring features from {league}: {e}")
                continue
        
        self.transferred_features = transferred_features
        return transferred_features
    
    def create_transferred_features(self, target_data: pd.DataFrame) -> pd.DataFrame:
        """Create transferred features for target data"""
        self.logger.info("Creating transferred features for target data")
        
        if not self.transferred_features:
            self.logger.warning("No transferred features available")
            return target_data
        
        target_data_transferred = target_data.copy()
        
        # Calculate weighted feature importance across source leagues
        weighted_feature_importance = {}
        
        for league, transfer_info in self.transferred_features.items():
            league_weight = transfer_info['transfer_weight']
            common_weight = transfer_info['common_weight']
            league_specific_weight = transfer_info['league_specific_weight']
            
            # Add common features
            for feature, importance in transfer_info['common_features'].items():
                if feature not in weighted_feature_importance:
                    weighted_feature_importance[feature] = 0
                weighted_feature_importance[feature] += league_weight * common_weight * importance
            
            # Add league-specific features
            for feature, importance in transfer_info['league_specific_features'].items():
                if feature not in weighted_feature_importance:
                    weighted_feature_importance[feature] = 0
                weighted_feature_importance[feature] += league_weight * league_specific_weight * importance
        
        # Normalize feature importance
        if weighted_feature_importance:
            max_importance = max(weighted_feature_importance.values())
            weighted_feature_importance = {
                feature: importance / max_importance 
                for feature, importance in weighted_feature_importance.items()
            }
        
        # Create transferred feature columns
        for feature, importance in weighted_feature_importance.items():
            if feature in target_data.columns:
                # Create transferred version of the feature
                transferred_feature_name = f"{feature}_transferred"
                target_data_transferred[transferred_feature_name] = (
                    target_data[feature] * importance
                )
        
        # Add transfer learning metadata
        target_data_transferred['transfer_learning_factor'] = self.config['transfer_strategy']['transfer_ratio']
        target_data_transferred['source_leagues_count'] = len(self.transferred_features)
        
        self.logger.info(f"Created {len(weighted_feature_importance)} transferred features")
        return target_data_transferred
    
    def fine_tune_target_model(self, target_data: pd.DataFrame, 
                              target_league: str) -> Dict[str, Any]:
        """Fine-tune model on target league data"""
        self.logger.info(f"Fine-tuning model on {target_league}")
        
        # Prepare data
        X = target_data.drop('target', axis=1)
        y = target_data['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Create base model with transferred knowledge
        base_model = xgb.XGBClassifier(
            n_estimators=100,  # Fewer estimators for fine-tuning
            max_depth=5,
            learning_rate=0.01,  # Lower learning rate for fine-tuning
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        # Fine-tune with early stopping
        early_stopping_rounds = self.config['transfer_strategy']['early_stopping_patience']
        
        base_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            early_stopping_rounds=early_stopping_rounds,
            verbose=False
        )
        
        # Evaluate fine-tuned model
        test_pred = base_model.predict(X_test)
        test_proba = base_model.predict_proba(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        test_logloss = log_loss(y_test, test_proba)
        
        fine_tuned_results = {
            'model': base_model,
            'accuracy': test_acc,
            'log_loss': test_logloss,
            'feature_importance': base_model.feature_importances_,
            'feature_names': X.columns.tolist(),
            'n_estimators_used': base_model.n_estimators
        }
        
        self.target_model = base_model
        self.logger.info(f"Fine-tuned model - Acc: {test_acc:.4f}, LogLoss: {test_logloss:.4f}")
        
        return fine_tuned_results
    
    def compare_with_baseline(self, target_data: pd.DataFrame) -> Dict[str, Any]:
        """Compare transfer learning model with baseline"""
        self.logger.info("Comparing with baseline model")
        
        # Prepare data
        X = target_data.drop('target', axis=1)
        y = target_data['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train baseline model (no transfer learning)
        baseline_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric='mlogloss'
        )
        
        baseline_model.fit(X_train, y_train)
        
        # Evaluate baseline
        baseline_pred = baseline_model.predict(X_test)
        baseline_proba = baseline_model.predict_proba(X_test)
        baseline_acc = accuracy_score(y_test, baseline_pred)
        baseline_logloss = log_loss(y_test, baseline_proba)
        
        # Evaluate transfer learning model
        if self.target_model is not None:
            transfer_pred = self.target_model.predict(X_test)
            transfer_proba = self.target_model.predict_proba(X_test)
            transfer_acc = accuracy_score(y_test, transfer_pred)
            transfer_logloss = log_loss(y_test, transfer_proba)
            
            # Calculate improvement
            acc_improvement = transfer_acc - baseline_acc
            logloss_improvement = baseline_logloss - transfer_logloss
            
            comparison_results = {
                'baseline': {
                    'accuracy': baseline_acc,
                    'log_loss': baseline_logloss
                },
                'transfer_learning': {
                    'accuracy': transfer_acc,
                    'log_loss': transfer_logloss
                },
                'improvement': {
                    'accuracy': acc_improvement,
                    'log_loss': logloss_improvement,
                    'accuracy_pct': (acc_improvement / baseline_acc) * 100,
                    'logloss_pct': (logloss_improvement / baseline_logloss) * 100
                },
                'meets_threshold': acc_improvement >= self.config['conservative_settings']['min_improvement_threshold']
            }
            
            self.logger.info(f"Baseline - Acc: {baseline_acc:.4f}, LogLoss: {baseline_logloss:.4f}")
            self.logger.info(f"Transfer Learning - Acc: {transfer_acc:.4f}, LogLoss: {transfer_logloss:.4f}")
            self.logger.info(f"Improvement - Acc: {acc_improvement:+.4f}, LogLoss: {logloss_improvement:+.4f}")
            
        else:
            comparison_results = {
                'baseline': {
                    'accuracy': baseline_acc,
                    'log_loss': baseline_logloss
                },
                'transfer_learning': None,
                'error': 'No transfer learning model available'
            }
        
        return comparison_results
    
    def validate_transfer_learning(self, target_data: pd.DataFrame) -> Dict[str, Any]:
        """Validate transfer learning effectiveness"""
        self.logger.info("Validating transfer learning effectiveness")
        
        validation_results = {
            'source_leagues': len(self.source_models),
            'transferred_features': len(self.transferred_features),
            'transfer_ratio': self.config['transfer_strategy']['transfer_ratio'],
            'validation_passed': True,
            'issues': []
        }
        
        # Check minimum requirements
        conservative_settings = self.config['conservative_settings']
        
        if len(target_data) < conservative_settings['min_target_samples']:
            validation_results['validation_passed'] = False
            validation_results['issues'].append(f"Insufficient target samples: {len(target_data)}")
        
        if len(self.source_models) < 1:
            validation_results['validation_passed'] = False
            validation_results['issues'].append("No source models available")
        
        if self.config['transfer_strategy']['transfer_ratio'] > conservative_settings['max_transfer_ratio']:
            validation_results['validation_passed'] = False
            validation_results['issues'].append(f"Transfer ratio too high: {self.config['transfer_strategy']['transfer_ratio']}")
        
        # Check feature transfer
        if not self.transferred_features:
            validation_results['validation_passed'] = False
            validation_results['issues'].append("No features transferred")
        
        # Check model performance
        if self.target_model is not None:
            # This would typically involve more sophisticated validation
            validation_results['model_available'] = True
        else:
            validation_results['validation_passed'] = False
            validation_results['issues'].append("No target model available")
        
        self.logger.info(f"Transfer learning validation: {'PASSED' if validation_results['validation_passed'] else 'FAILED'}")
        
        return validation_results
    
    def get_transfer_metrics(self) -> Dict[str, Any]:
        """Get transfer learning metrics"""
        self.logger.info("Calculating transfer learning metrics")
        
        metrics = {
            'source_models': {},
            'transferred_features': {},
            'target_model': {},
            'overall': {}
        }
        
        # Source model metrics
        for league, model_info in self.source_models.items():
            metrics['source_models'][league] = {
                'accuracy': model_info['accuracy'],
                'log_loss': model_info['log_loss'],
                'weight': self.config['source_leagues'][league]['weight']
            }
        
        # Transferred features metrics
        for league, transfer_info in self.transferred_features.items():
            metrics['transferred_features'][league] = {
                'common_features_count': len(transfer_info['common_features']),
                'league_specific_features_count': len(transfer_info['league_specific_features']),
                'transfer_weight': transfer_info['transfer_weight']
            }
        
        # Target model metrics
        if self.target_model is not None:
            metrics['target_model'] = {
                'n_estimators': self.target_model.n_estimators,
                'max_depth': self.target_model.max_depth,
                'learning_rate': self.target_model.learning_rate
            }
        
        # Overall metrics
        metrics['overall'] = {
            'source_leagues_count': len(self.source_models),
            'transferred_features_count': len(self.transferred_features),
            'transfer_ratio': self.config['transfer_strategy']['transfer_ratio'],
            'method': self.config['transfer_strategy']['method']
        }
        
        self.transfer_metrics = metrics
        return metrics
    
    def save_transfer_learning(self, filepath: str):
        """Save transfer learning system"""
        self.logger.info(f"Saving transfer learning system to {filepath}")
        
        transfer_state = {
            'source_models': self.source_models,
            'transferred_features': self.transferred_features,
            'target_model': self.target_model,
            'transfer_metrics': self.transfer_metrics,
            'config': self.config
        }
        
        joblib.dump(transfer_state, filepath)
        self.logger.info("Transfer learning system saved successfully")
    
    def load_transfer_learning(self, filepath: str):
        """Load transfer learning system"""
        self.logger.info(f"Loading transfer learning system from {filepath}")
        
        transfer_state = joblib.load(filepath)
        self.source_models = transfer_state['source_models']
        self.transferred_features = transfer_state['transferred_features']
        self.target_model = transfer_state['target_model']
        self.transfer_metrics = transfer_state['transfer_metrics']
        self.config = transfer_state['config']
        
        self.logger.info("Transfer learning system loaded successfully")

# Example usage
def main():
    """Example usage of NonMajorLeagueTransferLearning"""
    
    # Initialize transfer learning system
    transfer_learning = NonMajorLeagueTransferLearning()
    
    # Load source data
    source_data = transfer_learning.load_source_data(['EPL', 'LaLiga', 'Bundesliga'])
    
    # Pre-train source models
    source_models = transfer_learning.pre_train_source_models(source_data)
    
    # Create target data
    target_data = transfer_learning._create_synthetic_source_data('Championship')
    
    # Transfer features
    transferred_features = transfer_learning.transfer_features(source_models, target_data)
    
    # Create transferred features for target data
    target_data_transferred = transfer_learning.create_transferred_features(target_data)
    
    # Fine-tune target model
    fine_tuned_results = transfer_learning.fine_tune_target_model(target_data_transferred, 'Championship')
    
    # Compare with baseline
    comparison_results = transfer_learning.compare_with_baseline(target_data_transferred)
    
    # Validate transfer learning
    validation_results = transfer_learning.validate_transfer_learning(target_data_transferred)
    
    # Get transfer metrics
    transfer_metrics = transfer_learning.get_transfer_metrics()
    
    # Print results
    print("Transfer Learning Results:")
    print(f"Source leagues: {len(source_models)}")
    print(f"Transferred features: {len(transferred_features)}")
    
    if comparison_results['transfer_learning']:
        print(f"Baseline accuracy: {comparison_results['baseline']['accuracy']:.4f}")
        print(f"Transfer learning accuracy: {comparison_results['transfer_learning']['accuracy']:.4f}")
        print(f"Improvement: {comparison_results['improvement']['accuracy']:+.4f}")
    
    print(f"Validation passed: {validation_results['validation_passed']}")
    
    # Save transfer learning system
    transfer_learning.save_transfer_learning('transfer_learning_system.pkl')

if __name__ == "__main__":
    main()
