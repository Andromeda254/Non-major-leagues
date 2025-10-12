import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, log_loss, classification_report, confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeagueModelValidation:
    """
    Robust model validation system for non-major soccer leagues
    
    Key Features:
    - Time series aware validation
    - Conservative cross-validation
    - Model calibration validation
    - Uncertainty quantification
    - Performance stability analysis
    - Overfitting detection
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize model validation system
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.validation_results = {}
        self.calibration_results = {}
        self.stability_analysis = {}
        
    def setup_logging(self):
        """Setup logging for model validation"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load validation configuration"""
        if config is None:
            self.config = {
                'validation_strategies': {
                    'time_series_split': {
                        'enabled': True,
                        'n_splits': 3,  # Conservative for limited data
                        'test_size': 0.2
                    },
                    'walk_forward': {
                        'enabled': True,
                        'n_splits': 3,
                        'min_train_size': 100
                    },
                    'bootstrap': {
                        'enabled': True,
                        'n_bootstrap': 50,
                        'sample_size': 0.8
                    }
                },
                'metrics': {
                    'primary': ['accuracy', 'log_loss'],
                    'secondary': ['precision', 'recall', 'f1_score'],
                    'calibration': ['brier_score', 'reliability_diagram'],
                    'stability': ['coefficient_of_variation', 'confidence_interval']
                },
                'calibration': {
                    'enabled': True,
                    'method': 'isotonic',
                    'cv_folds': 3,
                    'test_size': 0.2
                },
                'stability': {
                    'enabled': True,
                    'min_cv_splits': 3,
                    'max_cv_std': 0.1,
                    'confidence_level': 0.95
                },
                'overfitting_detection': {
                    'enabled': True,
                    'max_train_val_gap': 0.15,
                    'min_validation_accuracy': 0.45,
                    'max_overfitting_ratio': 1.5
                },
                'conservative_settings': {
                    'min_samples_per_split': 20,
                    'min_total_samples': 100,
                    'max_missing_splits': 1,
                    'min_successful_splits': 2
                }
            }
        else:
            self.config = config
    
    def time_series_cross_validation(self, model: Any, X: pd.DataFrame, 
                                   y: pd.Series) -> Dict[str, Any]:
        """Perform time series cross-validation"""
        self.logger.info("Performing time series cross-validation")
        
        if not self.config['validation_strategies']['time_series_split']['enabled']:
            return {}
        
        n_splits = self.config['validation_strategies']['time_series_split']['n_splits']
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        cv_results = {
            'scores': [],
            'train_scores': [],
            'overfitting_gaps': [],
            'fold_details': [],
            'successful_folds': 0,
            'failed_folds': 0
        }
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            self.logger.info(f"Time series CV fold {fold + 1}/{n_splits}")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # Check minimum samples
            min_samples = self.config['conservative_settings']['min_samples_per_split']
            if len(X_train) < min_samples or len(X_val) < min_samples:
                self.logger.warning(f"Fold {fold + 1}: Insufficient samples")
                cv_results['failed_folds'] += 1
                continue
            
            # Track class distribution
            train_dist = pd.Series(y_train).value_counts(normalize=True)
            val_dist = pd.Series(y_val).value_counts(normalize=True)
            
            # Log warning if severe imbalance
            if any(train_dist < 0.15) or any(val_dist < 0.15):
                self.logger.warning(f"Class imbalance detected in fold {fold + 1}")
                self.logger.warning(f"Train distribution: {train_dist.to_dict()}")
                self.logger.warning(f"Val distribution: {val_dist.to_dict()}")
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate on validation set
                val_pred = model.predict(X_val)
                val_proba = model.predict_proba(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                val_logloss = log_loss(y_val, val_proba)
                
                # Evaluate on training set (for overfitting detection)
                train_pred = model.predict(X_train)
                train_proba = model.predict_proba(X_train)
                train_acc = accuracy_score(y_train, train_pred)
                train_logloss = log_loss(y_train, train_proba)
                
                # Calculate overfitting gap
                overfitting_gap = train_acc - val_acc
                
                # Store results
                fold_result = {
                    'fold': fold + 1,
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'train_accuracy': train_acc,
                    'val_accuracy': val_acc,
                    'train_logloss': train_logloss,
                    'val_logloss': val_logloss,
                    'overfitting_gap': overfitting_gap,
                    'success': True
                }
                
                cv_results['scores'].append(val_acc)
                cv_results['train_scores'].append(train_acc)
                cv_results['overfitting_gaps'].append(overfitting_gap)
                cv_results['fold_details'].append(fold_result)
                cv_results['successful_folds'] += 1
                
                self.logger.info(f"Fold {fold + 1} - Val Acc: {val_acc:.4f}, Overfitting: {overfitting_gap:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error in fold {fold + 1}: {e}")
                cv_results['failed_folds'] += 1
                continue
        
        # Calculate summary statistics
        if cv_results['scores']:
            cv_results['mean_score'] = np.mean(cv_results['scores'])
            cv_results['std_score'] = np.std(cv_results['scores'])
            cv_results['mean_overfitting'] = np.mean(cv_results['overfitting_gaps'])
            cv_results['max_overfitting'] = np.max(cv_results['overfitting_gaps'])
            cv_results['coefficient_of_variation'] = cv_results['std_score'] / cv_results['mean_score']
        
        self.logger.info(f"Time series CV complete: {cv_results['successful_folds']} successful folds")
        return cv_results
    
    def walk_forward_validation(self, model: Any, X: pd.DataFrame, 
                              y: pd.Series) -> Dict[str, Any]:
        """Perform walk-forward validation"""
        self.logger.info("Performing walk-forward validation")
        
        if not self.config['validation_strategies']['walk_forward']['enabled']:
            return {}
        
        n_splits = self.config['validation_strategies']['walk_forward']['n_splits']
        min_train_size = self.config['validation_strategies']['walk_forward']['min_train_size']
        
        walk_forward_results = {
            'scores': [],
            'train_sizes': [],
            'val_sizes': [],
            'fold_details': [],
            'successful_folds': 0,
            'failed_folds': 0
        }
        
        # Calculate split sizes
        total_size = len(X)
        split_size = total_size // (n_splits + 1)
        
        for fold in range(n_splits):
            self.logger.info(f"Walk-forward CV fold {fold + 1}/{n_splits}")
            
            # Calculate indices
            train_end = split_size * (fold + 1)
            val_start = train_end
            val_end = split_size * (fold + 2)
            
            if val_end > total_size:
                val_end = total_size
            
            X_train = X.iloc[:train_end]
            y_train = y.iloc[:train_end]
            X_val = X.iloc[val_start:val_end]
            y_val = y.iloc[val_start:val_end]
            
            # Check minimum training size
            if len(X_train) < min_train_size:
                self.logger.warning(f"Fold {fold + 1}: Insufficient training samples")
                walk_forward_results['failed_folds'] += 1
                continue
            
            if len(X_val) == 0:
                self.logger.warning(f"Fold {fold + 1}: Empty validation set")
                walk_forward_results['failed_folds'] += 1
                continue
            
            try:
                # Train model
                model.fit(X_train, y_train)
                
                # Evaluate
                val_pred = model.predict(X_val)
                val_proba = model.predict_proba(X_val)
                val_acc = accuracy_score(y_val, val_pred)
                val_logloss = log_loss(y_val, val_proba)
                
                # Store results
                fold_result = {
                    'fold': fold + 1,
                    'train_size': len(X_train),
                    'val_size': len(X_val),
                    'val_accuracy': val_acc,
                    'val_logloss': val_logloss,
                    'success': True
                }
                
                walk_forward_results['scores'].append(val_acc)
                walk_forward_results['train_sizes'].append(len(X_train))
                walk_forward_results['val_sizes'].append(len(X_val))
                walk_forward_results['fold_details'].append(fold_result)
                walk_forward_results['successful_folds'] += 1
                
                self.logger.info(f"Fold {fold + 1} - Val Acc: {val_acc:.4f}")
                
            except Exception as e:
                self.logger.error(f"Error in fold {fold + 1}: {e}")
                walk_forward_results['failed_folds'] += 1
                continue
        
        # Calculate summary statistics
        if walk_forward_results['scores']:
            walk_forward_results['mean_score'] = np.mean(walk_forward_results['scores'])
            walk_forward_results['std_score'] = np.std(walk_forward_results['scores'])
            walk_forward_results['coefficient_of_variation'] = (
                walk_forward_results['std_score'] / walk_forward_results['mean_score']
            )
        
        self.logger.info(f"Walk-forward CV complete: {walk_forward_results['successful_folds']} successful folds")
        return walk_forward_results
    
    def bootstrap_validation(self, model: Any, X: pd.DataFrame, 
                           y: pd.Series) -> Dict[str, Any]:
        """Perform bootstrap validation"""
        self.logger.info("Performing bootstrap validation")
        
        if not self.config['validation_strategies']['bootstrap']['enabled']:
            return {}
        
        n_bootstrap = self.config['validation_strategies']['bootstrap']['n_bootstrap']
        sample_size = self.config['validation_strategies']['bootstrap']['sample_size']
        
        bootstrap_results = {
            'scores': [],
            'out_of_bag_scores': [],
            'bootstrap_details': [],
            'successful_bootstraps': 0,
            'failed_bootstraps': 0
        }
        
        for i in range(n_bootstrap):
            self.logger.info(f"Bootstrap {i + 1}/{n_bootstrap}")
            
            try:
                # Bootstrap sample
                n_samples = int(len(X) * sample_size)
                bootstrap_indices = np.random.choice(len(X), size=n_samples, replace=True)
                oob_indices = np.setdiff1d(np.arange(len(X)), bootstrap_indices)
                
                X_bootstrap = X.iloc[bootstrap_indices]
                y_bootstrap = y.iloc[bootstrap_indices]
                
                if len(oob_indices) == 0:
                    self.logger.warning(f"Bootstrap {i + 1}: No out-of-bag samples")
                    bootstrap_results['failed_bootstraps'] += 1
                    continue
                
                X_oob = X.iloc[oob_indices]
                y_oob = y.iloc[oob_indices]
                
                # Train model
                model.fit(X_bootstrap, y_bootstrap)
                
                # Evaluate on bootstrap sample
                bootstrap_pred = model.predict(X_bootstrap)
                bootstrap_proba = model.predict_proba(X_bootstrap)
                bootstrap_acc = accuracy_score(y_bootstrap, bootstrap_pred)
                bootstrap_logloss = log_loss(y_bootstrap, bootstrap_proba)
                
                # Evaluate on out-of-bag sample
                oob_pred = model.predict(X_oob)
                oob_proba = model.predict_proba(X_oob)
                oob_acc = accuracy_score(y_oob, oob_pred)
                oob_logloss = log_loss(y_oob, oob_proba)
                
                # Store results
                bootstrap_result = {
                    'bootstrap': i + 1,
                    'bootstrap_size': len(X_bootstrap),
                    'oob_size': len(X_oob),
                    'bootstrap_accuracy': bootstrap_acc,
                    'oob_accuracy': oob_acc,
                    'bootstrap_logloss': bootstrap_logloss,
                    'oob_logloss': oob_logloss,
                    'success': True
                }
                
                bootstrap_results['scores'].append(bootstrap_acc)
                bootstrap_results['out_of_bag_scores'].append(oob_acc)
                bootstrap_results['bootstrap_details'].append(bootstrap_result)
                bootstrap_results['successful_bootstraps'] += 1
                
            except Exception as e:
                self.logger.error(f"Error in bootstrap {i + 1}: {e}")
                bootstrap_results['failed_bootstraps'] += 1
                continue
        
        # Calculate summary statistics
        if bootstrap_results['scores']:
            bootstrap_results['mean_score'] = np.mean(bootstrap_results['scores'])
            bootstrap_results['std_score'] = np.std(bootstrap_results['scores'])
            bootstrap_results['mean_oob_score'] = np.mean(bootstrap_results['out_of_bag_scores'])
            bootstrap_results['std_oob_score'] = np.std(bootstrap_results['out_of_bag_scores'])
            bootstrap_results['optimism'] = bootstrap_results['mean_score'] - bootstrap_results['mean_oob_score']
        
        self.logger.info(f"Bootstrap validation complete: {bootstrap_results['successful_bootstraps']} successful bootstraps")
        return bootstrap_results
    
    def validate_model_calibration(self, model: Any, X: pd.DataFrame, 
                                 y: pd.Series) -> Dict[str, Any]:
        """Validate model calibration"""
        self.logger.info("Validating model calibration")
        
        if not self.config['calibration']['enabled']:
            return {}
        
        try:
            # Split data for calibration
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config['calibration']['test_size'], 
                random_state=42
            )
            
            # Train model
            model.fit(X_train, y_train)
            
            # Get predictions
            test_proba = model.predict_proba(X_test)
            test_pred = model.predict(X_test)
            
            # Calculate calibration metrics
            calibration_results = {
                'brier_score': self._calculate_brier_score(y_test, test_proba),
                'reliability': self._calculate_reliability(y_test, test_proba),
                'sharpness': self._calculate_sharpness(test_proba),
                'calibration_error': self._calculate_calibration_error(y_test, test_proba)
            }
            
            # Calibrate model
            calibrated_model = CalibratedClassifierCV(
                model, 
                method=self.config['calibration']['method'],
                cv=self.config['calibration']['cv_folds']
            )
            calibrated_model.fit(X_train, y_train)
            
            # Evaluate calibrated model
            calibrated_proba = calibrated_model.predict_proba(X_test)
            calibrated_pred = calibrated_model.predict(X_test)
            
            calibrated_results = {
                'brier_score': self._calculate_brier_score(y_test, calibrated_proba),
                'reliability': self._calculate_reliability(y_test, calibrated_proba),
                'sharpness': self._calculate_sharpness(calibrated_proba),
                'calibration_error': self._calculate_calibration_error(y_test, calibrated_proba)
            }
            
            # Compare calibration
            calibration_comparison = {
                'original': calibration_results,
                'calibrated': calibrated_results,
                'improvement': {
                    'brier_score': calibration_results['brier_score'] - calibrated_results['brier_score'],
                    'calibration_error': calibration_results['calibration_error'] - calibrated_results['calibration_error']
                }
            }
            
            self.calibration_results = calibration_comparison
            self.logger.info("Model calibration validation complete")
            
            return calibration_comparison
            
        except Exception as e:
            self.logger.error(f"Error validating calibration: {e}")
            return {'error': str(e)}
    
    def _calculate_brier_score(self, y_true: pd.Series, y_proba: np.ndarray) -> float:
        """Calculate Brier score for calibration"""
        try:
            # Convert y_true to one-hot encoding
            y_onehot = np.zeros((len(y_true), 3))
            for i, label in enumerate(y_true):
                y_onehot[i, label] = 1
            
            # Calculate Brier score
            brier_score = np.mean(np.sum((y_onehot - y_proba) ** 2, axis=1))
            return brier_score
            
        except Exception as e:
            self.logger.error(f"Error calculating Brier score: {e}")
            return float('inf')
    
    def _calculate_reliability(self, y_true: pd.Series, y_proba: np.ndarray) -> float:
        """Calculate reliability for calibration"""
        try:
            # This is a simplified reliability calculation
            # In practice, you would bin predictions and calculate reliability
            max_proba = np.max(y_proba, axis=1)
            predicted_class = np.argmax(y_proba, axis=1)
            correct_predictions = (predicted_class == y_true).astype(int)
            
            # Calculate reliability as correlation between confidence and accuracy
            reliability = np.corrcoef(max_proba, correct_predictions)[0, 1]
            return reliability if not np.isnan(reliability) else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating reliability: {e}")
            return 0.0
    
    def _calculate_sharpness(self, y_proba: np.ndarray) -> float:
        """Calculate sharpness for calibration"""
        try:
            # Sharpness is the variance of the predicted probabilities
            sharpness = np.var(y_proba)
            return sharpness
            
        except Exception as e:
            self.logger.error(f"Error calculating sharpness: {e}")
            return 0.0
    
    def _calculate_calibration_error(self, y_true: pd.Series, y_proba: np.ndarray) -> float:
        """Calculate calibration error"""
        try:
            # Simplified calibration error calculation
            max_proba = np.max(y_proba, axis=1)
            predicted_class = np.argmax(y_proba, axis=1)
            correct_predictions = (predicted_class == y_true).astype(int)
            
            # Calculate expected calibration error
            calibration_error = np.mean(np.abs(max_proba - correct_predictions))
            return calibration_error
            
        except Exception as e:
            self.logger.error(f"Error calculating calibration error: {e}")
            return float('inf')
    
    def detect_overfitting(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Detect overfitting in validation results"""
        self.logger.info("Detecting overfitting")
        
        if not self.config['overfitting_detection']['enabled']:
            return {}
        
        overfitting_results = {
            'detected': False,
            'severity': 'none',
            'metrics': {},
            'recommendations': []
        }
        
        # Check time series CV results
        if 'time_series_cv' in validation_results:
            ts_cv = validation_results['time_series_cv']
            
            if 'overfitting_gaps' in ts_cv and ts_cv['overfitting_gaps']:
                mean_overfitting = np.mean(ts_cv['overfitting_gaps'])
                max_overfitting = np.max(ts_cv['overfitting_gaps'])
                
                overfitting_results['metrics']['mean_overfitting'] = mean_overfitting
                overfitting_results['metrics']['max_overfitting'] = max_overfitting
                
                # Check thresholds
                max_threshold = self.config['overfitting_detection']['max_train_val_gap']
                
                if mean_overfitting > max_threshold:
                    overfitting_results['detected'] = True
                    overfitting_results['severity'] = 'high' if mean_overfitting > max_threshold * 2 else 'medium'
                    overfitting_results['recommendations'].append(
                        f"High overfitting detected: {mean_overfitting:.4f} > {max_threshold:.4f}"
                    )
        
        # Check validation accuracy
        if 'time_series_cv' in validation_results:
            ts_cv = validation_results['time_series_cv']
            
            if 'mean_score' in ts_cv:
                val_acc = ts_cv['mean_score']
                min_threshold = self.config['overfitting_detection']['min_validation_accuracy']
                
                if val_acc < min_threshold:
                    overfitting_results['detected'] = True
                    overfitting_results['severity'] = 'high'
                    overfitting_results['recommendations'].append(
                        f"Low validation accuracy: {val_acc:.4f} < {min_threshold:.4f}"
                    )
        
        # Check coefficient of variation
        if 'time_series_cv' in validation_results:
            ts_cv = validation_results['time_series_cv']
            
            if 'coefficient_of_variation' in ts_cv:
                cv = ts_cv['coefficient_of_variation']
                max_cv = self.config['stability']['max_cv_std']
                
                if cv > max_cv:
                    overfitting_results['detected'] = True
                    overfitting_results['severity'] = 'medium'
                    overfitting_results['recommendations'].append(
                        f"High coefficient of variation: {cv:.4f} > {max_cv:.4f}"
                    )
        
        self.logger.info(f"Overfitting detection: {'DETECTED' if overfitting_results['detected'] else 'NOT DETECTED'}")
        return overfitting_results
    
    def analyze_stability(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze model stability"""
        self.logger.info("Analyzing model stability")
        
        if not self.config['stability']['enabled']:
            return {}
        
        stability_results = {
            'stable': True,
            'metrics': {},
            'issues': []
        }
        
        # Analyze time series CV stability
        if 'time_series_cv' in validation_results:
            ts_cv = validation_results['time_series_cv']
            
            if 'coefficient_of_variation' in ts_cv:
                cv = ts_cv['coefficient_of_variation']
                stability_results['metrics']['coefficient_of_variation'] = cv
                
                max_cv = self.config['stability']['max_cv_std']
                if cv > max_cv:
                    stability_results['stable'] = False
                    stability_results['issues'].append(f"High CV: {cv:.4f} > {max_cv:.4f}")
            
            if 'std_score' in ts_cv:
                std_score = ts_cv['std_score']
                stability_results['metrics']['std_score'] = std_score
        
        # Analyze walk-forward stability
        if 'walk_forward' in validation_results:
            wf = validation_results['walk_forward']
            
            if 'coefficient_of_variation' in wf:
                cv = wf['coefficient_of_variation']
                stability_results['metrics']['walk_forward_cv'] = cv
                
                max_cv = self.config['stability']['max_cv_std']
                if cv > max_cv:
                    stability_results['stable'] = False
                    stability_results['issues'].append(f"High WF CV: {cv:.4f} > {max_cv:.4f}")
        
        # Analyze bootstrap stability
        if 'bootstrap' in validation_results:
            bootstrap = validation_results['bootstrap']
            
            if 'optimism' in bootstrap:
                optimism = bootstrap['optimism']
                stability_results['metrics']['optimism'] = optimism
                
                if optimism > 0.1:  # 10% optimism threshold
                    stability_results['stable'] = False
                    stability_results['issues'].append(f"High optimism: {optimism:.4f}")
        
        self.stability_analysis = stability_results
        self.logger.info(f"Stability analysis: {'STABLE' if stability_results['stable'] else 'UNSTABLE'}")
        return stability_results
    
    def comprehensive_validation(self, model: Any, X: pd.DataFrame, 
                                y: pd.Series) -> Dict[str, Any]:
        """Perform comprehensive model validation"""
        self.logger.info("Performing comprehensive model validation")
        
        # Check minimum requirements
        min_samples = self.config['conservative_settings']['min_total_samples']
        if len(X) < min_samples:
            self.logger.error(f"Insufficient samples: {len(X)} < {min_samples}")
            return {'error': 'Insufficient samples'}
        
        validation_results = {}
        
        try:
            # Time series cross-validation
            ts_cv_results = self.time_series_cross_validation(model, X, y)
            validation_results['time_series_cv'] = ts_cv_results
            
            # Walk-forward validation
            wf_results = self.walk_forward_validation(model, X, y)
            validation_results['walk_forward'] = wf_results
            
            # Bootstrap validation
            bootstrap_results = self.bootstrap_validation(model, X, y)
            validation_results['bootstrap'] = bootstrap_results
            
            # Model calibration
            calibration_results = self.validate_model_calibration(model, X, y)
            validation_results['calibration'] = calibration_results
            
            # Overfitting detection
            overfitting_results = self.detect_overfitting(validation_results)
            validation_results['overfitting'] = overfitting_results
            
            # Stability analysis
            stability_results = self.analyze_stability(validation_results)
            validation_results['stability'] = stability_results
            
            # Overall validation assessment
            overall_assessment = self._assess_overall_validation(validation_results)
            validation_results['overall'] = overall_assessment
            
            self.validation_results = validation_results
            self.logger.info("Comprehensive validation complete")
            
            return validation_results
            
        except Exception as e:
            self.logger.error(f"Error in comprehensive validation: {e}")
            return {'error': str(e)}
    
    def _assess_overall_validation(self, validation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess overall validation results"""
        overall_assessment = {
            'validation_passed': True,
            'overall_score': 0.0,
            'issues': [],
            'recommendations': []
        }
        
        # Check if validation passed
        if 'overfitting' in validation_results and validation_results['overfitting']['detected']:
            overall_assessment['validation_passed'] = False
            overall_assessment['issues'].extend(validation_results['overfitting']['recommendations'])
        
        if 'stability' in validation_results and not validation_results['stability']['stable']:
            overall_assessment['validation_passed'] = False
            overall_assessment['issues'].extend(validation_results['stability']['issues'])
        
        # Calculate overall score
        scores = []
        
        if 'time_series_cv' in validation_results and 'mean_score' in validation_results['time_series_cv']:
            scores.append(validation_results['time_series_cv']['mean_score'])
        
        if 'walk_forward' in validation_results and 'mean_score' in validation_results['walk_forward']:
            scores.append(validation_results['walk_forward']['mean_score'])
        
        if 'bootstrap' in validation_results and 'mean_score' in validation_results['bootstrap']:
            scores.append(validation_results['bootstrap']['mean_score'])
        
        if scores:
            overall_assessment['overall_score'] = np.mean(scores)
        
        # Generate recommendations
        if overall_assessment['validation_passed']:
            overall_assessment['recommendations'].append("Model validation passed successfully")
        else:
            overall_assessment['recommendations'].append("Model validation failed - review issues")
        
        return overall_assessment
    
    def generate_validation_report(self, validation_results: Dict[str, Any], 
                                 save_path: str = None) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("=" * 80)
        report.append("NON-MAJOR LEAGUE MODEL VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Overall assessment
        if 'overall' in validation_results:
            overall = validation_results['overall']
            report.append("OVERALL ASSESSMENT:")
            report.append(f"  Validation Status: {'PASSED' if overall['validation_passed'] else 'FAILED'}")
            report.append(f"  Overall Score: {overall['overall_score']:.4f}")
            report.append("")
            
            if overall['issues']:
                report.append("  Issues:")
                for issue in overall['issues']:
                    report.append(f"    - {issue}")
                report.append("")
            
            if overall['recommendations']:
                report.append("  Recommendations:")
                for rec in overall['recommendations']:
                    report.append(f"    - {rec}")
                report.append("")
        
        # Detailed results
        report.append("DETAILED VALIDATION RESULTS:")
        report.append("-" * 40)
        
        # Time series CV
        if 'time_series_cv' in validation_results:
            ts_cv = validation_results['time_series_cv']
            report.append("\nTime Series Cross-Validation:")
            report.append(f"  Successful folds: {ts_cv.get('successful_folds', 0)}")
            report.append(f"  Failed folds: {ts_cv.get('failed_folds', 0)}")
            if 'mean_score' in ts_cv:
                report.append(f"  Mean accuracy: {ts_cv['mean_score']:.4f}")
                report.append(f"  Std accuracy: {ts_cv['std_score']:.4f}")
                report.append(f"  Coefficient of variation: {ts_cv['coefficient_of_variation']:.4f}")
            if 'mean_overfitting' in ts_cv:
                report.append(f"  Mean overfitting: {ts_cv['mean_overfitting']:.4f}")
                report.append(f"  Max overfitting: {ts_cv['max_overfitting']:.4f}")
        
        # Walk-forward validation
        if 'walk_forward' in validation_results:
            wf = validation_results['walk_forward']
            report.append("\nWalk-Forward Validation:")
            report.append(f"  Successful folds: {wf.get('successful_folds', 0)}")
            report.append(f"  Failed folds: {wf.get('failed_folds', 0)}")
            if 'mean_score' in wf:
                report.append(f"  Mean accuracy: {wf['mean_score']:.4f}")
                report.append(f"  Std accuracy: {wf['std_score']:.4f}")
                report.append(f"  Coefficient of variation: {wf['coefficient_of_variation']:.4f}")
        
        # Bootstrap validation
        if 'bootstrap' in validation_results:
            bootstrap = validation_results['bootstrap']
            report.append("\nBootstrap Validation:")
            report.append(f"  Successful bootstraps: {bootstrap.get('successful_bootstraps', 0)}")
            report.append(f"  Failed bootstraps: {bootstrap.get('failed_bootstraps', 0)}")
            if 'mean_score' in bootstrap:
                report.append(f"  Mean accuracy: {bootstrap['mean_score']:.4f}")
                report.append(f"  Mean OOB accuracy: {bootstrap['mean_oob_score']:.4f}")
                report.append(f"  Optimism: {bootstrap['optimism']:.4f}")
        
        # Calibration
        if 'calibration' in validation_results:
            calibration = validation_results['calibration']
            report.append("\nModel Calibration:")
            if 'original' in calibration:
                orig = calibration['original']
                report.append(f"  Original Brier score: {orig['brier_score']:.4f}")
                report.append(f"  Original calibration error: {orig['calibration_error']:.4f}")
            if 'calibrated' in calibration:
                cal = calibration['calibrated']
                report.append(f"  Calibrated Brier score: {cal['brier_score']:.4f}")
                report.append(f"  Calibrated calibration error: {cal['calibration_error']:.4f}")
            if 'improvement' in calibration:
                imp = calibration['improvement']
                report.append(f"  Brier score improvement: {imp['brier_score']:+.4f}")
                report.append(f"  Calibration error improvement: {imp['calibration_error']:+.4f}")
        
        # Overfitting detection
        if 'overfitting' in validation_results:
            overfitting = validation_results['overfitting']
            report.append("\nOverfitting Detection:")
            report.append(f"  Overfitting detected: {'YES' if overfitting['detected'] else 'NO'}")
            report.append(f"  Severity: {overfitting['severity']}")
            if overfitting['recommendations']:
                report.append("  Recommendations:")
                for rec in overfitting['recommendations']:
                    report.append(f"    - {rec}")
        
        # Stability analysis
        if 'stability' in validation_results:
            stability = validation_results['stability']
            report.append("\nStability Analysis:")
            report.append(f"  Model stable: {'YES' if stability['stable'] else 'NO'}")
            if stability['issues']:
                report.append("  Issues:")
                for issue in stability['issues']:
                    report.append(f"    - {issue}")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Validation report saved to {save_path}")
        
        return report_text
    
    def save_validation_results(self, filepath: str):
        """Save validation results"""
        self.logger.info(f"Saving validation results to {filepath}")
        
        validation_state = {
            'validation_results': self.validation_results,
            'calibration_results': self.calibration_results,
            'stability_analysis': self.stability_analysis,
            'config': self.config
        }
        
        joblib.dump(validation_state, filepath)
        self.logger.info("Validation results saved successfully")

# Example usage
def main():
    """Example usage of NonMajorLeagueModelValidation"""
    
    # Create sample data
    np.random.seed(42)
    n_samples = 500
    n_features = 15
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                     columns=[f'feature_{i}' for i in range(n_features)])
    y = pd.Series(np.random.choice([0, 1, 2], n_samples))
    
    # Create a simple model
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    
    # Initialize validation
    validator = NonMajorLeagueModelValidation()
    
    # Perform comprehensive validation
    validation_results = validator.comprehensive_validation(model, X, y)
    
    # Generate report
    report = validator.generate_validation_report(validation_results, 'validation_report.txt')
    print(report)
    
    # Save results
    validator.save_validation_results('validation_results.pkl')

if __name__ == "__main__":
    main()
