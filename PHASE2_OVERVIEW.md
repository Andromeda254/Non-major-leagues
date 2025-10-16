# Phase 2: Model Training & Validation - Overview

**Status:** Ready to Execute  
**Prerequisites:** ✅ Phase 1 Data Collection Complete  
**Dependencies:** ✅ All ML libraries installed

---

## 📋 Phase 2 Components

### 1. Model Architecture (`non_major_league_model_architecture.py`)
**Purpose:** Build and configure ML models for match prediction

**Models Implemented:**
- **XGBoost** - Gradient boosting for high accuracy
- **LightGBM** - Fast gradient boosting
- **Random Forest** - Ensemble decision trees
- **Logistic Regression** - Baseline linear model
- **Neural Networks** - Deep learning approach

**Key Features:**
- Custom architecture for non-major leagues
- Handles imbalanced data
- Feature importance analysis
- Cross-validation support

---

### 2. Ensemble Methods (`non_major_league_ensemble.py`)
**Purpose:** Combine multiple models for better predictions

**Ensemble Strategies:**
- **Weighted Averaging** - Weight models by performance
- **Stacking** - Meta-model learns from base models
- **Voting** - Majority vote from multiple models
- **Blending** - Holdout set for meta-model training

**Benefits:**
- Reduces overfitting
- Improves prediction stability
- Captures different patterns

---

### 3. Transfer Learning (`non_major_league_transfer_learning.py`)
**Purpose:** Leverage knowledge from major leagues

**Approach:**
- Train on major league data (Premier League, La Liga, etc.)
- Fine-tune on non-major league data
- Domain adaptation techniques
- Feature space alignment

**Advantages:**
- Compensates for limited data
- Improves initial performance
- Faster convergence

---

### 4. Hyperparameter Tuning (`non_major_league_hyperparameter_tuning.py`)
**Purpose:** Optimize model parameters

**Methods:**
- **Grid Search** - Exhaustive parameter search
- **Random Search** - Random parameter sampling
- **Bayesian Optimization** - Smart parameter search (Optuna)
- **Genetic Algorithms** - Evolutionary optimization

**Parameters Tuned:**
- Learning rate
- Tree depth
- Number of estimators
- Regularization strength
- Dropout rates

---

### 5. Model Validation (`non_major_league_model_validation.py`)
**Purpose:** Ensure model reliability and performance

**Validation Strategies:**
- **Time Series Split** - Respects temporal order
- **K-Fold Cross-Validation** - Multiple train/test splits
- **Stratified Sampling** - Maintains class distribution
- **Walk-Forward Validation** - Simulates real-world deployment

**Metrics Evaluated:**
- **Accuracy** - Overall correctness
- **Precision** - Positive prediction accuracy
- **Recall** - True positive rate
- **F1 Score** - Harmonic mean of precision/recall
- **ROC-AUC** - Classification performance
- **Log Loss** - Probabilistic accuracy
- **Brier Score** - Calibration quality

---

## 🎯 Phase 2 Workflow

```
1. Load Processed Data
   └─> From Phase 1 output
   └─> Features: 126 columns
   └─> Matches: 470+ samples

2. Feature Selection
   └─> Remove low-variance features
   └─> Correlation analysis
   └─> Feature importance ranking
   └─> Select top N features

3. Train Base Models
   └─> XGBoost
   └─> LightGBM
   └─> Random Forest
   └─> Logistic Regression
   └─> Neural Networks

4. Hyperparameter Tuning
   └─> Optimize each model
   └─> Cross-validation
   └─> Select best parameters

5. Ensemble Creation
   └─> Combine best models
   └─> Weight optimization
   └─> Meta-model training

6. Model Validation
   └─> Time series validation
   └─> Performance metrics
   └─> Calibration assessment
   └─> Overfitting checks

7. Model Selection
   └─> Compare all models
   └─> Select best performer
   └─> Save for deployment

8. Save Outputs
   └─> Trained models (.pkl)
   └─> Performance metrics (.json)
   └─> Feature importance (.csv)
   └─> Validation reports (.txt)
```

---

## 📊 Expected Outputs

### Models Directory
```
./pipeline_output/phase2_output/
├── models/
│   ├── xgboost_model.pkl
│   ├── lightgbm_model.pkl
│   ├── random_forest_model.pkl
│   ├── logistic_regression_model.pkl
│   ├── neural_network_model.pkl
│   └── ensemble_model.pkl
├── feature_importance.csv
├── model_performance.json
├── validation_report.txt
└── phase2_results.json
```

### Performance Metrics
```json
{
  "models": {
    "xgboost": {
      "accuracy": 0.65,
      "precision": 0.63,
      "recall": 0.65,
      "f1_score": 0.64,
      "roc_auc": 0.72
    },
    "random_forest": {
      "accuracy": 0.62,
      "precision": 0.60,
      "recall": 0.62,
      "f1_score": 0.61,
      "roc_auc": 0.69
    },
    "ensemble": {
      "accuracy": 0.68,
      "precision": 0.66,
      "recall": 0.68,
      "f1_score": 0.67,
      "roc_auc": 0.75
    }
  },
  "best_model": "ensemble",
  "training_time": "45.2 seconds",
  "features_used": 45
}
```

---

## 🎯 Performance Targets

### Minimum Acceptable Performance
- **Accuracy:** > 55% (better than random for 3-class problem)
- **ROC-AUC:** > 0.65
- **Calibration:** Brier score < 0.25

### Good Performance
- **Accuracy:** > 60%
- **ROC-AUC:** > 0.70
- **F1 Score:** > 0.58

### Excellent Performance
- **Accuracy:** > 65%
- **ROC-AUC:** > 0.75
- **F1 Score:** > 0.63

---

## 🔧 Configuration

### Model Parameters (Default)

**XGBoost:**
```python
{
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'num_class': 3
}
```

**Random Forest:**
```python
{
    'n_estimators': 100,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt'
}
```

**Logistic Regression:**
```python
{
    'C': 1.0,
    'max_iter': 1000,
    'multi_class': 'multinomial',
    'solver': 'lbfgs'
}
```

---

## 📈 Feature Importance

### Expected Top Features
1. **Recent Form** - Last 5 matches performance
2. **Home/Away Goals** - Historical scoring
3. **Shots on Target** - Shot accuracy
4. **Possession Metrics** - Ball control
5. **Head-to-Head** - Historical matchups
6. **League Position** - Current standings
7. **Goal Difference** - Attack vs defense balance
8. **Home Advantage** - Home vs away performance
9. **Betting Odds** - Market expectations
10. **Momentum** - Win/loss streaks

---

## 🚀 How to Run Phase 2

### Option 1: Full Pipeline
```bash
python3 master_pipeline.py --phase 2 --input-dir ./pipeline_output/phase1_output
```

### Option 2: Simplified Version
```bash
python3 run_phase2_simple.py
```

### Option 3: Individual Components
```bash
# Train specific model
python3 -c "
from non_major_league_model_architecture import NonMajorLeagueModelArchitecture
import pandas as pd

# Load data
data = pd.read_csv('data_collection_sample.csv')

# Initialize and train
model_arch = NonMajorLeagueModelArchitecture()
model = model_arch.train_xgboost(data)
"
```

---

## ✅ Success Criteria

Phase 2 is considered successful when:

1. ✅ All models train without errors
2. ✅ Validation accuracy > 55%
3. ✅ No significant overfitting (train/test gap < 10%)
4. ✅ Models saved successfully
5. ✅ Performance metrics documented
6. ✅ Feature importance analyzed
7. ✅ Ensemble model created
8. ✅ Ready for Phase 3 (Backtesting)

---

## 🔍 Troubleshooting

### Common Issues

**Issue:** "Insufficient data for training"
- **Solution:** Collect more historical data (Phase 1)
- **Workaround:** Use transfer learning from major leagues

**Issue:** "Poor model performance"
- **Solution:** Feature engineering improvements
- **Solution:** Hyperparameter tuning
- **Solution:** Try ensemble methods

**Issue:** "Overfitting detected"
- **Solution:** Increase regularization
- **Solution:** Reduce model complexity
- **Solution:** Add more training data

**Issue:** "Class imbalance"
- **Solution:** Use SMOTE for oversampling
- **Solution:** Adjust class weights
- **Solution:** Use stratified sampling

---

## 📚 Next Steps After Phase 2

1. **Phase 3: Backtesting**
   - Test models on historical data
   - Simulate betting strategies
   - Calculate ROI and profitability

2. **Model Refinement**
   - Analyze prediction errors
   - Improve feature engineering
   - Retrain with more data

3. **Phase 4: Production Deployment**
   - Deploy best model
   - Set up monitoring
   - Implement live predictions

---

## 📊 Expected Timeline

- **Data Loading:** 1-2 minutes
- **Feature Selection:** 2-3 minutes
- **Model Training:** 5-10 minutes per model
- **Hyperparameter Tuning:** 10-30 minutes
- **Ensemble Creation:** 3-5 minutes
- **Validation:** 2-3 minutes
- **Total:** ~30-60 minutes

---

## 🎓 Key Learnings

### For Non-Major Leagues
- Limited data requires careful validation
- Transfer learning is highly beneficial
- Ensemble methods improve stability
- Feature engineering is critical
- Odds data provides strong signals

### Model Selection
- XGBoost typically performs best
- Ensemble methods add 2-5% accuracy
- Simple models (Logistic Regression) provide good baselines
- Neural networks need more data to shine

---

*Ready to proceed with Phase 2 training when you're ready!*
