#!/usr/bin/env python3
"""
Simplified Phase 2 Runner - Train models on collected data
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("PHASE 2: MODEL TRAINING (SIMPLIFIED)")
print("=" * 80)

# Step 1: Load the collected data
logger.info("Loading collected data...")
try:
    # Use the sample data we collected earlier
    df = pd.read_csv('data_collection_sample.csv')
    logger.info(f"✅ Loaded {len(df)} matches with {len(df.columns)} features")
except Exception as e:
    logger.error(f"Failed to load data: {e}")
    sys.exit(1)

# Step 2: Prepare features and target
logger.info("Preparing features and target variable...")

# Select relevant features
feature_cols = ['FTHG', 'FTAG', 'HTHG', 'HTAG', 'HS', 'AS', 'HST', 'AST', 
                'HC', 'AC', 'HF', 'AF', 'HY', 'AY']

# Check which features are available
available_features = [col for col in feature_cols if col in df.columns]
logger.info(f"Using {len(available_features)} features: {available_features}")

# Target variable: Match result (FTR)
if 'FTR' not in df.columns:
    logger.error("Target variable 'FTR' not found in data")
    sys.exit(1)

# Prepare data
X = df[available_features].fillna(0)
y = df['FTR']

# Encode target variable
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

logger.info(f"Features shape: {X.shape}")
logger.info(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")

# Step 3: Split data
logger.info("Splitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

logger.info(f"Training set: {len(X_train)} samples")
logger.info(f"Test set: {len(X_test)} samples")

# Step 4: Train models
print("\n" + "=" * 80)
print("TRAINING MODELS")
print("=" * 80)

models = {}
results = {}

# Model 1: Random Forest
logger.info("\n1. Training Random Forest...")
try:
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    
    models['random_forest'] = rf_model
    results['random_forest'] = {
        'accuracy': accuracy_rf,
        'predictions': y_pred_rf
    }
    
    logger.info(f"✅ Random Forest Accuracy: {accuracy_rf:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': available_features,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    logger.info("\nTop 5 Important Features:")
    for idx, row in feature_importance.head().iterrows():
        logger.info(f"   {row['feature']}: {row['importance']:.4f}")
        
except Exception as e:
    logger.error(f"Random Forest training failed: {e}")

# Model 2: Logistic Regression
logger.info("\n2. Training Logistic Regression...")
try:
    lr_model = LogisticRegression(
        max_iter=1000,
        random_state=42,
        multi_class='multinomial'
    )
    lr_model.fit(X_train, y_train)
    
    # Predictions
    y_pred_lr = lr_model.predict(X_test)
    accuracy_lr = accuracy_score(y_test, y_pred_lr)
    
    models['logistic_regression'] = lr_model
    results['logistic_regression'] = {
        'accuracy': accuracy_lr,
        'predictions': y_pred_lr
    }
    
    logger.info(f"✅ Logistic Regression Accuracy: {accuracy_lr:.4f}")
    
except Exception as e:
    logger.error(f"Logistic Regression training failed: {e}")

# Step 5: Model Evaluation
print("\n" + "=" * 80)
print("MODEL EVALUATION")
print("=" * 80)

best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
best_accuracy = results[best_model_name]['accuracy']

logger.info(f"\n🏆 Best Model: {best_model_name}")
logger.info(f"   Accuracy: {best_accuracy:.4f}")

# Detailed classification report for best model
logger.info(f"\nDetailed Classification Report ({best_model_name}):")
y_pred_best = results[best_model_name]['predictions']
report = classification_report(y_test, y_pred_best, target_names=le.classes_)
print(report)

# Step 6: Save models
print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

output_dir = Path("./pipeline_output/phase2_output")
output_dir.mkdir(parents=True, exist_ok=True)

for model_name, model in models.items():
    model_path = output_dir / f"{model_name}.pkl"
    joblib.dump(model, model_path)
    logger.info(f"✅ Saved {model_name} to {model_path}")

# Save label encoder
le_path = output_dir / "label_encoder.pkl"
joblib.dump(le, le_path)
logger.info(f"✅ Saved label encoder to {le_path}")

# Save feature names
feature_path = output_dir / "feature_names.txt"
with open(feature_path, 'w') as f:
    f.write('\n'.join(available_features))
logger.info(f"✅ Saved feature names to {feature_path}")

# Save results summary
results_summary = {
    'models': {
        name: {'accuracy': res['accuracy']}
        for name, res in results.items()
    },
    'best_model': best_model_name,
    'best_accuracy': best_accuracy,
    'features_used': available_features,
    'training_samples': len(X_train),
    'test_samples': len(X_test),
    'target_classes': le.classes_.tolist()
}

import json
summary_path = output_dir / "phase2_results.json"
with open(summary_path, 'w') as f:
    json.dump(results_summary, f, indent=2)
logger.info(f"✅ Saved results summary to {summary_path}")

# Final summary
print("\n" + "=" * 80)
print("✅ PHASE 2 COMPLETE")
print("=" * 80)
print(f"\n📊 Summary:")
print(f"   • Models trained: {len(models)}")
print(f"   • Best model: {best_model_name}")
print(f"   • Best accuracy: {best_accuracy:.4f}")
print(f"   • Features used: {len(available_features)}")
print(f"   • Training samples: {len(X_train)}")
print(f"   • Test samples: {len(X_test)}")
print(f"\n📁 Output directory: {output_dir}")
print(f"\n🎯 Next steps:")
print(f"   • Review model performance")
print(f"   • Run Phase 3 for backtesting")
print(f"   • Deploy best model to production")
print("=" * 80)
