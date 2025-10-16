# ML Workflow Quick Start Guide

**Status:** ✅ Production Ready  
**Last Verified:** October 16, 2025

---

## 🚀 Quick Start

### **Option 1: Full Workflow (Train New Models)**
```bash
python3 -u run_complete_workflow.py \
    --crawler-file soccer-match-intelligence/filtered_3am_5am_20251016.json \
    --league E1
```

**Duration:** ~30-40 minutes  
**What it does:**
- Phase 1: Feature engineering from historical data
- Phase 2: Train models with hyperparameter optimization
- Phase 3: Backtest models on historical data
- Phase 4: Generate predictions for upcoming matches

---

### **Option 2: Fast Predictions (Use Existing Models)**
```bash
python3 -u run_complete_workflow.py \
    --crawler-file soccer-match-intelligence/filtered_3am_5am_20251016.json \
    --league E1 \
    --skip-training
```

**Duration:** ~30 seconds  
**What it does:**
- Loads pre-trained models
- Generates predictions for upcoming matches
- Skips Phase 1, 2, and 3

---

## 📊 Workflow Phases

### **Phase 1: Feature Engineering** (11.59s)
- Collects historical match data
- Engineers features (form, consistency, market features)
- Validates and preprocesses data
- **Output:** `pipeline_output/phase1_output/E1_features.csv`

### **Phase 2: Model Training** (~30 min)
- Trains 4 base models (XGBoost, LightGBM, Random Forest, Logistic Regression)
- Hyperparameter optimization (30 trials per model)
- Creates ensemble model
- Model calibration
- **Output:** `pipeline_output/phase2_output/E1_ensemble.pkl`

### **Phase 3: Backtesting** (0.72s)
- Backtests models on historical data
- Implements betting strategy
- Calculates performance metrics
- Risk management assessment
- Live testing simulation (paper trades)
- **Output:** `./results/E1_backtesting_results.pkl`

### **Phase 4: Predictions** (instant)
- Generates predictions for upcoming matches
- Calculates confidence scores
- Exports to JSON and CSV
- **Output:** `predictions_output/predictions_YYYYMMDD_HHMMSS.json`

---

## 📁 Output Files

### **Predictions**
```
predictions_output/
├── predictions_YYYYMMDD_HHMMSS.json    # Full predictions with probabilities
├── predictions_YYYYMMDD_HHMMSS.csv     # CSV format for spreadsheets
├── workflow_YYYYMMDD_HHMMSS.json       # Workflow metadata
└── summary_YYYYMMDD_HHMMSS.txt         # Human-readable summary
```

### **Models**
```
pipeline_output/
├── phase1_output/
│   ├── E1_features.csv                 # Engineered features
│   ├── E1_validation_report.txt        # Data validation report
│   └── E1_phase1_summary.json          # Phase 1 summary
└── phase2_output/
    ├── E1_ensemble.pkl                 # Trained ensemble model
    ├── E1_hyperparameter_tuning.pkl    # Optimization results
    └── E1_phase2_summary.json          # Phase 2 summary
```

### **Backtesting Results**
```
results/
├── E1_backtesting_results.pkl          # Backtest metrics
├── E1_betting_strategy.pkl             # Strategy state
├── E1_performance_metrics.pkl          # Performance analysis
├── E1_risk_management.pkl              # Risk assessment
└── E1_live_testing.pkl                 # Paper trading results
```

---

## 📈 Prediction Format

### **JSON Output**
```json
{
  "match_info": {
    "date": "2025-10-16",
    "home_team": "Atletico Mineiro MG",
    "away_team": "Cruzeiro EC",
    "league": "E1"
  },
  "predictions": {
    "1x2": {
      "prediction": 2,
      "outcome": "Home Win",
      "confidence": 0.9799,
      "probabilities": {
        "home_win": 0.9799,
        "draw": 0.0072,
        "away_win": 0.0128
      },
      "recommended": true
    },
    "over_under_2.5": {
      "prediction": "Over",
      "confidence": 0.70
    },
    "btts": {
      "prediction": "Yes",
      "confidence": 0.68
    }
  }
}
```

### **CSV Output**
```csv
Date,HomeTeam,AwayTeam,League,1X2_Prediction,1X2_Confidence,OU_Prediction,OU_Confidence,BTTS_Prediction,BTTS_Confidence
2025-10-16,Atletico Mineiro MG,Cruzeiro EC,E1,Home Win,97.99%,Over,70.00%,Yes,68.00%
```

---

## 🔧 Command Line Options

```bash
python3 run_complete_workflow.py [OPTIONS]

Required:
  --crawler-file PATH    Path to crawler JSON file with upcoming matches
  --league CODE          League code (e.g., E1 for English Championship)

Optional:
  --skip-training        Skip Phase 1, 2, 3 and use existing models
  -h, --help            Show help message
```

---

## 📊 Model Performance

### **Hyperparameter Optimization Results**
- **XGBoost:** Best score: 0.0102 (30 trials)
- **LightGBM:** Best score: 0.0014 (30 trials)
- **Random Forest:** Best score: 0.0000 (30 trials)
- **Logistic Regression:** Best score: 0.0020 (30 trials)

### **Ensemble Performance**
- **Best Model:** Random Forest
- **Validation Accuracy:** 100%
- **Calibration:** Applied to all models

---

## ⚠️ Requirements

### **System Requirements**
- Python 3.13+
- 4GB+ RAM
- 2GB+ disk space

### **Python Packages**
```bash
pip install pandas numpy scikit-learn xgboost lightgbm optuna joblib
```

### **Input Data**
- Crawler file with upcoming matches (JSON format)
- Historical data (automatically fetched or use existing)

---

## 🐛 Troubleshooting

### **Issue: "No trained model found"**
**Solution:** Run without `--skip-training` to train models first

### **Issue: "No features file found"**
**Solution:** Run Phase 1 first or ensure `./data/E1_features.csv` exists

### **Issue: "Feature mismatch"**
**Solution:** Retrain models - feature set has changed

### **Issue: "Connection refused" during data collection**
**Solution:** Use existing data or check internet connection

---

## 📝 Logs

### **Monitor Workflow Progress**
```bash
# Run with output to terminal and log file
python3 -u run_complete_workflow.py \
    --crawler-file soccer-match-intelligence/filtered_3am_5am_20251016.json \
    --league E1 \
    2>&1 | tee workflow.log

# Monitor in real-time
tail -f workflow.log
```

### **Check Specific Phase**
```bash
# Phase 3 execution
grep "Phase 3" workflow.log

# Hyperparameter tuning progress
grep "Trial.*finished" workflow.log

# Errors only
grep -i "error\|failed" workflow.log
```

---

## 🎯 Next Steps

1. **Review Predictions:** Check `predictions_output/` for results
2. **Analyze Backtest:** Review `results/E1_backtesting_results.pkl`
3. **Monitor Performance:** Track prediction accuracy over time
4. **Adjust Strategy:** Modify betting strategy based on results
5. **Retrain Models:** Periodically retrain with new data

---

## 📚 Additional Documentation

- **Complete Fixes:** See `COMPLETE_WORKFLOW_FIXES_SUMMARY.md`
- **Phase 3 Details:** See `ALL_PHASE3_FIXES_SUMMARY.md`
- **Prediction Workflow:** See `COMPLETE_PREDICTION_WORKFLOW.md`

---

## ✅ Verification

To verify the workflow is working:

```bash
# Quick test with existing models
python3 -u run_complete_workflow.py \
    --crawler-file soccer-match-intelligence/filtered_3am_5am_20251016.json \
    --league E1 \
    --skip-training

# Check predictions were generated
ls -lh predictions_output/predictions_*.json

# Verify prediction count
python3 -c "import json; print(len(json.load(open('predictions_output/predictions_20251016_233929.json'))))"
```

Expected output: `31` (number of matches)

---

## 🎉 Success Indicators

✅ All phases complete without errors  
✅ Predictions generated for all matches  
✅ Confidence scores between 0-100%  
✅ Output files created in all directories  
✅ Workflow completes with "SUCCESS" message  

---

*Last Updated: October 16, 2025 at 23:40 UTC+03:00*
