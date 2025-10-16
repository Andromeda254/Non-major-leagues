# Complete Workflow Using Existing Infrastructure

**Date:** October 16, 2025, 01:58 UTC+03:00  
**Purpose:** Comprehensive workflow using existing phase integration files

---

## 🎯 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    CRAWLER INPUT                             │
│  enhanced_soccer_match_crawler.js                           │
│  → upcoming_matches.json                                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              API DATA COLLECTION                             │
│  api_data_manager.py                                        │
│  config.yaml + api_keys_template.env                        │
│  → team_stats, odds_all_markets, historical_data           │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 1: Data Processing                        │
│  phase1_integration.py                                      │
│  ├── non_major_league_data_collector.py                    │
│  ├── non_major_league_preprocessor.py                      │
│  ├── non_major_league_validator.py                         │
│  └── non_major_league_feature_engineer.py                  │
│  Output: pipeline_output/phase1_output/                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 2: Model Training                         │
│  phase2_integration.py                                      │
│  ├── non_major_league_model_architecture.py                │
│  ├── non_major_league_ensemble.py                          │
│  ├── non_major_league_transfer_learning.py                 │
│  ├── non_major_league_hyperparameter_tuning.py             │
│  └── non_major_league_model_validation.py                  │
│  Output: pipeline_output/phase2_output/                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 3: Backtesting                            │
│  phase3_integration.py                                      │
│  ├── non_major_league_backtesting.py                       │
│  ├── non_major_league_betting_strategy.py                  │
│  ├── non_major_league_performance_metrics.py               │
│  ├── non_major_league_risk_management.py                   │
│  └── non_major_league_live_testing.py                      │
│  Output: pipeline_output/phase3_output/                    │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              PHASE 4: Predictions & Deployment               │
│  phase4_integration.py                                      │
│  ├── non_major_league_deployment.py                        │
│  ├── non_major_league_monitoring.py                        │
│  ├── non_major_league_data_pipeline.py                     │
│  ├── non_major_league_model_serving.py                     │
│  └── non_major_league_performance_tracking.py              │
│  Output: predictions_output/                               │
└─────────────────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              MASTER ORCHESTRATOR                             │
│  master_pipeline.py                                         │
│  Coordinates all phases + generates predictions             │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Existing File Structure

```
/home/kali/Non-major-leagues/
│
├── 🔧 CONFIGURATION FILES
│   ├── config.yaml                      # Main API configuration
│   ├── api_keys_template.env            # API keys template
│   ├── generate_secrets.py              # Secret generation utility
│   └── pipeline_config.json             # Pipeline configuration
│
├── 🤖 CRAWLER & BRIDGE
│   ├── enhanced_soccer_match_crawler.js # Crawler script
│   ├── crawler_to_pipeline_bridge.py    # Bridge to pipeline
│   └── run_crawler_pipeline.sh          # Crawler runner
│
├── 📊 API & DATA MANAGEMENT
│   ├── api_data_manager.py              # API data manager
│   ├── non_major_league_data_collector.py
│   ├── non_major_league_data_pipeline.py
│   └── non_major_league_preprocessor.py
│
├── 🎯 PHASE INTEGRATION FILES
│   ├── phase1_integration.py            # Phase 1 orchestrator
│   ├── phase2_integration.py            # Phase 2 orchestrator
│   ├── phase3_integration.py            # Phase 3 orchestrator
│   └── phase4_integration.py            # Phase 4 orchestrator
│
├── 🔬 PHASE 1 COMPONENTS
│   ├── non_major_league_data_collector.py
│   ├── non_major_league_preprocessor.py
│   ├── non_major_league_validator.py
│   └── non_major_league_feature_engineer.py
│
├── 🤖 PHASE 2 COMPONENTS
│   ├── non_major_league_model_architecture.py
│   ├── non_major_league_ensemble.py
│   ├── non_major_league_transfer_learning.py
│   ├── non_major_league_hyperparameter_tuning.py
│   └── non_major_league_model_validation.py
│
├── 📈 PHASE 3 COMPONENTS
│   ├── non_major_league_backtesting.py
│   ├── non_major_league_betting_strategy.py
│   ├── non_major_league_performance_metrics.py
│   ├── non_major_league_risk_management.py
│   └── non_major_league_live_testing.py
│
├── 🚀 PHASE 4 COMPONENTS
│   ├── non_major_league_deployment.py
│   ├── non_major_league_monitoring.py
│   ├── non_major_league_data_pipeline.py
│   ├── non_major_league_model_serving.py
│   └── non_major_league_performance_tracking.py
│
├── 🎛️ MASTER ORCHESTRATOR
│   └── master_pipeline.py               # Main orchestrator
│
└── 📤 OUTPUT DIRECTORIES
    ├── pipeline_output/
    │   ├── phase1_output/
    │   ├── phase2_output/
    │   ├── phase3_output/
    │   └── phase4_output/
    ├── predictions_output/
    ├── logs/
    └── models/
```

---

## 🚀 Complete Workflow

### Step 1: Setup & Configuration

```bash
# 1. Configure API keys
cp api_keys_template.env .env
nano .env  # Add your API keys

# 2. Generate secrets (if needed)
python3 generate_secrets.py

# 3. Verify config.yaml
cat config.yaml
```

---

### Step 2: Run Crawler (Get Upcoming Matches)

```bash
# Option A: Run crawler directly
node enhanced_soccer_match_crawler.js

# Option B: Use the run script
./run_crawler_pipeline.sh

# Output: soccer-match-intelligence/upcoming_matches.json
```

---

### Step 3: Run Complete Pipeline

#### Option A: Run All Phases at Once

```bash
# Run complete pipeline for a league
python3 master_pipeline.py --phase all --league E1 --config config.yaml

# This will execute:
# - Phase 1: Data collection & feature engineering
# - Phase 2: Model training & validation
# - Phase 3: Backtesting & strategy optimization
# - Phase 4: Deployment & prediction generation
```

#### Option B: Run Phases Individually

```bash
# Phase 1: Data Processing
python3 master_pipeline.py --phase 1 --league E1 --config config.yaml

# Phase 2: Model Training
python3 master_pipeline.py --phase 2 --input-dir ./pipeline_output/phase1_output

# Phase 3: Backtesting
python3 master_pipeline.py --phase 3 --model-dir ./pipeline_output/phase2_output

# Phase 4: Predictions
python3 master_pipeline.py --phase 4 --strategy-dir ./pipeline_output/phase3_output
```

---

### Step 4: Generate Predictions for Upcoming Matches

```bash
# Use Phase 4 to generate predictions
python3 phase4_integration.py \
    --environment production \
    --config config.yaml \
    --upcoming-matches soccer-match-intelligence/upcoming_matches.json
```

---

## 📊 Detailed Workflow Steps

### PHASE 1: Data Collection & Feature Engineering

**Script:** `phase1_integration.py`

**Process:**
1. **Data Collection** (`non_major_league_data_collector.py`)
   - Loads crawler output (upcoming_matches.json)
   - Fetches team stats via `api_data_manager.py`
   - Fetches odds data (all markets)
   - Fetches historical data

2. **Preprocessing** (`non_major_league_preprocessor.py`)
   - Cleans and normalizes data
   - Handles missing values
   - Merges data sources

3. **Validation** (`non_major_league_validator.py`)
   - Data quality checks
   - Schema validation
   - Outlier detection

4. **Feature Engineering** (`non_major_league_feature_engineer.py`)
   - Generates 241+ features
   - Market-specific features (1X2, O/U, BTTS, HT)
   - Time-based features
   - Team performance metrics

**Output:**
```
pipeline_output/phase1_output/
├── E1_features.csv              # All features
├── E1_raw_data.csv              # Raw data
├── phase1_results.json          # Metadata
└── data_quality_report.txt      # Quality report
```

**Usage:**
```python
from phase1_integration import Phase1Integration

phase1 = Phase1Integration(config_file='config.yaml')
result = phase1.run_phase1_pipeline(
    league_code='E1',
    seasons=['2324', '2223'],
    collect_live_odds=True
)
```

---

### PHASE 2: Model Training & Validation

**Script:** `phase2_integration.py`

**Process:**
1. **Model Architecture** (`non_major_league_model_architecture.py`)
   - Designs model architecture
   - Configures hyperparameters
   - Prepares training pipeline

2. **Ensemble Modeling** (`non_major_league_ensemble.py`)
   - Trains multiple models (XGBoost, LightGBM, RF, LR)
   - Creates weighted ensemble
   - Optimizes ensemble weights

3. **Transfer Learning** (`non_major_league_transfer_learning.py`)
   - Transfers knowledge from major leagues
   - Fine-tunes for non-major leagues
   - Adapts to limited data

4. **Hyperparameter Tuning** (`non_major_league_hyperparameter_tuning.py`)
   - Optimizes model parameters
   - Uses Bayesian optimization
   - Cross-validation

5. **Model Validation** (`non_major_league_model_validation.py`)
   - Time-series cross-validation
   - Performance metrics
   - Model comparison

**Output:**
```
pipeline_output/phase2_output/
├── E1_ensemble.pkl              # Trained ensemble model
├── E1_xgboost.pkl               # Individual models
├── E1_lightgbm.pkl
├── E1_random_forest.pkl
├── model_performance.json       # Performance metrics
└── feature_importance.json      # Feature rankings
```

**Usage:**
```python
from phase2_integration import Phase2Integration

phase2 = Phase2Integration(config_file='config.yaml')
result = phase2.run_phase2_pipeline(
    data_file='pipeline_output/phase1_output/E1_features.csv',
    league_code='E1'
)
```

---

### PHASE 3: Backtesting & Strategy Optimization

**Script:** `phase3_integration.py`

**Process:**
1. **Backtesting** (`non_major_league_backtesting.py`)
   - Tests model on historical data
   - Simulates betting scenarios
   - Calculates performance metrics

2. **Betting Strategy** (`non_major_league_betting_strategy.py`)
   - Implements Kelly Criterion
   - Position sizing
   - Bet selection filters

3. **Performance Metrics** (`non_major_league_performance_metrics.py`)
   - ROI, win rate, profit factor
   - Sharpe ratio, drawdown
   - Risk-adjusted returns

4. **Risk Management** (`non_major_league_risk_management.py`)
   - Stop-loss mechanisms
   - Daily loss limits
   - Emergency stops

5. **Live Testing** (`non_major_league_live_testing.py`)
   - Paper trading simulation
   - Real-time performance tracking
   - Trade logging

**Output:**
```
pipeline_output/phase3_output/
├── backtest_results.json        # Backtest results
├── strategy_performance.json    # Strategy metrics
├── risk_analysis.json           # Risk metrics
└── optimized_strategy.pkl       # Optimized strategy
```

**Usage:**
```python
from phase3_integration import Phase3Integration

phase3 = Phase3Integration(config_file='config.yaml')
result = phase3.run_phase3_pipeline(
    data_file='pipeline_output/phase1_output/E1_features.csv',
    league_code='E1',
    model_file='pipeline_output/phase2_output/E1_ensemble.pkl'
)
```

---

### PHASE 4: Predictions & Deployment

**Script:** `phase4_integration.py`

**Process:**
1. **Deployment** (`non_major_league_deployment.py`)
   - Deploys models to production
   - Sets up API endpoints
   - Configures environments

2. **Monitoring** (`non_major_league_monitoring.py`)
   - Real-time performance monitoring
   - Alert system
   - Dashboard

3. **Data Pipeline** (`non_major_league_data_pipeline.py`)
   - Automated data collection
   - Real-time data processing
   - Feature generation

4. **Model Serving** (`non_major_league_model_serving.py`)
   - Prediction API
   - Batch predictions
   - Real-time predictions

5. **Performance Tracking** (`non_major_league_performance_tracking.py`)
   - Track live performance
   - Compare with backtest
   - Generate reports

**Output:**
```
predictions_output/
├── predictions_YYYYMMDD_HHMMSS.json  # All predictions
├── predictions_YYYYMMDD_HHMMSS.csv   # CSV format
├── recommendations.json               # Top bets
└── performance_report.html            # Visual report
```

**Usage:**
```python
from phase4_integration import Phase4Integration

phase4 = Phase4Integration(config_file='config.yaml')

# Generate predictions for upcoming matches
predictions = phase4.generate_predictions(
    upcoming_matches='soccer-match-intelligence/upcoming_matches.json',
    models_dir='pipeline_output/phase2_output',
    strategy='pipeline_output/phase3_output/optimized_strategy.pkl'
)
```

---

## 🎯 Master Pipeline Usage

### Complete Pipeline

```python
from master_pipeline import MasterPipeline

# Initialize
pipeline = MasterPipeline(config_path='config.yaml')

# Run all phases
pipeline.run_all_phases(league='E1')

# Or run specific phase
pipeline.run_phase1(league='E1')
pipeline.run_phase2()
pipeline.run_phase3()
pipeline.run_phase4()

# Get results
results = pipeline.get_pipeline_state()
print(results)
```

### Command Line

```bash
# Run all phases
python3 master_pipeline.py --phase all --league E1

# Run specific phase
python3 master_pipeline.py --phase 1 --league E1
python3 master_pipeline.py --phase 2
python3 master_pipeline.py --phase 3
python3 master_pipeline.py --phase 4

# With custom config
python3 master_pipeline.py --phase all --config my_config.yaml

# Multiple leagues
python3 master_pipeline.py --phase all --league E1,E2,E3
```

---

## 📊 API Data Manager Integration

### Using api_data_manager.py

```python
from api_data_manager import APIDataManager

# Initialize with config
api_manager = APIDataManager(
    config_path='config.yaml',
    env_file='api_keys_template.env'
)

# Fetch team statistics
team_stats = api_manager.get_team_statistics(
    team_name='Team A',
    league='E1'
)

# Fetch match odds (all markets)
odds = api_manager.get_match_odds(
    home_team='Team A',
    away_team='Team B',
    markets=['1x2', 'over_under', 'btts', 'first_half']
)

# Fetch historical data
historical = api_manager.get_historical_matches(
    league='E1',
    seasons=['2023', '2024']
)
```

### Configuration (config.yaml)

```yaml
data_sources:
  football_data:
    enabled: true
    api_key: ${FOOTBALL_DATA_API_KEY}
    base_url: https://api.football-data.org/v4
    rate_limit: 10
    priority: 1
    
  odds_api:
    enabled: true
    api_key: ${ODDS_API_KEY}
    base_url: https://api.the-odds-api.com/v4
    rate_limit: 500
    priority: 2
    
  api_football:
    enabled: true
    api_key: ${API_FOOTBALL_KEY}
    base_url: https://v3.football.api-sports.io
    rate_limit: 100
    priority: 3

markets:
  - 1x2
  - over_under_2.5
  - btts
  - first_half
  - asian_handicap
```

---

## 🔐 API Keys Setup

### 1. Copy Template

```bash
cp api_keys_template.env .env
```

### 2. Add Your Keys

```bash
# .env file
FOOTBALL_DATA_API_KEY=your_key_here
ODDS_API_KEY=your_key_here
API_FOOTBALL_KEY=your_key_here
API_FOOTBALL_HOST=v3.football.api-sports.io
```

### 3. Generate Secrets (Optional)

```bash
python3 generate_secrets.py
```

---

## 📈 Output Files & Predictions

### Prediction Format (JSON)

```json
{
  "match_info": {
    "date": "2025-10-20",
    "home_team": "Team A",
    "away_team": "Team B",
    "league": "E1"
  },
  "predictions": {
    "1x2": {
      "prediction": 2,
      "outcome": "Home Win",
      "confidence": 0.87,
      "probabilities": {
        "home_win": 0.87,
        "draw": 0.08,
        "away_win": 0.05
      },
      "odds": {"home": 2.10, "draw": 3.20, "away": 3.50},
      "recommended": true,
      "expected_value": 0.827
    },
    "over_under_2.5": {
      "prediction": "Over",
      "confidence": 0.78,
      "probabilities": {"over": 0.78, "under": 0.22},
      "odds": {"over": 1.85, "under": 2.05},
      "recommended": true
    },
    "btts": {
      "prediction": "Yes",
      "confidence": 0.72,
      "probabilities": {"yes": 0.72, "no": 0.28},
      "odds": {"yes": 1.90, "no": 1.95},
      "recommended": true
    },
    "first_half": {
      "prediction": 2,
      "outcome": "Home Win",
      "confidence": 0.65,
      "recommended": false
    }
  },
  "best_bet": {
    "market": "1x2",
    "selection": "Home Win",
    "confidence": 0.87,
    "odds": 2.10,
    "expected_value": 0.827,
    "stake_recommendation": "$50"
  }
}
```

---

## ⏱️ Execution Timeline

| Phase | Duration | Description |
|-------|----------|-------------|
| **Crawler** | 2-5 min | Fetch upcoming matches |
| **Phase 1** | 10-20 min | Data collection & feature engineering |
| **Phase 2** | 20-40 min | Model training (all markets) |
| **Phase 3** | 10-20 min | Backtesting & optimization |
| **Phase 4** | 5-10 min | Prediction generation |
| **Total** | **47-95 min** | **Complete workflow** |

---

## ✅ Validation Checklist

### Before Running

- [ ] Crawler output exists (`upcoming_matches.json`)
- [ ] API keys configured (`.env` file)
- [ ] `config.yaml` properly configured
- [ ] All Python dependencies installed
- [ ] Output directories created

### After Phase 1

- [ ] Features generated (`E1_features.csv`)
- [ ] Data quality report created
- [ ] No validation errors

### After Phase 2

- [ ] Models trained (`.pkl` files)
- [ ] Model performance acceptable
- [ ] Feature importance calculated

### After Phase 3

- [ ] Backtest results positive (ROI > 10%)
- [ ] Win rate acceptable (> 50%)
- [ ] Risk metrics within limits

### After Phase 4

- [ ] Predictions generated
- [ ] All markets covered
- [ ] Recommendations provided

---

## 🎯 Quick Start Commands

```bash
# 1. Setup
cp api_keys_template.env .env
nano .env  # Add API keys

# 2. Run crawler
node enhanced_soccer_match_crawler.js

# 3. Run complete pipeline
python3 master_pipeline.py --phase all --league E1 --config config.yaml

# 4. View predictions
cat predictions_output/predictions_*.json | jq '.'

# 5. View CSV
cat predictions_output/predictions_*.csv
```

---

## 🔧 Troubleshooting

### Issue: API keys not working

```bash
# Check .env file
cat .env

# Verify config.yaml
cat config.yaml

# Test API connection
python3 -c "from api_data_manager import APIDataManager; api = APIDataManager(); print('OK')"
```

### Issue: Phase 1 fails

```bash
# Check logs
tail -f logs/master_pipeline_*.log

# Run Phase 1 separately
python3 phase1_integration.py --league E1
```

### Issue: No predictions generated

```bash
# Verify all phases completed
ls -la pipeline_output/phase*/

# Check Phase 4 logs
tail -f logs/phase4_integration.log
```

---

## ✅ Summary

**Complete Workflow:**
1. ✅ Crawler → `upcoming_matches.json`
2. ✅ API Manager → team stats, odds, historical data
3. ✅ Phase 1 → Feature engineering (241+ features)
4. ✅ Phase 2 → Model training (all markets)
5. ✅ Phase 3 → Backtesting & optimization
6. ✅ Phase 4 → Predictions (all markets)

**Key Files:**
- ✅ `master_pipeline.py` - Main orchestrator
- ✅ `phase1_integration.py` - Phase 1
- ✅ `phase2_integration.py` - Phase 2
- ✅ `phase3_integration.py` - Phase 3
- ✅ `phase4_integration.py` - Phase 4
- ✅ `api_data_manager.py` - API integration
- ✅ `config.yaml` - Configuration
- ✅ `.env` - API keys

**Ready for production!** 🚀

---

*Workflow guide created on October 16, 2025 at 01:58 UTC+03:00*  
*Status: ✅ COMPLETE - Using Existing Infrastructure*
