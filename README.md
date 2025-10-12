# Non-Major League ML Pipeline

A comprehensive machine learning pipeline for predicting soccer match outcomes in non-major leagues, with integrated betting strategy and risk management.

## 🚀 Quick Start

### 1. Setup Environment
```bash
python run_pipeline.py --setup
```

### 2. Configure API Keys
Edit `config.yaml` and add your API keys:
```yaml
data_sources:
  football_data:
    api_key: "your_football_data_api_key_here"
  odds_api:
    api_key: "your_odds_api_key_here"
  api_football:
    api_key: "your_api_football_key_here"
```

### 3. Run Pipeline
```bash
# Quick test run
python run_pipeline.py --quick

# Full pipeline run
python run_pipeline.py --full

# Run specific phase
python run_pipeline.py --phase 1 --league E1
```

## 📋 Pipeline Overview

The system consists of 4 integrated phases:

### Phase 1: Data Collection & Preprocessing
- **Data Collection**: Multi-source data aggregation (football-data.co.uk, The Odds API, API-Football)
- **Preprocessing**: Advanced data cleaning, missing value handling, outlier detection
- **Validation**: Comprehensive data quality assessment
- **Feature Engineering**: Advanced feature creation (form, consistency, market, temporal features)

### Phase 2: Model Development & Validation
- **Model Architecture**: XGBoost, LightGBM, Random Forest, Logistic Regression
- **Ensemble Modeling**: Conservative weighted ensemble with dynamic weighting
- **Transfer Learning**: Knowledge transfer from major leagues
- **Hyperparameter Tuning**: Optuna-based optimization with time-series validation
- **Model Validation**: Time-series split, walk-forward, bootstrap validation

### Phase 3: Backtesting & Strategy Validation
- **Backtesting**: Walk-forward backtesting with Kelly Criterion
- **Betting Strategy**: Conservative betting with multiple filters
- **Performance Metrics**: Comprehensive financial and betting metrics
- **Risk Management**: Multi-level risk controls and drawdown protection
- **Live Testing**: Paper trading simulation

### Phase 4: Production Deployment & Monitoring
- **Deployment**: Multi-environment deployment (dev/staging/prod)
- **Monitoring**: Real-time system and business monitoring
- **Data Pipeline**: Automated data ingestion and processing
- **Model Serving**: FastAPI-based prediction API
- **Performance Tracking**: Real-time performance analytics

## 🛠️ Installation

### Requirements
```bash
pip install pandas numpy scikit-learn xgboost lightgbm optuna joblib requests schedule pyyaml fastapi uvicorn matplotlib seaborn scipy
```

### Optional Dependencies
```bash
pip install psutil prometheus-client jwt passlib python-multipart
```

## 📁 Project Structure

```
TestGames/
├── master_pipeline.py          # Main pipeline orchestrator
├── run_pipeline.py             # Simple runner script
├── config.yaml                 # Configuration file
├── README.md                   # This file
├── phase1_integration.py       # Phase 1 integration
├── phase2_integration.py       # Phase 2 integration
├── phase3_integration.py       # Phase 3 integration
├── phase4_integration.py       # Phase 4 integration
├── non_major_league_*.py       # Individual components
├── pipeline_output/            # Pipeline results
├── logs/                       # Log files
├── models/                     # Trained models
├── data/                       # Data files
├── reports/                    # Generated reports
└── deployments/                # Deployment files
```

## 🔧 Configuration

### Basic Configuration
Edit `config.yaml` to customize:

```yaml
# Pipeline settings
pipeline:
  output_dir: "./pipeline_output"
  log_level: "INFO"

# Phase settings
phase1:
  enabled: true
  leagues: ["E1", "E2", "E3"]
  seasons: 3

phase2:
  enabled: true
  models: ["xgboost", "lightgbm", "random_forest"]

phase3:
  enabled: true
  initial_capital: 10000
  kelly_fraction: 0.02

phase4:
  enabled: true
  deployment_environment: "development"
```

### API Keys
Add your API keys to `config.yaml`:

```yaml
data_sources:
  football_data:
    api_key: "your_api_key_here"
  odds_api:
    api_key: "your_api_key_here"
  api_football:
    api_key: "your_api_key_here"
```

## 🚀 Usage Examples

### Run Complete Pipeline
```bash
python run_pipeline.py --full --league E1
```

### Run Single Phase
```bash
python run_pipeline.py --phase 1 --league E1
python run_pipeline.py --phase 2
python run_pipeline.py --phase 3
python run_pipeline.py --phase 4
```

### Deploy to Production
```bash
python run_pipeline.py --full --deploy prod
```

### Quick Test
```bash
python run_pipeline.py --quick
```

## 📊 Output Files

The pipeline generates comprehensive outputs:

### Phase 1 Outputs
- `processed_features.parquet` - Clean, feature-rich dataset
- `validation_report.txt` - Data quality report
- `phase1_results.json` - Phase 1 results

### Phase 2 Outputs
- `ensemble_model.pkl` - Trained ensemble model
- `model_validation_report.txt` - Model validation report
- `phase2_results.json` - Phase 2 results

### Phase 3 Outputs
- `betting_strategy.pkl` - Validated betting strategy
- `backtesting_results.json` - Backtesting results
- `performance_metrics.json` - Performance metrics
- `phase3_results.json` - Phase 3 results

### Phase 4 Outputs
- `deployment_package/` - Deployment files
- `monitoring_dashboard/` - Monitoring dashboard
- `api_server/` - Model serving API
- `phase4_results.json` - Phase 4 results

### Final Report
- `final_report.json` - Comprehensive pipeline report
- `final_report.txt` - Human-readable report

## 🔍 Monitoring

### Real-time Monitoring
The system includes comprehensive monitoring:

- **System Metrics**: CPU, memory, disk usage
- **Application Metrics**: Response time, throughput, error rates
- **Business Metrics**: Prediction accuracy, betting performance, ROI
- **Alerts**: Multi-channel alerting (email, Slack, webhooks)

### Dashboard
Access the monitoring dashboard at:
- Development: http://localhost:8080
- Staging: http://staging.example.com:8080
- Production: https://api.example.com:8080

## 🛡️ Risk Management

The system includes multi-layered risk management:

- **Capital Protection**: Position size limits, drawdown controls
- **Volatility Management**: Dynamic position sizing
- **Correlation Monitoring**: Diversification requirements
- **Stress Testing**: Scenario-based risk assessment
- **Emergency Protocols**: Automatic stop-loss mechanisms

## 🔧 Advanced Usage

### Custom Configuration
```python
from master_pipeline import MasterPipeline

# Create custom config
config = {
    'phase1': {'enabled': True, 'leagues': ['E1']},
    'phase2': {'enabled': True, 'models': ['xgboost']},
    # ... more config
}

# Run pipeline
pipeline = MasterPipeline()
pipeline.config = config
result = pipeline.run_all_phases(league='E1')
```

### Individual Components
```python
from non_major_league_data_collector import NonMajorLeagueDataCollector
from non_major_league_model_architecture import NonMajorLeagueModelArchitecture

# Use individual components
collector = NonMajorLeagueDataCollector()
data = collector.collect_historical_data('E1', seasons=2)

model_arch = NonMajorLeagueModelArchitecture()
model = model_arch.train_ensemble_model(data)
```

## 📈 Performance Metrics

The system tracks comprehensive performance metrics:

### Prediction Metrics
- Accuracy, Precision, Recall, F1-Score
- Brier Score, Log Loss
- Calibration, Reliability, Sharpness

### Betting Metrics
- ROI, Sharpe Ratio, Sortino Ratio
- Maximum Drawdown, VaR, CVaR
- Win Rate, Profit Factor, Kelly Efficiency

### Risk Metrics
- Volatility, Correlation
- Position Sizing, Diversification
- Stress Test Results

## 🚨 Troubleshooting

### Common Issues

1. **API Key Errors**
   - Check API keys in `config.yaml`
   - Verify API key permissions and limits

2. **Memory Issues**
   - Reduce number of seasons or leagues
   - Enable parallel processing in config

3. **Model Training Failures**
   - Check data quality in Phase 1 results
   - Reduce model complexity in config

4. **Deployment Issues**
   - Check Docker and Kubernetes setup
   - Verify environment configuration

### Logs
Check logs in `./logs/` directory for detailed error information.

## 📞 Support

For issues and questions:
1. Check the logs in `./logs/`
2. Review the configuration in `config.yaml`
3. Check the troubleshooting section above
4. Review the generated reports in `./pipeline_output/`

## 📄 License

This project is for educational and research purposes. Please ensure compliance with API terms of service and local regulations regarding sports betting.

## 🔄 Updates

The pipeline is designed to be modular and extensible. You can:
- Add new data sources
- Implement new models
- Customize risk management rules
- Add new performance metrics
- Extend monitoring capabilities

---

**Happy Predicting! 🎯⚽**







