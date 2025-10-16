# Complete Prediction Workflow - All Markets

**Date:** October 16, 2025, 01:33 UTC+03:00  
**Purpose:** End-to-end workflow for match predictions across all markets

---

## 🎯 Workflow Overview

```
Crawler Data → Phase 1 → Phase 2 → Phase 3 → Phase 4 → Predictions
                  ↓         ↓         ↓         ↓          ↓
              Features   Models   Backtest   Deploy   All Markets
```

---

## 📊 Complete Workflow Architecture

### Input
```
Crawler Data (CSV/JSON)
├── Match fixtures
├── Team statistics
├── Historical results
├── Odds data (all markets)
└── Player information
```

### Processing Pipeline
```
Phase 1: Data Processing
├── Load crawler data
├── Feature engineering (241+ features)
├── Add market-specific features
│   ├── 1X2 features
│   ├── O/U features
│   ├── BTTS features
│   ├── First Half features
│   └── Other market features
└── Output: Processed features

Phase 2: Model Training
├── Train 1X2 model
├── Train O/U 2.5 model
├── Train BTTS model
├── Train First Half model
├── Train other market models
└── Output: Trained models

Phase 3: Backtesting
├── Validate all models
├── Test betting strategies
├── Calculate performance metrics
└── Output: Validation results

Phase 4: Prediction Generation
├── Load new match data
├── Generate predictions for all markets
├── Calculate confidence scores
├── Apply betting filters
└── Output: Match predictions
```

### Output
```
Predictions (JSON/CSV)
├── Match information
├── 1X2 predictions
├── Over/Under predictions
├── BTTS predictions
├── First Half predictions
├── Other market predictions
├── Confidence scores
├── Recommended bets
└── Expected values
```

---

## 🛠️ Implementation Structure

### File Structure
```
/home/kali/Non-major-leagues/
│
├── crawler_data/                    # Input from crawler
│   ├── upcoming_matches.csv
│   ├── team_stats.csv
│   ├── odds_all_markets.csv
│   └── historical_data.csv
│
├── pipeline_scripts/                # Processing scripts
│   ├── run_complete_pipeline.py    # Main orchestrator
│   ├── phase1_multi_market.py      # Enhanced Phase 1
│   ├── phase2_multi_market.py      # Enhanced Phase 2
│   ├── phase3_multi_market.py      # Enhanced Phase 3
│   └── phase4_predictions.py       # Prediction generator
│
├── models/                          # Trained models
│   ├── model_1x2.pkl
│   ├── model_ou_2.5.pkl
│   ├── model_btts.pkl
│   ├── model_first_half.pkl
│   └── model_*.pkl
│
└── predictions_output/              # Final predictions
    ├── match_predictions.json
    ├── match_predictions.csv
    ├── betting_recommendations.json
    └── confidence_report.html
```

---

## 📋 Detailed Workflow Steps

### Step 1: Crawler Data Input

**Input Files You Specify:**
```python
config = {
    'crawler_data': {
        'matches_file': 'crawler_data/upcoming_matches.csv',
        'stats_file': 'crawler_data/team_stats.csv',
        'odds_file': 'crawler_data/odds_all_markets.csv',
        'historical_file': 'crawler_data/historical_data.csv'
    },
    'output_files': {
        'predictions': 'predictions_output/match_predictions.json',
        'csv_export': 'predictions_output/match_predictions.csv',
        'recommendations': 'predictions_output/betting_recommendations.json',
        'report': 'predictions_output/confidence_report.html'
    }
}
```

**Expected Crawler Data Format:**
```csv
# upcoming_matches.csv
Date,HomeTeam,AwayTeam,League,Season
2025-10-20,Team A,Team B,E1,2025
2025-10-20,Team C,Team D,E1,2025

# odds_all_markets.csv
Date,HomeTeam,AwayTeam,B365H,B365D,B365A,BbOU,BbAH,BbMxAHH,BbAvAHA
2025-10-20,Team A,Team B,2.10,3.20,3.50,2.5,1.85,1.90,1.95

# With additional markets:
odds_over_2.5,odds_under_2.5,odds_btts_yes,odds_btts_no,odds_ht_home,odds_ht_draw,odds_ht_away
```

---

### Step 2: Phase 1 - Enhanced Feature Engineering

**Process:**
```python
# phase1_multi_market.py

class MultiMarketPreprocessor:
    def process_crawler_data(self, crawler_files):
        """Process crawler data for all markets"""
        
        # Load data
        matches = pd.read_csv(crawler_files['matches'])
        stats = pd.read_csv(crawler_files['stats'])
        odds = pd.read_csv(crawler_files['odds'])
        historical = pd.read_csv(crawler_files['historical'])
        
        # Merge data
        data = self.merge_all_data(matches, stats, odds, historical)
        
        # Generate features for each market
        data = self.add_1x2_features(data)
        data = self.add_ou_features(data)
        data = self.add_btts_features(data)
        data = self.add_first_half_features(data)
        data = self.add_handicap_features(data)
        
        return data
    
    def add_1x2_features(self, data):
        """Features for match result prediction"""
        # Home/Away form
        data['home_form_5'] = ...
        data['away_form_5'] = ...
        # Head-to-head
        data['h2h_home_wins'] = ...
        # League position
        data['home_position'] = ...
        return data
    
    def add_ou_features(self, data):
        """Features for Over/Under prediction"""
        # Goal averages
        data['home_goals_avg'] = ...
        data['away_goals_avg'] = ...
        data['total_goals_avg'] = ...
        # Recent goal trends
        data['home_goals_last_5'] = ...
        data['away_goals_last_5'] = ...
        return data
    
    def add_btts_features(self, data):
        """Features for Both Teams to Score"""
        # Scoring consistency
        data['home_scored_pct'] = ...
        data['away_scored_pct'] = ...
        # Clean sheets
        data['home_clean_sheets'] = ...
        data['away_clean_sheets'] = ...
        return data
    
    def add_first_half_features(self, data):
        """Features for First Half markets"""
        # First half stats
        data['home_ht_goals_avg'] = ...
        data['away_ht_goals_avg'] = ...
        data['home_ht_form'] = ...
        return data
```

**Output:**
```
pipeline_output/phase1_output/
├── processed_features.csv          # All features
├── feature_importance.json         # Feature rankings
└── data_quality_report.txt         # Validation report
```

---

### Step 3: Phase 2 - Multi-Market Model Training

**Process:**
```python
# phase2_multi_market.py

class MultiMarketModelTrainer:
    def train_all_models(self, data):
        """Train models for all markets"""
        
        models = {}
        
        # 1. Train 1X2 Model
        logger.info("Training 1X2 model...")
        X_1x2 = data[self.get_1x2_features()]
        y_1x2 = data['target']  # 0=Away, 1=Draw, 2=Home
        models['1x2'] = self.train_model(X_1x2, y_1x2, 'classification')
        
        # 2. Train O/U 2.5 Model
        logger.info("Training Over/Under 2.5 model...")
        X_ou = data[self.get_ou_features()]
        y_ou = (data['FTHG'] + data['FTAG'] > 2.5).astype(int)
        models['ou_2.5'] = self.train_model(X_ou, y_ou, 'binary')
        
        # 3. Train BTTS Model
        logger.info("Training BTTS model...")
        X_btts = data[self.get_btts_features()]
        y_btts = ((data['FTHG'] > 0) & (data['FTAG'] > 0)).astype(int)
        models['btts'] = self.train_model(X_btts, y_btts, 'binary')
        
        # 4. Train First Half Model
        logger.info("Training First Half model...")
        X_ht = data[self.get_ht_features()]
        y_ht = data['HTR'].map({'H': 2, 'D': 1, 'A': 0})
        models['first_half'] = self.train_model(X_ht, y_ht, 'classification')
        
        # 5. Train O/U 1.5 Model
        logger.info("Training Over/Under 1.5 model...")
        y_ou_1_5 = (data['FTHG'] + data['FTAG'] > 1.5).astype(int)
        models['ou_1.5'] = self.train_model(X_ou, y_ou_1_5, 'binary')
        
        # 6. Train O/U 3.5 Model
        logger.info("Training Over/Under 3.5 model...")
        y_ou_3_5 = (data['FTHG'] + data['FTAG'] > 3.5).astype(int)
        models['ou_3.5'] = self.train_model(X_ou, y_ou_3_5, 'binary')
        
        # Save all models
        for market, model in models.items():
            joblib.dump(model, f'models/model_{market}.pkl')
        
        return models
```

**Output:**
```
pipeline_output/phase2_output/
├── model_1x2.pkl
├── model_ou_2.5.pkl
├── model_ou_1.5.pkl
├── model_ou_3.5.pkl
├── model_btts.pkl
├── model_first_half.pkl
├── model_performance.json          # All model metrics
└── training_report.html            # Visual report
```

---

### Step 4: Phase 3 - Multi-Market Backtesting

**Process:**
```python
# phase3_multi_market.py

class MultiMarketBacktester:
    def backtest_all_markets(self, data, models):
        """Backtest all market predictions"""
        
        results = {}
        
        # Backtest each market
        for market, model in models.items():
            logger.info(f"Backtesting {market} market...")
            
            # Get predictions
            predictions = model.predict(data[features])
            confidence = model.predict_proba(data[features])
            
            # Calculate performance
            performance = self.calculate_performance(
                predictions, 
                data[f'{market}_actual'],
                data[f'odds_{market}']
            )
            
            results[market] = performance
        
        # Find best opportunities
        best_bets = self.find_best_opportunities(results)
        
        return results, best_bets
```

**Output:**
```
pipeline_output/phase3_output/
├── backtest_1x2.json
├── backtest_ou.json
├── backtest_btts.json
├── backtest_first_half.json
├── combined_performance.json       # All markets summary
└── best_opportunities.json         # Top betting opportunities
```

---

### Step 5: Phase 4 - Prediction Generation

**Process:**
```python
# phase4_predictions.py

class PredictionGenerator:
    def generate_predictions(self, upcoming_matches, models):
        """Generate predictions for all markets"""
        
        predictions = []
        
        for idx, match in upcoming_matches.iterrows():
            match_prediction = {
                'match_info': {
                    'date': match['Date'],
                    'home_team': match['HomeTeam'],
                    'away_team': match['AwayTeam'],
                    'league': match['League']
                },
                'predictions': {}
            }
            
            # Prepare features
            features = self.prepare_match_features(match)
            
            # 1X2 Prediction
            pred_1x2 = models['1x2'].predict([features])[0]
            conf_1x2 = models['1x2'].predict_proba([features])[0]
            match_prediction['predictions']['1x2'] = {
                'prediction': int(pred_1x2),
                'outcome': ['Away Win', 'Draw', 'Home Win'][pred_1x2],
                'confidence': float(conf_1x2[pred_1x2]),
                'probabilities': {
                    'home_win': float(conf_1x2[2]),
                    'draw': float(conf_1x2[1]),
                    'away_win': float(conf_1x2[0])
                },
                'odds': {
                    'home': match['B365H'],
                    'draw': match['B365D'],
                    'away': match['B365A']
                },
                'recommended': conf_1x2[pred_1x2] > 0.75
            }
            
            # Over/Under 2.5 Prediction
            pred_ou = models['ou_2.5'].predict([features])[0]
            conf_ou = models['ou_2.5'].predict_proba([features])[0]
            match_prediction['predictions']['over_under_2.5'] = {
                'prediction': 'Over' if pred_ou == 1 else 'Under',
                'confidence': float(conf_ou[pred_ou]),
                'probabilities': {
                    'over': float(conf_ou[1]),
                    'under': float(conf_ou[0])
                },
                'odds': {
                    'over': match['odds_over_2.5'],
                    'under': match['odds_under_2.5']
                },
                'recommended': conf_ou[pred_ou] > 0.70
            }
            
            # BTTS Prediction
            pred_btts = models['btts'].predict([features])[0]
            conf_btts = models['btts'].predict_proba([features])[0]
            match_prediction['predictions']['btts'] = {
                'prediction': 'Yes' if pred_btts == 1 else 'No',
                'confidence': float(conf_btts[pred_btts]),
                'probabilities': {
                    'yes': float(conf_btts[1]),
                    'no': float(conf_btts[0])
                },
                'odds': {
                    'yes': match['odds_btts_yes'],
                    'no': match['odds_btts_no']
                },
                'recommended': conf_btts[pred_btts] > 0.70
            }
            
            # First Half Prediction
            pred_ht = models['first_half'].predict([features])[0]
            conf_ht = models['first_half'].predict_proba([features])[0]
            match_prediction['predictions']['first_half'] = {
                'prediction': int(pred_ht),
                'outcome': ['Away Win', 'Draw', 'Home Win'][pred_ht],
                'confidence': float(conf_ht[pred_ht]),
                'probabilities': {
                    'home_win': float(conf_ht[2]),
                    'draw': float(conf_ht[1]),
                    'away_win': float(conf_ht[0])
                },
                'recommended': conf_ht[pred_ht] > 0.65
            }
            
            # Calculate best opportunity
            match_prediction['best_bet'] = self.find_best_bet(
                match_prediction['predictions']
            )
            
            predictions.append(match_prediction)
        
        return predictions
```

**Output Example:**
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
      "odds": {
        "home": 2.10,
        "draw": 3.20,
        "away": 3.50
      },
      "recommended": true,
      "expected_value": 0.827
    },
    "over_under_2.5": {
      "prediction": "Over",
      "confidence": 0.78,
      "probabilities": {
        "over": 0.78,
        "under": 0.22
      },
      "odds": {
        "over": 1.85,
        "under": 2.05
      },
      "recommended": true,
      "expected_value": 0.443
    },
    "btts": {
      "prediction": "Yes",
      "confidence": 0.72,
      "probabilities": {
        "yes": 0.72,
        "no": 0.28
      },
      "odds": {
        "yes": 1.90,
        "no": 1.95
      },
      "recommended": true,
      "expected_value": 0.368
    },
    "first_half": {
      "prediction": 2,
      "outcome": "Home Win",
      "confidence": 0.65,
      "probabilities": {
        "home_win": 0.65,
        "draw": 0.25,
        "away_win": 0.10
      },
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

## 🚀 Main Orchestrator Script

```python
# run_complete_pipeline.py

#!/usr/bin/env python3
"""
Complete Pipeline Orchestrator
Runs Phase 1-4 for all markets
"""

import argparse
import json
from pathlib import Path

class CompletePipelineOrchestrator:
    def __init__(self, config_file):
        """Initialize with configuration"""
        with open(config_file, 'r') as f:
            self.config = json.load(f)
    
    def run_complete_pipeline(self):
        """Execute complete workflow"""
        
        print("=" * 80)
        print("COMPLETE PREDICTION PIPELINE - ALL MARKETS")
        print("=" * 80)
        
        # Phase 1: Data Processing
        print("\n📊 PHASE 1: Data Processing & Feature Engineering")
        from phase1_multi_market import MultiMarketPreprocessor
        preprocessor = MultiMarketPreprocessor()
        processed_data = preprocessor.process_crawler_data(
            self.config['crawler_data']
        )
        print("✅ Phase 1 Complete")
        
        # Phase 2: Model Training
        print("\n🤖 PHASE 2: Multi-Market Model Training")
        from phase2_multi_market import MultiMarketModelTrainer
        trainer = MultiMarketModelTrainer()
        models = trainer.train_all_models(processed_data)
        print("✅ Phase 2 Complete")
        
        # Phase 3: Backtesting
        print("\n📈 PHASE 3: Multi-Market Backtesting")
        from phase3_multi_market import MultiMarketBacktester
        backtester = MultiMarketBacktester()
        backtest_results = backtester.backtest_all_markets(
            processed_data, models
        )
        print("✅ Phase 3 Complete")
        
        # Phase 4: Generate Predictions
        print("\n🎯 PHASE 4: Prediction Generation")
        from phase4_predictions import PredictionGenerator
        generator = PredictionGenerator()
        
        # Load upcoming matches from crawler
        upcoming_matches = pd.read_csv(
            self.config['crawler_data']['matches_file']
        )
        
        # Generate predictions
        predictions = generator.generate_predictions(
            upcoming_matches, models
        )
        
        # Save predictions
        self.save_predictions(predictions)
        print("✅ Phase 4 Complete")
        
        # Generate reports
        self.generate_reports(predictions, backtest_results)
        
        print("\n" + "=" * 80)
        print("✅ PIPELINE COMPLETE - PREDICTIONS GENERATED")
        print("=" * 80)
        
        return predictions
    
    def save_predictions(self, predictions):
        """Save predictions to specified output files"""
        output_dir = Path(self.config['output_files']['predictions']).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON
        with open(self.config['output_files']['predictions'], 'w') as f:
            json.dump(predictions, f, indent=2)
        
        # Save CSV
        df = self.predictions_to_dataframe(predictions)
        df.to_csv(self.config['output_files']['csv_export'], index=False)
        
        # Save recommendations
        recommendations = self.extract_recommendations(predictions)
        with open(self.config['output_files']['recommendations'], 'w') as f:
            json.dump(recommendations, f, indent=2)
        
        print(f"✅ Predictions saved to:")
        print(f"   - {self.config['output_files']['predictions']}")
        print(f"   - {self.config['output_files']['csv_export']}")
        print(f"   - {self.config['output_files']['recommendations']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='Configuration file')
    args = parser.parse_args()
    
    orchestrator = CompletePipelineOrchestrator(args.config)
    predictions = orchestrator.run_complete_pipeline()
```

---

## 📋 Configuration File

```json
{
  "crawler_data": {
    "matches_file": "crawler_data/upcoming_matches.csv",
    "stats_file": "crawler_data/team_stats.csv",
    "odds_file": "crawler_data/odds_all_markets.csv",
    "historical_file": "crawler_data/historical_data.csv"
  },
  "output_files": {
    "predictions": "predictions_output/match_predictions.json",
    "csv_export": "predictions_output/match_predictions.csv",
    "recommendations": "predictions_output/betting_recommendations.json",
    "report": "predictions_output/confidence_report.html"
  },
  "markets": {
    "1x2": {
      "enabled": true,
      "min_confidence": 0.75,
      "min_odds": 1.5,
      "max_odds": 5.0
    },
    "over_under_2.5": {
      "enabled": true,
      "min_confidence": 0.70,
      "min_odds": 1.7
    },
    "btts": {
      "enabled": true,
      "min_confidence": 0.70,
      "min_odds": 1.7
    },
    "first_half": {
      "enabled": true,
      "min_confidence": 0.65,
      "min_odds": 2.0
    }
  },
  "betting_strategy": {
    "max_stake_per_bet": 50,
    "max_bets_per_day": 5,
    "min_expected_value": 0.1
  }
}
```

---

## 🎯 Usage

### Command Line
```bash
# Run complete pipeline
python3 run_complete_pipeline.py --config config.json

# With custom crawler data
python3 run_complete_pipeline.py \
    --config config.json \
    --crawler-data crawler_data/matches.csv

# Generate predictions only (skip training)
python3 run_complete_pipeline.py \
    --config config.json \
    --predict-only
```

---

## 📊 Output Files

### 1. match_predictions.json
Complete predictions for all matches and markets

### 2. match_predictions.csv
Simplified CSV format for easy viewing

### 3. betting_recommendations.json
Top betting opportunities with stake recommendations

### 4. confidence_report.html
Visual report with charts and statistics

---

## ⏱️ Timeline

| Phase | Duration | Output |
|-------|----------|--------|
| **Phase 1** | 5-10 min | Processed features |
| **Phase 2** | 10-20 min | Trained models (all markets) |
| **Phase 3** | 5-10 min | Backtest results |
| **Phase 4** | 2-5 min | Match predictions |
| **Total** | **25-45 min** | **Complete predictions** |

---

## ✅ Next Steps

1. **Prepare crawler data** in specified format
2. **Create configuration file** with your paths
3. **Run complete pipeline**
4. **Review predictions** in output files
5. **Deploy to production**

---

*Workflow design completed on October 16, 2025 at 01:33 UTC+03:00*  
*Ready for implementation*
