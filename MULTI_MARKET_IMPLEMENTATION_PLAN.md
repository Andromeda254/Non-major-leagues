# Multi-Market Prediction Implementation Plan

**Current Status:** Only 1X2 market has trained models  
**Goal:** Implement predictions for all markets (1X2, O/U, BTTS, First Half)

---

## 📊 Current State

### **Working Markets**
- ✅ **1X2 (Match Result)** - Fully trained ensemble model

### **Placeholder Markets** (Need Implementation)
- ⚠️ **Over/Under 2.5 Goals** - Hardcoded placeholder
- ⚠️ **Both Teams to Score (BTTS)** - Hardcoded placeholder
- ⚠️ **First Half Result** - Hardcoded placeholder

---

## 🎯 Implementation Strategy

### **Option 1: Use Existing 1X2 Model Features (Quick)**
Use the same features and model to derive predictions for other markets based on probabilities and heuristics.

**Pros:**
- Fast implementation
- No additional training needed
- Uses existing infrastructure

**Cons:**
- Less accurate than dedicated models
- Heuristic-based, not ML-based
- May not capture market-specific patterns

### **Option 2: Train Dedicated Models (Recommended)**
Train separate models for each market with market-specific targets.

**Pros:**
- Most accurate predictions
- Captures market-specific patterns
- Proper ML-based approach

**Cons:**
- Requires additional training time (~30 min per market)
- Needs labeled data for each market
- More complex implementation

---

## 🚀 Quick Implementation (Option 1)

### **Approach: Derive from 1X2 Probabilities**

#### **1. Over/Under 2.5 Goals**
```python
def predict_over_under(home_win_prob, draw_prob, away_win_prob, features):
    """
    Predict O/U 2.5 based on match outcome probabilities
    High-scoring outcomes: Home Win (often 2-1, 3-1), Away Win (1-2, 1-3)
    Low-scoring outcomes: Draw (0-0, 1-1)
    """
    # Use goal-related features
    avg_goals = features.get('avg_goals_per_match', 2.5)
    home_attack = features.get('home_team_avg_goals', 1.5)
    away_attack = features.get('away_team_avg_goals', 1.0)
    
    # Calculate expected goals
    expected_goals = (home_attack + away_attack) * 1.1  # Slight boost
    
    # Adjust based on outcome probabilities
    # Draws tend to be lower scoring
    expected_goals -= draw_prob * 0.5
    
    # High probability outcomes suggest confidence in scoring
    if max(home_win_prob, away_win_prob) > 0.7:
        expected_goals += 0.3
    
    over_prob = 1 / (1 + np.exp(-(expected_goals - 2.5)))
    
    return {
        'prediction': 'Over' if over_prob > 0.5 else 'Under',
        'confidence': float(max(over_prob, 1 - over_prob)),
        'probabilities': {
            'over': float(over_prob),
            'under': float(1 - over_prob)
        },
        'expected_goals': float(expected_goals),
        'recommended': max(over_prob, 1 - over_prob) > 0.65
    }
```

#### **2. Both Teams to Score (BTTS)**
```python
def predict_btts(home_win_prob, draw_prob, away_win_prob, features):
    """
    Predict BTTS based on team attacking/defensive strength
    """
    home_attack = features.get('home_team_avg_goals', 1.5)
    away_attack = features.get('away_team_avg_goals', 1.0)
    home_defense = features.get('home_conceded_avg', 1.0)
    away_defense = features.get('away_conceded_avg', 1.2)
    
    # Both teams scoring is more likely when:
    # 1. Both teams have good attack
    # 2. Both teams have weak defense
    # 3. Match is competitive (not too one-sided)
    
    attack_strength = (home_attack + away_attack) / 2
    defense_weakness = (home_defense + away_defense) / 2
    competitiveness = 1 - abs(home_win_prob - away_win_prob)
    
    btts_score = (attack_strength * 0.4 + 
                  defense_weakness * 0.3 + 
                  competitiveness * 0.3)
    
    # Normalize to probability
    btts_prob = min(0.95, max(0.05, btts_score / 3))
    
    return {
        'prediction': 'Yes' if btts_prob > 0.5 else 'No',
        'confidence': float(max(btts_prob, 1 - btts_prob)),
        'probabilities': {
            'yes': float(btts_prob),
            'no': float(1 - btts_prob)
        },
        'recommended': max(btts_prob, 1 - btts_prob) > 0.60
    }
```

#### **3. First Half Result**
```python
def predict_first_half(home_win_prob, draw_prob, away_win_prob, features):
    """
    Predict first half result
    First half tends to be more conservative (more draws)
    """
    # First half is typically more conservative
    # Increase draw probability, decrease extreme outcomes
    fh_draw_prob = draw_prob * 1.5
    fh_home_prob = home_win_prob * 0.8
    fh_away_prob = away_win_prob * 0.8
    
    # Normalize
    total = fh_home_prob + fh_draw_prob + fh_away_prob
    fh_home_prob /= total
    fh_draw_prob /= total
    fh_away_prob /= total
    
    # Determine prediction
    probs = [fh_away_prob, fh_draw_prob, fh_home_prob]
    prediction_idx = np.argmax(probs)
    
    return {
        'prediction': prediction_idx,
        'outcome': ['Away Win', 'Draw', 'Home Win'][prediction_idx],
        'confidence': float(probs[prediction_idx]),
        'probabilities': {
            'home_win': float(fh_home_prob),
            'draw': float(fh_draw_prob),
            'away_win': float(fh_away_prob)
        },
        'recommended': probs[prediction_idx] > 0.55
    }
```

---

## 🔧 Implementation Steps

### **Step 1: Update `generate_match_prediction` Method**

Replace the placeholder code with actual prediction logic:

```python
def generate_match_prediction(self, match, model, feature_cols, historical_data):
    """Generate prediction for a single match"""
    prediction = {
        'match_info': {
            'date': match.get('Date', match.get('date', 'N/A')),
            'home_team': match.get('HomeTeam', match.get('homeTeam', 'N/A')),
            'away_team': match.get('AwayTeam', match.get('awayTeam', 'N/A')),
            'league': match.get('League', match.get('league', 'E1'))
        },
        'predictions': {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Prepare features
    features = historical_data[feature_cols].iloc[-1:].fillna(0).replace([np.inf, -np.inf], 0)
    features_dict = features.iloc[0].to_dict()
    
    # Generate 1X2 prediction
    try:
        pred = model.predict(features)[0]
        proba = model.predict_proba(features)[0]
        
        home_win_prob = float(proba[2])
        draw_prob = float(proba[1])
        away_win_prob = float(proba[0])
        
        prediction['predictions']['1x2'] = {
            'prediction': int(pred),
            'outcome': ['Away Win', 'Draw', 'Home Win'][int(pred)],
            'confidence': float(proba[int(pred)]),
            'probabilities': {
                'home_win': home_win_prob,
                'draw': draw_prob,
                'away_win': away_win_prob
            },
            'recommended': float(proba[int(pred)]) > 0.75
        }
        
        # Generate multi-market predictions
        prediction['predictions']['over_under_2.5'] = self.predict_over_under(
            home_win_prob, draw_prob, away_win_prob, features_dict
        )
        
        prediction['predictions']['btts'] = self.predict_btts(
            home_win_prob, draw_prob, away_win_prob, features_dict
        )
        
        prediction['predictions']['first_half'] = self.predict_first_half(
            home_win_prob, draw_prob, away_win_prob, features_dict
        )
        
    except Exception as e:
        logger.warning(f"⚠️ Could not generate predictions: {e}")
    
    return prediction
```

### **Step 2: Add Helper Methods**

Add the three prediction methods to the `WorkflowOrchestrator` class.

### **Step 3: Test the Implementation**

```bash
python3 -u run_complete_workflow.py \
    --crawler-file soccer-match-intelligence/filtered_3am_5am_20251016.json \
    --league E1 \
    --skip-training
```

---

## 📊 Expected Output Format

### **Enhanced Prediction Structure**

```json
{
  "match_info": {
    "date": "2025-10-16",
    "home_team": "Atletico Mineiro MG",
    "away_team": "Cruzeiro EC",
    "league": "Brazil"
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
      "confidence": 0.72,
      "probabilities": {
        "over": 0.72,
        "under": 0.28
      },
      "expected_goals": 2.8,
      "recommended": true
    },
    "btts": {
      "prediction": "Yes",
      "confidence": 0.65,
      "probabilities": {
        "yes": 0.65,
        "no": 0.35
      },
      "recommended": true
    },
    "first_half": {
      "prediction": 1,
      "outcome": "Draw",
      "confidence": 0.58,
      "probabilities": {
        "home_win": 0.35,
        "draw": 0.58,
        "away_win": 0.07
      },
      "recommended": true
    }
  }
}
```

---

## 🎯 Advanced Implementation (Option 2)

### **Training Dedicated Models**

To train separate models for each market, modify Phase 2:

#### **1. Create Market-Specific Targets**

```python
# In phase2_integration.py
def create_market_targets(data):
    """Create targets for all markets"""
    targets = {}
    
    # 1X2 target (existing)
    targets['1x2'] = data['FTR'].map({'H': 2, 'D': 1, 'A': 0})
    
    # Over/Under 2.5 target
    targets['over_under'] = (data['FTHG'] + data['FTAG'] > 2.5).astype(int)
    
    # BTTS target
    targets['btts'] = ((data['FTHG'] > 0) & (data['FTAG'] > 0)).astype(int)
    
    # First Half target
    targets['first_half'] = data['HTR'].map({'H': 2, 'D': 1, 'A': 0})
    
    return targets
```

#### **2. Train Models for Each Market**

```python
# Train ensemble for each market
markets = ['1x2', 'over_under', 'btts', 'first_half']
models = {}

for market in markets:
    logger.info(f"Training model for {market} market")
    y_market = targets[market]
    
    # Train ensemble
    model = train_ensemble_model(X_train, y_market, X_val, y_market_val)
    models[market] = model
    
    # Save model
    joblib.dump(model, f'pipeline_output/phase2_output/E1_{market}_ensemble.pkl')
```

#### **3. Load Market-Specific Models in Phase 4**

```python
# Load all market models
models = {}
markets = ['1x2', 'over_under', 'btts', 'first_half']

for market in markets:
    model_file = f'pipeline_output/phase2_output/E1_{market}_ensemble.pkl'
    if os.path.exists(model_file):
        models[market] = load_ensemble_model(model_file)
    else:
        logger.warning(f"Model for {market} not found, using heuristics")
```

---

## 📝 Recommendation

**Start with Option 1 (Quick Implementation)** because:
1. ✅ Can be implemented immediately
2. ✅ Uses existing infrastructure
3. ✅ Provides reasonable predictions
4. ✅ No additional training time

**Then migrate to Option 2** when:
1. You have labeled data for all markets
2. You need maximum accuracy
3. You have time for additional training

---

## 🚀 Next Steps

1. **Implement Option 1** - Add heuristic-based multi-market predictions
2. **Test thoroughly** - Verify predictions make sense
3. **Collect feedback** - See which markets need improvement
4. **Plan Option 2** - If accuracy is insufficient, train dedicated models

---

*Created: October 16, 2025 at 23:48 UTC+03:00*
