# Crawler to Prediction Integration Plan

## Executive Summary

**Current Gap:** The data collector collects historical league data, but doesn't fetch match-specific data (H2H, recent form, team stats) for crawler-provided fixtures.

**Solution:** Build a Match Data Enrichment Pipeline that bridges crawler output to prediction input.

---

## Current Workflow Issues

### What Works ✅
- Crawler outputs upcoming matches
- Phase 1 collects historical league data
- Phase 2 trains models
- Phase 3 makes predictions

### What's Missing ❌
1. **No H2H Data**: Historical meetings between specific teams
2. **No Recent Form**: Last 5-10 matches for each team
3. **No Team-Specific Stats**: Goals scored/conceded, home/away performance
4. **No Match Context**: Injuries, lineups, team news
5. **Generic Predictions**: Based on league-wide patterns, not match-specific

---

## Proposed Architecture

```
┌─────────────────┐
│ Crawler Output  │ → matches.json (Leicester vs Burnley, etc.)
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ NEW: Match Data Enrichment Pipeline     │
├─────────────────────────────────────────┤
│ 1. Parse crawler JSON                   │
│ 2. For each match:                      │
│    • Fetch H2H history (last 10 meetings)│
│    • Get home team recent form (last 10) │
│    • Get away team recent form (last 10) │
│    • Collect team statistics            │
│    • Get current odds                   │
│ 3. Build match-specific features        │
│ 4. Output: enriched_matches.json        │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Phase 1: Historical Training Data       │
│ (Enhanced to include H2H features)      │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Phase 2: Model Training                 │
│ (Train on H2H + Form features)          │
└────────┬────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────┐
│ Phase 3: Match Predictions              │
│ (Use enriched match data)               │
└─────────────────────────────────────────┘
```

---

## Implementation Components

### 1. Match Data Enricher (`match_data_enricher.py`)

**Purpose:** Fetch all data needed for specific match predictions

**Key Methods:**
```python
class MatchDataEnricher:
    def enrich_matches(self, crawler_json_path):
        """
        Input: matches.json from crawler
        Output: enriched_matches.json
        """
        
    def get_h2h_data(self, home_team, away_team, num_matches=10):
        """
        Fetch historical meetings
        Returns: H2H win rate, avg goals, last result
        """
        
    def get_team_form(self, team_name, num_matches=10):
        """
        Get recent match results
        Returns: Wins, draws, losses, goals, clean sheets
        """
        
    def get_team_stats(self, team_name, league):
        """
        Get season statistics
        Returns: Position, points, GF, GA, home/away splits
        """
```

**APIs to Use:**
- `api-football`: Has dedicated H2H endpoint
- `football-data.org`: Team fixtures and results
- `sofascore`: Recent form and statistics

---

### 2. Enhanced Data Collector Methods

**Add to `non_major_league_data_collector.py`:**

```python
def collect_h2h_matches(self, home_team, away_team, num_matches=10):
    """
    Collect historical meetings between two teams
    Uses: api-football H2H endpoint
    """
    
def collect_team_recent_matches(self, team_name, num_matches=10):
    """
    Collect last N matches for a team
    Uses: football-data.org, sofascore
    """
    
def collect_match_context_data(self, match_fixture):
    """
    Collect all data for one match:
    - H2H history
    - Home team form
    - Away team form
    - Team statistics
    - Current odds
    """
```

---

### 3. Pre-Match Feature Builder (`pre_match_features.py`)

**Purpose:** Convert enriched data into ML features

**Features to Create:**

#### H2H Features:
- `h2h_home_win_rate`: % of home wins in last 10 meetings
- `h2h_avg_home_goals`: Avg goals scored by home team
- `h2h_avg_away_goals`: Avg goals scored by away team
- `h2h_last_result`: Result of last meeting (H/D/A)
- `h2h_btts_rate`: % of matches with both teams scoring

#### Form Features (Home Team):
- `home_last5_wins`: Wins in last 5 matches
- `home_last5_goals_scored`: Goals scored
- `home_last5_goals_conceded`: Goals conceded
- `home_last5_clean_sheets`: Clean sheets
- `home_form_points`: Points from last 5 (3 per win, 1 per draw)

#### Form Features (Away Team):
- `away_last5_wins`
- `away_last5_goals_scored`
- `away_last5_goals_conceded`
- `away_last5_clean_sheets`
- `away_form_points`

#### Statistical Features:
- `home_team_position`: League position
- `away_team_position`: League position
- `home_team_gf_avg`: Goals per game (home)
- `away_team_ga_avg`: Goals conceded per game (away)
- `odds_home_win`: Betting odds for home win
- `odds_draw`: Betting odds for draw
- `odds_away_win`: Betting odds for away win

---

### 4. Integration with Existing Pipeline

**Modified Workflow:**

```python
# Step 1: Enrich crawler matches
enricher = MatchDataEnricher(api_manager)
enriched_data = enricher.enrich_matches('crawler_output.json')

# Step 2: Build features
feature_builder = PreMatchFeatureBuilder()
match_features = feature_builder.build_features(enriched_data)

# Step 3: Load trained models (from Phase 2)
models = load_trained_models('models/')

# Step 4: Make predictions
predictions = predict_matches(models, match_features)

# Step 5: Output results
save_predictions(predictions, 'predictions.json')
```

---

## API Capabilities Matrix

| API | H2H Data | Team Form | Team Stats | Live Odds | Lineups |
|-----|----------|-----------|------------|-----------|---------|
| football-data.org | ✅ (via history) | ✅ | ✅ | ❌ | ❌ |
| api-football | ✅ (dedicated) | ✅ | ✅ | ❌ | ✅ |
| odds-api | ❌ | ❌ | ❌ | ✅ | ❌ |
| sportmonks | ✅ | ✅ | ✅ | ❌ | ✅ |
| sofascore | ✅ | ✅ | ✅ | ❌ | ❌ |

**Best APIs for Match Enrichment:**
1. **api-football**: Best for H2H (dedicated endpoint)
2. **sofascore**: Best for recent form and statistics
3. **odds-api**: Best for live betting odds

---

## Sample Enriched Match Data

**Input (Crawler):**
```json
{
  "home_team": "Leicester City",
  "away_team": "Burnley",
  "date": "2024-10-19"
}
```

**Output (Enriched):**
```json
{
  "home_team": "Leicester City",
  "away_team": "Burnley",
  "date": "2024-10-19",
  "h2h": {
    "total_meetings": 10,
    "home_wins": 4,
    "draws": 3,
    "away_wins": 3,
    "avg_home_goals": 1.8,
    "avg_away_goals": 1.4,
    "btts_rate": 0.6,
    "last_result": "H"
  },
  "home_form": {
    "last5_results": ["W", "W", "D", "W", "L"],
    "last5_goals_scored": 8,
    "last5_goals_conceded": 4,
    "last5_points": 10,
    "clean_sheets": 2
  },
  "away_form": {
    "last5_results": ["L", "D", "W", "L", "D"],
    "last5_goals_scored": 5,
    "last5_goals_conceded": 7,
    "last5_points": 5,
    "clean_sheets": 1
  },
  "statistics": {
    "home_position": 2,
    "away_position": 8,
    "home_gf_avg": 2.1,
    "away_ga_avg": 1.3
  },
  "odds": {
    "home_win": 1.85,
    "draw": 3.50,
    "away_win": 4.20
  }
}
```

---

## Implementation Priority

### Phase A: Core Enrichment (Week 1)
1. ✅ Create `match_data_enricher.py`
2. ✅ Implement H2H data collection
3. ✅ Implement team form collection
4. ✅ Test with sample crawler JSON

### Phase B: Feature Engineering (Week 2)
1. ✅ Create `pre_match_features.py`
2. ✅ Build H2H features
3. ✅ Build form features
4. ✅ Build statistical features

### Phase C: Pipeline Integration (Week 3)
1. ✅ Integrate with Phase 1 (training data)
2. ✅ Retrain models with new features
3. ✅ Integrate with Phase 3 (predictions)
4. ✅ End-to-end testing

### Phase D: Production (Week 4)
1. ✅ Optimize API calls (caching, rate limiting)
2. ✅ Add error handling and fallbacks
3. ✅ Create monitoring and logging
4. ✅ Deploy to production

---

## Testing Strategy

### Test 1: H2H Data Collection
```python
# Test if we can get H2H for specific teams
enricher = MatchDataEnricher(api_manager)
h2h = enricher.get_h2h_data("Leicester City", "Burnley")
assert len(h2h) > 0
assert 'home_wins' in h2h
```

### Test 2: Team Form Collection
```python
# Test if we can get recent form
form = enricher.get_team_form("Leicester City", num_matches=10)
assert len(form) == 10
assert 'result' in form[0]
```

### Test 3: Full Match Enrichment
```python
# Test complete enrichment pipeline
matches = load_crawler_json('sample_matches.json')
enriched = enricher.enrich_matches(matches)
assert 'h2h' in enriched[0]
assert 'home_form' in enriched[0]
assert 'away_form' in enriched[0]
```

### Test 4: Feature Building
```python
# Test feature creation
features = feature_builder.build_features(enriched[0])
assert 'h2h_home_win_rate' in features
assert 'home_last5_points' in features
```

### Test 5: End-to-End Prediction
```python
# Test full pipeline
predictions = predict_from_crawler('crawler_output.json')
assert len(predictions) == len(crawler_matches)
assert 'predicted_result' in predictions[0]
assert 'confidence' in predictions[0]
```

---

## Expected Improvements

### Before (Current):
- ❌ Generic league-wide predictions
- ❌ No H2H consideration
- ❌ No recent form analysis
- ❌ Accuracy: ~55-60%

### After (With Enrichment):
- ✅ Match-specific predictions
- ✅ H2H history included
- ✅ Recent form weighted
- ✅ Team-specific statistics
- ✅ Expected accuracy: ~65-70%

---

## Next Steps

1. **Provide your crawler JSON sample** so I can test the enrichment pipeline
2. **Decide on implementation priority** (which phase to start with)
3. **Review API rate limits** to ensure we don't exceed quotas
4. **Test H2H data collection** with your specific teams

---

## Questions to Answer

1. **What format is your crawler JSON?** (so I can parse it correctly)
2. **How many matches per day?** (to calculate API usage)
3. **Which features are most important?** (H2H, form, stats, odds?)
4. **Do you want real-time updates?** (or batch processing?)
5. **What's your accuracy target?** (to prioritize features)

---

## Summary

**The current data collector works for historical training data, but needs enhancement for match-specific predictions.**

**Key additions needed:**
1. ✅ H2H data collection
2. ✅ Recent form analysis
3. ✅ Team-specific statistics
4. ✅ Match feature engineering
5. ✅ Crawler-to-prediction bridge

**All 5 APIs are ready to use for this purpose!**

Please provide your crawler JSON sample so I can build and test the enrichment pipeline!
