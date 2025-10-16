# Crawler to ML Pipeline Workflow

This document explains how to use the JavaScript crawler with the ML prediction pipeline to get match predictions.

## Overview

The workflow consists of three main steps:

1. **JavaScript Crawler** → Scrapes live matches from betting sites
2. **Bridge Script** → Converts crawler output to ML pipeline format
3. **ML Pipeline** → Generates predictions for the matches

## Quick Start

### Prerequisites

```bash
# Install Node.js dependencies
npm install

# Install Python dependencies
pip install -r requirements.txt

# Ensure credentials are set in .env or api_keys_template.env
```

### Run Complete Workflow

```bash
# Make script executable
chmod +x run_crawler_pipeline.sh

# Run full workflow (crawler + predictions)
./run_crawler_pipeline.sh
```

## Step-by-Step Usage

### Step 1: Run Crawler Independently

```bash
# Run the enhanced crawler
node enhanced_soccer_match_crawler.js
```

**What it does:**
- Opens browser and navigates to betting site
- Logs in with credentials from `.env`
- Extracts upcoming soccer matches
- Filters matches for optimal time windows (5AM-11AM)
- Saves results to `soccer-match-intelligence/`

**Output files:**
- `soccer-match-intelligence/*_filtered_matches.json` - Filtered matches for ML
- `soccer-match-intelligence/*_match_data.json` - All matches
- `exports/soccer_matches_*.csv` - CSV export

### Step 2: Bridge to ML Pipeline

```bash
# Auto-detect latest crawler output
python3 crawler_to_pipeline_bridge.py

# Or specify a specific file
python3 crawler_to_pipeline_bridge.py --input soccer-match-intelligence/soccer_2025-10-14_filtered_matches.json
```

**What it does:**
- Reads crawler JSON output
- Formats matches for ML pipeline
- Runs predictions (requires trained models)
- Saves predictions to `predictions/`

**Output files:**
- `predictions/predictions_*.json` - Predictions in JSON format
- `predictions/predictions_*.csv` - Predictions in CSV format
- `predictions/summary_*.txt` - Human-readable summary

### Step 3: View Predictions

```bash
# View latest summary
cat predictions/summary_*.txt | tail -100

# View JSON predictions
cat predictions/predictions_*.json | jq '.'

# View CSV predictions
column -t -s, predictions/predictions_*.csv | less
```

## Advanced Usage

### Skip Crawler (Use Existing Data)

```bash
# Use existing crawler output
./run_crawler_pipeline.sh --skip-crawler
```

### Use Specific Crawler File

```bash
# Specify exact file to process
./run_crawler_pipeline.sh --skip-crawler --crawler-file soccer-match-intelligence/soccer_2025-10-14_filtered_matches.json
```

### Train Models First (Recommended)

For accurate predictions, train the ML models first:

```bash
# Train models on historical data
python run_pipeline.py --full --league E1

# Then run predictions on crawler data
./run_crawler_pipeline.sh
```

## File Structure

```
Non-major-leagues/
├── enhanced_soccer_match_crawler.js    # Main crawler script
├── crawler_to_pipeline_bridge.py       # Bridge script
├── run_crawler_pipeline.sh             # Orchestration script
├── master_pipeline.py                  # ML pipeline orchestrator
├── run_pipeline.py                     # Pipeline runner
├── config.yaml                         # Configuration
│
├── soccer-match-intelligence/          # Crawler output
│   ├── *_filtered_matches.json
│   └── *_match_data.json
│
├── predictions/                        # Prediction output
│   ├── predictions_*.json
│   ├── predictions_*.csv
│   └── summary_*.txt
│
├── models/                             # Trained ML models
│   └── *.pkl
│
└── pipeline_output/                    # Pipeline artifacts
    ├── phase1_output/
    ├── phase2_output/
    └── phase3_output/
```

## Crawler Output Format

The crawler produces JSON with this structure:

```json
[
  {
    "id": "match_123",
    "teams": ["Team A", "Team B"],
    "time": "2025-10-14 08:00:00",
    "odds": {
      "home": 2.5,
      "draw": 3.2,
      "away": 2.8
    },
    "league": "Championship",
    "source": "crawler"
  }
]
```

## Prediction Output Format

The bridge produces predictions in this format:

```json
{
  "timestamp": "2025-10-14T05:30:00",
  "source_file": "soccer-match-intelligence/...",
  "total_matches": 10,
  "predictions": [
    {
      "match_id": "match_123",
      "home_team": "Team A",
      "away_team": "Team B",
      "league": "Championship",
      "match_time": "2025-10-14 08:00:00",
      "home_odds": 2.5,
      "draw_odds": 3.2,
      "away_odds": 2.8,
      "prediction": "home_win",
      "confidence": 0.72,
      "recommended_bet": "home",
      "expected_value": 0.15
    }
  ]
}
```

## Configuration

### Crawler Configuration

Edit `enhanced_soccer_match_crawler.js` or set environment variables:

```bash
# In .env or api_keys_template.env
BETIKA_USERNAME=your_username
BETIKA_PASSWORD=your_password
```

### Pipeline Configuration

Edit `config.yaml` to customize:

- Data sources and API keys
- Model parameters
- Risk management settings
- Betting strategy

## Troubleshooting

### Crawler Issues

**Browser doesn't open:**
```bash
# Install dependencies
npm install puppeteer puppeteer-extra puppeteer-extra-plugin-stealth
```

**Login fails:**
- Check credentials in `.env`
- Verify site is accessible
- Check for CAPTCHA or rate limiting

**No matches found:**
- Check time filters in crawler
- Verify site structure hasn't changed
- Look at `screenshots/` for debugging

### Bridge Issues

**No crawler output found:**
```bash
# Check directory
ls -la soccer-match-intelligence/

# Run crawler first
node enhanced_soccer_match_crawler.js
```

**Predictions show "pending":**
- This is expected without trained models
- Train models first: `python run_pipeline.py --full --league E1`
- Models will be saved to `models/` directory

**Format errors:**
- Check crawler JSON structure
- Verify team names are extracted correctly
- Look at bridge logs for details

## Integration with Existing Pipeline

The bridge integrates with the existing 4-phase pipeline:

1. **Phase 1** (Data Collection) - Can use crawler data as additional source
2. **Phase 2** (Model Training) - Train on historical + crawler data
3. **Phase 3** (Backtesting) - Validate predictions
4. **Phase 4** (Deployment) - Serve predictions via API

### Using Crawler Data for Training

```python
# In your pipeline code
from crawler_to_pipeline_bridge import CrawlerPipelineBridge

bridge = CrawlerPipelineBridge()
crawler_matches = bridge.load_crawler_matches(file_path)
training_df = bridge.format_matches_for_pipeline(crawler_matches)

# Use training_df in Phase 1 preprocessing
```

## Automation

### Cron Job for Regular Predictions

```bash
# Add to crontab (every 6 hours)
0 */6 * * * cd /home/kali/Non-major-leagues && ./run_crawler_pipeline.sh >> logs/crawler_pipeline.log 2>&1
```

### Continuous Monitoring

```bash
# Watch for new predictions
watch -n 60 'ls -lh predictions/predictions_*.json | tail -5'
```

## Performance Tips

1. **Cache crawler results** - Reuse recent data to avoid re-scraping
2. **Train models offline** - Don't retrain on every prediction
3. **Batch predictions** - Process multiple matches together
4. **Filter matches** - Focus on high-confidence predictions
5. **Monitor API limits** - Respect rate limits on data sources

## Next Steps

1. ✅ Run crawler to get matches
2. ✅ Bridge to pipeline format
3. ⏳ Train ML models on historical data
4. ⏳ Generate predictions with trained models
5. ⏳ Validate predictions with backtesting
6. ⏳ Deploy prediction API
7. ⏳ Monitor performance and refine

## Support

For issues or questions:
- Check logs in `logs/`
- Review screenshots in `screenshots/`
- Examine crawler output in `soccer-match-intelligence/`
- Check pipeline output in `pipeline_output/`

## License

Same as main project license.
