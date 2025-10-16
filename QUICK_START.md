# Quick Start: Crawler to Predictions

## Data Flow Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    CRAWLER (JavaScript)                         │
│                                                                 │
│  enhanced_soccer_match_crawler.js                              │
│  • Scrapes matches from betting sites                          │
│  • Filters by time (5AM-11AM)                                  │
│  • Extracts odds and match details                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Saves JSON files to:
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              soccer-match-intelligence/                         │
│                                                                 │
│  • *_filtered_matches.json  ← Primary input                    │
│  • *_match_data.json        ← Fallback                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Automatically detected by:
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    BRIDGE (Python)                              │
│                                                                 │
│  crawler_to_pipeline_bridge.py                                 │
│  • Auto-finds latest crawler output                            │
│  • Formats for ML pipeline                                     │
│  • Runs predictions (if models exist)                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    Saves predictions to:
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    predictions/                                 │
│                                                                 │
│  • predictions_*.json  ← Full predictions                      │
│  • predictions_*.csv   ← CSV format                            │
│  • summary_*.txt       ← Human-readable                        │
└─────────────────────────────────────────────────────────────────┘
```

## Current Crawler Data

Your `soccer-match-intelligence/` directory already contains:

```bash
$ ls -lh soccer-match-intelligence/
soccer_2025-10-10T22-27-10-996Z_filtered_matches.json  (184K)
soccer_2025-10-10T22-32-26-507Z_filtered_matches.json  (182K)
soccer_2025-10-11_filtered_matches.json                 (53K)
soccer_2025-10-11T00-44-26-489Z_match_data.json        (2.0M)
soccer_2025-10-11T00-46-00-411Z_match_data.json        (723K)
soccer_2025-10-11T00-47-49-425Z_match_data.json        (2.2M)
test_filtered_matches_20251014_054020.json             (1.6K)
```

## How the Bridge Finds Crawler Data

The bridge (`crawler_to_pipeline_bridge.py`) automatically:

1. **Looks in `soccer-match-intelligence/` directory** (line 45)
2. **Searches for `*_filtered_matches.json` files first** (line 66)
3. **Falls back to `*_match_data.json` if no filtered files** (line 71)
4. **Picks the most recent file by modification time** (line 79)

```python
# From crawler_to_pipeline_bridge.py
self.crawler_output_dir = Path("soccer-match-intelligence")  # Line 45

def find_latest_crawler_output(self):
    # Look for filtered matches files
    filtered_files = list(self.crawler_output_dir.glob("*_filtered_matches.json"))
    
    if not filtered_files:
        # Fallback to regular match data files
        match_files = list(self.crawler_output_dir.glob("*_match_data.json"))
        if match_files:
            latest_file = max(match_files, key=lambda p: p.stat().st_mtime)
            return latest_file
    
    # Get the most recent file
    latest_file = max(filtered_files, key=lambda p: p.stat().st_mtime)
    return latest_file
```

## Usage Examples

### 1. Use Latest Crawler Data (Auto-detect)

```bash
# Bridge automatically finds latest file in soccer-match-intelligence/
python3 crawler_to_pipeline_bridge.py
```

### 2. Use Specific Crawler File

```bash
# Specify exact file to process
python3 crawler_to_pipeline_bridge.py --input soccer-match-intelligence/soccer_2025-10-11_filtered_matches.json
```

### 3. Full Workflow (Crawler + Bridge)

```bash
# Run crawler, then automatically bridge to pipeline
./run_crawler_pipeline.sh
```

### 4. Skip Crawler, Use Existing Data

```bash
# Use existing crawler data without re-running crawler
./run_crawler_pipeline.sh --skip-crawler
```

## Expected Crawler JSON Format

The bridge handles these formats from `soccer-match-intelligence/`:

### Format 1: Array of matches
```json
[
  {
    "id": "match_001",
    "teams": ["Team A", "Team B"],
    "time": "2025-10-14 08:00:00",
    "odds": {
      "home": 2.5,
      "draw": 3.2,
      "away": 2.8
    },
    "league": "Championship"
  }
]
```

### Format 2: Object with matches key
```json
{
  "matches": [...],
  "timestamp": "2025-10-14T05:00:00",
  "total": 10
}
```

### Format 3: Object with data/filteredMatches key
```json
{
  "data": [...],
  "filteredMatches": [...]
}
```

## Verification Commands

### Check what crawler data exists
```bash
ls -lh soccer-match-intelligence/*_filtered_matches.json
```

### View latest crawler output
```bash
cat $(ls -t soccer-match-intelligence/*_filtered_matches.json | head -1) | jq '.' | head -50
```

### Count matches in latest file
```bash
cat $(ls -t soccer-match-intelligence/*_filtered_matches.json | head -1) | jq 'length'
```

### Test bridge with existing data
```bash
python3 crawler_to_pipeline_bridge.py
```

## Directory Structure Confirmation

```
Non-major-leagues/
├── enhanced_soccer_match_crawler.js    # Crawler (produces data)
├── crawler_to_pipeline_bridge.py       # Bridge (consumes data)
│
├── soccer-match-intelligence/          # ← Crawler writes here
│   ├── *_filtered_matches.json         # ← Bridge reads from here
│   └── *_match_data.json               # ← Fallback source
│
└── predictions/                        # ← Bridge writes here
    ├── predictions_*.json
    ├── predictions_*.csv
    └── summary_*.txt
```

## Next Steps

1. **Process existing crawler data:**
   ```bash
   python3 crawler_to_pipeline_bridge.py
   ```

2. **View the results:**
   ```bash
   cat predictions/summary_*.txt | tail -100
   ```

3. **Train models for predictions:**
   ```bash
   python run_pipeline.py --full --league E1
   ```

4. **Run fresh crawler + predictions:**
   ```bash
   ./run_crawler_pipeline.sh
   ```

## Troubleshooting

**Q: Bridge says "No crawler output file found"**
- Check: `ls soccer-match-intelligence/*.json`
- Run crawler first: `node enhanced_soccer_match_crawler.js`

**Q: Want to use specific file instead of latest**
- Use: `--input` flag with full path
- Example: `python3 crawler_to_pipeline_bridge.py --input soccer-match-intelligence/soccer_2025-10-11_filtered_matches.json`

**Q: Predictions show "pending"**
- This is expected without trained models
- Train first: `python run_pipeline.py --full --league E1`
- Then re-run bridge for actual predictions

## Summary

✅ **Crawler data location:** `soccer-match-intelligence/`  
✅ **Bridge auto-detects:** Latest `*_filtered_matches.json`  
✅ **Predictions saved to:** `predictions/`  
✅ **No manual file specification needed** (unless you want specific file)
