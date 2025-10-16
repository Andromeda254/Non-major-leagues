#!/bin/bash

###############################################################################
# Complete Prediction Pipeline - All Markets
###############################################################################
# Workflow:
# 1. Crawler provides: upcoming_matches.csv/json
# 2. APIs provide: team_stats, odds_all_markets, historical_data
# 3. Feature Engineering (Phase 1)
# 4. Model Training (Phase 2)
# 5. Backtesting (Phase 3)
# 6. Prediction Generation (Phase 4)
###############################################################################

set -e  # Exit on error

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║     COMPLETE PREDICTION PIPELINE - ALL MARKETS                ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Parse arguments
RUN_CRAWLER=true
CONFIG_FILE="pipeline_config.json"

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-crawler)
            RUN_CRAWLER=false
            shift
            ;;
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-crawler       Skip running the crawler"
            echo "  --config FILE        Use custom config file"
            echo "  --help              Show this help"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

# Step 1: Run Crawler (if not skipped)
if [ "$RUN_CRAWLER" = true ]; then
    echo -e "${GREEN}[STEP 1/5]${NC} Running Crawler to fetch upcoming matches..."
    echo ""
    
    if command -v node &> /dev/null; then
        if [ -f "enhanced_soccer_match_crawler.js" ]; then
            echo -e "${BLUE}🚀 Starting crawler...${NC}"
            node enhanced_soccer_match_crawler.js
            echo -e "${GREEN}✅ Crawler completed${NC}"
        else
            echo -e "${YELLOW}⚠️ Crawler script not found, skipping...${NC}"
        fi
    else
        echo -e "${YELLOW}⚠️ Node.js not installed, skipping crawler...${NC}"
    fi
    echo ""
else
    echo -e "${YELLOW}[STEP 1/5] Skipping crawler (using existing data)${NC}"
    echo ""
fi

# Step 2: Check for crawler output
echo -e "${GREEN}[STEP 2/5]${NC} Checking crawler output..."
echo ""

CRAWLER_OUTPUT="soccer-match-intelligence"
if [ -d "$CRAWLER_OUTPUT" ]; then
    MATCH_FILES=$(ls -t $CRAWLER_OUTPUT/*matches*.json 2>/dev/null | head -1)
    if [ -n "$MATCH_FILES" ]; then
        echo -e "${GREEN}✅ Found crawler output: $MATCH_FILES${NC}"
    else
        echo -e "${YELLOW}⚠️ No match files found in $CRAWLER_OUTPUT${NC}"
    fi
else
    echo -e "${YELLOW}⚠️ Crawler output directory not found${NC}"
fi
echo ""

# Step 3: Run Complete Pipeline
echo -e "${GREEN}[STEP 3/5]${NC} Running Complete Prediction Pipeline..."
echo ""

echo -e "${BLUE}📊 Pipeline will execute:${NC}"
echo "   1. Load upcoming matches from crawler"
echo "   2. Fetch team stats, odds, historical data from APIs"
echo "   3. Feature Engineering (Phase 1)"
echo "   4. Model Training (Phase 2)"
echo "   5. Backtesting (Phase 3)"
echo "   6. Prediction Generation (Phase 4)"
echo ""

echo -e "${BLUE}🚀 Starting pipeline...${NC}"
python3 complete_prediction_pipeline.py --config "$CONFIG_FILE"

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Pipeline completed successfully${NC}"
else
    echo -e "${RED}❌ Pipeline failed${NC}"
    exit 1
fi
echo ""

# Step 4: Display Results
echo -e "${GREEN}[STEP 4/5]${NC} Displaying Results..."
echo ""

PREDICTIONS_DIR="predictions_output"
if [ -d "$PREDICTIONS_DIR" ]; then
    LATEST_JSON=$(ls -t $PREDICTIONS_DIR/predictions_*.json 2>/dev/null | head -1)
    LATEST_CSV=$(ls -t $PREDICTIONS_DIR/predictions_*.csv 2>/dev/null | head -1)
    LATEST_SUMMARY=$(ls -t $PREDICTIONS_DIR/summary_*.txt 2>/dev/null | head -1)
    
    if [ -n "$LATEST_JSON" ]; then
        echo -e "${BLUE}📄 Latest Predictions:${NC} $LATEST_JSON"
        
        # Count matches
        if command -v jq &> /dev/null; then
            MATCH_COUNT=$(jq 'length' "$LATEST_JSON" 2>/dev/null || echo "N/A")
            echo -e "${GREEN}   Total Matches: $MATCH_COUNT${NC}"
        fi
        echo ""
    fi
    
    if [ -n "$LATEST_CSV" ]; then
        echo -e "${BLUE}📊 CSV Export:${NC} $LATEST_CSV"
        echo ""
    fi
    
    if [ -n "$LATEST_SUMMARY" ]; then
        echo -e "${BLUE}📋 Summary Report:${NC}"
        echo -e "${YELLOW}────────────────────────────────────────────────────────────────${NC}"
        cat "$LATEST_SUMMARY"
        echo -e "${YELLOW}────────────────────────────────────────────────────────────────${NC}"
        echo ""
    fi
fi

# Step 5: Summary
echo -e "${GREEN}[STEP 5/5]${NC} Pipeline Summary"
echo ""

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                  PIPELINE COMPLETED                            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

echo -e "${GREEN}✅ All steps completed successfully${NC}"
echo ""

echo -e "${YELLOW}📁 Output Files:${NC}"
echo "   - Predictions: $PREDICTIONS_DIR/"
echo "   - Models: models_multi_market/"
echo "   - Logs: complete_pipeline.log"
echo ""

echo -e "${YELLOW}📊 Markets Covered:${NC}"
echo "   ✓ 1X2 (Home/Draw/Away)"
echo "   ✓ Over/Under 2.5 Goals"
echo "   ✓ Both Teams to Score (BTTS)"
echo "   ✓ First Half Result"
echo ""

echo -e "${YELLOW}💡 Next Steps:${NC}"
echo "   1. Review predictions: $LATEST_JSON"
echo "   2. Check CSV export: $LATEST_CSV"
echo "   3. Deploy to Stage 2 live testing"
echo ""

echo -e "${BLUE}════════════════════════════════════════════════════════════════${NC}"
