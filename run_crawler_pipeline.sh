#!/bin/bash

###############################################################################
# Crawler to ML Pipeline Workflow
###############################################################################
# This script orchestrates the full workflow:
# 1. Run JavaScript crawler to get matches
# 2. Bridge crawler output to ML pipeline
# 3. Generate predictions
#
# Usage:
#   ./run_crawler_pipeline.sh [--skip-crawler] [--crawler-file FILE]
###############################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
SKIP_CRAWLER=false
CRAWLER_FILE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-crawler)
            SKIP_CRAWLER=true
            shift
            ;;
        --crawler-file)
            CRAWLER_FILE="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-crawler       Skip running the crawler, use existing output"
            echo "  --crawler-file FILE  Use specific crawler output file"
            echo "  --help              Show this help message"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║        CRAWLER TO ML PIPELINE WORKFLOW                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Step 1: Run crawler (unless skipped)
if [ "$SKIP_CRAWLER" = false ]; then
    echo -e "${GREEN}[STEP 1/3]${NC} Running JavaScript crawler to fetch matches..."
    echo -e "${YELLOW}⚽ This will open a browser and scrape match data${NC}"
    echo ""
    
    # Check if node is installed
    if ! command -v node &> /dev/null; then
        echo -e "${RED}❌ Node.js is not installed${NC}"
        echo -e "${YELLOW}💡 Install Node.js first:${NC}"
        echo "   sudo apt-get install nodejs npm"
        exit 1
    fi
    
    # Check if dependencies are installed
    if [ ! -d "node_modules" ]; then
        echo -e "${YELLOW}📦 Installing Node.js dependencies...${NC}"
        npm install
    fi
    
    # Run crawler
    echo -e "${BLUE}🚀 Starting crawler...${NC}"
    node enhanced_soccer_match_crawler.js
    
    echo -e "${GREEN}✅ Crawler completed${NC}"
    echo ""
else
    echo -e "${YELLOW}[STEP 1/3] Skipping crawler (using existing output)${NC}"
    echo ""
fi

# Step 2: Bridge crawler output to pipeline
echo -e "${GREEN}[STEP 2/3]${NC} Bridging crawler output to ML pipeline..."
echo ""

# Build bridge command
BRIDGE_CMD="python3 crawler_to_pipeline_bridge.py"
if [ -n "$CRAWLER_FILE" ]; then
    BRIDGE_CMD="$BRIDGE_CMD --input $CRAWLER_FILE"
fi

# Run bridge
echo -e "${BLUE}🔄 Running bridge...${NC}"
$BRIDGE_CMD

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✅ Bridge completed successfully${NC}"
else
    echo -e "${RED}❌ Bridge failed${NC}"
    exit 1
fi
echo ""

# Step 3: Display results
echo -e "${GREEN}[STEP 3/3]${NC} Displaying results..."
echo ""

# Find latest prediction files
LATEST_JSON=$(ls -t predictions/predictions_*.json 2>/dev/null | head -1)
LATEST_SUMMARY=$(ls -t predictions/summary_*.txt 2>/dev/null | head -1)

if [ -n "$LATEST_JSON" ]; then
    echo -e "${BLUE}📊 Latest predictions:${NC} $LATEST_JSON"
    
    # Display match count
    MATCH_COUNT=$(jq '.total_matches' "$LATEST_JSON" 2>/dev/null || echo "N/A")
    echo -e "${GREEN}   Total matches: $MATCH_COUNT${NC}"
    echo ""
fi

if [ -n "$LATEST_SUMMARY" ]; then
    echo -e "${BLUE}📄 Summary report:${NC}"
    echo -e "${YELLOW}────────────────────────────────────────────────────────────────${NC}"
    cat "$LATEST_SUMMARY"
    echo -e "${YELLOW}────────────────────────────────────────────────────────────────${NC}"
    echo ""
fi

# Final summary
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    WORKFLOW COMPLETED                          ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✅ All steps completed successfully${NC}"
echo ""
echo -e "${YELLOW}📁 Output files:${NC}"
echo "   - Predictions: predictions/"
echo "   - Crawler data: soccer-match-intelligence/"
echo ""
echo -e "${YELLOW}💡 Next steps:${NC}"
echo "   1. Review predictions in: $LATEST_JSON"
echo "   2. Train models for better predictions:"
echo "      python run_pipeline.py --full --league E1"
echo "   3. Re-run with trained models for actual predictions"
echo ""
