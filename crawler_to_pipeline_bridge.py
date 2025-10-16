#!/usr/bin/env python3
"""
Crawler to ML Pipeline Bridge
==============================

This script bridges the JavaScript crawler output with the ML prediction pipeline.
It reads matches from the crawler's JSON output, formats them for the pipeline,
runs predictions, and saves results.

Author: AI Assistant
Date: 2025-10-14
"""

import os
import sys
import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import yaml

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import pipeline components
from master_pipeline import MasterPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CrawlerPipelineBridge:
    """Bridge between crawler output and ML pipeline"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize the bridge"""
        self.config_path = config_path
        self.config = self._load_config()
        self.crawler_output_dir = Path("soccer-match-intelligence")
        self.predictions_output_dir = Path("predictions")
        self.predictions_output_dir.mkdir(exist_ok=True)
        
        logger.info("Initialized Crawler-Pipeline Bridge")
    
    def _load_config(self) -> Dict:
        """Load configuration"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def find_latest_crawler_output(self) -> Optional[Path]:
        """Find the latest filtered matches JSON from crawler"""
        try:
            # Look for filtered matches files
            filtered_files = list(self.crawler_output_dir.glob("*_filtered_matches.json"))
            
            if not filtered_files:
                logger.warning("No filtered matches files found")
                # Fallback to regular match data files
                match_files = list(self.crawler_output_dir.glob("*_match_data.json"))
                if match_files:
                    latest_file = max(match_files, key=lambda p: p.stat().st_mtime)
                    logger.info(f"Using match data file: {latest_file}")
                    return latest_file
                return None
            
            # Get the most recent file
            latest_file = max(filtered_files, key=lambda p: p.stat().st_mtime)
            logger.info(f"Found latest crawler output: {latest_file}")
            return latest_file
            
        except Exception as e:
            logger.error(f"Error finding crawler output: {e}")
            return None
    
    def load_crawler_matches(self, file_path: Path) -> List[Dict]:
        """Load matches from crawler JSON output"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Handle different JSON structures
            if isinstance(data, list):
                matches = data
            elif isinstance(data, dict):
                # Try common keys
                matches = data.get('matches', data.get('data', data.get('filteredMatches', [])))
            else:
                logger.error(f"Unexpected data structure: {type(data)}")
                return []
            
            logger.info(f"Loaded {len(matches)} matches from crawler")
            return matches
            
        except Exception as e:
            logger.error(f"Error loading crawler matches: {e}")
            return []
    
    def format_matches_for_pipeline(self, crawler_matches: List[Dict]) -> pd.DataFrame:
        """Convert crawler matches to pipeline-compatible DataFrame"""
        try:
            formatted_matches = []
            
            for match in crawler_matches:
                # Extract match details
                match_id = match.get('id', f"match_{len(formatted_matches)}")
                
                # Handle team names (various formats)
                teams = match.get('teams', [])
                if isinstance(teams, list) and len(teams) >= 2:
                    home_team = teams[0]
                    away_team = teams[1]
                elif isinstance(teams, str):
                    # Try to split "Team A vs Team B"
                    parts = teams.replace(' v ', ' vs ').split(' vs ')
                    if len(parts) >= 2:
                        home_team = parts[0].strip()
                        away_team = parts[1].strip()
                    else:
                        logger.warning(f"Could not parse teams: {teams}")
                        continue
                else:
                    home_team = match.get('home_team', match.get('homeTeam', 'Unknown'))
                    away_team = match.get('away_team', match.get('awayTeam', 'Unknown'))
                
                # Extract odds
                odds = match.get('odds', {})
                home_odds = odds.get('home', odds.get('1', odds.get('home_odds', None)))
                draw_odds = odds.get('draw', odds.get('X', odds.get('draw_odds', None)))
                away_odds = odds.get('away', odds.get('2', odds.get('away_odds', None)))
                
                # Extract other details
                match_time = match.get('time', match.get('match_time', match.get('start_time', '')))
                league = match.get('league', match.get('competition', 'Unknown'))
                
                formatted_match = {
                    'match_id': match_id,
                    'home_team': home_team,
                    'away_team': away_team,
                    'league': league,
                    'match_time': match_time,
                    'home_odds': home_odds,
                    'draw_odds': draw_odds,
                    'away_odds': away_odds,
                    'source': match.get('source', 'crawler'),
                    'timestamp': match.get('timestamp', datetime.now().isoformat())
                }
                
                formatted_matches.append(formatted_match)
            
            df = pd.DataFrame(formatted_matches)
            logger.info(f"Formatted {len(df)} matches for pipeline")
            return df
            
        except Exception as e:
            logger.error(f"Error formatting matches: {e}")
            return pd.DataFrame()
    
    def run_predictions(self, matches_df: pd.DataFrame) -> pd.DataFrame:
        """Run ML pipeline predictions on matches"""
        try:
            logger.info("Initializing ML pipeline for predictions...")
            
            # Save matches to temporary file for pipeline
            temp_input_file = self.predictions_output_dir / "temp_crawler_matches.csv"
            matches_df.to_csv(temp_input_file, index=False)
            logger.info(f"Saved matches to {temp_input_file}")
            
            # Initialize pipeline
            pipeline = MasterPipeline(config_path=self.config_path)
            
            # For now, we'll use a simplified prediction approach
            # In production, you'd run the full pipeline or use trained models
            logger.info("Running predictions...")
            
            # Add prediction columns (placeholder for actual model predictions)
            predictions_df = matches_df.copy()
            predictions_df['prediction'] = 'pending'
            predictions_df['confidence'] = 0.0
            predictions_df['recommended_bet'] = None
            predictions_df['expected_value'] = 0.0
            
            # TODO: Integrate with actual trained models from Phase 2
            # For now, we'll mark this as requiring model training
            logger.warning("⚠️ Predictions require trained models from Phase 2")
            logger.info("💡 Run the full pipeline first to train models:")
            logger.info("   python run_pipeline.py --full --league E1")
            
            return predictions_df
            
        except Exception as e:
            logger.error(f"Error running predictions: {e}")
            return matches_df
    
    def save_predictions(self, predictions_df: pd.DataFrame, source_file: Path):
        """Save predictions to JSON and CSV"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save as JSON
            json_output = self.predictions_output_dir / f"predictions_{timestamp}.json"
            predictions_dict = predictions_df.to_dict(orient='records')
            
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'source_file': str(source_file),
                'total_matches': len(predictions_df),
                'predictions': predictions_dict
            }
            
            with open(json_output, 'w') as f:
                json.dump(output_data, f, indent=2)
            logger.info(f"✅ Saved predictions to {json_output}")
            
            # Save as CSV
            csv_output = self.predictions_output_dir / f"predictions_{timestamp}.csv"
            predictions_df.to_csv(csv_output, index=False)
            logger.info(f"✅ Saved predictions to {csv_output}")
            
            # Create summary
            self._create_summary(predictions_df, timestamp)
            
            return json_output, csv_output
            
        except Exception as e:
            logger.error(f"Error saving predictions: {e}")
            return None, None
    
    def _create_summary(self, predictions_df: pd.DataFrame, timestamp: str):
        """Create a summary report"""
        try:
            summary_file = self.predictions_output_dir / f"summary_{timestamp}.txt"
            
            with open(summary_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("MATCH PREDICTIONS SUMMARY\n")
                f.write("=" * 80 + "\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total Matches: {len(predictions_df)}\n\n")
                
                f.write("MATCHES:\n")
                f.write("-" * 80 + "\n")
                
                for idx, row in predictions_df.iterrows():
                    f.write(f"\nMatch {idx + 1}:\n")
                    f.write(f"  {row['home_team']} vs {row['away_team']}\n")
                    f.write(f"  League: {row['league']}\n")
                    f.write(f"  Time: {row['match_time']}\n")
                    if row.get('home_odds'):
                        f.write(f"  Odds: {row['home_odds']} / {row.get('draw_odds', 'N/A')} / {row['away_odds']}\n")
                    f.write(f"  Prediction: {row.get('prediction', 'N/A')}\n")
                    f.write(f"  Confidence: {row.get('confidence', 0.0):.2%}\n")
                    if row.get('recommended_bet'):
                        f.write(f"  Recommended Bet: {row['recommended_bet']}\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n")
            
            logger.info(f"✅ Created summary report: {summary_file}")
            
        except Exception as e:
            logger.error(f"Error creating summary: {e}")
    
    def run(self, crawler_output_file: Optional[Path] = None):
        """Main execution flow"""
        try:
            logger.info("=" * 80)
            logger.info("CRAWLER TO PIPELINE BRIDGE - STARTING")
            logger.info("=" * 80)
            
            # Step 1: Find crawler output
            if crawler_output_file is None:
                crawler_output_file = self.find_latest_crawler_output()
            
            if crawler_output_file is None:
                logger.error("❌ No crawler output file found")
                logger.info("💡 Run the crawler first:")
                logger.info("   node enhanced_soccer_match_crawler.js")
                return False
            
            # Step 2: Load matches
            logger.info(f"📂 Loading matches from: {crawler_output_file}")
            crawler_matches = self.load_crawler_matches(crawler_output_file)
            
            if not crawler_matches:
                logger.error("❌ No matches found in crawler output")
                return False
            
            # Step 3: Format for pipeline
            logger.info("🔄 Formatting matches for ML pipeline...")
            matches_df = self.format_matches_for_pipeline(crawler_matches)
            
            if matches_df.empty:
                logger.error("❌ Failed to format matches")
                return False
            
            logger.info(f"✅ Formatted {len(matches_df)} matches")
            
            # Step 4: Run predictions
            logger.info("🤖 Running ML predictions...")
            predictions_df = self.run_predictions(matches_df)
            
            # Step 5: Save results
            logger.info("💾 Saving predictions...")
            json_file, csv_file = self.save_predictions(predictions_df, crawler_output_file)
            
            if json_file and csv_file:
                logger.info("=" * 80)
                logger.info("✅ BRIDGE EXECUTION COMPLETED SUCCESSFULLY")
                logger.info("=" * 80)
                logger.info(f"📊 Processed {len(predictions_df)} matches")
                logger.info(f"📄 JSON Output: {json_file}")
                logger.info(f"📄 CSV Output: {csv_file}")
                logger.info("=" * 80)
                return True
            else:
                logger.error("❌ Failed to save predictions")
                return False
            
        except Exception as e:
            logger.error(f"❌ Bridge execution failed: {e}")
            logger.exception("Stack trace:")
            return False


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Bridge crawler output to ML pipeline")
    parser.add_argument('--input', type=str, help='Path to crawler JSON file (optional, auto-detects latest)')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    
    args = parser.parse_args()
    
    # Initialize bridge
    bridge = CrawlerPipelineBridge(config_path=args.config)
    
    # Run bridge
    input_file = Path(args.input) if args.input else None
    success = bridge.run(crawler_output_file=input_file)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
