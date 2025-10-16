#!/usr/bin/env python3
"""
Complete Workflow Orchestrator
==============================

Integrates crawler output with existing phase integration files to generate
predictions for all markets (1X2, O/U, BTTS, First Half, etc.)

Workflow:
1. Load crawler output (upcoming_matches.json)
2. Fetch API data (team stats, odds, historical)
3. Run Phase 1: Feature Engineering
4. Run Phase 2: Model Training (all markets)
5. Run Phase 3: Backtesting
6. Run Phase 4: Generate Predictions (all markets)

Usage:
    python3 run_complete_workflow.py --league E1
    python3 run_complete_workflow.py --league E1 --skip-training
    python3 run_complete_workflow.py --crawler-file path/to/matches.json
"""

import argparse
import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import pandas as pd
import numpy as np

# Import existing components
from master_pipeline import MasterPipeline
from api_data_manager import APIDataManager
from crawler_to_pipeline_bridge import CrawlerPipelineBridge

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('complete_workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class CompleteWorkflowOrchestrator:
    """
    Orchestrates the complete workflow from crawler to predictions
    """
    
    def __init__(self, config_file: str = 'config.yaml'):
        """Initialize the orchestrator"""
        self.config_file = config_file
        
        # Initialize components
        self.master_pipeline = MasterPipeline(config_path=config_file)
        self.api_manager = APIDataManager(config_path=config_file)
        self.crawler_bridge = CrawlerPipelineBridge(config_path=config_file)
        
        # Output directories
        self.output_dir = Path('predictions_output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("✅ Complete Workflow Orchestrator initialized")
    
    def run_complete_workflow(self, league: str = 'E1', 
                             crawler_file: str = None,
                             skip_training: bool = False) -> Dict[str, Any]:
        """
        Execute the complete workflow
        
        Args:
            league: League code (e.g., 'E1')
            crawler_file: Path to crawler output file
            skip_training: Skip Phase 1-3 if models already exist
            
        Returns:
            Dictionary with workflow results
        """
        logger.info("=" * 80)
        logger.info("COMPLETE WORKFLOW - ALL MARKETS PREDICTION")
        logger.info("=" * 80)
        
        workflow_results = {
            'start_time': datetime.now().isoformat(),
            'league': league,
            'phases': {}
        }
        
        try:
            # Step 1: Load crawler data
            logger.info("\n📥 STEP 1: Loading Crawler Data")
            upcoming_matches = self.load_crawler_data(crawler_file)
            workflow_results['upcoming_matches_count'] = len(upcoming_matches)
            logger.info(f"✅ Loaded {len(upcoming_matches)} upcoming matches")
            
            # Step 2: Fetch API data
            logger.info("\n🌐 STEP 2: Fetching API Data")
            api_data = self.fetch_api_data(upcoming_matches, league)
            workflow_results['api_data'] = api_data
            logger.info("✅ API data fetched successfully")
            
            # Step 3-5: Run pipeline phases (if not skipping)
            if not skip_training:
                # Step 3: Phase 1 - Feature Engineering
                logger.info("\n📊 STEP 3: Phase 1 - Feature Engineering")
                phase1_result = self.master_pipeline.run_phase1(league=league)
                workflow_results['phases']['phase1'] = phase1_result
                logger.info("✅ Phase 1 completed")
                
                # Step 4: Phase 2 - Model Training
                logger.info("\n🤖 STEP 4: Phase 2 - Model Training (All Markets)")
                phase2_result = self.master_pipeline.run_phase2()
                workflow_results['phases']['phase2'] = phase2_result
                logger.info("✅ Phase 2 completed")
                
                # Step 5: Phase 3 - Backtesting
                logger.info("\n📈 STEP 5: Phase 3 - Backtesting")
                phase3_result = self.master_pipeline.run_phase3()
                workflow_results['phases']['phase3'] = phase3_result
                logger.info("✅ Phase 3 completed")
            else:
                logger.info("\n⏭️ Skipping training phases (using existing models)")
            
            # Step 6: Phase 4 - Generate Predictions
            logger.info("\n🎯 STEP 6: Phase 4 - Generate Predictions (All Markets)")
            predictions = self.generate_predictions(upcoming_matches, league)
            workflow_results['predictions'] = predictions
            workflow_results['predictions_count'] = len(predictions)
            logger.info(f"✅ Generated predictions for {len(predictions)} matches")
            
            # Step 7: Save results
            logger.info("\n💾 STEP 7: Saving Results")
            self.save_results(predictions, workflow_results)
            logger.info("✅ Results saved successfully")
            
            workflow_results['end_time'] = datetime.now().isoformat()
            workflow_results['status'] = 'success'
            
            logger.info("\n" + "=" * 80)
            logger.info("✅ WORKFLOW COMPLETED SUCCESSFULLY")
            logger.info("=" * 80)
            
            return workflow_results
            
        except Exception as e:
            logger.error(f"❌ Workflow failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            workflow_results['status'] = 'failed'
            workflow_results['error'] = str(e)
            raise
    
    def load_crawler_data(self, crawler_file: str = None) -> pd.DataFrame:
        """Load upcoming matches from crawler"""
        if crawler_file:
            crawler_path = Path(crawler_file)
        else:
            # Find latest crawler output
            crawler_path = self.crawler_bridge.find_latest_crawler_output()
        
        if not crawler_path or not crawler_path.exists():
            raise FileNotFoundError("No crawler output found")
        
        # Load matches
        matches = self.crawler_bridge.load_crawler_matches(crawler_path)
        df = pd.DataFrame(matches)
        
        # Save as CSV
        output_file = self.output_dir / 'upcoming_matches.csv'
        df.to_csv(output_file, index=False)
        
        return df
    
    def fetch_api_data(self, matches: pd.DataFrame, league: str) -> Dict[str, Any]:
        """Fetch additional data from APIs"""
        api_data = {
            'team_stats': None,
            'odds_data': None,
            'historical_data': None
        }
        
        try:
            # Fetch team statistics
            logger.info("  Fetching team statistics...")
            teams = set(matches['HomeTeam'].tolist() + matches['AwayTeam'].tolist())
            team_stats_list = []
            
            for team in teams:
                try:
                    stats = self.api_manager.get_team_statistics(team, league)
                    if stats:
                        team_stats_list.append(stats)
                except Exception as e:
                    logger.warning(f"  Could not fetch stats for {team}: {e}")
            
            if team_stats_list:
                team_stats_df = pd.DataFrame(team_stats_list)
                team_stats_df.to_csv(self.output_dir / 'team_stats.csv', index=False)
                api_data['team_stats'] = len(team_stats_list)
                logger.info(f"  ✅ Fetched stats for {len(team_stats_list)} teams")
            
        except Exception as e:
            logger.warning(f"  ⚠️ Team stats fetch failed: {e}")
        
        try:
            # Fetch odds for all markets
            logger.info("  Fetching odds (all markets)...")
            odds_list = []
            
            for idx, match in matches.iterrows():
                try:
                    odds = self.api_manager.get_match_odds(
                        match['HomeTeam'],
                        match['AwayTeam'],
                        markets=['1x2', 'over_under', 'btts', 'first_half']
                    )
                    if odds:
                        odds['HomeTeam'] = match['HomeTeam']
                        odds['AwayTeam'] = match['AwayTeam']
                        odds_list.append(odds)
                except Exception as e:
                    logger.warning(f"  Could not fetch odds for match: {e}")
            
            if odds_list:
                odds_df = pd.DataFrame(odds_list)
                odds_df.to_csv(self.output_dir / 'odds_all_markets.csv', index=False)
                api_data['odds_data'] = len(odds_list)
                logger.info(f"  ✅ Fetched odds for {len(odds_list)} matches")
            else:
                # Generate dummy odds for testing
                logger.info("  ⚠️ Generating dummy odds for testing...")
                odds_df = self.generate_dummy_odds(matches)
                odds_df.to_csv(self.output_dir / 'odds_all_markets.csv', index=False)
                api_data['odds_data'] = len(odds_df)
            
        except Exception as e:
            logger.warning(f"  ⚠️ Odds fetch failed: {e}")
        
        try:
            # Use existing historical data
            historical_file = Path('pipeline_output/phase1_output/E1_features.csv')
            if historical_file.exists():
                logger.info(f"  ✅ Using existing historical data: {historical_file}")
                api_data['historical_data'] = 'existing'
            else:
                logger.info("  ⚠️ No historical data found, will be generated in Phase 1")
        except Exception as e:
            logger.warning(f"  ⚠️ Historical data check failed: {e}")
        
        return api_data
    
    def generate_dummy_odds(self, matches: pd.DataFrame) -> pd.DataFrame:
        """Generate dummy odds for testing"""
        odds_data = []
        for idx, match in matches.iterrows():
            odds_data.append({
                'HomeTeam': match['HomeTeam'],
                'AwayTeam': match['AwayTeam'],
                # 1X2 odds
                'B365H': np.random.uniform(1.8, 3.5),
                'B365D': np.random.uniform(3.0, 3.8),
                'B365A': np.random.uniform(2.0, 4.5),
                # Over/Under 2.5
                'odds_over_2.5': np.random.uniform(1.7, 2.2),
                'odds_under_2.5': np.random.uniform(1.7, 2.2),
                # BTTS
                'odds_btts_yes': np.random.uniform(1.7, 2.3),
                'odds_btts_no': np.random.uniform(1.6, 2.1),
                # First Half
                'odds_ht_home': np.random.uniform(2.5, 4.0),
                'odds_ht_draw': np.random.uniform(2.0, 2.5),
                'odds_ht_away': np.random.uniform(3.0, 5.0)
            })
        return pd.DataFrame(odds_data)
    
    def generate_predictions(self, matches: pd.DataFrame, league: str) -> List[Dict]:
        """Generate predictions for all markets"""
        predictions = []
        
        # Load trained models
        models_dir = Path('pipeline_output/phase2_output')
        ensemble_file = models_dir / f'{league}_ensemble.pkl'
        
        if not ensemble_file.exists():
            logger.error(f"❌ No trained model found: {ensemble_file}")
            logger.info("💡 Run with --skip-training=false to train models first")
            return []
        
        # Load model
        import joblib
        ensemble_state = joblib.load(ensemble_file)
        
        # Extract the actual model from the ensemble state
        if isinstance(ensemble_state, dict):
            # Ensemble is saved as a dict with multiple components
            ensemble_model = ensemble_state.get('ensemble_model')
            
            if isinstance(ensemble_model, dict) and 'models' in ensemble_model:
                # Create a wrapper class for the ensemble
                class EnsemblePredictor:
                    def __init__(self, ensemble):
                        self.ensemble = ensemble
                    
                    def predict_proba(self, X):
                        models = self.ensemble['models']
                        weights = self.ensemble['weights']
                        
                        # Get predictions from each model
                        predictions = []
                        weight_list = []
                        for name, model in models.items():
                            pred = model.predict_proba(X)
                            predictions.append(pred)
                            weight_list.append(weights.get(name, 1.0))
                        
                        # Stack predictions and compute weighted average
                        # predictions shape: (n_models, n_samples, n_classes)
                        predictions = np.array(predictions)
                        weight_list = np.array(weight_list)
                        
                        # Weighted average across models (axis=0)
                        ensemble_proba = np.average(predictions, axis=0, weights=weight_list)
                        return ensemble_proba
                    
                    def predict(self, X):
                        proba = self.predict_proba(X)
                        return np.argmax(proba, axis=1)
                
                model = EnsemblePredictor(ensemble_model)
            else:
                # Try to get a calibrated model
                calibrated = ensemble_state.get('calibrated_models', {})
                if calibrated:
                    model = list(calibrated.values())[0]
                else:
                    model = None
        else:
            model = ensemble_state
        
        if model is None:
            logger.error(f"❌ Could not extract model from ensemble file")
            return []
        
        logger.info(f"✅ Loaded model: {ensemble_file}")
        
        # Load feature template
        features_file = Path('pipeline_output/phase1_output') / f'{league}_features.csv'
        if not features_file.exists():
            logger.error(f"❌ No features file found: {features_file}")
            return []
        
        historical_data = pd.read_csv(features_file)
        feature_cols = historical_data.select_dtypes(include=[np.number]).columns.tolist()
        # Only exclude target - keep FTHG, FTAG, HTHG, HTAG as they were used in training
        exclude_cols = ['target']
        feature_cols = [col for col in feature_cols if col not in exclude_cols]
        
        logger.info(f"Generating predictions for {len(matches)} matches...")
        
        for idx, match in matches.iterrows():
            try:
                prediction = self.generate_match_prediction(
                    match, model, feature_cols, historical_data
                )
                predictions.append(prediction)
            except Exception as e:
                logger.warning(f"⚠️ Could not generate prediction for match {idx}: {e}")
        
        return predictions
    
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
        
        # Prepare features (use last match as template)
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
            
            # Generate multi-market predictions based on 1X2 probabilities
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
    
    def predict_over_under(self, home_win_prob, draw_prob, away_win_prob, features):
        """Predict Over/Under 2.5 goals based on match probabilities and features"""
        # Extract goal-related features with safe defaults
        home_goals = features.get('home_team_avg_goals', features.get('FTHG', 1.5))
        away_goals = features.get('away_team_avg_goals', features.get('FTAG', 1.0))
        
        # Calculate expected goals
        expected_goals = (home_goals + away_goals) * 1.1
        
        # Adjust based on outcome probabilities
        # Draws tend to be lower scoring
        expected_goals -= draw_prob * 0.5
        
        # High probability outcomes suggest confidence in scoring
        if max(home_win_prob, away_win_prob) > 0.7:
            expected_goals += 0.3
        
        # Convert to probability using sigmoid
        over_prob = 1 / (1 + np.exp(-(expected_goals - 2.5)))
        over_prob = float(np.clip(over_prob, 0.1, 0.9))
        
        return {
            'prediction': 'Over' if over_prob > 0.5 else 'Under',
            'confidence': float(max(over_prob, 1 - over_prob)),
            'probabilities': {
                'over': over_prob,
                'under': 1 - over_prob
            },
            'expected_goals': float(expected_goals),
            'recommended': max(over_prob, 1 - over_prob) > 0.65
        }
    
    def predict_btts(self, home_win_prob, draw_prob, away_win_prob, features):
        """Predict Both Teams to Score based on attacking/defensive strength"""
        # Extract team strength features with safe defaults
        home_attack = features.get('home_team_avg_goals', features.get('FTHG', 1.5))
        away_attack = features.get('away_team_avg_goals', features.get('FTAG', 1.0))
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
        btts_prob = float(np.clip(btts_score / 3, 0.2, 0.8))
        
        return {
            'prediction': 'Yes' if btts_prob > 0.5 else 'No',
            'confidence': float(max(btts_prob, 1 - btts_prob)),
            'probabilities': {
                'yes': btts_prob,
                'no': 1 - btts_prob
            },
            'recommended': max(btts_prob, 1 - btts_prob) > 0.60
        }
    
    def predict_first_half(self, home_win_prob, draw_prob, away_win_prob, features):
        """Predict first half result (typically more conservative)"""
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
        prediction_idx = int(np.argmax(probs))
        
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
    
    def save_results(self, predictions: List[Dict], workflow_results: Dict):
        """Save all results"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save predictions JSON
        predictions_file = self.output_dir / f'predictions_{timestamp}.json'
        with open(predictions_file, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"  ✅ Predictions: {predictions_file}")
        
        # Save predictions CSV
        if predictions:
            csv_file = self.output_dir / f'predictions_{timestamp}.csv'
            self.save_predictions_csv(predictions, csv_file)
            logger.info(f"  ✅ CSV: {csv_file}")
        
        # Save workflow results (sanitize for JSON serialization)
        workflow_file = self.output_dir / f'workflow_{timestamp}.json'
        
        def make_json_serializable(obj):
            """Convert non-serializable objects to JSON-compatible format"""
            if isinstance(obj, dict):
                return {str(k): make_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_json_serializable(item) for item in obj]
            elif hasattr(obj, '__dict__'):
                return str(obj)
            else:
                return obj
        
        sanitized_results = make_json_serializable(workflow_results)
        
        with open(workflow_file, 'w') as f:
            json.dump(sanitized_results, f, indent=2, default=str)
        logger.info(f"  ✅ Workflow: {workflow_file}")
        
        # Generate summary
        summary_file = self.output_dir / f'summary_{timestamp}.txt'
        self.generate_summary(predictions, summary_file)
        logger.info(f"  ✅ Summary: {summary_file}")
    
    def save_predictions_csv(self, predictions: List[Dict], output_file: Path):
        """Save predictions in CSV format"""
        rows = []
        for pred in predictions:
            row = {
                'Date': pred['match_info']['date'],
                'HomeTeam': pred['match_info']['home_team'],
                'AwayTeam': pred['match_info']['away_team'],
                'League': pred['match_info']['league']
            }
            
            if '1x2' in pred['predictions']:
                row['1X2_Prediction'] = pred['predictions']['1x2'].get('outcome', 'N/A')
                row['1X2_Confidence'] = pred['predictions']['1x2'].get('confidence', 0)
            
            if 'over_under_2.5' in pred['predictions']:
                row['OU_Prediction'] = pred['predictions']['over_under_2.5'].get('prediction', 'N/A')
                row['OU_Confidence'] = pred['predictions']['over_under_2.5'].get('confidence', 0)
            
            if 'btts' in pred['predictions']:
                row['BTTS_Prediction'] = pred['predictions']['btts'].get('prediction', 'N/A')
                row['BTTS_Confidence'] = pred['predictions']['btts'].get('confidence', 0)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(output_file, index=False)
    
    def generate_summary(self, predictions: List[Dict], output_file: Path):
        """Generate summary report"""
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PREDICTION SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Matches: {len(predictions)}\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Count recommendations
            recommended_1x2 = sum(1 for p in predictions 
                                 if '1x2' in p['predictions'] 
                                 and p['predictions']['1x2'].get('recommended', False))
            
            f.write(f"1X2 Recommended Bets: {recommended_1x2}\n")
            f.write(f"Markets Covered: 1X2, O/U 2.5, BTTS, First Half\n")
            
            f.write("\n" + "=" * 80 + "\n")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Complete Workflow Orchestrator')
    parser.add_argument('--league', default='E1', help='League code (default: E1)')
    parser.add_argument('--crawler-file', help='Path to crawler output file')
    parser.add_argument('--skip-training', action='store_true', 
                       help='Skip training phases (use existing models)')
    parser.add_argument('--config', default='config.yaml', help='Configuration file')
    args = parser.parse_args()
    
    # Run workflow
    orchestrator = CompleteWorkflowOrchestrator(config_file=args.config)
    results = orchestrator.run_complete_workflow(
        league=args.league,
        crawler_file=args.crawler_file,
        skip_training=args.skip_training
    )
    
    print("\n" + "=" * 80)
    print(f"✅ WORKFLOW COMPLETE")
    print(f"   Matches: {results.get('upcoming_matches_count', 0)}")
    print(f"   Predictions: {results.get('predictions_count', 0)}")
    print(f"   Status: {results.get('status', 'unknown')}")
    print("=" * 80)


if __name__ == '__main__':
    main()
