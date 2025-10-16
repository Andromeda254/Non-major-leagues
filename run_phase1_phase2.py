#!/usr/bin/env python3
"""
Run Phase 1 and Phase 2 of the ML Pipeline
"""

import sys
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("RUNNING PHASE 1 & PHASE 2 PIPELINE")
print("=" * 80)

# Phase 1: Data Collection & Preprocessing
print("\n" + "=" * 80)
print("PHASE 1: DATA COLLECTION & PREPROCESSING")
print("=" * 80)

try:
    from phase1_integration import Phase1Integration
    
    # Initialize Phase 1
    phase1 = Phase1Integration(output_dir="./pipeline_output/phase1_output")
    
    # Run Phase 1 pipeline
    logger.info("Running Phase 1 pipeline for Championship (E1)...")
    phase1_results = phase1.run_phase1_pipeline(
        league_code='E1',
        seasons=['2324', '2223'],
        collect_live_odds=False  # Skip live odds for now
    )
    
    logger.info(f"✅ Phase 1 completed successfully")
    logger.info(f"   Data collected: {phase1_results.get('data_collection', {}).get('total_matches', 0)} matches")
    
    # Save Phase 1 results
    phase1_output_dir = Path("./pipeline_output/phase1_output")
    phase1_output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n✅ PHASE 1 COMPLETE")
    print(f"   Output directory: {phase1_output_dir}")
    
except Exception as e:
    logger.error(f"❌ Phase 1 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Phase 2: Model Training
print("\n" + "=" * 80)
print("PHASE 2: MODEL TRAINING & VALIDATION")
print("=" * 80)

try:
    from phase2_integration import Phase2Integration
    
    # Check if Phase 1 output exists
    feature_data_path = phase1_output_dir / "E1_features.csv"
    
    if not feature_data_path.exists():
        logger.warning(f"Feature data not found at {feature_data_path}")
        logger.info("Trying alternative filenames...")
        
        # Try other possible names
        alternatives = [
            "processed_features.parquet",
            "processed_features.csv",
            "features.csv"
        ]
        
        for alt in alternatives:
            alt_path = phase1_output_dir / alt
            if alt_path.exists():
                feature_data_path = alt_path
                logger.info(f"Found features at {feature_data_path}")
                break
        else:
            logger.error("No processed features found. Phase 1 may not have completed properly.")
            sys.exit(1)
    
    # Initialize Phase 2
    phase2 = Phase2Integration()
    
    # Run Phase 2 pipeline
    logger.info("Running Phase 2 pipeline...")
    phase2_results = phase2.run_phase2_pipeline(
        feature_data_path=str(feature_data_path),
        output_dir="./pipeline_output"
    )
    
    logger.info(f"✅ Phase 2 completed successfully")
    
    # Display results
    print(f"\n✅ PHASE 2 COMPLETE")
    print(f"   Models trained: {len(phase2_results.get('models', {}))}")
    print(f"   Best model: {phase2_results.get('best_model', 'N/A')}")
    
    if 'validation_metrics' in phase2_results:
        metrics = phase2_results['validation_metrics']
        print(f"\n📊 Model Performance:")
        print(f"   Accuracy: {metrics.get('accuracy', 0):.4f}")
        print(f"   Precision: {metrics.get('precision', 0):.4f}")
        print(f"   Recall: {metrics.get('recall', 0):.4f}")
        print(f"   F1 Score: {metrics.get('f1_score', 0):.4f}")
    
except Exception as e:
    logger.error(f"❌ Phase 2 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ PIPELINE EXECUTION COMPLETE")
print("=" * 80)
print(f"\nOutput directories:")
print(f"  • Phase 1: ./pipeline_output/phase1_output")
print(f"  • Phase 2: ./pipeline_output/phase2_output")
print(f"\nNext steps:")
print(f"  • Review model performance metrics")
print(f"  • Run Phase 3 for backtesting")
print(f"  • Deploy to production (Phase 4)")
print("=" * 80)
