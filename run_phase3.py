#!/usr/bin/env python3
"""
Run Phase 3: Backtesting & Strategy Development
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
print("PHASE 3: BACKTESTING & STRATEGY DEVELOPMENT")
print("=" * 80)

try:
    from phase3_integration import Phase3Integration
    
    # Initialize Phase 3
    phase3 = Phase3Integration(output_dir="./pipeline_output/phase3_output")
    
    # Paths to Phase 1 and Phase 2 outputs
    data_file = './pipeline_output/phase1_output/E1_features.csv'
    model_file = './pipeline_output/phase2_output/E1_ensemble.pkl'
    league_code = 'E1'
    
    logger.info(f"Loading data from: {data_file}")
    logger.info(f"Loading model from: {model_file}")
    logger.info(f"League: {league_code}")
    
    # Check if files exist
    if not Path(data_file).exists():
        logger.error(f"Data file not found: {data_file}")
        sys.exit(1)
    
    if not Path(model_file).exists():
        logger.error(f"Model file not found: {model_file}")
        sys.exit(1)
    
    # Run Phase 3 pipeline
    logger.info("\nStarting Phase 3 backtesting...")
    phase3_results = phase3.run_phase3_pipeline(
        data_file=data_file,
        league_code=league_code,
        model_file=model_file
    )
    
    logger.info("✅ Phase 3 completed successfully")
    
    # Display results
    print("\n" + "=" * 80)
    print("✅ PHASE 3 COMPLETE")
    print("=" * 80)
    
    if 'backtesting' in phase3_results:
        bt_results = phase3_results['backtesting']
        print(f"\n📊 Backtesting Results:")
        print(f"   • Total Bets: {bt_results.get('total_bets', 0)}")
        print(f"   • Win Rate: {bt_results.get('win_rate', 0):.2%}")
        print(f"   • ROI: {bt_results.get('roi', 0):.2%}")
        print(f"   • Profit: ${bt_results.get('profit', 0):.2f}")
    
    if 'betting_strategy' in phase3_results:
        strategy = phase3_results['betting_strategy']
        print(f"\n💰 Betting Strategy:")
        print(f"   • Strategy Type: {strategy.get('type', 'N/A')}")
        print(f"   • Confidence Threshold: {strategy.get('confidence_threshold', 0):.2%}")
        print(f"   • Max Stake: ${strategy.get('max_stake', 0):.2f}")
    
    if 'performance_metrics' in phase3_results:
        metrics = phase3_results['performance_metrics']
        print(f"\n📈 Performance Metrics:")
        print(f"   • Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
        print(f"   • Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
        print(f"   • Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    
    print(f"\n📁 Output directory: ./pipeline_output/phase3_output")
    print("=" * 80)
    
except Exception as e:
    logger.error(f"❌ Phase 3 failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("PHASE 3 EXECUTION COMPLETE")
print("=" * 80)
print(f"\nNext steps:")
print(f"  • Review backtesting results")
print(f"  • Analyze betting strategy performance")
print(f"  • Proceed to Phase 4 (Production Deployment)")
print("=" * 80)
