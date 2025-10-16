#!/usr/bin/env python3
"""
Test Phase 3 with forced module reload
"""

import sys
import importlib

# Remove cached modules
modules_to_reload = [
    'non_major_league_backtesting',
    'non_major_league_betting_strategy', 
    'non_major_league_performance_metrics',
    'phase3_integration'
]

for module_name in modules_to_reload:
    if module_name in sys.modules:
        del sys.modules[module_name]

# Now run Phase 3
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

print("=" * 80)
print("PHASE 3 TEST WITH FORCED MODULE RELOAD")
print("=" * 80)

try:
    from phase3_integration import Phase3Integration
    
    phase3 = Phase3Integration(output_dir="./pipeline_output/phase3_output")
    
    data_file = './pipeline_output/phase1_output/E1_features.csv'
    model_file = './pipeline_output/phase2_output/E1_ensemble.pkl'
    league_code = 'E1'
    
    logger.info(f"Testing Phase 3 with reloaded modules...")
    
    phase3_results = phase3.run_phase3_pipeline(
        data_file=data_file,
        league_code=league_code,
        model_file=model_file
    )
    
    print("\n" + "=" * 80)
    print("✅ PHASE 3 TEST COMPLETE")
    print("=" * 80)
    
    # Check for errors
    errors = []
    if not phase3_results.get('backtesting', {}).get('success', False):
        errors.append(f"Backtesting: {phase3_results.get('backtesting', {}).get('error', 'Unknown')}")
    
    if not phase3_results.get('betting_strategy', {}).get('success', False):
        errors.append(f"Betting Strategy: {phase3_results.get('betting_strategy', {}).get('error', 'Unknown')}")
    
    if not phase3_results.get('performance_metrics', {}).get('success', False):
        errors.append(f"Performance Metrics: {phase3_results.get('performance_metrics', {}).get('error', 'Unknown')}")
    
    if errors:
        print(f"\n⚠️ Errors Found ({len(errors)}):")
        for error in errors:
            print(f"  • {error}")
    else:
        print(f"\n✅ No Errors - All Components Working!")
    
    # Show what worked
    working = []
    if phase3_results.get('risk_management', {}).get('success', False):
        working.append("Risk Management")
    if phase3_results.get('live_testing', {}).get('success', False):
        working.append("Live Testing")
    
    if working:
        print(f"\n✅ Working Components ({len(working)}):")
        for comp in working:
            print(f"  • {comp}")
    
    print("=" * 80)
    
except Exception as e:
    logger.error(f"Test failed: {e}")
    import traceback
    traceback.print_exc()
