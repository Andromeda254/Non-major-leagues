#!/usr/bin/env python3
"""
Test Crawler Bridge with Sample Data
=====================================

This script tests the bridge functionality with sample match data
without requiring the actual crawler to run.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Create sample crawler output
def create_sample_crawler_output():
    """Create sample matches similar to crawler output"""
    
    # Sample matches for testing
    sample_matches = [
        {
            "id": "match_001",
            "teams": ["Manchester United", "Liverpool"],
            "time": (datetime.now() + timedelta(hours=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "odds": {
                "home": 2.5,
                "draw": 3.2,
                "away": 2.8
            },
            "league": "Premier League",
            "source": "test_crawler",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "match_002",
            "teams": ["Chelsea", "Arsenal"],
            "time": (datetime.now() + timedelta(hours=4)).strftime("%Y-%m-%d %H:%M:%S"),
            "odds": {
                "home": 2.1,
                "draw": 3.4,
                "away": 3.2
            },
            "league": "Premier League",
            "source": "test_crawler",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "match_003",
            "teams": ["Leeds United", "Norwich City"],
            "time": (datetime.now() + timedelta(hours=6)).strftime("%Y-%m-%d %H:%M:%S"),
            "odds": {
                "home": 1.8,
                "draw": 3.6,
                "away": 4.2
            },
            "league": "Championship",
            "source": "test_crawler",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "match_004",
            "teams": ["Sunderland", "Ipswich Town"],
            "time": (datetime.now() + timedelta(hours=8)).strftime("%Y-%m-%d %H:%M:%S"),
            "odds": {
                "home": 2.3,
                "draw": 3.1,
                "away": 3.0
            },
            "league": "Championship",
            "source": "test_crawler",
            "timestamp": datetime.now().isoformat()
        },
        {
            "id": "match_005",
            "teams": ["Portsmouth", "Bolton Wanderers"],
            "time": (datetime.now() + timedelta(hours=10)).strftime("%Y-%m-%d %H:%M:%S"),
            "odds": {
                "home": 2.0,
                "draw": 3.3,
                "away": 3.5
            },
            "league": "League One",
            "source": "test_crawler",
            "timestamp": datetime.now().isoformat()
        }
    ]
    
    return sample_matches


def main():
    """Test the bridge with sample data"""
    print("=" * 80)
    print("TESTING CRAWLER BRIDGE WITH SAMPLE DATA")
    print("=" * 80)
    print()
    
    # Create sample data
    print("📝 Creating sample match data...")
    sample_matches = create_sample_crawler_output()
    print(f"✅ Created {len(sample_matches)} sample matches")
    print()
    
    # Save to test file
    test_dir = Path("soccer-match-intelligence")
    test_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    test_file = test_dir / f"test_filtered_matches_{timestamp}.json"
    
    print(f"💾 Saving to: {test_file}")
    with open(test_file, 'w') as f:
        json.dump(sample_matches, f, indent=2)
    print("✅ Sample data saved")
    print()
    
    # Display sample matches
    print("📊 SAMPLE MATCHES:")
    print("-" * 80)
    for i, match in enumerate(sample_matches, 1):
        print(f"\n{i}. {match['teams'][0]} vs {match['teams'][1]}")
        print(f"   League: {match['league']}")
        print(f"   Time: {match['time']}")
        print(f"   Odds: {match['odds']['home']} / {match['odds']['draw']} / {match['odds']['away']}")
    print()
    print("-" * 80)
    print()
    
    # Test the bridge
    print("🔄 Testing bridge...")
    print()
    
    try:
        from crawler_to_pipeline_bridge import CrawlerPipelineBridge
        
        bridge = CrawlerPipelineBridge()
        success = bridge.run(crawler_output_file=test_file)
        
        if success:
            print()
            print("=" * 80)
            print("✅ BRIDGE TEST COMPLETED SUCCESSFULLY")
            print("=" * 80)
            print()
            print("📁 Check the following directories:")
            print("   - predictions/         (prediction outputs)")
            print("   - soccer-match-intelligence/  (crawler data)")
            print()
            print("💡 Next steps:")
            print("   1. Review predictions in predictions/ directory")
            print("   2. Train models: python run_pipeline.py --full --league E1")
            print("   3. Run actual crawler: node enhanced_soccer_match_crawler.js")
            print("   4. Run full workflow: ./run_crawler_pipeline.sh")
            print()
        else:
            print()
            print("=" * 80)
            print("⚠️ BRIDGE TEST COMPLETED WITH WARNINGS")
            print("=" * 80)
            print()
            print("This is expected if models are not trained yet.")
            print("The bridge can format data, but predictions require trained models.")
            print()
            
    except Exception as e:
        print(f"❌ Error testing bridge: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
