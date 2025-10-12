"""
API Integration Test Script
===========================

This script tests the API data manager with various data sources to ensure
proper integration and data collection for the ML pipeline.

Author: AI Assistant
Date: 2025-10-11
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from api_data_manager import APIDataManager

def test_api_manager_initialization():
    """Test API manager initialization"""
    print("=" * 60)
    print("Testing API Manager Initialization")
    print("=" * 60)
    
    try:
        manager = APIDataManager(env_file="api_keys_template.env")
        print("✅ API Manager initialized successfully")
        
        # Check available sources
        available_sources = manager.get_available_sources()
        print(f"📊 Available data sources: {len(available_sources)}")
        
        for source_name in available_sources:
            info = manager.get_source_info(source_name)
            print(f"  - {source_name}: Priority {info['priority']}, Types: {info['data_types']}")
        
        return manager
    except Exception as e:
        print(f"❌ Failed to initialize API manager: {e}")
        return None

def test_fixtures_retrieval(manager):
    """Test fixtures retrieval from various sources"""
    print("\n" + "=" * 60)
    print("Testing Fixtures Retrieval")
    print("=" * 60)
    
    if not manager:
        print("❌ No manager available for testing")
        return
    
    # Test different leagues
    test_leagues = ["E1", "E2", "E3", "PL", "BL1"]
    
    for league in test_leagues:
        print(f"\n🔍 Testing fixtures for league: {league}")
        
        try:
            fixtures = manager.get_fixtures(league=league)
            
            if fixtures:
                for response in fixtures:
                    print(f"  📡 Source: {response.source}")
                    print(f"     Success: {response.success}")
                    if response.success:
                        print(f"     Response time: {response.response_time:.2f}s")
                        print(f"     Data keys: {list(response.data.keys()) if isinstance(response.data, dict) else 'N/A'}")
                    else:
                        print(f"     Error: {response.error}")
            else:
                print("  ⚠️ No fixtures data received")
                
        except Exception as e:
            print(f"  ❌ Error testing fixtures for {league}: {e}")

def test_odds_retrieval(manager):
    """Test odds retrieval from various sources"""
    print("\n" + "=" * 60)
    print("Testing Odds Retrieval")
    print("=" * 60)
    
    if not manager:
        print("❌ No manager available for testing")
        return
    
    # Test different leagues
    test_leagues = ["E1", "E2", "E3"]
    
    for league in test_leagues:
        print(f"\n🎯 Testing odds for league: {league}")
        
        try:
            odds = manager.get_odds(league=league)
            
            if odds:
                for response in odds:
                    print(f"  📡 Source: {response.source}")
                    print(f"     Success: {response.success}")
                    if response.success:
                        print(f"     Response time: {response.response_time:.2f}s")
                        print(f"     Data keys: {list(response.data.keys()) if isinstance(response.data, dict) else 'N/A'}")
                    else:
                        print(f"     Error: {response.error}")
            else:
                print("  ⚠️ No odds data received")
                
        except Exception as e:
            print(f"  ❌ Error testing odds for {league}: {e}")

def test_standings_retrieval(manager):
    """Test standings retrieval from various sources"""
    print("\n" + "=" * 60)
    print("Testing Standings Retrieval")
    print("=" * 60)
    
    if not manager:
        print("❌ No manager available for testing")
        return
    
    # Test different leagues
    test_leagues = ["E1", "E2", "E3"]
    
    for league in test_leagues:
        print(f"\n🏆 Testing standings for league: {league}")
        
        try:
            standings = manager.get_standings(league=league)
            
            if standings:
                for response in standings:
                    print(f"  📡 Source: {response.source}")
                    print(f"     Success: {response.success}")
                    if response.success:
                        print(f"     Response time: {response.response_time:.2f}s")
                        print(f"     Data keys: {list(response.data.keys()) if isinstance(response.data, dict) else 'N/A'}")
                    else:
                        print(f"     Error: {response.error}")
            else:
                print("  ⚠️ No standings data received")
                
        except Exception as e:
            print(f"  ❌ Error testing standings for {league}: {e}")

def test_team_statistics(manager):
    """Test team statistics retrieval"""
    print("\n" + "=" * 60)
    print("Testing Team Statistics Retrieval")
    print("=" * 60)
    
    if not manager:
        print("❌ No manager available for testing")
        return
    
    # Test with sample team IDs
    test_team_ids = ["1", "2", "3", "4", "5"]
    
    for team_id in test_team_ids:
        print(f"\n👥 Testing statistics for team ID: {team_id}")
        
        try:
            stats = manager.get_team_statistics(team_id=team_id)
            
            if stats:
                for response in stats:
                    print(f"  📡 Source: {response.source}")
                    print(f"     Success: {response.success}")
                    if response.success:
                        print(f"     Response time: {response.response_time:.2f}s")
                        print(f"     Data keys: {list(response.data.keys()) if isinstance(response.data, dict) else 'N/A'}")
                    else:
                        print(f"     Error: {response.error}")
            else:
                print("  ⚠️ No team statistics received")
                
        except Exception as e:
            print(f"  ❌ Error testing team statistics for {team_id}: {e}")

def test_weather_data(manager):
    """Test weather data retrieval"""
    print("\n" + "=" * 60)
    print("Testing Weather Data Retrieval")
    print("=" * 60)
    
    if not manager:
        print("❌ No manager available for testing")
        return
    
    # Test with sample cities
    test_cities = ["London", "Manchester", "Birmingham", "Leeds", "Liverpool"]
    
    for city in test_cities:
        print(f"\n🌤️ Testing weather for city: {city}")
        
        try:
            weather = manager.get_weather_data(city=city)
            
            if weather:
                for response in weather:
                    print(f"  📡 Source: {response.source}")
                    print(f"     Success: {response.success}")
                    if response.success:
                        print(f"     Response time: {response.response_time:.2f}s")
                        print(f"     Data keys: {list(response.data.keys()) if isinstance(response.data, dict) else 'N/A'}")
                    else:
                        print(f"     Error: {response.error}")
            else:
                print("  ⚠️ No weather data received")
                
        except Exception as e:
            print(f"  ❌ Error testing weather for {city}: {e}")

def test_news_data(manager):
    """Test news data retrieval"""
    print("\n" + "=" * 60)
    print("Testing News Data Retrieval")
    print("=" * 60)
    
    if not manager:
        print("❌ No manager available for testing")
        return
    
    # Test with sample queries
    test_queries = ["football", "soccer", "premier league", "championship"]
    
    for query in test_queries:
        print(f"\n📰 Testing news for query: {query}")
        
        try:
            news = manager.get_news_data(query=query)
            
            if news:
                for response in news:
                    print(f"  📡 Source: {response.source}")
                    print(f"     Success: {response.success}")
                    if response.success:
                        print(f"     Response time: {response.response_time:.2f}s")
                        print(f"     Data keys: {list(response.data.keys()) if isinstance(response.data, dict) else 'N/A'}")
                    else:
                        print(f"     Error: {response.error}")
            else:
                print("  ⚠️ No news data received")
                
        except Exception as e:
            print(f"  ❌ Error testing news for {query}: {e}")

def test_data_quality_validation(manager):
    """Test data quality validation"""
    print("\n" + "=" * 60)
    print("Testing Data Quality Validation")
    print("=" * 60)
    
    if not manager:
        print("❌ No manager available for testing")
        return
    
    try:
        # Get multiple responses for comparison
        responses = []
        
        # Test fixtures from multiple sources
        fixtures1 = manager.get_fixtures(league="E1")
        responses.extend(fixtures1)
        
        fixtures2 = manager.get_fixtures(league="E2")
        responses.extend(fixtures2)
        
        # Validate data quality
        validation_results = manager.validate_data_quality(responses)
        
        print("📊 Data Quality Validation Results:")
        print(f"  Total sources: {validation_results['total_sources']}")
        print(f"  Successful sources: {validation_results['successful_sources']}")
        print(f"  Average response time: {validation_results['average_response_time']:.2f}s")
        print(f"  Data consistency: {validation_results['data_consistency']}")
        
        if validation_results['issues']:
            print("  Issues found:")
            for issue in validation_results['issues']:
                print(f"    - {issue}")
        else:
            print("  ✅ No issues found")
            
    except Exception as e:
        print(f"❌ Error testing data quality validation: {e}")

def test_cache_functionality(manager):
    """Test cache functionality"""
    print("\n" + "=" * 60)
    print("Testing Cache Functionality")
    print("=" * 60)
    
    if not manager:
        print("❌ No manager available for testing")
        return
    
    try:
        # Get cache stats before requests
        print("📊 Cache stats before requests:")
        stats_before = manager.get_cache_stats()
        print(f"  Total entries: {stats_before['total_entries']}")
        print(f"  Cache size: {stats_before['cache_size_mb']:.2f} MB")
        
        # Make some requests to populate cache
        print("\n🔄 Making requests to populate cache...")
        manager.get_fixtures(league="E1")
        manager.get_odds(league="E1")
        manager.get_standings(league="E1")
        
        # Get cache stats after requests
        print("\n📊 Cache stats after requests:")
        stats_after = manager.get_cache_stats()
        print(f"  Total entries: {stats_after['total_entries']}")
        print(f"  Cache size: {stats_after['cache_size_mb']:.2f} MB")
        
        # Test cache clearing
        print("\n🧹 Clearing cache...")
        manager.clear_cache()
        
        # Get cache stats after clearing
        print("📊 Cache stats after clearing:")
        stats_cleared = manager.get_cache_stats()
        print(f"  Total entries: {stats_cleared['total_entries']}")
        print(f"  Cache size: {stats_cleared['cache_size_mb']:.2f} MB")
        
    except Exception as e:
        print(f"❌ Error testing cache functionality: {e}")

def test_source_management(manager):
    """Test source management functionality"""
    print("\n" + "=" * 60)
    print("Testing Source Management")
    print("=" * 60)
    
    if not manager:
        print("❌ No manager available for testing")
        return
    
    try:
        # Get available sources
        available_sources = manager.get_available_sources()
        print(f"📊 Available sources: {available_sources}")
        
        if available_sources:
            # Test disabling a source
            test_source = available_sources[0]
            print(f"\n🔧 Testing source management with: {test_source}")
            
            # Disable source
            success = manager.disable_source(test_source)
            print(f"  Disable {test_source}: {success}")
            
            # Check if disabled
            info = manager.get_source_info(test_source)
            print(f"  Source enabled: {info['enabled']}")
            
            # Re-enable source
            success = manager.enable_source(test_source)
            print(f"  Enable {test_source}: {success}")
            
            # Check if enabled
            info = manager.get_source_info(test_source)
            print(f"  Source enabled: {info['enabled']}")
        
    except Exception as e:
        print(f"❌ Error testing source management: {e}")

def save_test_results(manager, output_file="api_test_results.json"):
    """Save test results to file"""
    print("\n" + "=" * 60)
    print("Saving Test Results")
    print("=" * 60)
    
    if not manager:
        print("❌ No manager available for saving results")
        return
    
    try:
        results = {
            "timestamp": datetime.now().isoformat(),
            "available_sources": manager.get_available_sources(),
            "source_info": {},
            "cache_stats": manager.get_cache_stats()
        }
        
        # Get info for each source
        for source_name in manager.get_available_sources():
            results["source_info"][source_name] = manager.get_source_info(source_name)
        
        # Save to file
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✅ Test results saved to: {output_file}")
        
    except Exception as e:
        print(f"❌ Error saving test results: {e}")

def main():
    """Main test function"""
    print("🚀 Starting API Integration Tests")
    print(f"⏰ Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Initialize manager
    manager = test_api_manager_initialization()
    
    if manager:
        # Run all tests
        test_fixtures_retrieval(manager)
        test_odds_retrieval(manager)
        test_standings_retrieval(manager)
        test_team_statistics(manager)
        test_weather_data(manager)
        test_news_data(manager)
        test_data_quality_validation(manager)
        test_cache_functionality(manager)
        test_source_management(manager)
        
        # Save results
        save_test_results(manager)
    
    print(f"\n🏁 Tests completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()



