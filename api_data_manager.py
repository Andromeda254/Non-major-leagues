"""
API Data Manager for Soccer Match Intelligence Pipeline
======================================================

This module manages API keys, data sources, and data collection for the ML pipeline.
It provides a centralized way to handle multiple data sources with fallback strategies,
rate limiting, and data quality monitoring.

Author: AI Assistant
Date: 2025-10-11
"""

import os
import yaml
import requests
import time
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import hashlib
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DataSource:
    """Data source configuration"""
    name: str
    api_key: str
    base_url: str
    rate_limit: int
    enabled: bool
    priority: int
    data_types: List[str]
    app_key: Optional[str] = None
    bearer_token: Optional[str] = None

@dataclass
class APIResponse:
    """API response wrapper"""
    data: Any
    source: str
    timestamp: datetime
    success: bool
    error: Optional[str] = None
    response_time: Optional[float] = None

class APIDataManager:
    """Manages API data sources and collection"""
    
    def __init__(self, config_path: str = "config.yaml", env_file: str = "api_keys_template.env"):
        """Initialize the API data manager"""
        self.config_path = config_path
        self.env_file = env_file
        self.config = self._load_config()
        self.data_sources = self._initialize_data_sources()
        self.cache = {}
        self.rate_limiters = {}
        self.session = requests.Session()
        
        # Set up session headers
        self.session.headers.update({
            'User-Agent': 'Soccer-Match-Intelligence/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })
        
        logger.info(f"Initialized APIDataManager with {len(self.data_sources)} data sources")
    
    def _load_config(self) -> Dict:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
    
    def _load_env_vars(self) -> Dict[str, str]:
        """Load environment variables from api_keys_template.env file"""
        env_vars = {}
        if os.path.exists(self.env_file):
            try:
                with open(self.env_file, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            env_vars[key] = value
                logger.info(f"Loaded {len(env_vars)} environment variables")
            except Exception as e:
                logger.error(f"Failed to load {self.env_file} file: {e}")
        return env_vars
    
    def _initialize_data_sources(self) -> Dict[str, DataSource]:
        """Initialize data sources from configuration"""
        data_sources = {}
        env_vars = self._load_env_vars()
        
        if 'data_sources' not in self.config:
            logger.warning("No data_sources found in config")
            return data_sources
        
        for source_name, source_config in self.config['data_sources'].items():
            if source_name in ['data_validation', 'fallback_strategy', 'caching', 'rate_limiting', 'quality_monitoring']:
                continue
                
            # Get API key from environment variables
            api_key = None
            if source_name == 'football_data':
                api_key = env_vars.get('FOOTBALL_DATA_API_KEY')
            elif source_name == 'odds_api':
                api_key = env_vars.get('ODDS_API_KEY')
            elif source_name == 'api_football':
                api_key = env_vars.get('API_FOOTBALL_KEY')
            elif source_name == 'rapidapi_football':
                api_key = env_vars.get('RAPIDAPI_KEY')
            elif source_name == 'sportsdata_io':
                api_key = env_vars.get('SPORTSDATA_IO_KEY')
            elif source_name == 'betfair_exchange':
                api_key = env_vars.get('BETFAIR_API_KEY')
                app_key = env_vars.get('BETFAIR_APP_KEY')
            elif source_name == 'openweathermap':
                api_key = env_vars.get('OPENWEATHERMAP_KEY')
            elif source_name == 'newsapi':
                api_key = env_vars.get('NEWSAPI_KEY')
            elif source_name == 'twitter_api':
                api_key = env_vars.get('TWITTER_BEARER_TOKEN')
            elif source_name == 'pinnacle':
                api_key = env_vars.get('PINNACLE_API_KEY')
            elif source_name == 'football_api_com':
                api_key = env_vars.get('FOOTBALL_API_COM_KEY')
            elif source_name == 'livescore':
                api_key = env_vars.get('LIVESCORE_KEY')
            elif source_name == 'football_api_net':
                api_key = env_vars.get('FOOTBALL_API_NET_KEY')
            elif source_name == 'fifa_data':
                api_key = env_vars.get('FIFA_DATA_KEY')
            elif source_name == 'smarkets':
                api_key = env_vars.get('SMARKETS_KEY')
            
            if api_key and api_key != f"your_{source_name}_key_here":
                data_source = DataSource(
                    name=source_name,
                    api_key=api_key,
                    base_url=source_config.get('base_url', ''),
                    rate_limit=source_config.get('rate_limit', 100),
                    enabled=source_config.get('enabled', False),
                    priority=source_config.get('priority', 999),
                    data_types=source_config.get('data_types', []),
                    app_key=app_key if source_name == 'betfair_exchange' else None,
                    bearer_token=api_key if source_name == 'twitter_api' else None
                )
                data_sources[source_name] = data_source
                logger.info(f"Initialized data source: {source_name}")
            else:
                logger.warning(f"Skipping {source_name} - no valid API key found")
        
        return data_sources
    
    def _check_rate_limit(self, source_name: str) -> bool:
        """Check if rate limit allows request"""
        if source_name not in self.rate_limiters:
            self.rate_limiters[source_name] = {
                'requests': [],
                'limit': self.data_sources[source_name].rate_limit
            }
        
        limiter = self.rate_limiters[source_name]
        now = time.time()
        
        # Remove old requests (older than 1 minute)
        limiter['requests'] = [req_time for req_time in limiter['requests'] if now - req_time < 60]
        
        # Check if under limit
        if len(limiter['requests']) < limiter['limit']:
            limiter['requests'].append(now)
            return True
        
        return False
    
    def _get_cache_key(self, source: str, endpoint: str, params: Dict) -> str:
        """Generate cache key for request"""
        key_data = f"{source}:{endpoint}:{json.dumps(params, sort_keys=True)}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid"""
        if cache_key not in self.cache:
            return False
        
        cache_duration = self.config.get('data_sources', {}).get('caching', {}).get('cache_duration', 3600)
        cache_time = self.cache[cache_key]['timestamp']
        
        return (datetime.now() - cache_time).total_seconds() < cache_duration
    
    def _make_request(self, source: DataSource, endpoint: str, params: Dict = None) -> APIResponse:
        """Make API request with error handling and rate limiting"""
        if not self._check_rate_limit(source.name):
            return APIResponse(
                data=None,
                source=source.name,
                timestamp=datetime.now(),
                success=False,
                error="Rate limit exceeded"
            )
        
        # Check cache first
        cache_key = self._get_cache_key(source.name, endpoint, params or {})
        if self._is_cache_valid(cache_key):
            logger.info(f"Using cached data for {source.name}:{endpoint}")
            return self.cache[cache_key]['response']
        
        # Prepare request
        url = f"{source.base_url}/{endpoint.lstrip('/')}"
        headers = {}
        
        # Set authentication headers
        if source.name == 'twitter_api':
            headers['Authorization'] = f"Bearer {source.bearer_token}"
        elif source.name == 'rapidapi_football':
            headers['X-RapidAPI-Key'] = source.api_key
            headers['X-RapidAPI-Host'] = 'api-football-v1.p.rapidapi.com'
        elif source.name == 'betfair_exchange':
            headers['X-Application'] = source.app_key
            headers['X-Authentication'] = source.api_key
        else:
            headers['X-Auth-Token'] = source.api_key
        
        # Make request
        start_time = time.time()
        try:
            response = self.session.get(url, headers=headers, params=params, timeout=30)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                data = response.json()
                api_response = APIResponse(
                    data=data,
                    source=source.name,
                    timestamp=datetime.now(),
                    success=True,
                    response_time=response_time
                )
                
                # Cache successful response
                self.cache[cache_key] = {
                    'response': api_response,
                    'timestamp': datetime.now()
                }
                
                logger.info(f"Successfully fetched data from {source.name}:{endpoint}")
                return api_response
            else:
                logger.error(f"API request failed: {source.name}:{endpoint} - Status: {response.status_code}")
                return APIResponse(
                    data=None,
                    source=source.name,
                    timestamp=datetime.now(),
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text}",
                    response_time=response_time
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"API request error: {source.name}:{endpoint} - {e}")
            return APIResponse(
                data=None,
                source=source.name,
                timestamp=datetime.now(),
                success=False,
                error=str(e),
                response_time=response_time
            )
    
    def get_fixtures(self, league: str = None, date: str = None) -> List[APIResponse]:
        """Get fixtures from available sources"""
        responses = []
        sources = sorted(self.data_sources.values(), key=lambda x: x.priority)
        
        for source in sources:
            if not source.enabled or 'fixtures' not in source.data_types:
                continue
            
            params = {}
            if league:
                params['league'] = league
            if date:
                params['date'] = date
            
            # Source-specific endpoint mapping
            if source.name == 'football_data':
                endpoint = 'matches'
            elif source.name == 'api_football':
                endpoint = 'fixtures'
            elif source.name == 'rapidapi_football':
                endpoint = 'fixtures'
            else:
                endpoint = 'fixtures'
            
            response = self._make_request(source, endpoint, params)
            if response.success:
                responses.append(response)
                break  # Use first successful response
        
        return responses
    
    def get_odds(self, match_id: str = None, league: str = None) -> List[APIResponse]:
        """Get odds from available sources"""
        responses = []
        sources = sorted(self.data_sources.values(), key=lambda x: x.priority)
        
        for source in sources:
            if not source.enabled or 'odds' not in source.data_types:
                continue
            
            params = {}
            if match_id:
                params['match_id'] = match_id
            if league:
                params['league'] = league
            
            # Source-specific endpoint mapping
            if source.name == 'odds_api':
                endpoint = 'odds'
            elif source.name == 'betfair_exchange':
                endpoint = 'odds'
            elif source.name == 'pinnacle':
                endpoint = 'odds'
            else:
                endpoint = 'odds'
            
            response = self._make_request(source, endpoint, params)
            if response.success:
                responses.append(response)
                break  # Use first successful response
        
        return responses
    
    def get_standings(self, league: str) -> List[APIResponse]:
        """Get league standings from available sources"""
        responses = []
        sources = sorted(self.data_sources.values(), key=lambda x: x.priority)
        
        for source in sources:
            if not source.enabled or 'standings' not in source.data_types:
                continue
            
            params = {'league': league}
            
            # Source-specific endpoint mapping
            if source.name == 'football_data':
                endpoint = 'standings'
            elif source.name == 'api_football':
                endpoint = 'standings'
            elif source.name == 'rapidapi_football':
                endpoint = 'standings'
            else:
                endpoint = 'standings'
            
            response = self._make_request(source, endpoint, params)
            if response.success:
                responses.append(response)
                break  # Use first successful response
        
        return responses
    
    def get_team_statistics(self, team_id: str) -> List[APIResponse]:
        """Get team statistics from available sources"""
        responses = []
        sources = sorted(self.data_sources.values(), key=lambda x: x.priority)
        
        for source in sources:
            if not source.enabled or 'statistics' not in source.data_types:
                continue
            
            params = {'team_id': team_id}
            
            # Source-specific endpoint mapping
            if source.name == 'api_football':
                endpoint = 'teams/statistics'
            elif source.name == 'rapidapi_football':
                endpoint = 'teams/statistics'
            else:
                endpoint = 'teams/statistics'
            
            response = self._make_request(source, endpoint, params)
            if response.success:
                responses.append(response)
                break  # Use first successful response
        
        return responses
    
    def get_weather_data(self, city: str, date: str = None) -> List[APIResponse]:
        """Get weather data for match location"""
        responses = []
        
        if 'openweathermap' in self.data_sources:
            source = self.data_sources['openweathermap']
            if source.enabled:
                params = {'q': city, 'appid': source.api_key}
                if date:
                    params['dt'] = date
                
                response = self._make_request(source, 'weather', params)
                if response.success:
                    responses.append(response)
        
        return responses
    
    def get_news_data(self, query: str, team: str = None) -> List[APIResponse]:
        """Get news data related to teams or matches"""
        responses = []
        
        if 'newsapi' in self.data_sources:
            source = self.data_sources['newsapi']
            if source.enabled:
                params = {
                    'q': f"{query} {team}" if team else query,
                    'apiKey': source.api_key,
                    'language': 'en',
                    'sortBy': 'publishedAt'
                }
                
                response = self._make_request(source, 'everything', params)
                if response.success:
                    responses.append(response)
        
        return responses
    
    def validate_data_quality(self, responses: List[APIResponse]) -> Dict[str, Any]:
        """Validate data quality across multiple sources"""
        validation_results = {
            'total_sources': len(responses),
            'successful_sources': len([r for r in responses if r.success]),
            'average_response_time': 0,
            'data_consistency': True,
            'issues': []
        }
        
        if not responses:
            validation_results['issues'].append('No responses received')
            return validation_results
        
        # Calculate average response time
        successful_responses = [r for r in responses if r.success and r.response_time]
        if successful_responses:
            validation_results['average_response_time'] = sum(r.response_time for r in successful_responses) / len(successful_responses)
        
        # Check data consistency
        if len(successful_responses) > 1:
            # Compare data across sources
            first_data = successful_responses[0].data
            for response in successful_responses[1:]:
                if not self._compare_data(first_data, response.data):
                    validation_results['data_consistency'] = False
                    validation_results['issues'].append(f'Data inconsistency between {successful_responses[0].source} and {response.source}')
        
        return validation_results
    
    def _compare_data(self, data1: Any, data2: Any) -> bool:
        """Compare data from different sources"""
        # Simple comparison - can be enhanced based on specific data structures
        try:
            return json.dumps(data1, sort_keys=True) == json.dumps(data2, sort_keys=True)
        except:
            return False
    
    def get_available_sources(self) -> List[str]:
        """Get list of available data sources"""
        return [name for name, source in self.data_sources.items() if source.enabled]
    
    def get_source_info(self, source_name: str) -> Optional[Dict]:
        """Get information about a specific data source"""
        if source_name in self.data_sources:
            source = self.data_sources[source_name]
            return {
                'name': source.name,
                'enabled': source.enabled,
                'priority': source.priority,
                'data_types': source.data_types,
                'rate_limit': source.rate_limit,
                'base_url': source.base_url
            }
        return None
    
    def enable_source(self, source_name: str) -> bool:
        """Enable a data source"""
        if source_name in self.data_sources:
            self.data_sources[source_name].enabled = True
            logger.info(f"Enabled data source: {source_name}")
            return True
        return False
    
    def disable_source(self, source_name: str) -> bool:
        """Disable a data source"""
        if source_name in self.data_sources:
            self.data_sources[source_name].enabled = False
            logger.info(f"Disabled data source: {source_name}")
            return True
        return False
    
    def clear_cache(self):
        """Clear all cached data"""
        self.cache.clear()
        logger.info("Cleared API cache")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        return {
            'total_entries': len(self.cache),
            'cache_size_mb': sum(len(str(v)) for v in self.cache.values()) / (1024 * 1024),
            'oldest_entry': min(entry['timestamp'] for entry in self.cache.values()) if self.cache else None,
            'newest_entry': max(entry['timestamp'] for entry in self.cache.values()) if self.cache else None
        }

def main():
    """Test the API data manager"""
    # Initialize the manager
    manager = APIDataManager()
    
    # Print available sources
    print("Available data sources:")
    for source_name in manager.get_available_sources():
        info = manager.get_source_info(source_name)
        print(f"  - {source_name}: {info['data_types']}")
    
    # Test fixtures retrieval
    print("\nTesting fixtures retrieval...")
    fixtures = manager.get_fixtures(league="E1")
    for response in fixtures:
        print(f"  {response.source}: {response.success}")
        if response.success:
            print(f"    Response time: {response.response_time:.2f}s")
    
    # Test odds retrieval
    print("\nTesting odds retrieval...")
    odds = manager.get_odds(league="E1")
    for response in odds:
        print(f"  {response.source}: {response.success}")
        if response.success:
            print(f"    Response time: {response.response_time:.2f}s")
    
    # Print cache stats
    print(f"\nCache stats: {manager.get_cache_stats()}")

if __name__ == "__main__":
    main()



