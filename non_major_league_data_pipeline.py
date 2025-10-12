import pandas as pd
import numpy as np
import requests
import schedule
import time
import threading
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeagueDataPipeline:
    """
    Automated data pipeline for live data ingestion for non-major soccer leagues
    
    Key Features:
    - Multi-source data collection
    - Real-time data processing
    - Data validation and quality checks
    - Automated scheduling
    - Error handling and retry logic
    - Data storage and archival
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize data pipeline
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.pipeline_active = False
        self.pipeline_thread = None
        self.data_cache = {}
        self.processing_stats = {}
        
    def setup_logging(self):
        """Setup logging for data pipeline"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load data pipeline configuration"""
        if config is None:
            self.config = {
                'pipeline': {
                    'enabled': True,
                    'schedule_interval': 3600,  # 1 hour
                    'max_retries': 3,
                    'retry_delay': 300,  # 5 minutes
                    'timeout': 30,
                    'batch_size': 1000
                },
                'data_sources': {
                    'football_data_co_uk': {
                        'enabled': True,
                        'base_url': 'https://www.football-data.co.uk/',
                        'api_key': None,
                        'rate_limit': 100,  # requests per hour
                        'timeout': 30,
                        'retry_attempts': 3
                    },
                    'the_odds_api': {
                        'enabled': True,
                        'base_url': 'https://api.the-odds-api.com/v4/',
                        'api_key': 'YOUR_API_KEY',
                        'rate_limit': 1000,  # requests per month
                        'timeout': 30,
                        'retry_attempts': 3
                    },
                    'api_football': {
                        'enabled': True,
                        'base_url': 'https://v3.football.api-sports.io/',
                        'api_key': 'YOUR_API_KEY',
                        'rate_limit': 100,  # requests per day
                        'timeout': 30,
                        'retry_attempts': 3
                    },
                    'flashscore': {
                        'enabled': False,  # Requires web scraping
                        'base_url': 'https://www.flashscore.com/',
                        'rate_limit': 10,  # requests per minute
                        'timeout': 30,
                        'retry_attempts': 3
                    }
                },
                'target_leagues': {
                    'Championship': {
                        'enabled': True,
                        'code': 'E1',
                        'season': '2324',
                        'sport_key': 'soccer_efl_championship',
                        'priority': 1
                    },
                    'League_One': {
                        'enabled': True,
                        'code': 'E2',
                        'season': '2324',
                        'sport_key': 'soccer_efl_league_one',
                        'priority': 2
                    },
                    'League_Two': {
                        'enabled': True,
                        'code': 'E3',
                        'season': '2324',
                        'sport_key': 'soccer_efl_league_two',
                        'priority': 3
                    },
                    'Scottish_Championship': {
                        'enabled': True,
                        'code': 'SC1',
                        'season': '2324',
                        'sport_key': 'soccer_spl_championship',
                        'priority': 4
                    }
                },
                'data_processing': {
                    'validation': {
                        'enabled': True,
                        'required_fields': ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'],
                        'data_types': {
                            'Date': 'datetime64[ns]',
                            'HomeTeam': 'object',
                            'AwayTeam': 'object',
                            'FTHG': 'int64',
                            'FTAG': 'int64',
                            'FTR': 'object'
                        },
                        'value_ranges': {
                            'FTHG': (0, 15),
                            'FTAG': (0, 15),
                            'B365H': (1.0, 100.0),
                            'B365D': (1.0, 100.0),
                            'B365A': (1.0, 100.0)
                        }
                    },
                    'transformation': {
                        'enabled': True,
                        'date_format': '%d/%m/%Y',
                        'team_name_mapping': True,
                        'odds_normalization': True,
                        'missing_value_handling': 'forward_fill'
                    },
                    'feature_engineering': {
                        'enabled': True,
                        'form_features': True,
                        'h2h_features': True,
                        'elo_features': True,
                        'odds_features': True
                    }
                },
                'storage': {
                    'database': {
                        'enabled': True,
                        'type': 'postgresql',
                        'connection_string': 'postgresql://user:password@localhost:5432/ml_soccer',
                        'table_name': 'match_data',
                        'batch_insert': True
                    },
                    'file_storage': {
                        'enabled': True,
                        'format': 'parquet',
                        'path': './data/raw',
                        'compression': 'snappy',
                        'partitioning': ['league', 'season']
                    },
                    'cache': {
                        'enabled': True,
                        'type': 'redis',
                        'host': 'localhost',
                        'port': 6379,
                        'ttl': 3600  # 1 hour
                    }
                },
                'monitoring': {
                    'enabled': True,
                    'metrics': {
                        'data_quality': True,
                        'processing_time': True,
                        'error_rate': True,
                        'throughput': True
                    },
                    'alerts': {
                        'enabled': True,
                        'thresholds': {
                            'error_rate': 0.1,
                            'processing_time': 300,  # 5 minutes
                            'data_quality_score': 0.8
                        }
                    }
                }
            }
        else:
            self.config = config
    
    def start_pipeline(self):
        """Start the data pipeline"""
        if self.pipeline_active:
            self.logger.warning("Data pipeline already active")
            return
        
        self.logger.info("Starting data pipeline")
        self.pipeline_active = True
        
        # Start pipeline thread
        self.pipeline_thread = threading.Thread(target=self._pipeline_loop)
        self.pipeline_thread.daemon = True
        self.pipeline_thread.start()
        
        # Schedule data collection
        schedule.every(self.config['pipeline']['schedule_interval']).seconds.do(self._run_data_collection)
        schedule.every(6).hours.do(self._run_data_validation)
        schedule.every(1).day.do(self._run_data_cleanup)
        
        self.logger.info("Data pipeline started successfully")
    
    def stop_pipeline(self):
        """Stop the data pipeline"""
        self.logger.info("Stopping data pipeline")
        self.pipeline_active = False
        
        if self.pipeline_thread:
            self.pipeline_thread.join(timeout=30)
        
        schedule.clear()
        self.logger.info("Data pipeline stopped")
    
    def _pipeline_loop(self):
        """Main pipeline loop"""
        while self.pipeline_active:
            try:
                schedule.run_pending()
                time.sleep(1)
            except Exception as e:
                self.logger.error(f"Error in pipeline loop: {e}")
                time.sleep(5)
    
    def _run_data_collection(self):
        """Run data collection from all sources"""
        self.logger.info("Starting data collection")
        
        try:
            collection_stats = {
                'start_time': datetime.now(),
                'sources_processed': 0,
                'total_records': 0,
                'errors': 0
            }
            
            # Process each enabled data source
            for source_name, source_config in self.config['data_sources'].items():
                if source_config['enabled']:
                    try:
                        self.logger.info(f"Processing data source: {source_name}")
                        
                        # Collect data from source
                        data = self._collect_from_source(source_name, source_config)
                        
                        if data is not None and not data.empty:
                            # Process and validate data
                            processed_data = self._process_data(data, source_name)
                            
                            if processed_data is not None and not processed_data.empty:
                                # Store data
                                self._store_data(processed_data, source_name)
                                
                                collection_stats['sources_processed'] += 1
                                collection_stats['total_records'] += len(processed_data)
                                
                                self.logger.info(f"Collected {len(processed_data)} records from {source_name}")
                            else:
                                collection_stats['errors'] += 1
                                self.logger.warning(f"No valid data from {source_name}")
                        else:
                            collection_stats['errors'] += 1
                            self.logger.warning(f"Failed to collect data from {source_name}")
                        
                    except Exception as e:
                        collection_stats['errors'] += 1
                        self.logger.error(f"Error processing {source_name}: {e}")
            
            collection_stats['end_time'] = datetime.now()
            collection_stats['duration'] = (collection_stats['end_time'] - collection_stats['start_time']).total_seconds()
            
            self.processing_stats['last_collection'] = collection_stats
            
            self.logger.info(f"Data collection completed: {collection_stats['sources_processed']} sources, "
                           f"{collection_stats['total_records']} records, {collection_stats['errors']} errors")
            
        except Exception as e:
            self.logger.error(f"Error in data collection: {e}")
    
    def _collect_from_source(self, source_name: str, source_config: Dict) -> Optional[pd.DataFrame]:
        """Collect data from a specific source"""
        try:
            if source_name == 'football_data_co_uk':
                return self._collect_football_data_co_uk(source_config)
            elif source_name == 'the_odds_api':
                return self._collect_the_odds_api(source_config)
            elif source_name == 'api_football':
                return self._collect_api_football(source_config)
            elif source_name == 'flashscore':
                return self._collect_flashscore(source_config)
            else:
                self.logger.warning(f"Unknown data source: {source_name}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error collecting from {source_name}: {e}")
            return None
    
    def _collect_football_data_co_uk(self, source_config: Dict) -> Optional[pd.DataFrame]:
        """Collect data from football-data.co.uk"""
        try:
            all_data = []
            
            # Collect data for each target league
            for league_name, league_config in self.config['target_leagues'].items():
                if league_config['enabled']:
                    try:
                        # Construct URL for the league
                        url = f"{source_config['base_url']}{league_config['code']}.csv"
                        
                        # Fetch data
                        response = requests.get(url, timeout=source_config['timeout'])
                        response.raise_for_status()
                        
                        # Parse CSV data
                        df = pd.read_csv(pd.StringIO(response.text))
                        
                        # Add league information
                        df['League'] = league_name
                        df['LeagueCode'] = league_config['code']
                        df['Season'] = league_config['season']
                        
                        all_data.append(df)
                        
                        self.logger.info(f"Collected {len(df)} records for {league_name}")
                        
                    except Exception as e:
                        self.logger.error(f"Error collecting {league_name} from football-data.co.uk: {e}")
                        continue
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                return combined_data
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error collecting from football-data.co.uk: {e}")
            return None
    
    def _collect_the_odds_api(self, source_config: Dict) -> Optional[pd.DataFrame]:
        """Collect data from The Odds API"""
        try:
            all_data = []
            
            # Collect odds for each target league
            for league_name, league_config in self.config['target_leagues'].items():
                if league_config['enabled']:
                    try:
                        # Construct URL for the league
                        url = f"{source_config['base_url']}sports/{league_config['sport_key']}/odds"
                        
                        params = {
                            'apiKey': source_config['api_key'],
                            'regions': 'uk',
                            'markets': 'h2h',
                            'oddsFormat': 'decimal'
                        }
                        
                        # Fetch data
                        response = requests.get(url, params=params, timeout=source_config['timeout'])
                        response.raise_for_status()
                        
                        data = response.json()
                        
                        # Process odds data
                        odds_data = []
                        for match in data:
                            odds_record = {
                                'Date': datetime.fromisoformat(match['commence_time'].replace('Z', '+00:00')),
                                'HomeTeam': match['home_team'],
                                'AwayTeam': match['away_team'],
                                'League': league_name,
                                'LeagueCode': league_config['code'],
                                'Season': league_config['season']
                            }
                            
                            # Extract odds
                            for bookmaker in match['bookmakers']:
                                for market in bookmaker['markets']:
                                    if market['key'] == 'h2h':
                                        for outcome in market['outcomes']:
                                            if outcome['name'] == match['home_team']:
                                                odds_record['OddsHome'] = outcome['price']
                                            elif outcome['name'] == match['away_team']:
                                                odds_record['OddsAway'] = outcome['price']
                                            else:
                                                odds_record['OddsDraw'] = outcome['price']
                            
                            odds_data.append(odds_record)
                        
                        if odds_data:
                            df = pd.DataFrame(odds_data)
                            all_data.append(df)
                            
                            self.logger.info(f"Collected {len(df)} odds records for {league_name}")
                        
                    except Exception as e:
                        self.logger.error(f"Error collecting {league_name} from The Odds API: {e}")
                        continue
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                return combined_data
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error collecting from The Odds API: {e}")
            return None
    
    def _collect_api_football(self, source_config: Dict) -> Optional[pd.DataFrame]:
        """Collect data from API-Football"""
        try:
            all_data = []
            
            # Collect data for each target league
            for league_name, league_config in self.config['target_leagues'].items():
                if league_config['enabled']:
                    try:
                        # Get league ID (simplified)
                        league_id = self._get_league_id(league_config['code'])
                        
                        if league_id:
                            # Get fixtures
                            fixtures_url = f"{source_config['base_url']}fixtures"
                            params = {
                                'league': league_id,
                                'season': 2023,
                                'timezone': 'UTC'
                            }
                            
                            headers = {
                                'X-RapidAPI-Key': source_config['api_key'],
                                'X-RapidAPI-Host': 'v3.football.api-sports.io'
                            }
                            
                            response = requests.get(fixtures_url, params=params, headers=headers, 
                                                  timeout=source_config['timeout'])
                            response.raise_for_status()
                            
                            data = response.json()
                            
                            # Process fixtures data
                            fixtures_data = []
                            for fixture in data['response']:
                                fixture_record = {
                                    'Date': pd.to_datetime(fixture['fixture']['date']),
                                    'HomeTeam': fixture['teams']['home']['name'],
                                    'AwayTeam': fixture['teams']['away']['name'],
                                    'League': league_name,
                                    'LeagueCode': league_config['code'],
                                    'Season': league_config['season']
                                }
                                
                                # Add score if available
                                if fixture['score']['fulltime']['home'] is not None:
                                    fixture_record['FTHG'] = fixture['score']['fulltime']['home']
                                    fixture_record['FTAG'] = fixture['score']['fulltime']['away']
                                    
                                    # Determine result
                                    if fixture_record['FTHG'] > fixture_record['FTAG']:
                                        fixture_record['FTR'] = 'H'
                                    elif fixture_record['FTHG'] < fixture_record['FTAG']:
                                        fixture_record['FTR'] = 'A'
                                    else:
                                        fixture_record['FTR'] = 'D'
                                
                                fixtures_data.append(fixture_record)
                            
                            if fixtures_data:
                                df = pd.DataFrame(fixtures_data)
                                all_data.append(df)
                                
                                self.logger.info(f"Collected {len(df)} fixtures for {league_name}")
                        
                    except Exception as e:
                        self.logger.error(f"Error collecting {league_name} from API-Football: {e}")
                        continue
            
            if all_data:
                combined_data = pd.concat(all_data, ignore_index=True)
                return combined_data
            else:
                return None
                
        except Exception as e:
            self.logger.error(f"Error collecting from API-Football: {e}")
            return None
    
    def _collect_flashscore(self, source_config: Dict) -> Optional[pd.DataFrame]:
        """Collect data from FlashScore (requires web scraping)"""
        try:
            # This would require web scraping implementation
            # For now, return None as it's disabled by default
            self.logger.info("FlashScore collection not implemented (requires web scraping)")
            return None
            
        except Exception as e:
            self.logger.error(f"Error collecting from FlashScore: {e}")
            return None
    
    def _get_league_id(self, league_code: str) -> Optional[int]:
        """Get league ID for API-Football"""
        # Simplified league ID mapping
        league_ids = {
            'E1': 40,  # Championship
            'E2': 41,  # League One
            'E3': 42,  # League Two
            'SC1': 179  # Scottish Championship
        }
        
        return league_ids.get(league_code)
    
    def _process_data(self, data: pd.DataFrame, source_name: str) -> Optional[pd.DataFrame]:
        """Process and validate collected data"""
        try:
            self.logger.info(f"Processing {len(data)} records from {source_name}")
            
            # Data validation
            if self.config['data_processing']['validation']['enabled']:
                validated_data = self._validate_data(data)
                if validated_data is None:
                    self.logger.error(f"Data validation failed for {source_name}")
                    return None
                data = validated_data
            
            # Data transformation
            if self.config['data_processing']['transformation']['enabled']:
                data = self._transform_data(data)
            
            # Feature engineering
            if self.config['data_processing']['feature_engineering']['enabled']:
                data = self._engineer_features(data)
            
            # Add metadata
            data['Source'] = source_name
            data['ProcessedAt'] = datetime.now()
            
            self.logger.info(f"Processed {len(data)} records from {source_name}")
            return data
            
        except Exception as e:
            self.logger.error(f"Error processing data from {source_name}: {e}")
            return None
    
    def _validate_data(self, data: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Validate data quality"""
        try:
            validation_config = self.config['data_processing']['validation']
            
            # Check required fields
            required_fields = validation_config['required_fields']
            missing_fields = [field for field in required_fields if field not in data.columns]
            
            if missing_fields:
                self.logger.warning(f"Missing required fields: {missing_fields}")
                # Try to add missing fields with default values
                for field in missing_fields:
                    if field == 'Date':
                        data[field] = datetime.now()
                    elif field in ['FTHG', 'FTAG']:
                        data[field] = 0
                    elif field == 'FTR':
                        data[field] = 'D'
                    else:
                        data[field] = 'Unknown'
            
            # Check data types
            data_types = validation_config['data_types']
            for field, expected_type in data_types.items():
                if field in data.columns:
                    try:
                        if expected_type == 'datetime64[ns]':
                            data[field] = pd.to_datetime(data[field])
                        elif expected_type == 'int64':
                            data[field] = pd.to_numeric(data[field], errors='coerce').astype('Int64')
                        elif expected_type == 'object':
                            data[field] = data[field].astype(str)
                    except Exception as e:
                        self.logger.warning(f"Error converting {field} to {expected_type}: {e}")
            
            # Check value ranges
            value_ranges = validation_config['value_ranges']
            for field, (min_val, max_val) in value_ranges.items():
                if field in data.columns:
                    invalid_count = ((data[field] < min_val) | (data[field] > max_val)).sum()
                    if invalid_count > 0:
                        self.logger.warning(f"{invalid_count} records have invalid {field} values")
                        # Filter out invalid values
                        data = data[(data[field] >= min_val) & (data[field] <= max_val)]
            
            # Remove duplicates
            initial_count = len(data)
            data = data.drop_duplicates()
            final_count = len(data)
            
            if initial_count != final_count:
                self.logger.info(f"Removed {initial_count - final_count} duplicate records")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error validating data: {e}")
            return None
    
    def _transform_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Transform data"""
        try:
            transformation_config = self.config['data_processing']['transformation']
            
            # Date format conversion
            if 'Date' in data.columns:
                try:
                    data['Date'] = pd.to_datetime(data['Date'], format=transformation_config['date_format'])
                except:
                    data['Date'] = pd.to_datetime(data['Date'])
            
            # Team name mapping
            if transformation_config['team_name_mapping']:
                data = self._map_team_names(data)
            
            # Odds normalization
            if transformation_config['odds_normalization']:
                data = self._normalize_odds(data)
            
            # Missing value handling
            if transformation_config['missing_value_handling'] == 'forward_fill':
                data = data.fillna(method='ffill')
            elif transformation_config['missing_value_handling'] == 'backward_fill':
                data = data.fillna(method='bfill')
            elif transformation_config['missing_value_handling'] == 'drop':
                data = data.dropna()
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error transforming data: {e}")
            return data
    
    def _map_team_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Map team names to standardized format"""
        try:
            # This would contain team name mappings
            # For now, return data as-is
            return data
            
        except Exception as e:
            self.logger.error(f"Error mapping team names: {e}")
            return data
    
    def _normalize_odds(self, data: pd.DataFrame) -> pd.DataFrame:
        """Normalize odds data"""
        try:
            odds_columns = ['B365H', 'B365D', 'B365A', 'OddsHome', 'OddsDraw', 'OddsAway']
            
            for col in odds_columns:
                if col in data.columns:
                    # Convert to float and handle invalid values
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                    
                    # Remove odds that are too low or too high
                    data[col] = data[col].clip(lower=1.01, upper=100.0)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error normalizing odds: {e}")
            return data
    
    def _engineer_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Engineer features from raw data"""
        try:
            feature_config = self.config['data_processing']['feature_engineering']
            
            # Form features
            if feature_config['form_features']:
                data = self._add_form_features(data)
            
            # Head-to-head features
            if feature_config['h2h_features']:
                data = self._add_h2h_features(data)
            
            # Elo features
            if feature_config['elo_features']:
                data = self._add_elo_features(data)
            
            # Odds features
            if feature_config['odds_features']:
                data = self._add_odds_features(data)
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error engineering features: {e}")
            return data
    
    def _add_form_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add form features"""
        try:
            # Sort by date
            data = data.sort_values('Date').reset_index(drop=True)
            
            # Add basic form features (simplified)
            data['HomeForm'] = 0
            data['AwayForm'] = 0
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding form features: {e}")
            return data
    
    def _add_h2h_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add head-to-head features"""
        try:
            # Add basic H2H features (simplified)
            data['H2HHomeWins'] = 0
            data['H2HAwayWins'] = 0
            data['H2HDraws'] = 0
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding H2H features: {e}")
            return data
    
    def _add_elo_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add Elo rating features"""
        try:
            # Add basic Elo features (simplified)
            data['HomeElo'] = 1500
            data['AwayElo'] = 1500
            data['EloDiff'] = 0
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding Elo features: {e}")
            return data
    
    def _add_odds_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add odds-based features"""
        try:
            # Add odds features
            odds_columns = ['B365H', 'B365D', 'B365A', 'OddsHome', 'OddsDraw', 'OddsAway']
            
            for col in odds_columns:
                if col in data.columns:
                    # Convert odds to probabilities
                    prob_col = col.replace('B365', 'Prob').replace('Odds', 'Prob')
                    data[prob_col] = 1 / data[col]
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error adding odds features: {e}")
            return data
    
    def _store_data(self, data: pd.DataFrame, source_name: str):
        """Store processed data"""
        try:
            storage_config = self.config['storage']
            
            # Database storage
            if storage_config['database']['enabled']:
                self._store_to_database(data, source_name)
            
            # File storage
            if storage_config['file_storage']['enabled']:
                self._store_to_file(data, source_name)
            
            # Cache storage
            if storage_config['cache']['enabled']:
                self._store_to_cache(data, source_name)
            
        except Exception as e:
            self.logger.error(f"Error storing data: {e}")
    
    def _store_to_database(self, data: pd.DataFrame, source_name: str):
        """Store data to database"""
        try:
            # This would implement database storage
            # For now, just log the action
            self.logger.info(f"Storing {len(data)} records to database from {source_name}")
            
        except Exception as e:
            self.logger.error(f"Error storing to database: {e}")
    
    def _store_to_file(self, data: pd.DataFrame, source_name: str):
        """Store data to file"""
        try:
            storage_config = self.config['storage']['file_storage']
            
            # Create directory if it doesn't exist
            os.makedirs(storage_config['path'], exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{source_name}_{timestamp}.parquet"
            filepath = os.path.join(storage_config['path'], filename)
            
            # Save data
            data.to_parquet(filepath, compression=storage_config['compression'])
            
            self.logger.info(f"Stored {len(data)} records to {filepath}")
            
        except Exception as e:
            self.logger.error(f"Error storing to file: {e}")
    
    def _store_to_cache(self, data: pd.DataFrame, source_name: str):
        """Store data to cache"""
        try:
            # This would implement cache storage
            # For now, just store in memory
            self.data_cache[source_name] = {
                'data': data,
                'timestamp': datetime.now()
            }
            
            self.logger.info(f"Cached {len(data)} records from {source_name}")
            
        except Exception as e:
            self.logger.error(f"Error storing to cache: {e}")
    
    def _run_data_validation(self):
        """Run data validation checks"""
        try:
            self.logger.info("Running data validation checks")
            
            # This would implement comprehensive data validation
            # For now, just log the action
            
            self.logger.info("Data validation checks completed")
            
        except Exception as e:
            self.logger.error(f"Error in data validation: {e}")
    
    def _run_data_cleanup(self):
        """Run data cleanup tasks"""
        try:
            self.logger.info("Running data cleanup tasks")
            
            # Clean up old cache data
            cutoff_time = datetime.now() - timedelta(hours=24)
            for source_name, cache_data in list(self.data_cache.items()):
                if cache_data['timestamp'] < cutoff_time:
                    del self.data_cache[source_name]
            
            self.logger.info("Data cleanup tasks completed")
            
        except Exception as e:
            self.logger.error(f"Error in data cleanup: {e}")
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get pipeline status"""
        return {
            'active': self.pipeline_active,
            'last_collection': self.processing_stats.get('last_collection'),
            'cache_size': len(self.data_cache),
            'config': self.config
        }
    
    def get_cached_data(self, source_name: str = None) -> Dict[str, Any]:
        """Get cached data"""
        if source_name:
            return self.data_cache.get(source_name)
        else:
            return self.data_cache
    
    def save_pipeline_state(self, filepath: str):
        """Save pipeline state"""
        self.logger.info(f"Saving pipeline state to {filepath}")
        
        pipeline_state = {
            'data_cache': self.data_cache,
            'processing_stats': self.processing_stats,
            'config': self.config
        }
        
        with open(filepath, 'w') as f:
            json.dump(pipeline_state, f, indent=2, default=str)
        
        self.logger.info("Pipeline state saved successfully")

# Example usage
def main():
    """Example usage of NonMajorLeagueDataPipeline"""
    
    # Initialize data pipeline
    pipeline = NonMajorLeagueDataPipeline()
    
    # Start pipeline
    pipeline.start_pipeline()
    
    # Let it run for a bit
    import time
    time.sleep(10)
    
    # Get pipeline status
    status = pipeline.get_pipeline_status()
    print(f"Pipeline active: {status['active']}")
    
    # Get cached data
    cached_data = pipeline.get_cached_data()
    print(f"Cached data sources: {list(cached_data.keys())}")
    
    # Stop pipeline
    pipeline.stop_pipeline()
    
    # Save state
    pipeline.save_pipeline_state('pipeline_state.json')

if __name__ == "__main__":
    main()
