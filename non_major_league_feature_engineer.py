import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeagueFeatureEngineer:
    """
    Advanced feature engineering for non-major soccer leagues
    
    Key Features:
    - Robust statistical features for limited data
    - Cross-league transfer learning features
    - Market inefficiency detection
    - Temporal and seasonal features
    - Team consistency and form features
    - League-specific adaptations
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize feature engineer with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.feature_cache = {}
        self.scalers = {}
        self.encoders = {}
        
    def setup_logging(self):
        """Setup logging for feature engineering"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load feature engineering configuration"""
        if config is None:
            self.config = {
                'form_features': {
                    'window_sizes': [3, 5, 10],
                    'use_median': True,  # More robust for limited data
                    'exponential_decay': True,
                    'decay_rate': 0.9
                },
                'consistency_features': {
                    'enabled': True,
                    'min_matches': 5,
                    'use_percentiles': True
                },
                'market_features': {
                    'enabled': True,
                    'odds_sources': ['home', 'draw', 'away'],
                    'disagreement_threshold': 0.1
                },
                'temporal_features': {
                    'enabled': True,
                    'seasonal_patterns': True,
                    'fatigue_factors': True,
                    'rest_days': True
                },
                'seasonal_features': {
                    'enabled': True,
                    'phases': ['early', 'mid', 'late'],
                    'month_encoding': 'cyclical'
                },
                'fixture_congestion': {
                    'enabled': True,
                    'windows': [7, 14, 21],
                    'fatigue_weights': [1.0, 0.8, 0.6]
                },
                'temporal_context': {
                    'enabled': True,
                    'day_of_week': True,
                    'match_time': True,
                    'special_matches': True
                },
                'momentum_features': {
                    'enabled': True,
                    'streak_threshold': 3,
                    'momentum_window': 10
                },
                'league_position': {
                    'enabled': True,
                    'update_frequency': 'match'
                },
                'cross_league': {
                    'enabled': True,
                    'reference_leagues': ['EPL', 'LaLiga', 'Bundesliga'],
                    'transfer_ratio': 0.3
                },
                'league_specific': {
                    'championship': {'home_advantage': 0.15, 'avg_goals': 2.5},
                    'ligue2': {'home_advantage': 0.12, 'avg_goals': 2.3},
                    '2bundesliga': {'home_advantage': 0.18, 'avg_goals': 2.7},
                    'serie_b': {'home_advantage': 0.14, 'avg_goals': 2.4},
                    'segunda_division': {'home_advantage': 0.16, 'avg_goals': 2.6},
                    'eredivisie': {'home_advantage': 0.13, 'avg_goals': 2.8},
                    'primeira_liga': {'home_advantage': 0.17, 'avg_goals': 2.9}
                }
            }
        else:
            self.config = config
    
    def create_all_features(self, df: pd.DataFrame, league_code: str = None) -> pd.DataFrame:
        """
        Create all features for non-major league data
        
        Args:
            df: Input DataFrame
            league_code: League identifier
            
        Returns:
            DataFrame with engineered features
        """
        self.logger.info(f"Starting feature engineering for {len(df)} matches")
        
        # Make a copy to avoid modifying original data
        df_features = df.copy()
        
        # 1. Basic match features
        df_features = self._create_basic_features(df_features)
        
        # 2. Team form features (robust for limited data)
        df_features = self._create_form_features(df_features)
        
        # 3. Consistency features
        df_features = self._create_consistency_features(df_features)
        
        # 4. Market features (if odds data available)
        df_features = self._create_market_features(df_features)
        
        # 5. Temporal features
        df_features = self._create_temporal_features(df_features)
        
        # 5.1. Advanced seasonal features
        df_features = self._create_seasonal_features(df_features)
        
        # 5.2. Fixture congestion features
        df_features = self._create_fixture_congestion_features(df_features)
        
        # 5.3. Temporal context features
        df_features = self._create_temporal_context_features(df_features)
        
        # 5.4. Momentum features
        df_features = self._create_momentum_features(df_features)
        
        # 5.5. League position features
        df_features = self._create_league_position_features(df_features)
        
        # 6. League-specific features
        df_features = self._create_league_specific_features(df_features, league_code)
        
        # 7. Cross-league features
        df_features = self._create_cross_league_features(df_features, league_code)
        
        # 8. Advanced statistical features
        df_features = self._create_advanced_statistical_features(df_features)
        
        # 9. Team embedding features
        df_features = self._create_team_embedding_features(df_features)
        
        # 10. Final feature selection and scaling
        df_features = self._finalize_features(df_features)
        
        self.logger.info(f"Feature engineering complete. Final features: {len(df_features.columns)}")
        return df_features
    
    def _create_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create basic match features"""
        self.logger.info("Creating basic features")
        
        # Match outcome features
        df['home_win'] = (df['FTR'] == 'H').astype(int)
        df['draw'] = (df['FTR'] == 'D').astype(int)
        df['away_win'] = (df['FTR'] == 'A').astype(int)
        
        # Goal features
        df['goal_difference'] = df['FTHG'] - df['FTAG']
        df['total_goals'] = df['FTHG'] + df['FTAG']
        df['home_goals_scored'] = df['FTHG']
        df['away_goals_scored'] = df['FTAG']
        df['home_goals_conceded'] = df['FTAG']
        df['away_goals_conceded'] = df['FTHG']
        
        # Match intensity
        df['match_intensity'] = df['total_goals'] / 2
        
        # Home advantage
        df['home_advantage'] = df['FTHG'] - df['FTAG']
        
        return df
    
    def _create_form_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team form features (robust for limited data)"""
        self.logger.info("Creating form features")
        
        window_sizes = self.config['form_features']['window_sizes']
        use_median = self.config['form_features']['use_median']
        exponential_decay = self.config['form_features']['exponential_decay']
        decay_rate = self.config['form_features']['decay_rate']
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Initialize form columns
        for window in window_sizes:
            df[f'home_form_{window}'] = 0.0
            df[f'away_form_{window}'] = 0.0
            df[f'home_goals_scored_{window}'] = 0.0
            df[f'away_goals_scored_{window}'] = 0.0
            df[f'home_goals_conceded_{window}'] = 0.0
            df[f'away_goals_conceded_{window}'] = 0.0
        
        # Calculate form for each team
        all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        
        for team in all_teams:
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
            team_matches = team_matches.sort_values('Date').reset_index(drop=True)
            
            if len(team_matches) < 2:
                continue
            
            for i in range(len(team_matches)):
                current_match = team_matches.iloc[i]
                current_date = current_match['Date']
                
                # Get previous matches
                previous_matches = team_matches[team_matches['Date'] < current_date].tail(max(window_sizes))
                
                if len(previous_matches) == 0:
                    continue
                
                # Calculate form for each window size
                for window in window_sizes:
                    window_matches = previous_matches.tail(window)
                    
                    if len(window_matches) == 0:
                        continue
                    
                    # Calculate points (3 for win, 1 for draw, 0 for loss)
                    points = []
                    goals_scored = []
                    goals_conceded = []
                    
                    for _, match in window_matches.iterrows():
                        if match['HomeTeam'] == team:
                            # Home match
                            if match['FTR'] == 'H':
                                points.append(3)
                            elif match['FTR'] == 'D':
                                points.append(1)
                            else:
                                points.append(0)
                            goals_scored.append(match['FTHG'])
                            goals_conceded.append(match['FTAG'])
                        else:
                            # Away match
                            if match['FTR'] == 'A':
                                points.append(3)
                            elif match['FTR'] == 'D':
                                points.append(1)
                            else:
                                points.append(0)
                            goals_scored.append(match['FTAG'])
                            goals_conceded.append(match['FTHG'])
                    
                    if points:
                        # Calculate form (points per game)
                        if use_median:
                            form = np.median(points) / 3.0  # Normalize to 0-1
                        else:
                            form = np.mean(points) / 3.0
                        
                        # Apply exponential decay if enabled
                        if exponential_decay and len(points) > 1:
                            weights = [decay_rate ** (len(points) - 1 - i) for i in range(len(points))]
                            form = np.average([p/3.0 for p in points], weights=weights)
                        
                        # Store form
                        match_idx = current_match.name
                        df.loc[match_idx, f'home_form_{window}'] = form if current_match['HomeTeam'] == team else 0
                        df.loc[match_idx, f'away_form_{window}'] = form if current_match['AwayTeam'] == team else 0
                        
                        # Store goals
                        if use_median:
                            goals_scored_avg = np.median(goals_scored)
                            goals_conceded_avg = np.median(goals_conceded)
                        else:
                            goals_scored_avg = np.mean(goals_scored)
                            goals_conceded_avg = np.mean(goals_conceded)
                        
                        df.loc[match_idx, f'home_goals_scored_{window}'] = goals_scored_avg if current_match['HomeTeam'] == team else 0
                        df.loc[match_idx, f'away_goals_scored_{window}'] = goals_scored_avg if current_match['AwayTeam'] == team else 0
                        df.loc[match_idx, f'home_goals_conceded_{window}'] = goals_conceded_avg if current_match['HomeTeam'] == team else 0
                        df.loc[match_idx, f'away_goals_conceded_{window}'] = goals_conceded_avg if current_match['AwayTeam'] == team else 0
        
        return df
    
    def _create_consistency_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team consistency features"""
        self.logger.info("Creating consistency features")
        
        if not self.config['consistency_features']['enabled']:
            return df
        
        min_matches = self.config['consistency_features']['min_matches']
        use_percentiles = self.config['consistency_features']['use_percentiles']
        
        # Initialize consistency columns
        df['home_consistency'] = 0.0
        df['away_consistency'] = 0.0
        df['home_goal_consistency'] = 0.0
        df['away_goal_consistency'] = 0.0
        
        all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        
        for team in all_teams:
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
            team_matches = team_matches.sort_values('Date').reset_index(drop=True)
            
            if len(team_matches) < min_matches:
                continue
            
            for i in range(len(team_matches)):
                current_match = team_matches.iloc[i]
                current_date = current_match['Date']
                
                # Get previous matches
                previous_matches = team_matches[team_matches['Date'] < current_date].tail(min_matches)
                
                if len(previous_matches) < min_matches:
                    continue
                
                # Calculate consistency metrics
                points = []
                goals_scored = []
                goals_conceded = []
                
                for _, match in previous_matches.iterrows():
                    if match['HomeTeam'] == team:
                        if match['FTR'] == 'H':
                            points.append(3)
                        elif match['FTR'] == 'D':
                            points.append(1)
                        else:
                            points.append(0)
                        goals_scored.append(match['FTHG'])
                        goals_conceded.append(match['FTAG'])
                    else:
                        if match['FTR'] == 'A':
                            points.append(3)
                        elif match['FTR'] == 'D':
                            points.append(1)
                        else:
                            points.append(0)
                        goals_scored.append(match['FTAG'])
                        goals_conceded.append(match['FTHG'])
                
                if len(points) >= min_matches:
                    # Calculate consistency (1 - coefficient of variation)
                    if use_percentiles:
                        points_consistency = 1 - (np.percentile(points, 75) - np.percentile(points, 25)) / np.median(points)
                        goals_consistency = 1 - (np.percentile(goals_scored, 75) - np.percentile(goals_scored, 25)) / np.median(goals_scored)
                    else:
                        points_consistency = 1 - (np.std(points) / np.mean(points)) if np.mean(points) > 0 else 0
                        goals_consistency = 1 - (np.std(goals_scored) / np.mean(goals_scored)) if np.mean(goals_scored) > 0 else 0
                    
                    # Store consistency
                    match_idx = current_match.name
                    df.loc[match_idx, 'home_consistency'] = points_consistency if current_match['HomeTeam'] == team else 0
                    df.loc[match_idx, 'away_consistency'] = points_consistency if current_match['AwayTeam'] == team else 0
                    df.loc[match_idx, 'home_goal_consistency'] = goals_consistency if current_match['HomeTeam'] == team else 0
                    df.loc[match_idx, 'away_goal_consistency'] = goals_consistency if current_match['AwayTeam'] == team else 0
        
        return df
    
    def _create_market_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create market-based features"""
        self.logger.info("Creating market features")
        
        if not self.config['market_features']['enabled']:
            return df
        
        # Check if odds data is available
        odds_columns = ['B365H', 'B365D', 'B365A', 'BW_H', 'BW_D', 'BW_A', 'IW_H', 'IW_D', 'IW_A']
        available_odds = [col for col in odds_columns if col in df.columns]
        
        if not available_odds:
            self.logger.warning("No odds data found, skipping market features")
            return df
        
        # Calculate market probabilities
        for odds_col in available_odds:
            if odds_col.endswith('H'):
                prob_col = odds_col.replace('H', 'prob_home')
                df[prob_col] = 1 / df[odds_col]
            elif odds_col.endswith('D'):
                prob_col = odds_col.replace('D', 'prob_draw')
                df[prob_col] = 1 / df[odds_col]
            elif odds_col.endswith('A'):
                prob_col = odds_col.replace('A', 'prob_away')
                df[prob_col] = 1 / df[odds_col]
        
        # Calculate market disagreement
        home_probs = [col for col in df.columns if 'prob_home' in col]
        draw_probs = [col for col in df.columns if 'prob_draw' in col]
        away_probs = [col for col in df.columns if 'prob_away' in col]
        
        if home_probs and draw_probs and away_probs:
            df['market_home_std'] = df[home_probs].std(axis=1)
            df['market_draw_std'] = df[draw_probs].std(axis=1)
            df['market_away_std'] = df[away_probs].std(axis=1)
            df['market_disagreement'] = df['market_home_std'] + df['market_draw_std'] + df['market_away_std']
        
        # Calculate market entropy
        if home_probs and draw_probs and away_probs:
            df['market_entropy'] = df.apply(
                lambda row: -sum([
                    row[col] * np.log(row[col] + 1e-10) 
                    for col in home_probs + draw_probs + away_probs
                ]), axis=1
            )
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features"""
        self.logger.info("Creating temporal features")
        
        if not self.config['temporal_features']['enabled']:
            return df
        
        # Date features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['is_midweek'] = df['day_of_week'].isin([1, 2, 3]).astype(int)
        
        # Seasonal features
        if self.config['temporal_features']['seasonal_patterns']:
            df['season'] = df['month'].apply(
                lambda x: 'Spring' if x in [3, 4, 5] 
                         else 'Summer' if x in [6, 7, 8] 
                         else 'Autumn' if x in [9, 10, 11] 
                         else 'Winter'
            )
            
            # Encode seasons
            season_encoder = LabelEncoder()
            df['season_encoded'] = season_encoder.fit_transform(df['season'])
            self.encoders['season'] = season_encoder
        
        # Fatigue factors
        if self.config['temporal_features']['fatigue_factors']:
            df = self._add_fatigue_features(df)
        
        # Rest days
        if self.config['temporal_features']['rest_days']:
            df = self._add_rest_day_features(df)
        
        return df
    
    def _add_fatigue_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add fatigue-related features"""
        df['home_fatigue'] = 0.0
        df['away_fatigue'] = 0.0
        
        all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        
        for team in all_teams:
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
            team_matches = team_matches.sort_values('Date').reset_index(drop=True)
            
            for i in range(1, len(team_matches)):
                current_match = team_matches.iloc[i]
                previous_match = team_matches.iloc[i-1]
                
                # Calculate days since last match
                days_since_last = (current_match['Date'] - previous_match['Date']).days
                
                # Fatigue factor (inverse of rest days)
                fatigue_factor = 1 / (days_since_last + 1)
                
                match_idx = current_match.name
                df.loc[match_idx, 'home_fatigue'] = fatigue_factor if current_match['HomeTeam'] == team else 0
                df.loc[match_idx, 'away_fatigue'] = fatigue_factor if current_match['AwayTeam'] == team else 0
        
        return df
    
    def _add_rest_day_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rest day features"""
        df['home_rest_days'] = 0
        df['away_rest_days'] = 0
        
        all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        
        for team in all_teams:
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
            team_matches = team_matches.sort_values('Date').reset_index(drop=True)
            
            for i in range(1, len(team_matches)):
                current_match = team_matches.iloc[i]
                previous_match = team_matches.iloc[i-1]
                
                # Calculate rest days
                rest_days = (current_match['Date'] - previous_match['Date']).days
                
                match_idx = current_match.name
                df.loc[match_idx, 'home_rest_days'] = rest_days if current_match['HomeTeam'] == team else 0
                df.loc[match_idx, 'away_rest_days'] = rest_days if current_match['AwayTeam'] == team else 0
        
        return df
    
    def _create_seasonal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced seasonal pattern features"""
        self.logger.info("Creating seasonal features")
        
        if not self.config['seasonal_features']['enabled']:
            return df
        
        # Month-of-season encoding (start: August=1, end: May=9)
        df['month_of_season'] = df['month'].apply(
            lambda x: x - 7 if x >= 8 else x + 5  # Aug=1, Sep=2, ..., May=9
        )
        
        # Season phase: early (Aug-Oct), mid (Nov-Feb), late (Mar-May)
        df['season_phase'] = df['month'].apply(
            lambda x: 'early' if x in [8, 9, 10] 
                     else 'mid' if x in [11, 12, 1, 2] 
                     else 'late'  # Mar, Apr, May
        )
        
        # Cyclical month encoding
        if self.config['seasonal_features']['month_encoding'] == 'cyclical':
            df['month_sin'] = np.sin(2 * np.pi * df['month_of_season'] / 9)
            df['month_cos'] = np.cos(2 * np.pi * df['month_of_season'] / 9)
        
        # Historical performance by season phase
        df = self._add_seasonal_performance_features(df)
        
        # League position trajectory
        df = self._add_position_trajectory_features(df)
        
        return df
    
    def _add_seasonal_performance_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add historical performance by season phase"""
        df['home_early_season_form'] = 0.0
        df['away_early_season_form'] = 0.0
        df['home_mid_season_form'] = 0.0
        df['away_mid_season_form'] = 0.0
        df['home_late_season_form'] = 0.0
        df['away_late_season_form'] = 0.0
        
        all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        
        for team in all_teams:
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
            team_matches = team_matches.sort_values('Date').reset_index(drop=True)
            
            if len(team_matches) < 5:
                continue
            
            for i in range(len(team_matches)):
                current_match = team_matches.iloc[i]
                current_date = current_match['Date']
                current_phase = current_match['season_phase']
                
                # Get previous matches in same season phase
                previous_matches = team_matches[
                    (team_matches['Date'] < current_date) & 
                    (team_matches['season_phase'] == current_phase)
                ].tail(5)
                
                if len(previous_matches) == 0:
                    continue
                
                # Calculate form in this season phase
                points = []
                for _, match in previous_matches.iterrows():
                    if match['HomeTeam'] == team:
                        if match['FTR'] == 'H':
                            points.append(3)
                        elif match['FTR'] == 'D':
                            points.append(1)
                        else:
                            points.append(0)
                    else:
                        if match['FTR'] == 'A':
                            points.append(3)
                        elif match['FTR'] == 'D':
                            points.append(1)
                        else:
                            points.append(0)
                
                if points:
                    phase_form = np.mean(points) / 3.0
                    
                    match_idx = current_match.name
                    if current_match['HomeTeam'] == team:
                        df.loc[match_idx, f'home_{current_phase}_season_form'] = phase_form
                    else:
                        df.loc[match_idx, f'away_{current_phase}_season_form'] = phase_form
        
        return df
    
    def _add_position_trajectory_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add league position trajectory features"""
        df['home_position_trend'] = 0.0
        df['away_position_trend'] = 0.0
        df['home_position_momentum'] = 0.0
        df['away_position_momentum'] = 0.0
        
        # This would require league table data - placeholder implementation
        # In practice, you'd calculate actual league positions over time
        
        return df
    
    def _create_fixture_congestion_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create fixture congestion and fatigue features"""
        self.logger.info("Creating fixture congestion features")
        
        if not self.config['fixture_congestion']['enabled']:
            return df
        
        windows = self.config['fixture_congestion']['windows']
        fatigue_weights = self.config['fixture_congestion']['fatigue_weights']
        
        # Initialize congestion features
        for window in windows:
            df[f'home_matches_last_{window}d'] = 0
            df[f'away_matches_last_{window}d'] = 0
            df[f'home_fatigue_index_{window}d'] = 0.0
            df[f'away_fatigue_index_{window}d'] = 0.0
        
        df['home_rest_days'] = 0
        df['away_rest_days'] = 0
        df['home_travel_factor'] = 0.0
        df['away_travel_factor'] = 0.0
        
        all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        
        for team in all_teams:
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
            team_matches = team_matches.sort_values('Date').reset_index(drop=True)
            
            for i in range(1, len(team_matches)):
                current_match = team_matches.iloc[i]
                previous_match = team_matches.iloc[i-1]
                current_date = current_match['Date']
                
                # Calculate rest days
                rest_days = (current_date - previous_match['Date']).days
                
                # Calculate matches in each window
                for j, window in enumerate(windows):
                    window_start = current_date - timedelta(days=window)
                    window_matches = team_matches[
                        (team_matches['Date'] >= window_start) & 
                        (team_matches['Date'] < current_date)
                    ]
                    
                    match_count = len(window_matches)
                    fatigue_index = 0.0
                    
                    if match_count > 0:
                        # Calculate fatigue based on match intensity and weights
                        for _, match in window_matches.iterrows():
                            intensity = (match['FTHG'] + match['FTAG']) / 2.0
                            days_ago = (current_date - match['Date']).days
                            weight = fatigue_weights[j] * (1.0 - days_ago / window)
                            fatigue_index += intensity * weight
                    
                    match_idx = current_match.name
                    if current_match['HomeTeam'] == team:
                        df.loc[match_idx, f'home_matches_last_{window}d'] = match_count
                        df.loc[match_idx, f'home_fatigue_index_{window}d'] = fatigue_index
                        df.loc[match_idx, 'home_rest_days'] = rest_days
                    else:
                        df.loc[match_idx, f'away_matches_last_{window}d'] = match_count
                        df.loc[match_idx, f'away_fatigue_index_{window}d'] = fatigue_index
                        df.loc[match_idx, 'away_rest_days'] = rest_days
                
                # Travel factor (simplified - home vs away concentration)
                recent_matches = team_matches[
                    (team_matches['Date'] >= current_date - timedelta(days=30)) & 
                    (team_matches['Date'] < current_date)
                ]
                
                if len(recent_matches) > 0:
                    home_matches = len(recent_matches[recent_matches['HomeTeam'] == team])
                    away_matches = len(recent_matches[recent_matches['AwayTeam'] == team])
                    travel_factor = away_matches / len(recent_matches)
                    
                    match_idx = current_match.name
                    if current_match['HomeTeam'] == team:
                        df.loc[match_idx, 'home_travel_factor'] = travel_factor
                    else:
                        df.loc[match_idx, 'away_travel_factor'] = travel_factor
        
        return df
    
    def _create_temporal_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal context features"""
        self.logger.info("Creating temporal context features")
        
        if not self.config['temporal_context']['enabled']:
            return df
        
        # Day of week effects
        if self.config['temporal_context']['day_of_week']:
            df['is_tuesday'] = (df['day_of_week'] == 1).astype(int)
            df['is_wednesday'] = (df['day_of_week'] == 2).astype(int)
            df['is_thursday'] = (df['day_of_week'] == 3).astype(int)
            df['is_friday'] = (df['day_of_week'] == 4).astype(int)
            df['is_saturday'] = (df['day_of_week'] == 5).astype(int)
            df['is_sunday'] = (df['day_of_week'] == 6).astype(int)
            df['is_monday'] = (df['day_of_week'] == 0).astype(int)
            
            # Midweek vs weekend
            df['is_midweek_game'] = df['day_of_week'].isin([1, 2, 3]).astype(int)
            df['is_weekend_game'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # Match time effects (simplified)
        if self.config['temporal_context']['match_time']:
            # Assume afternoon games (1) vs evening games (0)
            df['is_afternoon_game'] = np.random.choice([0, 1], len(df), p=[0.3, 0.7])
        
        # Special matches
        if self.config['temporal_context']['special_matches']:
            # Public holiday indicators (simplified)
            df['is_holiday'] = ((df['month'] == 12) & (df['Date'].dt.day.isin([24, 25, 26, 31]))).astype(int)
            
            # Derby/rivalry match flags (simplified - would need team rivalry data)
            df['is_derby'] = (df['HomeTeam'] == df['AwayTeam']).astype(int)  # Placeholder
        
        return df
    
    def _create_momentum_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create momentum and streak features"""
        self.logger.info("Creating momentum features")
        
        if not self.config['momentum_features']['enabled']:
            return df
        
        streak_threshold = self.config['momentum_features']['streak_threshold']
        momentum_window = self.config['momentum_features']['momentum_window']
        
        # Initialize momentum features
        df['home_winning_streak'] = 0
        df['away_winning_streak'] = 0
        df['home_losing_streak'] = 0
        df['away_losing_streak'] = 0
        df['home_momentum_score'] = 0.0
        df['away_momentum_score'] = 0.0
        df['home_points_trend'] = 0.0
        df['away_points_trend'] = 0.0
        df['home_goal_diff_trend'] = 0.0
        df['away_goal_diff_trend'] = 0.0
        df['home_clean_sheet_streak'] = 0
        df['away_clean_sheet_streak'] = 0
        
        all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        
        for team in all_teams:
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
            team_matches = team_matches.sort_values('Date').reset_index(drop=True)
            
            if len(team_matches) < momentum_window:
                continue
            
            for i in range(len(team_matches)):
                current_match = team_matches.iloc[i]
                current_date = current_match['Date']
                
                # Get recent matches for momentum calculation
                recent_matches = team_matches[
                    team_matches['Date'] < current_date
                ].tail(momentum_window)
                
                if len(recent_matches) == 0:
                    continue
                
                # Calculate streaks
                win_streak = 0
                loss_streak = 0
                clean_sheet_streak = 0
                
                for j in range(len(recent_matches) - 1, -1, -1):
                    match = recent_matches.iloc[j]
                    
                    if match['HomeTeam'] == team:
                        result = 'H' if match['FTR'] == 'H' else 'D' if match['FTR'] == 'D' else 'A'
                        goals_conceded = match['FTAG']
                    else:
                        result = 'A' if match['FTR'] == 'A' else 'D' if match['FTR'] == 'D' else 'H'
                        goals_conceded = match['FTHG']
                    
                    if result == 'W' and win_streak == (len(recent_matches) - 1 - j):
                        win_streak += 1
                    elif result == 'L' and loss_streak == (len(recent_matches) - 1 - j):
                        loss_streak += 1
                    else:
                        break
                    
                    if goals_conceded == 0 and clean_sheet_streak == (len(recent_matches) - 1 - j):
                        clean_sheet_streak += 1
                    else:
                        break
                
                # Calculate momentum score (weighted recent results)
                momentum_score = 0.0
                points_trend = 0.0
                goal_diff_trend = 0.0
                
                for j, match in enumerate(recent_matches.iterrows()):
                    _, match = match
                    weight = (momentum_window - j) / momentum_window  # More recent = higher weight
                    
                    if match['HomeTeam'] == team:
                        if match['FTR'] == 'H':
                            momentum_score += 3 * weight
                            points_trend += 3
                        elif match['FTR'] == 'D':
                            momentum_score += 1 * weight
                            points_trend += 1
                        goal_diff_trend += (match['FTHG'] - match['FTAG']) * weight
                    else:
                        if match['FTR'] == 'A':
                            momentum_score += 3 * weight
                            points_trend += 3
                        elif match['FTR'] == 'D':
                            momentum_score += 1 * weight
                            points_trend += 1
                        goal_diff_trend += (match['FTAG'] - match['FTHG']) * weight
                
                momentum_score /= momentum_window
                
                # Compare last 5 vs previous 5 matches
                if len(recent_matches) >= 10:
                    last_5_points = sum([
                        3 if (m['HomeTeam'] == team and m['FTR'] == 'H') or (m['AwayTeam'] == team and m['FTR'] == 'A')
                        else 1 if m['FTR'] == 'D' else 0
                        for _, m in recent_matches.tail(5).iterrows()
                    ])
                    
                    prev_5_points = sum([
                        3 if (m['HomeTeam'] == team and m['FTR'] == 'H') or (m['AwayTeam'] == team and m['FTR'] == 'A')
                        else 1 if m['FTR'] == 'D' else 0
                        for _, m in recent_matches.head(5).iterrows()
                    ])
                    
                    points_trend = (last_5_points - prev_5_points) / 5.0
                
                # Store features
                match_idx = current_match.name
                if current_match['HomeTeam'] == team:
                    df.loc[match_idx, 'home_winning_streak'] = win_streak
                    df.loc[match_idx, 'home_losing_streak'] = loss_streak
                    df.loc[match_idx, 'home_momentum_score'] = momentum_score
                    df.loc[match_idx, 'home_points_trend'] = points_trend
                    df.loc[match_idx, 'home_goal_diff_trend'] = goal_diff_trend
                    df.loc[match_idx, 'home_clean_sheet_streak'] = clean_sheet_streak
                else:
                    df.loc[match_idx, 'away_winning_streak'] = win_streak
                    df.loc[match_idx, 'away_losing_streak'] = loss_streak
                    df.loc[match_idx, 'away_momentum_score'] = momentum_score
                    df.loc[match_idx, 'away_points_trend'] = points_trend
                    df.loc[match_idx, 'away_goal_diff_trend'] = goal_diff_trend
                    df.loc[match_idx, 'away_clean_sheet_streak'] = clean_sheet_streak
        
        return df
    
    def _create_league_position_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create league position dynamics features"""
        self.logger.info("Creating league position features")
        
        if not self.config['league_position']['enabled']:
            return df
        
        # Initialize position features
        df['home_current_position'] = 0
        df['away_current_position'] = 0
        df['home_position_change'] = 0.0
        df['away_position_change'] = 0.0
        df['home_distance_from_promotion'] = 0
        df['away_distance_from_promotion'] = 0
        df['home_distance_from_relegation'] = 0
        df['away_distance_from_relegation'] = 0
        df['home_pressure_index'] = 0.0
        df['away_pressure_index'] = 0.0
        
        # This would require actual league table data
        # For now, create placeholder features based on team strength
        
        all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        team_strength = {}
        
        # Calculate team strength
        for team in all_teams:
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
            
            if len(team_matches) > 0:
                wins = ((team_matches['HomeTeam'] == team) & (team_matches['FTR'] == 'H')).sum() + \
                       ((team_matches['AwayTeam'] == team) & (team_matches['FTR'] == 'A')).sum()
                
                draws = ((team_matches['HomeTeam'] == team) & (team_matches['FTR'] == 'D')).sum() + \
                        ((team_matches['AwayTeam'] == team) & (team_matches['FTR'] == 'D')).sum()
                
                total_matches = len(team_matches)
                points = wins * 3 + draws
                strength_score = points / (total_matches * 3) if total_matches > 0 else 0.33
                
                team_strength[team] = strength_score
        
        # Assign positions based on strength (simplified)
        sorted_teams = sorted(team_strength.items(), key=lambda x: x[1], reverse=True)
        team_positions = {team: pos + 1 for pos, (team, _) in enumerate(sorted_teams)}
        
        # Add position features
        df['home_current_position'] = df['HomeTeam'].map(team_positions)
        df['away_current_position'] = df['AwayTeam'].map(team_positions)
        
        # Calculate pressure index (higher for teams near promotion/relegation zones)
        for team in all_teams:
            position = team_positions.get(team, 10)
            total_teams = len(all_teams)
            
            # Distance from promotion zone (top 3)
            promotion_distance = max(0, position - 3)
            
            # Distance from relegation zone (bottom 3)
            relegation_distance = max(0, total_teams - position - 2)
            
            # Pressure index (higher for teams in middle of table)
            pressure_index = 1.0 - abs(position - total_teams/2) / (total_teams/2)
            
            # Update features
            df.loc[df['HomeTeam'] == team, 'home_distance_from_promotion'] = promotion_distance
            df.loc[df['HomeTeam'] == team, 'home_distance_from_relegation'] = relegation_distance
            df.loc[df['HomeTeam'] == team, 'home_pressure_index'] = pressure_index
            
            df.loc[df['AwayTeam'] == team, 'away_distance_from_promotion'] = promotion_distance
            df.loc[df['AwayTeam'] == team, 'away_distance_from_relegation'] = relegation_distance
            df.loc[df['AwayTeam'] == team, 'away_pressure_index'] = pressure_index
        
        return df
    
    def _create_league_specific_features(self, df: pd.DataFrame, league_code: str) -> pd.DataFrame:
        """Create league-specific features"""
        self.logger.info("Creating league-specific features")
        
        if not league_code or league_code not in self.config['league_specific']:
            return df
        
        league_config = self.config['league_specific'][league_code]
        
        # League averages
        df['league_avg_goals'] = league_config['avg_goals']
        df['league_home_advantage'] = league_config['home_advantage']
        
        # League-specific adjustments
        df['home_advantage_adjusted'] = df['home_advantage'] - df['league_home_advantage']
        df['goals_vs_league_avg'] = df['total_goals'] - df['league_avg_goals']
        
        # League competitiveness (variance in results)
        df['league_competitiveness'] = df['total_goals'].var()
        
        return df
    
    def _create_cross_league_features(self, df: pd.DataFrame, league_code: str) -> pd.DataFrame:
        """Create cross-league transfer features"""
        self.logger.info("Creating cross-league features")
        
        if not self.config['cross_league']['enabled']:
            return df
        
        # This would typically involve comparing with reference league data
        # For now, create placeholder features
        
        df['cross_league_home_advantage'] = 0.15  # Placeholder
        df['cross_league_avg_goals'] = 2.5  # Placeholder
        df['cross_league_draw_rate'] = 0.3  # Placeholder
        
        # Transfer learning features
        transfer_ratio = self.config['cross_league']['transfer_ratio']
        df['transfer_learning_factor'] = transfer_ratio
        
        return df
    
    def _create_advanced_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced statistical features"""
        self.logger.info("Creating advanced statistical features")
        
        # Goal rate features
        df['home_goal_rate'] = df['home_goals_scored'] / (df['home_goals_scored'] + df['home_goals_conceded'] + 1e-10)
        df['away_goal_rate'] = df['away_goals_scored'] / (df['away_goals_scored'] + df['away_goals_conceded'] + 1e-10)
        
        # Defensive strength
        df['home_defensive_strength'] = 1 / (df['home_goals_conceded'] + 1e-10)
        df['away_defensive_strength'] = 1 / (df['away_goals_conceded'] + 1e-10)
        
        # Offensive strength
        df['home_offensive_strength'] = df['home_goals_scored']
        df['away_offensive_strength'] = df['away_goals_scored']
        
        # Match balance
        df['match_balance'] = abs(df['home_offensive_strength'] - df['away_offensive_strength'])
        
        # Goal difference ratio
        df['goal_difference_ratio'] = df['goal_difference'] / (df['total_goals'] + 1e-10)
        
        return df
    
    def _create_team_embedding_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create team embedding features"""
        self.logger.info("Creating team embedding features")
        
        # Team strength features
        all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        
        # Calculate team strength based on historical performance
        team_strength = {}
        for team in all_teams:
            team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)]
            
            if len(team_matches) > 0:
                # Calculate win rate
                wins = ((team_matches['HomeTeam'] == team) & (team_matches['FTR'] == 'H')).sum() + \
                       ((team_matches['AwayTeam'] == team) & (team_matches['FTR'] == 'A')).sum()
                
                draws = ((team_matches['HomeTeam'] == team) & (team_matches['FTR'] == 'D')).sum() + \
                        ((team_matches['AwayTeam'] == team) & (team_matches['FTR'] == 'D')).sum()
                
                total_matches = len(team_matches)
                win_rate = wins / total_matches
                draw_rate = draws / total_matches
                
                # Calculate average goals
                home_goals = team_matches[team_matches['HomeTeam'] == team]['FTHG'].sum()
                away_goals = team_matches[team_matches['AwayTeam'] == team]['FTAG'].sum()
                avg_goals = (home_goals + away_goals) / total_matches
                
                team_strength[team] = {
                    'win_rate': win_rate,
                    'draw_rate': draw_rate,
                    'avg_goals': avg_goals,
                    'strength_score': win_rate * 3 + draw_rate * 1
                }
            else:
                team_strength[team] = {
                    'win_rate': 0.33,
                    'draw_rate': 0.33,
                    'avg_goals': 1.0,
                    'strength_score': 1.0
                }
        
        # Add team strength features
        df['home_team_strength'] = df['HomeTeam'].map(lambda x: team_strength[x]['strength_score'])
        df['away_team_strength'] = df['AwayTeam'].map(lambda x: team_strength[x]['strength_score'])
        df['team_strength_diff'] = df['home_team_strength'] - df['away_team_strength']
        
        df['home_team_win_rate'] = df['HomeTeam'].map(lambda x: team_strength[x]['win_rate'])
        df['away_team_win_rate'] = df['AwayTeam'].map(lambda x: team_strength[x]['win_rate'])
        
        df['home_team_avg_goals'] = df['HomeTeam'].map(lambda x: team_strength[x]['avg_goals'])
        df['away_team_avg_goals'] = df['AwayTeam'].map(lambda x: team_strength[x]['avg_goals'])
        
        return df
    
    def _finalize_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Finalize features with scaling and selection"""
        self.logger.info("Finalizing features")
        
        # Remove rows with too many missing values
        missing_threshold = 0.5
        df = df.dropna(thresh=len(df.columns) * (1 - missing_threshold))
        
        # Fill remaining missing values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        
        # Remove infinite values
        df = df.replace([np.inf, -np.inf], np.nan)
        df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
        
        # Scale numerical features (exclude target columns)
        target_cols = [col for col in df.columns if col.startswith('target_')]
        feature_cols = [col for col in numerical_cols if col not in target_cols]
        
        if feature_cols:
            scaler = StandardScaler()
            df[feature_cols] = scaler.fit_transform(df[feature_cols])
            self.scalers['numerical'] = scaler
        
        # Create target variable
        if 'FTR' in df.columns:
            df['target'] = df['FTR'].map({'A': 0, 'D': 1, 'H': 2})
        
        return df
    
    def get_feature_importance(self, df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
        """Get feature importance for non-major leagues"""
        if target_col not in df.columns:
            return pd.DataFrame()
        
        # Calculate correlation with target
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        numerical_cols = [col for col in numerical_cols if col != target_col]
        
        correlations = df[numerical_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
        
        feature_importance = pd.DataFrame({
            'feature': correlations.index,
            'correlation': correlations.values,
            'importance': correlations.values / correlations.max()
        })
        
        return feature_importance
    
    def select_top_features(self, df: pd.DataFrame, target_col: str = 'target', n_features: int = 50) -> List[str]:
        """Select top N features based on importance"""
        feature_importance = self.get_feature_importance(df, target_col)
        
        if feature_importance.empty:
            return []
        
        top_features = feature_importance.head(n_features)['feature'].tolist()
        return top_features
    
    def save_feature_engineer(self, filepath: str):
        """Save feature engineer state"""
        import joblib
        
        state = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'config': self.config
        }
        
        joblib.dump(state, filepath)
        self.logger.info(f"Feature engineer state saved to {filepath}")
    
    def load_feature_engineer(self, filepath: str):
        """Load feature engineer state"""
        import joblib
        
        state = joblib.load(filepath)
        self.scalers = state['scalers']
        self.encoders = state['encoders']
        self.config = state['config']
        
        self.logger.info(f"Feature engineer state loaded from {filepath}")

# Example usage
def main():
    """Example usage of NonMajorLeagueFeatureEngineer"""
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'HomeTeam': ['Team A', 'Team B', 'Team C'] * 34,
        'AwayTeam': ['Team B', 'Team C', 'Team A'] * 34,
        'FTHG': np.random.randint(0, 5, 100),
        'FTAG': np.random.randint(0, 5, 100),
        'FTR': np.random.choice(['H', 'D', 'A'], 100)
    })
    
    # Initialize feature engineer
    feature_engineer = NonMajorLeagueFeatureEngineer()
    
    # Create features
    features_df = feature_engineer.create_all_features(sample_data, 'E1')
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Features data shape: {features_df.shape}")
    print(f"Features created: {len(features_df.columns)}")
    
    # Get feature importance
    feature_importance = feature_engineer.get_feature_importance(features_df)
    print(f"\nTop 10 features by importance:")
    print(feature_importance.head(10))
    
    # Select top features
    top_features = feature_engineer.select_top_features(features_df, n_features=20)
    print(f"\nTop 20 features: {top_features}")
    
    # Save feature engineer state
    feature_engineer.save_feature_engineer('feature_engineer.pkl')

if __name__ == "__main__":
    main()
