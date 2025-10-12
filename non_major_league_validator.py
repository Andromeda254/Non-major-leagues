import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeagueValidator:
    """
    Comprehensive data validation system for non-major soccer leagues
    
    Key Features:
    - Data quality assessment
    - Statistical validation
    - Temporal consistency checks
    - Cross-league comparison
    - Missing data analysis
    - Outlier detection
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize validator with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.validation_results = {}
        
    def setup_logging(self):
        """Setup logging for validation"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load validation configuration"""
        if config is None:
            self.config = {
                'quality_thresholds': {
                    'min_matches_per_team': 10,
                    'min_seasons': 2,
                    'max_missing_percentage': 0.3,
                    'min_teams_per_league': 16,
                    'max_team_changes_per_season': 0.5
                },
                'statistical_tests': {
                    'normality_test': 'shapiro',  # 'shapiro', 'kstest', 'normaltest'
                    'correlation_threshold': 0.8,
                    'variance_threshold': 0.1,
                    'outlier_threshold': 3.0  # Z-score threshold
                },
                'temporal_checks': {
                    'max_gap_days': 30,
                    'min_matches_per_month': 5,
                    'season_consistency': True
                },
                'cross_league': {
                    'enabled': True,
                    'reference_leagues': ['EPL', 'LaLiga', 'Bundesliga'],
                    'similarity_threshold': 0.7
                }
            }
        else:
            self.config = config
    
    def validate_dataset(self, df: pd.DataFrame, league_code: str = None) -> Dict[str, Any]:
        """
        Comprehensive dataset validation
        
        Args:
            df: Dataset to validate
            league_code: League identifier
            
        Returns:
            Dictionary with validation results
        """
        self.logger.info(f"Starting comprehensive validation for dataset with {len(df)} rows")
        
        validation_results = {
            'dataset_info': self._get_dataset_info(df),
            'data_quality': self._validate_data_quality(df),
            'statistical_validation': self._validate_statistics(df),
            'temporal_validation': self._validate_temporal_consistency(df),
            'team_validation': self._validate_teams(df),
            'score_validation': self._validate_scores(df),
            'missing_data_analysis': self._analyze_missing_data(df),
            'outlier_analysis': self._analyze_outliers(df),
            'cross_league_comparison': self._compare_with_reference_leagues(df, league_code),
            'overall_score': 0.0,
            'recommendations': []
        }
        
        # Calculate overall validation score
        validation_results['overall_score'] = self._calculate_overall_score(validation_results)
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_recommendations(validation_results)
        
        self.validation_results = validation_results
        return validation_results
    
    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        info = {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'memory_usage': df.memory_usage(deep=True).sum(),
            'data_types': df.dtypes.value_counts().to_dict(),
            'date_range': None,
            'teams': None,
            'seasons': None
        }
        
        # Date range
        if 'Date' in df.columns:
            info['date_range'] = {
                'start': df['Date'].min(),
                'end': df['Date'].max(),
                'span_days': (df['Date'].max() - df['Date'].min()).days
            }
        
        # Teams
        if 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
            all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
            info['teams'] = {
                'total': len(all_teams),
                'home_teams': len(df['HomeTeam'].unique()),
                'away_teams': len(df['AwayTeam'].unique()),
                'team_list': list(all_teams)
            }
        
        # Seasons
        if 'Date' in df.columns:
            df_copy = df.copy()
            df_copy['year'] = df_copy['Date'].dt.year
            info['seasons'] = {
                'total': len(df_copy['year'].unique()),
                'years': sorted(df_copy['year'].unique().tolist())
            }
        
        return info
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate overall data quality"""
        quality_results = {
            'completeness': self._check_completeness(df),
            'consistency': self._check_consistency(df),
            'accuracy': self._check_accuracy(df),
            'reliability': self._check_reliability(df),
            'score': 0.0
        }
        
        # Calculate quality score
        scores = [quality_results[key] for key in ['completeness', 'consistency', 'accuracy', 'reliability'] 
                 if isinstance(quality_results[key], (int, float))]
        quality_results['score'] = np.mean(scores) if scores else 0.0
        
        return quality_results
    
    def _check_completeness(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data completeness"""
        completeness = {
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'complete_rows': len(df.dropna()),
            'complete_percentage': (len(df.dropna()) / len(df)) * 100,
            'score': 0.0
        }
        
        # Calculate completeness score
        complete_pct = completeness['complete_percentage']
        if complete_pct >= 90:
            completeness['score'] = 1.0
        elif complete_pct >= 80:
            completeness['score'] = 0.8
        elif complete_pct >= 70:
            completeness['score'] = 0.6
        elif complete_pct >= 60:
            completeness['score'] = 0.4
        else:
            completeness['score'] = 0.2
        
        return completeness
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data consistency"""
        consistency = {
            'duplicate_rows': df.duplicated().sum(),
            'inconsistent_results': 0,
            'team_name_consistency': True,
            'date_consistency': True,
            'score': 0.0
        }
        
        # Check for inconsistent results
        if 'FTR' in df.columns and 'FTHG' in df.columns and 'FTAG' in df.columns:
            df_copy = df.copy()
            df_copy['calculated_FTR'] = df_copy.apply(
                lambda row: 'H' if row['FTHG'] > row['FTAG'] 
                           else 'A' if row['FTHG'] < row['FTAG'] 
                           else 'D', axis=1
            )
            consistency['inconsistent_results'] = (df_copy['FTR'] != df_copy['calculated_FTR']).sum()
        
        # Check team name consistency
        if 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
            home_teams = set(df['HomeTeam'].unique())
            away_teams = set(df['AwayTeam'].unique())
            consistency['team_name_consistency'] = home_teams == away_teams
        
        # Check date consistency
        if 'Date' in df.columns:
            consistency['date_consistency'] = df['Date'].is_monotonic_increasing
        
        # Calculate consistency score
        issues = sum([
            consistency['duplicate_rows'] > 0,
            consistency['inconsistent_results'] > 0,
            not consistency['team_name_consistency'],
            not consistency['date_consistency']
        ])
        
        consistency['score'] = max(0, 1.0 - (issues * 0.25))
        
        return consistency
    
    def _check_accuracy(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data accuracy"""
        accuracy = {
            'valid_dates': 0,
            'valid_scores': 0,
            'valid_teams': 0,
            'score': 0.0
        }
        
        # Check date accuracy
        if 'Date' in df.columns:
            accuracy['valid_dates'] = df['Date'].notna().sum()
        
        # Check score accuracy
        if 'FTHG' in df.columns and 'FTAG' in df.columns:
            valid_scores = ((df['FTHG'] >= 0) & (df['FTHG'] <= 20) & 
                          (df['FTAG'] >= 0) & (df['FTAG'] <= 20)).sum()
            accuracy['valid_scores'] = valid_scores
        
        # Check team accuracy
        if 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
            valid_teams = (df['HomeTeam'].notna() & df['AwayTeam'].notna() & 
                          (df['HomeTeam'] != '') & (df['AwayTeam'] != '')).sum()
            accuracy['valid_teams'] = valid_teams
        
        # Calculate accuracy score
        total_checks = len(df)
        valid_checks = accuracy['valid_dates'] + accuracy['valid_scores'] + accuracy['valid_teams']
        accuracy['score'] = valid_checks / (total_checks * 3) if total_checks > 0 else 0.0
        
        return accuracy
    
    def _check_reliability(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check data reliability"""
        reliability = {
            'team_coverage': 0.0,
            'season_coverage': 0.0,
            'match_frequency': 0.0,
            'score': 0.0
        }
        
        # Check team coverage
        if 'HomeTeam' in df.columns and 'AwayTeam' in df.columns:
            all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
            min_teams = self.config['quality_thresholds']['min_teams_per_league']
            reliability['team_coverage'] = min(1.0, len(all_teams) / min_teams)
        
        # Check season coverage
        if 'Date' in df.columns:
            df_copy = df.copy()
            df_copy['year'] = df_copy['Date'].dt.year
            seasons = len(df_copy['year'].unique())
            min_seasons = self.config['quality_thresholds']['min_seasons']
            reliability['season_coverage'] = min(1.0, seasons / min_seasons)
        
        # Check match frequency
        if 'Date' in df.columns:
            df_copy = df.copy()
            df_copy['month'] = df_copy['Date'].dt.to_period('M')
            monthly_matches = df_copy['month'].value_counts()
            min_matches = self.config['temporal_checks']['min_matches_per_month']
            reliability['match_frequency'] = min(1.0, monthly_matches.min() / min_matches)
        
        # Calculate reliability score
        scores = [reliability[key] for key in ['team_coverage', 'season_coverage', 'match_frequency']]
        reliability['score'] = np.mean(scores) if scores else 0.0
        
        return reliability
    
    def _validate_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate statistical properties"""
        stats_results = {
            'goal_distribution': self._analyze_goal_distribution(df),
            'result_distribution': self._analyze_result_distribution(df),
            'home_advantage': self._analyze_home_advantage(df),
            'seasonal_patterns': self._analyze_seasonal_patterns(df),
            'score': 0.0
        }
        
        # Calculate statistical validation score
        scores = [stats_results[key].get('score', 0.0) for key in stats_results.keys() 
                 if isinstance(stats_results[key], dict) and 'score' in stats_results[key]]
        stats_results['score'] = np.mean(scores) if scores else 0.0
        
        return stats_results
    
    def _analyze_goal_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze goal distribution"""
        if 'FTHG' not in df.columns or 'FTAG' not in df.columns:
            return {'score': 0.0, 'error': 'Goal columns not found'}
        
        home_goals = df['FTHG']
        away_goals = df['FTAG']
        total_goals = home_goals + away_goals
        
        analysis = {
            'home_goals_mean': home_goals.mean(),
            'away_goals_mean': away_goals.mean(),
            'total_goals_mean': total_goals.mean(),
            'home_goals_std': home_goals.std(),
            'away_goals_std': away_goals.std(),
            'total_goals_std': total_goals.std(),
            'home_goals_skewness': stats.skew(home_goals),
            'away_goals_skewness': stats.skew(away_goals),
            'total_goals_skewness': stats.skew(total_goals),
            'score': 0.0
        }
        
        # Check if distributions are reasonable
        reasonable_mean = 1.0 <= analysis['total_goals_mean'] <= 4.0
        reasonable_std = 0.5 <= analysis['total_goals_std'] <= 2.0
        reasonable_skewness = abs(analysis['total_goals_skewness']) <= 2.0
        
        analysis['score'] = sum([reasonable_mean, reasonable_std, reasonable_skewness]) / 3.0
        
        return analysis
    
    def _analyze_result_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze result distribution"""
        if 'FTR' not in df.columns:
            return {'score': 0.0, 'error': 'Result column not found'}
        
        result_counts = df['FTR'].value_counts()
        total_matches = len(df)
        
        analysis = {
            'home_win_rate': result_counts.get('H', 0) / total_matches,
            'draw_rate': result_counts.get('D', 0) / total_matches,
            'away_win_rate': result_counts.get('A', 0) / total_matches,
            'score': 0.0
        }
        
        # Check if distribution is reasonable
        reasonable_home = 0.4 <= analysis['home_win_rate'] <= 0.6
        reasonable_draw = 0.2 <= analysis['draw_rate'] <= 0.4
        reasonable_away = 0.2 <= analysis['away_win_rate'] <= 0.4
        
        analysis['score'] = sum([reasonable_home, reasonable_draw, reasonable_away]) / 3.0
        
        return analysis
    
    def _analyze_home_advantage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze home advantage"""
        if 'FTHG' not in df.columns or 'FTAG' not in df.columns:
            return {'score': 0.0, 'error': 'Goal columns not found'}
        
        home_advantage = df['FTHG'] - df['FTAG']
        
        analysis = {
            'mean_home_advantage': home_advantage.mean(),
            'std_home_advantage': home_advantage.std(),
            'positive_home_advantage_rate': (home_advantage > 0).mean(),
            'score': 0.0
        }
        
        # Check if home advantage is reasonable
        reasonable_mean = 0.1 <= analysis['mean_home_advantage'] <= 0.5
        reasonable_rate = 0.45 <= analysis['positive_home_advantage_rate'] <= 0.65
        
        analysis['score'] = sum([reasonable_mean, reasonable_rate]) / 2.0
        
        return analysis
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns"""
        if 'Date' not in df.columns:
            return {'score': 0.0, 'error': 'Date column not found'}
        
        df_copy = df.copy()
        df_copy['month'] = df_copy['Date'].dt.month
        df_copy['season'] = df_copy['month'].apply(lambda x: 'Spring' if x in [3,4,5] 
                                                   else 'Summer' if x in [6,7,8] 
                                                   else 'Autumn' if x in [9,10,11] 
                                                   else 'Winter')
        
        seasonal_goals = df_copy.groupby('season')['FTHG'].sum() + df_copy.groupby('season')['FTAG'].sum()
        seasonal_matches = df_copy.groupby('season').size()
        
        analysis = {
            'seasonal_goal_rates': (seasonal_goals / seasonal_matches).to_dict(),
            'seasonal_match_counts': seasonal_matches.to_dict(),
            'score': 0.0
        }
        
        # Check if seasonal patterns are reasonable
        goal_rates = list(analysis['seasonal_goal_rates'].values())
        if goal_rates:
            analysis['score'] = 1.0 - (np.std(goal_rates) / np.mean(goal_rates)) if np.mean(goal_rates) > 0 else 0.0
        
        return analysis
    
    def _validate_temporal_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate temporal consistency"""
        temporal_results = {
            'date_gaps': self._check_date_gaps(df),
            'match_frequency': self._check_match_frequency(df),
            'season_consistency': self._check_season_consistency(df),
            'score': 0.0
        }
        
        # Calculate temporal score
        scores = [temporal_results[key].get('score', 0.0) for key in temporal_results.keys() 
                 if isinstance(temporal_results[key], dict) and 'score' in temporal_results[key]]
        temporal_results['score'] = np.mean(scores) if scores else 0.0
        
        return temporal_results
    
    def _check_date_gaps(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for large gaps in dates"""
        if 'Date' not in df.columns:
            return {'score': 0.0, 'error': 'Date column not found'}
        
        df_sorted = df.sort_values('Date')
        date_diffs = df_sorted['Date'].diff().dt.days
        
        max_gap = self.config['temporal_checks']['max_gap_days']
        large_gaps = (date_diffs > max_gap).sum()
        
        return {
            'large_gaps': large_gaps,
            'max_gap_days': date_diffs.max(),
            'avg_gap_days': date_diffs.mean(),
            'score': max(0.0, 1.0 - (large_gaps / len(df)))
        }
    
    def _check_match_frequency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check match frequency consistency"""
        if 'Date' not in df.columns:
            return {'score': 0.0, 'error': 'Date column not found'}
        
        df_copy = df.copy()
        df_copy['month'] = df_copy['Date'].dt.to_period('M')
        monthly_counts = df_copy['month'].value_counts()
        
        min_matches = self.config['temporal_checks']['min_matches_per_month']
        low_frequency_months = (monthly_counts < min_matches).sum()
        
        return {
            'monthly_counts': monthly_counts.to_dict(),
            'low_frequency_months': low_frequency_months,
            'avg_matches_per_month': monthly_counts.mean(),
            'score': max(0.0, 1.0 - (low_frequency_months / len(monthly_counts)))
        }
    
    def _check_season_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check season consistency"""
        if 'Date' not in df.columns:
            return {'score': 0.0, 'error': 'Date column not found'}
        
        df_copy = df.copy()
        df_copy['year'] = df_copy['Date'].dt.year
        yearly_counts = df_copy['year'].value_counts()
        
        # Check if all seasons have reasonable number of matches
        min_season_matches = 200  # Minimum matches per season
        low_season_years = (yearly_counts < min_season_matches).sum()
        
        return {
            'yearly_counts': yearly_counts.to_dict(),
            'low_season_years': low_season_years,
            'avg_matches_per_season': yearly_counts.mean(),
            'score': max(0.0, 1.0 - (low_season_years / len(yearly_counts)))
        }
    
    def _validate_teams(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate team data"""
        team_results = {
            'team_coverage': self._check_team_coverage(df),
            'team_consistency': self._check_team_consistency(df),
            'team_changes': self._check_team_changes(df),
            'score': 0.0
        }
        
        # Calculate team validation score
        scores = [team_results[key].get('score', 0.0) for key in team_results.keys() 
                 if isinstance(team_results[key], dict) and 'score' in team_results[key]]
        team_results['score'] = np.mean(scores) if scores else 0.0
        
        return team_results
    
    def _check_team_coverage(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check team coverage"""
        if 'HomeTeam' not in df.columns or 'AwayTeam' not in df.columns:
            return {'score': 0.0, 'error': 'Team columns not found'}
        
        all_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        min_teams = self.config['quality_thresholds']['min_teams_per_league']
        
        return {
            'total_teams': len(all_teams),
            'min_required': min_teams,
            'coverage_rate': len(all_teams) / min_teams,
            'score': min(1.0, len(all_teams) / min_teams)
        }
    
    def _check_team_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check team consistency"""
        if 'HomeTeam' not in df.columns or 'AwayTeam' not in df.columns:
            return {'score': 0.0, 'error': 'Team columns not found'}
        
        home_teams = set(df['HomeTeam'].unique())
        away_teams = set(df['AwayTeam'].unique())
        
        return {
            'home_teams': len(home_teams),
            'away_teams': len(away_teams),
            'teams_in_both': len(home_teams & away_teams),
            'consistency_score': len(home_teams & away_teams) / len(home_teams | away_teams),
            'score': len(home_teams & away_teams) / len(home_teams | away_teams)
        }
    
    def _check_team_changes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check for team changes between seasons"""
        if 'Date' not in df.columns or 'HomeTeam' not in df.columns or 'AwayTeam' not in df.columns:
            return {'score': 0.0, 'error': 'Required columns not found'}
        
        df_copy = df.copy()
        df_copy['year'] = df_copy['Date'].dt.year
        
        yearly_teams = {}
        for year in df_copy['year'].unique():
            year_data = df_copy[df_copy['year'] == year]
            yearly_teams[year] = set(year_data['HomeTeam'].unique()) | set(year_data['AwayTeam'].unique())
        
        # Calculate team change rate
        years = sorted(yearly_teams.keys())
        total_changes = 0
        total_teams = 0
        
        for i in range(1, len(years)):
            prev_teams = yearly_teams[years[i-1]]
            curr_teams = yearly_teams[years[i]]
            
            changes = len(prev_teams.symmetric_difference(curr_teams))
            total_changes += changes
            total_teams += len(prev_teams | curr_teams)
        
        change_rate = total_changes / total_teams if total_teams > 0 else 0
        
        return {
            'total_changes': total_changes,
            'total_teams': total_teams,
            'change_rate': change_rate,
            'max_allowed_rate': self.config['quality_thresholds']['max_team_changes_per_season'],
            'score': max(0.0, 1.0 - change_rate / self.config['quality_thresholds']['max_team_changes_per_season'])
        }
    
    def _validate_scores(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate score data"""
        score_results = {
            'score_range': self._check_score_range(df),
            'score_distribution': self._check_score_distribution(df),
            'score_consistency': self._check_score_consistency(df),
            'score': 0.0
        }
        
        # Calculate score validation score
        scores = [score_results[key].get('score', 0.0) for key in score_results.keys() 
                 if isinstance(score_results[key], dict) and 'score' in score_results[key]]
        score_results['score'] = np.mean(scores) if scores else 0.0
        
        return score_results
    
    def _check_score_range(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check score range validity"""
        if 'FTHG' not in df.columns or 'FTAG' not in df.columns:
            return {'score': 0.0, 'error': 'Score columns not found'}
        
        home_goals = df['FTHG']
        away_goals = df['FTAG']
        
        return {
            'home_min': home_goals.min(),
            'home_max': home_goals.max(),
            'away_min': away_goals.min(),
            'away_max': away_goals.max(),
            'home_invalid': ((home_goals < 0) | (home_goals > 20)).sum(),
            'away_invalid': ((away_goals < 0) | (away_goals > 20)).sum(),
            'score': 1.0 - ((home_goals < 0) | (home_goals > 20) | (away_goals < 0) | (away_goals > 20)).sum() / len(df)
        }
    
    def _check_score_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check score distribution"""
        if 'FTHG' not in df.columns or 'FTAG' not in df.columns:
            return {'score': 0.0, 'error': 'Score columns not found'}
        
        home_goals = df['FTHG']
        away_goals = df['FTAG']
        total_goals = home_goals + away_goals
        
        return {
            'home_mean': home_goals.mean(),
            'away_mean': away_goals.mean(),
            'total_mean': total_goals.mean(),
            'home_std': home_goals.std(),
            'away_std': away_goals.std(),
            'total_std': total_goals.std(),
            'score': 1.0 if 1.0 <= total_goals.mean() <= 4.0 else 0.5
        }
    
    def _check_score_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Check score consistency with results"""
        if 'FTHG' not in df.columns or 'FTAG' not in df.columns or 'FTR' not in df.columns:
            return {'score': 0.0, 'error': 'Required columns not found'}
        
        df_copy = df.copy()
        df_copy['calculated_FTR'] = df_copy.apply(
            lambda row: 'H' if row['FTHG'] > row['FTAG'] 
                       else 'A' if row['FTHG'] < row['FTAG'] 
                       else 'D', axis=1
        )
        
        inconsistent = (df_copy['FTR'] != df_copy['calculated_FTR']).sum()
        
        return {
            'inconsistent_results': inconsistent,
            'total_matches': len(df),
            'consistency_rate': 1.0 - (inconsistent / len(df)),
            'score': 1.0 - (inconsistent / len(df))
        }
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing data patterns"""
        missing_analysis = {
            'missing_by_column': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'missing_patterns': self._identify_missing_patterns(df),
            'recommendations': []
        }
        
        # Generate recommendations
        for col, missing_pct in missing_analysis['missing_percentage'].items():
            if missing_pct > 50:
                missing_analysis['recommendations'].append(f"Consider dropping column {col} (missing: {missing_pct:.1f}%)")
            elif missing_pct > 20:
                missing_analysis['recommendations'].append(f"Consider imputation for column {col} (missing: {missing_pct:.1f}%)")
        
        return missing_analysis
    
    def _identify_missing_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify missing data patterns"""
        patterns = {
            'completely_missing_rows': df.isnull().all(axis=1).sum(),
            'partially_missing_rows': df.isnull().any(axis=1).sum(),
            'missing_by_date': {},
            'missing_by_team': {}
        }
        
        # Missing by date
        if 'Date' in df.columns:
            df_copy = df.copy()
            df_copy['year'] = df_copy['Date'].dt.year
            patterns['missing_by_date'] = df_copy.groupby('year').apply(lambda x: x.isnull().sum().sum()).to_dict()
        
        # Missing by team
        if 'HomeTeam' in df.columns:
            patterns['missing_by_team'] = df.groupby('HomeTeam').apply(lambda x: x.isnull().sum().sum()).to_dict()
        
        return patterns
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze outliers in the data"""
        outlier_analysis = {
            'numerical_outliers': {},
            'categorical_outliers': {},
            'recommendations': []
        }
        
        # Analyze numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if col in ['FTHG', 'FTAG']:  # Focus on goal columns
                outliers = self._detect_outliers(df[col])
                outlier_analysis['numerical_outliers'][col] = {
                    'count': outliers.sum(),
                    'percentage': (outliers.sum() / len(df)) * 100,
                    'indices': df[outliers].index.tolist()
                }
        
        # Generate recommendations
        for col, analysis in outlier_analysis['numerical_outliers'].items():
            if analysis['percentage'] > 5:
                outlier_analysis['recommendations'].append(f"High outlier percentage in {col}: {analysis['percentage']:.1f}%")
        
        return outlier_analysis
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """Detect outliers using IQR method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    
    def _compare_with_reference_leagues(self, df: pd.DataFrame, league_code: str) -> Dict[str, Any]:
        """Compare with reference leagues"""
        if not self.config['cross_league']['enabled']:
            return {'score': 0.0, 'message': 'Cross-league comparison disabled'}
        
        # This would typically compare with reference league data
        # For now, return a placeholder
        return {
            'score': 0.8,  # Placeholder score
            'message': 'Cross-league comparison not implemented',
            'reference_leagues': self.config['cross_league']['reference_leagues']
        }
    
    def _calculate_overall_score(self, validation_results: Dict) -> float:
        """Calculate overall validation score"""
        scores = []
        
        # Extract scores from each validation category
        for category, results in validation_results.items():
            if isinstance(results, dict) and 'score' in results:
                scores.append(results['score'])
        
        return np.mean(scores) if scores else 0.0
    
    def _generate_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        # Data quality recommendations
        if 'data_quality' in validation_results:
            quality = validation_results['data_quality']
            if quality.get('score', 0) < 0.7:
                recommendations.append("Data quality is below acceptable threshold. Consider data cleaning.")
        
        # Missing data recommendations
        if 'missing_data_analysis' in validation_results:
            missing = validation_results['missing_data_analysis']
            recommendations.extend(missing.get('recommendations', []))
        
        # Outlier recommendations
        if 'outlier_analysis' in validation_results:
            outliers = validation_results['outlier_analysis']
            recommendations.extend(outliers.get('recommendations', []))
        
        return recommendations
    
    def generate_validation_report(self, validation_results: Dict, save_path: str = None) -> str:
        """Generate comprehensive validation report"""
        report = []
        report.append("=" * 80)
        report.append("NON-MAJOR LEAGUE DATA VALIDATION REPORT")
        report.append("=" * 80)
        report.append("")
        
        # Dataset info
        if 'dataset_info' in validation_results:
            info = validation_results['dataset_info']
            report.append("DATASET INFORMATION:")
            report.append(f"  Total rows: {info['total_rows']:,}")
            report.append(f"  Total columns: {info['total_columns']}")
            report.append(f"  Memory usage: {info['memory_usage']:,} bytes")
            if info['date_range']:
                report.append(f"  Date range: {info['date_range']['start']} to {info['date_range']['end']}")
                report.append(f"  Span: {info['date_range']['span_days']} days")
            if info['teams']:
                report.append(f"  Teams: {info['teams']['total']}")
            if info['seasons']:
                report.append(f"  Seasons: {info['seasons']['total']}")
            report.append("")
        
        # Overall score
        overall_score = validation_results.get('overall_score', 0.0)
        report.append(f"OVERALL VALIDATION SCORE: {overall_score:.2f}/1.00")
        report.append("")
        
        # Category scores
        report.append("VALIDATION CATEGORIES:")
        categories = ['data_quality', 'statistical_validation', 'temporal_validation', 
                     'team_validation', 'score_validation']
        
        for category in categories:
            if category in validation_results:
                score = validation_results[category].get('score', 0.0)
                report.append(f"  {category.replace('_', ' ').title()}: {score:.2f}/1.00")
        report.append("")
        
        # Recommendations
        if 'recommendations' in validation_results:
            recommendations = validation_results['recommendations']
            if recommendations:
                report.append("RECOMMENDATIONS:")
                for i, rec in enumerate(recommendations, 1):
                    report.append(f"  {i}. {rec}")
                report.append("")
        
        # Detailed results
        report.append("DETAILED RESULTS:")
        report.append("-" * 40)
        
        for category, results in validation_results.items():
            if isinstance(results, dict) and category != 'recommendations':
                report.append(f"\n{category.replace('_', ' ').title()}:")
                for key, value in results.items():
                    if key != 'score':
                        if isinstance(value, dict):
                            report.append(f"  {key}:")
                            for subkey, subvalue in value.items():
                                report.append(f"    {subkey}: {subvalue}")
                        else:
                            report.append(f"  {key}: {value}")
        
        report.append("")
        report.append("=" * 80)
        
        report_text = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report_text)
            self.logger.info(f"Validation report saved to {save_path}")
        
        return report_text
    
    def plot_validation_summary(self, validation_results: Dict, save_path: str = None):
        """Plot validation summary"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Non-Major League Data Validation Summary', fontsize=16)
        
        # Overall score
        overall_score = validation_results.get('overall_score', 0.0)
        axes[0, 0].bar(['Overall Score'], [overall_score], color='green' if overall_score > 0.7 else 'red')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].set_title('Overall Validation Score')
        
        # Category scores
        categories = ['data_quality', 'statistical_validation', 'temporal_validation', 
                     'team_validation', 'score_validation']
        scores = []
        labels = []
        
        for category in categories:
            if category in validation_results:
                score = validation_results[category].get('score', 0.0)
                scores.append(score)
                labels.append(category.replace('_', ' ').title())
        
        axes[0, 1].bar(labels, scores, color=['green' if s > 0.7 else 'orange' if s > 0.5 else 'red' for s in scores])
        axes[0, 1].set_ylim(0, 1)
        axes[0, 1].set_title('Category Scores')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Missing data
        if 'missing_data_analysis' in validation_results:
            missing = validation_results['missing_data_analysis']['missing_percentage']
            if missing:
                axes[1, 0].bar(missing.keys(), missing.values(), color='red')
                axes[1, 0].set_title('Missing Data Percentage')
                axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Goal distribution
        if 'statistical_validation' in validation_results:
            goal_dist = validation_results['statistical_validation'].get('goal_distribution', {})
            if goal_dist:
                goals_data = [goal_dist.get('home_goals_mean', 0), goal_dist.get('away_goals_mean', 0)]
                axes[1, 1].bar(['Home Goals', 'Away Goals'], goals_data, color=['blue', 'orange'])
                axes[1, 1].set_title('Average Goals per Match')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Validation plot saved to {save_path}")
        
        plt.show()

# Example usage
def main():
    """Example usage of NonMajorLeagueValidator"""
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'HomeTeam': ['Team A', 'Team B', 'Team C'] * 34,
        'AwayTeam': ['Team B', 'Team C', 'Team A'] * 34,
        'FTHG': np.random.randint(0, 5, 100),
        'FTAG': np.random.randint(0, 5, 100),
        'FTR': np.random.choice(['H', 'D', 'A'], 100)
    })
    
    # Initialize validator
    validator = NonMajorLeagueValidator()
    
    # Validate dataset
    results = validator.validate_dataset(sample_data, 'E1')
    
    # Generate report
    report = validator.generate_validation_report(results, 'validation_report.txt')
    print(report)
    
    # Plot results
    validator.plot_validation_summary(results, 'validation_summary.png')

if __name__ == "__main__":
    main()
