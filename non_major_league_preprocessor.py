import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import KNNImputer
import warnings
warnings.filterwarnings('ignore')

class NonMajorLeaguePreprocessor:
    """
    Advanced preprocessing pipeline for non-major soccer leagues
    
    Key Features:
    - Robust missing data handling
    - Outlier detection and treatment
    - Feature scaling and normalization
    - Data augmentation for limited datasets
    - Cross-league standardization
    """
    
    def __init__(self, config: Dict = None):
        """
        Initialize preprocessor with configuration
        
        Args:
            config: Configuration dictionary
        """
        self.setup_logging()
        self.load_config(config)
        self.scalers = {}
        self.encoders = {}
        self.imputers = {}
        
    def setup_logging(self):
        """Setup logging for preprocessing"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
    def load_config(self, config: Dict):
        """Load preprocessing configuration"""
        if config is None:
            self.config = {
                'missing_data': {
                    'strategy': 'knn',  # 'mean', 'median', 'knn', 'drop'
                    'max_missing_percentage': 0.3,
                    'knn_neighbors': 5
                },
                'outliers': {
                    'method': 'iqr',  # 'iqr', 'zscore', 'isolation_forest'
                    'threshold': 1.5,
                    'treat_method': 'cap'  # 'cap', 'remove', 'transform'
                },
                'scaling': {
                    'method': 'standard',  # 'standard', 'minmax', 'robust'
                    'fit_on_train': True
                },
                'augmentation': {
                    'enabled': True,
                    'noise_factor': 0.05,
                    'synthetic_samples': 100
                },
                'validation': {
                    'min_matches_per_team': 10,
                    'min_seasons': 2,
                    'max_team_changes': 0.5
                }
            }
        else:
            self.config = config
    
    def preprocess_pipeline(self, df: pd.DataFrame, target_column: str = 'FTR') -> pd.DataFrame:
        """
        Complete preprocessing pipeline for non-major league data
        
        Args:
            df: Raw match data
            target_column: Target variable column name
            
        Returns:
            Preprocessed DataFrame
        """
        self.logger.info("Starting preprocessing pipeline")
        
        # 1. Initial data validation
        df = self._validate_input_data(df, target_column)
        
        # 2. Handle missing data
        df = self._handle_missing_data(df)
        
        # 3. Detect and treat outliers
        df = self._detect_and_treat_outliers(df)
        
        # 4. Feature engineering
        df = self._engineer_basic_features(df)
        
        # 5. Encode categorical variables
        df = self._encode_categorical_variables(df)
        
        # 6. Scale numerical features
        df = self._scale_numerical_features(df)
        
        # 7. Data augmentation (if enabled)
        if self.config['augmentation']['enabled']:
            df = self._augment_data(df)
        
        # 8. Final validation
        df = self._final_validation(df)
        
        self.logger.info(f"Preprocessing complete. Final dataset: {len(df)} rows, {len(df.columns)} columns")
        return df
    
    def _validate_input_data(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Validate input data structure and content"""
        self.logger.info("Validating input data")
        
        # Check required columns
        required_columns = ['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Remove completely empty rows
        initial_rows = len(df)
        df = df.dropna(how='all')
        if len(df) < initial_rows:
            self.logger.warning(f"Removed {initial_rows - len(df)} completely empty rows")
        
        # Validate date column
        df = self._validate_date_column(df)
        
        # Validate team columns
        df = self._validate_team_columns(df)
        
        # Validate score columns
        df = self._validate_score_columns(df)
        
        # Validate result column
        df = self._validate_result_column(df)
        
        self.logger.info(f"Data validation complete. Valid rows: {len(df)}")
        return df
    
    def _validate_date_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and standardize date column"""
        try:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            invalid_dates = df['Date'].isnull().sum()
            
            if invalid_dates > 0:
                self.logger.warning(f"Found {invalid_dates} invalid dates, removing them")
                df = df.dropna(subset=['Date'])
            
            # Sort by date
            df = df.sort_values('Date').reset_index(drop=True)
            
        except Exception as e:
            self.logger.error(f"Error validating dates: {e}")
            raise
        
        return df
    
    def _validate_team_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate team name columns"""
        # Remove rows with missing team names
        df = df.dropna(subset=['HomeTeam', 'AwayTeam'])
        
        # Standardize team names
        df['HomeTeam'] = df['HomeTeam'].astype(str).str.strip()
        df['AwayTeam'] = df['AwayTeam'].astype(str).str.strip()
        
        # Remove rows with empty team names
        df = df[(df['HomeTeam'] != '') & (df['AwayTeam'] != '')]
        
        # Log team statistics
        unique_teams = set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique())
        self.logger.info(f"Found {len(unique_teams)} unique teams")
        
        return df
    
    def _validate_score_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate score columns"""
        # Convert to numeric
        df['FTHG'] = pd.to_numeric(df['FTHG'], errors='coerce')
        df['FTAG'] = pd.to_numeric(df['FTAG'], errors='coerce')
        
        # Remove rows with missing scores
        df = df.dropna(subset=['FTHG', 'FTAG'])
        
        # Remove unrealistic scores
        df = df[(df['FTHG'] >= 0) & (df['FTAG'] >= 0)]
        df = df[(df['FTHG'] <= 20) & (df['FTAG'] <= 20)]
        
        return df
    
    def _validate_result_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate result column"""
        # Standardize result values
        df['FTR'] = df['FTR'].astype(str).str.upper().str.strip()
        
        # Keep only valid results
        valid_results = ['H', 'D', 'A']
        df = df[df['FTR'].isin(valid_results)]
        
        # Log result distribution
        result_counts = df['FTR'].value_counts()
        self.logger.info(f"Result distribution: {result_counts.to_dict()}")
        
        return df
    
    def _handle_missing_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing data using configured strategy"""
        self.logger.info("Handling missing data")
        
        strategy = self.config['missing_data']['strategy']
        max_missing = self.config['missing_data']['max_missing_percentage']
        
        # Calculate missing percentages
        missing_percentages = df.isnull().sum() / len(df)
        
        # Identify columns with high missing percentages
        high_missing_cols = missing_percentages[missing_percentages > max_missing].index.tolist()
        
        if high_missing_cols:
            self.logger.warning(f"Columns with high missing data: {high_missing_cols}")
            # Drop columns with too much missing data
            df = df.drop(columns=high_missing_cols)
        
        # Handle remaining missing data
        if strategy == 'knn':
            df = self._impute_with_knn(df)
        elif strategy == 'mean':
            df = self._impute_with_mean(df)
        elif strategy == 'median':
            df = self._impute_with_median(df)
        elif strategy == 'drop':
            df = df.dropna()
        
        self.logger.info(f"Missing data handling complete. Remaining missing values: {df.isnull().sum().sum()}")
        return df
    
    def _impute_with_knn(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using KNN"""
        # Identify numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if numerical_cols:
            # Use KNN imputation for numerical columns
            imputer = KNNImputer(n_neighbors=self.config['missing_data']['knn_neighbors'])
            df[numerical_cols] = imputer.fit_transform(df[numerical_cols])
            
            # Store imputer for later use
            self.imputers['knn'] = imputer
        
        # For categorical columns, use mode
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                df[col] = df[col].fillna(mode_value)
        
        return df
    
    def _impute_with_mean(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using mean"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                mean_value = df[col].mean()
                df[col] = df[col].fillna(mean_value)
        
        return df
    
    def _impute_with_median(self, df: pd.DataFrame) -> pd.DataFrame:
        """Impute missing values using median"""
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
        
        return df
    
    def _detect_and_treat_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detect and treat outliers in numerical columns"""
        self.logger.info("Detecting and treating outliers")
        
        method = self.config['outliers']['method']
        threshold = self.config['outliers']['threshold']
        treat_method = self.config['outliers']['treat_method']
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in numerical_cols:
            if col in ['FTHG', 'FTAG']:  # Don't treat score columns as outliers
                continue
                
            outliers = self._detect_outliers(df[col], method, threshold)
            
            if outliers.sum() > 0:
                self.logger.info(f"Found {outliers.sum()} outliers in {col}")
                
                if treat_method == 'cap':
                    df[col] = self._cap_outliers(df[col], outliers)
                elif treat_method == 'remove':
                    df = df[~outliers]
                elif treat_method == 'transform':
                    df[col] = self._transform_outliers(df[col], outliers)
        
        return df
    
    def _detect_outliers(self, series: pd.Series, method: str, threshold: float) -> pd.Series:
        """Detect outliers using specified method"""
        if method == 'iqr':
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            return (series < lower_bound) | (series > upper_bound)
        
        elif method == 'zscore':
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > threshold
        
        else:
            return pd.Series([False] * len(series))
    
    def _cap_outliers(self, series: pd.Series, outliers: pd.Series) -> pd.Series:
        """Cap outliers at 95th and 5th percentiles"""
        lower_bound = series.quantile(0.05)
        upper_bound = series.quantile(0.95)
        
        series_capped = series.copy()
        series_capped[series < lower_bound] = lower_bound
        series_capped[series > upper_bound] = upper_bound
        
        return series_capped
    
    def _transform_outliers(self, series: pd.Series, outliers: pd.Series) -> pd.Series:
        """Transform outliers using log transformation"""
        series_transformed = series.copy()
        
        # Add 1 to avoid log(0)
        series_transformed[outliers] = np.log1p(series_transformed[outliers])
        
        return series_transformed
    
    def _engineer_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer basic features for non-major leagues"""
        self.logger.info("Engineering basic features")
        
        # 1. Match outcome features
        df['home_win'] = (df['FTR'] == 'H').astype(int)
        df['draw'] = (df['FTR'] == 'D').astype(int)
        df['away_win'] = (df['FTR'] == 'A').astype(int)
        
        # 2. Goal difference
        df['goal_difference'] = df['FTHG'] - df['FTAG']
        df['total_goals'] = df['FTHG'] + df['FTAG']
        
        # 3. Match intensity (goals per match)
        df['match_intensity'] = df['total_goals'] / 2  # Average goals per team
        
        # 4. Home advantage indicator
        df['home_advantage'] = df['FTHG'] - df['FTAG']
        
        # 5. Season features
        df['year'] = df['Date'].dt.year
        df['month'] = df['Date'].dt.month
        df['day_of_week'] = df['Date'].dt.dayofweek
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        
        # 6. Match number in season (approximate)
        df['match_number'] = df.groupby('year').cumcount() + 1
        
        # 7. Team performance features (rolling averages)
        df = self._add_rolling_features(df)
        
        # 8. League-specific features
        df = self._add_league_features(df)
        
        return df
    
    def _add_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add rolling average features for team performance"""
        # Sort by date and team
        df = df.sort_values(['Date', 'HomeTeam']).reset_index(drop=True)
        
        # Initialize rolling features
        df['home_goals_avg_5'] = 0.0
        df['away_goals_avg_5'] = 0.0
        df['home_conceded_avg_5'] = 0.0
        df['away_conceded_avg_5'] = 0.0
        
        # Calculate rolling averages for each team
        for team in df['HomeTeam'].unique():
            team_matches = df[df['HomeTeam'] == team].copy()
            if len(team_matches) > 1:
                # Rolling averages for home team
                df.loc[df['HomeTeam'] == team, 'home_goals_avg_5'] = \
                    team_matches['FTHG'].rolling(window=5, min_periods=1).mean().values
                df.loc[df['HomeTeam'] == team, 'home_conceded_avg_5'] = \
                    team_matches['FTAG'].rolling(window=5, min_periods=1).mean().values
        
        for team in df['AwayTeam'].unique():
            team_matches = df[df['AwayTeam'] == team].copy()
            if len(team_matches) > 1:
                # Rolling averages for away team
                df.loc[df['AwayTeam'] == team, 'away_goals_avg_5'] = \
                    team_matches['FTAG'].rolling(window=5, min_periods=1).mean().values
                df.loc[df['AwayTeam'] == team, 'away_conceded_avg_5'] = \
                    team_matches['FTHG'].rolling(window=5, min_periods=1).mean().values
        
        return df
    
    def _add_league_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add league-specific features"""
        # League average goals per match
        df['league_avg_goals'] = df['total_goals'].mean()
        
        # League home advantage
        df['league_home_advantage'] = df['home_advantage'].mean()
        
        # League goal variance (indicates competitiveness)
        df['league_goal_variance'] = df['total_goals'].var()
        
        # League draw rate
        df['league_draw_rate'] = df['draw'].mean()
        
        return df
    
    def _encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        self.logger.info("Encoding categorical variables")
        
        # Encode team names
        all_teams = list(set(df['HomeTeam'].unique()) | set(df['AwayTeam'].unique()))
        
        # Create team encoders
        home_encoder = LabelEncoder()
        away_encoder = LabelEncoder()
        
        # Fit encoders
        home_encoder.fit(all_teams)
        away_encoder.fit(all_teams)
        
        # Transform team names
        df['home_team_encoded'] = home_encoder.transform(df['HomeTeam'])
        df['away_team_encoded'] = away_encoder.transform(df['AwayTeam'])
        
        # Store encoders for later use
        self.encoders['home_team'] = home_encoder
        self.encoders['away_team'] = away_encoder
        
        # Encode result
        result_encoder = LabelEncoder()
        df['result_encoded'] = result_encoder.fit_transform(df['FTR'])
        self.encoders['result'] = result_encoder
        
        return df
    
    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Scale numerical features"""
        self.logger.info("Scaling numerical features")
        
        method = self.config['scaling']['method']
        
        # Identify numerical columns to scale
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and encoded columns from scaling
        exclude_cols = ['home_team_encoded', 'away_team_encoded', 'result_encoded', 'FTR']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if method == 'standard':
            scaler = StandardScaler()
        elif method == 'minmax':
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
        elif method == 'robust':
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
        else:
            scaler = StandardScaler()
        
        # Scale numerical features
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        
        # Store scaler for later use
        self.scalers['numerical'] = scaler
        
        return df
    
    def _augment_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Augment data for non-major leagues with limited data"""
        self.logger.info("Augmenting data")
        
        if len(df) < 1000:  # Only augment if dataset is small
            noise_factor = self.config['augmentation']['noise_factor']
            synthetic_samples = self.config['augmentation']['synthetic_samples']
            
            # Create synthetic samples by adding noise
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['home_team_encoded', 'away_team_encoded', 'result_encoded']
            numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
            
            synthetic_data = []
            
            for _ in range(synthetic_samples):
                # Randomly select a row
                random_idx = np.random.randint(0, len(df))
                base_row = df.iloc[random_idx].copy()
                
                # Add noise to numerical features
                for col in numerical_cols:
                    if col in base_row:
                        noise = np.random.normal(0, noise_factor * base_row[col])
                        base_row[col] += noise
                
                synthetic_data.append(base_row)
            
            # Combine original and synthetic data
            synthetic_df = pd.DataFrame(synthetic_data)
            df = pd.concat([df, synthetic_df], ignore_index=True)
            
            self.logger.info(f"Data augmentation complete. Added {synthetic_samples} synthetic samples")
        
        return df
    
    def _final_validation(self, df: pd.DataFrame) -> pd.DataFrame:
        """Final validation of preprocessed data"""
        self.logger.info("Performing final validation")
        
        # Check for remaining missing values
        missing_values = df.isnull().sum().sum()
        if missing_values > 0:
            self.logger.warning(f"Remaining missing values: {missing_values}")
        
        # Check for infinite values
        infinite_values = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        if infinite_values > 0:
            self.logger.warning(f"Found {infinite_values} infinite values")
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(df.median())
        
        # Check data types
        self.logger.info(f"Data types: {df.dtypes.value_counts().to_dict()}")
        
        # Check data shape
        self.logger.info(f"Final data shape: {df.shape}")
        
        return df
    
    def transform_new_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors"""
        self.logger.info("Transforming new data")
        
        # Apply same preprocessing steps
        df = self._validate_input_data(df, 'FTR')
        df = self._handle_missing_data(df)
        df = self._detect_and_treat_outliers(df)
        df = self._engineer_basic_features(df)
        
        # Use fitted encoders
        if 'home_team' in self.encoders:
            df['home_team_encoded'] = self.encoders['home_team'].transform(df['HomeTeam'])
        if 'away_team' in self.encoders:
            df['away_team_encoded'] = self.encoders['away_team'].transform(df['AwayTeam'])
        if 'result' in self.encoders:
            df['result_encoded'] = self.encoders['result'].transform(df['FTR'])
        
        # Use fitted scalers
        if 'numerical' in self.scalers:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            exclude_cols = ['home_team_encoded', 'away_team_encoded', 'result_encoded']
            numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
            df[numerical_cols] = self.scalers['numerical'].transform(df[numerical_cols])
        
        return df
    
    def get_feature_importance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Get feature importance for non-major leagues"""
        # Calculate correlation with target
        target_col = 'result_encoded' if 'result_encoded' in df.columns else 'FTR'
        
        if target_col in df.columns:
            numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            correlations = df[numerical_cols].corrwith(df[target_col]).abs().sort_values(ascending=False)
            
            feature_importance = pd.DataFrame({
                'feature': correlations.index,
                'correlation': correlations.values,
                'importance': correlations.values / correlations.max()
            })
            
            return feature_importance
        
        return pd.DataFrame()
    
    def save_preprocessors(self, filepath: str):
        """Save fitted preprocessors"""
        import joblib
        
        preprocessors = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'imputers': self.imputers,
            'config': self.config
        }
        
        joblib.dump(preprocessors, filepath)
        self.logger.info(f"Preprocessors saved to {filepath}")
    
    def load_preprocessors(self, filepath: str):
        """Load fitted preprocessors"""
        import joblib
        
        preprocessors = joblib.load(filepath)
        self.scalers = preprocessors['scalers']
        self.encoders = preprocessors['encoders']
        self.imputers = preprocessors['imputers']
        self.config = preprocessors['config']
        
        self.logger.info(f"Preprocessors loaded from {filepath}")

# Example usage
def main():
    """Example usage of NonMajorLeaguePreprocessor"""
    
    # Create sample data
    sample_data = pd.DataFrame({
        'Date': pd.date_range('2023-01-01', periods=100, freq='D'),
        'HomeTeam': ['Team A', 'Team B', 'Team C'] * 34,
        'AwayTeam': ['Team B', 'Team C', 'Team A'] * 34,
        'FTHG': np.random.randint(0, 5, 100),
        'FTAG': np.random.randint(0, 5, 100),
        'FTR': np.random.choice(['H', 'D', 'A'], 100)
    })
    
    # Initialize preprocessor
    preprocessor = NonMajorLeaguePreprocessor()
    
    # Preprocess data
    processed_data = preprocessor.preprocess_pipeline(sample_data)
    
    print(f"Original data shape: {sample_data.shape}")
    print(f"Processed data shape: {processed_data.shape}")
    print(f"Processed columns: {list(processed_data.columns)}")
    
    # Get feature importance
    feature_importance = preprocessor.get_feature_importance(processed_data)
    print(f"Top 10 features by importance:")
    print(feature_importance.head(10))
    
    # Save preprocessors
    preprocessor.save_preprocessors('preprocessors.pkl')

if __name__ == "__main__":
    main()
