import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)

class FeatureEngineering:
    """
    Class for feature engineering on fantasy football data
    """
    
    def __init__(self, preprocessed_data, current_year=2024):
        """
        Initialize with preprocessed data
        
        Parameters:
        -----------
        preprocessed_data : dict
            Dictionary containing 'train' and 'test' dataframes
        current_year : int, optional
            Current year for age calculations and filtering
        """
        self.train_data = preprocessed_data['train']
        self.test_data = preprocessed_data['test']
        self.feature_columns = None
        self.scaler = None
        self.current_year = current_year
        self.cluster_models = {}  # Store cluster models by position
        
        # Add position column if missing but player_id is present
        if 'position' not in self.train_data.columns and 'player_id' in self.train_data.columns:
            logger.info("Position column not found in train data, adding from player_ids")
            # Implementation would go here if we had player_ids data
        
        logger.info(f"Initialized feature engineering with {len(self.train_data)} training samples and {len(self.test_data)} test samples")
    
    def create_positional_features(self):
        """
        Create position-specific features
        
        Returns:
        --------
        self : FeatureEngineering
            Returns self for method chaining
        """
        logger.info("Creating position-specific features")
        
        for df_name in ['train', 'test']:
            df = getattr(self, f"{df_name}_data")
            
            # QB-specific features
            # Check if necessary columns exist before using them
            if 'passing_yards' in df.columns:
                # Passing efficiency
                if 'passing_tds' in df.columns and 'interceptions' in df.columns:
                    df['passing_td_to_int_ratio'] = df['passing_tds'] / df['interceptions'].clip(lower=1)
                
                if 'completions' in df.columns and 'attempts' in df.columns:
                    df['completion_percentage'] = df['completions'] / df['attempts'].clip(lower=1) * 100
                
                if 'attempts' in df.columns:
                    df['yards_per_attempt'] = df['passing_yards'] / df['attempts'].clip(lower=1)
                
                if 'passing_tds' in df.columns and 'interceptions' in df.columns and 'attempts' in df.columns:
                    df['adjusted_yards_per_attempt'] = (df['passing_yards'] + 20*df['passing_tds'] - 45*df['interceptions']) / df['attempts'].clip(lower=1)
                
                if 'passing_air_yards' in df.columns:
                    df['deep_ball_percentage'] = df['passing_air_yards'] / df['passing_yards'].clip(lower=1)
                
                # Add advanced passing metrics if available
                if 'passing_epa' in df.columns and 'attempts' in df.columns:
                    df['epa_per_pass'] = df['passing_epa'] / df['attempts'].clip(lower=1)
                
                # TD% and INT%
                if 'passing_tds' in df.columns and 'attempts' in df.columns:
                    df['td_percentage'] = df['passing_tds'] / df['attempts'].clip(lower=1) * 100
                
                if 'interceptions' in df.columns and 'attempts' in df.columns:
                    df['int_percentage'] = df['interceptions'] / df['attempts'].clip(lower=1) * 100
                
                # QB rushing contribution
                if 'rushing_yards' in df.columns:
                    df['rushing_yards_percentage'] = df['rushing_yards'] / (df['passing_yards'] + df['rushing_yards']).clip(lower=1) * 100
            
            # RB-specific features
            if 'rushing_yards' in df.columns:
                # Check if carries column exists, if not try to use rush_attempts
                carries_col = None
                for col in ['carries', 'rush_attempts', 'rushing_att', 'rush_att']:
                    if col in df.columns:
                        carries_col = col
                        break
                
                # If we found a valid carries column
                if carries_col:
                    # Use the detected carries column
                    df['rushing_efficiency'] = df['rushing_yards'] / df[carries_col].clip(lower=1)
                    
                    if 'rushing_tds' in df.columns:
                        df['rushing_td_rate'] = df['rushing_tds'] / df[carries_col].clip(lower=1) * 100
                    
                    if 'receptions' in df.columns:
                        df['receiving_rb_ratio'] = df['receptions'] / (df[carries_col].clip(lower=1) + 1)
                    
                    # First down efficiency
                    if 'rushing_first_downs' in df.columns:
                        df['rushing_first_down_rate'] = df['rushing_first_downs'] / df[carries_col].clip(lower=1)
                    
                    # Add advanced rushing metrics if available
                    if 'rushing_epa' in df.columns:
                        df['epa_per_rush'] = df['rushing_epa'] / df[carries_col].clip(lower=1)
                
                # RB receiving contribution
                if 'receiving_yards' in df.columns:
                    df['receiving_percentage'] = df['receiving_yards'] / (df['rushing_yards'] + df['receiving_yards']).clip(lower=1) * 100
                    
                    if carries_col and 'receptions' in df.columns:
                        df['total_yards_per_touch'] = (df['rushing_yards'] + df['receiving_yards']) / (df[carries_col] + df['receptions']).clip(lower=1)
            
            # WR/TE-specific features
            if 'receiving_yards' in df.columns:
                if 'receptions' in df.columns:
                    df['yards_per_reception'] = df['receiving_yards'] / df['receptions'].clip(lower=1)
                    
                    if 'receiving_tds' in df.columns:
                        df['td_per_reception'] = df['receiving_tds'] / df['receptions'].clip(lower=1) * 100
                
                if 'targets' in df.columns and 'receptions' in df.columns:
                    df['reception_ratio'] = df['receptions'] / df['targets'].clip(lower=1)
                    df['yards_per_target'] = df['receiving_yards'] / df['targets'].clip(lower=1)
                
                # Advanced receiving metrics
                if all(col in df.columns for col in ['receiving_air_yards', 'receiving_yards_after_catch']):
                    df['air_yards_percentage'] = df['receiving_air_yards'] / df['receiving_yards'].clip(lower=1) * 100
                    df['yac_percentage'] = df['receiving_yards_after_catch'] / df['receiving_yards'].clip(lower=1) * 100
                
                # Add advanced receiving metrics if available
                if 'receiving_epa' in df.columns and 'targets' in df.columns:
                    df['epa_per_target'] = df['receiving_epa'] / df['targets'].clip(lower=1)
                
                # WR/TE first down metrics
                if 'receiving_first_downs' in df.columns:
                    if 'receptions' in df.columns:
                        df['first_down_rate'] = df['receiving_first_downs'] / df['receptions'].clip(lower=1)
                    
                    if 'targets' in df.columns:
                        df['first_down_per_target'] = df['receiving_first_downs'] / df['targets'].clip(lower=1)
            
            # General features
            td_columns = []
            yard_columns = []
            
            for col_prefix in ['passing', 'rushing', 'receiving']:
                td_col = f'{col_prefix}_tds'
                yard_col = f'{col_prefix}_yards'
                
                if td_col in df.columns:
                    td_columns.append(td_col)
                
                if yard_col in df.columns:
                    yard_columns.append(yard_col)
            
            if td_columns:
                df['total_td'] = df[td_columns].sum(axis=1)
            
            if yard_columns:
                df['total_yards'] = df[yard_columns].sum(axis=1)
            
            if 'games' in df.columns:
                if 'total_yards' in df.columns:
                    df['yards_per_game'] = df['total_yards'] / df['games'].clip(lower=1)
                
                if 'total_td' in df.columns:
                    df['td_per_game'] = df['total_td'] / df['games'].clip(lower=1)
                
                # Add per-game stats for key metrics
                for col_prefix in ['passing', 'rushing', 'receiving']:
                    for stat in ['yards', 'tds']:
                        col = f'{col_prefix}_{stat}'
                        if col in df.columns:
                            df[f'{col}_per_game'] = df[col] / df['games'].clip(lower=1)
                
                for col in ['targets', 'receptions', 'interceptions', 'fantasy_points']:
                    if col in df.columns:
                        df[f'{col}_per_game'] = df[col] / df['games'].clip(lower=1)
            
            # Player durability/availability
            if 'season' in df.columns and 'player_id' in df.columns and 'games' in df.columns:
                # Get games played per season for each player
                player_seasons = df.groupby(['player_id', 'season'])['games'].max().reset_index()
                # Calculate mean and std of games played per season
                player_durability = player_seasons.groupby('player_id')['games'].agg(['mean', 'std']).reset_index()
                player_durability.columns = ['player_id', 'avg_games_per_season', 'games_std']
                
                # Merge back to original data
                df = pd.merge(df, player_durability, on='player_id', how='left')
                df['durability_score'] = df['avg_games_per_season'] - df['games_std'].fillna(0)
            
            # Set the updated dataframe
            setattr(self, f"{df_name}_data", df)
        
        logger.info("Position-specific features created successfully")
        return self
    
    def create_age_features(self, player_info_df=None):
        """
        Create age-based features, considering player age and career trajectory
        
        Parameters:
        -----------
        player_info_df : DataFrame, optional
            DataFrame containing player birthdate information
            
        Returns:
        --------
        self : FeatureEngineering
            Returns self for method chaining
        """
        logger.info("Creating age-based features")
        
        # If no player info provided but we have birth_date or age in the data
        for df_name in ['train', 'test']:
            df = getattr(self, f"{df_name}_data")
            
            # If we already have age, use it
            if 'age' in df.columns:
                # Create age-based features
                df['age_squared'] = df['age'] ** 2  # Capture non-linear effect
                
                # Position-specific prime age indicators
                df['qb_prime'] = ((df['age'] >= 27) & (df['age'] <= 34)).astype(int)
                df['rb_prime'] = ((df['age'] >= 23) & (df['age'] <= 28)).astype(int)
                df['wr_prime'] = ((df['age'] >= 24) & (df['age'] <= 30)).astype(int)
                df['te_prime'] = ((df['age'] >= 25) & (df['age'] <= 32)).astype(int)
                
                # Create more specific age stage indicators
                df['early_career'] = (df['age'] <= 24).astype(int)
                df['prime_career'] = ((df['age'] > 24) & (df['age'] < 30)).astype(int)
                df['late_career'] = (df['age'] >= 30).astype(int)
                
                # Position-specific age adjustments
                if 'position' in df.columns:
                    df['position_adjusted_age'] = np.where(
                        df['position'] == 'RB',
                        df['age'] * 1.2,  # RBs age faster
                        np.where(
                            df['position'] == 'QB',
                            df['age'] * 0.9,  # QBs age slower
                            df['age']  # Other positions standard
                        )
                    )
            
            # If we have birth_date but no age
            elif 'birthdate' in df.columns:
                # Calculate age based on season and birth_date
                if 'season' in df.columns:
                    # Convert birthdate to datetime if it's not already
                    if pd.api.types.is_string_dtype(df['birthdate']):
                        df['birthdate'] = pd.to_datetime(df['birthdate'], errors='coerce')
                    
                    # Extract year from birthdate
                    df['birth_year'] = df['birthdate'].dt.year
                    
                    # Calculate age
                    df['age'] = df['season'] - df['birth_year']
                    
                    # Now create the same age features as above
                    df['age_squared'] = df['age'] ** 2
                    df['qb_prime'] = ((df['age'] >= 27) & (df['age'] <= 34)).astype(int)
                    df['rb_prime'] = ((df['age'] >= 23) & (df['age'] <= 28)).astype(int)
                    df['wr_prime'] = ((df['age'] >= 24) & (df['age'] <= 30)).astype(int)
                    df['te_prime'] = ((df['age'] >= 25) & (df['age'] <= 32)).astype(int)
                    df['early_career'] = (df['age'] <= 24).astype(int)
                    df['prime_career'] = ((df['age'] > 24) & (df['age'] < 30)).astype(int)
                    df['late_career'] = (df['age'] >= 30).astype(int)
            
            # Store updated dataframe
            setattr(self, f"{df_name}_data", df)
        
        logger.info("Age-based features created successfully")
        return self
    
    def create_trend_features(self):
        """
        Create trend-based features (year-over-year changes)
        
        Returns:
        --------
        self : FeatureEngineering
            Returns self for method chaining
        """
        logger.info("Creating trend-based features")
        
        # Group by player_id and season
        for df_name in ['train', 'test']:
            df = getattr(self, f"{df_name}_data")
            
            # Ensure we have player_id and season columns
            if 'player_id' not in df.columns or 'season' not in df.columns:
                logger.warning(f"player_id or season column missing in {df_name} data, skipping trend features")
                continue
            
            # Ensure we're sorted by player_id and season
            df = df.sort_values(['player_id', 'season'])
            
            # Create shift by 1 season for the same player
            df_shifted = df.groupby('player_id').shift(1)
            
            # Track previous seasons' averages for moving averages
            df_shifted_2 = df.groupby('player_id').shift(2)
            df_shifted_3 = df.groupby('player_id').shift(3)
            
            # Create trend features for key stats
            trend_metrics = [
                'fantasy_points_per_game', 'passing_yards', 'rushing_yards', 'receiving_yards', 
                'passing_tds', 'rushing_tds', 'receiving_tds', 'fantasy_points'
            ]
            
            # Add additional metrics if they exist
            potential_metrics = [
                'targets', 'receptions', 'interceptions', 'attempts', 'completions',
                'yards_per_attempt', 'yards_per_reception', 'yards_per_target',
                'rushing_efficiency', 'reception_ratio', 'epa_per_pass', 'epa_per_rush', 'epa_per_target'
            ]
            
            # Check which potential metrics exist in the data
            for metric in potential_metrics:
                if metric in df.columns:
                    trend_metrics.append(metric)
            
            # Add per-game metrics to the list if they exist
            per_game_metrics = []
            for base_metric in trend_metrics:
                pg_metric = f"{base_metric}_per_game"
                if pg_metric in df.columns:
                    per_game_metrics.append(pg_metric)
            
            trend_metrics.extend(per_game_metrics)
            
            # Make sure we're only using metrics that exist in the data
            trend_metrics = [m for m in trend_metrics if m in df.columns]
            
            for col in trend_metrics:
                # Previous season value
                df[f'{col}_prev_season'] = df_shifted[col]
                
                # Calculate absolute and percentage change
                df[f'{col}_change'] = df[col] - df[f'{col}_prev_season']
                # Avoid division by zero or NaN
                df[f'{col}_pct_change'] = df[f'{col}_change'] / df[f'{col}_prev_season'].clip(lower=0.1) * 100
                
                # Calculate simple moving average (3 seasons) if we have enough data
                if col in df_shifted_2.columns and col in df_shifted_3.columns:
                    df[f'{col}_3yr_avg'] = (
                        df[f'{col}_prev_season'].fillna(0) + 
                        df_shifted_2[col].fillna(0) + 
                        df_shifted_3[col].fillna(0)
                    ) / 3
                    
                    # Calculate deviation from 3-year average
                    df[f'{col}_vs_3yr_avg'] = df[col] - df[f'{col}_3yr_avg']
                    df[f'{col}_vs_3yr_avg_pct'] = df[f'{col}_vs_3yr_avg'] / df[f'{col}_3yr_avg'].clip(lower=0.1) * 100
                
                # Calculate weighted moving average (more weight to recent seasons)
                if col in df_shifted_2.columns and col in df_shifted_3.columns:
                    df[f'{col}_weighted_avg'] = (
                        3 * df[f'{col}_prev_season'].fillna(0) + 
                        2 * df_shifted_2[col].fillna(0) + 
                        1 * df_shifted_3[col].fillna(0)
                    ) / 6
                    
                    # Calculate deviation from weighted average
                    df[f'{col}_vs_weighted_avg'] = df[col] - df[f'{col}_weighted_avg']
                    df[f'{col}_vs_weighted_avg_pct'] = df[f'{col}_vs_weighted_avg'] / df[f'{col}_weighted_avg'].clip(lower=0.1) * 100
            
            # Create consistency metrics
            for col in ['fantasy_points', 'fantasy_points_per_game']:
                if col in df.columns and f'{col}_prev_season' in df.columns:
                    # Consistency measure (smaller absolute percentage change is more consistent)
                    df[f'{col}_consistency'] = 100 - abs(df[f'{col}_pct_change']).clip(upper=100)
            
            # Calculate season count for each player
            df['player_season_count'] = df.groupby('player_id')['season'].transform('count')
            
            # Calculate career trends
            if 'age' in df.columns:
                # Calculate age-related trends
                df['seasons_in_league'] = df['age'] - df.groupby('player_id')['age'].transform('min')
                
                # Create "peak season" indicator
                if 'position' in df.columns:
                    conditions = [
                        (df['position'] == 'QB') & (df['age'] >= 28) & (df['age'] <= 32),
                        (df['position'] == 'RB') & (df['age'] >= 24) & (df['age'] <= 27),
                        (df['position'] == 'WR') & (df['age'] >= 25) & (df['age'] <= 29),
                        (df['position'] == 'TE') & (df['age'] >= 26) & (df['age'] <= 30)
                    ]
                    df['peak_season'] = np.select(conditions, [1, 1, 1, 1], default=0)
            
            # Handle missing values for new features
            for col in df.columns:
                if col.endswith('_prev_season') or col.endswith('_change') or col.endswith('_pct_change') or \
                   col.endswith('_3yr_avg') or col.endswith('_vs_3yr_avg') or col.endswith('_vs_3yr_avg_pct') or \
                   col.endswith('_weighted_avg') or col.endswith('_vs_weighted_avg') or col.endswith('_vs_weighted_avg_pct') or \
                   col.endswith('_consistency'):
                    df[col] = df[col].fillna(0)
            
            # Store the updated dataframe
            setattr(self, f"{df_name}_data", df)
        
        logger.info("Trend-based features created successfully")
        return self
    
    def create_cluster_features(self, position='ALL', n_clusters=5):
        """
        Create features based on player clustering
        
        Parameters:
        -----------
        position : str
            Position to cluster ('ALL', 'QB', 'RB', 'WR', 'TE')
        n_clusters : int
            Number of clusters to create
            
        Returns:
        --------
        self : FeatureEngineering
            Returns self for method chaining
        """
        logger.info(f"Creating {n_clusters} cluster features for position: {position}")
        
        if position == 'ALL':
            positions = ['QB', 'RB', 'WR', 'TE']
        else:
            positions = [position]
        
        for pos in positions:
            for df_name in ['train', 'test']:
                df = getattr(self, f"{df_name}_data")
                
                # Check if we have position column
                if 'position' not in df.columns:
                    logger.warning(f"position column missing in {df_name} data, skipping cluster features")
                    continue
                
                # Filter to position
                pos_df = df[df['position'] == pos].copy()
                
                if len(pos_df) < n_clusters:
                    logger.warning(f"Not enough {pos} players ({len(pos_df)}) for clustering, skipping")
                    continue
                
                # Select relevant features for clustering based on position
                if pos == 'QB':
                    features = ['passing_yards_per_game', 'passing_tds_per_game', 
                              'interceptions', 'rushing_yards_per_game',
                              'completion_percentage', 'yards_per_attempt']
                elif pos == 'RB':
                    features = ['rushing_yards_per_game', 'rushing_tds_per_game',
                              'receptions_per_game', 'receiving_yards_per_game',
                              'rushing_efficiency', 'receiving_rb_ratio']
                elif pos == 'WR':
                    features = ['receptions_per_game', 'targets_per_game',
                              'receiving_yards_per_game', 'receiving_tds_per_game',
                              'yards_per_reception', 'reception_ratio']
                elif pos == 'TE':
                    features = ['receptions_per_game', 'targets_per_game',
                              'receiving_yards_per_game', 'receiving_tds_per_game',
                              'yards_per_reception', 'reception_ratio']
                
                # Filter to features that exist in the data
                features = [f for f in features if f in pos_df.columns]
                
                if len(features) < 2:
                    logger.warning(f"Not enough features for {pos} clustering, skipping")
                    continue
                
                # Prepare data for clustering
                X = pos_df[features].fillna(0)
                
                # Scale the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply K-means clustering
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_scaled)
                
                # Store the cluster model
                self.cluster_models[pos] = {
                    'kmeans': kmeans,
                    'scaler': scaler,
                    'features': features
                }
                
                # Add cluster assignments back to the position dataframe
                pos_df['cluster'] = clusters
                
                # Calculate cluster stats
                cluster_stats = pos_df.groupby('cluster')['fantasy_points_per_game'].mean().sort_values(ascending=False)
                
                # Rank clusters by performance and create tier assignments
                cluster_ranks = {cluster: rank + 1 for rank, cluster in enumerate(cluster_stats.index)}
                pos_df['cluster_rank'] = pos_df['cluster'].map(cluster_ranks)
                
                # Create one-hot encoding of clusters
                for i in range(n_clusters):
                    pos_df[f'{pos.lower()}_cluster_{i+1}'] = (pos_df['cluster'] == i).astype(int)
                
                # Identify top 3 clusters
                top_clusters = cluster_stats.head(3).index.tolist()
                pos_df['top_cluster'] = pos_df['cluster'].apply(lambda x: x in top_clusters)
                
                # Update the main dataframe with cluster info
                for col in ['cluster', 'cluster_rank', 'top_cluster'] + [f'{pos.lower()}_cluster_{i+1}' for i in range(n_clusters)]:
                    df.loc[pos_df.index, col] = pos_df[col]
                
                # Store the updated dataframe
                setattr(self, f"{df_name}_data", df)
        
        logger.info("Cluster features created successfully")
        return self
    
    def handle_missing_values(self):
        """
        Handle missing values in the feature set
        
        Returns:
        --------
        self : FeatureEngineering
            Returns self for method chaining
        """
        logger.info("Handling missing values")
        
        for df_name in ['train', 'test']:
            df = getattr(self, f"{df_name}_data")
            
            # Fill NaN values with 0 for numeric columns
            numeric_cols = df.select_dtypes(include=['number']).columns
            df[numeric_cols] = df[numeric_cols].fillna(0)
            
            # For any remaining columns, fill with the most frequent value
            for col in df.columns:
                if col not in numeric_cols and df[col].isnull().any():
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            
            # Store updated dataframe
            setattr(self, f"{df_name}_data", df)
        
        logger.info("Missing values handled successfully")
        return self
    
    def normalize_features(self, columns=None):
        """
        Normalize features using StandardScaler
        
        Parameters:
        -----------
        columns : list, optional
            List of columns to normalize
            
        Returns:
        --------
        self : FeatureEngineering
            Returns self for method chaining
        """
        logger.info("Normalizing features")
        
        if columns is None:
            # Use all numeric columns except player_id, season, and categorical features
            exclude_cols = ['player_id', 'season', 'name', 'position', 'team']
            numeric_cols = self.train_data.select_dtypes(include=['number']).columns
            columns = [col for col in numeric_cols if col not in exclude_cols and not col.startswith('cluster_') and not col.endswith('_cluster_')]
        
        self.feature_columns = columns
        
        # Initialize and fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(self.train_data[columns])
        
        # Transform the data
        for df_name in ['train', 'test']:
            df = getattr(self, f"{df_name}_data")
            df_scaled = pd.DataFrame(
                self.scaler.transform(df[columns]),
                columns=columns,
                index=df.index
            )
            
            # Replace original columns with scaled versions
            for col in columns:
                df[col] = df_scaled[col]
            
            # Store updated dataframe
            setattr(self, f"{df_name}_data", df)
        
        logger.info(f"Features normalized successfully: {len(columns)} columns")
        return self
    
    def get_processed_data(self, selected_only=False):
        """
        Return the processed data
        
        Parameters:
        -----------
        selected_only : bool, optional
            If True, return only the selected features
            
        Returns:
        --------
        dict
            Dictionary containing processed 'train' and 'test' dataframes
        """
        if hasattr(self, 'selected_features') and selected_only:
            return {
                'train': self.train_data[self.selected_features],
                'test': self.test_data[self.selected_features]
            }
        else:
            return {
                'train': self.train_data,
                'test': self.test_data
            }