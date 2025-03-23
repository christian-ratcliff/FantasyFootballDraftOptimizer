"""
Advanced feature engineering for fantasy football prediction
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import logging
from datetime import datetime
import re

logger = logging.getLogger(__name__)

class FantasyFeatureEngineering:
    """
    Class for advanced feature engineering on fantasy football data
    """
    
    def __init__(self, data_dict, target_year=None):
        """
        Initialize with data dictionary
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary containing processed data frames
        target_year : int, optional
            Year to use as target for projection (defaults to max year in data)
        """
        self.data_dict = data_dict
        self.feature_sets = {}
        self.target_year = target_year
        self.scalers = {}
        self.cluster_models = {}
        self.dropped_tiers = {}
        
        # Find the max year if target year not specified
        if not target_year and 'seasonal' in data_dict and not data_dict['seasonal'].empty:
            if 'season' in data_dict['seasonal'].columns:
                self.target_year = data_dict['seasonal']['season'].max()
                logger.info(f"Target year set to latest year in data: {self.target_year}")
        
        logger.info(f"Initialized feature engineering with {len(data_dict)} datasets")
    
    def create_position_features(self, positions=None):
        """
        Create position-specific feature sets
        
        Parameters:
        -----------
        positions : list, optional
            List of positions to process (defaults to ['QB', 'RB', 'WR', 'TE'])
            
        Returns:
        --------
        self : FantasyFeatureEngineering
            Returns self for method chaining
        """
        if positions is None:
            positions = ['QB', 'RB', 'WR', 'TE']
        
        logger.info(f"Creating position-specific features for {positions}")
        
        # Base seasonal data
        seasonal = self.data_dict.get('seasonal', pd.DataFrame())
        if seasonal.empty:
            logger.warning("No seasonal data found")
            return self
        
        # Check for position column
        if 'position' not in seasonal.columns:
            logger.warning("No position column in seasonal data")
            return self
        
        # Get NGS data if available
        ngs_passing = self.data_dict.get('ngs_passing', pd.DataFrame())
        ngs_rushing = self.data_dict.get('ngs_rushing', pd.DataFrame())
        ngs_receiving = self.data_dict.get('ngs_receiving', pd.DataFrame())
        
        # Process each position
        for position in positions:
            logger.info(f"Processing {position} features")
            
            # Filter seasonal data by position
            position_data = seasonal[seasonal['position'] == position].copy()
            
            if position_data.empty:
                logger.warning(f"No {position} data found")
                continue
            
            # Add position-specific features
            # These features should supplement what's already in the data
            if position == 'QB':
                # Add QB-specific metrics
                position_data = self._add_qb_specific_features(position_data)
                
                # Add NGS passing data if available
                if not ngs_passing.empty and 'player_id' in position_data.columns:
                    position_data = self._enhance_with_ngs_data(
                        position_data,
                        ngs_passing,
                        prefix='ngs_pass_',
                        key_metrics=[
                            'avg_time_to_throw', 'avg_completed_air_yards', 'avg_intended_air_yards',
                            'aggressiveness', 'avg_air_yards_differential', 'completion_percentage_above_expectation'
                        ]
                    )
            
            elif position == 'RB':
                # Add RB-specific metrics
                position_data = self._add_rb_specific_features(position_data)
                
                # Add NGS rushing data if available
                if not ngs_rushing.empty and 'player_id' in position_data.columns:
                    position_data = self._enhance_with_ngs_data(
                        position_data,
                        ngs_rushing,
                        prefix='ngs_rush_',
                        key_metrics=[
                            'efficiency', 'percent_attempts_gte_eight_defenders', 'avg_time_to_los',
                            'expected_rush_yards', 'rush_yards_over_expected', 'rush_yards_over_expected_per_att'
                        ]
                    )
            
            elif position in ['WR', 'TE']:
                # Add WR/TE-specific metrics
                position_data = self._add_receiver_specific_features(position_data)
                
                # Add NGS receiving data if available
                if not ngs_receiving.empty and 'player_id' in position_data.columns:
                    position_data = self._enhance_with_ngs_data(
                        position_data,
                        ngs_receiving,
                        prefix='ngs_rec_',
                        key_metrics=[
                            'avg_cushion', 'avg_separation', 'avg_intended_air_yards',
                            'percent_share_of_intended_air_yards', 'catch_percentage',
                            'avg_yac', 'avg_expected_yac', 'avg_yac_above_expectation'
                        ]
                    )
            
            # Store the enhanced dataset
            self.feature_sets[f"{position.lower()}_features"] = position_data
            logger.info(f"Created {position} feature set with {len(position_data)} rows and {len(position_data.columns)} columns")
        
        return self
    
    def _add_qb_specific_features(self, qb_data):
        """Add QB-specific advanced metrics"""
        df = qb_data.copy()
        
        # Advanced efficiency metrics
        if all(col in df.columns for col in ['passing_yards', 'attempts', 'passing_tds', 'interceptions']):
            # Adjusted Yards Per Attempt (AY/A)
            df['adjusted_yards_per_attempt'] = ((df['passing_yards'] + (20 * df['passing_tds']) - (45 * df['interceptions'])) / 
                                              df['attempts'].clip(lower=1))
        
        # Advanced success metrics
        if 'passing_epa' in df.columns and 'attempts' in df.columns:
            df['epa_per_attempt'] = df['passing_epa'] / df['attempts'].clip(lower=1)
        
        # Deep ball percentage (if air yards available)
        if all(col in df.columns for col in ['passing_air_yards', 'passing_yards']):
            df['deep_ball_percentage'] = df['passing_air_yards'] / df['passing_yards'].clip(lower=1) * 100
        
        # Rushing contribution
        if all(col in df.columns for col in ['rushing_yards', 'passing_yards']):
            df['rush_yards_percentage'] = df['rushing_yards'] / (df['passing_yards'] + df['rushing_yards']).clip(lower=1) * 100
        
        # Red zone efficiency
        if all(col in df.columns for col in ['passing_tds', 'passing_yards']):
            df['td_per_100yds'] = df['passing_tds'] * 100 / df['passing_yards'].clip(lower=1)
        
        # Add fantasy points per pass attempt
        if all(col in df.columns for col in ['fantasy_points', 'attempts']):
            df['fantasy_points_per_attempt'] = df['fantasy_points'] / df['attempts'].clip(lower=1)
        
        # Add career trajectory features
        if 'age' in df.columns:
            # QB age bands
            df['qb_early_career'] = (df['age'] <= 25).astype(int)
            df['qb_prime_career'] = ((df['age'] > 25) & (df['age'] <= 35)).astype(int)
            df['qb_late_career'] = (df['age'] > 35).astype(int)
            
            # Potential breakout indicator
            if 'seasons_in_league' in df.columns:
                df['potential_breakout'] = ((df['age'] <= 26) & (df['seasons_in_league'] == 2)).astype(int)
            
            # Age-adjusted performance
            if 'fantasy_points_per_game' in df.columns:
                # QBs often improve into late 20s, peak in early 30s
                df['age_adjusted_points'] = df['fantasy_points_per_game']
                
                # Adjust for expected growth/decline
                age_adjustment = pd.Series(1.0, index=df.index)
                age_adjustment[df['age'] < 24] = 1.15  # Young QBs likely to improve
                age_adjustment[(df['age'] >= 24) & (df['age'] < 26)] = 1.08  # Still improving
                age_adjustment[(df['age'] >= 26) & (df['age'] < 28)] = 1.04  # Approaching prime
                age_adjustment[(df['age'] >= 28) & (df['age'] <= 32)] = 1.0  # Prime years (baseline)
                age_adjustment[(df['age'] > 32) & (df['age'] <= 35)] = 0.96  # Slight decline
                age_adjustment[(df['age'] > 35) & (df['age'] <= 38)] = 0.92  # Moderate decline
                age_adjustment[df['age'] > 38] = 0.85  # Significant decline
                
                df['age_adjusted_points'] = df['fantasy_points_per_game'] * age_adjustment
        
        # Add consistency and ceiling metrics
        if 'fantasy_points_consistency' in df.columns:
            df['qb_consistency_tier'] = pd.qcut(df['fantasy_points_consistency'].clip(0, 100), 
                                            4, labels=['Low', 'Medium', 'Good', 'Elite'])
        
        return df
    
    def _add_rb_specific_features(self, rb_data):
        """Add RB-specific advanced metrics"""
        df = rb_data.copy()
        
        # Workload and efficiency metrics
        if all(col in df.columns for col in ['carries', 'receptions', 'games']):
            df['touches_per_game'] = (df['carries'] + df['receptions']) / df['games'].clip(lower=1)
        
        # Involvement in passing game
        if all(col in df.columns for col in ['targets', 'receptions', 'carries']):
            df['receiving_opportunity_share'] = df['targets'] / (df['targets'] + df['carries']).clip(lower=1) * 100
            df['target_success_rate'] = df['receptions'] / df['targets'].clip(lower=1) * 100
        
        # Scoring profile
        if all(col in df.columns for col in ['rushing_tds', 'receiving_tds']):
            df['total_tds'] = df['rushing_tds'] + df['receiving_tds']
            
            # TD distribution
            if 'total_tds' in df.columns:
                df['rushing_td_share'] = df['rushing_tds'] / df['total_tds'].clip(lower=1) * 100
                df['receiving_td_share'] = df['receiving_tds'] / df['total_tds'].clip(lower=1) * 100
        
        # Red zone efficiency
        if all(col in df.columns for col in ['rushing_tds', 'carries']):
            df['tds_per_carry'] = df['rushing_tds'] / df['carries'].clip(lower=1)
        
        # Add fantasy points per touch
        if all(col in df.columns for col in ['fantasy_points', 'carries', 'receptions']):
            df['fantasy_points_per_touch'] = df['fantasy_points'] / (df['carries'] + df['receptions']).clip(lower=1)
        
        # Add age-based features
        if 'age' in df.columns:
            # RB age bands
            df['rb_early_career'] = (df['age'] <= 23).astype(int)
            df['rb_prime_career'] = ((df['age'] > 23) & (df['age'] <= 27)).astype(int)
            df['rb_late_career'] = (df['age'] > 27).astype(int)
            
            # Potential breakout indicator
            if 'seasons_in_league' in df.columns:
                df['potential_breakout'] = ((df['age'] <= 24) & (df['seasons_in_league'] == 1)).astype(int)
            
            # Age-adjusted performance (RBs decline earlier than other positions)
            if 'fantasy_points_per_game' in df.columns:
                df['age_adjusted_points'] = df['fantasy_points_per_game']
                
                # Adjust for expected growth/decline
                age_adjustment = pd.Series(1.0, index=df.index)
                age_adjustment[df['age'] < 22] = 1.12  # Very young RBs likely to improve
                age_adjustment[(df['age'] >= 22) & (df['age'] < 24)] = 1.06  # Still improving
                age_adjustment[(df['age'] >= 24) & (df['age'] < 26)] = 1.0  # Prime years (baseline)
                age_adjustment[(df['age'] >= 26) & (df['age'] < 28)] = 0.94  # Early decline
                age_adjustment[(df['age'] >= 28) & (df['age'] < 30)] = 0.88  # Moderate decline
                age_adjustment[df['age'] >= 30] = 0.80  # Significant decline
                
                df['age_adjusted_points'] = df['fantasy_points_per_game'] * age_adjustment
        
        # High-value touch indicator
        if all(col in df.columns for col in ['targets', 'carries']):
            # Targets generally more valuable than carries
            df['high_value_touch_share'] = df['targets'] / (df['targets'] + df['carries']).clip(lower=1) * 100
        
        # Add consistency metrics
        if 'fantasy_points_consistency' in df.columns:
            df['rb_consistency_tier'] = pd.qcut(df['fantasy_points_consistency'].clip(0, 100), 
                                             4, labels=['Low', 'Medium', 'Good', 'Elite'])
        
        return df
    
    def _add_receiver_specific_features(self, receiver_data):
        """Add WR/TE-specific advanced metrics"""
        df = receiver_data.copy()
        
        # Target share and efficiency
        if all(col in df.columns for col in ['targets', 'receptions']):
            df['reception_rate'] = df['receptions'] / df['targets'].clip(lower=1) * 100
        
        # Air yards metrics
        if all(col in df.columns for col in ['receiving_air_yards', 'receiving_yards']):
            df['air_yards_share'] = df['receiving_air_yards'] / df['receiving_yards'].clip(lower=1) * 100
            
            # YAC share
            df['yac_share'] = 100 - df['air_yards_share']
        
        # RACR (Receiver Air Conversion Ratio)
        if all(col in df.columns for col in ['receiving_yards', 'receiving_air_yards']):
            df['racr'] = df['receiving_yards'] / df['receiving_air_yards'].clip(lower=1)
        
        # Scoring efficiency
        if all(col in df.columns for col in ['receiving_tds', 'receptions']):
            df['td_per_reception'] = df['receiving_tds'] / df['receptions'].clip(lower=1)
        
        # Fantasy efficiency
        if all(col in df.columns for col in ['fantasy_points', 'targets']):
            df['fantasy_points_per_target'] = df['fantasy_points'] / df['targets'].clip(lower=1)
        
        if 'position' in df.columns:
            # Add separate position indicators
            df['is_wr'] = (df['position'] == 'WR').astype(int)
            df['is_te'] = (df['position'] == 'TE').astype(int)
            
            # Add age-specific features
            if 'age' in df.columns:
                # Different age profiles for WR vs TE
                wr_mask = df['position'] == 'WR'
                te_mask = df['position'] == 'TE'
                
                # Age bands for WRs
                df.loc[wr_mask, 'early_career'] = (df.loc[wr_mask, 'age'] <= 24).astype(int)
                df.loc[wr_mask, 'prime_career'] = ((df.loc[wr_mask, 'age'] > 24) & (df.loc[wr_mask, 'age'] <= 29)).astype(int)
                df.loc[wr_mask, 'late_career'] = (df.loc[wr_mask, 'age'] > 29).astype(int)
                
                # Age bands for TEs (develop later than WRs)
                df.loc[te_mask, 'early_career'] = (df.loc[te_mask, 'age'] <= 25).astype(int)
                df.loc[te_mask, 'prime_career'] = ((df.loc[te_mask, 'age'] > 25) & (df.loc[te_mask, 'age'] <= 31)).astype(int)
                df.loc[te_mask, 'late_career'] = (df.loc[te_mask, 'age'] > 31).astype(int)
                
                # Potential breakout indicators
                if 'seasons_in_league' in df.columns:
                    # WRs often break out in year 2-3
                    df.loc[wr_mask, 'potential_breakout'] = ((df.loc[wr_mask, 'age'] <= 25) & 
                                                        (df.loc[wr_mask, 'seasons_in_league'].isin([1, 2]))).astype(int)
                    
                    # TEs often break out in year 3-4
                    df.loc[te_mask, 'potential_breakout'] = ((df.loc[te_mask, 'age'] <= 26) & 
                                                      (df.loc[te_mask, 'seasons_in_league'].isin([2, 3]))).astype(int)
                
                # Age-adjusted performance
                if 'fantasy_points_per_game' in df.columns:
                    df['age_adjusted_points'] = df['fantasy_points_per_game']
                    
                    # Adjust for WRs
                    wr_age_adjustment = pd.Series(1.0, index=df.index)
                    wr_age_adjustment[df['age'] < 23] = 1.15  # Very young WRs
                    wr_age_adjustment[(df['age'] >= 23) & (df['age'] < 25)] = 1.08  # Developing
                    wr_age_adjustment[(df['age'] >= 25) & (df['age'] < 27)] = 1.04  # Early prime
                    wr_age_adjustment[(df['age'] >= 27) & (df['age'] <= 29)] = 1.0  # Prime (baseline)
                    wr_age_adjustment[(df['age'] > 29) & (df['age'] <= 32)] = 0.94  # Early decline
                    wr_age_adjustment[(df['age'] > 32)] = 0.88  # Later decline
                    
                    # Adjust for TEs (peak later)
                    te_age_adjustment = pd.Series(1.0, index=df.index)
                    te_age_adjustment[df['age'] < 24] = 1.18  # Very young TEs
                    te_age_adjustment[(df['age'] >= 24) & (df['age'] < 26)] = 1.10  # Developing
                    te_age_adjustment[(df['age'] >= 26) & (df['age'] < 28)] = 1.05  # Early prime
                    te_age_adjustment[(df['age'] >= 28) & (df['age'] <= 31)] = 1.0  # Prime (baseline)
                    te_age_adjustment[(df['age'] > 31) & (df['age'] <= 34)] = 0.95  # Early decline
                    te_age_adjustment[(df['age'] > 34)] = 0.88  # Later decline
                    
                    # Apply adjustments based on position
                    df.loc[wr_mask, 'age_adjusted_points'] = df.loc[wr_mask, 'fantasy_points_per_game'] * wr_age_adjustment[wr_mask]
                    df.loc[te_mask, 'age_adjusted_points'] = df.loc[te_mask, 'fantasy_points_per_game'] * te_age_adjustment[te_mask]
        
        # Add consistency metrics
        if 'fantasy_points_consistency' in df.columns:
            df['rec_consistency_tier'] = pd.qcut(df['fantasy_points_consistency'].clip(0, 100), 
                                             4, labels=['Low', 'Medium', 'Good', 'Elite'])
        
        return df
    
    def _enhance_with_ngs_data(self, position_data, ngs_data, prefix='ngs_', key_metrics=None):
        """
        Enhance position data with NGS metrics
        
        Parameters:
        -----------
        position_data : DataFrame
            Position-specific seasonal data
        ngs_data : DataFrame
            NGS data for the relevant skill
        prefix : str, optional
            Prefix to add to NGS column names
        key_metrics : list, optional
            List of key NGS metrics to include
            
        Returns:
        --------
        DataFrame
            Enhanced position data
        """
        if position_data.empty or ngs_data.empty:
            return position_data
        
        # Check for required columns
        if 'player_id' not in position_data.columns or 'player_id' not in ngs_data.columns:
            # Try with player_gsis_id if direct player_id not available
            if 'player_id' not in ngs_data.columns and 'player_gsis_id' in ngs_data.columns:
                ngs_data['player_id'] = ngs_data['player_gsis_id']
            else:
                logger.warning("Cannot enhance with NGS data - missing player_id columns")
                return position_data
        
        # Ensure season column exists
        if 'season' not in position_data.columns or 'season' not in ngs_data.columns:
            logger.warning("Cannot enhance with NGS data - missing season columns")
            return position_data
        
        # Create a copy to avoid modifying the original
        enhanced_data = position_data.copy()
        
        # Select key metrics if specified
        if key_metrics:
            # Filter to metrics that actually exist in the data
            available_metrics = [m for m in key_metrics if m in ngs_data.columns]
            if not available_metrics:
                logger.warning(f"None of the specified NGS metrics are available: {key_metrics}")
                return enhanced_data
        else:
            # Use all numeric columns except player ID and season
            numeric_cols = ngs_data.select_dtypes(include=['number']).columns
            available_metrics = [col for col in numeric_cols if col not in ['player_id', 'player_gsis_id', 'season', 'week']]
        
        # Group NGS data by player and season (average across weeks)
        try:
            ngs_grouped = ngs_data.groupby(['player_id', 'season'])[available_metrics].mean().reset_index()
            
            # Add prefix to column names for clarity
            ngs_cols = {col: f"{prefix}{col}" for col in available_metrics}
            ngs_grouped = ngs_grouped.rename(columns=ngs_cols)
            
            # Merge with position data
            enhanced_data = pd.merge(
                enhanced_data,
                ngs_grouped,
                on=['player_id', 'season'],
                how='left'
            )
            
            # Fill NA values with zeros or appropriate defaults
            for col in ngs_cols.values():
                if col in enhanced_data.columns:
                    enhanced_data[col] = enhanced_data[col].fillna(0)
            
            logger.info(f"Enhanced position data with {len(ngs_cols)} NGS metrics")
        except Exception as e:
            logger.error(f"Error enhancing with NGS data: {e}")
        
        return enhanced_data
    
    def prepare_prediction_features(self):
        """
        Prepare features for prediction modeling
        
        Returns:
        --------
        self : FantasyFeatureEngineering
            Returns self for method chaining
        """
        logger.info("Preparing prediction features")
        
        # Process each position's feature set
        for position in ['qb', 'rb', 'wr', 'te']:
            feature_key = f"{position}_features"
            
            if feature_key not in self.feature_sets or self.feature_sets[feature_key].empty:
                logger.warning(f"No {position} feature set available")
                continue
            
            # Get position data
            position_data = self.feature_sets[feature_key].copy()
            
            # Select relevant columns for modeling
            features = self._select_prediction_features(position_data, position)
            
            # Split into training and projection sets
            if self.target_year and 'season' in position_data.columns:
                train_mask = position_data['season'] < self.target_year
                projection_mask = position_data['season'] == self.target_year
                
                train_data = position_data[train_mask]
                projection_data = position_data[projection_mask]
                
                # Store training and projection data
                self.feature_sets[f"{position}_train"] = train_data
                self.feature_sets[f"{position}_projection"] = projection_data
                
                logger.info(f"Split {position} data into {len(train_data)} training and {len(projection_data)} projection samples")
            else:
                logger.warning(f"Cannot split {position} data - missing target year or season column")
        
        return self
    
    def _select_prediction_features(self, data, position):
        """
        Select relevant features for prediction modeling
        
        Parameters:
        -----------
        data : DataFrame
            Position data
        position : str
            Position name
            
        Returns:
        --------
        list
            List of selected feature names
        """
        # Basic features for all positions
        common_features = [
            'player_id', 'name', 'season', 'age', 'fantasy_points', 'fantasy_points_per_game',
            'games', 'seasons_in_league'
        ]
        
        # Add trend features if they exist
        trend_features = [col for col in data.columns if any(
            pattern in col for pattern in ['_prev_season', '_change', '_pct_change', '_3yr_avg', '_weighted_avg']
        )]
        
        # Add consistency/ceiling metrics
        consistency_features = [col for col in data.columns if any(
            pattern in col for pattern in ['_consistency', 'consistency_', '_tier']
        )]
        
        # Add age-related features
        age_features = [col for col in data.columns if any(
            pattern in col for pattern in ['_career', 'career_', 'age_adjusted', 'potential_breakout']
        )]
        
        # Add NGS features
        ngs_features = [col for col in data.columns if col.startswith('ngs_')]
        
        # Position-specific features
        if position == 'qb':
            position_features = [
                'passing_yards', 'passing_yards_per_game', 'passing_tds', 'passing_tds_per_game',
                'interceptions', 'attempts', 'completions', 'completion_percentage', 
                'rushing_yards', 'rushing_tds', 'adjusted_yards_per_attempt',
                'td_percentage', 'int_percentage', 'fantasy_points_per_attempt',
                'epa_per_attempt', 'deep_ball_percentage', 'rush_yards_percentage',
                'td_per_100yds'
            ]
        elif position == 'rb':
            position_features = [
                'rushing_yards', 'rushing_yards_per_game', 'rushing_tds', 'rushing_tds_per_game',
                'carries', 'yards_per_carry', 'receptions', 'targets', 'receiving_yards',
                'receiving_tds', 'touches_per_game', 'receiving_opportunity_share',
                'total_tds', 'rushing_td_share', 'receiving_td_share',
                'tds_per_carry', 'fantasy_points_per_touch', 'high_value_touch_share'
            ]
        elif position in ['wr', 'te']:
            position_features = [
                'receiving_yards', 'receiving_yards_per_game', 'receiving_tds', 'receiving_tds_per_game',
                'receptions', 'targets', 'reception_rate', 'yards_per_reception', 'yards_per_target',
                'air_yards_share', 'yac_share', 'racr', 'td_per_reception',
                'fantasy_points_per_target', 'is_wr', 'is_te'
            ]
        else:
            position_features = []
        
        # Filter to features that actually exist in the data
        selected_features = common_features + position_features + trend_features + consistency_features + age_features + ngs_features
        selected_features = [col for col in selected_features if col in data.columns]
        
        logger.info(f"Selected {len(selected_features)} features for {position}")
        return selected_features
    
    def cluster_players(self, n_clusters=5, drop_tiers=1):
        """
        Cluster players by position
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters to create
        drop_tiers : int, optional
            Number of bottom tiers to drop
            
        Returns:
        --------
        self : FantasyFeatureEngineering
            Returns self for method chaining
        """
        logger.info(f"Clustering players with {n_clusters} clusters per position, dropping bottom {drop_tiers} tiers")
        
        # Process each position
        for position in ['qb', 'rb', 'wr', 'te']:
            train_key = f"{position}_train"
            projection_key = f"{position}_projection"
            
            if train_key not in self.feature_sets or self.feature_sets[train_key].empty:
                logger.warning(f"No {position} training data available for clustering")
                continue
            
            # Get training data
            train_data = self.feature_sets[train_key].copy()
            
            # Select features for clustering
            cluster_features = self._select_clustering_features(train_data, position)
            
            if len(cluster_features) < 2:
                logger.warning(f"Not enough features for {position} clustering")
                continue
            
            # Prepare data for clustering
            X = train_data[cluster_features].fillna(0)
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA for dimensionality reduction if we have many features
            if len(cluster_features) > 10:
                pca = PCA(n_components=min(8, len(cluster_features) - 1))
                X_pca = pca.fit_transform(X_scaled)
                logger.info(f"Applied PCA for {position}, explained variance: {sum(pca.explained_variance_ratio_):.2f}")
            else:
                X_pca = X_scaled
            
            # Apply K-means clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_pca)
            
            # Store the cluster model
            self.cluster_models[position] = {
                'kmeans': kmeans,
                'scaler': scaler,
                'features': cluster_features,
                'pca': pca if len(cluster_features) > 10 else None
            }
            
            # Add cluster assignments to training data
            train_data['cluster'] = clusters
            
            # Calculate cluster stats
            cluster_stats = train_data.groupby('cluster')['fantasy_points_per_game'].mean().sort_values(ascending=False)
            
            # Assign tier labels based on performance (rename clusters)
            tier_mapping = {}
            for i, (cluster, _) in enumerate(cluster_stats.items()):
                if i == 0:
                    tier_mapping[cluster] = 'Elite'
                elif i < n_clusters // 3:
                    tier_mapping[cluster] = 'High Tier'
                elif i < 2 * n_clusters // 3:
                    tier_mapping[cluster] = 'Mid Tier'
                else:
                    tier_mapping[cluster] = 'Low Tier'
            
            train_data['tier'] = train_data['cluster'].map(tier_mapping)
            
            # Identify clusters to drop (bottom tiers)
            bottom_clusters = cluster_stats.index[-drop_tiers:].tolist()
            self.dropped_tiers[position] = bottom_clusters
            
            # Create a filtered dataset excluding bottom tiers
            train_data_filtered = train_data[~train_data['cluster'].isin(bottom_clusters)].copy()
            
            # Apply clustering to projection data if available
            if projection_key in self.feature_sets and not self.feature_sets[projection_key].empty:
                projection_data = self.feature_sets[projection_key].copy()
                
                # Select same features used for training
                proj_features = [f for f in cluster_features if f in projection_data.columns]
                
                if len(proj_features) >= 2:
                    # Prepare projection data
                    X_proj = projection_data[proj_features].fillna(0)
                    
                    # Scale using the same scaler
                    X_proj_scaled = scaler.transform(X_proj)
                    
                    # Apply PCA if used in training
                    if len(cluster_features) > 10:
                        X_proj_pca = pca.transform(X_proj_scaled)
                    else:
                        X_proj_pca = X_proj_scaled
                    
                    # Predict clusters
                    proj_clusters = kmeans.predict(X_proj_pca)
                    
                    # Add cluster assignments to projection data
                    projection_data['cluster'] = proj_clusters
                    projection_data['tier'] = projection_data['cluster'].map(tier_mapping)
                    
                    # Create a filtered dataset excluding bottom tiers
                    projection_data_filtered = projection_data[~projection_data['cluster'].isin(bottom_clusters)].copy()
                    
                    # Store updated projection data
                    self.feature_sets[projection_key] = projection_data
                    self.feature_sets[f"{position}_projection_filtered"] = projection_data_filtered
                else:
                    logger.warning(f"Not enough matching features for {position} projection clustering")
            
            # Store updated training data
            self.feature_sets[train_key] = train_data
            self.feature_sets[f"{position}_train_filtered"] = train_data_filtered
            
            logger.info(f"Clustered {position} players into {n_clusters} tiers, dropped {len(bottom_clusters)} bottom tiers")
            logger.info(f"Retained {len(train_data_filtered)} of {len(train_data)} training samples after filtering")
        
        return self
    
    def _select_clustering_features(self, data, position):
        """
        Select features for clustering
        
        Parameters:
        -----------
        data : DataFrame
            Position data
        position : str
            Position name
            
        Returns:
        --------
        list
            List of selected feature names
        """
        # Basic fantasy performance metrics
        performance_features = ['fantasy_points_per_game']
        
        if position == 'qb':
            # QB clustering focuses on style and efficiency
            position_features = [
                'passing_yards_per_game', 'passing_tds_per_game', 'interceptions',
                'completion_percentage', 'adjusted_yards_per_attempt', 
                'rushing_yards_per_game', 'rushing_tds_per_game',
                'td_percentage', 'int_percentage'
            ]
            
            # Add NGS features if available
            ngs_features = [col for col in data.columns if col.startswith('ngs_pass_')]
            
        elif position == 'rb':
            # RB clustering focuses on workload and skill usage
            position_features = [
                'rushing_yards_per_game', 'rushing_tds_per_game',
                'receptions', 'receiving_yards_per_game',
                'touches_per_game', 'yards_per_carry',
                'receiving_opportunity_share'
            ]
            
            # Add NGS features if available
            ngs_features = [col for col in data.columns if col.startswith('ngs_rush_')]
            
        elif position in ['wr', 'te']:
            # WR/TE clustering focuses on volume and efficiency
            position_features = [
                'targets', 'receptions', 'receiving_yards_per_game',
                'receiving_tds_per_game', 'yards_per_reception',
                'yards_per_target', 'reception_rate'
            ]
            
            # Add air yards features if available
            if 'air_yards_share' in data.columns:
                position_features.append('air_yards_share')
            
            if 'racr' in data.columns:
                position_features.append('racr')
            
            # Add NGS features if available
            ngs_features = [col for col in data.columns if col.startswith('ngs_rec_')]
            
        else:
            position_features = []
            ngs_features = []
        
        # Filter to features that actually exist in the data
        selected_features = performance_features + position_features + ngs_features
        selected_features = [col for col in selected_features if col in data.columns]
        
        # Drop any constant columns or those with too many NaN values
        valid_features = []
        for col in selected_features:
            # Check if column has variation
            if data[col].nunique() > 1:
                # Check if column has reasonable amount of non-NaN values
                if data[col].notna().mean() > 0.7:  # At least 70% non-NaN
                    valid_features.append(col)
        
        logger.info(f"Selected {len(valid_features)} features for {position} clustering")
        return valid_features
    
    def finalize_features(self, apply_filtering=True):
        """
        Finalize feature sets for modeling
        
        Parameters:
        -----------
        apply_filtering : bool, optional
            Whether to apply tier filtering
            
        Returns:
        --------
        self : FantasyFeatureEngineering
            Returns self for method chaining
        """
        logger.info("Finalizing feature sets for modeling")
        
        # Create combined datasets for all positions
        train_data_combined = []
        projection_data_combined = []
        
        for position in ['qb', 'rb', 'wr', 'te']:
            if apply_filtering:
                train_key = f"{position}_train_filtered"
                projection_key = f"{position}_projection_filtered"
            else:
                train_key = f"{position}_train"
                projection_key = f"{position}_projection"
            
            if train_key in self.feature_sets and not self.feature_sets[train_key].empty:
                train_data_combined.append(self.feature_sets[train_key])
            
            if projection_key in self.feature_sets and not self.feature_sets[projection_key].empty:
                projection_data_combined.append(self.feature_sets[projection_key])
        
        # Combine datasets
        if train_data_combined:
            self.feature_sets['train_combined'] = pd.concat(train_data_combined, ignore_index=True)
            logger.info(f"Created combined training set with {len(self.feature_sets['train_combined'])} players")
        
        if projection_data_combined:
            self.feature_sets['projection_combined'] = pd.concat(projection_data_combined, ignore_index=True)
            logger.info(f"Created combined projection set with {len(self.feature_sets['projection_combined'])} players")
        
        return self
    
    def get_feature_sets(self):
        """
        Get the processed feature sets
        
        Returns:
        --------
        dict
            Dictionary of feature sets
        """
        return self.feature_sets
    
    def get_cluster_models(self):
        """
        Get the cluster models
        
        Returns:
        --------
        dict
            Dictionary of cluster models
        """
        return self.cluster_models