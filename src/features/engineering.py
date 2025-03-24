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
        """Add QB-specific advanced metrics with career trajectory awareness"""
        df = qb_data.copy()
        
        # PART 1: CAREER TRAJECTORY FEATURES
        # These will help detect declining playing time and role changes
        
        # TEAM CONTEXT: Add team-based analysis for QBs
        if 'posteam' in df.columns:
            # Number of quality receivers affects QB ceiling
            df['team_quality_receivers'] = 0
            
            # Group by team and season 
            for (team, season), group in df.groupby(['posteam', 'season']):
                # Count quality WRs (significant targets)
                quality_wr_count = ((self.data_dict.get('seasonal', pd.DataFrame())['position'] == 'WR') & 
                                (self.data_dict.get('seasonal', pd.DataFrame())['posteam'] == team) &
                                (self.data_dict.get('seasonal', pd.DataFrame())['season'] == season) &
                                (self.data_dict.get('seasonal', pd.DataFrame())['targets'] > 70)).sum()
                
                # Count quality TEs
                quality_te_count = ((self.data_dict.get('seasonal', pd.DataFrame())['position'] == 'TE') & 
                                (self.data_dict.get('seasonal', pd.DataFrame())['posteam'] == team) &
                                (self.data_dict.get('seasonal', pd.DataFrame())['season'] == season) &
                                (self.data_dict.get('seasonal', pd.DataFrame())['targets'] > 50)).sum()
                
                # Update all records for this team-season
                df.loc[(df['posteam'] == team) & (df['season'] == season), 
                    'team_quality_receivers'] = quality_wr_count + quality_te_count
            
            # Quality receivers boost QB ceiling
            df['receiver_quality_boost'] = df['team_quality_receivers'] * 0.1 + 1.0
            
            # Offensive line quality impacts QB performance
            if 'team_sack_rate' in df.columns:
                df['oline_quality'] = 1.0 - (df['team_sack_rate'] / 10.0)  # Lower sack rate = better line
            else:
                # Estimate from sack data if available
                if 'sacks' in df.columns and 'attempts' in df.columns:
                    df['oline_quality'] = 1.0 - (df['sacks'] / df['attempts'].clip(lower=20) * 5)
                else:
                    df['oline_quality'] = 1.0  # Default if no data
        
        # BREAKOUT DETECTION: Add a breakout candidate score for QBs
        df['breakout_candidate'] = 0
        if 'career_seasons' in df.columns:
            # QBs often break out in years 2-3
            qb_breakout_mask = (df['career_seasons'].between(1, 3)) & (df['attempts_per_game'] >= 25)
            df.loc[qb_breakout_mask, 'breakout_candidate'] = 1
            
            # Calculate breakout probability 0-100
            df['breakout_probability'] = (
                # Base probability
                15 +
                # Year 2 QBs often take a leap
                (df['career_seasons'] == 2) * 25 +
                # Already seeing good volume
                (df['attempts_per_game'] >= 30) * 15 +
                # Quality receivers helps breakout chances
                (df['team_quality_receivers'] >= 2) * 15
            ).clip(0, 100)
        
        # CEILING PROJECTIONS: Calculate ceiling factors
        if 'fantasy_points_per_game' in df.columns:
            # Baseline already exists in the model
            # Add ceiling projection factors
            df['ceiling_factor'] = 1.4  # Default
            
            # Factors that can lead to a higher QB ceiling:
            
            # 1. Young QB with breakout potential
            if 'breakout_candidate' in df.columns:
                df.loc[df['breakout_candidate'] == 1, 'ceiling_factor'] += 0.25
            
            # 2. Rushing ability dramatically increases ceiling
            if 'rushing_yards_per_game' in df.columns:
                df.loc[df['rushing_yards_per_game'] > 30, 'ceiling_factor'] += 0.2
                
            # 3. Quality receivers
            if 'team_quality_receivers' in df.columns:
                df.loc[df['team_quality_receivers'] >= 2, 'ceiling_factor'] += 0.15
            
            # 4. Good offensive line
            if 'oline_quality' in df.columns:
                df.loc[df['oline_quality'] > 0.8, 'ceiling_factor'] += 0.1
            
            # 5. Showing improvement
            if 'attempts_yoy_change' in df.columns:
                df.loc[df['attempts_yoy_change'] > 0.1, 'ceiling_factor'] += 0.15
                
            # df['ceiling_projection'] = df['fantasy_points_per_game'] * df['ceiling_factor']

        
        if 'season' in df.columns:
            # Sort by player and season to ensure time-based calculations are correct
            df = df.sort_values(['player_id', 'season'])
            
            # Calculate "recency" - how recent is this season compared to the max season
            max_season = df['season'].max()
            df['seasons_from_present'] = max_season - df['season']
            
            # Group by player to calculate trajectory features
            player_seasons = df.groupby('player_id')['season'].count()
            df['career_seasons'] = df['player_id'].map(player_seasons)
            
            # Calculate year-over-year changes in playing time
            if 'attempts' in df.columns:
                # Get previous season attempts for same player
                df['prev_attempts'] = df.groupby('player_id')['attempts'].shift(1)
                df['attempts_yoy_change'] = (df['attempts'] - df['prev_attempts']) / df['prev_attempts'].clip(lower=100)
                
                # Calculate 3-year attempts trend (negative means declining playing time)
                df['attempts_3yr'] = df.groupby('player_id')['attempts'].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
                most_recent_attempts = df.groupby('player_id')['attempts'].transform('last')
                df['attempts_vs_peak'] = df['attempts'] / df.groupby('player_id')['attempts'].transform('max')
                df['recent_role_decline'] = (most_recent_attempts < 200) & (df['attempts_vs_peak'] < 0.5)
                
            # Calculate recency-weighted attempts (emphasizes recent seasons)
            if 'attempts' in df.columns and 'seasons_from_present' in df.columns:
                # Give exponentially less weight to older seasons
                df['recency_weight'] = 0.5 ** df['seasons_from_present']
                df['weighted_attempts'] = df['attempts'] * df['recency_weight']
                
                # Aggregate weighted attempts by player
                total_weighted = df.groupby('player_id')['weighted_attempts'].transform('sum')
                total_weights = df.groupby('player_id')['recency_weight'].transform('sum')
                df['recency_weighted_attempts'] = total_weighted / total_weights
                
                # Flag players with significant playing time decline
                df['playing_time_declined'] = (df['attempts'] < 100) & (df['recency_weighted_attempts'] > 300)
        
        # PART 2: CURRENT ROLE ASSESSMENT
        # Define what makes a current starter vs backup more explicitly
        
        # Base starter status on attempts per game 
        if all(col in df.columns for col in ['attempts', 'games']):
            df['attempts_per_game'] = df['attempts'] / df['games'].clip(lower=1)
            
            # Define starter tiers
            df['full_time_starter'] = (df['attempts_per_game'] >= 25).astype(int)
            df['part_time_starter'] = ((df['attempts_per_game'] >= 15) & (df['attempts_per_game'] < 25)).astype(int)
            df['backup'] = (df['attempts_per_game'] < 15).astype(int)
            
            # For most recent season, create "current_role" feature
            if 'seasons_from_present' in df.columns:
                is_current_season = (df['seasons_from_present'] == 0)
                df['current_full_starter'] = (is_current_season & (df['full_time_starter'] == 1)).astype(int)
                df['current_part_starter'] = (is_current_season & (df['part_time_starter'] == 1)).astype(int)
                df['current_backup'] = (is_current_season & (df['backup'] == 1)).astype(int)
                
                # Propagate current role backward to all player seasons
                df['current_role'] = None
                for role in ['current_full_starter', 'current_part_starter', 'current_backup']:
                    role_value = 'Full Starter' if role == 'current_full_starter' else 'Part Starter' if role == 'current_part_starter' else 'Backup'
                    role_players = df.loc[df[role] == 1, 'player_id'].unique()
                    df.loc[df['player_id'].isin(role_players), 'current_role'] = role_value
        
        # PART 3: CAREER STAGE DETECTION
        # Identify rise, peak, and decline phases
        
        if all(col in df.columns for col in ['attempts', 'season', 'player_id']):
            # Calculate each player's peak season
            peak_seasons = df.loc[df.groupby('player_id')['attempts'].idxmax()][['player_id', 'season']]
            peak_seasons = dict(zip(peak_seasons['player_id'], peak_seasons['season']))
            df['peak_season'] = df['player_id'].map(peak_seasons)
            
            # Define career stage
            df['years_from_peak'] = df['season'] - df['peak_season']
            df['career_stage'] = pd.cut(
                df['years_from_peak'],
                bins=[-float('inf'), -2, 0, 1, float('inf')],
                labels=['Rise', 'Peak', 'Early_Decline', 'Late_Decline']
            )
            
            # For QBs in decline phase, check if they're getting worse
            df['is_declining'] = df['career_stage'].isin(['Early_Decline', 'Late_Decline']) & (df['attempts_yoy_change'] < 0)
        
        # Create explicit "do not draft" flag
        df['do_not_draft'] = 0
        
        # Flag QBs who are backups in recent years but had success in the past 
        if all(col in df.columns for col in ['current_role', 'career_stage', 'attempts_vs_peak']):
            df.loc[(df['current_role'] == 'Backup') & 
                (df['career_stage'].isin(['Early_Decline', 'Late_Decline'])) &
                (df['attempts_vs_peak'] < 0.5), 'do_not_draft'] = 1
        
        # Flag QBs who haven't played meaningful snaps in recent seasons
        recent_season_mask = df['seasons_from_present'] <= 1  # Current and previous season
        player_recent_attempts = df[recent_season_mask].groupby('player_id')['attempts'].sum()
        low_recent_volume_players = player_recent_attempts[player_recent_attempts < 200].index
        df.loc[df['player_id'].isin(low_recent_volume_players), 'do_not_draft'] = 1
        
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
        
        # Add age-based career trajectory features
        if 'age' in df.columns:
            # QB age bands
            df['qb_early_career'] = (df['age'] <= 25).astype(int)
            df['qb_prime_career'] = ((df['age'] > 25) & (df['age'] <= 35)).astype(int)
            df['qb_late_career'] = (df['age'] > 35).astype(int)
            
            # Flag very old QBs who are likely near retirement
            df.loc[(df['age'] >= 38) & (df['attempts_vs_peak'] < 0.7), 'do_not_draft'] = 1
        
        return df

    def _add_rb_specific_features(self, rb_data):
        """Add RB-specific advanced metrics with career trajectory awareness"""
        df = rb_data.copy()
        
        # PART 1: CAREER TRAJECTORY FEATURES
        # TEAM CONTEXT: Add team-based analysis for RBs
        if 'posteam' in df.columns:
            # Offensive line quality impacts RB performance
            if 'team_rush_yards_before_contact' in df.columns:
                df['oline_quality'] = df['team_rush_yards_before_contact'] / 2.0
            else:
                df['oline_quality'] = 1.0  # Default
            
            # RB committee situation
            df['rb_committee_count'] = 0
            
            # Group by team and season 
            for (team, season), group in df.groupby(['posteam', 'season']):
                # Count significant RBs (meaningful carries)
                sig_rb_count = ((self.data_dict.get('seasonal', pd.DataFrame())['position'] == 'RB') & 
                            (self.data_dict.get('seasonal', pd.DataFrame())['posteam'] == team) &
                            (self.data_dict.get('seasonal', pd.DataFrame())['season'] == season) &
                            (self.data_dict.get('seasonal', pd.DataFrame())['carries'] > 80)).sum()
                
                # Update all records for this team-season
                df.loc[(df['posteam'] == team) & (df['season'] == season), 
                    'rb_committee_count'] = sig_rb_count
            
            # RB committee situation impacts ceiling
            df['committee_factor'] = 1.0 / df['rb_committee_count'].clip(lower=1.0)
            
            # QB rushing affects RB opportunity
            df['qb_rush_threat'] = 0
            
            # Calculate if team has rushing QB that might vulture TDs
            for (team, season), group in df.groupby(['posteam', 'season']):
                # Identify if team has a rushing QB
                qb_rush_yards = ((self.data_dict.get('seasonal', pd.DataFrame())['position'] == 'QB') & 
                                (self.data_dict.get('seasonal', pd.DataFrame())['posteam'] == team) &
                                (self.data_dict.get('seasonal', pd.DataFrame())['season'] == season) &
                                (self.data_dict.get('seasonal', pd.DataFrame())['rushing_yards'] > 400)).any()
                
                # Update all records
                df.loc[(df['posteam'] == team) & (df['season'] == season), 
                    'qb_rush_threat'] = qb_rush_yards
        
        # BREAKOUT DETECTION: Add a breakout candidate score for RBs
        df['breakout_candidate'] = 0
        if 'career_seasons' in df.columns:
            # RBs often break out in years 1-2
            rb_breakout_mask = (df['career_seasons'].between(0, 2)) & (df['touches_per_game'] >= 10)
            df.loc[rb_breakout_mask, 'breakout_candidate'] = 1
            
            # Calculate breakout probability 0-100
            df['breakout_probability'] = (
                # Base probability
                20 +
                # Year 2 RBs often break out
                (df['career_seasons'] == 1) * 30 +
                # Already seeing good volume
                (df['touches_per_game'] >= 12) * 15 +
                # Low committee competition
                (df['rb_committee_count'] <= 1) * 15
            ).clip(0, 100)
        
        # CEILING PROJECTIONS: Calculate ceiling factors
        if 'fantasy_points_per_game' in df.columns:
            # Baseline already exists in the model
            # Add ceiling projection factors
            df['ceiling_factor'] = 1.5  # Default
            
            # Factors that can lead to a higher RB ceiling:
            
            # 1. Young RB with breakout potential
            if 'breakout_candidate' in df.columns:
                df.loc[df['breakout_candidate'] == 1, 'ceiling_factor'] += 0.3
            
            # 2. Workhorse role
            if 'touches_per_game' in df.columns:
                df.loc[df['touches_per_game'] > 20, 'ceiling_factor'] += 0.25
            
            # 3. Receiving involvement dramatically increases ceiling
            if 'targets_per_game' in df.columns:
                df.loc[df['targets_per_game'] > 4, 'ceiling_factor'] += 0.2
            
            # 4. Solo backfield (not committee)
            if 'rb_committee_count' in df.columns:
                df.loc[df['rb_committee_count'] <= 1, 'ceiling_factor'] += 0.15
            
            # 5. Good offensive line
            if 'oline_quality' in df.columns:
                df.loc[df['oline_quality'] > 1.5, 'ceiling_factor'] += 0.1
            
            # 6. QB doesn't vulture rushing TDs
            if 'qb_rush_threat' in df.columns:
                df.loc[df['qb_rush_threat'] == 0, 'ceiling_factor'] += 0.1
                
            # df['ceiling_projection'] = df['fantasy_points_per_game'] * df['ceiling_factor']
        
        
        if 'season' in df.columns:
            # Sort by player and season for time-based calculations
            df = df.sort_values(['player_id', 'season'])
            
            # Calculate recency
            max_season = df['season'].max()
            df['seasons_from_present'] = max_season - df['season']
            
            # Career length tracking
            player_seasons = df.groupby('player_id')['season'].count()
            df['career_seasons'] = df['player_id'].map(player_seasons)
            
            # Track touches (carries + receptions) for RB usage
            if all(col in df.columns for col in ['carries', 'receptions']):
                df['touches'] = df['carries'] + df['receptions']
                
                # Year-over-year changes in workload
                df['prev_touches'] = df.groupby('player_id')['touches'].shift(1)
                df['touches_yoy_change'] = (df['touches'] - df['prev_touches']) / df['prev_touches'].clip(lower=20)
                
                # Calculate career trajectory metrics
                df['touches_vs_peak'] = df['touches'] / df.groupby('player_id')['touches'].transform('max')
                
                # Calculate cumulative career touches (important injury predictor for RBs)
                df['career_touches_to_date'] = df.groupby('player_id')['touches'].cumsum()
                
                # Flag high-mileage RBs (over 1,500 career touches)
                high_mileage_players = df.groupby('player_id')['touches'].sum()
                high_mileage_players = high_mileage_players[high_mileage_players > 1500].index
                df['high_mileage'] = df['player_id'].isin(high_mileage_players).astype(int)
                
                # Calculate recency-weighted touches
                df['recency_weight'] = 0.5 ** df['seasons_from_present']
                df['weighted_touches'] = df['touches'] * df['recency_weight']
                
                # Aggregate weighted touches by player
                total_weighted = df.groupby('player_id')['weighted_touches'].transform('sum')
                total_weights = df.groupby('player_id')['recency_weight'].transform('sum')
                df['recency_weighted_touches'] = total_weighted / total_weights
                
                # Flag declining workload
                df['workload_declined'] = (df['touches'] < 100) & (df['recency_weighted_touches'] > 200)
        
        # PART 2: CURRENT ROLE ASSESSMENT
        if all(col in df.columns for col in ['touches', 'games']):
            df['touches_per_game'] = df['touches'] / df['games'].clip(lower=1)
            
            # Define RB roles
            df['feature_back'] = (df['touches_per_game'] >= 15).astype(int)
            df['committee_back'] = ((df['touches_per_game'] >= 8) & (df['touches_per_game'] < 15)).astype(int)
            df['backup_rb'] = (df['touches_per_game'] < 8).astype(int)
            
            # Current role assessment
            if 'seasons_from_present' in df.columns:
                is_current_season = (df['seasons_from_present'] == 0)
                df['current_feature'] = (is_current_season & (df['feature_back'] == 1)).astype(int)
                df['current_committee'] = (is_current_season & (df['committee_back'] == 1)).astype(int)
                df['current_backup'] = (is_current_season & (df['backup_rb'] == 1)).astype(int)
                
                # Propagate current role backward to all player seasons
                df['current_role'] = None
                for role in ['current_feature', 'current_committee', 'current_backup']:
                    role_value = 'Feature' if role == 'current_feature' else 'Committee' if role == 'current_committee' else 'Backup'
                    role_players = df.loc[df[role] == 1, 'player_id'].unique()
                    df.loc[df['player_id'].isin(role_players), 'current_role'] = role_value
        
        # PART 3: CAREER STAGE DETECTION
        if all(col in df.columns for col in ['touches', 'season', 'player_id']):
            # Calculate each player's peak season
            peak_seasons = df.loc[df.groupby('player_id')['touches'].idxmax()][['player_id', 'season']]
            peak_seasons = dict(zip(peak_seasons['player_id'], peak_seasons['season']))
            df['peak_season'] = df['player_id'].map(peak_seasons)
            
            # Define career stage - RBs decline faster than other positions
            df['years_from_peak'] = df['season'] - df['peak_season']
            df['career_stage'] = pd.cut(
                df['years_from_peak'],
                bins=[-float('inf'), -1, 0, 1, float('inf')],
                labels=['Rise', 'Peak', 'Early_Decline', 'Late_Decline']
            )
            
            # Flag declining RBs
            df['is_declining'] = df['career_stage'].isin(['Early_Decline', 'Late_Decline']) & (df['touches_yoy_change'] < 0)
        
        # Create explicit "do not draft" flag
        df['do_not_draft'] = 0
        
        # Flag RBs in decline with reduced role
        if all(col in df.columns for col in ['current_role', 'career_stage', 'touches_vs_peak']):
            df.loc[(df['current_role'].isin(['Committee', 'Backup'])) & 
                (df['career_stage'].isin(['Early_Decline', 'Late_Decline'])) &
                (df['touches_vs_peak'] < 0.6), 'do_not_draft'] = 1
        
        # Flag RBs who haven't played meaningful snaps recently
        recent_season_mask = df['seasons_from_present'] <= 1  # Current and previous season
        player_recent_touches = df[recent_season_mask].groupby('player_id')['touches'].sum()
        low_recent_volume_players = player_recent_touches[player_recent_touches < 100].index
        df.loc[df['player_id'].isin(low_recent_volume_players), 'do_not_draft'] = 1
        
        # Flag very high mileage RBs
        if 'career_touches_to_date' in df.columns:
            df.loc[(df['career_touches_to_date'] > 2000) & (df['touches_vs_peak'] < 0.7), 'do_not_draft'] = 1
        
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
        
        # Add age-based career trajectory
        if 'age' in df.columns:
            # RB age bands - RBs decline earlier than other positions
            df['rb_early_career'] = (df['age'] <= 23).astype(int)
            df['rb_prime_career'] = ((df['age'] > 23) & (df['age'] <= 27)).astype(int)
            df['rb_late_career'] = (df['age'] > 27).astype(int)
            
            # Flag older RBs who are likely declining
            df.loc[(df['age'] >= 30) & (df['touches_vs_peak'] < 0.8), 'do_not_draft'] = 1
        
        return df

    def _add_receiver_specific_features(self, receiver_data):
        """Add WR/TE-specific advanced metrics with career trajectory awareness"""
        df = receiver_data.copy()
        
        # PART 1: CAREER TRAJECTORY FEATURES
        if 'season' in df.columns:
            # Sort by player and season for time-based calculations
            df = df.sort_values(['player_id', 'season'])
            
            # Calculate recency
            max_season = df['season'].max()
            df['seasons_from_present'] = max_season - df['season']
            
            # Career length tracking
            player_seasons = df.groupby('player_id')['season'].count()
            df['career_seasons'] = df['player_id'].map(player_seasons)
            
            # Track targets for receiver usage
            if 'targets' in df.columns:
                # Year-over-year changes in opportunities
                df['prev_targets'] = df.groupby('player_id')['targets'].shift(1)
                df['targets_yoy_change'] = (df['targets'] - df['prev_targets']) / df['prev_targets'].clip(lower=20)
                
                # Calculate career trajectory metrics
                df['targets_vs_peak'] = df['targets'] / df.groupby('player_id')['targets'].transform('max')
                
                # Calculate recency-weighted targets
                df['recency_weight'] = 0.5 ** df['seasons_from_present']
                df['weighted_targets'] = df['targets'] * df['recency_weight']
                
                # Aggregate weighted targets by player
                total_weighted = df.groupby('player_id')['weighted_targets'].transform('sum')
                total_weights = df.groupby('player_id')['recency_weight'].transform('sum')
                df['recency_weighted_targets'] = total_weighted / total_weights
                
                # Flag declining usage
                df['usage_declined'] = (df['targets'] < 50) & (df['recency_weighted_targets'] > 100)
        
        # PART 2: CURRENT ROLE ASSESSMENT
        if all(col in df.columns for col in ['targets', 'games']):
            df['targets_per_game'] = df['targets'] / df['games'].clip(lower=1)
            
            # Define receiver roles based on position
            if 'position' in df.columns:
                # Different thresholds for WRs vs TEs
                wr_mask = df['position'] == 'WR'
                te_mask = df['position'] == 'TE'
                
                # WR role definitions
                df.loc[wr_mask, 'primary_target'] = (df.loc[wr_mask, 'targets_per_game'] >= 7).astype(int)
                df.loc[wr_mask, 'secondary_target'] = ((df.loc[wr_mask, 'targets_per_game'] >= 4) & 
                                                    (df.loc[wr_mask, 'targets_per_game'] < 7)).astype(int)
                df.loc[wr_mask, 'depth_receiver'] = (df.loc[wr_mask, 'targets_per_game'] < 4).astype(int)
                
                # TE role definitions (lower threshold)
                df.loc[te_mask, 'primary_target'] = (df.loc[te_mask, 'targets_per_game'] >= 5).astype(int)
                df.loc[te_mask, 'secondary_target'] = ((df.loc[te_mask, 'targets_per_game'] >= 3) & 
                                                    (df.loc[te_mask, 'targets_per_game'] < 5)).astype(int)
                df.loc[te_mask, 'depth_receiver'] = (df.loc[te_mask, 'targets_per_game'] < 3).astype(int)
            else:
                # Generic role definitions if position isn't available
                df['primary_target'] = (df['targets_per_game'] >= 6).astype(int)
                df['secondary_target'] = ((df['targets_per_game'] >= 3) & (df['targets_per_game'] < 6)).astype(int)
                df['depth_receiver'] = (df['targets_per_game'] < 3).astype(int)
            
            # Current role assessment
            if 'seasons_from_present' in df.columns:
                is_current_season = (df['seasons_from_present'] == 0)
                df['current_primary'] = (is_current_season & (df['primary_target'] == 1)).astype(int)
                df['current_secondary'] = (is_current_season & (df['secondary_target'] == 1)).astype(int)
                df['current_depth'] = (is_current_season & (df['depth_receiver'] == 1)).astype(int)
                
                # Propagate current role backward to all player seasons
                df['current_role'] = None
                for role in ['current_primary', 'current_secondary', 'current_depth']:
                    role_value = 'Primary' if role == 'current_primary' else 'Secondary' if role == 'current_secondary' else 'Depth'
                    role_players = df.loc[df[role] == 1, 'player_id'].unique()
                    df.loc[df['player_id'].isin(role_players), 'current_role'] = role_value
        
        # PART 3: CAREER STAGE DETECTION
        if all(col in df.columns for col in ['targets', 'season', 'player_id']):
            # Calculate each player's peak season
            peak_seasons = df.loc[df.groupby('player_id')['targets'].idxmax()][['player_id', 'season']]
            peak_seasons = dict(zip(peak_seasons['player_id'], peak_seasons['season']))
            df['peak_season'] = df['player_id'].map(peak_seasons)
            
            # Define career stage
            df['years_from_peak'] = df['season'] - df['peak_season']
            df['career_stage'] = pd.cut(
                df['years_from_peak'],
                bins=[-float('inf'), -2, 0, 1, float('inf')],
                labels=['Rise', 'Peak', 'Early_Decline', 'Late_Decline']
            )
            
            # Flag declining receivers
            df['is_declining'] = df['career_stage'].isin(['Early_Decline', 'Late_Decline']) & (df['targets_yoy_change'] < 0)
        
        # Create explicit "do not draft" flag
        df['do_not_draft'] = 0
        
        # Flag receivers in decline with reduced role
        if all(col in df.columns for col in ['current_role', 'career_stage', 'targets_vs_peak']):
            df.loc[(df['current_role'].isin(['Secondary', 'Depth'])) & 
                (df['career_stage'].isin(['Early_Decline', 'Late_Decline'])) &
                (df['targets_vs_peak'] < 0.5), 'do_not_draft'] = 1
        
        # FIX: Safely check recent targets by checking recent season mask first
        if 'seasons_from_present' in df.columns and 'targets' in df.columns:
            recent_season_mask = df['seasons_from_present'] <= 1
            if recent_season_mask.any():
                # Group by player_id to get recent targets for each player
                player_recent_targets = df[recent_season_mask].groupby('player_id')['targets'].sum()
                
                # Handle WRs - FIXED: Check for valid indices before lookup
                if 'position' in df.columns:
                    # Get all WR player IDs
                    wr_players = df[df['position'] == 'WR']['player_id'].unique()
                    # Filter to only get WR players that exist in the player_recent_targets index
                    wr_players_in_idx = [p for p in wr_players if p in player_recent_targets.index]
                    
                    if len(wr_players_in_idx) > 0:
                        # Now safely filter the player_recent_targets
                        wr_low_targets = player_recent_targets.loc[wr_players_in_idx]
                        low_wr_players = wr_low_targets[wr_low_targets < 70].index
                        df.loc[df['player_id'].isin(low_wr_players), 'do_not_draft'] = 1
                    
                    # Do the same for TEs
                    te_players = df[df['position'] == 'TE']['player_id'].unique()
                    te_players_in_idx = [p for p in te_players if p in player_recent_targets.index]
                    
                    if len(te_players_in_idx) > 0:
                        te_low_targets = player_recent_targets.loc[te_players_in_idx]
                        low_te_players = te_low_targets[te_low_targets < 40].index
                        df.loc[df['player_id'].isin(low_te_players), 'do_not_draft'] = 1
                else:
                    # Generic threshold if position isn't available
                    low_recent_volume_players = player_recent_targets[player_recent_targets < 60].index
                    df.loc[df['player_id'].isin(low_recent_volume_players), 'do_not_draft'] = 1
        
        # BREAKOUT DETECTION: Add a breakout candidate score
        df['breakout_candidate'] = 0
        if 'career_seasons' in df.columns and 'position' in df.columns:
            # WRs often break out in year 2-3
            wr_mask = df['position'] == 'WR'
            wr_breakout_mask = wr_mask & (df['career_seasons'].between(1, 3)) & (df['targets_per_game'] >= 5)
            df.loc[wr_breakout_mask, 'breakout_candidate'] = 1
            
            # TEs often break out in years 3-4
            te_mask = df['position'] == 'TE'
            te_breakout_mask = te_mask & (df['career_seasons'].between(2, 4)) & (df['targets_per_game'] >= 3)
            df.loc[te_breakout_mask, 'breakout_candidate'] = 1
            
            # Calculate breakout probability 0-100
            df['breakout_probability'] = 0
            
            # For WRs
            if wr_mask.any():
                # Factors that increase breakout likelihood
                df.loc[wr_mask, 'breakout_probability'] = (
                    # Base probability
                    20 +
                    # Year 2 WRs have highest breakout rate
                    (df.loc[wr_mask, 'career_seasons'] == 2) * 30 +
                    # Already seeing good volume but not yet elite production
                    (df.loc[wr_mask, 'targets_per_game'].between(5, 8)) * 15 +
                    # Growing target share
                    (df.loc[wr_mask, 'targets_yoy_change'] > 0.2).astype(int) * 15
                ).clip(0, 100)
            
            # For TEs (adjust values for TEs who break out later)
            if te_mask.any():
                df.loc[te_mask, 'breakout_probability'] = (
                    # Base probability (lower for TEs)
                    15 +
                    # Year 3 TEs have highest breakout rate
                    (df.loc[te_mask, 'career_seasons'] == 3) * 25 +
                    # Already seeing decent volume
                    (df.loc[te_mask, 'targets_per_game'].between(3, 6)) * 15 +
                    # Growing target share
                    (df.loc[te_mask, 'targets_yoy_change'] > 0.3).astype(int) * 15
                ).clip(0, 100)
        
        # Add TEAM-BASED CONTEXT features
        # Get QB quality impact
        if 'posteam' in df.columns:
            # Calculate average team passer rating/QBR by season and team
            team_seasons = df.groupby(['posteam', 'season'])
            
            # Add QB quality metrics if available
            if 'team_passer_rating' in df.columns:
                df['qb_quality'] = df['team_passer_rating']
            else:
                # Try to estimate QB quality from team passing stats
                team_stats = pd.DataFrame()
                if all(col in df.columns for col in ['posteam', 'season']):
                    # This would require additional team data - we'll use placeholder logic
                    # that you'd replace with actual team-level passing efficiency metrics
                    df['qb_quality'] = 0  # Placeholder
                    
            # Calculate number of quality receivers on team
            if 'posteam' in df.columns and 'season' in df.columns:
                # Count WRs with significant targets on same team
                df['team_quality_receivers'] = 0
                
                # Group by team and season 
                for (team, season), group in df.groupby(['posteam', 'season']):
                    # Count quality receivers (significant targets)
                    quality_wr_count = ((group['position'] == 'WR') & 
                                    (group['targets'] > 70)).sum()
                    
                    # Update all records for this team-season
                    df.loc[(df['posteam'] == team) & (df['season'] == season), 
                        'team_quality_receivers'] = quality_wr_count
                
                # Potential target competition factor
                df['target_competition'] = 1 + (df['team_quality_receivers'] * 0.15)
                
                # This impacts ceiling - more target competition = lower ceiling
                df['ceiling_modifier'] = 1.0 / df['target_competition'].clip(lower=1.0)
        
        # CEILING PROJECTIONS: Calculate both baseline and ceiling projections
        if 'fantasy_points_per_game' in df.columns:
            # Baseline already exists in the model
            # Add ceiling projection factors
            df['ceiling_factor'] = 1.7  # Default
            
            # Factors that can lead to a higher ceiling:
            
            # 1. Young player on the rise with breakout potential
            if 'breakout_candidate' in df.columns:
                df.loc[df['breakout_candidate'] == 1, 'ceiling_factor'] += 0.25
            
            # 2. Elite target share in offense
            if 'targets_per_game' in df.columns:
                df.loc[df['targets_per_game'] > 8, 'ceiling_factor'] += 0.15
            
            # 3. Elite QB throwing to them
            if 'qb_quality' in df.columns:
                df.loc[df['qb_quality'] > 90, 'ceiling_factor'] += 0.1
            
            # 4. Low target competition
            if 'team_quality_receivers' in df.columns:
                df.loc[df['team_quality_receivers'] <= 1, 'ceiling_factor'] += 0.1
            
            # 5. Already showing upward trajectory
            if 'targets_yoy_change' in df.columns:
                df.loc[df['targets_yoy_change'] > 0.15, 'ceiling_factor'] += 0.2
            
            # Calculate ceiling projection
            # df['ceiling_projection'] = df['fantasy_points_per_game'] * df['ceiling_factor']
        
        # Add additional receiver metrics
        if all(col in df.columns for col in ['targets', 'receptions']):
            df['reception_rate'] = df['receptions'] / df['targets'].clip(lower=1) * 100
        
        if all(col in df.columns for col in ['receiving_air_yards', 'receiving_yards']):
            df['air_yards_share'] = df['receiving_air_yards'] / df['receiving_yards'].clip(lower=1) * 100
            df['yac_share'] = 100 - df['air_yards_share']
        
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
    
    def cluster_players(self, n_clusters=5, position_tier_drops=None):
        """
        Cluster players by position with position-specific tier dropping
        
        Parameters:
        -----------
        n_clusters : int, optional
            Number of clusters to create
        position_tier_drops : dict, optional
            Dictionary specifying how many tiers to drop for each position.
            Example: {'qb': 2, 'rb': 1, 'wr': 1, 'te': 1}
            If None, defaults to dropping 2 tiers for QB and 1 for other positions.
            
        Returns:
        --------
        self : FantasyFeatureEngineering
            Returns self for method chaining
        """
        # Default tier drops if not specified
        if position_tier_drops is None:
            position_tier_drops = {'qb': 2, 'rb': 1, 'wr': 1, 'te': 1}
        
        logger.info(f"Clustering players with {n_clusters} clusters per position, with position-specific tier dropping")
        
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
            
            # Get position-specific tier drops
            tiers_to_drop = position_tier_drops.get(position, 1)
            logger.info(f"Dropping {tiers_to_drop} bottom tiers for {position}")
            
            # Identify clusters to drop (bottom tiers)
            bottom_clusters = cluster_stats.index[-tiers_to_drop:].tolist()
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
                    
                    # Log the filtering impact
                    filter_pct = (len(projection_data_filtered) / len(projection_data)) * 100
                    logger.info(f"Filtered projection data: kept {len(projection_data_filtered)} of {len(projection_data)} players ({filter_pct:.1f}%)")
                else:
                    logger.warning(f"Not enough matching features for {position} projection clustering")
            
            # Store updated training data
            self.feature_sets[train_key] = train_data
            self.feature_sets[f"{position}_train_filtered"] = train_data_filtered
            
            # Log the filtering impact
            filter_pct = (len(train_data_filtered) / len(train_data)) * 100
            logger.info(f"Filtered {position} players: kept {len(train_data_filtered)} of {len(train_data)} players ({filter_pct:.1f}%)")
            logger.info(f"Tier distribution after filtering: {train_data_filtered['tier'].value_counts().to_dict()}")
        
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