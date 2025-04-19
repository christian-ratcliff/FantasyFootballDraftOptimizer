import pandas as pd
import nfl_data_py as nfl
from espn_api.football import League
import logging
import numpy as np
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

logger = logging.getLogger(__name__)

def load_espn_league_data(league_id, year, espn_s2=None, swid=None):
    """
    Load league data from ESPN API - ONLY for scoring settings and roster rules
    
    Parameters:
    -----------
    league_id : int
        ESPN league ID
    year : int
        Season year
    espn_s2 : str, optional
        ESPN S2 cookie for private leagues
    swid : str, optional
        ESPN SWID cookie for private leagues
        
    Returns:
    --------
    dict
        Dictionary containing league settings
    """
    # Initialize connection to ESPN league
    try:
        league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
    except Exception as e:
        logger.error(f"Error connecting to ESPN league: {e}")
        logger.info("Creating default league settings")
        return {
            'league_info': {
                'name': 'Default League',
                'team_count': 10,
                'playoff_teams': 6
            },
            'scoring_settings': {},
            'roster_settings': {},
            'teams': pd.DataFrame()
        }
    
    # Extract relevant league settings
    settings = league.settings
    
    # Extract scoring settings
    scoring_dict = {}
    for item in settings.scoring_format:
        scoring_dict[item['abbr']] = item['points']
    
    # Extract roster settings
    roster_dict = settings.position_slot_counts
    
    # Get league teams
    teams_list = []
    for team in league.teams:
        teams_list.append({
            'team_id': team.team_id,
            'team_name': team.team_name,
            'owner': team.owners[0] if team.owners else None,
            'division': team.division_name
        })
    
    logger.info(f"Successfully loaded league settings for {settings.name}")
    return {
        'league_info': {
            'name': settings.name,
            'team_count': settings.team_count,
            'playoff_teams': settings.playoff_team_count
        },
        'scoring_settings': scoring_dict,
        'roster_settings': roster_dict,
        'teams': pd.DataFrame(teams_list)
    }

def load_nfl_data(years, include_ngs=True, ngs_min_year=2016, use_threads=True):
    """
    Load NFL data from nfl_data_py
    
    Parameters:
    -----------
    years : list
        List of years to pull data for
    include_ngs : bool, optional
        Whether to include Next Gen Stats data
    ngs_min_year : int, optional
        Minimum year for NGS data availability
    use_threads : bool, optional
        Whether to use threading for faster data loading
        
    Returns:
    --------
    dict
        Dictionary containing loaded data
    """
    data = {}
    
    def load_seasonal_with_retry(years, max_retries=3):
        for attempt in range(max_retries):
            try:
                return nfl.import_seasonal_data(years)
            except Exception as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Seasonal data fetch failed (attempt {attempt+1}/{max_retries}). Retrying...")
                    time.sleep(2)  # Wait before retrying
                else:
                    logger.error(f"Failed to fetch seasonal data after {max_retries} attempts: {e}")
                    return pd.DataFrame()  # 
    
    # Load player IDs for mapping
    logger.info("Loading player ID mapping data")
    try:
        player_ids = nfl.import_ids()
        # Ensure position column exists
        if 'position' not in player_ids.columns and 'pos' in player_ids.columns:
            player_ids['position'] = player_ids['pos']
        data['player_ids'] = player_ids
        logger.info(f"Loaded player ID data with {len(player_ids)} rows")
    except Exception as e:
        logger.error(f"Error loading player ID data: {e}")
        data['player_ids'] = pd.DataFrame()
    
    # Load seasonal data
    logger.info(f"Loading seasonal data for years: {years}")
    try:
        seasonal_data = load_seasonal_with_retry(years)
        # Add season type column if missing
        if 'season_type' not in seasonal_data.columns:
            seasonal_data['season_type'] = 'REG'
        data['seasonal'] = seasonal_data
        logger.info(f"Loaded seasonal data with {len(seasonal_data)} rows")
    except Exception as e:
        logger.error(f"Error loading seasonal data: {e}")
        data['seasonal'] = pd.DataFrame()
        logger.info("Continuing without season data")
    
    # Load weekly data
    logger.info(f"Loading weekly data for years: {years}")
    try:
        weekly_data = nfl.import_weekly_data(years)
        data['weekly'] = weekly_data
        logger.info(f"Loaded weekly data with {len(weekly_data)} rows")
    except Exception as e:
        logger.error(f"Error loading weekly data: {e}")
        data['weekly'] = pd.DataFrame()
    
    # Load rosters
    logger.info(f"Loading roster data for years: {years}")
    try:
        roster_data = nfl.import_seasonal_rosters(years)
        data['rosters'] = roster_data
        logger.info(f"Loaded roster data with {len(roster_data)} rows")
    except Exception as e:
        logger.error(f"Error loading roster data: {e}")
        data['rosters'] = pd.DataFrame()
    
    # Load schedules
    logger.info(f"Loading schedule data for years: {years}")
    try:
        schedule_data = nfl.import_schedules(years)
        data['schedules'] = schedule_data
        logger.info(f"Loaded schedule data with {len(schedule_data)} rows")
    except Exception as e:
        logger.error(f"Error loading schedule data: {e}")
        data['schedules'] = pd.DataFrame()
    
    # Load NGS data if requested
    if include_ngs:
        # Filter years for NGS data
        ngs_years = [year for year in years if year >= ngs_min_year]
        if not ngs_years:
            logger.warning(f"No years >= {ngs_min_year} for NGS data")
        else:
            logger.info(f"Loading NGS data for years: {ngs_years}")
            
            # Function to load NGS data for a specific type and year
            def load_ngs_data_for_year(stat_type, year):
                try:
                    start_time = time.time()
                    year_data = nfl.import_ngs_data(stat_type, [year])
                    
                    if not year_data.empty:
                        # Add year column if not present
                        if 'season' not in year_data.columns:
                            year_data['season'] = year
                            
                        elapsed = time.time() - start_time
                        logger.info(f"Loaded NGS {stat_type} data for {year} with {len(year_data)} rows in {elapsed:.2f}s")
                        return year_data
                    else:
                        logger.warning(f"Empty NGS {stat_type} data for year {year}")
                        return None
                except Exception as e:
                    logger.warning(f"Error loading NGS {stat_type} data for year {year}: {e}")
                    return None
            
            # Load NGS data for each type
            for stat_type in ['passing', 'rushing', 'receiving']:
                logger.info(f"Loading {stat_type} NGS data for years: {ngs_years}")
                
                # Use threading to speed up data loading if enabled
                if use_threads:
                    all_ngs_data = []
                    with ThreadPoolExecutor(max_workers=min(10, len(ngs_years))) as executor:
                        future_to_year = {executor.submit(load_ngs_data_for_year, stat_type, year): year for year in ngs_years}
                        for future in as_completed(future_to_year):
                            year = future_to_year[future]
                            try:
                                result = future.result()
                                if result is not None:
                                    all_ngs_data.append(result)
                            except Exception as e:
                                logger.error(f"Exception for NGS {stat_type} data for year {year}: {e}")
                else:
                    # Sequential loading
                    all_ngs_data = []
                    for year in ngs_years:
                        result = load_ngs_data_for_year(stat_type, year)
                        if result is not None:
                            all_ngs_data.append(result)
                
                # Combine all years' data
                if all_ngs_data:
                    try:
                        combined_ngs = pd.concat(all_ngs_data, ignore_index=True)
                        data[f'ngs_{stat_type}'] = combined_ngs
                        logger.info(f"Combined {stat_type} NGS data with {len(combined_ngs)} rows")
                    except Exception as e:
                        logger.error(f"Error combining NGS {stat_type} data: {e}")
                        data[f'ngs_{stat_type}'] = pd.DataFrame()
                else:
                    data[f'ngs_{stat_type}'] = pd.DataFrame()
                    logger.warning(f"No valid {stat_type} NGS data loaded for any year")
    
    # Load snap counts
    logger.info(f"Loading snap counts data for years: {years}")
    try:
        snap_data = nfl.import_snap_counts(years)
        data['snap_counts'] = snap_data
        logger.info(f"Loaded snap counts data with {len(snap_data)} rows")
    except Exception as e:
        logger.error(f"Error loading snap counts data: {e}")
        data['snap_counts'] = pd.DataFrame()
    
    return data

def process_player_data(data_dict):
    """
    Process player data for analysis
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing various data frames
        
    Returns:
    --------
    dict
        Dictionary containing processed data frames
    """
    processed_data = {}
    
    # Get player IDs
    player_ids = data_dict.get('player_ids', pd.DataFrame())
    
    # Process seasonal data
    if 'seasonal' in data_dict and not data_dict['seasonal'].empty:
        seasonal = data_dict['seasonal'].copy()
        
        # Add player info
        if not player_ids.empty:
            # Select relevant columns from player_ids
            id_columns = ['gsis_id', 'name', 'position', 'birthdate', 'college', 'height', 'weight']
            id_columns = [col for col in id_columns if col in player_ids.columns]
            
            # Merge data
            seasonal = pd.merge(
                seasonal,
                player_ids[id_columns],
                left_on='player_id',
                right_on='gsis_id',
                how='left'
            )
            
            # Add age column if birthdate is available
            if 'birthdate' in seasonal.columns and 'season' in seasonal.columns:
                try:
                    # Convert birthdate to datetime if it's not already
                    if pd.api.types.is_string_dtype(seasonal['birthdate']):
                        seasonal['birthdate'] = pd.to_datetime(seasonal['birthdate'], errors='coerce')
                    
                    # Calculate age at season
                    seasonal['age'] = seasonal.apply(
                        lambda row: calculate_age(row['birthdate'], row['season']), 
                        axis=1
                    )
                except Exception as e:
                    logger.error(f"Error calculating player ages: {e}")
        
        # Ensure key columns have consistent naming
        seasonal = standardize_columns(seasonal)
        
        # Calculate derived statistics
        seasonal = add_derived_features(seasonal)
        
        processed_data['seasonal'] = seasonal
        logger.info(f"Processed seasonal data with {len(seasonal)} rows")
    
    # Process NGS data
    for ngs_type in ['ngs_passing', 'ngs_rushing', 'ngs_receiving']:
        if ngs_type in data_dict and not data_dict[ngs_type].empty:
            ngs_data = data_dict[ngs_type].copy()
            
            # Add player info if available
            if not player_ids.empty and 'player_gsis_id' in ngs_data.columns:
                # Select relevant columns from player_ids
                id_columns = ['gsis_id', 'name', 'position', 'birthdate']
                id_columns = [col for col in id_columns if col in player_ids.columns]
                
                # Merge data
                ngs_data = pd.merge(
                    ngs_data,
                    player_ids[id_columns],
                    left_on='player_gsis_id',
                    right_on='gsis_id',
                    how='left'
                )
                
                # Add age column if birthdate is available
                if 'birthdate' in ngs_data.columns and 'season' in ngs_data.columns:
                    try:
                        # Convert birthdate to datetime if it's not already
                        if pd.api.types.is_string_dtype(ngs_data['birthdate']):
                            ngs_data['birthdate'] = pd.to_datetime(ngs_data['birthdate'], errors='coerce')
                        
                        # Calculate age at season
                        ngs_data['age'] = ngs_data.apply(
                            lambda row: calculate_age(row['birthdate'], row['season']), 
                            axis=1
                        )
                    except Exception as e:
                        logger.error(f"Error calculating player ages in NGS data: {e}")
            
            # Ensure key columns have consistent naming
            ngs_data = standardize_columns(ngs_data)
            
            # Add player_id column for consistency
            if 'player_id' not in ngs_data.columns and 'player_gsis_id' in ngs_data.columns:
                ngs_data['player_id'] = ngs_data['player_gsis_id']
            
            processed_data[ngs_type] = ngs_data
            logger.info(f"Processed {ngs_type} data with {len(ngs_data)} rows")
    
    # Process weekly data
    if 'weekly' in data_dict and not data_dict['weekly'].empty:
        weekly = data_dict['weekly'].copy()
        
        # Add player info
        if not player_ids.empty:
            # Select relevant columns from player_ids
            id_columns = ['gsis_id', 'name', 'position', 'birthdate']
            id_columns = [col for col in id_columns if col in player_ids.columns]
            
            # Merge data
            weekly = pd.merge(
                weekly,
                player_ids[id_columns],
                left_on='player_id',
                right_on='gsis_id',
                how='left'
            )
            
            # Add age column if birthdate is available
            if 'birthdate' in weekly.columns and 'season' in weekly.columns:
                try:
                    # Convert birthdate to datetime if it's not already
                    if pd.api.types.is_string_dtype(weekly['birthdate']):
                        weekly['birthdate'] = pd.to_datetime(weekly['birthdate'], errors='coerce')
                    
                    # Calculate age at season
                    weekly['age'] = weekly.apply(
                        lambda row: calculate_age(row['birthdate'], row['season']), 
                        axis=1
                    )
                except Exception as e:
                    logger.error(f"Error calculating player ages in weekly data: {e}")
        
        # Ensure key columns have consistent naming
        weekly = standardize_columns(weekly)
        
        # Add derived weekly features
        weekly = add_derived_weekly_features(weekly)
        
        processed_data['weekly'] = weekly
        logger.info(f"Processed weekly data with {len(weekly)} rows")
    
    # Add combined player data for easier analysis
    processed_data = add_combined_player_data(processed_data)
    
    return processed_data

def add_combined_player_data(data_dict):
    """
    Create combined player data merging seasonal and NGS data
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary containing processed data frames
        
    Returns:
    --------
    dict
        Dictionary with additional combined data
    """
    seasonal = data_dict.get('seasonal', pd.DataFrame())
    if seasonal.empty:
        logger.warning("No seasonal data to combine")
        return data_dict
    
    # Make a deep copy of the dictionary
    result = data_dict.copy()
    
    # Combine seasonal data with NGS data by position
    for position, ngs_type in [('QB', 'ngs_passing'), ('RB', 'ngs_rushing'), ('WR', 'ngs_receiving'), ('TE', 'ngs_receiving')]:
        if position == 'QB':
            # QB data = seasonal + NGS passing
            if 'position' in seasonal.columns and ngs_type in data_dict and not data_dict[ngs_type].empty:
                seasonal_pos = seasonal[seasonal['position'] == position].copy()
                ngs_data = data_dict[ngs_type].copy()
                
                # Prepare for merge
                merge_key = 'player_gsis_id' if 'player_gsis_id' in ngs_data.columns else 'player_id'
                
                # Group NGS data by player and season to get averages
                if 'season' in ngs_data.columns:
                    ngs_grouped = ngs_data.groupby(['player_id', 'season']).mean(numeric_only=True).reset_index()
                    
                    # Merge seasonal and NGS data
                    pos_data = pd.merge(
                        seasonal_pos,
                        ngs_grouped,
                        left_on=['player_id', 'season'],
                        right_on=['player_id', 'season'],
                        how='left',
                        suffixes=('', '_ngs')
                    )
                    
                    # Add position suffix for clarity
                    result[f'combined_{position.lower()}'] = pos_data
                    logger.info(f"Created combined {position} data with {len(pos_data)} rows")
                else:
                    logger.warning(f"Cannot group NGS data for {position} - missing season column")
        elif position in ['RB', 'WR', 'TE']:
            # Rushing/receiving positions with respective NGS data
            if 'position' in seasonal.columns and ngs_type in data_dict and not data_dict[ngs_type].empty:
                seasonal_pos = seasonal[seasonal['position'] == position].copy()
                ngs_data = data_dict[ngs_type].copy()
                
                # Group NGS data by player and season to get averages
                if 'season' in ngs_data.columns:
                    ngs_grouped = ngs_data.groupby(['player_id', 'season']).mean(numeric_only=True).reset_index()
                    
                    # Merge seasonal and NGS data
                    pos_data = pd.merge(
                        seasonal_pos,
                        ngs_grouped,
                        left_on=['player_id', 'season'],
                        right_on=['player_id', 'season'],
                        how='left',
                        suffixes=('', '_ngs')
                    )
                    
                    # Add position suffix for clarity
                    result[f'combined_{position.lower()}'] = pos_data
                    logger.info(f"Created combined {position} data with {len(pos_data)} rows")
                else:
                    logger.warning(f"Cannot group NGS data for {position} - missing season column")
    
    # Create filtered data with active players only for predictions
    last_year = seasonal['season'].max() if not seasonal.empty and 'season' in seasonal.columns else None
    
    if last_year:
        active_players = set()
        
        # Get active players from last season's data
        last_season = seasonal[seasonal['season'] == last_year]
        active_players.update(last_season['player_id'].unique())
        
        # Filter seasonal data to active players
        active_seasonal = seasonal[seasonal['player_id'].isin(active_players)].copy()
        result['active_players'] = active_seasonal
        logger.info(f"Created active players dataset with {len(active_seasonal)} players")
    
    return result

def calculate_age(birthdate, season):
    """
    Calculate player age for a given season
    
    Parameters:
    -----------
    birthdate : datetime
        Player's date of birth
    season : int
        NFL season year
        
    Returns:
    --------
    float
        Player's age during the season
    """
    if pd.isnull(birthdate):
        return np.nan
    
    # Use Sept 1 as reference date for NFL season
    reference_date = datetime(season, 9, 1)
    
    # Calculate age in years
    age = reference_date.year - birthdate.year
    
    # Adjust if birthday hasn't occurred yet that year
    if (reference_date.month, reference_date.day) < (birthdate.month, birthdate.day):
        age -= 1
    
    return age

def standardize_columns(df):
    """
    Standardize column names and handle known issues
    
    Parameters:
    -----------
    df : DataFrame
        DataFrame to standardize
        
    Returns:
    --------
    DataFrame
        Standardized DataFrame
    """
    # Create a copy to avoid modifying the original
    df = df.copy()
    
    # Map common NFL stats column variations 
    column_mapping = {
        'rush_att': 'carries',
        'rush_attempts': 'carries',
        'rushing_att': 'carries',
        'rushing_attempts': 'carries',
        'att': 'attempts',
        'passing_att': 'attempts',
        'passing_attempts': 'attempts',
        'rec': 'receptions',
        'receiving_rec': 'receptions',
        'rush_yds': 'rushing_yards',
        'rushing_yds': 'rushing_yards',
        'rec_yds': 'receiving_yards',
        'receiving_yds': 'receiving_yards',
        'pass_yds': 'passing_yards',
        'passing_yds': 'passing_yards',
        'rush_td': 'rushing_tds',
        'rushing_td': 'rushing_tds',
        'rec_td': 'receiving_tds',
        'receiving_td': 'receiving_tds',
        'pass_td': 'passing_tds',
        'passing_td': 'passing_tds',
        'int': 'interceptions',
        'pass_int': 'interceptions',
        'passing_int': 'interceptions',
        'fumble_lost': 'fumbles_lost',
        'games_played': 'games',
        'pos': 'position',
        'player_display_name': 'player_name',
        'player_short_name': 'player_name' 
    }
    
    # Apply column mapping where columns exist
    for old_col, new_col in column_mapping.items():
        if old_col in df.columns and new_col not in df.columns:
            df[new_col] = df[old_col]
    
    # Create missing key columns if needed by using alternative calculations
    if 'carries' not in df.columns and 'rushing_yards' in df.columns and 'rushing_yards_per_att' in df.columns:
        # Calculate carries from yards and yards per attempt
        mask = df['rushing_yards_per_att'] > 0
        df.loc[mask, 'carries'] = df.loc[mask, 'rushing_yards'] / df.loc[mask, 'rushing_yards_per_att']
        df['carries'] = df['carries'].fillna(0)
    
    if 'attempts' not in df.columns and 'passing_yards' in df.columns and 'yards_per_attempt' in df.columns:
        # Calculate attempts from yards and yards per attempt
        mask = df['yards_per_attempt'] > 0
        df.loc[mask, 'attempts'] = df.loc[mask, 'passing_yards'] / df.loc[mask, 'yards_per_attempt'] 
        df['attempts'] = df['attempts'].fillna(0)
    
    # Ensure numeric cols are numeric (sometimes strings slip in)
    numeric_cols = ['carries', 'attempts', 'receptions', 'rushing_yards', 'receiving_yards', 
                    'passing_yards', 'rushing_tds', 'receiving_tds', 'passing_tds', 'interceptions',
                    'fumbles_lost', 'games']
    
    for col in numeric_cols:
        if col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
            except Exception as e:
                logger.error(f"Error converting {col} to numeric: {e}")
    
    return df

def add_derived_features(seasonal_data):
    """
    Add derived features to seasonal data - optimized to avoid fragmentation
    
    Parameters:
    -----------
    seasonal_data : DataFrame
        Seasonal player data
        
    Returns:
    --------
    DataFrame
        Enhanced seasonal data with derived features
    """
    if seasonal_data.empty:
        return seasonal_data
    
    # Create a copy to avoid modifying the original
    df = seasonal_data.copy()
    
    # Dictionary to store all new derived features
    derived_features = {}
    
    # Per game stats
    if 'games' in df.columns:
        for col in ['passing_yards', 'rushing_yards', 'receiving_yards', 
                    'passing_tds', 'rushing_tds', 'receiving_tds',
                    'targets', 'receptions', 'interceptions', 'carries', 'attempts']:
            if col in df.columns:
                derived_features[f'{col}_per_game'] = df[col] / df['games'].clip(lower=1)
    
    # Add position-specific features
    if 'position' in df.columns:
        # QB features
        qb_mask = df['position'] == 'QB'
        if qb_mask.any():
            # Passing efficiency
            if all(col in df.columns for col in ['completions', 'attempts']):
                derived_features['completion_percentage'] = pd.Series(
                    (df.loc[qb_mask, 'completions'] / df.loc[qb_mask, 'attempts'].clip(lower=1)) * 100,
                    index=df.index
                ).where(qb_mask)
            
            if all(col in df.columns for col in ['passing_yards', 'attempts']):
                derived_features['yards_per_attempt'] = pd.Series(
                    df.loc[qb_mask, 'passing_yards'] / df.loc[qb_mask, 'attempts'].clip(lower=1),
                    index=df.index
                ).where(qb_mask)
            
            if all(col in df.columns for col in ['passing_tds', 'attempts']):
                derived_features['td_percentage'] = pd.Series(
                    (df.loc[qb_mask, 'passing_tds'] / df.loc[qb_mask, 'attempts'].clip(lower=1)) * 100,
                    index=df.index
                ).where(qb_mask)
            
            if all(col in df.columns for col in ['interceptions', 'attempts']):
                derived_features['int_percentage'] = pd.Series(
                    (df.loc[qb_mask, 'interceptions'] / df.loc[qb_mask, 'attempts'].clip(lower=1)) * 100,
                    index=df.index
                ).where(qb_mask)
            
            if all(col in df.columns for col in ['passing_yards', 'attempts', 'passing_tds', 'interceptions']):
                # Advanced QB metrics
                derived_features['adjusted_yards_per_attempt'] = pd.Series(
                    ((df.loc[qb_mask, 'passing_yards'] + 
                      (20 * df.loc[qb_mask, 'passing_tds']) - 
                      (45 * df.loc[qb_mask, 'interceptions'])) / 
                    df.loc[qb_mask, 'attempts'].clip(lower=1)),
                    index=df.index
                ).where(qb_mask)
            
            if all(col in df.columns for col in ['passing_tds', 'interceptions']):
                derived_features['td_to_int_ratio'] = pd.Series(
                    df.loc[qb_mask, 'passing_tds'] / df.loc[qb_mask, 'interceptions'].clip(lower=1),
                    index=df.index
                ).where(qb_mask)
        
        # RB features
        rb_mask = df['position'] == 'RB'
        if rb_mask.any():
            if all(col in df.columns for col in ['rushing_yards', 'carries']):
                derived_features['yards_per_carry'] = pd.Series(
                    df.loc[rb_mask, 'rushing_yards'] / df.loc[rb_mask, 'carries'].clip(lower=1),
                    index=df.index
                ).where(rb_mask)
            
            if all(col in df.columns for col in ['rushing_tds', 'carries']):
                derived_features['rushing_td_rate'] = pd.Series(
                    (df.loc[rb_mask, 'rushing_tds'] / df.loc[rb_mask, 'carries'].clip(lower=1)) * 100,
                    index=df.index
                ).where(rb_mask)
            
            if all(col in df.columns for col in ['rushing_yards', 'receiving_yards']):
                # Create total yards
                derived_features['total_yards'] = pd.Series(
                    df.loc[rb_mask, 'rushing_yards'] + df.loc[rb_mask, 'receiving_yards'],
                    index=df.index
                ).where(rb_mask)
                
                # Create total_yards first, then use it for the calculations below
                # This uses the original dataframe columns, not the new derived ones
                
                # Calculate yards distribution
                derived_features['rushing_yards_percentage'] = pd.Series(
                    (df.loc[rb_mask, 'rushing_yards'] / (df.loc[rb_mask, 'rushing_yards'] + 
                                                       df.loc[rb_mask, 'receiving_yards']).clip(lower=1)) * 100,
                    index=df.index
                ).where(rb_mask)
                
                derived_features['receiving_yards_percentage'] = pd.Series(
                    (df.loc[rb_mask, 'receiving_yards'] / (df.loc[rb_mask, 'rushing_yards'] + 
                                                         df.loc[rb_mask, 'receiving_yards']).clip(lower=1)) * 100,
                    index=df.index
                ).where(rb_mask)
            
            if all(col in df.columns for col in ['rushing_yards', 'receiving_yards', 'carries', 'receptions']):
                derived_features['total_yards_per_touch'] = pd.Series(
                    ((df.loc[rb_mask, 'rushing_yards'] + df.loc[rb_mask, 'receiving_yards']) / 
                    (df.loc[rb_mask, 'carries'] + df.loc[rb_mask, 'receptions']).clip(lower=1)),
                    index=df.index
                ).where(rb_mask)
            
            if all(col in df.columns for col in ['receptions', 'targets']):
                derived_features['reception_ratio'] = pd.Series(
                    df.loc[rb_mask, 'receptions'] / df.loc[rb_mask, 'targets'].clip(lower=1),
                    index=df.index
                ).where(rb_mask)
        
        # WR/TE features
        wr_te_mask = df['position'].isin(['WR', 'TE'])
        if wr_te_mask.any():
            if all(col in df.columns for col in ['receiving_yards', 'receptions']):
                derived_features['yards_per_reception'] = pd.Series(
                    df.loc[wr_te_mask, 'receiving_yards'] / df.loc[wr_te_mask, 'receptions'].clip(lower=1),
                    index=df.index
                ).where(wr_te_mask)
            
            if all(col in df.columns for col in ['receiving_yards', 'targets']):
                derived_features['yards_per_target'] = pd.Series(
                    df.loc[wr_te_mask, 'receiving_yards'] / df.loc[wr_te_mask, 'targets'].clip(lower=1),
                    index=df.index
                ).where(wr_te_mask)
            
            if all(col in df.columns for col in ['receiving_tds', 'receptions']):
                derived_features['receiving_td_rate'] = pd.Series(
                    (df.loc[wr_te_mask, 'receiving_tds'] / df.loc[wr_te_mask, 'receptions'].clip(lower=1)) * 100,
                    index=df.index
                ).where(wr_te_mask)
            
            if all(col in df.columns for col in ['receptions', 'targets']):
                derived_features['reception_ratio'] = pd.Series(
                    df.loc[wr_te_mask, 'receptions'] / df.loc[wr_te_mask, 'targets'].clip(lower=1),
                    index=df.index
                ).where(wr_te_mask)
            
            # Advanced receiving metrics
            if all(col in df.columns for col in ['receiving_air_yards', 'receiving_yards']):
                derived_features['air_yards_percentage'] = pd.Series(
                    (df.loc[wr_te_mask, 'receiving_air_yards'] / df.loc[wr_te_mask, 'receiving_yards'].clip(lower=1)) * 100,
                    index=df.index
                ).where(wr_te_mask)
                
                # Calculate air_yards_percentage first, then use it for yac_percentage
                derived_features['yac_percentage'] = 100 - derived_features['air_yards_percentage']
            
            # RACR (Receiver Air Conversion Ratio)
            if all(col in df.columns for col in ['receiving_yards', 'receiving_air_yards']):
                derived_features['racr'] = pd.Series(
                    df.loc[wr_te_mask, 'receiving_yards'] / df.loc[wr_te_mask, 'receiving_air_yards'].clip(lower=1),
                    index=df.index
                ).where(wr_te_mask)
            
            # WOPR (Weighted Opportunity Rating)
            if all(col in df.columns for col in ['target_share', 'air_yards_share']):
                derived_features['wopr'] = pd.Series(
                    (1.5 * df.loc[wr_te_mask, 'target_share']) + (0.7 * df.loc[wr_te_mask, 'air_yards_share']),
                    index=df.index
                ).where(wr_te_mask)
    
    # Add fantasy points per game
    if 'fantasy_points' in df.columns and 'games' in df.columns:
        derived_features['fantasy_points_per_game'] = df['fantasy_points'] / df['games'].clip(lower=1)
    
    # Calculate career trends (if player played multiple seasons)
    if 'season' in df.columns and 'player_id' in df.columns:
        # Ensure data is sorted by player_id and season
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
            'yards_per_carry', 'reception_ratio', 'racr'
        ]
        
        # Check which potential metrics exist in the data
        for metric in potential_metrics:
            if metric in df.columns:
                trend_metrics.append(metric)
        
        # Dictionary to hold all trend-related features
        trend_features = {}
        
        # Process each metric for trend analysis
        for col in trend_metrics:
            if col in df.columns:
                # Previous season value
                trend_features[f'{col}_prev_season'] = df_shifted[col]
                
                # Calculate absolute and percentage change
                trend_features[f'{col}_change'] = df[col] - df_shifted[col]
                # Avoid division by zero or NaN
                trend_features[f'{col}_pct_change'] = (
                    (df[col] - df_shifted[col]) / df_shifted[col].clip(lower=0.1) * 100
                )
                
                # Calculate simple moving average (3 seasons) if we have enough data
                if col in df_shifted_2.columns and col in df_shifted_3.columns:
                    trend_features[f'{col}_3yr_avg'] = (
                        df_shifted[col].fillna(0) + 
                        df_shifted_2[col].fillna(0) + 
                        df_shifted_3[col].fillna(0)
                    ) / 3
                    
                    # Calculate deviation from 3-year average
                    trend_features[f'{col}_vs_3yr_avg'] = df[col] - trend_features[f'{col}_3yr_avg']
                    trend_features[f'{col}_vs_3yr_avg_pct'] = (
                        trend_features[f'{col}_vs_3yr_avg'] / trend_features[f'{col}_3yr_avg'].clip(lower=0.1) * 100
                    )
                
                # Calculate weighted moving average (more weight to recent seasons)
                if col in df_shifted_2.columns and col in df_shifted_3.columns:
                    trend_features[f'{col}_weighted_avg'] = (
                        3 * df_shifted[col].fillna(0) + 
                        2 * df_shifted_2[col].fillna(0) + 
                        1 * df_shifted_3[col].fillna(0)
                    ) / 6
                    
                    # Calculate deviation from weighted average
                    trend_features[f'{col}_vs_weighted_avg'] = df[col] - trend_features[f'{col}_weighted_avg']
                    trend_features[f'{col}_vs_weighted_avg_pct'] = (
                        trend_features[f'{col}_vs_weighted_avg'] / trend_features[f'{col}_weighted_avg'].clip(lower=0.1) * 100
                    )
        
        # Create consistency metrics
        for col in ['fantasy_points', 'fantasy_points_per_game']:
            if col in df.columns and f'{col}_prev_season' in trend_features:
                # Consistency measure (smaller absolute percentage change is more consistent)
                trend_features[f'{col}_consistency'] = 100 - abs(trend_features[f'{col}_pct_change']).clip(upper=100)
        
        # Calculate season count for each player
        derived_features['player_season_count'] = df.groupby('player_id')['season'].transform('count')
        
        # Calculate seasons in league for each player
        if 'age' in df.columns:
            # Calculate age-related trends
            derived_features['seasons_in_league'] = derived_features['player_season_count'] - 1  # 0-indexed
            
            # Create "peak season" indicator based on position
            if 'position' in df.columns:
                conditions = [
                    (df['position'] == 'QB') & (df['age'] >= 28) & (df['age'] <= 32),
                    (df['position'] == 'RB') & (df['age'] >= 24) & (df['age'] <= 27),
                    (df['position'] == 'WR') & (df['age'] >= 25) & (df['age'] <= 29),
                    (df['position'] == 'TE') & (df['age'] >= 26) & (df['age'] <= 30)
                ]
                derived_features['peak_season'] = np.select(conditions, [1, 1, 1, 1], default=0)
                
                # Create career trajectory indicators
                derived_features['early_career'] = (derived_features['seasons_in_league'] <= 2).astype(int)
                derived_features['prime_career'] = ((derived_features['seasons_in_league'] > 2) & 
                                                    (derived_features['seasons_in_league'] <= 6)).astype(int)
                derived_features['late_career'] = (derived_features['seasons_in_league'] > 6).astype(int)
        
        # Add all trend features to derived features
        derived_features.update(trend_features)
    
    # Create a DataFrame with all derived features
    derived_df = pd.DataFrame(derived_features, index=df.index)
    
    # Join derived features with original data
    result = pd.concat([df, derived_df], axis=1)
    
    # Fill NA values in derived features
    for col in derived_features.keys():
        if col in result.columns:
            result[col] = result[col].fillna(0)
    
    return result

def add_derived_weekly_features(weekly_data):
    """
    Add derived features to weekly data for better analysis
    
    Parameters:
    -----------
    weekly_data : DataFrame
        Weekly player data
        
    Returns:
    --------
    DataFrame
        Enhanced weekly data
    """
    if weekly_data.empty:
        return weekly_data
    
    # Create a copy to avoid modifying the original
    enhanced_data = weekly_data.copy()
    
    # Calculate positional benchmarks
    if 'position' in enhanced_data.columns:
        # Group by season, week, and position
        grouped = enhanced_data.groupby(['season', 'week', 'position'])
        
        # Calculate positional averages and ranks for key metrics
        for metric in ['fantasy_points', 'passing_yards', 'rushing_yards', 'receiving_yards']:
            if metric in enhanced_data.columns:
                try:
                    # Calculate position average
                    enhanced_data[f'{metric}_pos_avg'] = grouped[metric].transform('mean')
                    
                    # Calculate standard deviation
                    enhanced_data[f'{metric}_pos_std'] = grouped[metric].transform('std')
                    
                    # Calculate z-score (how many std devs above/below average)
                    enhanced_data[f'{metric}_z_score'] = (enhanced_data[metric] - enhanced_data[f'{metric}_pos_avg']) / enhanced_data[f'{metric}_pos_std'].replace(0, 1)
                    
                    # Calculate percentile rank within position (higher is better)
                    enhanced_data[f'{metric}_pos_rank'] = grouped[metric].transform(lambda x: x.rank(pct=True, ascending=False))
                except Exception as e:
                    logger.error(f"Error calculating positional benchmarks for {metric}: {e}")
    
    # Calculate consistency metrics
    if 'fantasy_points' in enhanced_data.columns and 'player_id' in enhanced_data.columns and 'season' in enhanced_data.columns:
        try:
            # Group by player and season
            player_season = enhanced_data.groupby(['player_id', 'season'])
            
            # Sort data by season and week
            enhanced_data = enhanced_data.sort_values(['season', 'week'])
            
            # Calculate rolling averages and standard deviations
            enhanced_data['fantasy_points_rolling_avg'] = player_season['fantasy_points'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).mean())
            enhanced_data['fantasy_points_rolling_std'] = player_season['fantasy_points'].transform(lambda x: x.shift(1).rolling(window=3, min_periods=1).std())
            
            # Calculate coefficient of variation (lower is more consistent)
            enhanced_data['fantasy_points_cv'] = enhanced_data['fantasy_points_rolling_std'] / enhanced_data['fantasy_points_rolling_avg'].replace(0, 1)
            
            # Calculate consistency score (higher is more consistent)
            enhanced_data['consistency_score'] = 1 - enhanced_data['fantasy_points_cv'].clip(0, 1)
        except Exception as e:
            logger.error(f"Error calculating consistency metrics: {e}")
    
    # Calculate matchup difficulty
    if 'opponent_team' in enhanced_data.columns and 'fantasy_points' in enhanced_data.columns:
        try:
            # Group by opponent, season, and week
            defense_groups = enhanced_data.groupby(['opponent_team', 'season', 'week'])
            
            # Calculate average fantasy points allowed by each defense
            defense_allowed = defense_groups['fantasy_points'].transform('sum')
            
            # Group by opponent and position
            if 'position' in enhanced_data.columns:
                pos_defense_groups = enhanced_data.groupby(['opponent_team', 'position', 'season', 'week'])
                
                # Calculate position-specific fantasy points allowed
                pos_defense_allowed = pos_defense_groups['fantasy_points'].transform('sum')
                
                # Add matchup difficulty score
                enhanced_data['matchup_difficulty'] = pos_defense_allowed
        except Exception as e:
            logger.error(f"Error calculating matchup difficulty: {e}")
    
    # Calculate boom/bust indicators (fantasy points exceeding expected)
    if 'fantasy_points' in enhanced_data.columns and 'fantasy_points_pos_avg' in enhanced_data.columns:
        try:
            # Boom: 30% above position average
            enhanced_data['boom_game'] = (enhanced_data['fantasy_points'] > (enhanced_data['fantasy_points_pos_avg'] * 1.3)).astype(int)
            
            # Bust: 30% below position average
            enhanced_data['bust_game'] = (enhanced_data['fantasy_points'] < (enhanced_data['fantasy_points_pos_avg'] * 0.7)).astype(int)
            
            # Calculate boom/bust ratio for each player-season
            if 'player_id' in enhanced_data.columns and 'season' in enhanced_data.columns:
                player_season_boom = enhanced_data.groupby(['player_id', 'season'])['boom_game'].transform('sum')
                player_season_bust = enhanced_data.groupby(['player_id', 'season'])['bust_game'].transform('sum')
                player_season_games = enhanced_data.groupby(['player_id', 'season'])['boom_game'].transform('count')
                
                enhanced_data['boom_rate'] = player_season_boom / player_season_games
                enhanced_data['bust_rate'] = player_season_bust / player_season_games
                enhanced_data['boom_bust_ratio'] = player_season_boom / player_season_bust.replace(0, 1)
        except Exception as e:
            logger.error(f"Error calculating boom/bust indicators: {e}")
    
    return enhanced_data