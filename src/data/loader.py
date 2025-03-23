"""
Data loading utilities for fantasy football analysis
"""

import pandas as pd
import nfl_data_py as nfl
from espn_api.football import League
import logging
import numpy as np

logger = logging.getLogger(__name__)

def load_espn_league_data(league_id, year, espn_s2=None, swid=None):
    """
    Load league data from ESPN API
    
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
        Dictionary containing league data and settings
    """
    # Initialize connection to ESPN league
    league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
    
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
    
    logger.info(f"Successfully loaded league data for {settings.name}")
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

def load_nfl_historical_data(years, include_weekly=True):
    """
    Load historical NFL stats from nfl_data_py for ALL specified years
    
    Parameters:
    -----------
    years : list
        List of years to pull data for
    include_weekly : bool, optional
        Whether to include weekly data
        
    Returns:
    --------
    dict
        Dictionary containing loaded data
    """
    data = {}
    
    # Load player IDs for mapping first (we'll need this throughout)
    logger.info("Loading player ID mapping data")
    try:
        player_ids = nfl.import_ids()
        # Ensure player_ids has position information
        if 'position' not in player_ids.columns and 'pos' in player_ids.columns:
            player_ids['position'] = player_ids['pos']
        data['player_ids'] = player_ids
        logger.info(f"Loaded player ID data with {len(player_ids)} rows")
    except Exception as e:
        logger.error(f"Error loading player ID data: {e}")
        data['player_ids'] = pd.DataFrame()
    
    # Load seasonal data for ALL years at once
    logger.info(f"Loading seasonal data for ALL years: {years}")
    try:
        seasonal_data = nfl.import_seasonal_data(years)
        # Add season type column if missing
        if 'season_type' not in seasonal_data.columns:
            seasonal_data['season_type'] = 'REG'
        data['seasonal'] = seasonal_data
        logger.info(f"Loaded seasonal data with {len(seasonal_data)} rows")
    except Exception as e:
        logger.error(f"Error loading seasonal data: {e}")
        data['seasonal'] = pd.DataFrame()
    
    # Load weekly data if requested for ALL years at once
    if include_weekly:
        logger.info(f"Loading weekly data for ALL years: {years}")
        try:
            weekly_data = nfl.import_weekly_data(years)
            data['weekly'] = weekly_data
            logger.info(f"Loaded weekly data with {len(weekly_data)} rows")
        except Exception as e:
            logger.error(f"Error loading weekly data: {e}")
            data['weekly'] = pd.DataFrame()
    
    # Load rosters for ALL years at once
    logger.info(f"Loading roster data for ALL years: {years}")
    try:
        roster_data = nfl.import_seasonal_rosters(years)
        data['rosters'] = roster_data
        logger.info(f"Loaded roster data with {len(roster_data)} rows")
    except Exception as e:
        logger.error(f"Error loading roster data: {e}")
        data['rosters'] = pd.DataFrame()
    
    # Load NGS data for ALL historical years
    for stat_type in ['passing', 'rushing', 'receiving']:
        logger.info(f"Loading {stat_type} NGS data for ALL years: {years}")
        try:
            # Load NGS data for each year to avoid potential issues with bulk loading
            all_ngs_data = []
            for year in years:
                logger.info(f"Loading {stat_type} NGS data for year: {year}")
                try:
                    year_data = nfl.import_ngs_data(stat_type, [year])
                    if not year_data.empty:
                        # Add year column if not present
                        if 'season' not in year_data.columns:
                            year_data['season'] = year
                        all_ngs_data.append(year_data)
                except Exception as e:
                    logger.warning(f"Error loading NGS {stat_type} data for year {year}: {e}")
            
            # Combine all years' data
            if all_ngs_data:
                combined_ngs = pd.concat(all_ngs_data, ignore_index=True)
                data[f'ngs_{stat_type}'] = combined_ngs
                logger.info(f"Loaded combined {stat_type} NGS data with {len(combined_ngs)} rows")
            else:
                data[f'ngs_{stat_type}'] = pd.DataFrame()
                logger.warning(f"No valid {stat_type} NGS data loaded for any year")
        except Exception as e:
            logger.error(f"Error loading combined NGS {stat_type} data: {e}")
            data[f'ngs_{stat_type}'] = pd.DataFrame()
    
    return data

def load_nfl_current_season_data(year):
    """
    Load current season NFL stats from nfl_data_py
    
    Parameters:
    -----------
    year : int
        Season year
        
    Returns:
    --------
    dict
        Dictionary containing loaded data
    """
    data = {}
    
    # Load seasonal data if available
    logger.info(f"Loading seasonal data for year: {year}")
    try:
        seasonal_data = nfl.import_seasonal_data([year])
        # Add season type column if missing
        if 'season_type' not in seasonal_data.columns:
            seasonal_data['season_type'] = 'REG'
        data['seasonal'] = seasonal_data
        logger.info(f"Loaded seasonal data with {len(seasonal_data)} rows")
    except Exception as e:
        logger.error(f"Error loading seasonal data: {e}")
        data['seasonal'] = pd.DataFrame()
    
    # Load NGS data for current season
    for stat_type in ['passing', 'rushing', 'receiving']:
        logger.info(f"Loading {stat_type} NGS data for year: {year}")
        try:
            ngs_data = nfl.import_ngs_data(stat_type, [year])
            data[f'ngs_{stat_type}'] = ngs_data
            logger.info(f"Loaded {stat_type} NGS data with {len(ngs_data)} rows")
        except Exception as e:
            logger.error(f"Error loading NGS {stat_type} data: {e}")
            data[f'ngs_{stat_type}'] = pd.DataFrame()
    
    # Load weekly data if available
    logger.info(f"Loading weekly data for year: {year}")
    try:
        weekly_data = nfl.import_weekly_data([year])
        data['weekly'] = weekly_data
        logger.info(f"Loaded weekly data with {len(weekly_data)} rows")
    except Exception as e:
        logger.error(f"Error loading weekly data: {e}")
        data['weekly'] = pd.DataFrame()
    
    # Load current rosters
    logger.info(f"Loading roster data for year: {year}")
    try:
        roster_data = nfl.import_seasonal_rosters([year])
        data['rosters'] = roster_data
        logger.info(f"Loaded roster data with {len(roster_data)} rows")
    except Exception as e:
        logger.error(f"Error loading roster data: {e}")
        data['rosters'] = pd.DataFrame()
    
    return data

def merge_player_data(seasonal_data, player_ids):
    """
    Merge player name and ID information with seasonal data
    
    Parameters:
    -----------
    seasonal_data : DataFrame
        Seasonal stats data
    player_ids : DataFrame
        Player ID mapping data
        
    Returns:
    --------
    DataFrame
        Merged data
    """
    if seasonal_data.empty or player_ids.empty:
        logger.warning("One of the input dataframes is empty")
        return seasonal_data
    
    # Select relevant columns from player_ids
    id_columns = ['gsis_id', 'name', 'position', 'birthdate', 'college']
    id_columns = [col for col in id_columns if col in player_ids.columns]
    
    if 'position' not in id_columns:
        logger.warning("No position column found in player_ids. Trying alternative columns.")
        if 'pos' in player_ids.columns:
            player_ids['position'] = player_ids['pos']
            id_columns.append('position')
    
    # Merge data
    merged_data = pd.merge(
        seasonal_data,
        player_ids[id_columns],
        left_on='player_id',
        right_on='gsis_id',
        how='left'
    )
    
    logger.info(f"Merged player data: {len(merged_data)} rows, Position column exists: {'position' in merged_data.columns}")
    
    return merged_data

def map_ngs_to_player_ids(ngs_data, player_ids):
    """
    Map NGS data to player IDs
    
    Parameters:
    -----------
    ngs_data : DataFrame
        NGS stats data
    player_ids : DataFrame
        Player ID mapping data
        
    Returns:
    --------
    DataFrame
        Merged data
    """
    if ngs_data.empty or player_ids.empty:
        logger.warning("One of the input dataframes is empty")
        return ngs_data
    
    # Ensure position is in player_ids
    if 'position' not in player_ids.columns and 'pos' in player_ids.columns:
        player_ids['position'] = player_ids['pos']
    
    # NGS data typically uses 'player_gsis_id'
    if 'player_gsis_id' in ngs_data.columns:
        merged_data = pd.merge(
            ngs_data,
            player_ids[['gsis_id', 'name', 'position']],
            left_on='player_gsis_id',
            right_on='gsis_id',
            how='left'
        )
        
        # Add player_id column for consistency
        if 'player_id' not in merged_data.columns:
            merged_data['player_id'] = merged_data['player_gsis_id']
    else:
        logger.warning("No player_gsis_id column found in NGS data")
        merged_data = ngs_data
    
    return merged_data

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
        'pos': 'position'
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
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    return df

def clean_and_merge_all_data(historical_data, current_data):
    """
    Clean and merge all data sources
    
    Parameters:
    -----------
    historical_data : dict
        Dictionary containing historical data
    current_data : dict
        Dictionary containing current season data
        
    Returns:
    --------
    dict
        Dictionary containing cleaned and merged data
    """
    result = {}
    
    # Get player IDs from historical data
    player_ids = historical_data.get('player_ids', pd.DataFrame())
    
    # Ensure position column exists in player_ids
    if 'position' not in player_ids.columns and 'pos' in player_ids.columns:
        player_ids['position'] = player_ids['pos']
    
    # Process seasonal data - merging historical and current data
    all_seasonal = []
    
    if 'seasonal' in historical_data and not historical_data['seasonal'].empty:
        hist_seasonal = standardize_columns(historical_data['seasonal'])
        hist_seasonal = merge_player_data(hist_seasonal, player_ids)
        if 'position' not in hist_seasonal.columns:
            logger.warning("Position column missing after merging historical seasonal data")
        all_seasonal.append(hist_seasonal)
        result['historical_seasonal'] = hist_seasonal
    
    if 'seasonal' in current_data and not current_data['seasonal'].empty:
        current_seasonal = standardize_columns(current_data['seasonal'])
        current_seasonal = merge_player_data(current_seasonal, player_ids)
        if 'position' not in current_seasonal.columns:
            logger.warning("Position column missing after merging current seasonal data")
        all_seasonal.append(current_seasonal)
        result['current_seasonal'] = current_seasonal
    
    # Combine all seasonal data
    if all_seasonal:
        combined_seasonal = pd.concat(all_seasonal, ignore_index=True)
        
        # If position still missing, try to add from player_ids based on player_id
        if 'position' not in combined_seasonal.columns and 'player_id' in combined_seasonal.columns:
            logger.info("Attempting to add position from player_ids based on player_id")
            position_map = dict(zip(player_ids['gsis_id'], player_ids['position']))
            combined_seasonal['position'] = combined_seasonal['player_id'].map(position_map)
        
        result['all_seasonal'] = combined_seasonal
        
        # Log column presence for debugging
        logger.info(f"Final all_seasonal columns: {', '.join(combined_seasonal.columns.tolist())}")
        logger.info(f"Position column exists: {'position' in combined_seasonal.columns}")
        logger.info(f"player_id column exists: {'player_id' in combined_seasonal.columns}")
    
    # Process NGS data - both historical and current
    for ngs_type in ['ngs_passing', 'ngs_rushing', 'ngs_receiving']:
        all_ngs = []
        
        if ngs_type in historical_data and not historical_data[ngs_type].empty:
            hist_ngs = map_ngs_to_player_ids(historical_data[ngs_type], player_ids)
            all_ngs.append(hist_ngs)
        
        if ngs_type in current_data and not current_data[ngs_type].empty:
            current_ngs = map_ngs_to_player_ids(current_data[ngs_type], player_ids)
            all_ngs.append(current_ngs)
        
        if all_ngs:
            combined_ngs = pd.concat(all_ngs, ignore_index=True)
            # Ensure player_id and position columns exist
            if 'player_id' not in combined_ngs.columns and 'player_gsis_id' in combined_ngs.columns:
                combined_ngs['player_id'] = combined_ngs['player_gsis_id']
                
            if 'position' not in combined_ngs.columns and 'player_position' in combined_ngs.columns:
                combined_ngs['position'] = combined_ngs['player_position']
                
            result[ngs_type] = combined_ngs
    
    # Process weekly data
    all_weekly = []
    
    if 'weekly' in historical_data and not historical_data['weekly'].empty:
        hist_weekly = standardize_columns(historical_data['weekly'])
        hist_weekly = merge_player_data(hist_weekly, player_ids)
        all_weekly.append(hist_weekly)
    
    if 'weekly' in current_data and not current_data['weekly'].empty:
        current_weekly = standardize_columns(current_data['weekly'])
        current_weekly = merge_player_data(current_weekly, player_ids)
        all_weekly.append(current_weekly)
    
    if all_weekly:
        combined_weekly = pd.concat(all_weekly, ignore_index=True)
        
        # If position still missing, try to add from player_ids based on player_id
        if 'position' not in combined_weekly.columns and 'player_id' in combined_weekly.columns:
            position_map = dict(zip(player_ids['gsis_id'], player_ids['position']))
            combined_weekly['position'] = combined_weekly['player_id'].map(position_map)
            
        result['all_weekly'] = combined_weekly
    
    # Process roster data
    all_rosters = []
    
    if 'rosters' in historical_data and not historical_data['rosters'].empty:
        all_rosters.append(historical_data['rosters'])
    
    if 'rosters' in current_data and not current_data['rosters'].empty:
        all_rosters.append(current_data['rosters'])
    
    if all_rosters:
        result['rosters'] = pd.concat(all_rosters, ignore_index=True)
    
    # Make sure to include player_ids in the result
    result['player_ids'] = player_ids
    
    logger.info(f"Data cleaning and merging complete. Result contains {len(result)} datasets.")
    
    return result