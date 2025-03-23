#!/usr/bin/env python3
"""
Fantasy Football Draft Optimizer

This script orchestrates the complete data analysis pipeline:
1. Loads data from NFL data sources
2. Extracts league settings from ESPN API
3. Engineers features for player performance prediction
4. Performs clustering and tier-based analysis
5. Creates comprehensive visualizations
6. Evaluates projection model performance using proper validation
"""

import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import sys
import re
import warnings
warnings.filterwarnings('ignore')

#######################################################
# CONFIGURATION - Modify these parameters as needed
#######################################################

# ESPN League Information
LEAGUE_ID = 697625923  # Your ESPN fantasy league ID
LEAGUE_YEAR = 2024  # The year to analyze

# ESPN Credentials (required for private leagues)
ESPN_S2 = "AECeHAkR7FZuTvVWSEnMRIA29wVeroZhvd7fHHy5tZvIUdIp4XIZaglA17V6g2rulDDFMCUiC%2BpeXNqWzEJTpjTJsz5Zv2DTMOIjeX0JrC6CYs8kDidYeF0HHkI78OG2O%2Bs6f%2FUPVSwXRBZEGMKPDdKl%2BE0a7na225JN4bC80tD9RXFv32kqqtEk%2Bgw1hgQ968ARiAdt69axAvjryW57rj58sYK4oMJPxjtPbh9tATi%2BSI2AmQ0dNPXZfRTA%2FFgCtgzyxWNwbKT2boYeDfFe7rm8idW47lnavsfYGwWjVpddVY6%2BcupF7uoc9AeVFQ5xNUY%3D"  # Your ESPN_S2 cookie value
SWID = "{42614A28-F6F5-4052-A14A-28F6F52052AF}"  # Your SWID cookie value 

# Data Analysis Parameters
START_YEAR = 2019  # First year to include in historical analysis
INCLUDE_NGS = True  # Whether to include Next Gen Stats (slower but more accurate)
DEBUG_MODE = False  # Enable for verbose logging

# Processing Options
CLUSTER_COUNT = 5  # Number of player tiers/clusters to create
DROP_BOTTOM_TIERS = 1  # Number of bottom tiers to drop
USE_FILTERED = True #Set use the filtered dataset in the training


# Projection Evaluation Parameters
VALIDATION_YEAR = 2023  # Year to use for validation
PROJECTION_YEAR = 2024  # Year to project
PERFORM_CV = True  # Whether to perform cross-validation during model training

# Caching and Reuse Options
USE_CACHED_RAW_DATA = False  # Set to True to use previously downloaded raw data
USE_CACHED_PROCESSED_DATA = False  # Set to True to use previously processed data
USE_CACHED_FEATURE_SETS = False  # Set to True to use previously engineered features
CREATE_VISUALIZATIONS = True  # Whether to generate visualization plots
SKIP_MODEL_TRAINING = False  # Set to True to skip model training/evaluation


# File Paths
CONFIG_PATH = 'configs/league_settings.json'  # Path to optional config file
DATA_DIR = 'data'  # Base directory for all data
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs')  # Directory for visualizations
MODELS_DIR = os.path.join(DATA_DIR, 'models')  # Directory for saved models

#######################################################
# END OF CONFIGURATION
#######################################################

# Set up logging
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fantasy_football.log")
    ]
)
logger = logging.getLogger("fantasy_football")

# Import project modules
from src.data.loader import (
    load_espn_league_data,
    load_nfl_data,
    process_player_data
)
from src.features.engineering import FantasyFeatureEngineering
from src.analysis.visualizer import FantasyDataVisualizer
from src.analysis.ml_explorer import MLExplorer
from src.models.projections import PlayerProjectionModel

def load_config(config_path):
    """Load configuration from file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Error loading config file: {e}")
        return {}

def create_output_dirs():
    """Create organized output directories"""
    # Create base directories
    dirs = {
        'raw': os.path.join(DATA_DIR, 'raw'),
        'processed': os.path.join(DATA_DIR, 'processed'),
        'outputs': OUTPUT_DIR,
        'models': MODELS_DIR,
    }
    
    # Create position-specific directories
    positions = ['qb', 'rb', 'wr', 'te', 'overall']
    
    overall_analysis_types = [
        'time_trends',
        'distributions',
        'league_settings'
    ]
    # Create analysis-type directories
    analysis_types = [
        'correlations', 
        'clusters',
        'feature_importance',
        'time_trends',
        'distributions',
    ]
    
    # Create all directories
    for dir_name, dir_path in dirs.items():
        os.makedirs(dir_path, exist_ok=True)
    
    # Create organized visualization directories
    for position in positions:
        if position == 'overall':
            for analysis_type in overall_analysis_types:
                path = os.path.join(OUTPUT_DIR, position, analysis_type)
                os.makedirs(path, exist_ok=True)
        else:
            for analysis_type in analysis_types:
                path = os.path.join(OUTPUT_DIR, position, analysis_type)
                os.makedirs(path, exist_ok=True)
    
    return dirs

def load_cached_raw_data(raw_data_dir):
    """
    Load previously downloaded raw data files
    
    Parameters:
    -----------
    raw_data_dir : str
        Directory containing raw data files
        
    Returns:
    --------
    dict
        Dictionary containing raw data frames
    """
    logger.info("Loading raw data from cache...")
    raw_data = {}
    
    # List of expected data files
    expected_files = [
        'seasonal.csv', 
        'weekly.csv', 
        'rosters.csv',
        'ngs_passing.csv',
        'ngs_rushing.csv',
        'ngs_receiving.csv',
        'player_ids.csv',
        'schedules.csv'
    ]
    
    # Load each expected file if it exists
    for file_name in expected_files:
        file_path = os.path.join(raw_data_dir, file_name)
        base_name = file_name.split('.')[0]
        
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                raw_data[base_name] = df
                logger.info(f"Loaded {base_name} from {file_path} ({len(df)} rows)")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
        else:
            logger.warning(f"File not found: {file_path}")
    
    return raw_data

def load_cached_processed_data(processed_data_dir):
    """
    Load previously processed data files
    
    Parameters:
    -----------
    processed_data_dir : str
        Directory containing processed data files
        
    Returns:
    --------
    dict
        Dictionary containing processed data frames
    """
    logger.info("Loading processed data from cache...")
    processed_data = {}
    
    # List all CSV files in the directory
    for file_name in os.listdir(processed_data_dir):
        if file_name.endswith(".csv") and not file_name.startswith("features_"):
            file_path = os.path.join(processed_data_dir, file_name)
            base_name = file_name.split('.')[0]
            
            try:
                df = pd.read_csv(file_path)
                processed_data[base_name] = df
                logger.info(f"Loaded {base_name} from {file_path} ({len(df)} rows)")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    
    return processed_data

def load_cached_feature_sets(processed_data_dir):
    """
    Load previously engineered feature sets
    
    Parameters:
    -----------
    processed_data_dir : str
        Directory containing feature set files
        
    Returns:
    --------
    dict
        Dictionary containing feature sets
    """
    logger.info("Loading feature sets from cache...")
    feature_sets = {}
    
    # List all feature set files (those starting with "features_")
    for file_name in os.listdir(processed_data_dir):
        if file_name.startswith("features_") and file_name.endswith(".csv"):
            file_path = os.path.join(processed_data_dir, file_name)
            key = file_name[9:-4]  # Remove "features_" prefix and ".csv" suffix
            
            try:
                df = pd.read_csv(file_path)
                feature_sets[key] = df
                logger.info(f"Loaded feature set {key} from {file_path} ({len(df)} rows)")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")
    
    # Load dropped tiers information if available
    dropped_tiers_path = os.path.join(processed_data_dir, 'dropped_tiers.json')
    if os.path.exists(dropped_tiers_path):
        try:
            with open(dropped_tiers_path, 'r') as f:
                dropped_tiers = json.load(f)
                logger.info(f"Loaded dropped tiers information from {dropped_tiers_path}")
                return feature_sets, dropped_tiers
        except Exception as e:
            logger.error(f"Error loading dropped tiers: {e}")
    
    return feature_sets, None

def evaluate_projections(processed_data, projections, projection_year, output_dir, create_visualizations=True):
    """
    Evaluate projections against actual results
    
    Parameters:
    -----------
    processed_data : dict
        Dictionary of processed data
    projections : dict
        Dictionary of projections by position
    projection_year : int
        Year being projected
    output_dir : str
        Directory to save evaluation results
    create_visualizations : bool
        Whether to create visualization plots
        
    Returns:
    --------
    dict
        Evaluation results by position
    """
    # Get seasonal data
    seasonal = processed_data.get('seasonal', pd.DataFrame())
    if seasonal.empty:
        logger.error("No seasonal data for evaluation")
        return {}
    
    # Check for the actual projection year data
    actual_data = seasonal[seasonal['season'] == projection_year].copy()
    if actual_data.empty:
        logger.error(f"No actual data for {projection_year}")
        return {}
    
    # Print some information about available players to help with debugging
    for position in ['QB', 'RB', 'WR', 'TE']:
        pos_count = len(actual_data[actual_data['position'] == position])
        logger.info(f"Found {pos_count} {position} players in actual data for {projection_year}")
    
    # Evaluate each position
    results = {}
    for position in ['qb', 'rb', 'wr', 'te']:
        if position not in projections:
            logger.warning(f"No projections for {position}")
            continue
        
        # Get position data
        pos_projections = projections[position]
        pos_actual = actual_data[actual_data['position'] == position.upper()]
        
        # Log info about available data
        if 'name' in pos_projections.columns:
            logger.info(f"Projected players for {position}: {len(pos_projections)}")
            # List the top 5 players in projections to help with debugging
            top_proj_players = pos_projections.nlargest(5, 'projected_points')['name'].tolist()
            logger.info(f"Top 5 projected {position} players: {top_proj_players}")
        
        if 'name' in pos_actual.columns:
            logger.info(f"Actual players for {position}: {len(pos_actual)}")
            # List the top 5 players in actual data to help with debugging
            if 'fantasy_points_per_game' in pos_actual.columns:
                top_actual_players = pos_actual.nlargest(5, 'fantasy_points_per_game')['name'].tolist()
                logger.info(f"Top 5 actual {position} players: {top_actual_players}")
        
        if 'name' not in pos_projections.columns or 'name' not in pos_actual.columns:
            logger.warning(f"Missing name column for {position}")
            continue
        
        # Try different ways to standardize names for better matching
        pos_projections['std_name'] = pos_projections['name'].str.lower().str.strip()
        pos_actual['std_name'] = pos_actual['name'].str.lower().str.strip()
        
        # Merge projections with actual data using standardized names
        merged = pd.merge(
            pos_projections[['name', 'std_name', 'projected_points', 'projection_low', 'projection_high']],
            pos_actual[['name', 'std_name', 'fantasy_points_per_game']],
            on='std_name',
            how='inner',
            suffixes=('_proj', '_actual')
        )
        
        # If we couldn't find matches using standard method, try fuzzy matching
        if merged.empty and 'name_proj' in merged.columns and 'name_actual' in merged.columns:
            logger.warning(f"No exact name matches for {position}, trying name cleaning...")
            
            # Clean names further by removing suffixes, Jr., etc.
            def clean_name(name):
                if not isinstance(name, str):
                    return ""
                # Remove suffixes, numbers, and common abbrevations
                name = re.sub(r'\s+(Jr\.|Sr\.|I{1,3}|IV|V)$', '', name)
                name = re.sub(r'\s+\d+$', '', name)
                return name.strip()
            
            pos_projections['clean_name'] = pos_projections['name'].apply(clean_name).str.lower()
            pos_actual['clean_name'] = pos_actual['name'].apply(clean_name).str.lower()
            
            # Try merging on cleaned names
            merged = pd.merge(
                pos_projections[['name', 'clean_name', 'projected_points', 'projection_low', 'projection_high']],
                pos_actual[['name', 'clean_name', 'fantasy_points_per_game']],
                on='clean_name',
                how='inner',
                suffixes=('_proj', '_actual')
            )
        
        if merged.empty:
            logger.warning(f"No matched players for {position}")
            continue
        
        logger.info(f"Successfully matched {len(merged)} {position} players for evaluation")
        
        # Use the actual name for final output 
        if 'name_proj' in merged.columns and 'name_actual' in merged.columns:
            merged['name'] = merged['name_actual']
        
        # Before calculating metrics, filter out players with zero or near-zero projections
        min_projection_threshold = 0.5  # Adjust this threshold as needed
        filtered_merged = merged[merged['projected_points'] >= min_projection_threshold].copy()
        
        # Log the filtering impact
        original_count = len(merged)
        filtered_count = len(filtered_merged)
        logger.info(f"Filtered evaluation data from {original_count} to {filtered_count} players")
        logger.info(f"Removed {original_count - filtered_count} players with projections below {min_projection_threshold}")
        
        # Use filtered data for metrics calculation
        error = filtered_merged['fantasy_points_per_game'] - filtered_merged['projected_points']
        abs_error = error.abs()
        
        # Calculate projection accuracy metrics
        metrics = {
            'n_players': len(filtered_merged),
            'mae': abs_error.mean(),
            'rmse': np.sqrt((error ** 2).mean()),
            'mean_error': error.mean(),  # Positive means underprojection
            'median_error': error.median(),
            'max_error': error.max(),
            'min_error': error.min(),
            'correlation': filtered_merged['fantasy_points_per_game'].corr(filtered_merged['projected_points']),
            'within_range': (
                (filtered_merged['fantasy_points_per_game'] >= filtered_merged['projection_low']) & 
                (filtered_merged['fantasy_points_per_game'] <= filtered_merged['projection_high'])
            ).mean() * 100  # Percentage of players whose actual points were within projected range
        }
        
        # Add player-level evaluation
        metrics['players'] = filtered_merged.copy()
        
        # Log results
        logger.info(f"{position} projection evaluation:")
        logger.info(f"  Players: {metrics['n_players']}")
        logger.info(f"  MAE: {metrics['mae']:.2f}")
        logger.info(f"  RMSE: {metrics['rmse']:.2f}")
        logger.info(f"  Mean Error: {metrics['mean_error']:.2f}")
        logger.info(f"  Correlation: {metrics['correlation']:.3f}")
        logger.info(f"  % Within Range: {metrics['within_range']:.1f}%")
        
        # Save player-level evaluation to CSV
        merged_path = os.path.join(output_dir, f'{position}_projection_evaluation.csv')
        filtered_merged[['name', 'projected_points', 'fantasy_points_per_game', 'projection_low', 'projection_high']].to_csv(merged_path, index=False)
        
        # Save the full (unfiltered) data too for comparison
        unfiltered_path = os.path.join(output_dir, f'{position}_projection_evaluation_unfiltered.csv')
        merged[['name', 'projected_points', 'fantasy_points_per_game', 'projection_low', 'projection_high']].to_csv(unfiltered_path, index=False)
        
        results[position] = metrics
        
        # Create visualization if requested
        if create_visualizations:
            # Create scatter plot of projected vs actual using filtered data
            plt.figure(figsize=(10, 8))
            
            # Plot the points
            plt.scatter(
                filtered_merged['projected_points'], 
                filtered_merged['fantasy_points_per_game'],
                alpha=0.7
            )
            
            # Add diagonal line (perfect projection)
            max_val = max(
                filtered_merged['projected_points'].max(),
                filtered_merged['fantasy_points_per_game'].max()
            ) * 1.1
            plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
            
            # Add labels and title
            plt.xlabel('Projected Fantasy Points per Game')
            plt.ylabel('Actual Fantasy Points per Game')
            plt.title(f'{position.upper()} Projection Accuracy - {projection_year}')
            
            # Add correlation and error info
            corr = filtered_merged['projected_points'].corr(filtered_merged['fantasy_points_per_game'])
            plt.annotate(
                f"Correlation: {corr:.3f}\nMAE: {metrics['mae']:.2f}",
                xy=(0.05, 0.95),
                xycoords='axes fraction',
                fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
            )
            
            # Add player labels for top and bottom performers
            for _, player in filtered_merged.nlargest(3, 'fantasy_points_per_game').iterrows():
                plt.annotate(
                    player['name'],
                    (player['projected_points'], player['fantasy_points_per_game']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='lightgreen', alpha=0.7)
                )
                
            for _, player in filtered_merged.nsmallest(3, 'fantasy_points_per_game').iterrows():
                plt.annotate(
                    player['name'],
                    (player['projected_points'], player['fantasy_points_per_game']),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.1", facecolor='lightcoral', alpha=0.7)
                )
            
            # Add regression line
            sns.regplot(
                x='projected_points',
                y='fantasy_points_per_game',
                data=filtered_merged,
                scatter=False,
                line_kws={'color': 'blue'}
            )
            
            # Add grid for better readability
            plt.grid(True, alpha=0.3)
            
            # Save the plot
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, f'{position}_projection_accuracy.png'))
            plt.close()
            
            # Also create an unfiltered version for comparison
            if len(merged) > len(filtered_merged):
                plt.figure(figsize=(10, 8))
                
                # Plot the points
                plt.scatter(
                    merged['projected_points'], 
                    merged['fantasy_points_per_game'],
                    alpha=0.7
                )
                
                # Add diagonal line (perfect projection)
                max_val = max(
                    merged['projected_points'].max(),
                    merged['fantasy_points_per_game'].max()
                ) * 1.1
                plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
                
                # Add labels and title
                plt.xlabel('Projected Fantasy Points per Game')
                plt.ylabel('Actual Fantasy Points per Game')
                plt.title(f'{position.upper()} Projection Accuracy (Unfiltered) - {projection_year}')
                
                # Add correlation and error info
                unfiltered_corr = merged['projected_points'].corr(merged['fantasy_points_per_game'])
                unfiltered_mae = (merged['fantasy_points_per_game'] - merged['projected_points']).abs().mean()
                plt.annotate(
                    f"Correlation: {unfiltered_corr:.3f}\nMAE: {unfiltered_mae:.2f}",
                    xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
                )
                
                # Add regression line
                sns.regplot(
                    x='projected_points',
                    y='fantasy_points_per_game',
                    data=merged,
                    scatter=False,
                    line_kws={'color': 'blue'}
                )
                
                # Add grid for better readability
                plt.grid(True, alpha=0.3)
                
                # Save the unfiltered plot
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'{position}_projection_accuracy_unfiltered.png'))
                plt.close()
    
    # Create overall evaluation summary if requested
    if results and create_visualizations:
        plt.figure(figsize=(10, 6))
        
        # Gather metrics for all positions
        positions = list(results.keys())
        metrics_data = {
            'MAE': [results[pos]['mae'] for pos in positions],
            'RMSE': [results[pos]['rmse'] for pos in positions],
            'Correlation': [results[pos]['correlation'] for pos in positions],
            'Within Range (%)': [results[pos]['within_range'] for pos in positions]
        }
        
        # Create bar chart
        x = np.arange(len(positions))
        width = 0.2  # Narrower bars to fit more metrics
        
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.bar(x - 1.5*width, metrics_data['MAE'], width, label='MAE')
        ax.bar(x - 0.5*width, metrics_data['RMSE'], width, label='RMSE')
        ax.bar(x + 0.5*width, metrics_data['Correlation'], width, label='Correlation')
        ax.bar(x + 1.5*width, [value / 10 for value in metrics_data['Within Range (%)']], width, label='Within Range (% ÷ 10)')        
        # Add labels and title
        ax.set_xlabel('Position')
        ax.set_ylabel('Value')
        ax.set_title(f'Projection Accuracy by Position - {projection_year}')
        ax.set_xticks(x)
        ax.set_xticklabels([p.upper() for p in positions])
        ax.legend()
        
        # Add values on top of bars
        for i, metric in enumerate(['MAE', 'RMSE', 'Correlation', 'Within Range (%)']):
            for j, pos in enumerate(positions):
                value = metrics_data[metric][j]
                if metric == 'Within Range (%)':
                    value = value / 10  # Scale down for display
                    text = f"{metrics_data[metric][j]:.1f}%"
                else:
                    text = f"{value:.2f}"
                
                ax.text(x[j] + (i-1.5)*width, value + 0.1, text,
                       ha='center', va='bottom', fontsize=8, rotation=45)
        
        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        
        # Save the plot
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'projection_metrics_summary.png'))
        plt.close()
    
    return results


def main():
    """Main function to run the fantasy football analysis pipeline"""
    start_time = datetime.now()
    logger.info(f"Starting fantasy football analysis at {start_time}")
    
    # Load configuration from file (optional, will use defaults if not found)
    config = load_config(CONFIG_PATH)
    
    # Use config values if available, otherwise use the ones defined at the top
    league_id = config.get('league_id', LEAGUE_ID)
    year = config.get('year', LEAGUE_YEAR)
    espn_s2 = config.get('espn_s2', ESPN_S2)
    swid = config.get('swid', SWID)
    start_year = config.get('start_year', START_YEAR)
    
    # Cache options from config
    use_cached_raw = config.get('use_cached_raw_data', USE_CACHED_RAW_DATA)
    use_cached_processed = config.get('use_cached_processed_data', USE_CACHED_PROCESSED_DATA)
    use_cached_features = config.get('use_cached_feature_sets', USE_CACHED_FEATURE_SETS)
    create_viz = config.get('create_visualizations', CREATE_VISUALIZATIONS)
    skip_model = config.get('skip_model_training', SKIP_MODEL_TRAINING)
    
    if league_id is None:
        logger.error("League ID is required. Set LEAGUE_ID at the top of the script.")
        return
    
    # Create output directories with improved organization
    dirs = create_output_dirs()
    
    # Initialize containers for data
    nfl_data = {}
    processed_data = {}
    feature_sets = {}
    dropped_tiers = None
    
    # Step 1: Load league data from ESPN
    logger.info(f"Step 1: Loading ESPN league data for league {league_id}, year {year}...")
    league_data = load_espn_league_data(league_id, year, espn_s2, swid)
    
    # Save league data
    with open(os.path.join(dirs['raw'], 'league_data.json'), 'w') as f:
        # Convert DataFrame to list for JSON serialization
        if 'teams' in league_data and isinstance(league_data['teams'], pd.DataFrame):
            league_data_json = league_data.copy()
            league_data_json['teams'] = league_data_json['teams'].to_dict(orient='records')
            json.dump(league_data_json, f, indent=2)
        else:
            json.dump(league_data, f, indent=2)
    
    # Step 2: Load NFL data (either from cache or fresh download)
    years = list(range(start_year, year + 1))
    
    if use_cached_raw:
        logger.info("Using cached raw data...")
        nfl_data = load_cached_raw_data(dirs['raw'])
        if not nfl_data:
            logger.warning("No cached raw data found or error loading. Falling back to fresh download.")
            use_cached_raw = False
    
    if not use_cached_raw:
        logger.info(f"Step 2: Loading NFL data for years {start_year}-{year}...")
        nfl_data = load_nfl_data(
            years=years,
            include_ngs=INCLUDE_NGS,
            ngs_min_year=2016,
            use_threads=True
        )
        
        # Save raw data
        for key, df in nfl_data.items():
            if not df.empty and isinstance(df, pd.DataFrame):
                output_path = os.path.join(dirs['raw'], f"{key}.csv")
                df.to_csv(output_path, index=False)
                logger.info(f"Saved raw {key} data to {output_path}")
    
    # Step 3: Process player data (either from cache or process raw data)
    if use_cached_processed and not use_cached_features:
        logger.info("Using cached processed data...")
        processed_data = load_cached_processed_data(dirs['processed'])
        if not processed_data:
            logger.warning("No cached processed data found or error loading. Will process raw data.")
            use_cached_processed = False
    
    if not use_cached_processed and not use_cached_features:
        logger.info("Step 3: Processing player data...")
        processed_data = process_player_data(nfl_data)
        
        # Save processed data
        for key, df in processed_data.items():
            if not df.empty and isinstance(df, pd.DataFrame):
                output_path = os.path.join(dirs['processed'], f"{key}.csv")
                df.to_csv(output_path, index=False)
                logger.info(f"Saved processed {key} data to {output_path}")
    
    # Step 4: Feature engineering (either from cache or run engineering)
    if use_cached_features:
        logger.info("Using cached feature sets...")
        feature_sets, dropped_tiers = load_cached_feature_sets(dirs['processed'])
        if not feature_sets:
            logger.warning("No cached feature sets found or error loading. Will run feature engineering.")
            use_cached_features = False
    
    if not use_cached_features:
        # Ensure we have processed data
        if not processed_data and use_cached_processed:
            logger.info("Loading processed data for feature engineering...")
            processed_data = load_cached_processed_data(dirs['processed'])
        
        if not processed_data:
            logger.error("No processed data available for feature engineering.")
            return
            
        logger.info("Step 4: Performing feature engineering...")
        feature_eng = FantasyFeatureEngineering(processed_data, target_year=year)
        
        # Create position-specific features
        feature_eng.create_position_features()
        
        # Prepare prediction features
        feature_eng.prepare_prediction_features()
        
        # Perform clustering with position-specific tier drops
        # Drop 2 tiers for QBs (more depth available) and 1 tier for other positions
        position_tier_drops = {'qb': 2, 'rb': 1, 'wr': 1, 'te': 1}
        feature_eng.cluster_players(n_clusters=CLUSTER_COUNT, position_tier_drops=position_tier_drops)
        
        # Check if filtered feature sets were created and create them if missing
        for position in ['qb', 'rb', 'wr', 'te']:
            filtered_key = f"{position}_train_filtered"
            if filtered_key not in feature_eng.feature_sets or feature_eng.feature_sets[filtered_key].empty:
                # If filtered sets don't exist, create them manually
                logger.warning(f"Creating missing {filtered_key} feature set manually")
                train_key = f"{position}_train"
                if train_key in feature_eng.feature_sets and not feature_eng.feature_sets[train_key].empty:
                    # Copy the unfiltered dataset as a fallback
                    feature_eng.feature_sets[filtered_key] = feature_eng.feature_sets[train_key].copy()
                    
                    # Add required columns if missing
                    if 'tier' not in feature_eng.feature_sets[filtered_key].columns:
                        # Create a basic tier assignment
                        if 'fantasy_points_per_game' in feature_eng.feature_sets[filtered_key].columns:
                            # Divide players into tiers based on fantasy points
                            feature_eng.feature_sets[filtered_key]['cluster'] = 0  # Default cluster
                            feature_eng.feature_sets[filtered_key]['tier'] = 'Mid Tier'  # Default tier
                            
                            # Assign Elite tier to top players
                            top_mask = feature_eng.feature_sets[filtered_key]['fantasy_points_per_game'] > \
                                feature_eng.feature_sets[filtered_key]['fantasy_points_per_game'].quantile(0.8)
                            feature_eng.feature_sets[filtered_key].loc[top_mask, 'tier'] = 'Elite'
                        else:
                            # If no fantasy points, just use default tier
                            feature_eng.feature_sets[filtered_key]['cluster'] = 0
                            feature_eng.feature_sets[filtered_key]['tier'] = 'Mid Tier'
                    
                    logger.info(f"Created fallback {filtered_key} feature set with {len(feature_eng.feature_sets[filtered_key])} rows")
        
        # Finalize features with filtering applied
        feature_eng.finalize_features(apply_filtering=USE_FILTERED)
        
        # Get processed feature sets
        feature_sets = feature_eng.get_feature_sets()
        
        # Save feature sets
        for key, df in feature_sets.items():
            if not df.empty and isinstance(df, pd.DataFrame):
                output_path = os.path.join(dirs['processed'], f"features_{key}.csv")
                df.to_csv(output_path, index=False)
                logger.info(f"Saved feature set {key} to {output_path}")
        
        # Get cluster models and dropped tiers
        cluster_models = feature_eng.get_cluster_models()
        dropped_tiers = feature_eng.dropped_tiers
        
        # Save the dropped tier information for future reference
        with open(os.path.join(dirs['processed'], 'dropped_tiers.json'), 'w') as f:
            json.dump({pos: [int(x) for x in tiers] for pos, tiers in dropped_tiers.items()}, f)
    
    # Step 5: Create visualizations if requested
    if create_viz:
        logger.info("Step 5: Creating visualizations...")
        
        # Basic visualizations
        logger.info("Creating basic visualizations...")
        visualizer = FantasyDataVisualizer(
            data_dict=processed_data,
            feature_sets=feature_sets,
            output_dir=OUTPUT_DIR
        )
        
        # Run basic visualizations
        visualizer.run_all_visualizations(league_data)
        
        # Advanced ML-focused visualizations
        logger.info("Creating advanced ML visualizations...")
        ml_explorer = MLExplorer(
            data_dict=processed_data,
            feature_sets=feature_sets,
            output_dir=OUTPUT_DIR
        )
        
        # Run advanced ML analyses
        ml_explorer.run_advanced_eda()
    else:
        logger.info("Skipping visualizations as specified in configuration")
    
    # Step 6: Train and evaluate projection models if requested
    if not skip_model:
        logger.info("Step 6: Training and evaluating projection models...")
        
        # Initialize the projection model
        projection_model = PlayerProjectionModel(feature_sets, output_dir=MODELS_DIR, use_filtered=USE_FILTERED)

        
        # Train models for all positions with proper validation
        all_metrics = projection_model.train_all_positions(
            model_type='random_forest',
            validation_season=VALIDATION_YEAR,
            perform_cv=PERFORM_CV,
            evaluate_overfit=True 
        )
        
        # Log validation metrics
        logger.info("Model validation metrics:")
        for position, metrics in all_metrics.items():
            logger.info(f"  {position.upper()}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.2f}")
        
        # Create projection filters for specific year evaluation
        projection_filters = {}
        for position in ['qb', 'rb', 'wr', 'te']:
            # Look for projection data in the correct feature sets (position_projection)
            projection_key = f"{position}_projection"  # Use projection data directly
            if projection_key in feature_sets and not feature_sets[projection_key].empty:
                projection_filters[position] = feature_sets[projection_key]
        
        # Generate projections
        logger.info(f"Generating projections for {PROJECTION_YEAR}...")
        projections = projection_model.generate_full_projections(projection_filters)
        
        # Make sure we have processed data for evaluation
        if use_cached_features and (not processed_data or 'seasonal' not in processed_data or processed_data['seasonal'].empty):
            logger.info("Loading seasonal data for projection evaluation...")
            # Try to load seasonal data from processed directory
            seasonal_path = os.path.join(dirs['processed'], 'seasonal.csv')
            if os.path.exists(seasonal_path):
                try:
                    seasonal_data = pd.read_csv(seasonal_path)
                    if not seasonal_data.empty:
                        if not processed_data:
                            processed_data = {}
                        processed_data['seasonal'] = seasonal_data
                        logger.info(f"Loaded seasonal data from {seasonal_path} ({len(seasonal_data)} rows)")
                except Exception as e:
                    logger.error(f"Error loading seasonal data: {e}")
            
            # If still no seasonal data, try loading from raw directory and processing
            if 'seasonal' not in processed_data or processed_data['seasonal'].empty:
                raw_seasonal_path = os.path.join(dirs['raw'], 'seasonal.csv')
                if os.path.exists(raw_seasonal_path):
                    try:
                        raw_seasonal = pd.read_csv(raw_seasonal_path)
                        if not raw_seasonal.empty:
                            logger.info("Processing raw seasonal data...")
                            # Get player IDs for mapping
                            player_ids_path = os.path.join(dirs['raw'], 'player_ids.csv')
                            if os.path.exists(player_ids_path):
                                player_ids = pd.read_csv(player_ids_path)
                                if not player_ids.empty:
                                    # Merge player names and positions
                                    seasonal_data = pd.merge(
                                        raw_seasonal,
                                        player_ids[['gsis_id', 'name', 'position']],
                                        left_on='player_id',
                                        right_on='gsis_id',
                                        how='left'
                                    )
                                    if not processed_data:
                                        processed_data = {}
                                    processed_data['seasonal'] = seasonal_data
                                    logger.info(f"Processed raw seasonal data ({len(seasonal_data)} rows)")
                    except Exception as e:
                        logger.error(f"Error processing raw seasonal data: {e}")
        
        # Check if we have the necessary data for evaluation
        if not processed_data or 'seasonal' not in processed_data or processed_data['seasonal'].empty:
            logger.error(f"Missing seasonal data for evaluation. Cannot evaluate projections for {PROJECTION_YEAR}.")
            evaluation_results = {}
        else:
            # Evaluate projections against actual results
            logger.info("Evaluating projection accuracy...")
            evaluation_results = evaluate_projections(
                processed_data, 
                projections, 
                PROJECTION_YEAR,
                output_dir=MODELS_DIR
            )
        
        # Store results for return
        results = {
            "all_metrics": all_metrics,
            "projections": projections,
            "evaluation_results": evaluation_results
        }
    else:
        logger.info("Skipping model training and evaluation as specified in configuration")
        results = {}
    
    # Summarize results
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Analysis completed in {duration}")
    
    # Display summary of analysis
    print("\n========== Fantasy Football Analysis Summary ==========")
    print(f"League: {league_data.get('league_info', {}).get('name', 'Unknown')}")
    print(f"Teams: {league_data.get('league_info', {}).get('team_count', 'Unknown')}")
    print(f"Analysis years: {min(years)}-{max(years)}")
    
    # Count players in each position tier after filtering
    for position in ['qb', 'rb', 'wr', 'te']:
        tier_key = f"{position}_train_filtered"
        if tier_key in feature_sets and not feature_sets[tier_key].empty:
            tier_counts = feature_sets[tier_key]['tier'].value_counts()
            print(f"\n{position.upper()} Tier distribution:")
            for tier, count in tier_counts.items():
                print(f"  {tier}: {count} players")
    
    # Highlight top players in each position
    print("\nTop players by position after tier filtering:")
    for position in ['qb', 'rb', 'wr', 'te']:
        tier_key = f"{position}_train_filtered"
        if tier_key in feature_sets and not feature_sets[tier_key].empty:
            if 'fantasy_points_per_game' in feature_sets[tier_key].columns and 'name' in feature_sets[tier_key].columns:
                top_players = feature_sets[tier_key].nlargest(5, 'fantasy_points_per_game')
                print(f"\nTop 5 {position.upper()} players:")
                for _, player in top_players.iterrows():
                    print(f"  {player['name']}: {player['fantasy_points_per_game']:.2f} pts/game")
    
    # Display projection model evaluation results if available
    if not skip_model and 'all_metrics' in results:
        print("\nProjection Model Evaluation:")
        for position, metrics in results['all_metrics'].items():
            print(f"\n{position.upper()} Validation Metrics:")
            print(f"  RMSE: {metrics['rmse']:.2f}")
            print(f"  MAE: {metrics['mae']:.2f}")
            print(f"  R²: {metrics['r2']:.2f}")
    
        # Display projection accuracy evaluation if available
        if 'evaluation_results' in results:
            print("\nProjection Accuracy Evaluation:")
            for position, eval_results in results['evaluation_results'].items():
                print(f"\n{position.upper()} Projection Accuracy:")
                print(f"  Players Evaluated: {eval_results['n_players']}")
                print(f"  MAE: {eval_results['mae']:.2f}")
                print(f"  RMSE: {eval_results['rmse']:.2f}")
                print(f"  Mean Error: {eval_results['mean_error']:.2f}")
                print(f"  % Within Range: {eval_results['within_range']:.1f}%")
    
    print("\nAnalysis completed successfully!")
    print(f"Output data and visualizations saved to the '{DATA_DIR}' directory")
    
    return {
        "league_data": league_data,
        "nfl_data": nfl_data if not use_cached_raw else None,
        "processed_data": processed_data,
        "feature_sets": feature_sets,
        "dropped_tiers": dropped_tiers,
        "results": results
    }

if __name__ == "__main__":
    main()
    
# def test_fantasy_points_calculation(player_name, league_id, year=2024, espn_s2=None, swid=None):
#     """
#     Test function to compare fantasy points calculation between nfl_data_py and ESPN API
    
#     Parameters:
#     -----------
#     player_name : str
#         Name of the player to check (must match both in nfl_data_py and ESPN)
#     league_id : int
#         ESPN league ID
#     year : int
#         Season year to check (default 2024)
#     espn_s2 : str, optional
#         ESPN S2 cookie for private leagues
#     swid : str, optional
#         SWID cookie for private leagues
        
#     Returns:
#     --------
#     dict
#         Comparison of fantasy points calculations
#     """
#     import nfl_data_py as nfl
#     from espn_api.football import League
#     import pandas as pd
    
#     print(f"Comparing fantasy points calculation for {player_name} in {year}")
    
#     # Step 1: Get data from nfl_data_py
#     try:
#         # Import seasonal data for the specified year
#         nfl_seasonal = nfl.import_seasonal_data([year])
        
#         # Get player IDs for name matching
#         player_ids = nfl.import_ids()
        
#         # Merge to get player names
#         nfl_data = pd.merge(
#             nfl_seasonal,
#             player_ids[['gsis_id', 'name', 'position']],
#             left_on='player_id',
#             right_on='gsis_id',
#             how='left'
#         )
        
#         # Find the player by name (case-insensitive)
#         player_mask = nfl_data['name'].str.lower() == player_name.lower()
#         if not player_mask.any():
#             # Try partial matching
#             player_mask = nfl_data['name'].str.lower().str.contains(player_name.lower())
#             if not player_mask.any():
#                 print(f"Player {player_name} not found in nfl_data_py data")
#                 return None
        
#         player_nfl_data = nfl_data[player_mask].copy()
        
#         if len(player_nfl_data) > 1:
#             print(f"Multiple matches found for {player_name}. Using first match.")
#             print(f"Matched players: {player_nfl_data['name'].tolist()}")
#             player_nfl_data = player_nfl_data.iloc[0:1]
        
#         print(f"Found player in nfl_data_py: {player_nfl_data['name'].iloc[0]} ({player_nfl_data['position'].iloc[0]})")
        
#         # Step 2: Calculate fantasy points using our method
#         # This is a simplified version - you'd need to implement the actual scoring system
#         nfl_fantasy_points = 0
        
#         # Basic scoring - adjust according to your league settings
#         if 'passing_yards' in player_nfl_data.columns:
#             nfl_fantasy_points += player_nfl_data['passing_yards'].iloc[0] * 0.04  # 1 point per 25 yards
        
#         if 'passing_tds' in player_nfl_data.columns:
#             nfl_fantasy_points += player_nfl_data['passing_tds'].iloc[0] * 4  # 4 points per TD
            
#         if 'interceptions' in player_nfl_data.columns:
#             nfl_fantasy_points += player_nfl_data['interceptions'].iloc[0] * -2  # -2 points per INT
            
#         if 'rushing_yards' in player_nfl_data.columns:
#             nfl_fantasy_points += player_nfl_data['rushing_yards'].iloc[0] * 0.1  # 1 point per 10 yards
            
#         if 'rushing_tds' in player_nfl_data.columns:
#             nfl_fantasy_points += player_nfl_data['rushing_tds'].iloc[0] * 6  # 6 points per TD
            
#         if 'receiving_yards' in player_nfl_data.columns:
#             nfl_fantasy_points += player_nfl_data['receiving_yards'].iloc[0] * 0.1  # 1 point per 10 yards
            
#         if 'receiving_tds' in player_nfl_data.columns:
#             nfl_fantasy_points += player_nfl_data['receiving_tds'].iloc[0] * 6  # 6 points per TD
            
#         if 'receptions' in player_nfl_data.columns:
#             nfl_fantasy_points += player_nfl_data['receptions'].iloc[0] * 0.5  # 0.5 points per reception (PPR)
        
#         # Calculate points per game
#         if 'games' in player_nfl_data.columns and player_nfl_data['games'].iloc[0] > 0:
#             nfl_fantasy_ppg = nfl_fantasy_points / player_nfl_data['games'].iloc[0]
#         else:
#             nfl_fantasy_ppg = nfl_fantasy_points
        
#         # Get fantasy points from nfl_data_py if available
#         nfl_provided_fantasy = None
#         if 'fantasy_points_ppr' in player_nfl_data.columns:
#             nfl_provided_fantasy = player_nfl_data['fantasy_points_ppr'].iloc[0]
#         elif 'fantasy_points' in player_nfl_data.columns:
#             nfl_provided_fantasy = player_nfl_data['fantasy_points'].iloc[0]
        
#         nfl_provided_ppg = None
#         if nfl_provided_fantasy is not None and 'games' in player_nfl_data.columns and player_nfl_data['games'].iloc[0] > 0:
#             nfl_provided_ppg = nfl_provided_fantasy / player_nfl_data['games'].iloc[0]
        
#         # Step 3: Get data from ESPN API
#         try:
#             # Connect to ESPN league
#             league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
            
#             # Get all players from ESPN
#             espn_player = None
            
#             # Search for the player in ESPN API
#             for team in league.teams:
#                 for roster_player in team.roster:
#                     if player_name.lower() in roster_player.name.lower():
#                         espn_player = roster_player
#                         break
#                 if espn_player:
#                     break
            
#             if not espn_player:
#                 print(f"Player {player_name} not found in ESPN API roster data")
#                 espn_fantasy_ppg = None
#             else:
#                 print(f"Found player in ESPN API: {espn_player.name} ({espn_player.position})")

#                 # Get fantasy points from ESPN
#                 espn_fantasy_ppg = espn_player.avg_points
                
                
            
#         except Exception as e:
#             print(f"Error accessing ESPN API: {e}")
#             espn_fantasy_ppg = None
        
#         # Step 4: Return comparison
#         result = {
#             "player_name": player_nfl_data['name'].iloc[0],
#             "position": player_nfl_data['position'].iloc[0],
#             "games_played": player_nfl_data['games'].iloc[0] if 'games' in player_nfl_data.columns else None,
#             "nfl_data_py_stats": {
#                 "passing_yards": player_nfl_data['passing_yards'].iloc[0] if 'passing_yards' in player_nfl_data.columns else 0,
#                 "passing_tds": player_nfl_data['passing_tds'].iloc[0] if 'passing_tds' in player_nfl_data.columns else 0,
#                 "interceptions": player_nfl_data['interceptions'].iloc[0] if 'interceptions' in player_nfl_data.columns else 0,
#                 "rushing_yards": player_nfl_data['rushing_yards'].iloc[0] if 'rushing_yards' in player_nfl_data.columns else 0,
#                 "rushing_tds": player_nfl_data['rushing_tds'].iloc[0] if 'rushing_tds' in player_nfl_data.columns else 0,
#                 "receiving_yards": player_nfl_data['receiving_yards'].iloc[0] if 'receiving_yards' in player_nfl_data.columns else 0,
#                 "receiving_tds": player_nfl_data['receiving_tds'].iloc[0] if 'receiving_tds' in player_nfl_data.columns else 0,
#                 "receptions": player_nfl_data['receptions'].iloc[0] if 'receptions' in player_nfl_data.columns else 0
#             },
#             "fantasy_points_calculations": {
#                 "our_calculation_ppg": round(nfl_fantasy_ppg, 2) if nfl_fantasy_ppg is not None else None,
#                 "nfl_data_py_ppg": round(nfl_provided_ppg, 2) if nfl_provided_ppg is not None else None,
#                 "espn_api_ppg": round(espn_fantasy_ppg, 2) if espn_fantasy_ppg is not None else None
#             }
#         }
        
#         # Print summary
#         print("\nFantasy Points Per Game Comparison:")
#         print(f"Our calculation:     {result['fantasy_points_calculations']['our_calculation_ppg']}")
#         print(f"ESPN API value:      {result['fantasy_points_calculations']['espn_api_ppg']}")
        
#         if nfl_provided_ppg is not None and espn_fantasy_ppg is not None:
#             diff = abs(nfl_fantasy_ppg - espn_fantasy_ppg)
#             diff_percent = diff / max(espn_fantasy_ppg, 0.01) * 100
#             print(f"\nDifference: {round(diff, 2)} points ({round(diff_percent, 1)}%)")
        
#         return result
        
#     except Exception as e:
#         print(f"Error comparing fantasy points: {e}")
#         import traceback
#         traceback.print_exc()
#         return None
    
# comparison = test_fantasy_points_calculation(
#     player_name="Ja'Marr Chase",  # Enter player name here
#     league_id=697625923,            # Your league ID
#     year=2024,                      
#     espn_s2="AECeHAkR7FZuTvVWSEnMRIA29wVeroZhvd7fHHy5tZvIUdIp4XIZaglA17V6g2rulDDFMCUiC%2BpeXNqWzEJTpjTJsz5Zv2DTMOIjeX0JrC6CYs8kDidYeF0HHkI78OG2O%2Bs6f%2FUPVSwXRBZEGMKPDdKl%2BE0a7na225JN4bC80tD9RXFv32kqqtEk%2Bgw1hgQ968ARiAdt69axAvjryW57rj58sYK4oMJPxjtPbh9tATi%2BSI2AmQ0dNPXZfRTA%2FFgCtgzyxWNwbKT2boYeDfFe7rm8idW47lnavsfYGwWjVpddVY6%2BcupF7uoc9AeVFQ5xNUY%3D",
#     swid="{42614A28-F6F5-4052-A14A-28F6F52052AF}"
# )