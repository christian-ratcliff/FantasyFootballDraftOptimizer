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
import shutil
import time
import random
import copy
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
START_YEAR = 2016  # First year to include in historical analysis
INCLUDE_NGS = True  # Whether to include Next Gen Stats (slower but more accurate)
DEBUG_MODE = False  # Enable for verbose logging

# Processing Options
CLUSTER_COUNT = 5  # Number of player tiers/clusters to create
DROP_BOTTOM_TIERS = 1  # Number of bottom tiers to drop
USE_FILTERED = True #Set use the filtered dataset in the training
USE_DO_NOT_DRAFT = True


# Projection Evaluation Parameters
# VALIDATION_YEARS = [2021, 2022, 2023]  # Years to use for validation
PROJECTION_YEAR = 2024  # Year to project
PERFORM_CV = True  # Whether to perform cross-validation during model training

# Caching and Reuse Options
USE_CACHED_RAW_DATA =True  # Set to True to use previously downloaded raw data
USE_CACHED_PROCESSED_DATA =True  # Set to True to use previously processed data
USE_CACHED_FEATURE_SETS = True # Set to True to use previously engineered features
CREATE_VISUALIZATIONS = False  # Whether to generate visualization plots
SKIP_MODEL_TRAINING = False  # Set to True to skip model training/evaluation


# File Paths
CONFIG_PATH = 'configs/league_settings.json'  # Path to optional config file
DATA_DIR = 'data'  # Base directory for all data
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs')  # Directory for visualizations
MODELS_DIR = os.path.join(DATA_DIR, 'models')  # Directory for saved models


# Draft Simulator Settings
RUN_DRAFT_SIMULATIONS = True  # Whether to run draft simulations
NUM_DRAFT_SIMULATIONS = 200    # Number of draft simulations to run
DRAFT_STRATEGIES_TO_TEST = ["VBD", "ESPN", "ZeroRB", "HeroRB", "TwoRB", "BestAvailable", "RL"]
USER_DRAFT_POSITION = None    # Set to a position (1-10) to simulate drafting as that position

# Season Simulator Settings
RUN_SEASON_SIMULATIONS = True  # Whether to run season simulations
NUM_SEASON_SIMULATIONS = 150   # Number of seasons to simulate per draft
RANDOMNESS_FACTOR = 0.2        # Amount of randomness in weekly scoring (0.0 = deterministic, higher = more random)
NUM_REGULAR_WEEKS = 14         # Number of regular season weeks
NUM_PLAYOFF_TEAMS = 6          # Number of playoff teams
NUM_PLAYOFF_WEEKS = 3          # Number of playoff weeks

# Reinforcement Learning Settings
TRAIN_RL_MODEL = True       # Whether to train the RL model
NUM_RL_EPISODES = 200          # Number of episodes to train for
RL_EVAL_INTERVAL = 10          # Number of episodes between evaluations
USE_EXISTING_RL_MODEL = False  # Whether to use a pre-trained RL model
RL_MODEL_PATH = 'data/models/rl_drafter_final'  # Path to pre-trained RL model

# Draft Analysis Output
SAVE_DRAFT_RESULTS = True      # Whether to save draft results
DRAFT_RESULTS_PATH = 'data/outputs/draft_results'  # Path to save draft results

#######################################################
# END OF CONFIGURATION
#######################################################

# Set up logging
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
file_handler = logging.FileHandler("fantasy_football.log")
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.WARNING)
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        file_handler,
        console_handler
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
from src.models.rl_drafter import RLDrafter
from src.models.draft_simulator import DraftSimulator, Player
from src.models.season_simulator import SeasonSimulator, SeasonEvaluator

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
                min_games = 8 
                qualified_data = pos_actual[pos_actual['games'] >= min_games]
                top_actual_players = qualified_data.nlargest(5, 'fantasy_points_per_game')['name'].tolist()
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

def run_draft_simulations(projections, league_info, roster_limits, scoring_settings, results_dir):


    
    
    logger.info("Running draft simulations...")
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract league size
    league_size = league_info.get('league_info', {}).get('team_count', 10)
    
    # Prepare baseline values for VBD
    # Define baseline indices by position (typically the last starter at each position)
    baseline_indices = {
        "QB": league_size,  # Last starting QB
        "RB": league_size * 2 + 2,  # Last starting RB including 2 FLEX
        "WR": league_size * 2 + 2,  # Last starting WR including 2 FLEX
        "TE": league_size,  # Last starting TE
        "K": league_size,  # Last starting K
        "DST": league_size  # Last starting DST
    }
    
    # Calculate baseline values
    vbd_baseline = {}
    for position, df in projections.items():
        pos = position.upper()
        if df is not None and not df.empty and "projected_points" in df.columns:
            # Sort by projected points
            sorted_df = df.sort_values("projected_points", ascending=False)
            
            # Get the baseline player
            index = baseline_indices.get(pos, 0)
            if index < len(sorted_df):
                vbd_baseline[pos] = sorted_df.iloc[index]["projected_points"]
            else:
                # If not enough players, use the last player's points
                vbd_baseline[pos] = sorted_df.iloc[-1]["projected_points"] if not sorted_df.empty else 0
    
    # Load players from projections
    all_players = DraftSimulator.load_players_from_projections(projections, vbd_baseline)
    
    # Log players loaded by position
    position_counts = {}
    for player in all_players:
        position_counts[player.position] = position_counts.get(player.position, 0) + 1
    
    logger.info(f"Loaded {len(all_players)} players for draft simulation:")
    for pos, count in position_counts.items():
        logger.info(f"  {pos}: {count} players")
    
    # Results containers
    all_teams = []
    all_draft_histories = []
    strategy_metrics = {strategy: [] for strategy in DRAFT_STRATEGIES_TO_TEST}

    # Run simulations
    for sim in range(NUM_DRAFT_SIMULATIONS):
        logger.info(f"Running draft simulation {sim+1}/{NUM_DRAFT_SIMULATIONS}")
        
        # Create a fresh deep copy of players for each simulation
        fresh_players = copy.deepcopy(all_players)
        
        # Reset draft status for all players
        for player in fresh_players:
            player.is_drafted = False
            player.drafted_round = None
            player.drafted_pick = None
            player.drafted_team = None
            
            # Add some randomness to projected points (within a small range) 
            # to create variety between simulations
            variation = random.uniform(0.95, 1.05)  # 5% variation
            player.projected_points = player.projected_points * variation
        
        # Create simulator with the fresh players
        draft_sim = DraftSimulator(
            players=fresh_players,
            league_size=league_size,
            roster_limits=roster_limits,
            num_rounds=sum(roster_limits.values()),
            scoring_settings=scoring_settings,
            user_pick=USER_DRAFT_POSITION
        )
        
        # Run draft
        teams, draft_history = draft_sim.run_draft()
        
        # Reset random seed for next simulation
        random.seed(time.time() + sim)
        
        # Store results
        all_teams.append(teams)
        all_draft_histories.append(draft_history)
        
        # Generate draft report
        report_path = os.path.join(results_dir, f"draft_simulation_{sim+1}.csv")
        draft_sim.create_draft_report(output_path=report_path)
        
        # Calculate metrics by strategy
        for team in teams:
            if team.strategy in strategy_metrics:
                strategy_metrics[team.strategy].append({
                    "total_points": team.get_total_projected_points(),
                    "starting_points": team.get_starting_lineup_points(),
                    "team_name": team.name,
                    "draft_position": team.draft_position
                })
    
    # Aggregate metrics
    summary = {}
    
    for strategy, metrics in strategy_metrics.items():
        if not metrics:
            continue
            
        # Calculate averages
        avg_total = sum(m["total_points"] for m in metrics) / len(metrics)
        avg_starters = sum(m["starting_points"] for m in metrics) / len(metrics)
        
        summary[strategy] = {
            "avg_total_points": avg_total,
            "avg_starter_points": avg_starters,
            "num_simulations": len(metrics),
            "details": metrics
        }
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame([
        {
            "Strategy": strategy,
            "Avg Total Points": data["avg_total_points"],
            "Avg Starter Points": data["avg_starter_points"],
            "Simulations": data["num_simulations"]
        }
        for strategy, data in summary.items()
    ])
    
    # Save summary
    summary_path = os.path.join(results_dir, "draft_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    
    # Plot average starter points by strategy
    ax = sns.barplot(
        x="Strategy", 
        y="Avg Starter Points", 
        data=summary_df, 
        palette="viridis"
    )
    
    # Add data labels
    for i, row in enumerate(summary_df.itertuples()):
        ax.text(
            i, 
            row._3 + 5,  # Add a small offset
            f"{row._3:.1f}",
            ha='center'
        )
    
    plt.title("Average Starter Points by Draft Strategy")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "draft_strategy_comparison.png"), dpi=300)
    plt.close()
    
    # Return summary
    return {
        "summary": summary,
        "summary_df": summary_df,
        "teams": all_teams,
        "draft_histories": all_draft_histories
    }

def run_season_simulations(draft_results, league_info, results_dir):

    logger.info("Running season simulations...")
    
    # Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)
    
    # Extract teams from draft results
    all_teams = draft_results["teams"]
    
    # Results containers
    all_season_results = []
    all_evaluations = []
    strategy_metrics = {strategy: [] for strategy in DRAFT_STRATEGIES_TO_TEST}
    
    # Configure season parameters based on league settings
    num_regular_weeks = NUM_REGULAR_WEEKS
    num_playoff_teams = NUM_PLAYOFF_TEAMS
    
    # Override with actual league settings if available
    if league_info:
        settings = league_info.get('settings', {})
        if isinstance(settings, dict):
            # Regular season weeks
            if 'scheduleSettings' in settings and 'matchupPeriodCount' in settings['scheduleSettings']:
                num_regular_weeks = settings['scheduleSettings']['matchupPeriodCount']
            
            # Playoff teams
            if 'scheduleSettings' in settings and 'playoffTeamCount' in settings['scheduleSettings']:
                num_playoff_teams = settings['scheduleSettings']['playoffTeamCount']
    
    logger.info(f"Season settings: {num_regular_weeks} regular weeks, {num_playoff_teams} playoff teams")
    
    # Run simulations for each draft result
    for draft_idx, teams in enumerate(all_teams):
        logger.info(f"Processing draft result {draft_idx+1}/{len(all_teams)}")
        
        # Run multiple seasons for this draft
        draft_seasons = []
        draft_evaluations = []
        
        for season_idx in range(NUM_SEASON_SIMULATIONS):
            logger.info(f"  Season simulation {season_idx+1}/{NUM_SEASON_SIMULATIONS}")
            
            # Create simulator
            season_sim = SeasonSimulator(
                teams=teams,
                num_regular_weeks=num_regular_weeks,
                num_playoff_teams=num_playoff_teams,
                num_playoff_weeks=NUM_PLAYOFF_WEEKS,
                randomness=RANDOMNESS_FACTOR
            )
            
            # Run simulation
            season_results = season_sim.simulate_season()
            
            # Evaluate results
            evaluation = SeasonEvaluator(teams, season_results)
            
            # Store results
            draft_seasons.append(season_results)
            draft_evaluations.append(evaluation)
            
            # Update metrics by strategy
            for strategy, metrics in evaluation.metrics.items():
                if strategy in strategy_metrics:
                    strategy_metrics[strategy].append({
                        "avg_rank": metrics["avg_rank"],
                        "avg_wins": metrics["avg_wins"],
                        "avg_points_for": metrics["avg_points_for"],
                        "championship_rate": metrics["championship_rate"],
                        "playoff_rate": metrics["playoff_rate"],
                        "draft_idx": draft_idx,
                        "season_idx": season_idx
                    })
        
        # Add to overall results
        all_season_results.append(draft_seasons)
        all_evaluations.append(draft_evaluations)
    
    # Aggregate metrics
    summary = {}
    
    for strategy, metrics in strategy_metrics.items():
        if not metrics:
            continue
            
        # Calculate averages
        avg_rank = sum(m["avg_rank"] for m in metrics) / len(metrics)
        avg_wins = sum(m["avg_wins"] for m in metrics) / len(metrics)
        avg_points = sum(m["avg_points_for"] for m in metrics) / len(metrics)
        championship_rate = sum(m["championship_rate"] for m in metrics) / len(metrics)
        playoff_rate = sum(m["playoff_rate"] for m in metrics) / len(metrics)
        
        summary[strategy] = {
            "avg_rank": avg_rank,
            "avg_wins": avg_wins,
            "avg_points": avg_points,
            "championship_rate": championship_rate * 100,  # Convert to percentage
            "playoff_rate": playoff_rate * 100,  # Convert to percentage
            "num_simulations": len(metrics),
            "details": metrics
        }
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame([
        {
            "Strategy": strategy,
            "Avg Rank": data["avg_rank"],
            "Avg Wins": data["avg_wins"],
            "Avg Points": data["avg_points"],
            "Championship %": data["championship_rate"],
            "Playoff %": data["playoff_rate"],
            "Simulations": data["num_simulations"]
        }
        for strategy, data in summary.items()
    ])
    
    # Save summary
    summary_path = os.path.join(results_dir, "season_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # Create visualizations
    # 1. Championship rate by strategy
    plt.figure(figsize=(12, 8))
    
    # Plot championship rate
    ax = sns.barplot(
        x="Strategy", 
        y="Championship %", 
        data=summary_df, 
        palette="viridis"
    )
    
    # Add data labels
    for i, row in enumerate(summary_df.itertuples()):
        ax.text(
            i, 
            row._5 + 1,  # Add a small offset
            f"{row._5:.1f}%",
            ha='center'
        )
    
    plt.title("Championship Rate by Draft Strategy")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "championship_rate_by_strategy.png"), dpi=300)
    plt.close()
    
    # 2. Average rank by strategy
    plt.figure(figsize=(12, 8))
    
    # Plot average rank (invert y-axis so lower is better visually)
    ax = sns.barplot(
        x="Strategy", 
        y="Avg Rank", 
        data=summary_df, 
        palette="viridis"
    )
    
    # Add data labels
    for i, row in enumerate(summary_df.itertuples()):
        ax.text(
            i, 
            row._3 - 0.2,  # Add a small offset
            f"{row._3:.1f}",
            ha='center'
        )
    
    # Invert y-axis so lower ranks are at the top
    plt.gca().invert_yaxis()
    
    plt.title("Average Finish Position by Draft Strategy")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "avg_rank_by_strategy.png"), dpi=300)
    plt.close()
    
    # Return summary
    return {
        "summary": summary,
        "summary_df": summary_df,
        "season_results": all_season_results,
        "evaluations": all_evaluations
    }

def train_rl_model(projections, league_info, roster_limits, scoring_settings, models_dir):
    """
    Train an RL model for draft optimization
    
    Parameters:
    -----------
    projections : dict
        Dictionary containing player projections by position
    league_info : dict
        Dictionary containing league information
    roster_limits : dict
        Dictionary containing roster position limits
    scoring_settings : dict
        Dictionary containing scoring settings
    models_dir : str
        Directory to save trained models
        
    Returns:
    --------
    dict
        Training results
    """
    logger.info("Training RL draft model...")
    
    # Ensure models directory exists
    os.makedirs(models_dir, exist_ok=True)
    
    # Create specific directory for RL models
    rl_models_dir = os.path.join(models_dir, "rl_models")
    os.makedirs(rl_models_dir, exist_ok=True)
    
    # Extract league size
    league_size = league_info.get('league_info', {}).get('team_count', 10)
    
    # Prepare baseline values for VBD
    # Define baseline indices by position (typically the last starter at each position)
    baseline_indices = {
        "QB": league_size,  # Last starting QB
        "RB": league_size * 2 + 2,  # Last starting RB including 2 FLEX
        "WR": league_size * 2 + 2,  # Last starting WR including 2 FLEX
        "TE": league_size,  # Last starting TE
        "K": league_size,  # Last starting K
        "DST": league_size  # Last starting DST
    }
    
    # Calculate baseline values
    vbd_baseline = {}
    for position, df in projections.items():
        pos = position.upper()
        if df is not None and not df.empty and "projected_points" in df.columns:
            # Sort by projected points
            sorted_df = df.sort_values("projected_points", ascending=False)
            
            # Get the baseline player
            index = baseline_indices.get(pos, 0)
            if index < len(sorted_df):
                vbd_baseline[pos] = sorted_df.iloc[index]["projected_points"]
            else:
                # If not enough players, use the last player's points
                vbd_baseline[pos] = sorted_df.iloc[-1]["projected_points"] if not sorted_df.empty else 0
    
    # Load players from projections
    all_players = DraftSimulator.load_players_from_projections(projections, vbd_baseline)
    
    # Log players loaded by position
    position_counts = {}
    for player in all_players:
        position_counts[player.position] = position_counts.get(player.position, 0) + 1
    
    logger.info(f"Loaded {len(all_players)} players for RL training:")
    for pos, count in position_counts.items():
        logger.info(f"  {pos}: {count} players")
    
    # Configure season parameters based on league settings
    num_regular_weeks = NUM_REGULAR_WEEKS
    num_playoff_teams = NUM_PLAYOFF_TEAMS
    
    # Override with actual league settings if available
    if league_info:
        settings = league_info.get('settings', {})
        if isinstance(settings, dict):
            # Regular season weeks
            if 'scheduleSettings' in settings and 'matchupPeriodCount' in settings['scheduleSettings']:
                num_regular_weeks = settings['scheduleSettings']['matchupPeriodCount']
            
            # Playoff teams
            if 'scheduleSettings' in settings and 'playoffTeamCount' in settings['scheduleSettings']:
                num_playoff_teams = settings['scheduleSettings']['playoffTeamCount']
    
    logger.info(f"Season settings: {num_regular_weeks} regular weeks, {num_playoff_teams} playoff teams")
    
    # Create simulators
    draft_sim = DraftSimulator(
        players=all_players.copy(),
        league_size=league_size,
        roster_limits=roster_limits,
        num_rounds=sum(roster_limits.values()),
        scoring_settings=scoring_settings
    )
    
    season_sim = SeasonSimulator(
        teams=draft_sim.teams,
        num_regular_weeks=num_regular_weeks,
        num_playoff_teams=num_playoff_teams,
        num_playoff_weeks=NUM_PLAYOFF_WEEKS,
        randomness=RANDOMNESS_FACTOR
    )
    
    # Initialize or load RL model
    if USE_EXISTING_RL_MODEL and os.path.exists(RL_MODEL_PATH):
        logger.info(f"Loading existing RL model from {RL_MODEL_PATH}")
        rl_model = RLDrafter.load_model(RL_MODEL_PATH)
    else:
        logger.info("Initializing new RL model")
        
        # Create a sample state to determine the input dimension
        # Find an RL team to use for the sample state
        rl_team = next((team for team in draft_sim.teams if team.strategy == "RL"), None)
        if not rl_team:
            logger.warning("No RL team found in the draft simulator")
            rl_team = draft_sim.teams[0]  # Use first team as fallback
        
        # Create a sample state
        from src.models.rl_drafter import DraftState
        sample_state = DraftState(
            team=rl_team,
            available_players=all_players,  # Use all players for sample state
            round_num=1,
            overall_pick=1,
            league_size=league_size,
            roster_limits=roster_limits,
            max_rounds=draft_sim.num_rounds
        )
        
        # Get a sample feature vector to determine input dimension
        sample_feature = sample_state.to_feature_vector(0)
        input_dim = len(sample_feature)
        
        logger.info(f"Creating RL model with input dimension {input_dim}")
        rl_model = RLDrafter(input_dim=input_dim)
    
    # Train model with more frequent saving
    training_results = rl_model.train(
        draft_simulator=draft_sim,
        season_simulator=season_sim,
        num_episodes=NUM_RL_EPISODES,
        eval_interval=RL_EVAL_INTERVAL,
        save_interval=1,  # Save after every episode
        save_path=rl_models_dir
    )
    
    # Create visualizations
    # 1. Rewards over time
    plt.figure(figsize=(12, 8))
    
    rewards = training_results["rewards_history"]
    episodes = range(1, len(rewards) + 1)
    
    plt.plot(episodes, rewards, 'b-', alpha=0.7)
    plt.title("RL Training Rewards Over Time")
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Add trend line
    if len(rewards) > 1:
        z = np.polyfit(episodes, rewards, 1)
        p = np.poly1d(z)
        plt.plot(episodes, p(episodes), "r--", alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(models_dir, "rl_reward_history.png"), dpi=300)
    plt.close()
    
    # 2. Win rates by strategy
    if training_results["win_rates"]:
        plt.figure(figsize=(12, 8))
        
        # Extract win rates for each strategy
        strategies = list(training_results["win_rates"][0].keys())
        win_rates = {strategy: [] for strategy in strategies}
        
        for wr in training_results["win_rates"]:
            for strategy, rate in wr.items():
                win_rates[strategy].append(rate)
        
        # Plot win rates for each strategy
        for strategy, rates in win_rates.items():
            evals = range(RL_EVAL_INTERVAL, len(rates) * RL_EVAL_INTERVAL + 1, RL_EVAL_INTERVAL)
            plt.plot(evals, rates, label=strategy)
        
        plt.title("Win Rates by Strategy During Training")
        plt.xlabel("Episode")
        plt.ylabel("Win Rate")
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(models_dir, "rl_win_rates.png"), dpi=300)
        plt.close()
    
    # Also save final model to the main model directory for easy access
    final_model_path = os.path.join(models_dir, "rl_drafter_final")
    if os.path.exists(os.path.join(rl_models_dir, "rl_drafter_final.keras")):
        shutil.copy(os.path.join(rl_models_dir, "rl_drafter_final.keras"), final_model_path + ".keras")
    elif os.path.exists(os.path.join(rl_models_dir, "rl_drafter_final.pkl")):
        shutil.copy(os.path.join(rl_models_dir, "rl_drafter_final.pkl"), final_model_path + ".pkl")
    
    # Return training results
    return training_results
    return training_results

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
    
    # Simulation options from config
    run_draft_sims = config.get('run_draft_simulations', RUN_DRAFT_SIMULATIONS)
    run_season_sims = config.get('run_season_simulations', RUN_SEASON_SIMULATIONS)
    train_rl = config.get('train_rl_model', TRAIN_RL_MODEL)
    
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
    
    # Extract roster limits and scoring settings
    roster_limits, scoring_settings = extract_league_settings(league_data)
    
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
    projections = {}
    if not skip_model:
        logger.info("Step 6: Training and evaluating projection models...")
        
        # Initialize the projection model
        projection_model = PlayerProjectionModel(feature_sets, output_dir=MODELS_DIR, use_filtered=USE_FILTERED)

        
        # Train models for all positions with proper validation
        all_metrics = projection_model.train_all_positions(
            model_type='random_forest',
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
        projections = projection_model.generate_full_projections(projection_filters, use_do_not_draft=USE_DO_NOT_DRAFT)
        
    else:
        logger.info("Skipping model training and evaluation as specified in configuration")
        
        # Try to load pre-trained projections
        for position in ['qb', 'rb', 'wr', 'te']:
            model_path = os.path.join(MODELS_DIR, f"{position}_model.joblib")
            if os.path.exists(model_path):
                try:
                    logger.info(f"Loading pre-trained {position} model from {model_path}")
                    model = PlayerProjectionModel.load_model(model_path, feature_sets)
                    
                    # Generate projections
                    projection_key = f"{position}_projection"
                    if projection_key in feature_sets and not feature_sets[projection_key].empty:
                        projection_data = model.project_players(position, feature_sets[projection_key])
                        projections[position] = projection_data
                except Exception as e:
                    logger.error(f"Error loading pre-trained model: {e}")
    
    # Step 7: Run draft simulations if requested
    draft_results = None
    if run_draft_sims and projections:
        logger.info("Step 7: Running draft simulations...")
        
        # Create output directory
        draft_results_dir = os.path.join(OUTPUT_DIR, 'draft_simulations')
        os.makedirs(draft_results_dir, exist_ok=True)
        
        # Run draft simulations
        draft_results = run_draft_simulations(
            projections=projections,
            league_info=league_data,
            roster_limits=roster_limits,
            scoring_settings=scoring_settings,
            results_dir=draft_results_dir
        )
        
        logger.info("Draft simulations complete")
    elif run_draft_sims:
        logger.warning("Cannot run draft simulations without projections")
    
    # Step 8: Run season simulations if requested
    season_results = None
    if run_season_sims and draft_results:
        logger.info("Step 8: Running season simulations...")
        
        # Create output directory
        season_results_dir = os.path.join(OUTPUT_DIR, 'season_simulations')
        os.makedirs(season_results_dir, exist_ok=True)
        
        # Run season simulations
        season_results = run_season_simulations(
            draft_results=draft_results,
            league_info=league_data,
            results_dir=season_results_dir
        )
        
        logger.info("Season simulations complete")
    elif run_season_sims:
        logger.warning("Cannot run season simulations without draft results")
    
    # Step 9: Train RL model if requested
    rl_results = None
    if train_rl and projections:
        logger.info("Step 9: Training reinforcement learning model...")
        
        # Train RL model
        rl_results = train_rl_model(
            projections=projections,
            league_info=league_data,
            roster_limits=roster_limits,
            scoring_settings=scoring_settings,
            models_dir=MODELS_DIR
        )
        
        logger.info("RL model training complete")
    elif train_rl:
        logger.warning("Cannot train RL model without projections")
    
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
    
    # Display draft simulation results if available
    if draft_results and 'summary_df' in draft_results:
        print("\nDraft Strategy Comparison:")
        print(draft_results['summary_df'].to_string(index=False))
    
    # Display season simulation results if available
    if season_results and 'summary_df' in season_results:
        print("\nSeason Simulation Results:")
        print(season_results['summary_df'].to_string(index=False))
    
    # Display RL training results if available
    if rl_results:
        print("\nRL Model Training Results:")
        print(f"  Episodes trained: {NUM_RL_EPISODES}")
        print(f"  Final epsilon: {rl_results['final_epsilon']:.4f}")
        print(f"  Best reward: {rl_results['best_reward']:.2f}")
    
    print("\nAnalysis completed successfully!")
    print(f"Output data and visualizations saved to the '{DATA_DIR}' directory")
    
    return {
        "league_data": league_data,
        "nfl_data": nfl_data if not use_cached_raw else None,
        "processed_data": processed_data,
        "feature_sets": feature_sets,
        "dropped_tiers": dropped_tiers,
        "projections": projections,
        "draft_results": draft_results,
        "season_results": season_results,
        "rl_results": rl_results
    }

def extract_league_settings(league_data):
    """
    Extract roster limits and scoring settings from league data
    
    Parameters:
    -----------
    league_data : Dict[str, Any]
        League information from ESPN API
        
    Returns:
    --------
    Tuple[Dict[str, int], Dict[str, Any]]
        Tuple of (roster_limits, scoring_settings)
    """
    # Default roster limits
    roster_limits = {
    "QB": 3,
    "RB": 10,
    "WR": 10,
    "TE": 4}
    
    # Default scoring settings
    scoring_settings = {
        "PA0": 5.0,
        "PA1": 4.0,
        "PA28": -1.0,
        "2PRET": 2.0,
        "YA449": -3.0,
        "RTD": 6.0,
        "1PSF": 1.0,
        "FUML": -2.0,
        "YA499": -5.0,
        "INT": 2.0,
        "YA549": -6.0,
        "YA550": -7.0,
        "BLKKRTD": 6.0,
        "SF": 2.0,
        "PA7": 3.0,
        "FR": 2.0,
        "BLKK": 2.0,
        "2PC": 2.0,
        "RETD": 6.0,
        "INTT": -2.0,
        "PTD": 4.0,
        "PRTD": 6.0,
        "RY": 0.1,
        "REC": 0.5,
        "INTTD": 6.0,
        "FTD": 6.0,
        "KRTD": 6.0,
        "FG0": 3.0,
        "2PR": 2.0,
        "FRTD": 6.0,
        "FG40": 4.0,
        "PA14": 1.0,
        "YA100": 5.0,
        "SK": 1.0,
        "PA46": -5.0,
        "FGM": -1.0,
        "PY": 0.04,
        "PA35": -3.0,
        "YA399": -1.0,
        "PAT": 1.0,
        "REY": 0.1,
        "YA199": 3.0,
        "2PRE": 2.0,
        "YA299": 2.0,
        "FG60": 6.0,
        "FG50": 5.0
    }
    
    # Get settings from league data if available
    if league_data and 'settings' in league_data:
        settings = league_data['settings']
        
        # Extract roster limits
        if hasattr(settings, 'position_slot_counts') and isinstance(settings.position_slot_counts, dict):
            for pos, count in settings.position_slot_counts.items():
                if pos in roster_limits:
                    roster_limits[pos] = count
        
        # Extract scoring format
        if hasattr(settings, 'scoring_format') and isinstance(settings.scoring_format, list):
            for item in settings.scoring_format:
                # Convert ESPN scoring to our format
                if item['abbr'] == 'PY':
                    scoring_settings['passing_yards'] = item['points']
                elif item['abbr'] == 'PTD':
                    scoring_settings['passing_td'] = item['points']
                elif item['abbr'] == 'INT':
                    scoring_settings['interception'] = item['points']
                elif item['abbr'] == 'RY':
                    scoring_settings['rushing_yards'] = item['points']
                elif item['abbr'] == 'RTD':
                    scoring_settings['rushing_td'] = item['points']
                elif item['abbr'] == 'REY':
                    scoring_settings['receiving_yards'] = item['points']
                elif item['abbr'] == 'RETD':
                    scoring_settings['receiving_td'] = item['points']
                elif item['abbr'] == 'REC':
                    scoring_settings['reception'] = item['points']
                elif item['abbr'] == 'FUML':
                    scoring_settings['fumble_lost'] = item['points']
    
    # Add roster settings for starter lineup slots
    scoring_settings['starter_limits'] = {
        "QB": 1,
        "RB": 2,
        "WR": 3,
        "TE": 1,
        "FLEX": 2,
        "K": 0,
        "DST": 0,
        "OP": 1
    }
    
    # Get starter slots from league data if available
    if league_data and 'settings' in league_data:
        settings = league_data['settings']
        # Extract lineup settings if available (implementation depends on ESPN API structure)
    
    return roster_limits, scoring_settings


if __name__ == "__main__":
    main()
    
