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
7. Trains PPO reinforcement learning model for optimal draft strategy
"""

import os
import pickle
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
USE_FILTERED = False  # Set use the filtered dataset in the training
USE_DO_NOT_DRAFT = True

# Projection Evaluation Parameters
PROJECTION_YEAR = 2024  # Year to project
PERFORM_CV = True  # Whether to perform cross-validation during model training

# Caching and Reuse Options
USE_CACHED_RAW_DATA = True  # Set to True to use previously downloaded raw data
USE_CACHED_PROCESSED_DATA = True  # Set to True to use previously processed data
USE_CACHED_FEATURE_SETS = True  # Set to True to use previously engineered features
CREATE_VISUALIZATIONS = False  # Whether to generate visualization plots
SKIP_MODEL_TRAINING = False  # Set to True to skip model training/evaluation

# File Paths
CONFIG_PATH = 'configs/league_settings.json'  # Path to optional config file
DATA_DIR = 'data'  # Base directory for all data
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs')  # Directory for visualizations
MODELS_DIR = os.path.join(DATA_DIR, 'models')  # Directory for saved models

# Draft Simulator Settings
RUN_DRAFT_SIMULATIONS = False  # Whether to run draft simulations
NUM_DRAFT_SIMULATIONS = 200  # Number of draft simulations to run
DRAFT_STRATEGIES_TO_TEST = ["VBD", "ESPN", "ZeroRB", "HeroRB", "TwoRB", "BestAvailable", "PPO"]
USER_DRAFT_POSITION = None  # Set to a position (1-10) to simulate drafting as that position

# Season Simulator Settings
RUN_SEASON_SIMULATIONS = False  # Whether to run season simulations
NUM_SEASON_SIMULATIONS = 150  # Number of seasons to simulate per draft
RANDOMNESS_FACTOR = 0.2  # Amount of randomness in weekly scoring (0.0 = deterministic, higher = more random)
NUM_REGULAR_WEEKS = 14  # Number of regular season weeks
NUM_PLAYOFF_TEAMS = 6  # Number of playoff teams
NUM_PLAYOFF_WEEKS = 3  # Number of playoff weeks

# PPO Reinforcement Learning Settings
TRAIN_PPO_MODEL = True  # Whether to train the PPO model
NUM_PPO_EPISODES = 2000  # Number of episodes to train PPO for
PPO_EVAL_INTERVAL = 10  # Number of episodes between PPO evaluations
PPO_SAVE_INTERVAL = 50  # Number of episodes between PPO model saves
USE_EXISTING_PPO_MODEL = False  # Whether to use a pre-trained PPO model
PPO_MODEL_PATH = 'data/models/ppo_models/ppo_drafter_final'  # Path to pre-trained PPO model
PPO_LEARNING_RATE = 0.0003  # Learning rate for PPO
PPO_GAMMA = 0.96  # Discount factor for PPO
PPO_BATCH_SIZE = 32  # Batch size for PPO training
PPO_POLICY_CLIP = 0.1  # Policy clipping parameter for PPO
PPO_ENTROPY_COEF = 0.02  # Entropy coefficient for PPO
USE_TOP_N_FEATURES = 7
CURRICULUM_ENABLED = True
CURRICULUM_PARAMS = {
    'phase_durations': {
        1: 200,  # Episodes in phase 1
        2: 500,  # Episodes in phase 2 
        3: 700,  # Episodes in phase 3
        4: float('inf')  # Phase 4 continues until end
    },
    'phase_thresholds': {
        1: 0.7,  # 70% valid rosters  
        2: 100.0,  # Minimum total projected points
        3: 120.0,  # Minimum starter points
    },
    'reward_mix_weights': {  # How much to include previous phase rewards
        1: [1.0, 0.0, 0.0, 0.0],
        2: [0.3, 0.7, 0.0, 0.0],
        3: [0.1, 0.3, 0.6, 0.0],
        4: [0.05, 0.15, 0.3, 0.5]
    },
    'max_stuck_episodes': 100,  # Maximum episodes to be stuck before forcing advancement
    'phase_stability_window': 20  # Episodes to consider for stability
}

# Opponent Modeling Settings
OPPONENT_MODELING_ENABLED = True  # Whether to use opponent modeling
POSITION_SCARCITY_WEIGHT = 1.2    # Weight for position scarcity (higher = more importance)
RUN_DETECTION_THRESHOLD = 0.25    # Threshold to detect a position run (percentage of recent picks)
VALUE_CLIFF_THRESHOLD = 0.15      # Threshold to detect a value cliff (percentage drop)
ADAPTIVE_POSITION_WEIGHTS = True  # Use adaptive position weighting
PICK_PREDICTION_DEPTH = 8         # Number of future picks to predict

# Hierarchical PPO settings
USE_HIERARCHICAL_PPO = True  # Whether to use hierarchical PPO
HIERARCHICAL_PPO_MODEL_PATH = 'data/models/ppo_models/hierarchical_ppo_final'
META_POLICY_LR = 0.0003  # Learning rate for meta policy
SUB_POLICY_LR = 0.0003  # Learning rate for sub policies

# Population-based Training Settings
USE_POPULATION_TRAINING = True  # Whether to use population-based training
POPULATION_SIZE = 5              # Number of agents in the population
HIERARCHICAL_RATIO = 0.4         # Ratio of hierarchical PPO agents (0.0 to 1.0)
EVOLUTION_INTERVAL = 50          # Number of episodes between evolution events
TOURNAMENT_SIZE = 3              # Number of agents in tournament selection
MUTATION_RATE = 0.1              # Probability of parameter mutation 
MUTATION_STRENGTH = 0.2          # Magnitude of mutations
ELITISM_COUNT = 1                # Number of top agents preserved unchanged
ENABLE_CROSSOVER = True          # Whether to enable crossover between agents


# Draft Analysis Output
SAVE_DRAFT_RESULTS = True  # Whether to save draft results
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
from src.analysis.analyzer import FantasyFootballAnalyzer
from src.models.projections import PlayerProjectionModel
from src.models.ppo_drafter import PPODrafter, DraftState
from src.models.draft_simulator import DraftSimulator, Player
from src.models.season_simulator import SeasonSimulator, SeasonEvaluator
from src.models.projections import ProjectionModelLoader
from src.models.lineup_evaluator import LineupEvaluator
from src.models.hierarchical_ppo_drafter import HierarchicalPPODrafter
from src.models.ppo_population import PPOPopulation


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
    
    # Create PPO models directory
    ppo_models_dir = os.path.join(MODELS_DIR, 'ppo_models')
    os.makedirs(ppo_models_dir, exist_ok=True)
    
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
        Dictionary with additional combined data
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


def run_season_simulations(draft_results, league_info, results_dir):
    """
    Simulate seasons based on draft results
    
    Parameters:
    -----------
    draft_results : dict
        Results from draft simulations
    league_info : dict
        Dictionary containing league information
    results_dir : str
        Directory to save results
        
    Returns:
    --------
    dict
        Dictionary of season simulation results
    """
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
    num_regular_weeks = league_info.get('schedule_settings', {}).get('regular_season_weeks', 14)
    num_playoff_teams = league_info.get('league_info', {}).get('playoff_teams', 6)
    num_playoff_weeks = league_info.get('schedule_settings', {}).get('playoff_matchup_period_length', 3)
    nfl_games_per_player = league_info.get('league_info', {}).get('nfl_games_per_player', 17)
    
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


# def train_ppo_model(projections, league_info, roster_limits, scoring_settings, models_dir, projection_models=None):
#     """
#     Train a PPO model for draft optimization with opponent modeling
    
#     Parameters:
#     -----------
#     projections : dict
#         Dictionary of player projections by position
#     league_info : dict
#         Dictionary containing league information
#     roster_limits : dict
#         Dictionary of roster position limits
#     scoring_settings : dict
#         Dictionary of scoring settings
#     models_dir : str
#         Directory to save models
#     projection_models : dict, optional
#         Dictionary of pre-loaded projection models by position
        
#     Returns:
#     --------
#     tuple
#         (PPO model, training results)
#     """
#     logger.info("Training PPO model for draft optimization...")
    
#     # Make sure we have projection models
#     if projection_models is None:
#         logger.info("Loading projection models...")
#         projection_loader = ProjectionModelLoader(models_dir)
#         projection_models = projection_loader.models
        
#         # Check if we loaded any models
#         if not projection_models:
#             logger.warning("No projection models found! Using default features.")
#         else:
#             logger.info(f"Loaded {len(projection_models)} projection models: {list(projection_models.keys())}")
    
#     # Create directory for PPO models
#     ppo_models_dir = os.path.join(models_dir, 'ppo_models')
#     os.makedirs(ppo_models_dir, exist_ok=True)
    
#     # Check if pre-trained model should be used
#     if USE_EXISTING_PPO_MODEL and os.path.exists(PPO_MODEL_PATH + "_actor.weights.h5"):
#         logger.info(f"Loading existing PPO model from {PPO_MODEL_PATH}")
#         try:
#             ppo_model = PPODrafter.load_model(PPO_MODEL_PATH)
#             # Return early with the loaded model
#             return ppo_model, {"loaded_from": PPO_MODEL_PATH}
#         except Exception as e:
#             logger.error(f"Error loading PPO model: {e}")
#             logger.info("Will train a new model instead")
    
#     # Extract league size
#     league_size = league_info.get('league_info', {}).get('team_count', 10)
    
#     # Prepare baseline values for VBD
#     baseline_indices = {
#         "QB": league_size,
#         "RB": league_size * 2 + 2,
#         "WR": league_size * 2 + 2,
#         "TE": league_size,
#         "K": league_size,
#         "DST": league_size
#     }
    
#     # Calculate baseline values
#     vbd_baseline = {}
#     for position, df in projections.items():
#         pos = position.upper()
#         if df is not None and not df.empty and "projected_points" in df.columns:
#             sorted_df = df.sort_values("projected_points", ascending=False)
#             index = baseline_indices.get(pos, 0)
#             if index < len(sorted_df):
#                 vbd_baseline[pos] = sorted_df.iloc[index]["projected_points"]
#             else:
#                 vbd_baseline[pos] = sorted_df.iloc[-1]["projected_points"] if not sorted_df.empty else 0
    
#     # Load players from projections
#     all_players = DraftSimulator.load_players_from_projections(projections, vbd_baseline)
    
#     # Create draft simulator with projection models
#     draft_simulator = DraftSimulator(
#         players=all_players,
#         league_size=league_size,
#         roster_limits=roster_limits,
#         num_rounds=18,  # Standard number of rounds
#         scoring_settings=scoring_settings,
#         user_pick=None,
#         projection_models=projection_models  # Pass projection models here
#     )
    
#     # Find or create the RL team
#     ppo_draft_position = random.randint(1, league_size)
#     rl_team = next((team for team in draft_simulator.teams 
#                  if team.draft_position == ppo_draft_position), None)
#     if not rl_team:
#         # If no RL team found, set the first team to use RL
#         rl_team = random.choice(draft_simulator.teams)
#         rl_team.strategy = "PPO"
    
#     # Create a sample state to determine dimensions
#     available_players = [p for p in draft_simulator.players if not p.is_drafted]
    
#     # Get starter limits if available in scoring settings
#     starter_limits = {}
#     if scoring_settings and 'starter_limits' in scoring_settings:
#         starter_limits = scoring_settings['starter_limits']
    
#     # Create a sample state with opponent modeling if enabled
#     if OPPONENT_MODELING_ENABLED:
#         sample_state = DraftState(
#             team=rl_team,
#             available_players=available_players,
#             round_num=1,
#             overall_pick=1,
#             league_size=draft_simulator.league_size,
#             roster_limits=draft_simulator.roster_limits,
#             max_rounds=draft_simulator.num_rounds,
#             all_teams=draft_simulator.teams,
#             starter_limits=starter_limits
#         )
#     else:
#         sample_state = DraftState(
#             team=rl_team,
#             available_players=available_players,
#             round_num=1,
#             overall_pick=1,
#             league_size=draft_simulator.league_size,
#             roster_limits=draft_simulator.roster_limits,
#             max_rounds=draft_simulator.num_rounds
#         )
    
#     # Get dimensions from sample state
#     state_dim = len(sample_state.to_feature_vector())
#     action_feature_dim = len(sample_state.get_action_features(0))
#     action_dim = 256  # Maximum number of players to consider at once
    
#     logger.info(f"State dimension: {state_dim}, Action feature dimension: {action_feature_dim}")
    
#     # Create PPO model with opponent modeling
#     ppo_model = PPODrafter(
#         state_dim=state_dim,
#         action_feature_dim=action_feature_dim,
#         action_dim=action_dim,
#         lr_actor=PPO_LEARNING_RATE,
#         lr_critic=PPO_LEARNING_RATE,
#         gamma=PPO_GAMMA,
#         batch_size=PPO_BATCH_SIZE,
#         policy_clip=PPO_POLICY_CLIP,
#         entropy_coef=PPO_ENTROPY_COEF,
#         use_top_n_features=USE_TOP_N_FEATURES,
#         curriculum_enabled=CURRICULUM_ENABLED,
#         opponent_modeling_enabled=OPPONENT_MODELING_ENABLED
#     )
    
#     # Set opponent modeling parameters if enabled
#     if OPPONENT_MODELING_ENABLED:
#         # Initialize position weights
#         ppo_model.position_priority_weights = {
#             "QB": 1.0,
#             "RB": 1.5,  # Start with slightly higher weight for RB
#             "WR": 1.2,  # Start with slightly higher weight for WR
#             "TE": 0.9,
#             "K": 0.5,
#             "DST": 0.5
#         }
        
#         # Set additional parameters
#         ppo_model.position_scarcity_weight = POSITION_SCARCITY_WEIGHT
#         ppo_model.run_detection_threshold = RUN_DETECTION_THRESHOLD
#         ppo_model.value_cliff_threshold = VALUE_CLIFF_THRESHOLD
#         ppo_model.pick_prediction_depth = PICK_PREDICTION_DEPTH
        
#         logger.info(f"Opponent modeling enabled with parameters:")
#         logger.info(f"  Position scarcity weight: {POSITION_SCARCITY_WEIGHT}")
#         logger.info(f"  Run detection threshold: {RUN_DETECTION_THRESHOLD}")
#         logger.info(f"  Value cliff threshold: {VALUE_CLIFF_THRESHOLD}")
    
#     # Set curriculum parameters if enabled
#     if CURRICULUM_ENABLED and CURRICULUM_PARAMS:
#         for param_name, param_value in CURRICULUM_PARAMS.items():
#             if hasattr(ppo_model, param_name):
#                 setattr(ppo_model, param_name, param_value)
#             elif isinstance(param_value, dict) and hasattr(ppo_model, param_name):
#                 # For dictionary parameters like phase_durations
#                 getattr(ppo_model, param_name).update(param_value)
    
#     # Create season simulator for training
#     season_simulator = SeasonSimulator(
#         teams=draft_simulator.teams,
#         num_regular_weeks=NUM_REGULAR_WEEKS,
#         num_playoff_teams=NUM_PLAYOFF_TEAMS,
#         num_playoff_weeks=NUM_PLAYOFF_WEEKS,
#         randomness=RANDOMNESS_FACTOR
#     )
    
#     # Train the model
#     logger.info(f"Training PPO model for {NUM_PPO_EPISODES} episodes")
#     start_time = time.time()
    
#     training_results = ppo_model.train(
#         draft_simulator=draft_simulator,
#         season_simulator=season_simulator,
#         num_episodes=NUM_PPO_EPISODES,
#         eval_interval=PPO_EVAL_INTERVAL,
#         save_interval=PPO_SAVE_INTERVAL,
#         save_path=ppo_models_dir
#     )
    
#     training_time = time.time() - start_time
#     logger.info(f"PPO training completed in {training_time:.2f} seconds")
    
#     # Save final model to the main path
#     ppo_model.save_model(PPO_MODEL_PATH)
    
#     # Generate learning verification visualizations
#     try:
#         # Generate standard learning visualizations
#         if hasattr(ppo_model, 'generate_learning_visualizations'):
#             ppo_model.generate_learning_visualizations(output_dir=ppo_models_dir)
#         else:
#             from src.analysis.visualizer import generate_learning_visualizations
#             generate_learning_visualizations(ppo_model, output_dir=ppo_models_dir)
        
#         logger.info("Generated learning verification visualizations")
        
#         # Generate opponent modeling visualizations if enabled
#         if OPPONENT_MODELING_ENABLED and hasattr(ppo_model, 'generate_opponent_modeling_visualizations'):
#             ppo_model.generate_opponent_modeling_visualizations(output_dir=ppo_models_dir)
#             logger.info("Generated opponent modeling visualizations")
#     except Exception as e:
#         logger.error(f"Error generating visualizations: {e}")
    
#     # Create simplified training progress plot if not already created
#     plt.figure(figsize=(12, 6))
#     plt.plot(training_results['rewards_history'])
#     plt.title('PPO Training Progress')
#     plt.xlabel('Episode')
#     plt.ylabel('Reward')
#     plt.grid(True)
#     plt.savefig(os.path.join(ppo_models_dir, 'training_progress.png'))
    
#     # Plot win rates if available
#     if training_results['win_rates']:
#         plt.figure(figsize=(12, 6))
        
#         # Extract win rates for each strategy
#         strategies = list(training_results['win_rates'][0].keys())
#         for strategy in strategies:
#             win_rates = [wr.get(strategy, 0) for wr in training_results['win_rates']]
#             plt.plot(
#                 range(PPO_EVAL_INTERVAL, NUM_PPO_EPISODES + 1, PPO_EVAL_INTERVAL),
#                 win_rates,
#                 label=strategy
#             )
        
#         plt.title('Win Rates by Strategy')
#         plt.xlabel('Episode')
#         plt.ylabel('Win Rate')
#         plt.legend()
#         plt.grid(True)
#         plt.savefig(os.path.join(ppo_models_dir, 'win_rates.png'))
    
#     # Plot opponent modeling metrics if enabled
#     if OPPONENT_MODELING_ENABLED and 'opponent_predictions' in training_results:
#         plt.figure(figsize=(12, 6))
        
#         # Extract accuracy data
#         accuracies = [data['accuracy'] for data in training_results['opponent_predictions']]
#         episodes = [data['episode'] for data in training_results['opponent_predictions']]
        
#         plt.plot(episodes, accuracies, 'o-')
#         plt.axhline(y=0.25, color='r', linestyle='--', label='Random Guess (4 positions)')
        
#         plt.title('Opponent Pick Prediction Accuracy')
#         plt.xlabel('Episode')
#         plt.ylabel('Accuracy')
#         plt.ylim(0, 1)
#         plt.grid(True)
#         plt.legend()
#         plt.savefig(os.path.join(ppo_models_dir, 'opponent_prediction_accuracy.png'))
    
#     plt.close('all')  # Close all figures to free memory
    
#     return ppo_model, training_results

def train_ppo_model(projections, league_info, roster_limits, scoring_settings, models_dir, projection_models=None):
    """
    Train a PPO model for draft optimization with opponent modeling
    
    Parameters:
    -----------
    projections : dict
        Dictionary of player projections by position
    league_info : dict
        Dictionary containing league information
    roster_limits : dict
        Dictionary of roster position limits
    scoring_settings : dict
        Dictionary of scoring settings
    models_dir : str
        Directory to save models
    projection_models : dict, optional
        Dictionary of pre-loaded projection models by position
        
    Returns:
    --------
    tuple
        (PPO model, training results)
    """
    logger.info("Training PPO model for draft optimization...")
    
    # Make sure we have projection models
    if projection_models is None:
        logger.info("Loading projection models...")
        projection_loader = ProjectionModelLoader(models_dir)
        projection_models = projection_loader.models
        
        # Check if we loaded any models
        if not projection_models:
            logger.warning("No projection models found! Using default features.")
        else:
            logger.info(f"Loaded {len(projection_models)} projection models: {list(projection_models.keys())}")
    
    # Create directory for PPO models
    ppo_models_dir = os.path.join(models_dir, 'ppo_models')
    os.makedirs(ppo_models_dir, exist_ok=True)
    
    # Check if pre-trained model should be used
    if USE_EXISTING_PPO_MODEL:
        if USE_HIERARCHICAL_PPO and os.path.exists(HIERARCHICAL_PPO_MODEL_PATH + "_meta_policy.weights.h5"):
            logger.info(f"Loading existing hierarchical PPO model from {HIERARCHICAL_PPO_MODEL_PATH}")
            try:
                ppo_model = HierarchicalPPODrafter.load_model(HIERARCHICAL_PPO_MODEL_PATH)
                return ppo_model, {"loaded_from": HIERARCHICAL_PPO_MODEL_PATH}
            except Exception as e:
                logger.error(f"Error loading hierarchical PPO model: {e}")
                logger.info("Will train a new model instead")
        elif not USE_HIERARCHICAL_PPO and os.path.exists(PPO_MODEL_PATH + "_actor.weights.h5"):
            logger.info(f"Loading existing PPO model from {PPO_MODEL_PATH}")
            try:
                ppo_model = PPODrafter.load_model(PPO_MODEL_PATH)
                return ppo_model, {"loaded_from": PPO_MODEL_PATH}
            except Exception as e:
                logger.error(f"Error loading PPO model: {e}")
                logger.info("Will train a new model instead")
    
    # Extract league size
    league_size = league_info.get('league_info', {}).get('team_count', 10)
    
    # Prepare baseline values for VBD
    baseline_indices = {
        "QB": league_size,
        "RB": league_size * 2 + 2,
        "WR": league_size * 2 + 2,
        "TE": league_size,
        "K": league_size,
        "DST": league_size
    }
    
    # Calculate baseline values
    vbd_baseline = {}
    for position, df in projections.items():
        pos = position.upper()
        if df is not None and not df.empty and "projected_points" in df.columns:
            sorted_df = df.sort_values("projected_points", ascending=False)
            index = baseline_indices.get(pos, 0)
            if index < len(sorted_df):
                vbd_baseline[pos] = sorted_df.iloc[index]["projected_points"]
            else:
                vbd_baseline[pos] = sorted_df.iloc[-1]["projected_points"] if not sorted_df.empty else 0
    
    # Load players from projections
    all_players = DraftSimulator.load_players_from_projections(projections, vbd_baseline)
    
    # Create draft simulator with projection models
    draft_simulator = DraftSimulator(
        players=all_players,
        league_size=league_size,
        roster_limits=roster_limits,
        num_rounds=18,  # Standard number of rounds
        scoring_settings=scoring_settings,
        user_pick=None,
        projection_models=projection_models  # Pass projection models here
    )
    
    # Find or create the RL team
    ppo_draft_position = random.randint(1, league_size)
    rl_team = next((team for team in draft_simulator.teams 
                 if team.draft_position == ppo_draft_position), None)
    if not rl_team:
        # If no RL team found, set the first team to use RL
        rl_team = random.choice(draft_simulator.teams)
        rl_team.strategy = "PPO"
    
    # Create a sample state to determine dimensions
    available_players = [p for p in draft_simulator.players if not p.is_drafted]
    
    # Get starter limits if available in scoring settings
    starter_limits = {}
    if scoring_settings and 'starter_limits' in scoring_settings:
        starter_limits = scoring_settings['starter_limits']
    
    # Create a sample state with opponent modeling
    sample_state = DraftState(
        team=rl_team,
        available_players=available_players,
        round_num=1,
        overall_pick=1,
        league_size=draft_simulator.league_size,
        roster_limits=draft_simulator.roster_limits,
        max_rounds=draft_simulator.num_rounds,
        all_teams=draft_simulator.teams,
        starter_limits=starter_limits
    )
    
    # Get dimensions from sample state
    state_dim = len(sample_state.to_feature_vector())
    action_feature_dim = len(sample_state.get_action_features(0))
    action_dim = 256  # Maximum number of players to consider at once
    
    logger.info(f"State dimension: {state_dim}, Action feature dimension: {action_feature_dim}")
    # Create season simulator for training
    
    if USE_POPULATION_TRAINING:
        logger.info(f"Using population-based training with {POPULATION_SIZE} agents")
        season_simulator = SeasonSimulator(
            teams=draft_simulator.teams,
            num_regular_weeks=NUM_REGULAR_WEEKS,
            num_playoff_teams=NUM_PLAYOFF_TEAMS,
            num_playoff_weeks=NUM_PLAYOFF_WEEKS,
            randomness=RANDOMNESS_FACTOR
        )
        # Create population
        population = PPOPopulation(
            population_size=POPULATION_SIZE,
            state_dim=state_dim,
            action_feature_dim=action_feature_dim,
            action_dim=action_dim,
            hierarchical_ratio=HIERARCHICAL_RATIO,
            evolution_interval=EVOLUTION_INTERVAL,
            tournament_size=TOURNAMENT_SIZE,
            mutation_rate=MUTATION_RATE,
            mutation_strength=MUTATION_STRENGTH,
            elitism_count=ELITISM_COUNT,
            output_dir=ppo_models_dir,
            use_top_n_features=USE_TOP_N_FEATURES,
            enable_crossover=ENABLE_CROSSOVER,
            curriculum_enabled=CURRICULUM_ENABLED,
            opponent_modeling_enabled=OPPONENT_MODELING_ENABLED
        )
        
        # Train the population
        start_time = time.time()
        
        population_results = population.train_population(
            draft_simulator=draft_simulator,
            season_simulator=season_simulator,
            num_episodes=NUM_PPO_EPISODES,
            eval_interval=PPO_EVAL_INTERVAL,
            save_interval=PPO_SAVE_INTERVAL
        )
        
        training_time = time.time() - start_time
        logger.info(f"Population training completed in {training_time:.2f} seconds")
        
        # Get the best agent from the population
        ppo_model = population.get_best_agent()
        
        # Save the best agent to the main path
        if isinstance(ppo_model, HierarchicalPPODrafter):
            ppo_model.save_model(HIERARCHICAL_PPO_MODEL_PATH)
        else:
            ppo_model.save_model(PPO_MODEL_PATH)
        
        return ppo_model, population_results
    else:
        # Create PPO model - either hierarchical or standard
        if USE_HIERARCHICAL_PPO:
            logger.info("Creating hierarchical PPO model")
            ppo_model = HierarchicalPPODrafter(
                state_dim=state_dim,
                action_feature_dim=action_feature_dim,
                action_dim=action_dim,
                lr_meta=META_POLICY_LR,
                lr_sub=SUB_POLICY_LR,
                lr_critic=PPO_LEARNING_RATE,
                gamma=PPO_GAMMA,
                batch_size=PPO_BATCH_SIZE,
                policy_clip=PPO_POLICY_CLIP,
                entropy_coef=PPO_ENTROPY_COEF,
                use_top_n_features=USE_TOP_N_FEATURES,
                curriculum_enabled=CURRICULUM_ENABLED,
                opponent_modeling_enabled=OPPONENT_MODELING_ENABLED
            )
        else:
            logger.info("Creating standard PPO model")
            ppo_model = PPODrafter(
                state_dim=state_dim,
                action_feature_dim=action_feature_dim,
                action_dim=action_dim,
                lr_actor=PPO_LEARNING_RATE,
                lr_critic=PPO_LEARNING_RATE,
                gamma=PPO_GAMMA,
                batch_size=PPO_BATCH_SIZE,
                policy_clip=PPO_POLICY_CLIP,
                entropy_coef=PPO_ENTROPY_COEF,
                use_top_n_features=USE_TOP_N_FEATURES,
                curriculum_enabled=CURRICULUM_ENABLED,
                opponent_modeling_enabled=OPPONENT_MODELING_ENABLED
            )
        
        # Set opponent modeling parameters if enabled
        if OPPONENT_MODELING_ENABLED:
            # Initialize position weights
            ppo_model.position_priority_weights = {
                "QB": 1.0,
                "RB": 1.5,  # Start with slightly higher weight for RB
                "WR": 1.2,  # Start with slightly higher weight for WR
                "TE": 0.9,
                "K": 0.5,
                "DST": 0.5
            }
            
            # Set additional parameters
            ppo_model.position_scarcity_weight = POSITION_SCARCITY_WEIGHT
            ppo_model.run_detection_threshold = RUN_DETECTION_THRESHOLD
            ppo_model.value_cliff_threshold = VALUE_CLIFF_THRESHOLD
            ppo_model.pick_prediction_depth = PICK_PREDICTION_DEPTH
            
            logger.info(f"Opponent modeling enabled with parameters:")
            logger.info(f"  Position scarcity weight: {POSITION_SCARCITY_WEIGHT}")
            logger.info(f"  Run detection threshold: {RUN_DETECTION_THRESHOLD}")
            logger.info(f"  Value cliff threshold: {VALUE_CLIFF_THRESHOLD}")
        
        # Set curriculum parameters if enabled
        if CURRICULUM_ENABLED and CURRICULUM_PARAMS:
            for param_name, param_value in CURRICULUM_PARAMS.items():
                if hasattr(ppo_model, param_name):
                    setattr(ppo_model, param_name, param_value)
                elif isinstance(param_value, dict) and hasattr(ppo_model, param_name):
                    # For dictionary parameters like phase_durations
                    getattr(ppo_model, param_name).update(param_value)
        
        # Create season simulator for training
        season_simulator = SeasonSimulator(
            teams=draft_simulator.teams,
            num_regular_weeks=NUM_REGULAR_WEEKS,
            num_playoff_teams=NUM_PLAYOFF_TEAMS,
            num_playoff_weeks=NUM_PLAYOFF_WEEKS,
            randomness=RANDOMNESS_FACTOR
        )
        
        # Train the model
        logger.info(f"Training {'hierarchical' if USE_HIERARCHICAL_PPO else 'standard'} PPO model for {NUM_PPO_EPISODES} episodes")
        start_time = time.time()
        
        training_results = ppo_model.train(
            draft_simulator=draft_simulator,
            season_simulator=season_simulator,
            num_episodes=NUM_PPO_EPISODES,
            eval_interval=PPO_EVAL_INTERVAL,
            save_interval=PPO_SAVE_INTERVAL,
            save_path=ppo_models_dir
        )
        
        training_time = time.time() - start_time
        logger.info(f"PPO training completed in {training_time:.2f} seconds")
        
        # Save final model to the main path
        if USE_HIERARCHICAL_PPO:
            ppo_model.save_model(HIERARCHICAL_PPO_MODEL_PATH)
        else:
            ppo_model.save_model(PPO_MODEL_PATH)
        
        # Generate visualizations
        try:
            # Standard learning visualizations
            if hasattr(ppo_model, 'generate_learning_visualizations'):
                ppo_model.generate_learning_visualizations(output_dir=ppo_models_dir)
            
            # Opponent modeling visualizations if enabled
            if OPPONENT_MODELING_ENABLED and hasattr(ppo_model, 'generate_opponent_modeling_visualizations'):
                ppo_model.generate_opponent_modeling_visualizations(output_dir=ppo_models_dir)
        except Exception as e:
            logger.error(f"Error generating visualizations: {e}")
        
        return ppo_model, training_results


# def evaluate_ppo_model(ppo_model, projections, league_info, roster_limits, scoring_settings, output_dir, projection_models=None):
#     """
#     Evaluate the PPO model with multiple draft simulations including opponent modeling
    
#     Parameters:
#     -----------
#     ppo_model : PPODrafter
#         Trained PPO model
#     projections : dict
#         Dictionary of player projections by position
#     league_info : dict
#         Dictionary containing league information
#     roster_limits : dict
#         Dictionary of roster position limits
#     scoring_settings : dict
#         Dictionary of scoring settings
#     output_dir : str
#         Directory to save evaluation results
#     projection_models : dict, optional
#         Dictionary of projection models by position
        
#     Returns:
#     --------
#     dict
#         Evaluation results
#     """
#     logger.info("Evaluating PPO model...")
    
#     # Create output directory
#     eval_dir = os.path.join(output_dir, 'ppo_evaluation')
#     os.makedirs(eval_dir, exist_ok=True)
    
#     # If no projection models provided, load them
#     if projection_models is None:
#         projection_loader = ProjectionModelLoader()
#         projection_models = projection_loader.models
    
#     # Extract league size
#     league_size = league_info.get('league_info', {}).get('team_count', 10)
    
#     # Number of trials to run
#     num_trials = 20
    
#     # Prepare baseline values for VBD
#     baseline_indices = {
#         "QB": league_size,
#         "RB": league_size * 2 + 2,
#         "WR": league_size * 2 + 2,
#         "TE": league_size,
#         "K": league_size,
#         "DST": league_size
#     }
    
#     # Calculate baseline values
#     vbd_baseline = {}
#     for position, df in projections.items():
#         pos = position.upper()
#         if df is not None and not df.empty and "projected_points" in df.columns:
#             sorted_df = df.sort_values("projected_points", ascending=False)
#             index = baseline_indices.get(pos, 0)
#             if index < len(sorted_df):
#                 vbd_baseline[pos] = sorted_df.iloc[index]["projected_points"]
#             else:
#                 vbd_baseline[pos] = sorted_df.iloc[-1]["projected_points"] if not sorted_df.empty else 0
    
#     # Results tracking
#     rewards = []
#     ranks = []
#     win_rates = {}
    
#     # Additional opponent modeling metrics if enabled
#     opponent_metrics = {
#         'prediction_accuracy': [],
#         'value_cliff_usage': [],
#         'position_run_detection': []
#     }
    
#     # Run trials
#     for trial in range(num_trials):
#         logger.info(f"Running evaluation trial {trial+1}/{num_trials}")
        
#         # Load fresh players for each trial
#         all_players = DraftSimulator.load_players_from_projections(projections, vbd_baseline)
        
#         # Create draft simulator
#         draft_simulator = DraftSimulator(
#             players=all_players,
#             league_size=league_size,
#             roster_limits=roster_limits,
#             num_rounds=18,
#             scoring_settings=scoring_settings,
#             user_pick=None,
#             projection_models=projection_models 
#         )
        
#         # Get starter limits if available in scoring settings
#         starter_limits = {}
#         if scoring_settings and 'starter_limits' in scoring_settings:
#             starter_limits = scoring_settings['starter_limits']
        
#         # Find or create a team that will use PPO
#         ppo_team = None
#         for team in draft_simulator.teams:
#             if team.strategy == "RL":
#                 team.strategy = "PPO"  # Rename to PPO
#                 ppo_team = team
#                 break
        
#         if not ppo_team:
#             # If no suitable team found, set the first team to use PPO
#             ppo_team = random.choice(draft_simulator.teams)
#             ppo_team.strategy = "PPO"
        
#         # Track opponent modeling metrics for this trial
#         trial_predictions = []
#         trial_cliff_decisions = []
#         trial_run_decisions = []
        
#         # Run draft simulation
#         current_round = 1
#         current_pick = 1
        
#         # For opponent modeling, predict the next few picks
#         predicted_picks = {}  # {pick_number: predicted_position}
        
#         # Run until draft completion
#         while current_round <= draft_simulator.num_rounds:
#             # Determine team picking
#             team_idx = (current_pick - 1) % draft_simulator.league_size
#             if current_round % 2 == 0:  # Snake draft
#                 team_idx = draft_simulator.league_size - 1 - team_idx
            
#             team = draft_simulator.teams[team_idx]
            
#             # If this is the PPO team, use our model in evaluation mode
#             if team.strategy == "PPO":
#                 # Get available players
#                 available_players = [p for p in draft_simulator.players if not p.is_drafted]
                
#                 # Create state with opponent modeling if enabled
#                 if hasattr(ppo_model, 'opponent_modeling_enabled') and ppo_model.opponent_modeling_enabled:
#                     state = DraftState(
#                         team=team,
#                         available_players=available_players,
#                         round_num=current_round,
#                         overall_pick=current_pick,
#                         league_size=draft_simulator.league_size,
#                         roster_limits=draft_simulator.roster_limits,
#                         max_rounds=draft_simulator.num_rounds,
#                         projection_models=projection_models,
#                         use_top_n_features=getattr(ppo_model, 'use_top_n_features', 0),
#                         all_teams=draft_simulator.teams,
#                         starter_limits=starter_limits
#                     )
                    
#                     # Check for value cliffs
#                     if hasattr(state, 'value_cliffs'):
#                         for position in ["QB", "RB", "WR", "TE"]:
#                             if position in state.value_cliffs:
#                                 cliff_info = state.value_cliffs[position]
                                
#                                 if cliff_info.get('has_cliff', False) and cliff_info.get('first_cliff_position', 10) == 0:
#                                     # Found a position with an immediate cliff
#                                     trial_cliff_decisions.append({
#                                         "pick": current_pick,
#                                         "position": position,
#                                         "cliff_magnitude": cliff_info.get('first_cliff_magnitude', 0)
#                                     })
                    
#                     # Check for position runs
#                     if hasattr(state, 'position_runs'):
#                         for position, run_info in state.position_runs.items():
#                             if run_info.get('is_run', False):
#                                 # Found a position with an active run
#                                 trial_run_decisions.append({
#                                     "pick": current_pick,
#                                     "position": position,
#                                     "run_percentage": run_info.get('run_percentage', 0)
#                                 })
                    
#                     # Predict opponent picks
#                     for future_pick in range(current_pick + 1, current_pick + 8):
#                         # Only predict for picks before our next turn
#                         if future_pick < current_pick + state.picks_until_next_turn:
#                             # Find the team picking at this position
#                             future_round = (future_pick - 1) // draft_simulator.league_size + 1
#                             future_pick_in_round = (future_pick - 1) % draft_simulator.league_size
                            
#                             # Determine if this round is ascending or descending
#                             future_is_ascending = (future_round % 2 == 1)
                            
#                             # Calculate team index
#                             if future_is_ascending:
#                                 future_team_idx = future_pick_in_round
#                             else:
#                                 future_team_idx = draft_simulator.league_size - 1 - future_pick_in_round
                            
#                             # Get the team at this position
#                             future_team = None
#                             for t in draft_simulator.teams:
#                                 if t.draft_position == future_team_idx + 1:  # Convert to 1-indexed
#                                     future_team = t
#                                     break
                            
#                             if future_team:
#                                 # Use opponent needs to predict position
#                                 opponent_needs = state.opponent_needs.get(future_team.name, {})
                                
#                                 # Determine most likely position target
#                                 max_urgency = 0
#                                 predicted_position = None
                                
#                                 for position, need_info in opponent_needs.items():
#                                     if position in ["QB", "RB", "WR", "TE", "K", "DST"]:
#                                         urgency = need_info.get('urgency', 0) * need_info.get('remaining', 0)
                                        
#                                         if urgency > max_urgency:
#                                             max_urgency = urgency
#                                             predicted_position = position
                                
#                                 # Record the prediction
#                                 if predicted_position:
#                                     predicted_picks[future_pick] = predicted_position
#                 else:
#                     # Create regular state without opponent modeling
#                     state = DraftState(
#                         team=team,
#                         available_players=available_players,
#                         round_num=current_round,
#                         overall_pick=current_pick,
#                         league_size=draft_simulator.league_size,
#                         roster_limits=draft_simulator.roster_limits,
#                         max_rounds=draft_simulator.num_rounds,
#                         projection_models=projection_models,
#                         use_top_n_features=getattr(ppo_model, 'use_top_n_features', 0)
#                     )
                
#                 # Select action (without training)
#                 action, _, _ = ppo_model.select_action(state, training=False)
                
#                 # Execute action
#                 if action is not None and action < len(state.valid_players):
#                     player = state.valid_players[action]
#                     team.add_player(player, current_round, current_pick)
#                 else:
#                     # Fallback to best available
#                     draft_simulator._make_pick(team, current_round, current_pick)
#             else:
#                 # Use simulator's strategy
#                 picked_player = draft_simulator._make_pick(team, current_round, current_pick)
                
#                 # Check if we predicted this pick
#                 if current_pick in predicted_picks and picked_player:
#                     predicted_position = predicted_picks[current_pick]
#                     actual_position = picked_player.position
                    
#                     # Record prediction accuracy
#                     prediction_correct = (predicted_position == actual_position)
                    
#                     # Track prediction
#                     trial_predictions.append({
#                         "pick": current_pick,
#                         "predicted_position": predicted_position,
#                         "actual_position": actual_position,
#                         "correct": prediction_correct
#                     })
            
#             # Move to next pick
#             current_pick += 1
#             if current_pick > current_round * draft_simulator.league_size:
#                 current_round += 1
                
#             if current_pick > draft_simulator.num_rounds * draft_simulator.league_size:
#                 break
        
#         # Track opponent modeling metrics
#         if trial_predictions:
#             correct_predictions = sum(1 for pred in trial_predictions if pred.get("correct", False))
#             total_predictions = len(trial_predictions)
#             prediction_accuracy = correct_predictions / max(1, total_predictions)
#             opponent_metrics['prediction_accuracy'].append(prediction_accuracy)
        
#         opponent_metrics['value_cliff_usage'].append(len(trial_cliff_decisions))
#         opponent_metrics['position_run_detection'].append(len(trial_run_decisions))
        
#         # Simulate season
#         season_simulator = SeasonSimulator(
#             teams=draft_simulator.teams,
#             num_regular_weeks=NUM_REGULAR_WEEKS,
#             num_playoff_teams=NUM_PLAYOFF_TEAMS,
#             num_playoff_weeks=NUM_PLAYOFF_WEEKS,
#             randomness=RANDOMNESS_FACTOR
#         )
        
#         season_results = season_simulator.simulate_season()
#         evaluation = SeasonEvaluator(draft_simulator.teams, season_results)
        
#         # Get PPO team results
#         ppo_metrics = None
#         for metrics in evaluation.metrics.get("PPO", {}).get("teams", []):
#             if metrics["team_name"] == ppo_team.name:
#                 ppo_metrics = metrics
#                 break
        
#         if ppo_metrics:
#             # Calculate reward using the same logic as in training
#             reward = (
#                 -1.0 * ppo_metrics["rank"] +
#                 2.0 * ppo_metrics["wins"] +
#                 0.02 * ppo_metrics["points_for"] +
#                 10.0 * (1 if ppo_metrics.get("playoff_result") == "Champion" else 0) +
#                 5.0 * (1 if ppo_metrics.get("playoff_result") in ["Runner-up", "Third Place"] else 0) +
#                 2.0 * (1 if ppo_metrics.get("playoff_result") == "Playoff Qualification" else 0)
#             )
            
#             rewards.append(reward)
#             ranks.append(ppo_metrics["rank"])
            
#             logger.info(f"Trial {trial+1} - Reward: {reward:.2f}, Rank: {ppo_metrics['rank']}")
        
#         # Track win rates
#         for strategy, metrics in evaluation.metrics.items():
#             if strategy not in win_rates:
#                 win_rates[strategy] = []
            
#             win_rate = metrics["avg_wins"] / (metrics["avg_wins"] + metrics.get("avg_losses", 0))
#             win_rates[strategy].append(win_rate)
    
#     # Calculate average metrics
#     avg_reward = sum(rewards) / len(rewards) if rewards else 0
#     avg_rank = sum(ranks) / len(ranks) if ranks else 0
#     avg_win_rates = {s: sum(rates) / len(rates) for s, rates in win_rates.items()}
    
#     # Calculate average opponent modeling metrics
#     if opponent_metrics['prediction_accuracy']:
#         avg_prediction_accuracy = sum(opponent_metrics['prediction_accuracy']) / len(opponent_metrics['prediction_accuracy'])
#     else:
#         avg_prediction_accuracy = 0
    
#     avg_cliff_usage = sum(opponent_metrics['value_cliff_usage']) / len(opponent_metrics['value_cliff_usage'])
#     avg_run_detection = sum(opponent_metrics['position_run_detection']) / len(opponent_metrics['position_run_detection'])
    
#     logger.info(f"PPO Evaluation results:")
#     logger.info(f"  Average reward: {avg_reward:.2f}")
#     logger.info(f"  Average rank: {avg_rank:.2f}")
#     logger.info(f"  Win rates by strategy:")
#     for strategy, rate in avg_win_rates.items():
#         logger.info(f"    {strategy}: {rate:.3f}")
    
#     # Log opponent modeling metrics if enabled
#     if hasattr(ppo_model, 'opponent_modeling_enabled') and ppo_model.opponent_modeling_enabled:
#         logger.info(f"  Opponent modeling metrics:")
#         logger.info(f"    Pick prediction accuracy: {avg_prediction_accuracy:.3f}")
#         logger.info(f"    Value cliff usage: {avg_cliff_usage:.2f} per draft")
#         logger.info(f"    Position run detection: {avg_run_detection:.2f} per draft")
    
#     # Create evaluation plots
#     # Rank distribution plot
#     plt.figure(figsize=(10, 6))
#     sns.histplot(ranks, bins=range(1, max(ranks) + 2), kde=True)
#     plt.title('PPO Rank Distribution')
#     plt.xlabel('Rank')
#     plt.ylabel('Frequency')
#     plt.savefig(os.path.join(eval_dir, 'rank_distribution.png'))
    
#     # Win rate comparison plot
#     plt.figure(figsize=(12, 6))
#     strategies = list(avg_win_rates.keys())
#     win_rates_values = [avg_win_rates[s] for s in strategies]
    
#     plt.bar(strategies, win_rates_values)
#     plt.title('Average Win Rate by Strategy')
#     plt.xlabel('Strategy')
#     plt.ylabel('Win Rate')
#     plt.ylim(0, max(win_rates_values) * 1.2)
    
#     # Add win rate values on top of bars
#     for i, v in enumerate(win_rates_values):
#         plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
#     plt.savefig(os.path.join(eval_dir, 'win_rate_comparison.png'))
    
#     # Create opponent modeling plots if enabled
#     if hasattr(ppo_model, 'opponent_modeling_enabled') and ppo_model.opponent_modeling_enabled:
#         plt.figure(figsize=(10, 6))
        
#         # Create bar chart for opponent modeling metrics
#         metrics_names = ["Prediction\nAccuracy", "Value Cliff\nUsage", "Position Run\nDetection"]
#         metrics_values = [avg_prediction_accuracy, avg_cliff_usage, avg_run_detection]
        
#         plt.bar(metrics_names, metrics_values, color=['blue', 'green', 'orange'])
#         plt.title('Opponent Modeling Metrics')
#         plt.ylabel('Value')
        
#         # Add values on top of bars
#         for i, v in enumerate(metrics_values):
#             plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
        
#         plt.savefig(os.path.join(eval_dir, 'opponent_modeling_metrics.png'))
    
#     # Save results to JSON
#     results = {
#         'avg_reward': avg_reward,
#         'avg_rank': avg_rank,
#         'avg_win_rates': avg_win_rates,
#         'ranks': ranks,
#         'rewards': rewards
#     }
    
#     # Add opponent modeling metrics if enabled
#     if hasattr(ppo_model, 'opponent_modeling_enabled') and ppo_model.opponent_modeling_enabled:
#         results['opponent_modeling'] = {
#             'avg_prediction_accuracy': avg_prediction_accuracy,
#             'avg_cliff_usage': avg_cliff_usage,
#             'avg_run_detection': avg_run_detection
#         }
    
#     with open(os.path.join(eval_dir, 'evaluation_results.json'), 'w') as f:
#         # Convert numpy arrays to lists for JSON serialization
#         for key, value in results.items():
#             if isinstance(value, np.ndarray):
#                 results[key] = value.tolist()
#             elif isinstance(value, dict):
#                 for k, v in value.items():
#                     if isinstance(v, np.ndarray):
#                         results[key][k] = v.tolist()
        
#         json.dump(results, f, indent=2)
    
#     plt.close('all')  # Close all figures to free memory
    
#     return results

def evaluate_ppo_model(ppo_model, projections, league_info, roster_limits, scoring_settings, output_dir, projection_models=None):
    """
    Evaluate the PPO model with multiple draft simulations including opponent modeling
    
    Parameters:
    -----------
    ppo_model : PPODrafter or HierarchicalPPODrafter
        Trained PPO model
    projections : dict
        Dictionary of player projections by position
    league_info : dict
        Dictionary containing league information
    roster_limits : dict
        Dictionary of roster position limits
    scoring_settings : dict
        Dictionary of scoring settings
    output_dir : str
        Directory to save evaluation results
    projection_models : dict, optional
        Dictionary of projection models by position
        
    Returns:
    --------
    dict
        Evaluation results
    """
    # Determine if we're using hierarchical PPO
    is_hierarchical = isinstance(ppo_model, HierarchicalPPODrafter)
    logger.info(f"Evaluating {'hierarchical' if is_hierarchical else 'standard'} PPO model...")
    
    # Create output directory
    eval_dir = os.path.join(output_dir, 'ppo_evaluation')
    os.makedirs(eval_dir, exist_ok=True)
    
    # If no projection models provided, load them
    if projection_models is None:
        projection_loader = ProjectionModelLoader()
        projection_models = projection_loader.models
    
    # Extract league size
    league_size = league_info.get('league_info', {}).get('team_count', 10)
    
    # Number of trials to run
    num_trials = 20
    
    # Prepare baseline values for VBD
    baseline_indices = {
        "QB": league_size,
        "RB": league_size * 2 + 2,
        "WR": league_size * 2 + 2,
        "TE": league_size,
        "K": league_size,
        "DST": league_size
    }
    
    # Calculate baseline values and prepare for simulation
    vbd_baseline = {}
    for position, df in projections.items():
        pos = position.upper()
        if df is not None and not df.empty and "projected_points" in df.columns:
            sorted_df = df.sort_values("projected_points", ascending=False)
            index = baseline_indices.get(pos, 0)
            if index < len(sorted_df):
                vbd_baseline[pos] = sorted_df.iloc[index]["projected_points"]
            else:
                vbd_baseline[pos] = sorted_df.iloc[-1]["projected_points"] if not sorted_df.empty else 0
    
    # Results tracking
    rewards = []
    ranks = []
    win_rates = {}
    
    # Additional opponent modeling metrics if enabled
    opponent_metrics = {
        'prediction_accuracy': [],
        'value_cliff_usage': [],
        'position_run_detection': []
    }
    
    # Run trials
    for trial in range(num_trials):
        logger.info(f"Running evaluation trial {trial+1}/{num_trials}")
        
        # Load fresh players for each trial
        all_players = DraftSimulator.load_players_from_projections(projections, vbd_baseline)
        
        # Create draft simulator
        draft_simulator = DraftSimulator(
            players=all_players,
            league_size=league_size,
            roster_limits=roster_limits,
            num_rounds=18,
            scoring_settings=scoring_settings,
            user_pick=None,
            projection_models=projection_models 
        )
        
        # Get starter limits if available in scoring settings
        starter_limits = {}
        if scoring_settings and 'starter_limits' in scoring_settings:
            starter_limits = scoring_settings['starter_limits']
        
        # Find or create a team that will use PPO
        ppo_team = None
        for team in draft_simulator.teams:
            if team.strategy == "RL":
                team.strategy = "PPO"  # Rename to PPO
                ppo_team = team
                break
        
        if not ppo_team:
            # If no suitable team found, set the first team to use PPO
            ppo_team = random.choice(draft_simulator.teams)
            ppo_team.strategy = "PPO"
        
        # Track opponent modeling metrics for this trial
        trial_predictions = []
        trial_cliff_decisions = []
        trial_run_decisions = []
        
        # Run draft simulation
        current_round = 1
        current_pick = 1
        
        # For opponent modeling, predict the next few picks
        predicted_picks = {}  # {pick_number: predicted_position}
        
        # Run until draft completion
        while current_round <= draft_simulator.num_rounds:
            # Determine team picking
            team_idx = (current_pick - 1) % draft_simulator.league_size
            if current_round % 2 == 0:  # Snake draft
                team_idx = draft_simulator.league_size - 1 - team_idx
            
            team = draft_simulator.teams[team_idx]
            
            # If this is the PPO team, use our model in evaluation mode
            if team.strategy == "PPO":
                # Get available players
                available_players = [p for p in draft_simulator.players if not p.is_drafted]
                
                # Create state with opponent modeling if enabled
                state = DraftState(
                    team=team,
                    available_players=available_players,
                    round_num=current_round,
                    overall_pick=current_pick,
                    league_size=draft_simulator.league_size,
                    roster_limits=draft_simulator.roster_limits,
                    max_rounds=draft_simulator.num_rounds,
                    projection_models=projection_models,
                    use_top_n_features=getattr(ppo_model, 'use_top_n_features', 0),
                    all_teams=draft_simulator.teams,
                    starter_limits=starter_limits
                )
                
                # Track opponent modeling metrics
                if hasattr(ppo_model, 'opponent_modeling_enabled') and ppo_model.opponent_modeling_enabled:
                    # Check for value cliffs
                    if hasattr(state, 'value_cliffs'):
                        for position in ["QB", "RB", "WR", "TE"]:
                            if position in state.value_cliffs:
                                cliff_info = state.value_cliffs[position]
                                
                                if cliff_info.get('has_cliff', False) and cliff_info.get('first_cliff_position', 10) == 0:
                                    # Found a position with an immediate cliff
                                    trial_cliff_decisions.append({
                                        "pick": current_pick,
                                        "position": position,
                                        "cliff_magnitude": cliff_info.get('first_cliff_magnitude', 0)
                                    })
                    
                    # Check for position runs
                    if hasattr(state, 'position_runs'):
                        for position, run_info in state.position_runs.items():
                            if run_info.get('is_run', False):
                                # Found a position with an active run
                                trial_run_decisions.append({
                                    "pick": current_pick,
                                    "position": position,
                                    "run_percentage": run_info.get('run_percentage', 0)
                                })
                
                # Select action based on model type (hierarchical or standard)
                if is_hierarchical:
                    # Hierarchical PPO returns position, action, pos_prob, action_prob, value, features
                    selected_position, action, pos_prob, action_prob, value, features = ppo_model.select_action(state, training=False)
                    
                    # Execute action if valid
                    if selected_position is not None and action is not None and action < len(state.valid_players):
                        player = state.valid_players[action]
                        team.add_player(player, current_round, current_pick)
                    else:
                        # Fallback to best available
                        action = None  # Set action to None to trigger fallback
                else:
                    # Standard PPO returns action, prob, value, features
                    action, prob, value, features = ppo_model.select_action(state, training=False)
                    
                    # Execute action if valid
                    if action is not None and action < len(state.valid_players):
                        player = state.valid_players[action]
                        team.add_player(player, current_round, current_pick)
                
                # If action is None or invalid, fall back to best available
                if action is None or (is_hierarchical and selected_position is None):
                    # Fallback to best available
                    valid_players = [p for p in available_players if p.position in 
                                     [pos for pos in draft_simulator.roster_limits.keys() 
                                      if team.can_draft_position(pos)]]
                    
                    if valid_players:
                        # Sort by projected points
                        valid_players.sort(key=lambda p: p.projected_points, reverse=True)
                        player = valid_players[0]
                        team.add_player(player, current_round, current_pick)
                        logger.info(f"Fallback draft: {player.name} ({player.position})")
            
            # Otherwise, use the team's strategy
            else:
                # Make the pick using the team's strategy
                picked_player = draft_simulator._make_pick(team, current_round, current_pick)
                
                # Validate opponent pick prediction if we made one
                if current_pick in predicted_picks and picked_player:
                    predicted_position = predicted_picks[current_pick]
                    actual_position = picked_player.position
                    
                    # Record prediction accuracy
                    prediction_correct = (predicted_position == actual_position)
                    
                    # Track prediction
                    prediction_data = {
                        "pick": current_pick,
                        "predicted_position": predicted_position,
                        "actual_position": actual_position,
                        "correct": prediction_correct
                    }
                    trial_predictions.append(prediction_data)
            
            # Move to next pick
            current_pick += 1
            if current_pick > current_round * draft_simulator.league_size:
                current_round += 1
                
            if current_pick > draft_simulator.num_rounds * draft_simulator.league_size:
                break
        
        # Track opponent modeling metrics
        if trial_predictions:
            correct_predictions = sum(1 for pred in trial_predictions if pred.get("correct", False))
            total_predictions = len(trial_predictions)
            prediction_accuracy = correct_predictions / max(1, total_predictions)
            opponent_metrics['prediction_accuracy'].append(prediction_accuracy)
        
        opponent_metrics['value_cliff_usage'].append(len(trial_cliff_decisions))
        opponent_metrics['position_run_detection'].append(len(trial_run_decisions))
        
        # Simulate season
        season_simulator = SeasonSimulator(
            teams=draft_simulator.teams,
            num_regular_weeks=NUM_REGULAR_WEEKS,
            num_playoff_teams=NUM_PLAYOFF_TEAMS,
            num_playoff_weeks=NUM_PLAYOFF_WEEKS,
            randomness=RANDOMNESS_FACTOR
        )
        
        season_results = season_simulator.simulate_season()
        evaluation = SeasonEvaluator(draft_simulator.teams, season_results)
        
        # Get PPO team results
        ppo_metrics = None
        for metrics in evaluation.metrics.get("PPO", {}).get("teams", []):
            if metrics["team_name"] == ppo_team.name:
                ppo_metrics = metrics
                break
        
        if ppo_metrics:
            # Calculate reward using the same logic as in training
            reward = (
                -1.0 * ppo_metrics["rank"] +
                2.0 * ppo_metrics["wins"] +
                0.02 * ppo_metrics["points_for"] +
                10.0 * (1 if ppo_metrics.get("playoff_result") == "Champion" else 0) +
                5.0 * (1 if ppo_metrics.get("playoff_result") in ["Runner-up", "Third Place"] else 0) +
                2.0 * (1 if ppo_metrics.get("playoff_result") == "Playoff Qualification" else 0)
            )
            
            rewards.append(reward)
            ranks.append(ppo_metrics["rank"])
            
            logger.info(f"Trial {trial+1} - Reward: {reward:.2f}, Rank: {ppo_metrics['rank']}")
        
        # Track win rates
        for strategy, metrics in evaluation.metrics.items():
            if strategy not in win_rates:
                win_rates[strategy] = []
            
            win_rate = metrics["avg_wins"] / (metrics["avg_wins"] + metrics.get("avg_losses", 0))
            win_rates[strategy].append(win_rate)
    
    # Calculate average metrics
    avg_reward = sum(rewards) / len(rewards) if rewards else 0
    avg_rank = sum(ranks) / len(ranks) if ranks else 0
    avg_win_rates = {s: sum(rates) / len(rates) for s, rates in win_rates.items()}
    
    # Calculate average opponent modeling metrics
    if opponent_metrics['prediction_accuracy']:
        avg_prediction_accuracy = sum(opponent_metrics['prediction_accuracy']) / len(opponent_metrics['prediction_accuracy'])
    else:
        avg_prediction_accuracy = 0
    
    avg_cliff_usage = sum(opponent_metrics['value_cliff_usage']) / len(opponent_metrics['value_cliff_usage'])
    avg_run_detection = sum(opponent_metrics['position_run_detection']) / len(opponent_metrics['position_run_detection'])
    
    logger.info(f"{'Hierarchical' if is_hierarchical else 'Standard'} PPO Evaluation results:")
    logger.info(f"  Average reward: {avg_reward:.2f}")
    logger.info(f"  Average rank: {avg_rank:.2f}")
    logger.info(f"  Win rates by strategy:")
    for strategy, rate in avg_win_rates.items():
        logger.info(f"    {strategy}: {rate:.3f}")
    
    # Log opponent modeling metrics if enabled
    if hasattr(ppo_model, 'opponent_modeling_enabled') and ppo_model.opponent_modeling_enabled:
        logger.info(f"  Opponent modeling metrics:")
        logger.info(f"    Pick prediction accuracy: {avg_prediction_accuracy:.3f}")
        logger.info(f"    Value cliff usage: {avg_cliff_usage:.2f} per draft")
        logger.info(f"    Position run detection: {avg_run_detection:.2f} per draft")
    
    # Create evaluation plots
    # Rank distribution plot
    plt.figure(figsize=(10, 6))
    sns.histplot(ranks, bins=range(1, max(ranks) + 2), kde=True)
    plt.title(f"{'Hierarchical' if is_hierarchical else 'Standard'} PPO Rank Distribution")
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.savefig(os.path.join(eval_dir, 'rank_distribution.png'))
    
    # Win rate comparison plot
    plt.figure(figsize=(12, 6))
    strategies = list(avg_win_rates.keys())
    win_rates_values = [avg_win_rates[s] for s in strategies]
    
    plt.bar(strategies, win_rates_values)
    plt.title('Average Win Rate by Strategy')
    plt.xlabel('Strategy')
    plt.ylabel('Win Rate')
    plt.ylim(0, max(win_rates_values) * 1.2)
    
    # Add win rate values on top of bars
    for i, v in enumerate(win_rates_values):
        plt.text(i, v + 0.01, f"{v:.3f}", ha='center')
    
    plt.savefig(os.path.join(eval_dir, 'win_rate_comparison.png'))
    
    # Save results to JSON
    results = {
        'model_type': 'hierarchical' if is_hierarchical else 'standard',
        'avg_reward': avg_reward,
        'avg_rank': avg_rank,
        'avg_win_rates': avg_win_rates,
        'ranks': ranks,
        'rewards': rewards
    }
    
    # Add opponent modeling metrics if enabled
    if hasattr(ppo_model, 'opponent_modeling_enabled') and ppo_model.opponent_modeling_enabled:
        results['opponent_modeling'] = {
            'avg_prediction_accuracy': avg_prediction_accuracy,
            'avg_cliff_usage': avg_cliff_usage,
            'avg_run_detection': avg_run_detection
        }
    
    with open(os.path.join(eval_dir, 'evaluation_results.json'), 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        for key, value in results.items():
            if isinstance(value, np.ndarray):
                results[key] = value.tolist()
            elif isinstance(value, dict):
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        results[key][k] = v.tolist()
        
        json.dump(results, f, indent=2)
    
    plt.close('all')  # Close all figures to free memory
    
    return results


def run_draft_simulations(projections, league_info, roster_limits, scoring_settings, ppo_model=None, results_dir=None):
    """
    Run draft simulations with opponent modeling
    
    Parameters:
    -----------
    projections : dict
        Dictionary of player projections by position
    league_info : dict
        Dictionary containing league information
    roster_limits : dict
        Dictionary of roster position limits
    scoring_settings : dict
        Dictionary of scoring settings
    ppo_model : PPODrafter
        PPO model to use for RL strategy (if None, use baseline strategies)
    results_dir : str
        Directory to save results
        
    Returns:
    --------
    dict
        Dictionary of simulation results
    """
    
    logger.info("Running draft simulations...")
    
    # Ensure results directory exists
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    # Extract league size
    league_size = league_info.get('league_info', {}).get('team_count', 10)
    
    # Prepare baseline values for VBD
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
    
    # Create a new dictionary to track performance against different opponent types
    opponent_type_performance = {}
    
    # Get starter limits if available in scoring settings
    starter_limits = {}
    if scoring_settings and 'starter_limits' in scoring_settings:
        starter_limits = scoring_settings['starter_limits']
    
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
        
        # Inject PPO model if available
        if ppo_model is not None:
            # Replace "RL" strategy with "PPO" strategy
            for team in draft_sim.teams:
                if team.strategy == "RL":
                    team.strategy = "PPO"
            
            # Rename strategy in metrics tracking
            if "RL" in strategy_metrics and "PPO" not in strategy_metrics:
                strategy_metrics["PPO"] = strategy_metrics.pop("RL")
        
        # Run draft
        teams, draft_history = draft_sim.run_draft(ppo_model=ppo_model)
        
        # Reset random seed for next simulation
        random.seed(time.time() + sim)
        
        # Store results
        all_teams.append(teams)
        all_draft_histories.append(draft_history)
        
        # Generate draft report
        if results_dir:
            report_path = os.path.join(results_dir, f"draft_simulation_{sim+1}.csv")
            draft_sim.create_draft_report(output_path=report_path)
        
        # Record which teams are using which strategies
        team_strategies = {team.name: team.strategy for team in draft_sim.teams}
        
        # Calculate metrics by strategy
        for team in teams:
            if team.strategy in strategy_metrics:
                strategy_metrics[team.strategy].append({
                    "total_points": team.get_total_projected_points(),
                    "starting_points": team.get_starting_lineup_points(),
                    "team_name": team.name,
                    "draft_position": team.draft_position,
                    "opponent_strategies": [s for s in team_strategies.values() if s != team.strategy]
                })
            
            # If opponent modeling is enabled and we're using a PPO model,
            # track performance against different opponent types
            if team.strategy == "PPO" and ppo_model and hasattr(ppo_model, 'opponent_modeling_enabled') and ppo_model.opponent_modeling_enabled:
                ppo_metrics = {
                    "total_points": team.get_total_projected_points(),
                    "starting_points": team.get_starting_lineup_points(),
                    "team_name": team.name,
                    "draft_position": team.draft_position,
                    "opponent_strategies": [s for s in team_strategies.values() if s != "PPO"]
                }
                
                # Track PPO performance against each strategy
                for strategy in set(team_strategies.values()):
                    if strategy != "PPO":
                        if strategy not in opponent_type_performance:
                            opponent_type_performance[strategy] = []
                        
                        opponent_type_performance[strategy].append(ppo_metrics)
    
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
    
    # Analyze opponent type performance if available
    opponent_df = None
    if opponent_type_performance:
        opponent_summary = {}
        
        for strategy, metrics_list in opponent_type_performance.items():
            avg_total = sum(m["total_points"] for m in metrics_list) / max(1, len(metrics_list))
            avg_starters = sum(m["starting_points"] for m in metrics_list) / max(1, len(metrics_list))
            
            opponent_summary[strategy] = {
                "avg_total_points": avg_total,
                "avg_starter_points": avg_starters,
                "num_simulations": len(metrics_list)
            }
        
        # Create opponent comparison DataFrame
        opponent_df = pd.DataFrame([
            {
                "Opponent Type": strategy,
                "Avg Total Points": data["avg_total_points"],
                "Avg Starter Points": data["avg_starter_points"],
                "Simulations": data["num_simulations"]
            }
            for strategy, data in opponent_summary.items()
        ])
        
        # Save opponent summary
        if results_dir:
            opponent_summary_path = os.path.join(results_dir, "opponent_strategy_comparison.csv")
            opponent_df.to_csv(opponent_summary_path, index=False)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            
            # Plot average starter points by opponent type
            ax = sns.barplot(
                x="Opponent Type", 
                y="Avg Starter Points", 
                data=opponent_df, 
                palette="viridis"
            )
            
            # Add data labels
            for i, row in enumerate(opponent_df.itertuples()):
                ax.text(
                    i, 
                    row._3 + 5,  # Add a small offset
                    f"{row._3:.1f}",
                    ha='center'
                )
            
            plt.title("PPO Performance Against Different Opponent Types")
            plt.grid(axis='y', linestyle='--', alpha=0.7)
            
            # Save figure
            plt.tight_layout()
            plt.savefig(os.path.join(results_dir, "opponent_strategy_comparison.png"), dpi=300)
            plt.close()
        
        # Add to summary dictionary
        summary["opponent_analysis"] = opponent_summary
    
    # Save summary
    if results_dir:
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
        "opponent_df": opponent_df,
        "teams": all_teams,
        "draft_histories": all_draft_histories
    }

def load_league_settings(league_id, year, espn_s2, swid):
    """Load league settings from ESPN API or config file"""
    # First, try to generate/update the config file
    try:
        from generate_config import generate_config
        config_generated = generate_config(league_id, year, espn_s2, swid)
        logger.info("Config file successfully generated/updated")
    except Exception as e:
        logger.warning(f"Failed to generate config file: {e}")
        config_generated = False
    
    # Try to load league data from ESPN API
    try:
        league_data = load_espn_league_data(league_id, year, espn_s2, swid)
        
        if league_data:
            # Extract roster limits and scoring settings from league data
            roster_limits = league_data.get('roster_settings', {})
            scoring_settings = league_data.get('scoring_settings', {})
            
            logger.info(f"Successfully loaded league settings from ESPN API")
            return league_data, roster_limits, scoring_settings
    except Exception as e:
        logger.warning(f"Error loading from ESPN API: {e}")
    
    # If API failed, try loading from config file
    try:
        config_path = "configs/league_settings.json"
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            roster_limits = config.get('roster_settings', {})
            scoring_settings = config.get('scoring_settings', {})
            
            logger.info(f"Successfully loaded league settings from config file")
            return config, roster_limits, scoring_settings
        else:
            logger.warning(f"Config file not found: {config_path}")
    except Exception as e:
        logger.warning(f"Error loading config file: {e}")
    
    # Only as a last resort, use minimal defaults
    logger.error("Could not load league settings from API or config file. Using minimal defaults.")
    
    # # Create minimal league data with defaults
    # league_data = {
    #     "league_info": {
    #         "name": "Default League",
    #         "team_count": 10,
    #         "playoff_teams": 6
    #     }
    # }
    
    # These should be replaced with API data, we only include defaults as a fallback
    # roster_limits = {
    #     "QB": 3,
    #     "RB": 10,
    #     "WR": 10,
    #     "TE": 2,
    #     "FLEX": 10,
    #     "D/ST": 10,
    #     "K": 10,
    #     "BE": 8,
    #     "IR": 1
    # }
    
    # scoring_settings = {
    #     "passing_yards": 0.04,
    #     "passing_td": 4,
    #     "interception": -2,
    #     "rushing_yards": 0.1,
    #     "rushing_td": 6,
    #     "receiving_yards": 0.1,
    #     "receiving_td": 6,
    #     "reception": 0.5,
    #     "fumble_lost": -2
    # }
    
    return league_data, roster_limits, scoring_settings



def main():
    start_time = datetime.now()
    logger.info(f"Starting fantasy football analysis at {start_time}")
    
    # Load configuration from file (optional, will use defaults if not found)
    config = load_config(CONFIG_PATH)
    
    # Use config values if available, otherwise use the ones defined at the top
    league_id = config.get('league_id', LEAGUE_ID)
    year = config.get('league_year', LEAGUE_YEAR)
    espn_s2 = config.get('espn_s2', ESPN_S2)
    swid = config.get('swid', SWID)
    start_year = config.get('start_year', START_YEAR)
    
    # Load league settings
    league_data, roster_limits, scoring_settings = load_league_settings(league_id, year, espn_s2, swid)
    
    # roster_limits = {
    #         "QB": 3,  # Maximum 3 QBs
    #         "RB": 10,  # Effectively unlimited
    #         "WR": 10,  # Effectively unlimited
    #         "TE": 3,  # Effectively unlimited
    #         "K": 3,    # Effectively unlimited
    #         "DST": 3,  # Effectively unlimited
    #         "BE": 8,  # Calculated bench spots
    #         "IR": 1    # One IR spot
    #     }
    
    roster_limits = config.get('roster_settings', {})
    
    # Cache options from config
    use_cached_raw = config.get('use_cached_raw_data', USE_CACHED_RAW_DATA)
    use_cached_processed = config.get('use_cached_processed_data', USE_CACHED_PROCESSED_DATA)
    use_cached_features = config.get('use_cached_feature_sets', USE_CACHED_FEATURE_SETS)
    create_viz = config.get('create_visualizations', CREATE_VISUALIZATIONS)
    skip_model = config.get('skip_model_training', SKIP_MODEL_TRAINING)
    
    # Simulation options from config
    run_draft_sims = config.get('run_draft_simulations', RUN_DRAFT_SIMULATIONS)
    run_season_sims = config.get('run_season_simulations', RUN_SEASON_SIMULATIONS)
    train_ppo = config.get('train_ppo_model', TRAIN_PPO_MODEL)
    
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
        analyzer = FantasyFootballAnalyzer(
            league_id=league_id,
            year=year,
            espn_s2=espn_s2,
            swid=swid
        )
        
        # Run exploratory data analysis
        analyzer.explore_league_settings()
        for position in ['QB', 'RB', 'WR', 'TE']:
            analyzer.explore_player_data(position)
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
        
        # Cache projections for draft assistant
        try:
            with open(os.path.join(MODELS_DIR, 'player_projections.pkl'), 'wb') as f:
                pickle.dump(projections, f)
            logger.info("Cached player projections for draft assistant")
        except Exception as e:
            logger.error(f"Error caching projections: {e}")
        
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
    
    
    # Step 7: Train PPO model if requested
    ppo_model = None
    if train_ppo:
        logger.info("Step 7: Training PPO reinforcement learning model...")
        projection_loader = ProjectionModelLoader()
        projection_models = projection_loader.models
        # Train PPO model
        ppo_model, ppo_results = train_ppo_model(
            projections=projections,
            league_info=league_data,
            roster_limits=roster_limits,
            scoring_settings=scoring_settings,
            models_dir=MODELS_DIR,
            projection_models=projection_models 
        )
        
        # Evaluate PPO model
        ppo_eval_results = evaluate_ppo_model(
            ppo_model=ppo_model,
            projections=projections,
            league_info=league_data,
            roster_limits=roster_limits,
            scoring_settings=scoring_settings,
            output_dir=OUTPUT_DIR,
            projection_models=projection_models 
        )
        
        logger.info(f"PPO model training and evaluation complete")
    elif USE_EXISTING_PPO_MODEL:
        logger.info(f"Loading existing {'hierarchical' if USE_HIERARCHICAL_PPO else 'standard'} PPO model")
        
        try:
            if USE_HIERARCHICAL_PPO:
                ppo_model = HierarchicalPPODrafter.load_model(HIERARCHICAL_PPO_MODEL_PATH)
            else:
                ppo_model = PPODrafter.load_model(PPO_MODEL_PATH)
            logger.info("Successfully loaded PPO model")
        except Exception as e:
            logger.error(f"Error loading PPO model: {e}")
    
    # Step 8: Run draft simulations if requested
    draft_results = None
    if run_draft_sims and projections:
        logger.info("Step 8: Running draft simulations...")
        
        # Create output directory
        draft_results_dir = os.path.join(OUTPUT_DIR, 'draft_simulations')
        os.makedirs(draft_results_dir, exist_ok=True)
        
        # Run draft simulations
        draft_results = run_draft_simulations(
            projections=projections,
            league_info=league_data,
            roster_limits=roster_limits,
            scoring_settings=scoring_settings,
            ppo_model=ppo_model,  # Pass PPO model if available
            results_dir=draft_results_dir
        )
        
        logger.info("Draft simulations complete")
    elif run_draft_sims:
        logger.warning("Cannot run draft simulations without projections")
    
    # Step 9: Run season simulations if requested
    season_results = None
    if run_season_sims and draft_results:
        logger.info("Step 9: Running season simulations...")
        
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
    
    # Display PPO training status if available
    if ppo_model:
        print("\nPPO Model:")
        print(f"  Status: {'Trained' if train_ppo else 'Loaded from existing model'}")
        print(f"  Model path: {PPO_MODEL_PATH}")
    
    print("\nDraft Assistant Script:")
    print(f"  Path: {os.path.join(MODELS_DIR, 'draft_assistant.py')}")
    print(f"  Usage: python {os.path.join(MODELS_DIR, 'draft_assistant.py')} [draft_position]")
    
    print("\nAnalysis completed successfully!")
    print(f"Output data and visualizations saved to the '{DATA_DIR}' directory")
    
    return {
        "league_data": league_data,
        "nfl_data": nfl_data if not use_cached_raw else None,
        "processed_data": processed_data,
        "feature_sets": feature_sets,
        "dropped_tiers": dropped_tiers,
        "projections": projections,
        "ppo_model": ppo_model,
        "draft_results": draft_results,
        "season_results": season_results
    }
    

if __name__ == "__main__":
    main()