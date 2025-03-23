#!/usr/bin/env python3
"""
Fantasy Football Draft Optimizer

This script orchestrates the complete data analysis pipeline:
1. Loads data from NFL data sources
2. Extracts league settings from ESPN API
3. Engineers features for player performance prediction
4. Performs clustering and tier-based analysis
5. Creates comprehensive visualizations
"""

import os
import json
import pandas as pd
import logging
from datetime import datetime
import sys
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
CREATE_VISUALIZATIONS = True  # Whether to generate visualization plots
DEBUG_MODE = False  # Enable for verbose logging

# Processing Options
CLUSTER_COUNT = 5  # Number of player tiers/clusters to create
DROP_BOTTOM_TIERS = 1  # Number of bottom tiers to drop

# File Paths
CONFIG_PATH = 'configs/league_settings.json'  # Path to optional config file
DATA_DIR = 'data'  # Base directory for all data
OUTPUT_DIR = os.path.join(DATA_DIR, 'outputs')  # Directory for visualizations

#######################################################
# END OF CONFIGURATION
#######################################################

# Set up logging
log_level = logging.DEBUG if DEBUG_MODE else logging.INFO
logging.basicConfig(
    level=log_level,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fantasy_football.log"),
        logging.StreamHandler(sys.stdout)
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

def load_config(config_path):
    """Load configuration from file"""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.warning(f"Error loading config file: {e}")
        return {}

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
    
    if league_id is None:
        logger.error("League ID is required. Set LEAGUE_ID at the top of the script.")
        return
    
    # Create output directories
    raw_dir = os.path.join(DATA_DIR, 'raw')
    processed_dir = os.path.join(DATA_DIR, 'processed')
    
    for directory in [DATA_DIR, raw_dir, processed_dir, OUTPUT_DIR]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Load league data from ESPN
    logger.info(f"Step 1: Loading ESPN league data for league {league_id}, year {year}...")
    league_data = load_espn_league_data(league_id, year, espn_s2, swid)
    
    # Save league data
    with open(os.path.join(raw_dir, 'league_data.json'), 'w') as f:
        # Convert DataFrame to list for JSON serialization
        if 'teams' in league_data and isinstance(league_data['teams'], pd.DataFrame):
            league_data_json = league_data.copy()
            league_data_json['teams'] = league_data_json['teams'].to_dict(orient='records')
            json.dump(league_data_json, f, indent=2)
        else:
            json.dump(league_data, f, indent=2)
    
    # Step 2: Load NFL data from nfl_data_py
    logger.info(f"Step 2: Loading NFL data for years {start_year}-{year}...")
    years = list(range(start_year, year + 1))
    
    nfl_data = load_nfl_data(
        years=years,
        include_ngs=INCLUDE_NGS,
        ngs_min_year=2016,
        use_threads=True
    )
    
    # Save raw data
    for key, df in nfl_data.items():
        if not df.empty and isinstance(df, pd.DataFrame):
            output_path = os.path.join(raw_dir, f"{key}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved raw {key} data to {output_path}")
    
    # Step 3: Process player data
    logger.info("Step 3: Processing player data...")
    processed_data = process_player_data(nfl_data)
    
    # Save processed data
    for key, df in processed_data.items():
        if not df.empty and isinstance(df, pd.DataFrame):
            output_path = os.path.join(processed_dir, f"{key}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved processed {key} data to {output_path}")
    
    # Step 4: Feature engineering
    logger.info("Step 4: Performing feature engineering...")
    feature_eng = FantasyFeatureEngineering(processed_data, target_year=year)
    
    # Create position-specific features
    feature_eng.create_position_features()
    
    # Prepare prediction features
    feature_eng.prepare_prediction_features()
    
    # Perform clustering
    feature_eng.cluster_players(n_clusters=CLUSTER_COUNT, drop_tiers=DROP_BOTTOM_TIERS)
    
    # Finalize features
    feature_eng.finalize_features(apply_filtering=True)
    
    # Get processed feature sets
    feature_sets = feature_eng.get_feature_sets()
    
    # Save feature sets
    for key, df in feature_sets.items():
        if not df.empty and isinstance(df, pd.DataFrame):
            output_path = os.path.join(processed_dir, f"features_{key}.csv")
            df.to_csv(output_path, index=False)
            logger.info(f"Saved feature set {key} to {output_path}")
    
    # Get cluster models
    cluster_models = feature_eng.get_cluster_models()
    
    # Step 5: Create visualizations
    if CREATE_VISUALIZATIONS:
        logger.info("Step 5: Creating visualizations...")
        visualizer = FantasyDataVisualizer(
            data_dict=processed_data,
            feature_sets=feature_sets,
            output_dir=OUTPUT_DIR
        )
        
        # Run all visualizations
        visualizer.run_all_visualizations(league_data)
    else:
        logger.info("Skipping visualizations as specified in configuration")
    
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
    
    print("\nAnalysis completed successfully!")
    print(f"Output data and visualizations saved to the '{DATA_DIR}' directory")
    print("\nNext steps:")
    print("1. Review visualizations in data/outputs/")
    print("2. Run the draft simulation module (coming soon)")
    print("3. Use the draft helper GUI during your draft (coming soon)")
    
    return {
        "league_data": league_data,
        "processed_data": processed_data,
        "feature_sets": feature_sets,
        "cluster_models": cluster_models
    }

if __name__ == "__main__":
    main()