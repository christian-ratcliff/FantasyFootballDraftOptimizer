#!/usr/bin/env python3
"""
Fantasy Football Draft Optimizer - Main Script

This script orchestrates the complete data analysis pipeline by calling functions
from the various modules in the project.
"""

import os
import json
import pandas as pd
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


# Import from project modules
from src.data.loader import (
    load_espn_league_data,
    load_nfl_historical_data,
    load_nfl_current_season_data,
    clean_and_merge_all_data
)
from src.analysis.analyzer import FantasyFootballAnalyzer
from src.analysis.visualizer import FantasyDataVisualizer
from src.features.engineering import FeatureEngineering
from src.features.engineering_extensions import FeatureEngineeringExtensions

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("fantasy_football.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("fantasy_football")

def main():
    """Main function to run the fantasy football analysis pipeline"""
    # Load configuration
    config_path = "configs/league_settings.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        league_id = config['league_id']
        year = config['year']
        espn_s2 = config.get('espn_s2')
        swid = config.get('swid')
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Please run generate_config.py")
        
    
    # Create output directories
    data_dir = 'data'
    raw_dir = os.path.join(data_dir, 'raw')
    processed_dir = os.path.join(data_dir, 'processed')
    output_dir = os.path.join(data_dir, 'outputs')
    feature_dir = os.path.join(output_dir, 'feature_analysis')
    
    for directory in [data_dir, raw_dir, processed_dir, output_dir, feature_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Step 1: Load league data from ESPN
    logger.info("Step 1: Loading ESPN league data...")
    league_data = load_espn_league_data(league_id, year, espn_s2, swid)
    
    # Save league data
    with open(os.path.join(raw_dir, 'league_data.json'), 'w') as f:
        # Convert DataFrame to dictionary for JSON serialization
        league_data_json = league_data.copy()
        if 'teams' in league_data_json and isinstance(league_data_json['teams'], pd.DataFrame):
            league_data_json['teams'] = league_data_json['teams'].to_dict(orient='records')
        json.dump(league_data_json, f, indent=2)
    
    # Step 2: Load historical NFL data
    logger.info("Step 2: Loading historical NFL data...")
    historical_years = list(range(year-7, year))
    historical_data = load_nfl_historical_data(historical_years)
    
    # Save key historical data
    for key, df in historical_data.items():
        if not df.empty and isinstance(df, pd.DataFrame):
            df.to_csv(os.path.join(raw_dir, f"historical_{key}.csv"), index=False)
    
    # Step 3: Load current season NFL data
    logger.info("Step 3: Loading current season NFL data...")
    current_data = load_nfl_current_season_data(year)
    
    # Save key current data
    for key, df in current_data.items():
        if not df.empty and isinstance(df, pd.DataFrame):
            df.to_csv(os.path.join(raw_dir, f"current_{key}.csv"), index=False)
    
    # Step 4: Clean and merge data
    logger.info("Step 4: Cleaning and merging all data...")
    merged_data = clean_and_merge_all_data(historical_data, current_data)
    
    # Save merged data
    for key, df in merged_data.items():
        if not df.empty:
            df.to_csv(os.path.join(processed_dir, f"{key}.csv"), index=False)
    
    # Step 5: Initialize analyzer
    logger.info("Step 5: Initializing fantasy football analyzer...")
    analyzer = FantasyFootballAnalyzer(
        league_id=league_id,
        year=year,
        espn_s2=espn_s2,
        swid=swid
    )
    
    # Load data into analyzer
    if league_data:
        if 'league_info' in league_data:
            analyzer.settings = league_data['league_info']
        analyzer.scoring_format = league_data['scoring_settings']
        analyzer.roster_positions = league_data['roster_settings']
    
    if 'all_seasonal' in merged_data:
        analyzer.historical_data = {
            'seasonal': merged_data['all_seasonal'],
            'player_ids': historical_data.get('player_ids', pd.DataFrame()),
            'weekly': historical_data.get('weekly', pd.DataFrame())
        }
    
    # Step 6: Preprocess data
    logger.info("Step 6: Preprocessing data...")
    preprocessed_data = analyzer.preprocess_data()
    
    # Save preprocessed data
    for key, df in preprocessed_data.items():
        df.to_csv(os.path.join(processed_dir, f"preprocessed_{key}.csv"), index=False)
    
    # Step 7: Feature engineering
    logger.info("Step 7: Applying basic feature engineering...")
    feature_eng = FeatureEngineering(preprocessed_data)
    
    # Apply feature engineering steps
    feature_eng.create_positional_features() \
              .create_trend_features() \
              .create_age_features() \
              .handle_missing_values() \
              .normalize_features()
    
    # Step 7b: Apply enhanced feature engineering with clustering (using 5 clusters)
    logger.info("Step 7b: Applying enhanced feature engineering with clustering...")
    
    # Apply clustering before other visualizations to enable filtering
    logger.info("Creating player clusters...")
    positions = ['QB', 'RB', 'WR', 'TE']
    for position in positions:
        logger.info(f"Creating clusters for {position}...")
        feature_eng.create_cluster_features(position=position, n_clusters=5)
    
    # Apply advanced feature engineering extensions
    logger.info("Applying advanced feature engineering...")
    feature_eng_ext = FeatureEngineeringExtensions(feature_eng)
    
    # Add NGS data features if available
    ngs_data = {
        'ngs_passing': merged_data.get('ngs_passing', pd.DataFrame()),
        'ngs_rushing': merged_data.get('ngs_rushing', pd.DataFrame()),
        'ngs_receiving': merged_data.get('ngs_receiving', pd.DataFrame())
    }
    
    # Create advanced metrics and interactions
    feature_eng_ext.create_advanced_ngs_features(ngs_data) \
                  .create_fantasy_scoring_features(league_data['scoring_settings']) \
                  .create_interaction_features(top_n=15) \
                  .analyze_feature_importance() \
                  .visualize_feature_importance(output_dir=feature_dir) \
                  .analyze_feature_correlations() \
                  .visualize_feature_correlations(output_dir=feature_dir) \
                  .remove_multicollinear_features() \
                  .select_best_features()
    
    # Get the fully processed data
    processed_data = feature_eng_ext.get_processed_data()
    
    # Save processed data
    for key, df in processed_data.items():
        df.to_csv(os.path.join(processed_dir, f"engineered_{key}.csv"), index=False)
    
    feature_eng_ext.save_feature_engineered_data(output_dir=processed_dir)
    
    # Step 8: Create visualizations
    logger.info("Step 8: Creating visualizations...")
    
    # Create enhanced visualizer with updated player age filtering and top cluster filtering
    visualizer = FantasyDataVisualizer(
        merged_data, 
        output_dir=output_dir, 
        current_year=year
    )
    
    # Run all visualizations - now with improved clustering and filtering
    visualizer.run_all_visualizations()
    
    # Save all data for later use
    analyzer.save_data(os.path.join(processed_dir, 'fantasy_football_data.pkl'))
    
    logger.info("Analysis pipeline completed successfully!")
    
    return {
        "league_data": league_data,
        "merged_data": merged_data,
        "preprocessed_data": preprocessed_data,
        "processed_data": processed_data
    }

if __name__ == "__main__":
    start_time = datetime.now()
    logger.info(f"Starting fantasy football analysis at {start_time}")
    
    results = main()
    
    end_time = datetime.now()
    duration = end_time - start_time
    logger.info(f"Analysis completed in {duration}")
    
    print(f"\nAnalysis completed successfully in {duration}!")
    print(f"Output data and visualizations saved to the 'data' directory.")