"""
Player projection pipeline that handles the end-to-end process
of generating fantasy football player projections.
"""

import logging
import os
import pickle
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Import project modules
from src.data.loader import (
    load_espn_league_data,
    load_nfl_data,
    process_player_data
)
from src.features.engineering import FantasyFeatureEngineering
from src.models.projections import PlayerProjectionModel

class ProjectionPipeline:
    """
    Orchestrates the end-to-end pipeline for player projections
    """
    
    def __init__(self, config):
        """Initialize pipeline with configuration"""
        self.config = config
        self.logger = logging.getLogger("fantasy_football")
        self.data_dirs = None
        self.nfl_data = {}
        self.processed_data = {}
        self.feature_sets = {}
        self.dropped_tiers = None
        self.projections = {}
        self.league_data = {}
        self.roster_limits = {}
        self.scoring_settings = {}
        
    def run(self):
        """Execute the full projection pipeline"""
        start_time = datetime.now()
        self.logger.info(f"Starting projection pipeline at {start_time}")
        
        # Setup directories 
        self.data_dirs = self.create_output_dirs()
        
        # Load league settings
        self.league_data, self.roster_limits, self.scoring_settings = self.load_league_settings()
        
        # Load and process data
        self.load_data()
        
        # Engineer features
        self.engineer_features()
        
       # Check if we should use existing models or train new ones
        use_existing_models = self.config.get('models', {}).get('use_existing', False)
        
        if use_existing_models:
            # Load existing models instead of training
            self.load_existing_models()
        else:
            # Train models and generate projections
            self.train_models_and_project()
        
        # Create visualizations if enabled
        if self.config.get('visualizations', {}).get('enabled', True):
            self.create_visualizations()
        
        # Evaluate projections if enabled
        if self.config.get('evaluation', {}).get('enabled', True):
            self.evaluate_projections()
        
        # Log completion
        duration = datetime.now() - start_time
        self.logger.info(f"Pipeline completed in {duration}")
        
        return self.projections
    
    def create_output_dirs(self):
        """Create organized output directories based on config"""
        # Get directory paths from config
        data_dir = self.config['paths']['data_dir']
        output_dir = self.config['paths']['output_dir']
        models_dir = self.config['paths']['models_dir']
        
        # Create base directories
        dirs = {
            'raw': os.path.join(data_dir, 'raw'),
            'processed': os.path.join(data_dir, 'processed'),
            'outputs': output_dir,
            'models': models_dir,
        }
        
        # Create position-specific directories
        positions = ['qb', 'rb', 'wr', 'te', 'overall']
        
        # Create all directories
        for dir_name, dir_path in dirs.items():
            os.makedirs(dir_path, exist_ok=True)
        
        # Create organized visualization directories
        for position in positions:
            position_dir = os.path.join(output_dir, position)
            os.makedirs(position_dir, exist_ok=True)
        
        self.logger.info(f"Created output directories in {data_dir}")
        return dirs
        
    def load_league_settings(self):
        """Load league settings from ESPN API or config file"""
        # Get league info from config
        league_id = self.config['league']['id']
        year = self.config['league']['year']
        espn_s2 = self.config['league']['espn_s2']
        swid = self.config['league']['swid']
        
        # First, try to generate/update the config file
        try:
            from generate_config import generate_config
            config_generated = generate_config(league_id, year, espn_s2, swid)
            self.logger.info("Config file successfully generated/updated")
        except Exception as e:
            self.logger.warning(f"Failed to generate config file: {e}")
        
        # Try to load league data from ESPN API
        try:
            league_data = load_espn_league_data(league_id, year, espn_s2, swid)
            
            if league_data:
                # Extract roster limits and scoring settings from league data
                roster_limits = league_data.get('roster_settings', {})
                scoring_settings = league_data.get('scoring_settings', {})
                
                self.logger.info(f"Successfully loaded league settings from ESPN API")
                return league_data, roster_limits, scoring_settings
        except Exception as e:
            self.logger.warning(f"Error loading from ESPN API: {e}")
        
        # If API failed, try loading from config file
        try:
            config_path = self.config['paths']['config_path']
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    league_config = json.load(f)
                
                roster_limits = league_config.get('roster_settings', {})
                scoring_settings = league_config.get('scoring_settings', {})
                
                self.logger.info(f"Successfully loaded league settings from config file")
                return league_config, roster_limits, scoring_settings
            else:
                self.logger.warning(f"Config file not found: {config_path}")
        except Exception as e:
            self.logger.warning(f"Error loading config file: {e}")
        
        self.logger.error("Could not load league settings from API or config file.")
        return {}, {}, {}
    
    def load_data(self):
        """Load raw or cached data based on config settings"""
        # Get configuration parameters for data loading
        year = self.config['league']['year']
        start_year = self.config['data']['start_year']
        include_ngs = self.config['data']['include_ngs']
        use_cached_raw = self.config['caching']['use_cached_raw_data']
        use_cached_processed = self.config['caching']['use_cached_processed_data']
        
        # Generate year range
        years = list(range(start_year, year + 1))
        
        # Step 1: Load NFL data (either from cache or fresh download)
        if use_cached_raw:
            self.logger.info("Using cached raw data...")
            self.nfl_data = self._load_cached_raw_data()
            if not self.nfl_data:
                self.logger.warning("No cached raw data found or error loading. Falling back to fresh download.")
                use_cached_raw = False
        
        if not use_cached_raw:
            self.logger.info(f"Loading NFL data for years {start_year}-{year}...")
            self.nfl_data = load_nfl_data(
                years=years,
                include_ngs=include_ngs,
                ngs_min_year=start_year,  # Using 2016 as the first year with reliable NGS data
                use_threads=True
            )
            
            # Save raw data
            for key, df in self.nfl_data.items():
                if not df.empty and isinstance(df, pd.DataFrame):
                    output_path = os.path.join(self.data_dirs['raw'], f"{key}.csv")
                    df.to_csv(output_path, index=False)
                    self.logger.info(f"Saved raw {key} data to {output_path}")
        
        # Step 2: Process player data (either from cache or process raw data)
        use_cached_features = self.config['caching']['use_cached_feature_sets']
        if use_cached_processed and use_cached_features:
            self.logger.info("Using cached processed data...")
            self.processed_data = self._load_cached_processed_data()
            
        if use_cached_processed and not use_cached_features:
            self.logger.info("Using cached processed data...")
            self.processed_data = self._load_cached_processed_data()
            if not self.processed_data:
                self.logger.warning("No cached processed data found or error loading. Will process raw data.")
                use_cached_processed = False
        
        if not use_cached_processed and not use_cached_features:
            self.logger.info("Processing player data...")
            self.processed_data = process_player_data(self.nfl_data)
            
            # Save processed data
            for key, df in self.processed_data.items():
                if not df.empty and isinstance(df, pd.DataFrame):
                    output_path = os.path.join(self.data_dirs['processed'], f"{key}.csv")
                    df.to_csv(output_path, index=False)
                    self.logger.info(f"Saved processed {key} data to {output_path}")
                    
        return self.nfl_data, self.processed_data
        
    def _load_cached_raw_data(self):
        """Load previously downloaded raw data files"""
        raw_data_dir = self.data_dirs['raw']
        self.logger.info("Loading raw data from cache...")
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
                    self.logger.info(f"Loaded {base_name} from {file_path} ({len(df)} rows)")
                except Exception as e:
                    self.logger.error(f"Error loading {file_path}: {e}")
            else:
                self.logger.warning(f"File not found: {file_path}")
        
        return raw_data
        
    def _load_cached_processed_data(self):
        """Load previously processed data files"""
        processed_data_dir = self.data_dirs['processed']
        self.logger.info("Loading processed data from cache...")
        processed_data = {}
        
        # List all CSV files in the directory
        for file_name in os.listdir(processed_data_dir):
            if file_name.endswith(".csv") and not file_name.startswith("features_"):
                file_path = os.path.join(processed_data_dir, file_name)
                base_name = file_name.split('.')[0]
                
                try:
                    df = pd.read_csv(file_path)
                    processed_data[base_name] = df
                    self.logger.info(f"Loaded {base_name} from {file_path} ({len(df)} rows)")
                except Exception as e:
                    self.logger.error(f"Error loading {file_path}: {e}")
        
        return processed_data
        
    def _load_cached_feature_sets(self):
        """Load previously engineered feature sets"""
        processed_data_dir = self.data_dirs['processed']
        self.logger.info("Loading feature sets from cache...")
        feature_sets = {}
        
        # List all feature set files (those starting with "features_")
        for file_name in os.listdir(processed_data_dir):
            if file_name.startswith("features_") and file_name.endswith(".csv"):
                file_path = os.path.join(processed_data_dir, file_name)
                key = file_name[9:-4]  # Remove "features_" prefix and ".csv" suffix
                
                try:
                    df = pd.read_csv(file_path)
                    feature_sets[key] = df
                    self.logger.info(f"Loaded feature set {key} from {file_path} ({len(df)} rows)")
                except Exception as e:
                    self.logger.error(f"Error loading {file_path}: {e}")
        
        # Load dropped tiers information if available
        dropped_tiers_path = os.path.join(processed_data_dir, 'dropped_tiers.json')
        dropped_tiers = None
        if os.path.exists(dropped_tiers_path):
            try:
                with open(dropped_tiers_path, 'r') as f:
                    dropped_tiers = json.load(f)
                    self.logger.info(f"Loaded dropped tiers information from {dropped_tiers_path}")
            except Exception as e:
                self.logger.error(f"Error loading dropped tiers: {e}")
        
        return feature_sets, dropped_tiers
    
    def engineer_features(self):
        """Run feature engineering process using config parameters"""
        # Check if we should use cached feature sets
        use_cached_features = self.config['caching']['use_cached_feature_sets']
        year = self.config['league']['year']
        
        if use_cached_features:
            self.logger.info("Using cached feature sets...")
            self.feature_sets, self.dropped_tiers = self._load_cached_feature_sets()
            if not self.feature_sets:
                self.logger.warning("No cached feature sets found or error loading. Will run feature engineering.")
                use_cached_features = False
        
        if not use_cached_features:
            # Ensure we have processed data
            if not self.processed_data:
                self.logger.info("Loading processed data for feature engineering...")
                self.processed_data = self._load_cached_processed_data()
            
            if not self.processed_data:
                self.logger.error("No processed data available for feature engineering.")
                return
                
            self.logger.info("Performing feature engineering...")
            feature_eng = FantasyFeatureEngineering(self.processed_data, target_year=year)
            
            # Create position-specific features
            feature_eng.create_position_features()
            
            
            
            # Prepare prediction features
            feature_eng.prepare_prediction_features()
            
            # Get clustering parameters from config
            cluster_count = self.config['clustering']['cluster_count']
            drop_bottom_tiers = self.config['clustering']['drop_bottom_tiers']
            
            # Perform clustering with position-specific tier drops
            position_tier_drops = {
                'qb': drop_bottom_tiers,
                'rb': drop_bottom_tiers,
                'wr': drop_bottom_tiers,
                'te': drop_bottom_tiers
            }
            feature_eng.cluster_players(n_clusters=cluster_count, position_tier_drops=position_tier_drops)
            
            # Finalize features with filtering applied
            use_filtered = self.config['clustering']['use_filtered']
            feature_eng.finalize_features(apply_filtering=use_filtered)
            
            # Get processed feature sets
            self.feature_sets = feature_eng.get_feature_sets()
            
            # Save feature sets
            for key, df in self.feature_sets.items():
                if not df.empty and isinstance(df, pd.DataFrame):
                    output_path = os.path.join(self.data_dirs['processed'], f"features_{key}.csv")
                    df.to_csv(output_path, index=False)
                    self.logger.info(f"Saved feature set {key} to {output_path}")
            
            # Get cluster models and dropped tiers
            self.dropped_tiers = feature_eng.dropped_tiers
            
            # Save the dropped tier information for future reference
            with open(os.path.join(self.data_dirs['processed'], 'dropped_tiers.json'), 'w') as f:
                json.dump({pos: [int(x) for x in tiers] for pos, tiers in self.dropped_tiers.items()}, f)
        
        return self.feature_sets, self.dropped_tiers
        
    def train_models_and_project(self):
        """Train projection models and generate projections using config settings"""
        # Get projection parameters from config
        models_dir = self.config['paths']['models_dir']
        use_filtered = self.config['clustering']['use_filtered']
        use_do_not_draft = self.config['projections']['use_do_not_draft']
        use_hierarchical = self.config['projections']['use_hierarchical']
        model_type = self.config['projections']['model_type']
        optimize_hyperparams = self.config['projections']['optimize_hyperparams']
        feature_selection_method = self.config['projections'].get('feature_selection_method', 'importance')
        prioritize_ngs = self.config['projections'].get('prioritize_ngs', True)
        
        # Initialize the projection model
        self.logger.info(f"Initializing projection model with {model_type} and {feature_selection_method} feature selection...")
        if prioritize_ngs:
            self.logger.info("Prioritizing NGS metrics in feature selection")
            
        projection_model = PlayerProjectionModel(self.feature_sets, output_dir=models_dir, use_filtered=use_filtered)
        
        # Train standard models for all positions
        all_metrics = projection_model.train_all_positions(
            model_type=model_type,
            optimize_hyperparams=optimize_hyperparams
        )
        
        # Log validation metrics
        self.logger.info("Model validation metrics:")
        for position, metrics in all_metrics.items():
            self.logger.info(f"  {position.upper()}: RMSE={metrics['rmse']:.2f}, MAE={metrics['mae']:.2f}, R²={metrics['r2']:.2f}")
        
        # Train hierarchical models if enabled
        if use_hierarchical:
            self.logger.info("Training hierarchical component models...")
            projection_model.train_all_hierarchical_models(model_type=model_type)
        
        # Generate projections
        projection_year = self.config['projections']['projection_year']
        self.logger.info(f"Generating projections for {projection_year}...")
        self.projections = projection_model.generate_full_projections(
            use_do_not_draft=use_do_not_draft,
            use_hierarchical=use_hierarchical
        )
        
        # Cache projections for future use
        try:
            with open(os.path.join(models_dir, 'player_projections.pkl'), 'wb') as f:
                pickle.dump(self.projections, f)
            self.logger.info("Cached player projections")
        except Exception as e:
            self.logger.error(f"Error caching projections: {e}")
            
        return self.projections
        
    def evaluate_projections(self):
        """Evaluate projections against actual results if available"""
        # Get projection year from config
        projection_year = self.config['projections']['projection_year']
        start_year = self.config['data']['start_year']
        year = self.config['league']['year']
        
        # Create a range of years that we have data for
        years = list(range(start_year, year + 1))
        
        # Only evaluate if projection year is in our data range
        if projection_year not in years:
            self.logger.info(f"Evaluation not possible: projection year {projection_year} not in data range {start_year}-{year}")
            return None
            
        # Get seasonal data
        seasonal = self.processed_data.get('seasonal', pd.DataFrame())
        if seasonal.empty:
            self.logger.error("No seasonal data for evaluation")
            return {}
        
        # Check for the actual projection year data
        actual_data = seasonal[seasonal['season'] == projection_year].copy()
        if actual_data.empty:
            self.logger.error(f"No actual data for {projection_year}")
            return {}
        
        # Print some information about available players to help with debugging
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_count = len(actual_data[actual_data['position'] == position])
            self.logger.info(f"Found {pos_count} {position} players in actual data for {projection_year}")
        
        # Evaluate each position
        results = {}
        for position in ['qb', 'rb', 'wr', 'te']:
            if position not in self.projections:
                self.logger.warning(f"No projections for {position}")
                continue
            
            # Get position data
            pos_projections = self.projections[position]
            pos_actual = actual_data[actual_data['position'] == position.upper()].copy()
            
            # Skip if missing required columns
            if 'name' not in pos_projections.columns or 'name' not in pos_actual.columns:
                self.logger.warning(f"Missing name column for {position}")
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
            
            # Handle no matches scenario
            if merged.empty:
                self.logger.warning(f"No matched players for {position}")
                continue
            
            self.logger.info(f"Successfully matched {len(merged)} {position} players for evaluation")
            
            # Use the actual name for final output 
            if 'name_proj' in merged.columns and 'name_actual' in merged.columns:
                merged['name'] = merged['name_actual']
            
            # Filter out players with zero or near-zero projections
            min_projection_threshold = 0.5
            filtered_merged = merged[merged['projected_points'] >= min_projection_threshold].copy()
            
            # Calculate metrics
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
            self.logger.info(f"{position} projection evaluation:")
            self.logger.info(f"  Players: {metrics['n_players']}")
            self.logger.info(f"  MAE: {metrics['mae']:.2f}")
            self.logger.info(f"  RMSE: {metrics['rmse']:.2f}")
            self.logger.info(f"  Mean Error: {metrics['mean_error']:.2f}")
            self.logger.info(f"  Correlation: {metrics['correlation']:.3f}")
            self.logger.info(f"  % Within Range: {metrics['within_range']:.1f}%")
            
            # Save player-level evaluation to CSV
            merged_path = os.path.join(self.data_dirs['outputs'], f'{position}_projection_evaluation.csv')
            filtered_merged[['name', 'projected_points', 'fantasy_points_per_game', 'projection_low', 'projection_high']].to_csv(merged_path, index=False)
            
            results[position] = metrics
            
            # Create projection accuracy visualizations
            for position in ['qb', 'rb', 'wr', 'te']:
                if position not in results:
                    continue
                    
                metrics = results[position]
                if 'players' not in metrics:
                    continue
                    
                players_df = metrics['players']
                
                # Verify the required columns exist
                required_cols = ['projected_points', 'fantasy_points_per_game', 'name']
                if not all(col in players_df.columns for col in required_cols):
                    self.logger.warning(f"Missing required columns for {position} visualization")
                    continue
                
                # Create directory
                viz_dir = os.path.join(self.data_dirs['outputs'], position, 'projections')
                os.makedirs(viz_dir, exist_ok=True)
                
                # Create accuracy scatter plot
                plt.figure(figsize=(10, 8))
                
                # Calculate projection error directly (add as a new column)
                players_df['projection_error'] = players_df['projected_points'] - players_df['fantasy_points_per_game']
                
                # Plot the points
                plt.scatter(
                    players_df['projected_points'], 
                    players_df['fantasy_points_per_game'],
                    alpha=0.7
                )
                
                # Add perfect prediction line
                max_val = max(
                    players_df['projected_points'].max(),
                    players_df['fantasy_points_per_game'].max()
                ) * 1.1
                plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7)
                
                # Add regression line
                sns.regplot(
                    x='projected_points',
                    y='fantasy_points_per_game',
                    data=players_df,
                    scatter=False,
                    line_kws={'color': 'blue'}
                )
                
                # Add metrics annotation
                corr = metrics['correlation']
                mae = metrics['mae']
                plt.annotate(
                    f"Correlation: {corr:.3f}\nMAE: {mae:.2f}",
                    xy=(0.05, 0.95),
                    xycoords='axes fraction',
                    fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8)
                )
                
                # Add labels for notable players (using top 3 and worst 3 projections)
                try:
                    # Top performers
                    for _, player in players_df.nlargest(3, 'fantasy_points_per_game').iterrows():
                        plt.annotate(
                            player['name'],
                            (player['projected_points'], player['fantasy_points_per_game']),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightgreen', alpha=0.7)
                        )
                        
                    # Biggest underperformers (using the column we calculated)
                    for _, player in players_df.nlargest(3, 'projection_error').iterrows():
                        plt.annotate(
                            player['name'],
                            (player['projected_points'], player['fantasy_points_per_game']),
                            xytext=(5, -10),
                            textcoords='offset points',
                            fontsize=8,
                            bbox=dict(boxstyle="round,pad=0.2", facecolor='lightcoral', alpha=0.7)
                        )
                except Exception as e:
                    self.logger.error(f"Error adding player annotations: {e}")
                target_year = self.config.get('projections', {}).get('projection_year', 2024)
                # Add title and labels
                plt.title(f'{position.upper()} Projection Accuracy - {target_year}')
                plt.xlabel('Projected Fantasy Points per Game')
                plt.ylabel('Actual Fantasy Points per Game')
                plt.grid(True, alpha=0.3)
                
                # Save figure
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'projection_accuracy.png'), dpi=300)
                plt.close()
        
        return results
        
    def get_summary(self):
        """Create a summary of the projections for display"""
        summary = {
            'league': self.league_data.get('league_info', {}).get('name', 'Unknown'),
            'teams': self.league_data.get('league_info', {}).get('team_count', 'Unknown'),
            'projection_year': self.config['projections']['projection_year'],
            'used_hierarchical': self.config['projections']['use_hierarchical'],
            'used_ngs': self.config['data']['include_ngs'],
            'top_players': {}
        }
        
        # Add top players by position
        for position in ['qb', 'rb', 'wr', 'te']:
            if position in self.projections and 'name' in self.projections[position].columns:
                top_players = self.projections[position].nlargest(5, 'projected_points')
                summary['top_players'][position] = [
                    {"name": player['name'], "points": player['projected_points']} 
                    for _, player in top_players.iterrows()
                ]
                
        return summary
    
    def create_visualizations(self):
        """Generate visualizations for models and data"""
        self.logger.info("Creating visualizations...")
        
        # Create MLExplorer instance
        from src.analysis.ml_explorer import MLExplorer
        explorer = MLExplorer(
            data_dict=self.processed_data,
            feature_sets=self.feature_sets,
            output_dir=self.data_dirs['outputs']
        )
        
        # Generate ML-focused visualizations
        explorer.create_correlation_matrices()
        explorer.create_feature_importance_plots(target='fantasy_points_per_game')
        explorer.create_pair_plots()
        explorer.create_cluster_visualizations()
        explorer.save_analysis_results()
        
        # Create basic visualizations
        from src.analysis.visualizer import FantasyDataVisualizer
        visualizer = FantasyDataVisualizer(
            data_dict=self.processed_data,
            feature_sets=self.feature_sets,
            output_dir=self.data_dirs['outputs']
        )
        
        # Create league setting visualizations if available
        if self.league_data:
            visualizer.explore_league_settings(self.league_data)
        
        # Create basic data distributions
        visualizer.explore_data_distributions()
        
        # Create performance trends
        visualizer.explore_performance_trends()
        
        # Generate projection visualizations if projections exist
        if hasattr(self, 'projections') and self.projections:
            self._create_projection_visualizations()
        
        self.logger.info("Visualization generation complete")
        
        return self
    
    
    def load_existing_models(self):
        """Load existing projection models"""
        self.logger.info("Loading existing projection models")
        
        # Use models_dir from config or default to 'data/models'
        models_dir = self.config.get('models', {}).get('models_dir', 'data/models')
        
        # Initialize container for projections
        self.projections = {}
        
        # Load models for each position
        for position in ['qb', 'rb', 'wr', 'te']:
            model_path = os.path.join(models_dir, f"{position}_model.joblib")
            
            if os.path.exists(model_path):
                try:
                    self.logger.info(f"Loading {position} model from {model_path}")
                    
                    # Load model
                    from src.models.projections import PlayerProjectionModel
                    projection_model = PlayerProjectionModel.load_model(model_path, self.feature_sets)
                    
                    # Generate projections for this position
                    projection_key = f"{position}_projection"
                    if projection_key in self.feature_sets and not self.feature_sets[projection_key].empty:
                        # Use hierarchical models if available
                        use_hierarchical = self.config['projections'].get('use_hierarchical', True)
                        
                        if use_hierarchical and hasattr(projection_model, 'hierarchical_models') and position in projection_model.hierarchical_models:
                            self.logger.info(f"Using hierarchical projections for {position}")
                            projection_data = projection_model._project_with_hierarchical_model(position, self.feature_sets[projection_key])
                        else:
                            self.logger.info(f"Using direct projections for {position}")
                            use_do_not_draft = self.config['projections'].get('use_do_not_draft', True)
                            projection_data = projection_model.project_players(position, self.feature_sets[projection_key], use_do_not_draft)
                        
                        self.projections[position] = projection_data
                        self.logger.info(f"Generated projections for {len(projection_data)} {position} players")
                    else:
                        self.logger.warning(f"No projection data available for {position}")
                except Exception as e:
                    self.logger.error(f"Error loading {position} model: {e}")
            else:
                self.logger.warning(f"No model file found at {model_path}")
        
        return self.projections
    
    
    def _create_projection_visualizations(self):
        """Create visualizations specific to projections"""
        self.logger.info("Creating projection visualizations...")
        
        # Create output directory
        proj_viz_dir = os.path.join(self.data_dirs['outputs'], 'projections')
        os.makedirs(proj_viz_dir, exist_ok=True)
        
        # Generate position-specific projection visualizations
        for position in ['qb', 'rb', 'wr', 'te']:
            if position in self.projections and not self.projections[position].empty:
                pos_data = self.projections[position]
                
                # Create output directory for this position
                pos_dir = os.path.join(proj_viz_dir, position)
                os.makedirs(pos_dir, exist_ok=True)
                
                # Only continue if we have projected points
                if 'projected_points' not in pos_data.columns or 'name' not in pos_data.columns:
                    continue
                
                # Sort by projected points
                sorted_data = pos_data.sort_values('projected_points', ascending=False)
                
                # Get top players for visualization
                top_n = min(30, len(sorted_data))
                top_players = sorted_data.head(top_n)
                
                # Create bar chart of top players
                import matplotlib.pyplot as plt
                import seaborn as sns
                
                plt.figure(figsize=(14, 10))
                
                # Create horizontal bar chart
                bars = plt.barh(
                    top_players['name'], 
                    top_players['projected_points'],
                    color=sns.color_palette('viridis', len(top_players))
                )
                
                # Add projection values to bars
                for i, bar in enumerate(bars):
                    plt.text(
                        bar.get_width() + 0.3,
                        bar.get_y() + bar.get_height()/2,
                        f"{top_players.iloc[i]['projected_points']:.1f}",
                        va='center'
                    )
                
                # Add titles and labels
                plt.title(f'Top {top_n} Projected {position.upper()} Players', fontsize=16)
                plt.xlabel('Projected Fantasy Points per Game')
                plt.ylabel('Player')
                
                # Adjust layout and save
                plt.tight_layout()
                plt.savefig(os.path.join(pos_dir, f'top_players.png'), dpi=300)
                plt.close()