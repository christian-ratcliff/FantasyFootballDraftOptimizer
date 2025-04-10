#!/usr/bin/env python3
"""
Player projection pipeline that handles the end-to-end process
of generating fantasy football player projections.
Includes functionality to generate projections from pre-trained models.
"""

import logging
import os
import pickle # Using pickle for saving/loading projections dictionary
import json
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import joblib # Keep if models are saved/loaded with joblib

# Import project modules
# Adjusted relative imports assuming standard structure
try:
    from src.data.loader import load_espn_league_data, load_nfl_data, process_player_data
    from src.features.engineering import FantasyFeatureEngineering
    from src.models.projections import PlayerProjectionModel
    # Import visualization modules only if needed (inside methods)
except ImportError as e:
     # Fallback for different structures or direct execution (less ideal)
     logging.warning(f"Using fallback imports due to: {e}")
     from data.loader import load_espn_league_data, load_nfl_data, process_player_data
     from features.engineering import FantasyFeatureEngineering
     from models.projections import PlayerProjectionModel


# --- Projection Pipeline Class ---
class ProjectionPipeline:
    """
    Orchestrates the end-to-end pipeline for player projections.
    Saves the final projections to a file for downstream use (e.g., RL).
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
        self.projections = {} # Stores final projections {pos: DataFrame}
        self.league_settings = {}
        self.roster_limits = {}
        self.scoring_settings = {}
        self.projection_model_instance = None # To hold the instance of PlayerProjectionModel

    def run(self):
        """Execute the appropriate pipeline steps based on config"""
        start_time = datetime.now()
        self.logger.info(f"Starting projection pipeline run at {start_time}")

        # --- Setup ---
        self.data_dirs = self.create_output_dirs()
        self.league_settings = self.load_league_settings_from_file()
        if not self.league_settings:
             self.logger.error("Failed to load league settings. Aborting pipeline.")
             return None # Indicate failure

        self.roster_limits = self.league_settings.get('roster_settings', {})
        self.scoring_settings = self.league_settings.get('scoring_settings', {})

        use_existing_models = self.config.get('models_loading', {}).get('use_existing_projection_models', False)
        projections_generated = False # Flag

        # --- Generate or Load Projections ---
        if use_existing_models:
            self.logger.info("Mode: Generating projections from existing models.")
            self.logger.info("Loading necessary feature sets for projection...")

            # Load feature sets needed for making predictions
            self.feature_sets, self.dropped_tiers = self._load_cached_feature_sets()

            # *** FIX: Implement the verification check directly here ***
            projection_features_valid = False
            required_proj_keys = [f"{pos}_projection" for pos in ['qb', 'rb', 'wr', 'te']]
            if not self.feature_sets: # Check if feature_sets dict is empty
                 self.logger.error("Feature sets dictionary is empty after loading cache.")
            else:
                for key in required_proj_keys:
                    if (key in self.feature_sets and
                        isinstance(self.feature_sets[key], pd.DataFrame) and
                        not self.feature_sets[key].empty):
                        projection_features_valid = True
                        self.logger.debug(f"Found valid projection feature set: {key}")
                        # break # Found at least one, can proceed
                    else:
                         self.logger.warning(f"Required projection feature set '{key}' missing, empty, or invalid type in cache.")

            if not projection_features_valid:
                 self.logger.error("Required projection feature sets missing or empty in cache. Cannot generate projections from saved models.")
                 self.projections = {} # Ensure projections is empty dict
            # *** End FIX ***
            else:
                # This method now populates self.projections AND calls save_projections internally
                # It should also handle cases where models might be missing for *some* positions
                self.projections = self.generate_and_save_projections_from_saved_models()
                if self.projections: # Check if generate_... returned a non-empty dict
                     # Further check if any DataFrame inside is non-empty
                     if any(isinstance(df, pd.DataFrame) and not df.empty for df in self.projections.values()):
                          projections_generated = True
                     else:
                          self.logger.warning("generate_and_save_projections_from_saved_models returned a dict, but all DataFrames were empty.")
                else:
                     self.logger.error("generate_and_save_projections_from_saved_models returned None or empty dict.")


        else: # Train new models
            self.logger.info("Mode: Training new models and generating projections.")
            self.load_data() # Loads raw/processed data
            self.engineer_features() # Creates feature sets
            # _train_models_and_project returns the projections dict
            self.projections = self._train_models_and_project()
            if self.projections:
                # Check if any DataFrame inside is non-empty
                if any(isinstance(df, pd.DataFrame) and not df.empty for df in self.projections.values()):
                    self.save_projections() # Save the generated projections
                    projections_generated = True
                else:
                    self.logger.error("Model training/projection resulted in empty DataFrames. No projections saved.")
            else:
                self.logger.error("Model training/projection failed. No projections generated.")

        # --- Check if projections were actually created/loaded ---
        if not projections_generated:
             self.logger.error("Pipeline finished but no valid projections were generated or loaded. Cannot proceed with post-projection steps.")
             return None # Return None explicitly if no projections

        # --- Post-Projection Steps ---
        self.logger.info("Projections generated/loaded successfully.")

        # Visualization (requires data)
        if self.config.get('visualizations', {}).get('enabled', True):
            if not self.processed_data: self.processed_data = self._load_cached_processed_data()
            if not self.feature_sets: self.feature_sets, self.dropped_tiers = self._load_cached_feature_sets() # Try loading again if needed for viz
            if self.processed_data or self.feature_sets:
                 self.create_visualizations()
            else:
                 self.logger.warning("Skipping visualizations: Processed data and feature sets could not be loaded/generated.")

        # Evaluation (requires data)
        if self.config.get('evaluation', {}).get('enabled', True):
             if not self.processed_data: self.processed_data = self._load_cached_processed_data()
             if self.processed_data:
                 self.evaluate_projections()
             else:
                 self.logger.warning("Skipping evaluation: Processed data could not be loaded.")

        # --- Finish ---
        duration = datetime.now() - start_time
        self.logger.info(f"Projection pipeline run completed in {duration}")
        return self.projections # Return the final dictionary


    def save_projections(self):
        """Saves the generated projections dictionary to a pickle file."""
        # 1. Validation Checks
        if not self.projections:
            self.logger.error("Cannot save projections: 'self.projections' is empty or None.")
            return
        if not isinstance(self.projections, dict):
            self.logger.error(f"Cannot save projections: 'self.projections' is not a dictionary (type: {type(self.projections)}).")
            return

        valid_data_found = False
        # Check for essential columns expected by downstream tasks (like RL)
        essential_cols = ['player_id', 'name', 'position', 'projected_points',
                          'projection_low', 'projection_high', 'ceiling_projection'] # Added range cols

        # 2. Detailed Inspection Loop
        self.logger.info(f"Preparing to save projections dictionary. Inspecting contents...")
        positions_to_save = {} # Store only valid dataframes
        for pos, df in self.projections.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                missing_cols = [col for col in essential_cols if col not in df.columns]
                if not missing_cols:
                    valid_data_found = True
                    # Check for NaNs in critical columns
                    nan_check_cols = ['player_id', 'projected_points']
                    if df[nan_check_cols].isnull().any().any():
                         self.logger.warning(f"  '{pos}': DataFrame contains NaNs in critical columns (player_id, projected_points). Saving anyway, but check source.")
                    self.logger.info(f"  '{pos}': DataFrame valid - Shape {df.shape}, Columns: {df.columns.tolist()[:7]}...") # Show more cols
                    positions_to_save[pos] = df # Add valid df to save dict
                else:
                    self.logger.error(f"  '{pos}': DataFrame invalid - Missing essential columns: {missing_cols}. Shape: {df.shape}. Skipping save for this position.")
            elif isinstance(df, pd.DataFrame) and df.empty:
                 self.logger.warning(f"  '{pos}': DataFrame is empty. Skipping save for this position.")
            else:
                 self.logger.error(f"  '{pos}': Contains invalid data type: {type(df)}. Skipping save for this position.")

        # 3. Check if *any* valid data was found
        if not valid_data_found:
             self.logger.error("Projections dictionary contains no valid DataFrames with essential columns. Aborting save.")
             # Ensure self.projections is cleared if nothing valid was found
             self.projections = {}
             return

        # 4. Define Path and Save using the filtered dict
        proj_cache_path = os.path.join(self.data_dirs['models'], 'player_projections.pkl')
        try:
            with open(proj_cache_path, 'wb') as f:
                pickle.dump(positions_to_save, f) # Save only the valid data
            self.logger.info(f"Successfully saved {len(positions_to_save)} valid player projection DataFrames to {proj_cache_path}")
        except Exception as e:
            self.logger.error(f"Error saving projections to {proj_cache_path}: {e}", exc_info=True)

    # --- Load/Cache Methods ---
    def create_output_dirs(self):
        """Create organized output directories based on config"""
        paths = self.config.get('paths', {})
        dirs = {
            'raw': paths.get('raw_dir', 'data/raw'),
            'processed': paths.get('processed_dir', 'data/processed'),
            'outputs': paths.get('output_dir', 'data/outputs'),
            'models': paths.get('models_dir', 'data/models')
        }
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        # Create subdirs within outputs if needed by viz/eval
        for subdir in ['evaluation', 'projections', 'qb', 'rb', 'wr', 'te', 'overall']:
             os.makedirs(os.path.join(dirs['outputs'], subdir), exist_ok=True)
        self.logger.info("Ensured output directories exist.")
        return dirs

    def load_league_settings_from_file(self):
        """Load league settings primarily from the generated JSON file."""
        config_path = self.config['paths'].get('config_path', 'configs/league_settings.json')
        settings = None
        if not os.path.exists(config_path):
            self.logger.warning(f"League settings JSON not found: {config_path}. Attempting generation.")
            try:
                # Lazy import to avoid circular dependencies if generate_config uses pipeline parts
                from generate_config import generate_config
                league_cfg = self.config.get('league', {})
                league_id = league_cfg.get('id'); year = league_cfg.get('year')
                espn_s2 = league_cfg.get('espn_s2'); swid = league_cfg.get('swid')
                if not league_id or not year: raise ValueError("Missing league id or year in main config.")
                success = generate_config(league_id, year, espn_s2, swid, output_path=config_path)
                if not success: self.logger.error("Failed to generate league settings JSON."); return None
            except ImportError: self.logger.error("generate_config.py not found or failed import."); return None
            except ValueError as e: self.logger.error(f"Config Error for generating settings: {e}"); return None
            except Exception as e: self.logger.error(f"Error generating league settings: {e}", exc_info=True); return None

        # Now try loading the potentially generated file
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f: settings = json.load(f)
                self.logger.info(f"Successfully loaded league settings JSON from {config_path}")
                # Basic Validation
                required_keys = ['league_info', 'starter_limits', 'roster_settings', 'scoring_settings', 'player_map']
                missing_keys = [k for k in required_keys if k not in settings]
                if missing_keys: raise ValueError(f"Loaded JSON missing required keys: {missing_keys}")
                if 'team_count' not in settings['league_info']: raise ValueError("Missing 'team_count'.")
                return settings
            except json.JSONDecodeError as e: self.logger.error(f"Error decoding JSON from {config_path}: {e}"); return None
            except ValueError as e: self.logger.error(f"Validation Error in {config_path}: {e}"); return None
            except Exception as e: self.logger.error(f"Unexpected error loading {config_path}: {e}", exc_info=True); return None
        else:
            self.logger.error(f"File {config_path} still not found after generation attempt."); return None


    # def load_data(self):
    #     """Load raw or cached data based on config settings"""
    #     data_cfg = self.config.get('data', {}); cache_cfg = self.config.get('caching', {})
    #     league_cfg = self.config.get('league', {})

    #     year = league_cfg.get('year')
    #     start_year = data_cfg.get('start_year', 2016)
    #     if not year: self.logger.error("Target year missing in config['league']."); return
    #     years = list(range(start_year, year + 1)) # Include target year for processing

    #     include_ngs = data_cfg.get('include_ngs', True)
    #     use_cached_raw = cache_cfg.get('use_cached_raw_data', True)
    #     use_cached_processed = cache_cfg.get('use_cached_processed_data', True)
    #     use_cached_features = cache_cfg.get('use_cached_feature_sets', False)

    #     # 1. Load Raw Data (Download or Cache)
    #     if use_cached_raw:
    #         self.nfl_data = self._load_cached_raw_data()
    #         if self.nfl_data: self.logger.info("Using cached raw data.")
    #         else: use_cached_raw = False; self.logger.warning("Raw cache empty/failed, downloading.")
    #     if not use_cached_raw:
    #         self.nfl_data = load_nfl_data(years=years, include_ngs=include_ngs, ngs_min_year=start_year, use_threads=True)
    #         self._save_cache(self.nfl_data, self.data_dirs['raw'], "raw") # Save downloaded data

    #     if not self.nfl_data:
    #         self.logger.error("Failed to load or download raw NFL data. Aborting data loading.")
    #         return

    #     # 2. Load Processed Data (Process or Cache)
    #     # Only process if we don't intend to load features directly
    #     should_load_processed = use_cached_processed and not use_cached_features
    #     if should_load_processed:
    #         self.processed_data = self._load_cached_processed_data()
    #         if self.processed_data: self.logger.info("Using cached processed data.")
    #         else: should_load_processed = False; self.logger.warning("Processed cache empty/failed, reprocessing.")

    #     if not should_load_processed and not use_cached_features:
    #         if not self.nfl_data: self.logger.error("No raw data available to process."); return
    #         self.logger.info("Processing raw NFL data...")
    #         self.processed_data = process_player_data(self.nfl_data)
    #         self._save_cache(self.processed_data, self.data_dirs['processed'], "processed")

    #     if not use_cached_features and not self.processed_data:
    #          self.logger.error("Failed to load or generate processed data.")


    def load_data(self):
        """Load raw or cached data based on config settings,
           ensuring processed data is always read from disk if generated."""
        data_cfg = self.config.get('data', {}); cache_cfg = self.config.get('caching', {})
        league_cfg = self.config.get('league', {})

        year = league_cfg.get('year')
        start_year = data_cfg.get('start_year', 2016)
        if not year: self.logger.error("Target year missing in config['league']."); return
        years = list(range(start_year, year + 1)) # Include target year for processing

        include_ngs = data_cfg.get('include_ngs', True)
        use_cached_raw = cache_cfg.get('use_cached_raw_data', True)
        use_cached_processed = cache_cfg.get('use_cached_processed_data', True)
        # Feature set cache handling is done in engineer_features

        # --- 1. Load Raw Data ---
        if use_cached_raw:
            self.nfl_data = self._load_cached_raw_data()
            if self.nfl_data: self.logger.info("Using cached raw data.")
            else: use_cached_raw = False; self.logger.warning("Raw cache empty/failed, downloading.")
        if not use_cached_raw:
            self.nfl_data = load_nfl_data(years=years, include_ngs=include_ngs, ngs_min_year=start_year, use_threads=True)
            self._save_cache(self.nfl_data, self.data_dirs['raw'], "raw") # Save downloaded data

        if not self.nfl_data:
            self.logger.error("Failed to load or download raw NFL data. Aborting data loading.")
            return

        # --- 2. Process Data (or Load Cache) & Ensure Disk Read ---
        processed_data_loaded = False
        if use_cached_processed:
            self.processed_data = self._load_cached_processed_data()
            if self.processed_data:
                self.logger.info("Using cached processed data.")
                processed_data_loaded = True
            else:
                self.logger.warning("Configured to use processed cache, but cache empty/failed. Will reprocess.")

        # If not loaded from cache, process, save, then reload
        if not processed_data_loaded:
            if not self.nfl_data:
                self.logger.error("No raw data available to process.")
                return # Cannot proceed
            self.logger.info("Processing raw NFL data...")
            processed_data_temp = process_player_data(self.nfl_data)

            if not processed_data_temp:
                 self.logger.error("Processing raw data resulted in empty dictionary. Cannot proceed.")
                 self.processed_data = {}
                 return

            self.logger.info("Saving newly processed data to cache before use...")
            self._save_cache(processed_data_temp, self.data_dirs['processed'], "processed")

            self.logger.info("Reloading processed data from cache to ensure consistency...")
            self.processed_data = self._load_cached_processed_data() # Read it back immediately
            if not self.processed_data:
                self.logger.error("Failed to reload processed data immediately after saving! Check permissions or saving logic.")
                # If reload fails, we might still try to use the in-memory version, but log a severe warning
                self.processed_data = processed_data_temp # Fallback to in-memory version
                self.logger.warning("Using in-memory processed data due to reload failure.")
            else:
                 self.logger.info("Successfully reloaded processed data from cache.")

        if not self.processed_data:
             self.logger.error("Failed to load or generate processed data.")

    def _save_cache(self, data_dict, cache_dir, cache_type):
        """Helper to save dataframes to CSV cache."""
        self.logger.info(f"Saving {cache_type} data to cache: {cache_dir}")
        for key, df in data_dict.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                path = os.path.join(cache_dir, f"{key}.csv")
                try:
                    df.to_csv(path, index=False)
                    self.logger.debug(f"Saved {cache_type} cache: {key}.csv")
                except Exception as e:
                    self.logger.error(f"Failed to save {cache_type} cache {key}: {e}")
            elif isinstance(df, pd.DataFrame) and df.empty:
                 self.logger.debug(f"Skipping empty DataFrame for {cache_type} cache: {key}")
            # else: ignore non-dataframes

    def _load_cached_raw_data(self):
        """Load raw data from CSV cache files."""
        cache_dir = self.data_dirs.get('raw'); data = {}; found_any = False
        if not cache_dir or not os.path.isdir(cache_dir): return {}
        self.logger.info(f"Attempting to load raw data from cache: {cache_dir}")
        expected_files = ['seasonal.csv', 'weekly.csv', 'rosters.csv', 'player_ids.csv', 'schedules.csv',
                          'ngs_passing.csv', 'ngs_rushing.csv', 'ngs_receiving.csv', 'snap_counts.csv']
        for file in expected_files:
            path = os.path.join(cache_dir, file); name = file.split('.')[0]
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, low_memory=False)
                    if len(df) > 0: data[name] = df; found_any = True; self.logger.debug(f"Loaded raw cache: {name}.csv")
                    else: self.logger.warning(f"Raw cache file {file} empty.")
                except pd.errors.EmptyDataError: self.logger.warning(f"Raw cache file {file} is empty.")
                except Exception as e: self.logger.error(f"Error loading raw cache {file}: {e}", exc_info=True)
        return data if found_any else {}

    def _load_cached_processed_data(self):
        """Load processed data from CSV cache files."""
        cache_dir = self.data_dirs.get('processed'); data = {}; found_any = False
        if not cache_dir or not os.path.isdir(cache_dir): return {}
        self.logger.info(f"Attempting to load processed data from cache: {cache_dir}")
        # Include base processed files and combined files
        expected_names = ['seasonal', 'weekly', 'active_players',
                          'combined_qb', 'combined_rb', 'combined_wr', 'combined_te']
        for name in expected_names:
            path = os.path.join(cache_dir, f"{name}.csv")
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, low_memory=False)
                    if len(df) > 0: data[name] = df; found_any = True; self.logger.debug(f"Loaded processed cache: {name}.csv")
                    else: self.logger.warning(f"Processed cache file {name}.csv empty.")
                except pd.errors.EmptyDataError: self.logger.warning(f"Processed cache file {name}.csv is empty.")
                except Exception as e: self.logger.error(f"Error loading processed cache {name}: {e}", exc_info=True)
        return data if found_any else {}

    # def _load_cached_feature_sets(self):
    #     """Load engineered feature sets from CSV cache files."""
    #     cache_dir = self.data_dirs.get('processed'); feature_sets = {}; found_any = False
    #     dropped_tiers = None
    #     if not cache_dir or not os.path.isdir(cache_dir): return {}, None
    #     self.logger.info(f"Attempting to load feature sets from cache: {cache_dir}")
    #     for file in os.listdir(cache_dir):
    #         if file.startswith("features_") and file.endswith(".csv"):
    #             path = os.path.join(cache_dir, file); key = file[len("features_"):-len(".csv")]
    #             try:
    #                 df = pd.read_csv(path, low_memory=False)
    #                 if len(df) > 0 and 'player_id' in df.columns:
    #                      feature_sets[key] = df; found_any = True; self.logger.info(f"Loaded feature set cache: '{key}'")
    #                 elif len(df) == 0: self.logger.warning(f"Feature set cache {file} empty.")
    #                 else: self.logger.warning(f"Feature set cache {file} missing 'player_id'.")
    #             except pd.errors.EmptyDataError: self.logger.warning(f"Feature set cache {file} is empty.")
    #             except Exception as e: self.logger.error(f"Error loading feature set cache {file}: {e}", exc_info=True)

    #     # Load dropped tiers info
    #     tiers_path = os.path.join(cache_dir, 'dropped_tiers.json')
    #     if os.path.exists(tiers_path):
    #         try:
    #             with open(tiers_path, 'r') as f:
    #                  loaded = json.load(f)
    #                  # Ensure keys are strings and values are lists of ints
    #                  dropped_tiers = {str(k): [int(i) for i in v] for k, v in loaded.items()}
    #             self.logger.info(f"Loaded dropped tiers info from cache.")
    #         except Exception as e: self.logger.error(f"Error loading dropped tiers cache: {e}", exc_info=True)

    #     return feature_sets if found_any else {}, dropped_tiers

    def _load_cached_feature_sets(self):
        """Load engineered feature sets from CSV cache files."""
        cache_dir = self.data_dirs.get('processed')
        feature_sets = {}
        found_any = False
        dropped_tiers = None # Initialize dropped_tiers

        if not cache_dir or not os.path.isdir(cache_dir):
            self.logger.error(f"Processed cache directory not found or invalid: {cache_dir}")
            return {}, None # Return empty dict and None for tiers

        self.logger.info(f"Attempting to load feature sets from cache: {cache_dir}")

        # Define expected key patterns based on how they are likely saved
        # (from your tree output and the engineer_features logic)
        expected_key_bases = []
        for pos in ['qb', 'rb', 'wr', 'te']:
            expected_key_bases.extend([
                f"{pos}_features",
                f"{pos}_train",
                f"{pos}_projection",
                f"{pos}_train_filtered",
                f"{pos}_projection_filtered"
            ])
        # Add combined keys if they might be saved (check _save_cache call if unsure)
        # expected_key_bases.extend(['train_combined', 'projection_combined'])

        # Iterate through expected keys and try to load corresponding CSV
        for key in expected_key_bases:
            filename = f"{key}.csv"
            path = os.path.join(cache_dir, filename)

            if os.path.exists(path):
                try:
                    # Check file size before reading fully
                    if os.path.getsize(path) == 0:
                         self.logger.warning(f"Feature set cache {filename} is empty (0 bytes). Skipping.")
                         continue

                    df = pd.read_csv(path, low_memory=False)
                    # Validate DataFrame content
                    if not df.empty and 'player_id' in df.columns:
                        feature_sets[key] = df
                        found_any = True
                        self.logger.info(f"Loaded feature set cache: '{key}' from {filename}")
                    elif df.empty:
                        self.logger.warning(f"Feature set cache {filename} loaded as empty DataFrame. Skipping.")
                    else: # Non-empty but missing player_id
                        self.logger.warning(f"Feature set cache {filename} missing 'player_id' column. Skipping.")

                except pd.errors.EmptyDataError:
                    # This case might be caught by os.path.getsize == 0, but good to have
                    self.logger.warning(f"Feature set cache {filename} is empty (pandas error). Skipping.")
                except Exception as e:
                    self.logger.error(f"Error loading feature set cache {filename}: {e}", exc_info=True)
            # else: # File doesn't exist for this expected key - silently ignore, it might not be generated in all runs

        # Load dropped tiers info (no changes needed here)
        tiers_path = os.path.join(cache_dir, 'dropped_tiers.json')
        if os.path.exists(tiers_path):
            try:
                with open(tiers_path, 'r') as f:
                    loaded = json.load(f)
                    # Ensure keys are strings and values are lists of ints
                    dropped_tiers = {str(k): [int(i) for i in v] for k, v in loaded.items()}
                self.logger.info(f"Loaded dropped tiers info from cache.")
            except Exception as e:
                self.logger.error(f"Error loading dropped tiers cache: {e}", exc_info=True)
                dropped_tiers = None # Ensure it's None on error

        # Return the loaded sets (or empty dict) and tiers info
        return feature_sets if found_any else {}, dropped_tiers

    # --- Feature Engineering ---
    # def engineer_features(self):
    #     """Run feature engineering process using config parameters"""
    #     cache_cfg = self.config.get('caching', {})
    #     cluster_cfg = self.config.get('clustering', {})
    #     use_cached = cache_cfg.get('use_cached_feature_sets', False)
    #     year = self.config.get('league', {}).get('year')

    #     if use_cached:
    #         self.feature_sets, self.dropped_tiers = self._load_cached_feature_sets()
    #         if self.feature_sets: self.logger.info("Using cached feature sets."); return self.feature_sets, self.dropped_tiers
    #         else: self.logger.warning("Feature cache empty/failed, re-engineering.")

    #     if not self.processed_data:
    #          self.logger.error("No processed data available for feature engineering.")
    #          return {}, None

    #     self.logger.info("Performing feature engineering...")
    #     fe = FantasyFeatureEngineering(self.processed_data, target_year=year)
    #     fe.create_position_features().prepare_prediction_features()

    #     cluster_count = cluster_cfg.get('cluster_count', 5)
    #     drop_tiers_count = cluster_cfg.get('drop_bottom_tiers', 1)
    #     # Define position-specific drops based on config (e.g., {'qb': 2, 'rb': 1, ...})
    #     pos_tier_drops = {p: drop_tiers_count for p in ['qb','rb','wr','te']} # Simple uniform drop for now
    #     # You could add specific overrides in config if needed:
    #     # pos_tier_drops.update(cluster_cfg.get('position_tier_drops', {}))

    #     fe.cluster_players(n_clusters=cluster_count, position_tier_drops=pos_tier_drops)
    #     fe.finalize_features(apply_filtering=cluster_cfg.get('use_filtered', True)) # Apply filtering based on config

    #     self.feature_sets = fe.get_feature_sets()
    #     self.dropped_tiers = fe.dropped_tiers

    #     # Save the generated features and tiers info
    #     self._save_cache(self.feature_sets, self.data_dirs['processed'], "features")
    #     if self.dropped_tiers:
    #          tiers_path = os.path.join(self.data_dirs['processed'], 'dropped_tiers.json')
    #          try:
    #              st = {k: [int(i) for i in v] for k, v in self.dropped_tiers.items()} # Ensure int format
    #              with open(tiers_path, 'w') as f: json.dump(st, f, indent=2)
    #              self.logger.info(f"Saved dropped tiers info to {tiers_path}")
    #          except Exception as e: self.logger.error(f"Error saving dropped tiers JSON: {e}")

    #     return self.feature_sets, self.dropped_tiers


    def engineer_features(self):
        """Run feature engineering process, ensuring features are read from disk if generated."""
        cache_cfg = self.config.get('caching', {})
        cluster_cfg = self.config.get('clustering', {})
        year = self.config.get('league', {}).get('year')
        use_cached_features = cache_cfg.get('use_cached_feature_sets', False)

        feature_sets_loaded = False
        if use_cached_features:
            self.feature_sets, self.dropped_tiers = self._load_cached_feature_sets()
            if self.feature_sets:
                self.logger.info("Using cached feature sets.")
                feature_sets_loaded = True
            else:
                self.logger.warning("Configured to use feature cache, but cache empty/failed. Will re-engineer.")

        # If not loaded from cache, engineer, save, then reload
        if not feature_sets_loaded:
            if not self.processed_data:
                self.logger.error("No processed data available for feature engineering.")
                self.feature_sets = {}
                self.dropped_tiers = None
                return {}, None

            self.logger.info("Performing feature engineering...")
            fe = FantasyFeatureEngineering(self.processed_data, target_year=year)
            # --- Run Feature Engineering Steps ---
            fe.create_position_features()
            # Prepare prediction features (creates _train and _projection keys)
            fe.prepare_prediction_features()
            # Clustering (creates _filtered keys potentially, and cluster/tier columns)
            cluster_count = cluster_cfg.get('cluster_count', 5)
            drop_tiers_count = cluster_cfg.get('drop_bottom_tiers', 1)
            pos_tier_drops = {p: drop_tiers_count for p in ['qb','rb','wr','te']}
            pos_tier_drops.update(cluster_cfg.get('position_tier_drops', {}))
            fe.cluster_players(n_clusters=cluster_count, position_tier_drops=pos_tier_drops)
            # Finalize (creates combined keys potentially) - pass apply_filtering flag
            fe.finalize_features(apply_filtering=cluster_cfg.get('use_filtered', True))
            # --- End Feature Engineering Steps ---

            feature_sets_temp = fe.get_feature_sets()
            dropped_tiers_temp = fe.dropped_tiers

            if not feature_sets_temp:
                 self.logger.error("Feature engineering resulted in empty dictionary. Cannot proceed.")
                 self.feature_sets = {}
                 self.dropped_tiers = None
                 return {}, None

            # Save the generated features and tiers info
            self.logger.info("Saving newly engineered feature sets to cache before use...")
            self._save_cache(feature_sets_temp, self.data_dirs['processed'], "features")
            # Save dropped tiers info
            if dropped_tiers_temp:
                 tiers_path = os.path.join(self.data_dirs['processed'], 'dropped_tiers.json')
                 try:
                     st = {k: [int(i) for i in v] for k, v in dropped_tiers_temp.items()} # Ensure int format
                     with open(tiers_path, 'w') as f: json.dump(st, f, indent=2)
                     self.logger.info(f"Saved dropped tiers info to {tiers_path}")
                 except Exception as e: self.logger.error(f"Error saving dropped tiers JSON: {e}")

            # Reload features and tiers from cache
            self.logger.info("Reloading feature sets from cache to ensure consistency...")
            self.feature_sets, self.dropped_tiers = self._load_cached_feature_sets()
            if not self.feature_sets:
                self.logger.error("Failed to reload feature sets immediately after saving! Check permissions or saving logic.")
                # Fallback to in-memory version with warning
                self.feature_sets = feature_sets_temp
                self.dropped_tiers = dropped_tiers_temp
                self.logger.warning("Using in-memory feature sets due to reload failure.")
            else:
                 self.logger.info("Successfully reloaded feature sets from cache.")


        # Check again if feature sets are valid after potential reload/fallback
        if not self.feature_sets:
             self.logger.error("Feature sets dictionary is empty or failed to load.")
             return {}, None

        return self.feature_sets, self.dropped_tiers
    
    # --- Model Training & Projection ---
    def _train_models_and_project(self):
        """Train models and generate projections."""
        proj_cfg = self.config.get('projections', {})
        cluster_cfg = self.config.get('clustering', {})
        models_dir = self.config['paths'].get('models_dir', 'data_testing/models')

        use_filtered = cluster_cfg.get('use_filtered', True)
        use_do_not_draft = proj_cfg.get('use_do_not_draft', True)
        use_hierarchical = proj_cfg.get('use_hierarchical', False)
        model_type = proj_cfg.get('model_type', 'xgboost')
        optimize_hyperparams = proj_cfg.get('optimize_hyperparams', True)
        feature_method = proj_cfg.get('feature_selection_method', 'shap')
        prioritize_ngs = proj_cfg.get('prioritize_ngs', True) # Get NGS priority flag

        if not self.feature_sets: self.logger.error("No feature sets available for training."); return {}

        self.logger.info(f"Initializing training: type={model_type}, use_filtered={use_filtered}, hierarchical={use_hierarchical}, optimize={optimize_hyperparams}, featsel={feature_method}, ngs_priority={prioritize_ngs}")

        # Instantiate the model class (pass feature sets)
        try:
            # Pass use_filtered flag to the constructor
            self.projection_model_instance = PlayerProjectionModel(self.feature_sets, output_dir=models_dir, use_filtered=use_filtered)
            # Store dropped tiers info in the instance if available
            if self.dropped_tiers:
                 self.projection_model_instance.dropped_tiers = self.dropped_tiers

        except Exception as e:
             self.logger.error(f"Failed to instantiate PlayerProjectionModel: {e}", exc_info=True)
             return {}

        # Train direct models (or load if configured, though this method assumes training)
        all_metrics = self.projection_model_instance.train_all_positions(
            model_type=model_type,
            optimize_hyperparams=optimize_hyperparams,
            # feature_selection_method=feature_method, # Pass method from config
            # prioritize_ngs=prioritize_ngs # Pass NGS flag
        )

        # Log direct model validation metrics
        self.logger.info("--- Direct Model Validation Metrics ---")
        for pos, mets in (all_metrics or {}).items():
            if mets and 'validation_metrics' in mets and 'rmse' in mets['validation_metrics']:
                 vm = mets['validation_metrics']
                 self.logger.info(f"  {pos.upper()}: FinalModel RMSE={vm.get('rmse',0):.3f}, MAE={vm.get('mae',0):.3f}, R2={vm.get('r2',0):.3f}")
                 # Log average temporal RMSE if available
                 temporal_results = vm.get('temporal_validation', [])
                 if temporal_results:
                      temporal_rmses = [r['test_rmse'] for r in temporal_results if 'test_rmse' in r]
                      avg_temporal_rmse = np.mean(temporal_rmses) if temporal_rmses else np.nan
                      self.logger.info(f"  {pos.upper()}: Avg Temporal RMSE = {avg_temporal_rmse:.3f}")
            else:
                 self.logger.warning(f"Validation metrics missing or incomplete for {pos}.")


        # Train hierarchical models if enabled
        if use_hierarchical:
            self.logger.info("Training hierarchical component models...")
            self.projection_model_instance.train_all_hierarchical_models(
                 model_type=model_type,
                 optimize_hyperparams=optimize_hyperparams, # Optionally optimize component models too
                 feature_selection_method=feature_method
            )
            # Save models again after training hierarchical components (overwrites previous save)
            for pos in ['qb', 'rb', 'wr', 'te']:
                self.projection_model_instance.save_model(pos)

        # Generate projections for the target year
        proj_year = proj_cfg.get('projection_year')
        if not proj_year: self.logger.error("projection_year missing in config."); return {}

        self.logger.info(f"Generating projections for {proj_year}...")
        if not self.projection_model_instance: self.logger.error("Model instance not created."); return {}

        try:
            projections = self.projection_model_instance.generate_full_projections(
                use_do_not_draft=use_do_not_draft,
                use_hierarchical=use_hierarchical # Use flag from config
            )
            if not projections: self.logger.error("Projection generation returned empty dict."); return {}
            self.logger.info("Projection generation successful.")
            return projections
        except Exception as e:
            self.logger.exception("Error during generate_full_projections.")
            return {}

    def generate_and_save_projections_from_saved_models(self):
        """Loads models and generates projections using cached feature sets."""
        self.logger.info("--- Generating Projections from Saved Models ---")
        models_dir = self.config['paths'].get('models_dir', 'data/models')
        proj_cfg = self.config.get('projections', {})
        cluster_cfg = self.config.get('clustering', {})

        use_filtered = cluster_cfg.get('use_filtered', True) # Need to know which feature set to use
        use_do_not_draft = proj_cfg.get('use_do_not_draft', True)
        use_hierarchical = proj_cfg.get('use_hierarchical', False) # Ensure this matches how models were saved if applicable

        generated_projections = {}

        # Instance needs feature sets for projection method
        # Pass use_filtered to constructor so it knows which features to load/expect
        self.projection_model_instance = PlayerProjectionModel(self.feature_sets, output_dir=models_dir, use_filtered=use_filtered)
        # Load dropped tiers if filtering is used
        if use_filtered and self.dropped_tiers:
             self.projection_model_instance.dropped_tiers = self.dropped_tiers

        any_model_loaded = False
        for position in ['qb', 'rb', 'wr', 'te']:
            model_path = os.path.join(models_dir, f"{position}_model.joblib")
            if not os.path.exists(model_path):
                self.logger.warning(f"Model file not found: {model_path}. Skipping {position}.")
                continue
            try:
                # Load the entire saved dictionary
                model_save_data = joblib.load(model_path)
                # Basic validation of loaded data
                if not isinstance(model_save_data, dict) or 'model' not in model_save_data or 'features' not in model_save_data:
                     self.logger.error(f"Loaded {position} model data is invalid or missing keys. Skipping.")
                     continue

                # Populate the instance's models dictionary
                self.projection_model_instance.models[position] = {
                     'model': model_save_data['model'],
                     'features': model_save_data['features'],
                     'model_type': model_save_data.get('model_type', 'unknown'),
                     'training_samples': model_save_data.get('training_samples', 'N/A'),
                     'validation_metrics': model_save_data.get('validation_metrics', {})
                 }
                # Populate hierarchical models if they exist in the saved file
                if 'hierarchical_models' in model_save_data:
                     self.projection_model_instance.hierarchical_models[position] = model_save_data['hierarchical_models']
                     self.logger.info(f"Loaded hierarchical model components for {position}.")

                any_model_loaded = True
                self.logger.info(f"Successfully loaded model and metadata for {position}.")
                # Optionally log more details from loaded data
                # self.logger.debug(f"  Features: {model_save_data['features'][:10]}...")
                # metrics = model_save_data.get('validation_metrics', {})
                # if metrics: self.logger.debug(f"  Validation RMSE: {metrics.get('rmse', 'N/A'):.3f}")

            except Exception as e:
                self.logger.error(f"Error loading {position} model from {model_path}: {e}", exc_info=True)
                continue

        if not any_model_loaded:
             self.logger.error("No valid projection models were loaded. Cannot generate projections.")
             return {}

        # --- Generate Projections ---
        self.logger.info("Generating projections using loaded models...")
        try:
            generated_projections = self.projection_model_instance.generate_full_projections(
                use_do_not_draft=use_do_not_draft,
                use_hierarchical=use_hierarchical # Use config flag here
            )
        except Exception as e:
            self.logger.error(f"Error during generate_full_projections using loaded models: {e}", exc_info=True)
            return {}

        self.logger.info("--- Finished Generating Projections from Saved Models ---")

        # Store and save the projections
        self.projections = generated_projections
        if self.projections:
             # Check if any dataframe inside is valid before saving
             if any(isinstance(df, pd.DataFrame) and not df.empty for df in self.projections.values()):
                  self.save_projections() # Save the generated projections dict
             else:
                  self.logger.warning("Projection generation resulted in empty dataframes. Not saving.")
                  return {} # Return empty if nothing valid generated
        else:
             self.logger.error("Projection generation returned None or empty dict.")
             return {}

        return self.projections

    # --- Evaluation & Visualization ---
    def evaluate_projections(self):
        """Evaluate projections against actual results if available."""
        eval_cfg = self.config.get('evaluation', {}); paths = self.config.get('paths', {})
        proj_cfg = self.config.get('projections', {})
        proj_year = proj_cfg.get('projection_year'); curr_year = datetime.now().year

        if not proj_year or proj_year >= curr_year:
             self.logger.info(f"Skipping evaluation: Year {proj_year} is not in the past or not specified."); return None
        if not self.processed_data or 'seasonal' not in self.processed_data or self.processed_data['seasonal'].empty:
             self.logger.error("No valid processed seasonal data found for evaluation."); return None
        if not self.projections: self.logger.error("No projections available to evaluate."); return None

        actual_data = self.processed_data['seasonal']
        actual_year_data = actual_data[actual_data['season'] == proj_year].copy()
        if actual_year_data.empty: self.logger.error(f"No actual data found for evaluation year {proj_year}."); return None
        # Ensure target column exists
        if 'fantasy_points_per_game' not in actual_year_data.columns:
             self.logger.error("Actual data missing 'fantasy_points_per_game' column for evaluation."); return None

        self.logger.info(f"--- Evaluating Projections for {proj_year} ---")
        results = {}; eval_dfs = []; eval_dir = os.path.join(paths.get('output_dir', 'data/outputs'), 'evaluation'); os.makedirs(eval_dir, exist_ok=True)

        for pos, pos_proj_df in self.projections.items():
            if not isinstance(pos_proj_df, pd.DataFrame) or pos_proj_df.empty: continue
            pos_actual_df = actual_year_data[actual_year_data['position'] == pos.upper()].copy()
            if pos_actual_df.empty: continue

            # Verify required columns
            req_p = ['player_id', 'name', 'projected_points', 'projection_low', 'projection_high']
            req_a = ['player_id', 'name', 'fantasy_points_per_game']
            if not all(c in pos_proj_df.columns for c in req_p): self.logger.warning(f"Eval skip {pos}: Proj missing cols: {[c for c in req_p if c not in pos_proj_df.columns]}."); continue
            if not all(c in pos_actual_df.columns for c in req_a): self.logger.warning(f"Eval skip {pos}: Actual missing cols: {[c for c in req_a if c not in pos_actual_df.columns]}."); continue

            try:
                # Ensure consistent ID types for merging
                pos_proj_df['player_id'] = pos_proj_df['player_id'].astype(str)
                pos_actual_df['player_id'] = pos_actual_df['player_id'].astype(str)

                # Merge projections and actuals
                merged = pd.merge(pos_proj_df[req_p], pos_actual_df[req_a], on='player_id', how='inner', suffixes=('_proj', '_actual'))
                if merged.empty: continue
                merged['name'] = merged['name_actual']; merged['position'] = pos.upper() # Use actual name, assign position

                # Filter to players with meaningful scores
                filt = merged[(merged['fantasy_points_per_game'] > 1.0) & (merged['projected_points'] > 1.0)].copy()
                if filt.empty: continue

                # Calculate errors and metrics
                err = filt['fantasy_points_per_game'] - filt['projected_points']
                mets = {'pos': pos.upper(), 'n': len(filt), 'mae': np.round(err.abs().mean(), 3), 'rmse': np.round(np.sqrt((err**2).mean()), 3),
                        'bias': np.round(err.mean(), 3), 'corr': np.round(filt['fantasy_points_per_game'].corr(filt['projected_points']), 3),
                        'range_acc': np.round(((filt['fantasy_points_per_game'] >= filt['projection_low']) & (filt['fantasy_points_per_game'] <= filt['projection_high'])).mean()*100, 1)}

                results[pos] = mets; eval_dfs.append(filt) # Add filtered data for combined plot
                self.logger.info(f"{mets['pos']} Eval (N={mets['n']}): MAE={mets['mae']:.2f}, RMSE={mets['rmse']:.2f}, Corr={mets['corr']:.3f}, Range%={mets['range_acc']:.1f}")

                # Save position-specific evaluation data
                merged.round(2).to_csv(os.path.join(eval_dir, f'{pos}_eval_{proj_year}.csv'), index=False)

                # Create position-specific scatter plot
                plt.figure(figsize=(8, 8)); sns.scatterplot(data=filt, x='projected_points', y='fantasy_points_per_game', alpha=0.7, label=f"{pos.upper()} (N={mets['n']})")
                max_val = max(filt['projected_points'].max(), filt['fantasy_points_per_game'].max()) * 1.1; max_val = max(10, max_val) # Ensure decent plot range
                plt.plot([0, max_val], [0, max_val], 'r--', alpha=0.7, label='Perfect Prediction')
                sns.regplot(data=filt, x='projected_points', y='fantasy_points_per_game', scatter=False, color='blue', line_kws={'label': 'Regression Fit'})
                plt.title(f"{mets['pos']} Projection vs Actual ({proj_year})"); plt.xlabel('Projected FPTS/Game'); plt.ylabel('Actual FPTS/Game')
                plt.text(0.05, 0.95, f"Corr: {mets['corr']:.3f}\nRMSE: {mets['rmse']:.2f}", transform=plt.gca().transAxes, va='top', bbox=dict(boxstyle='round', alpha=0.8))
                plt.xlim(0, max_val); plt.ylim(0, max_val); plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
                plt.savefig(os.path.join(eval_dir, f'{pos}_scatter_{proj_year}.png'), dpi=150); plt.close()

            except Exception as e: self.logger.error(f"Error during evaluation for {pos}: {e}", exc_info=True)

        # --- Create Combined Scatter Plot ---
        if eval_dfs:
             try:
                 combined_eval_df = pd.concat(eval_dfs, ignore_index=True)
                 if not combined_eval_df.empty and 'position' in combined_eval_df.columns:
                     self.logger.info(f"Combined evaluation data for plotting: {combined_eval_df.shape}")
                     plt.figure(figsize=(10, 10))
                     sns.scatterplot(data=combined_eval_df, x='projected_points', y='fantasy_points_per_game', hue='position', alpha=0.6, palette='colorblind')
                     max_proj = combined_eval_df['projected_points'].max() if not combined_eval_df['projected_points'].empty else 10
                     max_actual = combined_eval_df['fantasy_points_per_game'].max() if not combined_eval_df['fantasy_points_per_game'].empty else 10
                     max_val_comb = max(max_proj, max_actual) * 1.1; max_val_comb = max(10, max_val_comb)
                     plt.plot([0, max_val_comb], [0, max_val_comb], 'r--', alpha=0.7, label='Perfect Prediction')
                     plt.title(f'Overall Projection vs Actual ({proj_year})'); plt.xlabel('Projected FPTS/Game'); plt.ylabel('Actual FPTS/Game')
                     plt.xlim(0, max_val_comb); plt.ylim(0, max_val_comb); plt.grid(True, alpha=0.3); plt.legend(title='Position'); plt.tight_layout()
                     plt.savefig(os.path.join(eval_dir, f'overall_accuracy_scatter_{proj_year}.png'), dpi=150); plt.close()
                     self.logger.info(f"Saved overall accuracy scatter plot to {eval_dir}")
                 else:
                     self.logger.warning("Combined evaluation DataFrame empty or missing 'position' column. Skipping overall plot.")
             except Exception as concat_err: self.logger.error(f"Error during combined eval plot: {concat_err}", exc_info=True)

        self.logger.info(f"--- Projection Evaluation Finished ---")
        return results

    def get_summary(self):
        """Create a human-readable summary of the projections."""
        try:
             league_info = self.league_settings.get('league_info', {})
             proj_cfg = self.config.get('projections', {})
             data_cfg = self.config.get('data', {})
             summary = {
                 'league': league_info.get('name', 'N/A'),
                 'teams': league_info.get('team_count', 'N/A'),
                 'projection_year': proj_cfg.get('projection_year', 'N/A'),
                 'used_hierarchical': proj_cfg.get('use_hierarchical', 'N/A'),
                 'used_ngs': data_cfg.get('include_ngs', 'N/A'),
                 'top_players': {}
             }
             if isinstance(self.projections, dict):
                 for pos in ['qb', 'rb', 'wr', 'te']:
                     summary['top_players'][pos] = []
                     if pos in self.projections and isinstance(self.projections[pos], pd.DataFrame):
                         df = self.projections[pos]
                         req = ['name', 'projected_points']
                         if all(c in df.columns for c in req):
                             df['projected_points'] = pd.to_numeric(df['projected_points'], errors='coerce')
                             filt = df.dropna(subset=['projected_points'])
                             if not filt.empty:
                                  # Include player_id in summary data
                                  summary['top_players'][pos] = [
                                      {"name": p['name'], "points": p['projected_points'], "id": p.get('player_id', 'N/A')}
                                      for _, p in filt.nlargest(5, 'projected_points').iterrows()
                                  ]
             return summary
        except Exception as e:
             self.logger.error(f"Error creating summary: {e}")
             return {} # Return empty dict on error

    def create_visualizations(self):
        """Generate visualizations for models and data."""
        self.logger.info("Attempting to create visualizations...")
        # Ensure data/features are loaded if they weren't already
        if not self.processed_data: self.processed_data = self._load_cached_processed_data()
        if not self.feature_sets: self.feature_sets, self.dropped_tiers = self._load_cached_feature_sets()

        # Only proceed if we have something to visualize
        if not self.processed_data and not self.feature_sets and not self.projections:
            self.logger.warning("No data, features, or projections available for visualization.")
            return self

        output_dir = self.data_dirs.get('outputs', 'data/outputs')

        # --- ML Explorer Visualizations (requires feature sets) ---
        if self.feature_sets:
            try:
                # Lazy import inside method
                from src.analysis.ml_explorer import MLExplorer
                explorer = MLExplorer(self.processed_data or {}, self.feature_sets, output_dir)
                # Pass dropped tiers info if available
                if self.dropped_tiers: explorer.dropped_tiers = self.dropped_tiers
                explorer.run_advanced_eda()
            except ImportError: self.logger.warning("MLExplorer not found or failed import. Skipping advanced EDA plots.")
            except Exception as e: self.logger.error(f"ML Explorer visualization error: {e}", exc_info=True)
        else:
            self.logger.info("Skipping ML Explorer visualizations: Feature sets not available.")

        # --- Basic Data Visualizations (requires processed data) ---
        if self.processed_data:
             try:
                # Lazy import inside method
                from src.analysis.visualizer import FantasyDataVisualizer
                visualizer = FantasyDataVisualizer(self.processed_data, self.feature_sets or {}, output_dir)
                # Pass cluster models if available from projection instance
                if self.projection_model_instance and hasattr(self.projection_model_instance, 'cluster_models'):
                    visualizer.cluster_models = self.projection_model_instance.cluster_models
                # Pass league settings
                if self.league_settings: visualizer.explore_league_settings(self.league_settings)
                # Generate general plots
                visualizer.explore_data_distributions()
                visualizer.explore_performance_trends()
                # Call cluster viz only if models exist
                if hasattr(visualizer, 'cluster_models') and visualizer.cluster_models:
                     visualizer.visualize_clusters(visualizer.feature_sets, visualizer.cluster_models)

             except ImportError: self.logger.warning("FantasyDataVisualizer not found or failed import. Skipping basic data plots.")
             except Exception as e: self.logger.error(f"Basic data visualization error: {e}", exc_info=True)
        else:
            self.logger.info("Skipping basic data visualizations: Processed data not available.")

        # --- Projection-Specific Visualizations ---
        if self.projections:
            self._create_projection_visualizations()
        else:
            self.logger.info("Skipping projection visualizations: Projections not available.")

        self.logger.info("Visualization generation attempt complete.")
        return self


    def _create_projection_visualizations(self):
        """Create visualizations specific to the generated projections."""
        self.logger.info("Creating projection visualizations...")
        proj_viz_dir = os.path.join(self.data_dirs.get('outputs', 'data/outputs'), 'projections')
        os.makedirs(proj_viz_dir, exist_ok=True)

        all_pos_dfs = [] # For combined plot

        for position in ['qb', 'rb', 'wr', 'te']:
            if position in self.projections and isinstance(self.projections[position], pd.DataFrame) and not self.projections[position].empty:
                pos_data = self.projections[position].copy() # Work on a copy
                pos_dir = os.path.join(proj_viz_dir, position); os.makedirs(pos_dir, exist_ok=True)

                if 'projected_points' not in pos_data.columns or 'name' not in pos_data.columns:
                     self.logger.warning(f"Skipping projection viz for {position}: Missing required columns."); continue

                try:
                    # --- Top Players Bar Chart ---
                    pos_data['projected_points'] = pd.to_numeric(pos_data['projected_points'], errors='coerce')
                    pos_data.dropna(subset=['projected_points'], inplace=True) # Ensure points are valid
                    sorted_data = pos_data.sort_values('projected_points', ascending=False)
                    top_n = min(30, len(sorted_data)); top_players = sorted_data.head(top_n)

                    if not top_players.empty:
                        plt.figure(figsize=(12, 8)); # Adjusted size
                        # Create bars with error bars for projection range
                        # Ensure range columns exist and are numeric
                        low_col, high_col = 'projection_low', 'projection_high'
                        has_range = low_col in top_players.columns and high_col in top_players.columns
                        if has_range:
                             top_players[low_col] = pd.to_numeric(top_players[low_col], errors='coerce').fillna(top_players['projected_points'])
                             top_players[high_col] = pd.to_numeric(top_players[high_col], errors='coerce').fillna(top_players['projected_points'])
                             errors = [top_players['projected_points'] - top_players[low_col],
                                       top_players[high_col] - top_players['projected_points']]
                             bars = plt.barh(top_players['name'], top_players['projected_points'],
                                             xerr=errors, capsize=3, # Add error bars
                                             color=sns.color_palette('viridis', len(top_players)), alpha=0.8)
                        else:
                             bars = plt.barh(top_players['name'], top_players['projected_points'],
                                             color=sns.color_palette('viridis', len(top_players)))

                        # Add point value labels
                        for i, bar in enumerate(bars):
                             plt.text(top_players.iloc[i]['projected_points'] + 0.3, bar.get_y() + bar.get_height()/2,
                                      f"{top_players.iloc[i]['projected_points']:.1f}", va='center', fontsize=8)

                        plt.title(f'Top {top_n} Projected {position.upper()} Players (with Range)', fontsize=14); # Updated title
                        plt.xlabel('Projected FPTS/Game'); plt.ylabel(''); # Removed Player label on Y
                        plt.gca().invert_yaxis(); plt.grid(axis='x', linestyle='--', alpha=0.6)
                        plt.tight_layout(); plt.savefig(os.path.join(pos_dir, f'top_players_range.png'), dpi=150); plt.close()
                        all_pos_dfs.append(pos_data) # Add to list for combined plot

                    # --- Distribution Plot ---
                    plt.figure(figsize=(10, 6));
                    sns.histplot(pos_data['projected_points'], kde=True, bins=20)
                    mean_proj = pos_data['projected_points'].mean()
                    plt.axvline(mean_proj, color='r', linestyle='--', label=f'Mean: {mean_proj:.2f}')
                    plt.title(f'{position.upper()} Projected Points Distribution', fontsize=14); plt.xlabel('Projected FPTS/Game'); plt.ylabel('Count'); plt.legend()
                    plt.tight_layout(); plt.savefig(os.path.join(pos_dir, f'distribution.png'), dpi=150); plt.close()

                except Exception as e: self.logger.error(f"Error creating projection viz for {position}: {e}", exc_info=True)
            else:
                self.logger.info(f"No projection data found for {position} to visualize.")

        # --- Combined Distribution Plot ---
        if all_pos_dfs:
             combined_df = pd.concat(all_pos_dfs, ignore_index=True)
             if not combined_df.empty and 'position' in combined_df.columns:
                  plt.figure(figsize=(12, 7))
                  sns.kdeplot(data=combined_df, x='projected_points', hue='position', fill=True, common_norm=False, palette='colorblind', alpha=0.5)
                  plt.title('Projected Points Distribution by Position', fontsize=14); plt.xlabel('Projected FPTS/Game'); plt.ylabel('Density'); plt.xlim(left=max(0, combined_df['projected_points'].min() - 1)) # Start near 0
                  plt.grid(True, alpha=0.3); plt.tight_layout(); plt.savefig(os.path.join(proj_viz_dir, f'combined_distribution.png'), dpi=150); plt.close()

        self.logger.info("Projection visualizations creation attempt complete.")


# --- Main Execution ---
if __name__ == '__main__':
    print("This script defines the ProjectionPipeline class and should be run via main2.py.")
    # Example usage (for testing)
    # config = load_config() # Assuming load_config is defined or imported
    # setup_logging(config)
    # pipeline = ProjectionPipeline(config)
    # projections = pipeline.run()
    # if projections:
    #     print("\nPipeline finished. Example projections:")
    #     for pos, df in projections.items():
    #         if isinstance(df, pd.DataFrame) and not df.empty:
    #             print(f"\n--- Top 5 {pos.upper()} ---")
    #             print(df[['name', 'projected_points']].head().round(2))