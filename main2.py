#!/usr/bin/env python3
"""
Fantasy Football Player Projection System and RL Draft Agent Training Orchestrator

This script orchestrates the player projection pipeline and optionally
trains a reinforcement learning agent for drafting.
"""

import sys
import yaml
import logging
import logging.config # Use dictConfig for more robust setup
import os
from datetime import datetime
from src.pipeline.projection_pipeline import ProjectionPipeline
from src.pipeline.rl_pipeline import RLPipeline
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    # Use print here, logging might not be configured yet
    print("INFO: setup_logging - Optuna not found, Optuna logging setup will be skipped.")
    optuna = None # Define optuna as None if import fails
    OPTUNA_AVAILABLE = False
import multiprocessing
RL_AVAILABLE = True


try:
    from src.analysis.tensorboard_parser import process_and_plot_tb_data
    TB_PARSER_AVAILABLE = True
except ImportError:
    logging.getLogger("fantasy_football").warning("tensorboard_parser script not found or failed import. Cannot generate plots from TB logs.")
    TB_PARSER_AVAILABLE = False



def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"ERROR: Configuration file not found at '{config_path}'")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Error loading config file '{config_path}': {e}")
        sys.exit(1)

# def setup_logging(config):
#     """
#     Set up logging based on configuration using dictConfig,
#     handling multiprocessing by creating separate log files for child processes,
#     and silencing Optuna logs.
#     """
#     log_config = config.get('logging', {})
#     log_level_file = log_config.get('level', 'INFO').upper() # Level for the file
#     log_file_base = log_config.get('file', 'fantasy_football.log') # Get base name from config
#     console_log_level = 'WARNING' # Hardcode console level

#     log_file = None # Final log file path
#     pid = os.getpid()
#     is_child_process = False

#     # --- Determine Log Filename Based on Process ---
#     if log_file_base:
#         try:
#             current_process = multiprocessing.current_process()
#             if current_process.name != 'MainProcess' or 'PoolWorker' in current_process.name or 'Fork' in current_process.name or 'Spawn' in current_process.name:
#                  is_child_process = True

#             if is_child_process:
#                 base, ext = os.path.splitext(log_file_base)
#                 log_file = f"{base}_pid{pid}{ext}"
#                 print(f"INFO: setup_logging - Child process detected (PID {pid}, Name: {current_process.name}), using separate log file: {log_file}")
#             else:
#                 log_file = log_file_base
#                 print(f"INFO: setup_logging - Main process (PID {pid}, Name: {current_process.name}), using primary log file: {log_file}")

#         except Exception as e:
#              print(f"WARNING: setup_logging - Could not determine child process status ({e}), using base log file: {log_file_base}")
#              log_file = log_file_base

#     # --- Prepare File Handler Info ---
#     file_handler_config = None
#     if log_file:
#         log_dir = os.path.dirname(log_file)
#         if log_dir and not os.path.exists(log_dir):
#             try:
#                 os.makedirs(log_dir)
#                 print(f"INFO: setup_logging - Created log directory: {log_dir}")
#             except OSError as e:
#                 log_file = None
#                 print(f"ERROR: setup_logging - Creating log dir '{log_dir}' failed: {e}. Disabling file log.")

#         if log_file:
#              write_target_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else '.'
#              if not os.access(write_target_dir, os.W_OK):
#                  log_file = None
#                  print(f"ERROR: setup_logging - Cannot write to log directory '{write_target_dir}'. Disabling file log.")

#         if log_file:
#             file_handler_config = {
#                 'level': log_level_file,
#                 'class': 'logging.FileHandler',
#                 'formatter': 'standard',
#                 'filename': log_file,
#                 'mode': 'a',
#             }
#     else:
#          print("INFO: setup_logging - File logging disabled.")


#     # --- Define Logging Configuration Dictionary ---
#     logging_dict = {
#         'version': 1,
#         'disable_existing_loggers': False,
#         'formatters': {
#             'standard': {
#                 'format': '%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s',
#                 'datefmt': '%Y-%m-%d %H:%M:%S',
#             },
#         },
#         'handlers': {
#             'console': {
#                 'level': console_log_level,
#                 'class': 'logging.StreamHandler',
#                 'formatter': 'standard',
#                 'stream': sys.stdout,
#             },
#             **({'file': file_handler_config} if file_handler_config else {}),
#             'null': {
#                 'class': 'logging.NullHandler',
#             },
#         },
#         'loggers': {
#             'fantasy_football': {
#                 'handlers': ['console'] + (['file'] if file_handler_config else []),
#                 'level': log_level_file,
#                 'propagate': False,
#             },
#             'fantasy_draft_rl': {
#                 'handlers': ['console'] + (['file'] if file_handler_config else []),
#                 'level': log_level_file,
#                 'propagate': False,
#             },
#              'nfl_data_py': {'level': 'WARNING', 'propagate': False, 'handlers': ['console'] + (['file'] if file_handler_config else [])},
#              'matplotlib': {'level': 'WARNING', 'propagate': False, 'handlers': ['console'] + (['file'] if file_handler_config else [])},
#              'stable_baselines3': {'level': 'WARNING', 'propagate': False, 'handlers': ['console'] + (['file'] if file_handler_config else [])},
#              'joblib': {'level': 'WARNING', 'propagate': False, 'handlers': ['console'] + (['file'] if file_handler_config else [])},
#              # Configure Optuna logger - will use 'null' handler if OPTUNA_AVAILABLE is False
#              'optuna': {
#                   'handlers': ['null'], # Always send to null initially
#                   'level': 'CRITICAL', # Set level high
#                   'propagate': False,
#              },
#         },
#         'root': {
#             'level': 'CRITICAL',
#             'handlers': [],
#         }
#     }

#     # --- Apply dictConfig ---
#     try:
#         logging.config.dictConfig(logging_dict)
#         final_log_file_path = log_file if file_handler_config else "Disabled"
#         print(f"INFO: setup_logging - Initial logging config applied. Console: {console_log_level}, File: {log_level_file}, Log File: {final_log_file_path} (PID: {pid})")
#     except Exception as e:
#         print(f"FATAL: setup_logging - Failed to configure logging using dictConfig: {e}")
#         logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
#         logging.error("Logging configuration failed, using basic WARNING level output.")
#         return logging.getLogger("fantasy_football")


#     # --- Apply Optuna-specific logging controls AFTER dictConfig ---
#     # Now OPTUNA_AVAILABLE is defined within this function's scope
#     if OPTUNA_AVAILABLE and optuna:
#         try:
#             optuna.logging.disable_default_handler()
#             print("INFO: setup_logging - Disabled Optuna's default logging handler.")
#             optuna.logging.set_verbosity(optuna.logging.CRITICAL)
#             print("INFO: setup_logging - Set Optuna's internal verbosity to CRITICAL.")

#             optuna_logger = logging.getLogger("optuna")
#             for handler in optuna_logger.handlers[:]:
#                 optuna_logger.removeHandler(handler)
#             optuna_logger.addHandler(logging.NullHandler())
#             optuna_logger.setLevel(logging.CRITICAL)
#             optuna_logger.propagate = False
#             print(f"INFO: setup_logging - Optuna logger explicitly configured to discard messages.")

#         except Exception as e:
#             print(f"WARNING: setup_logging - Could not apply Optuna-specific logging controls: {e}")

#     # Return the main application logger instance
#     logger = logging.getLogger("fantasy_football")
#     logger.debug(f"Setup Logging: DEBUG message test (PID {pid}, File only if level=DEBUG)")
#     logger.info(f"--- Logging setup complete (PID {pid}) ---")
#     logger.warning(f"--- Logging setup complete (PID {pid} - WARNING test) ---")

#     return logger


def setup_logging(config):
    """
    Set up logging based on configuration using dictConfig.
    This version is intended to be called by the MAIN process.
    Child processes should ideally use a different configuration or method.
    """
    log_config = config.get('logging', {})
    log_level_file = log_config.get('level', 'INFO').upper()
    log_file_base = log_config.get('file', 'fantasy_football.log') # Base file name
    console_log_level = 'WARNING'

    pid = os.getpid() # Get PID of the process calling this function
    log_file = log_file_base # Use the base name by default

    # --- Prepare File Handler Info ---
    file_handler_config = None
    if log_file: # Only proceed if a base log file is specified
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            try:
                os.makedirs(log_dir)
                print(f"INFO: setup_logging (PID {pid}) - Created log directory: {log_dir}")
            except OSError as e:
                log_file = None
                print(f"ERROR: setup_logging (PID {pid}) - Creating log dir '{log_dir}' failed: {e}. Disabling file log.")

        if log_file:
             write_target_dir = os.path.dirname(log_file) if os.path.dirname(log_file) else '.'
             if not os.access(write_target_dir, os.W_OK):
                 log_file = None
                 print(f"ERROR: setup_logging (PID {pid}) - Cannot write to log directory '{write_target_dir}'. Disabling file log.")

        if log_file:
            file_handler_config = {
                'level': log_level_file,
                'class': 'logging.FileHandler', # Standard handler
                'formatter': 'standard',
                'filename': log_file,       # Writes to the base log file
                'mode': 'a',
            }
    else:
         print(f"INFO: setup_logging (PID {pid}) - File logging disabled.")

    # --- Define Logging Configuration Dictionary ---
    logging_dict = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                 # Include PID here to know which process wrote which message IF they share file
                'format': '%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s',
                'datefmt': '%Y-%m-%d %H:%M:%S',
            },
        },
        'handlers': {
            'console': {
                'level': console_log_level,
                'class': 'logging.StreamHandler',
                'formatter': 'standard',
                'stream': sys.stdout,
            },
            **({'file': file_handler_config} if file_handler_config else {}),
            'null': { 'class': 'logging.NullHandler' },
        },
        'loggers': {
            # Configure loggers to use handlers
            'fantasy_football': { 'handlers': ['console'] + (['file'] if file_handler_config else []), 'level': log_level_file, 'propagate': False },
            'fantasy_draft_rl': { 'handlers': ['console'] + (['file'] if file_handler_config else []), 'level': log_level_file, 'propagate': False },
            'nfl_data_py': { 'handlers': ['console'] + (['file'] if file_handler_config else []), 'level': 'WARNING', 'propagate': False },
            'matplotlib': { 'handlers': ['console'] + (['file'] if file_handler_config else []), 'level': 'WARNING', 'propagate': False },
            'stable_baselines3': { 'handlers': ['console'] + (['file'] if file_handler_config else []), 'level': 'WARNING', 'propagate': False },
            'joblib': { 'handlers': ['console'] + (['file'] if file_handler_config else []), 'level': 'WARNING', 'propagate': False },
            'optuna': { 'handlers': ['null'], 'level': 'CRITICAL', 'propagate': False },
        },
        'root': { 'level': 'CRITICAL', 'handlers': [] } # Keep root clean
    }

    # --- Apply dictConfig ---
    try:
        logging.config.dictConfig(logging_dict)
        final_log_file_path = log_file if file_handler_config else "Disabled"
        print(f"INFO: setup_logging (PID {pid}) - Logging configured. Console: {console_log_level}, File: {log_level_file}, Log File: {final_log_file_path}")
    except Exception as e:
        print(f"FATAL: setup_logging (PID {pid}) - Failed to configure logging: {e}")
        logging.basicConfig(level=logging.WARNING, format='%(levelname)s: %(message)s')
        logging.error("Logging configuration failed, using basic WARNING level output.")
        return logging.getLogger("fantasy_football")

    # --- Apply Optuna-specific logging controls ---
    if OPTUNA_AVAILABLE and optuna:
        try:
            optuna.logging.disable_default_handler()
            optuna.logging.set_verbosity(optuna.logging.CRITICAL)
            optuna_logger = logging.getLogger("optuna")
            for handler in optuna_logger.handlers[:]: optuna_logger.removeHandler(handler)
            optuna_logger.addHandler(logging.NullHandler())
            optuna_logger.propagate = False
            print(f"INFO: setup_logging (PID {pid}) - Optuna logging silenced.")
        except Exception as e:
            print(f"WARNING: setup_logging (PID {pid}) - Could not apply Optuna logging controls: {e}")

    logger = logging.getLogger("fantasy_football")
    logger.info(f"--- Logging setup complete (PID {pid}) ---")
    return logger

def print_summary(pipeline_instance):
    """Print a human-readable summary of the projection results"""
    if not pipeline_instance or not hasattr(pipeline_instance, 'get_summary'):
        print("\nNo valid projection pipeline instance to summarize.")
        return

    try:
        summary = pipeline_instance.get_summary()
    except Exception as e:
        logging.getLogger("fantasy_football").error(f"Error getting summary from pipeline: {e}")
        print("\nError generating projection summary.")
        return

    print("\n========== Fantasy Football Projections Summary ==========")
    print(f"League: {summary.get('league', 'N/A')}")
    print(f"Teams: {summary.get('teams', 'N/A')}")
    print(f"Projection year: {summary.get('projection_year', 'N/A')}")
    print(f"Used hierarchical projections: {summary.get('used_hierarchical', 'N/A')}")
    print(f"Used NGS data: {summary.get('used_ngs', 'N/A')}")

    top_players = summary.get('top_players', {})
    for position in ['qb', 'rb', 'wr', 'te']:
        players = top_players.get(position)
        if players and isinstance(players, list):
            print(f"\nTop 5 projected {position.upper()} players:")
            for player in players:
                if isinstance(player, dict):
                    name = player.get('name', 'Unknown Player')
                    points = player.get('points', 0.0)
                    try: points_f = float(points); print(f"  {name}: {points_f:.2f} pts/game")
                    except (ValueError, TypeError): print(f"  {name}: {points} pts/game (Invalid points format)")
                else: print(f"  Invalid player data format for {position}")

    print(f"\nProjections completed successfully!")
    print(f"Output data, projections, and models saved to configured directories.")


def main():
    config = load_config()
    logger = setup_logging(config)
    logger.info("===== Starting Fantasy Football System =====")

    # --- (Projection Pipeline logic as before) ---
    logger.info("--- Running Main Projection Pipeline ---")
    main_projections = None; projection_pipeline = None
    try:
        # Dynamically get class if needed
        if 'projection_pipeline_class' not in locals(): from src.pipeline.projection_pipeline import ProjectionPipeline as projection_pipeline_class
        projection_pipeline = projection_pipeline_class(config)
        main_projections = projection_pipeline.run()
    except Exception as e: logger.exception("Projection Pipeline failed."); print("ERROR: Projection Pipeline failed."); return 1
    if main_projections is None: logger.error("Main projection pipeline failed. Aborting."); return 1
    logger.info("--- Main Projection Pipeline Finished ---")
    if projection_pipeline: print_summary(projection_pipeline)


    # --- Run RL ---
    rl_config = config.get('rl_training', {})
    run_main_rl = rl_config.get('enabled', False)
    rl_pipeline_class = None # Reset class variable
    RL_AVAILABLE = False
    try: # Check RL deps again just before using
        from src.pipeline.rl_pipeline import RLPipeline, SB3_AVAILABLE, FantasyDraftEnv
        if SB3_AVAILABLE and FantasyDraftEnv is not None: rl_pipeline_class = RLPipeline; RL_AVAILABLE = True
    except ImportError: logger.warning("RLPipeline module/deps not found.")

    if run_main_rl and RL_AVAILABLE and rl_pipeline_class:
         logger.info(f"--- Running Main RL Training/Evaluation ---")
         tensorboard_log_dir = None # Variable to store the actual log path
         try:
              rl_pipeline_main = rl_pipeline_class(config) # Pass config
              rl_pipeline_main.run() # This runs training & eval
              # *** Store the path used by the pipeline ***
              tensorboard_log_dir = rl_config.get('tensorboard_log_path', "./ppo_fantasydraft_tensorboard/")
              # Adjust if the run modifies the base path (e.g., adds PPO_DraftAgent_1)
              # This might require inspecting the actual directory structure SB3 creates
              # Example: Find the latest PPO run directory inside the base path
              try:
                  run_dirs = [d for d in os.listdir(tensorboard_log_dir) if d.startswith("PPO_") and os.path.isdir(os.path.join(tensorboard_log_dir, d))]
                  if run_dirs:
                       latest_run_dir = max(run_dirs, key=lambda d: os.path.getmtime(os.path.join(tensorboard_log_dir, d)))
                       tensorboard_log_dir = os.path.join(tensorboard_log_dir, latest_run_dir)
                       logger.info(f"Found specific run log directory: {tensorboard_log_dir}")
              except FileNotFoundError:
                   logger.warning(f"Base TensorBoard dir not found: {tensorboard_log_dir}. Cannot find specific run.")
                   tensorboard_log_dir = None # Reset if not found
              except Exception as find_err:
                   logger.warning(f"Error finding specific run dir: {find_err}. Using base path.")
                   tensorboard_log_dir = rl_config.get('tensorboard_log_path', "./ppo_fantasydraft_tensorboard/")


              logger.info(f"--- Finished Main RL Run ---")
         except Exception as e: logger.exception("Main RL Pipeline failed.")

         # --- Generate Plots from TensorBoard Data ---
         # Check if the parser is available and we have a log directory
         if TB_PARSER_AVAILABLE and tensorboard_log_dir and os.path.isdir(tensorboard_log_dir):
              output_dir = config.get('paths', {}).get('output_dir', 'data/outputs')
              csv_path = os.path.join(output_dir, "rl_training_data_from_tb.csv")
              plot_path = os.path.join(output_dir, "rl_training_plots_from_tb.png")
              logger.info("Attempting to process TensorBoard logs for custom plotting...")
              process_and_plot_tb_data(tensorboard_log_dir, csv_path, plot_path)
         elif not TB_PARSER_AVAILABLE:
              logger.warning("Skipping TB plot generation: tbparse not available.")
         else:
              logger.warning(f"Skipping TB plot generation: Log directory not found or invalid: {tensorboard_log_dir}")

    elif run_main_rl and (not RL_AVAILABLE or rl_pipeline_class is None):
         logger.warning("RL training enabled, but dependencies missing or class failed import.")
    else:
        logger.info("Main RL training/evaluation disabled.")


    logger.info("===== Fantasy Football System Finished Successfully =====")
    return 0

if __name__ == "__main__":
    main_exit_code = 0
    try: main_exit_code = main()
    except SystemExit as e: main_exit_code = e.code if isinstance(e.code, int) else 1
    except Exception as e: print(f"FATAL ERROR in main: {e}"); logging.exception("Fatal error:"); main_exit_code = 1
    finally: logging.shutdown(); sys.exit(main_exit_code)