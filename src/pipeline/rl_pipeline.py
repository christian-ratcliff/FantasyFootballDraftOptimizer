#!/usr/bin/env python3
"""
Reinforcement Learning Pipeline for training the Fantasy Football Draft Agent.
Includes Optuna hyperparameter optimization.
Loads projections directly from individual position CSV files.
Includes Monitor wrapper for proper episode logging to TensorBoard for the final run.
Relies on external script/TensorBoard for plotting.
ADDED DEBUG LOGGING around PPO initialization.
"""
import warnings

# Filter out only the specific warning about mlp_extractor
warnings.filterwarnings(
    "ignore", 
    message="As shared layers in the mlp_extractor are removed since SB3 v1.8.0.*", 
    category=UserWarning
)
from tqdm import tqdm

import os
import signal
import logging
import yaml
import json
import joblib
import pandas as pd
import logging.config
import numpy as np
import random
import gymnasium as gym
import time
from datetime import datetime # For Optuna study name timestamp
import sys # For memory usage check
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
import multiprocessing # To get CPU count
import torch
import traceback
import concurrent.futures
import psutil

_GLOBAL_INSTANCE = None


def kill_all_children_and_exit(sig=None, frame=None):
    """Kill all child processes and exit forcefully"""
    print("\n\nFORCE TERMINATING ALL PROCESSES")
    
    # Get the current process
    current_process = psutil.Process(os.getpid())
    
    # Kill all child processes
    for child in current_process.children(recursive=True):
        try:
            print(f"Killing child process {child.pid}")
            child.kill()
        except:
            pass
    
    # Exit immediately without cleanup
    os._exit(1)

def force_exit_handler(sig, frame):
    print("\nForce exiting the program")
    # This bypasses normal exit procedures and immediately kills the process
    os._exit(1) 

# --- Optuna Import ---
OPTUNA_AVAILABLE = False
try:
    import optuna
    from optuna.exceptions import TrialPruned # Import TrialPruned for callback
    OPTUNA_AVAILABLE = True
except ImportError:
    logging.getLogger("fantasy_draft_rl").warning("Optuna not installed. Run 'pip install optuna' to enable hyperparameter optimization.")
    optuna = None # Set to None if not available

# --- Stable Baselines 3 Imports ---
SB3_AVAILABLE = True
try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.env_checker import check_env
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
except ImportError:
    SB3_AVAILABLE = False
    logging.getLogger("fantasy_draft_rl").error("Stable Baselines 3 not installed. RL Pipeline disabled.")
    class BaseCallback: pass
    PPO = None; DummyVecEnv = None; check_env = None; Monitor = None; CheckpointCallback = None; CallbackList = None

# --- Custom Environment Import ---
try:
    from src.models.rl_draft_agent import FantasyDraftEnv, NUM_ACTIONS, ACTION_BEST_AVAILABLE
except ImportError:
     FantasyDraftEnv = None
     NUM_ACTIONS = 6
     ACTION_BEST_AVAILABLE = 5
     SB3_AVAILABLE = False
     logging.getLogger("fantasy_draft_rl").error("Could not import FantasyDraftEnv or constants. RL Pipeline disabled.")

# --- Custom Callback Import ---
# Assuming callback is in src/models/callback.py based on user's import
try:
    from src.models.callback import TrialEvalCallback
except ImportError:
     TrialEvalCallback = None # Set to None if import fails
     logging.getLogger("fantasy_draft_rl").warning("TrialEvalCallback not found or failed import. Optuna HPO might not function correctly.")


# --- Helper Functions ---
def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    logger_init = logging.getLogger("fantasy_football_init")
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        return config_data
    except FileNotFoundError:
        logger_init.error(f"Config file not found: {config_path}")
        raise
    except Exception as e:
        logger_init.error(f"Error loading config file {config_path}: {e}")
        raise

def load_league_settings(config_path='configs/league_settings.json'):
    """Load detailed league settings from JSON file"""
    logger_init = logging.getLogger("fantasy_football_init")
    try:
        with open(config_path, 'r') as f:
            settings = json.load(f)
        logger_init.info(f"Loaded league settings from {config_path}")
        if 'league_info' not in settings or 'starter_limits' not in settings:
             raise ValueError("League settings JSON missing required keys ('league_info', 'starter_limits').")
        if 'team_count' not in settings['league_info']:
             raise ValueError("League settings JSON missing 'team_count' in 'league_info'.")
        return settings
    except FileNotFoundError:
         logger_init.error(f"League settings file not found: {config_path}")
         raise
    except json.JSONDecodeError as e:
         logger_init.error(f"Error decoding JSON from league settings file {config_path}: {e}")
         raise
    except ValueError as e:
         logger_init.error(f"Invalid league settings structure in {config_path}: {e}")
         raise
    except Exception as e:
        logger_init.error(f"Unexpected error loading league settings file {config_path}: {e}")
        raise

def load_projections_from_csvs(models_dir, league_settings, evaluation_year=None):
    """Loads projections dict from year-specific CSV files."""
    rl_logger = logging.getLogger("fantasy_draft_rl"); player_map = league_settings.get('player_map', {})
    if evaluation_year: file_suffix = f"_{evaluation_year}.csv"; load_desc = f"hist year {evaluation_year}"
    else: file_suffix = ".csv"; load_desc = "main proj year"
    file_path_base = os.path.join(models_dir, "top_{pos}_projections" + file_suffix)
    rl_logger.info(f"Load projections for {load_desc} from CSVs in: {models_dir} (suffix: '{file_suffix}')")
    all_projections_list = []; positions = ['qb', 'rb', 'wr', 'te']
    essential_cols = ['player_id', 'name', 'position', 'projected_points', 'age', 'projection_low', 'projection_high', 'ceiling_projection']
    for pos in positions:
        file_path = file_path_base.format(pos=pos); rl_logger.debug(f"Checking for file: {file_path}")
        if not os.path.exists(file_path): rl_logger.warning(f"File not found: {file_path}. Skip {pos}."); continue
        try:
            if os.path.getsize(file_path) == 0: rl_logger.warning(f"File empty: {file_path}. Skip {pos}."); continue
        except OSError as e: rl_logger.error(f"Cannot get size: {file_path}: {e}. Skip {pos}."); continue
        try:
            df = pd.read_csv(file_path);
            if df.empty: rl_logger.warning(f"{file_path} loaded empty."); continue
            df_processed = df.copy(); missing = [c for c in essential_cols if c not in df_processed.columns]
            if missing: rl_logger.error(f"{file_path} missing {missing}. Skip {pos}."); continue
            df_processed['position'] = df_processed['position'].astype(str).str.upper();
            if (df_processed['position'] != pos.upper()).any(): df_processed['position'] = pos.upper()
            if 'name' not in df_processed or df_processed['name'].isnull().any():
                 df_processed['player_id_str'] = df_processed['player_id'].astype(str)
                 df_processed['name'] = df_processed['player_id_str'].map(player_map).fillna("Unknown_" + df_processed['player_id_str'])
                 df_processed.drop(columns=['player_id_str'], inplace=True)
            else: df_processed['name'] = df_processed['name'].astype(str)
            df_processed['age'] = pd.to_numeric(df_processed['age'], errors='coerce').fillna(27).astype(int)
            try:
                df_processed['player_id'] = pd.to_numeric(df_processed['player_id'], errors='coerce'); df_processed['projected_points'] = pd.to_numeric(df_processed['projected_points'], errors='coerce')
                df_processed.dropna(subset=['player_id', 'projected_points'], inplace=True)
                if df_processed.empty: rl_logger.warning(f"Empty DataFrame after dropping NaNs for {pos}."); continue
                df_processed['player_id'] = df_processed['player_id'].astype(int)
                for col in ['projection_low','projection_high','ceiling_projection']: df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
                df_processed['projection_low'].fillna(df_processed['projected_points'] * 0.8, inplace=True); df_processed['projection_high'].fillna(df_processed['projected_points'] * 1.2, inplace=True); df_processed['ceiling_projection'].fillna(df_processed['projected_points'] * 1.5, inplace=True)
                for col in ['projected_points','projection_low','projection_high','ceiling_projection']:
                    if col in df_processed.columns: df_processed[col] = df_processed[col].astype(float)
            except Exception as type_err: rl_logger.error(f"Type convert err {pos}: {type_err}. Skip."); continue
            all_projections_list.append(df_processed[essential_cols])
        except Exception as e: rl_logger.error(f"Error processing {file_path}: {e}", exc_info=True)
    if not all_projections_list: rl_logger.error(f"No valid DFs loaded for {load_desc}."); return pd.DataFrame()
    try:
        final_df = pd.concat(all_projections_list, ignore_index=True, join='inner');
        if final_df.empty: rl_logger.error("Concat empty DF."); return pd.DataFrame()
        final_df['combined_score'] = (0.5*final_df['projected_points'] + 0.2*final_df['projection_low'] + 0.3*final_df['ceiling_projection']).fillna(0.0)
        final_df = final_df.sort_values('combined_score', ascending=False).reset_index(drop=True)
        rl_logger.info(f"Loaded & combined projections for {load_desc} ({len(final_df)} players).")
        # DEBUG: Log memory usage of the final DataFrame
        try:
            mem_usage_mb = final_df.memory_usage(deep=True).sum() / (1024 * 1024)
            rl_logger.info(f"Projections DataFrame memory usage: {mem_usage_mb:.2f} MB")
        except Exception:
            rl_logger.warning("Could not determine projection DataFrame memory usage.")
        return final_df
    except Exception as concat_err: rl_logger.error(f"Final concat/cleanup err: {concat_err}", exc_info=True); return pd.DataFrame()


def _init_worker(instance_config, study_name, storage_path):
    """Initialize each worker process with the necessary data"""
    global _GLOBAL_INSTANCE
    
    
    # Create a new instance with the same config
    _GLOBAL_INSTANCE = RLPipeline(config=instance_config)
    
    # Set study information
    _GLOBAL_INSTANCE._current_study_name = study_name
    _GLOBAL_INSTANCE._current_storage_path = storage_path
    

def _run_optuna_trial_in_process(trial_id):
    """Standalone function that runs a single Optuna trial in a separate process."""
    global _GLOBAL_INSTANCE
    
    pid = os.getpid()
    
    if _GLOBAL_INSTANCE is None:
        return trial_id, float("inf"), pid
    
    # Get access to the instance's configuration and logger
    self = _GLOBAL_INSTANCE
    
    try:
        # Create a trial and run it
        study = optuna.load_study(
            study_name=self._current_study_name,
            storage=self._current_storage_path
        )
        
        trial = study.ask()
        
        # Run the objective function
        value = self._optuna_objective(trial)
        
        # Report the result
        study.tell(trial, value)
        
        return trial_id, value, pid
        
    except Exception as e:
        traceback.print_exc()
        return trial_id, float("inf"), pid

# --- RLPipeline Class ---
class RLPipeline:
    """Handles the RL training process, including Optuna HPO."""

    def __init__(self, config):
        self.config = config
        self.logger = logging.getLogger("fantasy_football")
        self.rl_logger = logging.getLogger("fantasy_draft_rl")
        if not SB3_AVAILABLE or FantasyDraftEnv is None:
            self.logger.error("Missing RL dependencies or FantasyDraftEnv. RL Pipeline is disabled.")
            raise ImportError("RL dependencies or FantasyDraftEnv missing.")
        if OPTUNA_AVAILABLE and TrialEvalCallback is None:
             self.rl_logger.warning("Optuna is available but TrialEvalCallback failed to import. Optuna HPO will likely fail.")

    def _setup_worker_logging(self, config, pid):
        """Configures logging specifically for a worker process."""
        log_config = config.get('logging', {})
        log_level_file = log_config.get('level', 'INFO').upper()
        log_file_base = log_config.get('file', 'fantasy_football.log')
        console_log_level = 'WARNING' # Keep console less noisy for workers

        # --- Create PID-specific filename in PID_logs folder ---
        log_file = None
        if log_file_base:
            # Determine the base path and filename
            orig_log_dir = os.path.dirname(log_file_base)
            base_filename = os.path.basename(log_file_base)
            base, ext = os.path.splitext(base_filename)
            
            # Create the PID_logs directory path
            pid_logs_dir = os.path.join(orig_log_dir, "PID_logs") if orig_log_dir else "PID_logs"
            
            # Ensure PID_logs directory exists
            os.makedirs(pid_logs_dir, exist_ok=True)
            
            # Create the full path for the PID-specific log file
            log_file = os.path.join(pid_logs_dir, f"{base}_pid{pid}{ext}")
        # -----------------------------------------------------

        # Define worker's file handler config
        worker_file_handler_config = None
        if log_file:
            # Minimal check for writability
            write_target_dir = os.path.dirname(log_file)
            if os.access(write_target_dir, os.W_OK):
                worker_file_handler_config = {
                'level': log_level_file,
                'class': 'logging.FileHandler',
                'formatter': 'standard',
                'filename': log_file,
                'mode': 'a',
                }
            else:
                print(f"ERROR: Worker (PID {pid}) cannot write to log directory '{write_target_dir}'. Disabling file log for worker.")

        # Define the logging dict *for the worker*
        worker_logging_dict = {
            'version': 1,
            'disable_existing_loggers': False, # Let it use existing logger names
            'formatters': { # Define formatter needed by handlers
                'standard': {
                    'format': '%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s',
                    'datefmt': '%Y-%m-%d %H:%M:%S',
                },
            },
            'handlers': {
                # Console handler (optional for workers, could be removed to reduce clutter)
                'console': {
                    'level': console_log_level,
                    'class': 'logging.StreamHandler',
                    'formatter': 'standard',
                    'stream': sys.stdout, # Or sys.stderr
                },
                # Worker-specific file handler
                 **({'worker_file': worker_file_handler_config} if worker_file_handler_config else {}),
                 # Null handler for Optuna/other libraries if needed
                 'null': {'class': 'logging.NullHandler'},
            },
            'loggers': {
                 # Configure ONLY the loggers used within the worker/env/SB3
                 # Route them to the worker's specific file handler + console
                 'fantasy_draft_rl': {
                     'handlers': ['console'] + (['worker_file'] if worker_file_handler_config else []),
                     'level': log_level_file, # Use detailed level for worker file
                     'propagate': False,
                 },
                 # Silence Optuna within worker too
                 'optuna': {'handlers': ['null'], 'level': 'CRITICAL', 'propagate': False},
                 # Optionally configure SB3 logger for worker if needed
                 'stable_baselines3': {'handlers': ['console'] + (['worker_file'] if worker_file_handler_config else []), 'level': 'WARNING', 'propagate': False},
                 # Add other loggers used inside the objective if necessary
            },
            # Keep root minimal in worker too
            'root': {'level': 'CRITICAL', 'handlers': []}
        }
        try:
            logging.config.dictConfig(worker_logging_dict)
        except Exception as e:
            # Fallback to basic config for this worker if dictConfig fails
            logging.basicConfig(level=logging.WARNING, format=f'%(asctime)s - {pid} - %(name)s - %(levelname)s - %(message)s')

    

    # Optuna Objective Function
    def _optuna_objective(self, trial) -> float:
        """Objective function for Optuna HPO. Returns negative mean reward."""
        pid = os.getpid() # Get current worker PID
        # --- Configure Worker Logging ---
        self._setup_worker_logging(self.config, pid)
        # Now use self.rl_logger safely within this worker
        self.rl_logger.info(f"--- Starting Optuna Trial {trial.number} (PID {pid}) ---")
        self.rl_logger.info(f"\n--- Starting Optuna Trial {trial.number} ---")
        
        # --- Set PyTorch Threads  ---
        try:
            num_threads = 1
            torch.set_num_threads(num_threads)
            # Optional but good practice: Limit inter-op parallelism too
            # torch.set_num_interop_threads(num_threads)
            self.rl_logger.debug(f"Optuna Trial {trial.number}: Set torch.set_num_threads({num_threads})")
        except Exception as torch_err:
            self.rl_logger.error(f"Optuna Trial {trial.number}: Failed to set PyTorch threads: {torch_err}")
        # --------------------------------------------------------
        trial_env = None # For finally block
        eval_trial_env = None # For finally block
        model = None # For logging params

        # Check if TrialEvalCallback is available before proceeding
        if TrialEvalCallback is None:
             self.rl_logger.error("TrialEvalCallback is not available. Cannot run Optuna trial.")
             return float('inf') # Return high penalty

        try:
            # --- Load Data ---
            league_settings_path = self.config['paths']['config_path']
            league_settings = load_league_settings(league_settings_path)
            models_dir = self.config['paths']['models_dir']
            projections_df = load_projections_from_csvs(models_dir, league_settings)
            if projections_df.empty:
                self.rl_logger.error(f"Optuna Trial {trial.number}: Failed to load projections. Returning high penalty.")
                return float('inf')

            # --- Suggest Hyperparameters ---
            optuna_cfg = self.config.get('rl_training', {}).get('optuna', {}); search_space = optuna_cfg.get('search_space', {})
            ppo_params = self.config.get('rl_training', {}).get('ppo_hyperparams', {}).copy()
            tuned_params = {}
            for param, settings in search_space.items():
                 if param == 'policy_kwargs': continue
                 try:
                     if settings['type'] == 'loguniform': tuned_params[param] = trial.suggest_float(param, settings['low'], settings['high'], log=True)
                     elif settings['type'] == 'uniform': tuned_params[param] = trial.suggest_float(param, settings['low'], settings['high'])
                     elif settings['type'] == 'int': tuned_params[param] = trial.suggest_int(param, settings['low'], settings['high'])
                     elif settings['type'] == 'categorical': tuned_params[param] = trial.suggest_categorical(param, settings['choices'])
                 except Exception as suggest_err:
                     self.rl_logger.warning(f"Optuna Trial {trial.number}: Error suggesting param '{param}': {suggest_err}. Using default.")
            ppo_params.update(tuned_params)

            if 'policy_kwargs_net_arch' in search_space:
                 try:
                     arch_choice = trial.suggest_categorical('policy_kwargs_net_arch', search_space['policy_kwargs_net_arch']['choices'])
                     pi_str, vf_str = arch_choice.split('_')
                     pi_layers = [int(x) for x in pi_str.split('=')[1].split(',')]
                     vf_layers = [int(x) for x in vf_str.split('=')[1].split(',')]
                     ppo_params['policy_kwargs'] = dict(net_arch=[dict(pi=pi_layers, vf=vf_layers)])
                 except Exception as arch_err:
                     self.rl_logger.warning(f"Optuna Trial {trial.number}: Error parsing net_arch '{arch_choice}': {arch_err}. Using default policy_kwargs.")
                     ppo_params['policy_kwargs'] = self.config.get('rl_training', {}).get('ppo_hyperparams', {}).get('policy_kwargs')
            elif 'policy_kwargs' not in ppo_params:
                ppo_params['policy_kwargs'] = None

            self.rl_logger.info(f"Optuna Trial {trial.number} Sampled Parameters: {ppo_params}")

            # --- Environment Setup for Trial ---
            agent_draft_pos = self.config.get('rl_training', {}).get('agent_draft_position', 1)
            try:
                # DEBUG: Log before creating env factory
                self.rl_logger.debug(f"Optuna Trial {trial.number}: Defining make_trial_env_func...")
                def make_trial_env_func():
                     self.rl_logger.debug(f"Optuna Trial {trial.number}: make_trial_env_func called - Creating FantasyDraftEnv instance...")
                     # Pass copy to prevent modifications across processes/trials
                     env_instance = FantasyDraftEnv(projections_df.copy(), league_settings, agent_draft_pos)
                     self.rl_logger.debug(f"Optuna Trial {trial.number}: FantasyDraftEnv instance created.")
                     return env_instance

                self.rl_logger.debug(f"Optuna Trial {trial.number}: Creating DummyVecEnv...")
                trial_env = DummyVecEnv([make_trial_env_func]) # Training env VecEnv
                self.rl_logger.debug(f"Optuna Trial {trial.number}: DummyVecEnv created.")
                self.rl_logger.debug(f"Optuna Trial {trial.number}: Creating evaluation env instance...")
                eval_trial_env = make_trial_env_func() # Separate instance for eval callback
                self.rl_logger.debug(f"Optuna Trial {trial.number}: Evaluation env instance created.")
                self.rl_logger.debug(f"Optuna Trial {trial.number}: Environments setup complete.")
            except Exception as env_err:
                self.rl_logger.error(f"Optuna Trial {trial.number}: Failed to create environments: {env_err}", exc_info=True)
                return float('inf')
            
            if 'policy_kwargs_net_arch' in ppo_params:
                del ppo_params['policy_kwargs_net_arch']
                self.rl_logger.debug("Removed temporary 'policy_kwargs_net_arch' key from params before PPO init.")

            # --- Define Model for Trial ---
            try:
                 self.rl_logger.debug(f"Optuna Trial {trial.number}: >>> BEFORE PPO Initialization <<<")
                 # --- TEMPORARY DEBUG --- Add a small sleep
                 # time.sleep(0.5)
                 # --- END TEMPORARY DEBUG ---
                 model = PPO(policy="MlpPolicy", env=trial_env, verbose=2, # Set verbose=0 for Optuna trials
                            tensorboard_log=None, # Disable TB logging for individual trials
                             **ppo_params)
                 self.rl_logger.debug(f"Optuna Trial {trial.number}: >>> AFTER PPO Initialization <<<")
                 self.rl_logger.debug(f"Optuna Trial {trial.number}: PPO model defined successfully.")
            except Exception as model_err:
                 self.rl_logger.error(f"Optuna Trial {trial.number}: Failed during PPO model definition: {model_err}", exc_info=True)
                 # Attempt to close envs even if model definition fails
                 if trial_env: trial_env.close()
                 if eval_trial_env: eval_trial_env.close()
                 return float('inf') # Penalize trial

            # --- Setup Evaluation Callback for Trial ---
            try:
                 self.rl_logger.debug(f"Optuna Trial {trial.number}: >>> BEFORE Callback Initialization <<<")
                 eval_freq = int(ppo_params.get('n_steps', 2048) * 3)
                 eval_callback = TrialEvalCallback(eval_trial_env, trial, n_eval_episodes=5, eval_freq=eval_freq, verbose=2) # Keep verbose low for callback
                 self.rl_logger.debug(f"Optuna Trial {trial.number}: >>> AFTER Callback Initialization <<<")
                 self.rl_logger.debug(f"Optuna Trial {trial.number}: TrialEvalCallback setup complete.")
            except Exception as cb_err:
                 self.rl_logger.error(f"Optuna Trial {trial.number}: Failed to setup TrialEvalCallback: {cb_err}", exc_info=True)
                 return float('inf')


            # --- Train Trial Model ---
            trial_timesteps = optuna_cfg.get('trial_timesteps', 100000)
            mean_reward = -np.inf
            self.rl_logger.info(f"Optuna Trial {trial.number}: Starting training for {trial_timesteps} timesteps...")
            try:
                model.learn(total_timesteps=trial_timesteps, callback=eval_callback, progress_bar=False) # Progress bar off for trials
                mean_reward = eval_callback.get_last_mean_reward()
                self.rl_logger.info(f"Optuna Trial {trial.number} finished training. Last reported Mean Reward: {mean_reward:.4f}")
            except TrialPruned:
                 self.rl_logger.info(f"Optuna Trial {trial.number} successfully pruned.")
                 raise # Re-raise prune exception for Optuna to handle it
            except Exception as learn_err:
                 self.rl_logger.error(f"Optuna Trial {trial.number} training failed during model.learn: {learn_err}", exc_info=True)
                 mean_reward = -np.inf

        except Exception as e:
            self.rl_logger.error(f"Optuna Trial {trial.number} failed unexpectedly: {e}", exc_info=True)
            mean_reward = -np.inf
        finally:
            self.rl_logger.debug(f"Optuna Trial {trial.number}: Entering finally block for cleanup.")
            if trial_env:
                try: trial_env.close(); self.rl_logger.debug(f"Optuna Trial {trial.number}: Training env closed.")
                except Exception as close_err: self.rl_logger.error(f"Optuna Trial {trial.number}: Error closing training env: {close_err}")
            if eval_trial_env:
                try: eval_trial_env.close(); self.rl_logger.debug(f"Optuna Trial {trial.number}: Evaluation env closed.")
                except Exception as close_err: self.rl_logger.error(f"Optuna Trial {trial.number}: Error closing evaluation env: {close_err}")
            self.rl_logger.info(f"--- Finished Optuna Trial {trial.number} ---")

        if not np.isfinite(mean_reward):
            self.rl_logger.warning(f"Optuna Trial {trial.number}: Resulting mean_reward is not finite ({mean_reward}). Returning large penalty.")
            return float('inf')
        else:
            return -mean_reward


    def run(self):
        """Executes the RL training pipeline, optionally running HPO first."""
        signal.signal(signal.SIGINT, kill_all_children_and_exit)
        signal.signal(signal.SIGTERM, kill_all_children_and_exit)
        target_year = self.config.get('projections', {}).get('projection_year', 2024)
        is_training_run = self.config.get('rl_training', {}).get('enabled', False)

        if not is_training_run:
            self.rl_logger.info("RL training disabled in config. Skipping RL pipeline.")
            return
        if not SB3_AVAILABLE or FantasyDraftEnv is None:
             self.logger.error("RL Dependencies or Environment not available. Cannot run RL pipeline.")
             return

        self.rl_logger.info(f"--- Starting RL Pipeline Run for {target_year} ---")
        env = None
        best_hyperparams = self.config.get('rl_training', {}).get('ppo_hyperparams', {}).copy()

        # --- Optuna HPO (Optional) ---
        optuna_cfg = self.config.get('rl_training', {}).get('optuna', {})
        run_optuna = optuna_cfg.get('enabled', False)

        # if run_optuna:
        #     if not OPTUNA_AVAILABLE:
        #         self.rl_logger.error("Optuna is enabled in config but not installed. Skipping HPO and using default hyperparameters.")
        #     elif TrialEvalCallback is None:
        #          self.rl_logger.error("Optuna is enabled but TrialEvalCallback is not available. Skipping HPO.")
        #     else:
        #         self.rl_logger.info("--- Starting Optuna Hyperparameter Optimization ---")
        #         n_trials = optuna_cfg.get('n_trials', 25)
        #         timeout_seconds = optuna_cfg.get('timeout_seconds', 26 * 3600)
        #         study_name = f"ppo-fantasy-draft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        #         storage_url = optuna_cfg.get('storage_url', None)
        #         pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)

        #         study = optuna.create_study(study_name=study_name, storage=storage_url, direction="minimize", pruner=pruner, load_if_exists=True)
        #         self.rl_logger.info(f"Optuna Study '{study_name}' created/loaded. Storage: {storage_url}. Pruner: MedianPruner.")
        #         self.rl_logger.info(f"Running Optuna optimization for {n_trials} trials with timeout {timeout_seconds}s...")

        #         try:
        #             study.optimize(self._optuna_objective, n_trials=n_trials, timeout=timeout_seconds, gc_after_trial=True)
        #             if study.best_trial:
        #                 tuned_params = study.best_params
        #                 if 'policy_kwargs_net_arch' in tuned_params:
        #                      arch_choice = tuned_params.pop('policy_kwargs_net_arch')
        #                      try:
        #                          pi_str, vf_str = arch_choice.split('_')
        #                          pi_layers = [int(x) for x in pi_str.split('=')[1].split(',')]
        #                          vf_layers = [int(x) for x in vf_str.split('=')[1].split(',')]
        #                          best_hyperparams['policy_kwargs'] = dict(net_arch=[dict(pi=pi_layers, vf=vf_layers)])
        #                      except Exception as parse_err:
        #                          self.rl_logger.error(f"Error parsing best net_arch '{arch_choice}' from Optuna: {parse_err}. Keeping default policy_kwargs.")
        #                 best_hyperparams.update(tuned_params)
        #                 self.rl_logger.info(f"Optuna finished. Best Trial: {study.best_trial.number}, Value (Negative Reward): {study.best_value:.4f}")
        #                 self.rl_logger.info(f"Using Optuna Best Hyperparameters for final run: {best_hyperparams}")
        #             else:
        #                  self.rl_logger.warning("Optuna finished, but no best trial found. Proceeding with default hyperparameters.")
        #         except Exception as opt_err:
        #             self.rl_logger.error(f"Optuna optimization failed: {opt_err}", exc_info=True)
        #             self.rl_logger.warning("Proceeding with default hyperparameters from config.")
        # else:
        #     self.rl_logger.info("Optuna hyperparameter optimization disabled.")


        optuna_cfg = self.config.get('rl_training', {}).get('optuna', {})
        run_optuna = optuna_cfg.get('enabled', False)

        # if run_optuna:
        #     if not OPTUNA_AVAILABLE:
        #         self.rl_logger.error("Optuna is enabled but not installed. Skipping HPO.")
        #     elif TrialEvalCallback is None:
        #          self.rl_logger.error("Optuna is enabled but TrialEvalCallback is not available. Skipping HPO.")
        #     else:
        #         self.rl_logger.info("--- Starting Optuna Hyperparameter Optimization (Parallel) ---")
        #         n_trials = optuna_cfg.get('n_trials', 50)
        #         timeout_seconds = optuna_cfg.get('timeout_seconds', 26 * 3600)
        #         study_name = f"ppo-fantasy-draft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

        #         # --- Setup Persistent Storage (SQLite example) ---
        #         storage_db_dir = os.path.join(self.config['paths']['output_dir'], 'optuna_db')
        #         os.makedirs(storage_db_dir, exist_ok=True)
        #         storage_path = f"sqlite:///{os.path.join(storage_db_dir, f'{study_name}.db')}"
        #         self.rl_logger.info(f"Using Optuna storage: {storage_path}")
        #         # ------------------------------------------------

        #         pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)

        #         study = optuna.create_study(
        #             study_name=study_name,
        #             storage=storage_path, # Use the database storage
        #             direction="minimize",
        #             pruner=pruner,
        #             load_if_exists=True # Load previous results if study name/db exists
        #         )

        #         # --- Determine number of parallel jobs ---
        #         n_jobs = 2 #optuna_cfg.get('n_jobs', -1) # Default to all cores if not specified
        #         if n_jobs == -1:
        #             # import multiprocessing
        #             n_jobs = multiprocessing.cpu_count()
        #             self.rl_logger.info(f"Using n_jobs=-1, detected {n_jobs} CPU cores.")
        #         else:
        #             self.rl_logger.info(f"Using n_jobs={n_jobs} based on config.")
        #         # -----------------------------------------

        #         self.rl_logger.info(f"Running Optuna optimization in PARALLEL with n_jobs={n_jobs} for {n_trials} trials (timeout {timeout_seconds}s)...")


        #         # Create a shared counter for tracking trials
        #         trial_counter = multiprocessing.Value('i', 0)

        #         def run_trial_in_separate_process():
        #             """Independent process function that creates and runs its own trial"""
        #             # Get a unique trial ID
        #             with trial_counter.get_lock():
        #                 trial_id = trial_counter.value
        #                 trial_counter.value += 1
                    
        #             pid = os.getpid()
        #             print(f"Process {pid} starting trial {trial_id}")
                    
        #             # Create a new trial
        #             trial = study.ask()
                    
        #             try:
        #                 # This calls your existing objective function
        #                 value = self._optuna_objective(trial)
        #                 # Report result back to study
        #                 study.tell(trial, value)
        #                 return (trial_id, value, pid)
        #             except Exception as e:
        #                 print(f"Error in trial {trial_id} (PID {pid}): {e}")
        #                 study.tell(trial, float("inf"))
        #                 return (trial_id, float("inf"), pid)

        #         try:
        #             # Use ProcessPoolExecutor to explicitly create separate processes
        #             completed_trials = []
                    
        #             with concurrent.futures.ProcessPoolExecutor(max_workers=n_jobs) as executor:
        #                 # Submit all trial jobs
        #                 futures = [executor.submit(run_trial_in_separate_process) for _ in range(n_trials)]
                        
        #                 # Process results as they complete
        #                 for future in concurrent.futures.as_completed(futures):
        #                     try:
        #                         trial_id, value, pid = future.result()
        #                         completed_trials.append((trial_id, value))
        #                         self.rl_logger.info(f"Trial {trial_id} completed by process {pid} with value: {value}")
        #                     except Exception as exc:
        #                         self.rl_logger.error(f"Trial execution failed: {exc}")
                    
        #             self.rl_logger.info(f"All {len(completed_trials)} trials completed.")
                    
        #             # --- Save the study object itself ---
        #             study_save_path = os.path.join(self.config['paths']['output_dir'], f'{study_name}_study.pkl')
        #             try:
        #                 joblib.dump(study, study_save_path)
        #                 self.rl_logger.info(f"Saved Optuna study object to: {study_save_path}")
        #             except Exception as study_save_err:
        #                 self.rl_logger.error(f"Failed to save Optuna study object: {study_save_err}")

        #             if study.best_trial:
        #                 # ... (process best_hyperparams as before) ...
        #                 self.rl_logger.info(f"Optuna (Parallel) finished. Best Trial: {study.best_trial.number}, Value: {study.best_value:.4f}")
        #                 self.rl_logger.info(f"Using Optuna Best Hyperparameters: {best_hyperparams}")
        #             else:
        #                 self.rl_logger.warning("Optuna (Parallel) finished, but no best trial found. Using defaults.")

        #         except Exception as opt_err:
        #             self.rl_logger.error(f"Optuna parallel optimization failed: {opt_err}", exc_info=True)
        #             self.rl_logger.warning("Proceeding with default hyperparameters.")
        # else:
        #     self.rl_logger.info("Optuna hyperparameter optimization disabled.")


        if run_optuna:
            self.rl_logger.info("--- Starting Optuna Hyperparameter Optimization (Parallel) ---")
            n_trials = optuna_cfg.get('n_trials', 50)
            timeout_seconds = optuna_cfg.get('timeout_seconds', 26 * 3600)
            study_name = f"ppo-fantasy-draft-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

            # --- Setup Persistent Storage (SQLite example) ---
            storage_db_dir = os.path.join(self.config['paths']['output_dir'], 'optuna_db')
            os.makedirs(storage_db_dir, exist_ok=True)
            storage_path = f"sqlite:///{os.path.join(storage_db_dir, f'{study_name}.db')}"
            self.rl_logger.info(f"Using Optuna storage: {storage_path}")

            # Store study information for worker processes
            self._current_study_name = study_name
            self._current_storage_path = storage_path

            # Initialize the global instance reference
            global _GLOBAL_INSTANCE
            _GLOBAL_INSTANCE = self

            # Create the study
            pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3, interval_steps=1)
            study = optuna.create_study(
                study_name=study_name,
                storage=storage_path,
                direction="minimize",
                pruner=pruner,
                load_if_exists=True
            )

            # --- Determine number of parallel jobs ---
            n_jobs = optuna_cfg.get('n_jobs', -1)
            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()
                self.rl_logger.info(f"Using n_jobs=-1, detected {n_jobs} CPU cores.")
            else:
                self.rl_logger.info(f"Using n_jobs={n_jobs} based on config.")

            self.rl_logger.info(f"Running Optuna optimization in PARALLEL with n_jobs={n_jobs} for {n_trials} trials (timeout {timeout_seconds}s)...")

            try:
                progress_bar = tqdm(total=n_trials, desc="Optuna Trials", unit="trial")

                # Use ProcessPoolExecutor for true parallelization
                completed_trials = []
                
                with concurrent.futures.ProcessPoolExecutor(
                    max_workers=n_jobs,
                    initializer=_init_worker,
                    initargs=(self.config, study_name, storage_path)
                ) as executor:
                    # Submit all trial jobs - use the module-level function
                    futures = [executor.submit(_run_optuna_trial_in_process, i) for i in range(n_trials)]
                    
                    # Process results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            trial_id, value, pid = future.result()
                            completed_trials.append((trial_id, value))
                            self.rl_logger.info(f"Trial {trial_id} completed by process {pid} with value: {value}")
                            
                            # Update the progress bar
                            progress_bar.update(1)
                            progress_bar.set_postfix({"best": f"{study.best_value:.4f}" if study.best_trial else "None"})
                        except Exception as exc:
                            self.rl_logger.error(f"Trial execution failed: {exc}")
                            progress_bar.update(1)
                progress_bar.close()
                self.rl_logger.info(f"All {len(completed_trials)} trials completed.")
                
                # --- Save the study object itself ---
                study_save_path = os.path.join(self.config['paths']['output_dir'], f'{study_name}_study.pkl')
                try:
                    joblib.dump(study, study_save_path)
                    self.rl_logger.info(f"Saved Optuna study object to: {study_save_path}")
                except Exception as study_save_err:
                    self.rl_logger.error(f"Failed to save Optuna study object: {study_save_err}")

                if study.best_trial:
                    # Process best parameters
                    tuned_params = study.best_params.copy()
                    if 'policy_kwargs_net_arch' in tuned_params:
                        arch_choice = tuned_params.pop('policy_kwargs_net_arch')
                        try:
                            pi_str, vf_str = arch_choice.split('_')
                            pi_layers = [int(x) for x in pi_str.split('=')[1].split(',')]
                            vf_layers = [int(x) for x in vf_str.split('=')[1].split(',')]
                            best_hyperparams['policy_kwargs'] = dict(net_arch=[dict(pi=pi_layers, vf=vf_layers)])
                        except Exception as parse_err:
                            self.rl_logger.error(f"Error parsing net_arch '{arch_choice}': {parse_err}")
                    
                    best_hyperparams.update(tuned_params)
                    self.rl_logger.info(f"Optuna (Parallel) finished. Best Trial: {study.best_trial.number}, Value: {study.best_value:.4f}")
                    self.rl_logger.info(f"Using Optuna Best Hyperparameters: {best_hyperparams}")
                else:
                    self.rl_logger.warning("Optuna (Parallel) finished, but no best trial found. Using defaults.")

            except Exception as opt_err:
                kill_all_children_and_exit()
                self.rl_logger.error(f"Optuna parallel optimization failed: {opt_err}", exc_info=True)
                self.rl_logger.warning("Proceeding with default hyperparameters.")
        else:
            self.rl_logger.info("--- No Optuna Optimization Requested ---")

        # --- Final Training Run ---
        # self.rl_logger.info("--- Starting Final RL Agent Training ---")
        # final_model = None # Keep track of the final model instance
        self.rl_logger.info("--- Starting Final RL Agent Training (Parallel Environments) ---")
        final_model = None
        vec_env = None # Use vec_env to manage the environment instance

        try:
            # Load data for final run
            league_settings_path = self.config['paths']['config_path']
            league_settings = load_league_settings(league_settings_path)
            models_dir = self.config['paths']['models_dir']
            projections_df = load_projections_from_csvs(models_dir, league_settings)
            if projections_df.empty:
                self.rl_logger.error("Failed projection load for final run. Aborting final training.")
                return

            # Log Memory Usage Before Env Creation
            try:
                proj_mem_usage_mb = projections_df.memory_usage(deep=True).sum() / (1024 * 1024)
                self.rl_logger.debug(f"Memory usage of projections_df before env creation: {proj_mem_usage_mb:.2f} MB")
            except Exception as mem_err:
                self.rl_logger.warning(f"Could not get projections_df memory usage: {mem_err}")

            # --- Determine Number of Parallel Environments ---
            num_envs = self.config.get('rl_training', {}).get('num_parallel_envs', multiprocessing.cpu_count())
            # Limit if needed, e.g., based on available RAM
            max_envs = 16 # Example cap
            if num_envs > max_envs:
                self.rl_logger.warning(f"Requested {num_envs} parallel envs, capping at {max_envs}.")
                num_envs = max_envs
            if num_envs <= 0:
                self.rl_logger.warning(f"Invalid num_parallel_envs ({num_envs}), defaulting to 1.")
                num_envs = 1
            self.rl_logger.info(f"Using {num_envs} parallel environments for final training.")
            # -----------------------------------------------


            # Get final run configs
            rl_config = self.config.get('rl_training', {}); models_save_dir = self.config['paths'].get('models_dir', 'data/models')
            os.makedirs(models_save_dir, exist_ok=True); agent_draft_pos = rl_config.get('agent_draft_position', 1);
            total_timesteps = rl_config.get('final_training_timesteps', 1000000)
            model_filename_base = rl_config.get('model_save_path', "ppo_fantasy_draft_agent"); tensorboard_base_path = rl_config.get('tensorboard_log_path', "./ppo_fantasydraft_tensorboard/")
            model_filename = f"{model_filename_base}.zip" if not model_filename_base.endswith(".zip") else model_filename_base
            model_save_path = os.path.join(models_save_dir, model_filename); tensorboard_log_path = tensorboard_base_path
            checkpoint_save_path = os.path.join(models_save_dir, 'rl_checkpoints/'); os.makedirs(checkpoint_save_path, exist_ok=True)
            os.makedirs(tensorboard_log_path, exist_ok=True)
            monitor_base_dir = self.config['paths'].get('output_dir', 'data/outputs'); os.makedirs(os.path.join(monitor_base_dir, 'rl_monitor_logs'), exist_ok=True)

            # Final Env Setup w/ Monitor
            # self.rl_logger.debug("Defining make_final_env function...")
            self.rl_logger.debug(f"Defining make_env function for parallel envs (Index {{i}})...")
            # def make_final_env():
            #      run_timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            #      monitor_log_dir = os.path.join(monitor_base_dir, 'rl_monitor_logs', f'final_run_{run_timestamp}')
            #      os.makedirs(monitor_log_dir, exist_ok=True)
            #      self.rl_logger.info(f"Monitor logs for final run will be saved to: {monitor_log_dir}")
            #      self.rl_logger.debug("make_final_env: Creating FantasyDraftEnv instance...")
            #      # Pass a copy of the projections DF to avoid potential issues if env modifies it
            #      env_instance = FantasyDraftEnv(projections_df.copy(), league_settings, agent_draft_pos)
            #      self.rl_logger.debug("make_final_env: FantasyDraftEnv created. Wrapping with Monitor...")
            #      env_instance = Monitor(env_instance, filename=os.path.join(monitor_log_dir, "monitor.csv"), allow_early_resets=True)
            #      self.rl_logger.debug("make_final_env: Monitor wrapping complete.")
            #      return env_instance
            
            def make_env(rank: int, seed: int = 0):
                """
                Utility function for multiprocessed env.
                :param rank: index of the subprocess
                :param seed: the initial seed for RNG
                """
                def _init():
                    # Use a unique directory for each Monitor instance if needed,
                    # though often one aggregate monitor file is sufficient.
                    # For simplicity, let's point them all to one dir for now.
                    monitor_log_dir = os.path.join(monitor_base_dir, 'rl_monitor_logs', f'final_run_env_{rank}')
                    os.makedirs(monitor_log_dir, exist_ok=True)

                    # Ensure each env gets a different seed potentially based on rank + global seed
                    env_seed = seed + rank + int(time.time() * 1000 + random.randint(0,1000)) % (2**32 - 1)

                    # Pass a COPY of projections_df
                    env = FantasyDraftEnv(projections_df.copy(), league_settings, agent_draft_pos)
                    # Seed the environment if it supports it (optional but good practice)
                    try:
                         env.reset(seed=env_seed)
                    except TypeError:
                         env.reset() # Fallback if seed not accepted by reset

                    # Wrap with Monitor
                    env = Monitor(env, filename=os.path.join(monitor_log_dir, f"monitor_{rank}.csv"), allow_early_resets=True)
                    self.rl_logger.debug(f"Created env instance {rank} with seed {env_seed}")
                    return env
                # Set a unique seed for each environment process based on rank
                # set_random_seed(seed + rank) # SB3 utility, or manage manually
                return _init
            
            self.rl_logger.info(f"Creating SubprocVecEnv with {num_envs} environments...")
            if num_envs > 1:
                vec_env = SubprocVecEnv([make_env(i) for i in range(num_envs)])
            else:
                vec_env = DummyVecEnv([make_env(0)]) # Use DummyVecEnv for a single environment case
            self.rl_logger.info(f"{'SubprocVecEnv' if num_envs > 1 else 'DummyVecEnv'} created.")

            # self.rl_logger.debug("Creating DummyVecEnv for final training...")
            # env = DummyVecEnv([make_final_env])
            # self.rl_logger.debug("DummyVecEnv created.")
            # try:
            #     self.rl_logger.debug("Running check_env on underlying environment...")
            #     check_env(env.envs[0].env, warn=True);
            #     self.rl_logger.info("Final Env check passed.")
            # except Exception as check_err:
            #      self.rl_logger.error(f"Final Env check failed: {check_err}", exc_info=True)
            #      if env: env.close() # Close if check fails
            #      return

            # Model Definition
            self.rl_logger.info(f"Defining FINAL PPO model with params: {best_hyperparams}")
            if 'policy_kwargs' in best_hyperparams and not isinstance(best_hyperparams.get('policy_kwargs'), (dict, type(None))):
                 self.rl_logger.warning(f"Invalid type for policy_kwargs in best_hyperparams. Setting to None.")
                 best_hyperparams['policy_kwargs'] = None
            elif 'policy_kwargs' not in best_hyperparams:
                 best_hyperparams['policy_kwargs'] = self.config.get('rl_training', {}).get('ppo_hyperparams', {}).get('policy_kwargs', None)
                 if not isinstance(best_hyperparams['policy_kwargs'], (dict, type(None))):
                      best_hyperparams['policy_kwargs'] = None
                      self.rl_logger.warning("Using default SB3 policy_kwargs as config value was invalid.")

            # --- Debugging Point ---
            self.rl_logger.debug(">>> BEFORE FINAL PPO Initialization <<<")
            time.sleep(0.5) # Tiny pause before potential hang point
            # ---

            # Define the PPO model instance
            # --- TEMPORARY TEST: Disable TensorBoard Logging ---
            # final_tensorboard_log = None
            # self.rl_logger.warning("TEMPORARY DEBUG: TensorBoard logging disabled for PPO init.")
            final_tensorboard_log = tensorboard_log_path # Restore normal path
            # ---
            final_model = PPO(
                policy="MlpPolicy",
                env=vec_env, # Use the vectorized environment
                verbose=1,
                tensorboard_log=tensorboard_log_path,
                **best_hyperparams
            )
            self.rl_logger.info("Final PPO model defined.")
            # final_model = PPO(policy="MlpPolicy", env=env, verbose=1, # Changed verbose to 1 for final run
            #             tensorboard_log=final_tensorboard_log, **best_hyperparams)

            # --- Debugging Point ---
            self.rl_logger.debug(">>> AFTER FINAL PPO Initialization <<<")
            # ---

            # Callbacks for Final Run
            self.rl_logger.debug(">>> BEFORE FINAL CallbackList Initialization <<<")
            callbacks_final = []
            checkpoint_freq = max(20000, total_timesteps // 50)
            # cp_callback = CheckpointCallback(save_freq=checkpoint_freq, save_path=checkpoint_save_path, name_prefix="final_"+model_filename_base.replace('.zip',''), save_replay_buffer=False, save_vecnormalize=True)
            # callbacks_final.append(cp_callback)
            # self.rl_logger.debug(f"CheckpointCallback enabled for final run (freq: {checkpoint_freq} steps).")
            # # --- TEMPORARY TEST: Disable Callbacks ---
            # # final_callback_list = None
            # # self.rl_logger.warning("TEMPORARY DEBUG: Callbacks disabled for model.learn.")
            # final_callback_list = CallbackList(callbacks_final) # Restore normal callbacks
            
            cp_callback = CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=checkpoint_save_path,
                name_prefix="final_"+model_filename_base.replace('.zip',''),
                save_replay_buffer=False, # Usually false for PPO
                save_vecnormalize=True # Save VecNormalize stats if used (not used here currently)
            )
            callbacks_final.append(cp_callback)
            final_callback_list = CallbackList(callbacks_final)
            self.rl_logger.info("Callbacks defined for final run.")
            # ---
            self.rl_logger.debug(">>> AFTER FINAL CallbackList Initialization <<<")


            # # Final Training
            # self.rl_logger.info(f"Starting FINAL training for {total_timesteps} timesteps...")
            # tb_log_name = f"PPO_Final"
            # final_model.learn( total_timesteps=total_timesteps, progress_bar=True, callback=final_callback_list, tb_log_name=tb_log_name, reset_num_timesteps=False )
            # self.rl_logger.info("FINAL RL Training finished.")
            
            # --- Final Training ---
            self.rl_logger.info(f"Starting FINAL training with {num_envs} parallel envs for {total_timesteps} total timesteps...")
            tb_log_name = f"PPO_Final_{num_envs}envs" # Include env count in log name
            final_model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True,
                callback=final_callback_list,
                tb_log_name=tb_log_name,
                reset_num_timesteps=False # Usually false unless continuing a run
            )
            self.rl_logger.info("FINAL RL Training finished.")
            # ----------------------

            # Save Final Model
            if model_save_path:
                 try:
                     final_model.save(model_save_path);
                     self.rl_logger.info(f"FINAL RL Model saved to {model_save_path}")
                     hyperparams_save_path = model_save_path.replace('.zip', '_hyperparams.json')
                     with open(hyperparams_save_path, 'w') as f:
                         serializable_params = best_hyperparams.copy()
                         # Basic check for non-serializable items if needed
                         # if serializable_params.get('policy_kwargs') and not isinstance(serializable_params['policy_kwargs'], (dict, type(None))):
                         #      serializable_params['policy_kwargs'] = str(serializable_params['policy_kwargs']) # Example fallback
                         json.dump(serializable_params, f, indent=4)
                     self.rl_logger.info(f"FINAL RL Hyperparameters saved to {hyperparams_save_path}")
                 except Exception as save_err:
                     self.rl_logger.error(f"Error saving FINAL RL model or hyperparameters: {save_err}", exc_info=True)

            # Final Evaluation
            self.evaluate_agent(final_model, projections_df, league_settings, agent_draft_pos, target_year)

            self.rl_logger.info(f"--- RL Pipeline Finished (Final Run for {target_year}) ---")
            
            

        except ImportError as imp_err:
             self.logger.error(f"Import Error during RL final run setup: {imp_err}. Aborting.")
        # except Exception as e:
        #      self.rl_logger.error(f"Unexpected error in RL pipeline final run: {e}", exc_info=True)
        # finally:
        #      if env is not None:
        #           try:
        #               env.close();
        #               self.rl_logger.info("Final training environment closed.")
        #           except Exception as close_err:
        #               self.rl_logger.error(f"Error closing final training env: {close_err}")
        
        except Exception as e:
            self.rl_logger.error(f"Unexpected error in RL pipeline final run: {e}", exc_info=True)
        finally:
            if vec_env is not None: # Close the VecEnv
                try:
                    vec_env.close()
                    self.rl_logger.info("Final training vectorized environment closed.")
                except Exception as close_err:
                    self.rl_logger.error(f"Error closing final training vec_env: {close_err}")


    def evaluate_agent(self, model, projections_df, league_settings, agent_draft_pos, eval_year):
        """Evaluates the RL agent for a specific year."""
        self.rl_logger.info(f"--- Evaluating Agent for Year: {eval_year} ---")
        eval_env = None
        try:
            eval_env = FantasyDraftEnv(projections_df.copy(), league_settings, agent_draft_pos)
            eval_env.set_render_mode('logging')

            n_eval_episodes = 5
            episode_rewards = []
            self.rl_logger.info(f"Running {n_eval_episodes} evaluation episodes...")

            for i_episode in range(n_eval_episodes):
                 obs, info = eval_env.reset(seed=int(time.time() * 1000 + i_episode) % (2**32 - 1))
                 done, truncated, ep_reward, step_count = False, False, 0.0, 0
                 max_ep_steps = getattr(eval_env, 'total_picks', 200) + 10

                 self.rl_logger.debug(f"\n--- Eval Episode {i_episode+1}/{n_eval_episodes} ---")

                 while not done and not truncated and step_count < max_ep_steps:
                      action_output, _states = model.predict(obs, deterministic=True)
                      action_int = -1
                      try:
                           if isinstance(action_output, np.ndarray): action_int = int(action_output.item()) if action_output.size==1 else -1
                           elif isinstance(action_output, (int, np.integer, np.int64, np.int32)): action_int = int(action_output)
                           else: self.rl_logger.error(f"Eval Ep {i_episode+1} Predict unexpected type: {type(action_output)}")
                      except Exception as e: self.rl_logger.error(f"Eval Ep {i_episode+1} Error processing action '{action_output}': {e}"); action_int = -1

                      if action_int < 0 or action_int >= NUM_ACTIONS:
                           self.rl_logger.warning(f"Eval Ep {i_episode+1} Invalid action {action_int}. Using fallback {ACTION_BEST_AVAILABLE}.")
                           action_int = ACTION_BEST_AVAILABLE

                      try:
                           obs, reward, done, truncated, info = eval_env.step(action_int)
                      except Exception as step_err:
                           self.rl_logger.error(f"Eval Ep {i_episode+1} Error in env.step(action={action_int}) at step {step_count+1}: {step_err}", exc_info=True)
                           done = True

                      step_count += 1
                      if done or truncated:
                           ep_reward = reward;
                           status = "Done" if done else "Truncated";
                           self.rl_logger.debug(f"Eval Ep {i_episode+1} Finished ({status} after {step_count} steps). Final Reward: {ep_reward:.4f}")
                           self.rl_logger.info(f"Agent's Final Roster (Eval Ep {i_episode+1}):"); roster_list = eval_env.teams_rosters.get(eval_env.agent_team_id, [])
                           if roster_list:
                                try:
                                   df_roster = pd.DataFrame(roster_list); cols = ['name', 'position', 'projected_points', 'vorp', 'risk_adjusted_vorp']; avail_cols = [c for c in cols if c in df_roster.columns];
                                   if avail_cols: self.rl_logger.debug(f"\n{df_roster[avail_cols].round(2).to_string(index=False)}")
                                   else: self.rl_logger.warning("  Roster missing display cols.")
                                except Exception as df_err: self.rl_logger.error(f"Error logging roster DF for Ep {i_episode+1}: {df_err}"); self.rl_logger.info("  Could not log roster.")
                           else: self.rl_logger.warning("  Agent roster is empty.")
                           episode_rewards.append(ep_reward)
                           break
                 if step_count >= max_ep_steps:
                      self.rl_logger.warning(f"Eval Ep {i_episode+1} hit max steps ({max_ep_steps}) without finishing naturally.")


            if episode_rewards:
                 mean_reward = np.mean(episode_rewards)
                 std_reward = np.std(episode_rewards)
                 self.rl_logger.info(f"\nEvaluation Summary ({eval_year}, {len(episode_rewards)} episodes finished):") # Corrected count
                 self.rl_logger.info(f"  Mean Reward (Total VORP): {mean_reward:.4f}")
                 self.rl_logger.info(f"  Std Dev Reward: {std_reward:.4f}")
                 self.rl_logger.info(f"  Individual Rewards: {[f'{r:.2f}' for r in episode_rewards]}")
            else:
                 self.rl_logger.warning(f"No episodes finished successfully during evaluation for {eval_year}.")

            self.rl_logger.info(f"--- RL Draft Agent Evaluation Finished for Year: {eval_year} ---")

        except ImportError as imp_err:
            self.logger.error(f"Import Error during evaluation setup for {eval_year}: {imp_err}. Aborting evaluation.")
        except Exception as e:
             self.rl_logger.error(f"Error during evaluation for {eval_year}: {e}", exc_info=True)
        finally:
             if eval_env is not None:
                  try: eval_env.close(); self.rl_logger.info(f"Evaluation environment for {eval_year} closed.")
                  except Exception as ce: self.rl_logger.error(f"Error closing evaluation env for {eval_year}: {ce}")





