# # src/models/plotting_callback.py

# import os
# import numpy as np
# import matplotlib.pyplot as plt
# from stable_baselines3.common.callbacks import BaseCallback
# import logging
# import pandas as pd
# from collections import deque

# # Use standard Python logging for messages FROM the callback
# # The callback itself accesses SB3's internal logger differently
# callback_logger = logging.getLogger("fantasy_draft_rl.PlottingCallback")

# class PlottingCallback(BaseCallback):
#     """
#     Callback that collects training metrics logged by SB3
#     and plots them at the end of training. Correctly accesses SB3 logger values.

#     :param plot_freq: (int) Check logs every `plot_freq` calls. Plot at end.
#     :param log_dir: (str) Path to save the plot.
#     :param plot_name: (str) Name of the plot file.
#     :param verbose: (int) Verbosity level.
#     """
#     def __init__(self, log_dir: str, plot_name: str = "rl_training_metrics.png", plot_freq: int = 1000, verbose: int = 0):
#         super().__init__(verbose)
#         self.plot_freq = plot_freq
#         self.log_dir = log_dir
#         self.plot_path = os.path.join(log_dir, plot_name)
#         # Define the metrics we want to try and plot from SB3 logs
#         self.metrics_to_plot = [
#             'rollout/ep_rew_mean',
#             'rollout/ep_len_mean',
#             'time/fps',
#             'train/loss',
#             'train/value_loss',
#             'train/policy_gradient_loss', # Confirm this key
#             'train/entropy_loss',
#             'train/approx_kl',
#             'train/clip_fraction',
#             'train/explained_variance',
#         ]
#         # Initialize storage
#         self.metric_data = {metric: {'timesteps': [], 'values': []} for metric in self.metrics_to_plot}
#         self.fig = None
#         self.axes = None
#         self.metric_info = {} # To store plot index and title
#         os.makedirs(self.log_dir, exist_ok=True)

#     def _clean_metric_name(self, metric_key):
#         """Removes common prefixes for cleaner plot titles."""
#         if metric_key.startswith("train/"): return metric_key[len("train/"):]
#         if metric_key.startswith("rollout/"): return metric_key[len("rollout/"):]
#         if metric_key.startswith("time/"): return metric_key[len("time/"):]
#         return metric_key

#     def _on_training_start(self) -> None:
#         """Initialize the plot figure and axes."""
#         callback_logger.info("Initializing plots...") # Use callback_logger instance
#         num_plots = len(self.metrics_to_plot)
#         if num_plots == 0: callback_logger.warning("No metrics selected."); return
#         ncols = 3; nrows = (num_plots + ncols - 1) // ncols
#         try:
#             self.fig, self.axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
#             self.axes = self.axes.flatten()
#             for ax in self.axes: ax.cla()
#             for i, metric_name in enumerate(self.metrics_to_plot):
#                 if i < len(self.axes):
#                     clean_title = self._clean_metric_name(metric_name)
#                     self.axes[i].set_title(clean_title, fontsize=10)
#                     self.axes[i].set_xlabel("Timesteps", fontsize=8); self.axes[i].set_ylabel("Value", fontsize=8)
#                     self.axes[i].tick_params(axis='both', which='major', labelsize=8); self.axes[i].grid(True, linestyle='--', alpha=0.6)
#                     self.metric_info[metric_name] = {'index': i, 'title': clean_title} # Store info
#             for i in range(num_plots, len(self.axes)):
#                 if i < len(self.axes): self.axes[i].set_visible(False)
#             callback_logger.info(f"Plot figure initialized ({nrows}x{ncols}).")
#         except Exception as e:
#              callback_logger.error(f"Error initializing plot figure: {e}", exc_info=True)
#              self.fig, self.axes = None, None

#     def _on_step(self) -> bool:
#         """Collect metrics logged by SB3 at specified frequency."""
#         # Check logger values every self.plot_freq steps
#         # Use num_timesteps for consistency with SB3 logging intervals
#         if self.num_timesteps > 0 and self.num_timesteps % self.plot_freq == 0:
#             # *** CORRECT WAY to access logged values ***
#             # SB3's logger object has `name_to_value` and `name_to_count` dictionaries
#             log_dict = self.logger.name_to_value

#             if self.verbose > 0: # Log available keys less frequently if verbose=1
#                  if self.num_timesteps % (self.plot_freq * 5) == 0:
#                      callback_logger.info(f"Plot CB @ step {self.num_timesteps}: Checking logs. Keys: {list(log_dict.keys())}")

#             # Store the metrics we want to plot if they exist in the logs
#             for metric_name in self.metrics_to_plot:
#                 if metric_name in log_dict:
#                     # The value in name_to_value is often the smoothed value
#                     value = log_dict[metric_name]
#                     if isinstance(value, (int, float, np.number)) and np.isfinite(value): # Check type and finiteness
#                         self.metric_data[metric_name]['timesteps'].append(self.num_timesteps)
#                         self.metric_data[metric_name]['values'].append(value)
#                         # callback_logger.debug(f"Plot CB: Stored {metric_name}={value:.4f}") # Optional debug
#                     else:
#                          # Log unexpected types or non-finite values less often
#                          if self.num_timesteps % (self.plot_freq * 10) == 0:
#                             callback_logger.warning(f"Plot CB: Skipping invalid value for {metric_name}: {value} (type: {type(value)}) @ {self.num_timesteps}")
#                 # No need for warning if key is missing, it might appear later

#         return True # Continue training


#     def _on_training_end(self) -> None:
#         """Plot and save the final metrics with log scale option."""
#         callback_logger.info("Training ended. Generating final plots...")
#         if self.fig is None or self.axes is None: callback_logger.error("Plot CB: Figure/Axes not init."); return
#         try:
#             metrics_plotted_count = 0
#             for metric_name in self.metrics_to_plot:
#                 info = self.metric_info.get(metric_name);
#                 if info is None: continue
#                 idx = info['index']
#                 if idx < len(self.axes):
#                     ax = self.axes[idx]; data = self.metric_data.get(metric_name, {'timesteps': [], 'values': []})
#                     if data['timesteps'] and data['values']:
#                         timesteps = np.array(data['timesteps']); values = np.array(data['values'])
#                         use_log_scale = metric_name in ['train/loss', 'train/value_loss']
#                         label_main = info['title']; label_avg = 'Avg' # Use cleaned title

#                         if use_log_scale:
#                             mask = values > 1e-8; # Avoid log(0) or log(negative)
#                             if np.any(mask):
#                                 ax.plot(timesteps[mask], values[mask], label=label_main, lw=1.5); ax.set_yscale('log');
#                                 w = max(5, len(values[mask])//20); # Rolling window
#                                 if len(values[mask])>=w:
#                                      try: s=pd.Series(values[mask]); r=s.rolling(w,min_periods=1,center=True).mean(); ax.plot(timesteps[mask],r,ls='--',alpha=0.7,label=label_avg,lw=1.0)
#                                      except Exception as roll_err: callback_logger.warning(f"Roll avg err {metric_name}: {roll_err}")
#                             else: callback_logger.warning(f"Plot CB: Cannot log scale {metric_name}."); ax.plot(timesteps, values, label=label_main, lw=1.5) # Linear fallback
#                         else: # Linear plot
#                             ax.plot(timesteps, values, label=label_main, lw=1.5)
#                             w = max(5, len(values)//20);
#                             if len(values)>=w:
#                                 try: s=pd.Series(values); r=s.rolling(window=w, min_periods=1, center=True).mean(); ax.plot(timesteps,r,ls='--',alpha=0.7,label=label_avg,lw=1.0)
#                                 except Exception as roll_err: callback_logger.warning(f"Roll avg err {metric_name}: {roll_err}")

#                         ax.legend(loc='best', fontsize='x-small'); metrics_plotted_count += 1
#                     else: callback_logger.warning(f"Plot CB: No data collected for '{metric_name}'. Plot empty."); ax.text(0.5,0.5,"No data",ha='center',va='center',transform=ax.transAxes,fontsize=9,color='grey')
#                 else: callback_logger.warning(f"Plot CB: Index mismatch {metric_name}.")

#             if metrics_plotted_count == 0: callback_logger.error("Plot CB: No data plotted.")
#             self.fig.suptitle("RL Agent Training Metrics", fontsize=14)
#             self.fig.tight_layout(rect=[0, 0.03, 1, 0.96]) # Adjust after plotting
#             self.fig.savefig(self.plot_path, dpi=200); callback_logger.info(f"Plotting Callback: Saved plot to {self.plot_path}")
#             plt.close(self.fig) # Close figure after saving
#         except Exception as e: callback_logger.error(f"Plot CB: Error final plot/save: {e}", exc_info=True);
#         if self.fig: plt.close(self.fig) # Ensure closed on error

from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList, BaseCallback
import gymnasium as gym
import numpy as np
import logging
import time
import random
from optuna import TrialPruned
from src.models.rl_draft_agent import NUM_ACTIONS, ACTION_BEST_AVAILABLE

# --- Optuna Objective Callback ---
class TrialEvalCallback(BaseCallback):
    """
    Callback used by Optuna objective function to evaluate the agent
    trained during a trial and report the reward. Stops training early
    once enough data is gathered for evaluation or if pruned.

    :param eval_env: Environment instance for evaluation.
    :param trial: Optuna trial object.
    :param n_eval_episodes: Number of episodes for evaluation.
    :param eval_freq: Check for evaluation every n steps.
    :param deterministic: Use deterministic actions for evaluation.
    :param verbose: Verbosity level.
    """
    def __init__(self, eval_env: gym.Env, trial, n_eval_episodes: int = 5,
                 eval_freq: int = 5000, deterministic: bool = True, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.eval_env = eval_env
        self.trial = trial
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.eval_count = 0
        self._is_pruned = False # Flag to track pruning status
        self.rl_logger = logging.getLogger("fantasy_draft_rl") # Get RL logger instance

        # Reset the eval env at the start of the callback
        try:
            # Seed with time if possible
            self.eval_env.reset(seed=int(time.time() * 1000 + random.randint(0,1000)) % (2**32 -1))
            self.rl_logger.debug("Optuna eval env reset with seed.")
        except TypeError:
            self.eval_env.reset() # Fallback if seed not accepted
            self.rl_logger.debug("Optuna eval env reset does not accept seed.")
        except Exception as e:
             self.rl_logger.error(f"Error resetting eval_env in Optuna callback init: {e}")
             # Potentially raise error or handle depending on desired robustness


    def _on_step(self) -> bool:
        if self._is_pruned:
             self.rl_logger.debug(f"Optuna Trial {self.trial.number} is pruned, stopping training.")
             return False # Stop training if pruned

        # Evaluate periodically
        if self.eval_freq > 0 and self.n_calls > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_count += 1
            self.rl_logger.debug(f"Optuna Trial {self.trial.number} - Step: {self.num_timesteps} | Starting Eval# {self.eval_count}")
            episode_rewards = []
            total_eval_steps = 0 # Track steps within this specific eval loop
            max_allowed_eval_steps = getattr(self.eval_env, 'total_picks', 200) * self.n_eval_episodes * 1.5 # Safety break

            for i_episode in range(self.n_eval_episodes):
                if total_eval_steps > max_allowed_eval_steps:
                    self.rl_logger.error(f"Optuna Trial {self.trial.number} Eval {self.eval_count} Exceeded max steps {max_allowed_eval_steps}. Aborting eval loop.")
                    break
                try:
                    obs, _ = self.eval_env.reset(seed=int(time.time()*1000 + i_episode + self.n_calls) % (2**32 - 1))
                except TypeError:
                    obs, _ = self.eval_env.reset()
                except Exception as reset_err:
                     self.rl_logger.error(f"Optuna Trial {self.trial.number} Eval {self.eval_count} - Error resetting env for episode {i_episode+1}: {reset_err}")
                     continue # Skip this episode if reset fails

                done, truncated = False, False
                ep_reward = 0.0
                steps_in_ep = 0
                max_ep_steps = getattr(self.eval_env, 'total_picks', 200) + 5 # Max steps per episode

                while not done and not truncated and steps_in_ep < max_ep_steps:
                    try:
                         action, _ = self.model.predict(obs, deterministic=self.deterministic)
                    except Exception as pred_err:
                         self.rl_logger.error(f"Optuna Trial {self.trial.number} Eval {self.eval_count} - Error during predict: {pred_err}")
                         action = ACTION_BEST_AVAILABLE # Fallback action

                    # Ensure action is standard int before passing
                    action_int = -1
                    try:
                        if isinstance(action, np.ndarray): action_int = int(action.item()) if action.size==1 else -1
                        elif isinstance(action, (int, np.integer, np.int64, np.int32)): action_int = int(action)
                        else: self.rl_logger.warning(f"Optuna Eval Predict unexpected action type: {type(action)}")
                    except Exception as action_conv_err:
                         self.rl_logger.error(f"Optuna Eval Error converting action '{action}': {action_conv_err}")
                         action_int = -1 # Fallback

                    # Fallback if conversion failed
                    if action_int < 0 or action_int >= NUM_ACTIONS:
                         action_int = ACTION_BEST_AVAILABLE

                    try:
                        obs, reward, done, truncated, info = self.eval_env.step(action_int)
                    except Exception as step_err:
                         self.rl_logger.error(f"Optuna Trial {self.trial.number} Eval {self.eval_count} - Error during env.step(action={action_int}): {step_err}")
                         done = True # Force stop this episode on step error

                    if done or truncated:
                        ep_reward = reward # Capture final VORP reward
                    steps_in_ep += 1
                    total_eval_steps += 1 # Increment total eval step counter

                if steps_in_ep >= max_ep_steps:
                     self.rl_logger.warning(f"Optuna Trial {self.trial.number} Eval {self.eval_count} Episode {i_episode+1} hit max steps {max_ep_steps}.")

                # Only append if episode finished naturally or truncated (or forced done by error)
                if done or truncated:
                     # Ensure reward is a float
                     if not isinstance(ep_reward, (float, np.floating)):
                          self.rl_logger.warning(f"Optuna Trial {self.trial.number} Eval {self.eval_count} Ep {i_episode+1} - Invalid reward type {type(ep_reward)}, using 0.0.")
                          ep_reward = 0.0
                     episode_rewards.append(ep_reward)
                else: # Hit max steps without finishing
                     self.rl_logger.warning(f"Optuna Trial {self.trial.number} Eval {self.eval_count} Ep {i_episode+1} did not finish, reward discarded.")


            if episode_rewards: # Check if any episodes finished
                mean_reward = np.mean(episode_rewards)
                self.last_mean_reward = mean_reward

                if self.verbose > 0 or self.rl_logger.isEnabledFor(logging.INFO): # Log if verbose or INFO enabled
                    self.rl_logger.info(f"Optuna Trial {self.trial.number} - Step: {self.num_timesteps} | Eval# {self.eval_count} | Mean Reward: {mean_reward:.3f}")

                # Report intermediate result to Optuna (report the positive reward)
                if np.isfinite(mean_reward):
                    self.trial.report(mean_reward, self.num_timesteps)
                else:
                    self.rl_logger.warning(f"Optuna Trial {self.trial.number} - Non-finite mean reward ({mean_reward}), not reporting.")

                # Prune trial if needed
                if self.trial.should_prune():
                    self.rl_logger.info(f"Optuna Trial {self.trial.number} pruned at step {self.num_timesteps} based on intermediate value: {mean_reward:.3f}.")
                    self._is_pruned = True # Set prune flag
                    raise TrialPruned() # Raise exception to stop learning gracefully
            else:
                # Handle case where no episodes finished during eval
                self.rl_logger.warning(f"Optuna Trial {self.trial.number} - Step: {self.num_timesteps} | Eval# {self.eval_count} | No valid episodes finished during evaluation.")


        return True # Continue training

    def get_last_mean_reward(self) -> float:
        # Return -inf if pruned or no valid eval happened, otherwise the last calculated mean
        if self._is_pruned:
            return -np.inf
        elif self.eval_count > 0 and np.isfinite(self.last_mean_reward):
            return self.last_mean_reward
        else: # No eval happened or last reward was non-finite
            return -np.inf