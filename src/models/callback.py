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