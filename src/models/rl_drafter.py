"""
Reinforcement Learning Drafter for Fantasy Football

This module implements a reinforcement learning model that learns optimal draft strategies
through simulating many drafts and seasons.
"""
import numpy as np
import pandas as pd
import random
import json
import os
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import defaultdict
import time
import pickle
import copy
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Import simulator classes
from .draft_simulator import DraftSimulator, Team, Player
from .season_simulator import SeasonSimulator, SeasonEvaluator

# Set up logging
logger = logging.getLogger(__name__)

class DraftState:
    """Represents the state of a fantasy football draft for RL"""
    
    def __init__(self, team: Team, available_players: List[Player], 
                 round_num: int, overall_pick: int, league_size: int, 
                 roster_limits: Dict[str, int], max_rounds: int = 16):
        """
        Initialize a draft state
        
        Parameters:
        -----------
        team : Team
            Team making the current pick
        available_players : List[Player]
            List of available players
        round_num : int
            Current draft round
        overall_pick : int
            Overall pick number
        league_size : int
            Number of teams in the league
        roster_limits : Dict[str, int]
            Maximum players by position
        max_rounds : int, optional
            Maximum number of draft rounds
        """
        self.team = team
        self.available_players = available_players
        self.round_num = round_num
        self.overall_pick = overall_pick
        self.league_size = league_size
        self.roster_limits = roster_limits
        self.max_rounds = max_rounds
        
        # Filter players to positions team can draft
        self.valid_positions = [pos for pos in self.roster_limits.keys() 
                               if team.can_draft_position(pos)]
        self.valid_players = [p for p in available_players 
                             if p.position in self.valid_positions]
        
        # Calculate draft progress
        self.draft_progress = (overall_pick - 1) / (league_size * max_rounds)
        
        # Cache some roster stats for features
        self.roster_by_position = {pos: len(team.roster_by_position[pos]) 
                                  for pos in roster_limits.keys()}
        
        # Calculate position needs
        self.position_needs = team.get_position_needs()
    
    def to_feature_vector(self, player_index: int = None) -> np.ndarray:
        """
        Convert state to a feature vector for the RL model
        
        Parameters:
        -----------
        player_index : int, optional
            Index of player being considered
                
        Returns:
        --------
        np.ndarray
            Feature vector
        """
        # If player_index is provided, include player-specific features
        if player_index is not None and player_index < len(self.valid_players):
            player = self.valid_players[player_index]
            
            # Player features
            player_features = [
                player.projected_points,  # Projected points
                getattr(player, 'vbd', player.projected_points),  # VBD
                player.projection_high - player.projection_low,  # Range (uncertainty)
                player.ceiling_projection - player.projected_points,  # Ceiling upside
                
                # Position one-hot encoding
                1 if player.position == 'QB' else 0,
                1 if player.position == 'RB' else 0,
                1 if player.position == 'WR' else 0,
                1 if player.position == 'TE' else 0,
                1 if player.position == 'K' else 0,
                1 if player.position == 'DST' else 0,
                
                # ADP value relative to current pick
                max(0, player.adp - self.overall_pick) if hasattr(player, 'adp') else 0,
            ]
        else:
            # Default player features if no player provided
            player_features = [0] * 11
        
        # Draft state features
        draft_features = [
            self.round_num,  # Current round
            self.overall_pick,  # Overall pick number
            self.draft_progress,  # Draft progress (0 to 1)
            self.team.draft_position,  # Team's draft position
            
            # Picks until next turn
            (2 * self.league_size - (self.overall_pick % (2 * self.league_size))) 
            if (self.overall_pick % (2 * self.league_size)) != 0 
            else 2 * self.league_size,
        ]
        
        # Team composition features
        roster_features = []
        # Use only major positions
        for position in ['QB', 'RB', 'WR', 'TE']:
            roster_features.append(self.roster_by_position.get(position, 0))
            roster_features.append(self.position_needs.get(position, 0))
        
        # Player pool features
        pool_features = []
        # Use only major positions
        for position in ['QB', 'RB', 'WR', 'TE']:
            position_players = [p for p in self.available_players if p.position == position]
            if position_players:
                # Number of players
                pool_features.append(len(position_players))
                
                # Average projected points of top 5 players
                top_players = sorted(position_players, key=lambda p: p.projected_points, reverse=True)[:5]
                avg_points = sum(p.projected_points for p in top_players) / len(top_players)
                pool_features.append(avg_points)
            else:
                pool_features.append(0)  # No players
                pool_features.append(0)  # No average
        
        # Combine all features
        feature_vector = player_features + draft_features + roster_features + pool_features

        
        # Ensure we have exactly 39 features
        # assert len(feature_vector) == 39, f"Feature vector has {len(feature_vector)} features, expected 39"
        
        return np.array(feature_vector, dtype=np.float32)


class RLDrafter:
    """Reinforcement Learning model for fantasy football drafting"""
    
    def __init__(self, learning_rate: float = 0.001, gamma: float = 0.99,
                epsilon_start: float = 1.0, epsilon_end: float = 0.01, 
                epsilon_decay: float = 0.995, batch_size: int = 64,
                memory_size: int = 10000, target_update: int = 100,
                input_dim: int = None):  # Add input_dim parameter
        """
        Initialize the RL drafter
        
        Parameters:
        -----------
        learning_rate : float, optional
            Learning rate for the model
        gamma : float, optional
            Discount factor
        epsilon_start : float, optional
            Starting exploration rate
        epsilon_end : float, optional
            Ending exploration rate
        epsilon_decay : float, optional
            Rate at which to decay epsilon
        batch_size : int, optional
            Batch size for training
        memory_size : int, optional
            Size of replay memory
        target_update : int, optional
            Number of steps between target network updates
        input_dim : int, optional
            Input dimension for neural network (feature vector size)
        """
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.target_update = target_update
        self.input_dim = input_dim  # Store input dimension
        
        # Initialize model
        self.q_network = None
        self.target_network = None
        self.memory = []
        self.optimizer = None
        
        # Training stats
        self.training_steps = 0
        self.episodes = 0
        self.rewards_history = []
        self.win_rates = []
        
        # Try to import tensorflow with proper error handling
        try:
            # Define the neural network architecture
            def create_network(input_shape=None):
                # We'll create the network only if we know the input shape
                if input_shape is None:
                    logger.info("Network creation deferred until input shape is known")
                    return None
                    
                model = Sequential([
                    Dense(128, activation='relu', input_shape=(input_shape,)),
                    Dropout(0.2),
                    Dense(128, activation='relu'),
                    Dropout(0.2),
                    Dense(64, activation='relu'),
                    Dense(1, activation='linear')  # Q-value output
                ])
                model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')
                return model
            
            # If input_dim is specified, create the networks now
            if input_dim is not None:
                self.q_network = create_network(input_dim)
                self.target_network = create_network(input_dim)
                self.target_network.set_weights(self.q_network.get_weights())
            else:
                # Otherwise, defer network creation until we have a sample
                self.create_network_func = create_network
            
            # Use TensorFlow if available
            self.tf_available = True
            logger.info("Using TensorFlow for RL model")
            
        except ImportError:
            # Fall back to a simpler model if TensorFlow is not available
            logger.warning("TensorFlow not found. Using simplified linear RL model.")
            self.tf_available = False
            
            # Initialize simple linear model 
            if input_dim is not None:
                self.weights = np.random.randn(input_dim)
                self.target_weights = self.weights.copy()
            else:
                # Defer weight initialization
                self.weights = None
                self.target_weights = None

    def select_action(self, state: DraftState, training: bool = True) -> int:
        """
        Select an action (player) based on the current state
        """
        # Get available players
        valid_players = state.valid_players
        
        if not valid_players:
            return None
        
        # With probability epsilon, choose a random player (exploration)
        if training and random.random() < self.epsilon:
            return random.randint(0, len(valid_players) - 1)
        
        # Otherwise, choose the player with the highest Q-value (exploitation)
        q_values = []
        
        for i in range(len(valid_players)):
            # Get feature vector for this state-action pair
            features = state.to_feature_vector(i)
            
            # Initialize networks if not done yet
            if self.tf_available and self.q_network is None:
                input_dim = len(features)
                logger.info(f"Initializing networks with input dimension: {input_dim}")
                self.q_network = self.create_network_func(input_dim)
                self.target_network = self.create_network_func(input_dim)
                self.target_network.set_weights(self.q_network.get_weights())
            elif not self.tf_available and self.weights is None:
                input_dim = len(features)
                logger.info(f"Initializing linear model with input dimension: {input_dim}")
                self.weights = np.random.randn(input_dim)
                self.target_weights = self.weights.copy()
            
            # Calculate Q-value
            if self.tf_available:
                # Use TensorFlow model
                q_value = self.q_network.predict(features.reshape(1, -1), verbose=0)[0][0]
            else:
                # Use simple linear model
                q_value = np.dot(features, self.weights)
            
            q_values.append(q_value)
        
        # Choose the action with the highest Q-value
        return np.argmax(q_values)
    
    def update_model(self, state: DraftState, action: int, reward: float, 
                    next_state: Optional[DraftState], done: bool) -> None:
        """
        Update the RL model based on experience
        """
        # Get feature vector for this state-action
        features = state.to_feature_vector(action)
        
        # Initialize networks if they haven't been yet
        if self.tf_available and self.q_network is None:
            input_dim = len(features)
            logger.info(f"Initializing networks with input dimension: {input_dim}")
            self.q_network = self.create_network_func(input_dim)
            self.target_network = self.create_network_func(input_dim)
            self.target_network.set_weights(self.q_network.get_weights())
        
        # Add to memory
        self.memory.append((state, action, reward, next_state, done))
        
        # Limit memory size
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        
        # Only start training once we have enough samples
        if len(self.memory) < self.batch_size:
            return
        
        # Update training steps counter
        self.training_steps += 1
        
        # Only train periodically to reduce overhead
        if self.training_steps % 5 == 0:  # Train every 5 steps 
            # Sample a batch from memory
            indices = random.sample(range(len(self.memory)), self.batch_size)
            batch = [self.memory[i] for i in indices]
            
            # Train on batch
            if self.tf_available:
                self._train_batch_tf(batch)
            else:
                self._train_batch_simple(batch)
        
        # Update target network periodically
        if self.training_steps % self.target_update == 0:
            if self.tf_available:
                self.target_network.set_weights(self.q_network.get_weights())
            else:
                self.target_weights = self.weights.copy()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
    
    def _train_batch_tf(self, batch):
        """
        Train on a batch using TensorFlow - with performance optimizations
        """

        
        # Extract data from batch
        states = []
        actions = []
        rewards = []
        next_states = []
        dones = []
        
        for state, action, reward, next_state, done in batch:
            states.append(state.to_feature_vector(action))
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            # For next states, we'll handle them in a batch below
            next_states.append(next_state)
        
        # Efficiently compute next state values in a single batch
        next_state_values = np.zeros(len(batch))
        
        # Only process valid next states
        valid_indices = [i for i, ns in enumerate(next_states) if ns is not None]
        if valid_indices:
            # For each valid next state, get all valid actions and batch them
            all_next_features = []
            next_state_mapping = []  # To map back to the original indices
            
            for i in valid_indices:
                next_state = next_states[i]
                valid_players = next_state.valid_players
                
                for j in range(len(valid_players)):
                    all_next_features.append(next_state.to_feature_vector(j))
                    next_state_mapping.append(i)  # Remember which next_state this belongs to
            
            if all_next_features:
                # Make a single batch prediction for all actions from all next states
                all_next_features = np.array(all_next_features)
                all_next_q_values = self.target_network.predict(all_next_features, verbose=0).flatten()
                
                # Group by next state and take the max for each
                next_q_by_state = {}
                for j, q_value in enumerate(all_next_q_values):
                    state_idx = next_state_mapping[j]
                    if state_idx not in next_q_by_state or q_value > next_q_by_state[state_idx]:
                        next_q_by_state[state_idx] = q_value
                
                # Set the max Q-values
                for state_idx, q_value in next_q_by_state.items():
                    next_state_values[state_idx] = q_value
        
        # Convert to numpy arrays
        states = np.array(states)
        rewards = np.array(rewards)
        dones = np.array(dones)
        
        # Calculate target Q-values
        target_q = rewards + self.gamma * next_state_values * (1 - dones)
        
        # Train model
        self.q_network.fit(states, target_q, batch_size=self.batch_size, verbose=0)
    
    def _train_batch_simple(self, batch: List[Tuple]) -> None:
        """
        Train on a batch using simple linear model
        
        Parameters:
        -----------
        batch : List[Tuple]
            Batch of experiences
        """
        # Extract data from batch
        for state, action, reward, next_state, done in batch:
            # Get feature vector for this state-action pair
            features = state.to_feature_vector(action)
            
            # Calculate current Q-value
            current_q = np.dot(features, self.weights)
            
            # Calculate target Q-value
            if done:
                target_q = reward
            else:
                # Get best action for next state
                valid_players = next_state.valid_players if next_state else []
                next_q_values = []
                
                for i in range(len(valid_players)):
                    next_features = next_state.to_feature_vector(i)
                    next_q_value = np.dot(next_features, self.target_weights)
                    next_q_values.append(next_q_value)
                
                max_next_q = max(next_q_values) if next_q_values else 0
                target_q = reward + self.gamma * max_next_q
            
            # Update weights using gradient descent
            error = target_q - current_q
            self.weights += self.learning_rate * error * features
    
    def train(self, draft_simulator: DraftSimulator, season_simulator: SeasonSimulator,
          num_episodes: int = 100, eval_interval: int = 10, 
          save_interval: int = 1, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the RL model through multiple draft-season simulations
        
        Parameters:
        -----------
        draft_simulator : DraftSimulator
            Draft simulator instance
        season_simulator : SeasonSimulator
            Season simulator instance
        num_episodes : int, optional
            Number of episodes to train for
        eval_interval : int, optional
            Number of episodes between evaluations
        save_interval : int, optional
            Number of episodes between model saves
        save_path : Optional[str], optional
            Path to save the model
                
        Returns:
        --------
        Dict[str, Any]
            Training results
        """
        logger.info(f"Training RL drafter for {num_episodes} episodes")
        
        # Keep track of best model
        best_reward = -float('inf')
        best_weights = None
        
        # Store original parameters to recreate simulator each episode
        original_players = copy.deepcopy(draft_simulator.players)
        league_size = draft_simulator.league_size
        roster_limits = draft_simulator.roster_limits.copy()
        num_rounds = draft_simulator.num_rounds
        scoring_settings = draft_simulator.scoring_settings.copy()
        
        # Save initial model before training if path is provided
        if save_path:
            self.save_model(save_path, episode=0, is_initial=True)
            logger.info("Saved initial model before training")
        
        # Training loop
        for episode in range(1, num_episodes + 1):
            self.episodes = episode
            logger.info(f"Episode {episode}/{num_episodes} (epsilon: {self.epsilon:.3f})")
            
            # Completely reinitialize the draft simulator for each episode
            fresh_players = copy.deepcopy(original_players)
            draft_simulator = DraftSimulator(
                players=fresh_players,
                league_size=league_size,
                roster_limits=roster_limits,
                num_rounds=num_rounds,
                scoring_settings=scoring_settings,
                rl_model=self  # Inject the RL model into the simulator
            )
            
            # Need to also reinitialize the season simulator with the new teams
            season_simulator = SeasonSimulator(
                teams=draft_simulator.teams,
                num_regular_weeks=season_simulator.num_regular_weeks,
                num_playoff_teams=season_simulator.num_playoff_teams,
                num_playoff_weeks=season_simulator.num_playoff_weeks,
                randomness=season_simulator.randomness
            )
            
            # Track episode data
            episode_rewards = []
            
            # Run the draft simulation
            draft_history = []
            
            # Current state
            current_round = 1
            current_pick = 1
            
            # Run until all rounds are complete
            while current_round <= draft_simulator.num_rounds:
                # Get the team picking
                team_idx = (current_pick - 1) % draft_simulator.league_size
                if current_round % 2 == 0:  # Snake draft
                    team_idx = draft_simulator.league_size - 1 - team_idx
                
                team = draft_simulator.teams[team_idx]
                
                # If this is the RL team, use our model
                if team.strategy == "RL":
                    # Get available players
                    available_players = [p for p in draft_simulator.players if not p.is_drafted]
                    
                    # Create state
                    state = DraftState(
                        team=team,
                        available_players=available_players,
                        round_num=current_round,
                        overall_pick=current_pick,
                        league_size=draft_simulator.league_size,
                        roster_limits=draft_simulator.roster_limits,
                        max_rounds=draft_simulator.num_rounds
                    )
                    
                    # Select action
                    action = self.select_action(state, training=True)
                    
                    # Execute action
                    if action is not None and action < len(state.valid_players):
                        player = state.valid_players[action]
                        team.add_player(player, current_round, current_pick)
                        draft_history.append((state, action, player))
                    else:
                        logger.warning(f"Invalid action: {action}, valid players: {len(state.valid_players)}")
                
                # Otherwise, use the team's strategy
                else:
                    # Get available players
                    available_players = [p for p in draft_simulator.players if not p.is_drafted]
                    
                    # Make the pick (using draft_simulator methods)
                    picked_player = draft_simulator._make_pick(team, current_round, current_pick)
                    
                    if not picked_player:
                        logger.warning(f"Team {team.name} could not make a valid pick!")
                
                # Move to next pick
                current_pick += 1
                if current_pick > current_round * draft_simulator.league_size:
                    current_round += 1
            
            # Simulate the season
            season_results = season_simulator.simulate_season()
            
            # Evaluate season
            evaluation = SeasonEvaluator(draft_simulator.teams, season_results)
            
            # Calculate reward for RL team
            rl_team = next((team for team in draft_simulator.teams if team.strategy == "RL"), None)
            if rl_team:
                # Get team's metrics
                rl_metrics = None
                for metrics in evaluation.metrics["RL"]["teams"]:
                    if metrics["team_name"] == rl_team.name:
                        rl_metrics = metrics
                        break
                
                if rl_metrics:
                    # Design reward function based on performance
                    # This can be customized based on what you want to optimize for
                    reward = (
                        -1.0 * rl_metrics["rank"] +  # Lower rank is better
                        2.0 * rl_metrics["wins"] +
                        0.02 * rl_metrics["points_for"] +
                        10.0 * (1 if rl_metrics.get("playoff_result") == "Champion" else 0) +
                        5.0 * (1 if rl_metrics.get("playoff_result") in ["Runner-up", "Third Place"] else 0) +
                        2.0 * (1 if rl_metrics.get("playoff_result") == "Playoff Qualification" else 0)
                    )
                    
                    # Track episode reward
                    episode_rewards.append(reward)
                    
                    # Update model for each decision made
                    for i, (state, action, player) in enumerate(draft_history):
                        # Get next state
                        next_state = draft_history[i+1][0] if i+1 < len(draft_history) else None
                        
                        # Is this the last action?
                        done = (i == len(draft_history) - 1)
                        
                        # Apply reward shaping - give small reward for each good pick
                        # but the majority of reward comes at the end
                        if done:
                            action_reward = reward
                        else:
                            # Small immediate reward for drafting high-value players
                            # scaled by how far into draft we are
                            draft_progress = state.draft_progress
                            action_reward = player.vbd * (0.1 + 0.4 * draft_progress)
                        
                        # Update model
                        self.update_model(state, action, action_reward, next_state, done)
            
            # Track metrics for this episode
            if episode_rewards:
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                self.rewards_history.append(avg_reward)
                
                # Check if this is the best model so far
                if avg_reward > best_reward:
                    best_reward = avg_reward
                    if self.tf_available:
                        best_weights = self.q_network.get_weights()
                    else:
                        best_weights = self.weights.copy()
                
                logger.info(f"Episode {episode} - Reward: {avg_reward:.2f}, RL Team Rank: {rl_metrics['rank']}")
            
            # Save model more frequently than evaluation
            if save_path and episode % save_interval == 0:
                self.save_model(save_path, episode=episode)
                logger.info(f"Saved model at episode {episode}")
            
            # Evaluate performance periodically
            if episode % eval_interval == 0:
                # Get win rates for each strategy
                win_rates = {}
                for strategy, metrics in evaluation.metrics.items():
                    win_rate = metrics["avg_wins"] / (metrics["avg_wins"] + metrics.get("avg_losses", 0)) if (metrics["avg_wins"] + metrics.get("avg_losses", 0)) > 0 else 0.0
                    win_rates[strategy] = win_rate
                
                self.win_rates.append(win_rates)
                
                # Calculate current win percentage for RL vs baseline
                for strategy, win_rate in win_rates.items():
                    logger.info(f"  {strategy}: Win Rate = {win_rate:.3f}")
        
        # Restore best model
        if best_weights is not None:
            if self.tf_available:
                self.q_network.set_weights(best_weights)
                self.target_network.set_weights(best_weights)
            else:
                self.weights = best_weights
                self.target_weights = best_weights
        
        # Save final model
        if save_path:
            self.save_model(save_path, episode=num_episodes, is_final=True)
        
        # Return training results
        return {
            "rewards_history": self.rewards_history,
            "win_rates": self.win_rates,
            "best_reward": best_reward,
            "final_epsilon": self.epsilon,
            "episodes": self.episodes
        }
    
    def save_model(self, path: str, episode: int = None, is_final: bool = False, is_initial: bool = False) -> str:
        """
        Save the RL model to disk
        
        Parameters:
        -----------
        path : str
            Directory to save the model
        episode : int, optional
            Current episode number
        is_final : bool, optional
            Whether this is the final model
        is_initial : bool, optional
            Whether this is the initial model before training
            
        Returns:
        --------
        str
            Path to the saved model
        """
        os.makedirs(path, exist_ok=True)
        
        if is_final:
            model_path = os.path.join(path, "rl_drafter_final")
        elif is_initial:
            model_path = os.path.join(path, "rl_drafter_initial")
        else:
            model_path = os.path.join(path, f"rl_drafter_episode_{episode}")
        
        # Save the model
        if self.tf_available:
            try:
                # Save TensorFlow model
                self.q_network.save(model_path + ".keras")
            except Exception as e:
                logger.error(f"Error saving TensorFlow model: {e}")
                
                # Fall back to saving weights only
                try:
                    self.q_network.save_weights(model_path + "_weights.keras")
                except Exception as e2:
                    logger.error(f"Error saving TensorFlow weights: {e2}")
        else:
            # Save simple model weights
            with open(model_path + ".pkl", "wb") as f:
                pickle.dump({
                    "weights": self.weights,
                    "target_weights": self.target_weights,
                    "epsilon": self.epsilon,
                    "training_steps": self.training_steps,
                    "episodes": self.episodes
                }, f)
        
        # Save training history
        history_path = os.path.join(path, f"rl_history{'_final' if is_final else '_initial' if is_initial else f'_episode_{episode}'}.json")
        with open(history_path, "w") as f:
            json.dump({
                "rewards_history": self.rewards_history,
                "win_rates": self.win_rates,
                "episodes": self.episodes,
                "epsilon": self.epsilon,
                "training_steps": self.training_steps
            }, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        
        return model_path
    
    @classmethod
    def load_model(cls, path: str) -> "RLDrafter":
        """
        Load a saved model
        
        Parameters:
        -----------
        path : str
            Path to the saved model
            
        Returns:
        --------
        RLDrafter
            Loaded model
        """
        instance = cls()
        
        # Try to load TensorFlow model first
        if os.path.exists(path + ".keras"):
            try:
                import tensorflow as tf
                instance.q_network = tf.keras.models.load_model(path + ".keras")
                instance.target_network = tf.keras.models.clone_model(instance.q_network)
                instance.target_network.set_weights(instance.q_network.get_weights())
                instance.tf_available = True
                logger.info(f"Loaded TensorFlow model from {path}.keras")
            except ImportError:
                logger.warning("TensorFlow not available, falling back to simple model")
                instance.tf_available = False
            except Exception as e:
                logger.error(f"Error loading TensorFlow model: {e}")
                instance.tf_available = False
        
        # If TensorFlow model not available, try weights
        elif os.path.exists(path + "_weights.keras"):
            try:
                # import tensorflow as tf
                # from keras.models import Sequential
                # from keras.layers import Dense, Dropout
                # from keras.optimizers import Adam
                
                # Create model architecture
                instance.q_network = Sequential([
                    Dense(128, activation='relu', input_shape=(39,)),
                    Dropout(0.2),
                    Dense(128, activation='relu'),
                    Dropout(0.2),
                    Dense(64, activation='relu'),
                    Dense(1, activation='linear')
                ])
                instance.q_network.compile(optimizer=Adam(learning_rate=instance.learning_rate), loss='mse')
                
                # Load weights
                instance.q_network.load_weights(path + "_weights.keras")
                
                # Create target network
                instance.target_network = tf.keras.models.clone_model(instance.q_network)
                instance.target_network.set_weights(instance.q_network.get_weights())
                instance.tf_available = True
                logger.info(f"Loaded TensorFlow weights from {path}_weights.keras")
            except ImportError:
                logger.warning("TensorFlow not available, falling back to simple model")
                instance.tf_available = False
            except Exception as e:
                logger.error(f"Error loading TensorFlow weights: {e}")
                instance.tf_available = False
        
        # Try loading simple model
        elif os.path.exists(path + ".pkl"):
            try:
                with open(path + ".pkl", "rb") as f:
                    data = pickle.load(f)
                
                instance.weights = data["weights"]
                instance.target_weights = data["target_weights"]
                instance.epsilon = data["epsilon"]
                instance.training_steps = data["training_steps"]
                instance.episodes = data["episodes"]
                instance.tf_available = False
                logger.info(f"Loaded simple model from {path}.pkl")
            except Exception as e:
                logger.error(f"Error loading simple model: {e}")
                # Initialize a new model
                instance.weights = np.random.randn(39)
                instance.target_weights = instance.weights.copy()
                instance.tf_available = False
        
        # Try loading training history
        history_path = path.replace("rl_drafter", "rl_history") + ".json"
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as f:
                    history = json.load(f)
                
                instance.rewards_history = history.get("rewards_history", [])
                instance.win_rates = history.get("win_rates", [])
                logger.info(f"Loaded training history from {history_path}")
            except Exception as e:
                logger.error(f"Error loading training history: {e}")
        
        return instance