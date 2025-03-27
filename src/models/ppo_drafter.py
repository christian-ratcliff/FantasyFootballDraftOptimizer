"""
PPO-based Reinforcement Learning Drafter for Fantasy Football

This module implements a PPO (Proximal Policy Optimization) reinforcement learning model 
that learns optimal draft strategies through simulating many drafts and seasons.
"""
import numpy as np
import pandas as pd
import joblib
import random
import json
import os
import logging
from typing import List, Dict, Tuple, Optional, Any, Union
from collections import defaultdict, deque
import time
import pickle
import copy
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

from .draft_simulator import DraftSimulator, Team, Player
from .season_simulator import SeasonSimulator, SeasonEvaluator
from src.models.projections import ProjectionModelLoader
from src.models.lineup_evaluator import LineupEvaluator




logger = logging.getLogger(__name__)

class DraftState:
    _model_cache = {}  # Class variable to cache models
    def __init__(self, team, available_players, round_num, overall_pick, 
             league_size, roster_limits, max_rounds=18, 
             projection_models=None, use_top_n_features = 0):
        """Initialize draft state with optional projection models"""
        self.team = team
        self.available_players = available_players
        self.round_num = round_num
        self.overall_pick = overall_pick
        self.league_size = league_size
        self.roster_limits = roster_limits
        self.max_rounds = max_rounds
        self.use_top_n_features = use_top_n_features

        
        # Define mappings for different roster slots to actual player positions
        slot_to_position_map = {
            "QB": ["QB"],
            "RB": ["RB"],
            "WR": ["WR"],
            "TE": ["TE"],
            "K": ["K"],
            "DST": ["DST"],
            "FLEX": ["RB", "WR", "TE"],
            "RB/WR": ["RB", "WR"],
            "WR/TE": ["WR", "TE"],
            "RB/WR/TE": ["RB", "WR", "TE"],
            "OP": ["QB", "RB", "WR", "TE"],  # Offensive Player can be any offensive position
            "DL": ["DL", "DE", "DT"],
            "LB": ["LB"],
            "DB": ["DB", "CB", "S"],
            "DP": ["DL", "DE", "DT", "LB", "DB", "CB", "S"],
            "BE": ["QB", "RB", "WR", "TE", "K", "DST", "DL", "LB", "DB", "DE", "DT", "CB", "S"],  # Bench can hold any position
            "IR": ["QB", "RB", "WR", "TE", "K", "DST", "DL", "LB", "DB", "DE", "DT", "CB", "S"]   # IR can hold any position
        }
        
        # Get valid roster slots the team can still fill
        self.valid_positions = [pos for pos in self.roster_limits.keys() 
                            if team.can_draft_position(pos)]
        
        # Determine valid player positions based on available roster slots
        valid_player_positions = set()
        for slot in self.valid_positions:
            if slot in slot_to_position_map:
                valid_player_positions.update(slot_to_position_map[slot])
            else:
                # For any unrecognized slot type, assume it can hold any position
                valid_player_positions.update(["QB", "RB", "WR", "TE", "K", "DST"])
        
        # Check standard positions first - ensure we have room for each position
        position_counts = {
            "QB": len(team.roster_by_position.get("QB", [])),
            "RB": len(team.roster_by_position.get("RB", [])),
            "WR": len(team.roster_by_position.get("WR", [])),
            "TE": len(team.roster_by_position.get("TE", [])),
            "K": len(team.roster_by_position.get("K", [])), 
            "DST": len(team.roster_by_position.get("DST", []))
        }
        
        # Check against the roster limits
        for pos, count in position_counts.items():
            # If we've reached the limit for a position, remove it from valid positions
            if pos in roster_limits and count >= roster_limits.get(pos, 0):
                if pos in valid_player_positions:
                    valid_player_positions.remove(pos)
        
        # Filter available players to those with valid positions
        self.valid_players = [p for p in available_players 
                            if p.position in valid_player_positions and not p.is_drafted]
        
        # Calculate draft progress
        self.draft_progress = (overall_pick - 1) / (league_size * max_rounds)
        
        # Cache some roster stats for features
        self.roster_by_position = {pos: len(team.roster_by_position.get(pos, [])) 
                                for pos in roster_limits.keys()}
        
        # Calculate position needs
        self.position_needs = team.get_position_needs()
        
        self.projection_models = projection_models or {}
        
        

        
    def _load_projection_models(self, models_dir):
        """
        Load pre-trained projection models for each position
        
        Parameters:
        -----------
        models_dir : str
            Directory containing joblib model files
        
        Returns:
        --------
        dict
            Dictionary of loaded projection models by position
        """
        projection_models = {}
        positions = ['qb', 'rb', 'wr', 'te']
        
        for position in positions:
            model_path = os.path.join(models_dir, f'{position}_model.joblib')
            try:
                model_data = joblib.load(model_path)
                projection_models[position] = model_data
                logger.info(f"Loaded projection model for {position}")
            except Exception as e:
                logger.warning(f"Could not load {position} projection model: {e}")
        
        return projection_models
    
    def to_feature_vector(self) -> np.ndarray:
        """
        Convert state to a feature vector for the RL model with enhanced projection features
        
        Returns:
        --------
        np.ndarray
            Feature vector representing the state
        """
        # Draft state features
        draft_features = [
            self.round_num / self.max_rounds,  # Normalized round
            self.overall_pick / (self.league_size * self.max_rounds),  # Normalized pick
            self.draft_progress,  # Draft progress (0 to 1)
            self.team.draft_position / self.league_size,  # Normalized team position
            
            # Normalized picks until next turn
            (2 * self.league_size - (self.overall_pick % (2 * self.league_size))) 
            / (2 * self.league_size) if (self.overall_pick % (2 * self.league_size)) != 0 
            else 1.0,
        ]
        
        # Team composition features
        roster_features = []
        # Use all major positions including TE (important)
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
            # Current count and max count
            current = self.roster_by_position.get(position, 0)
            max_count = self.roster_limits.get(position, 0)
            
            # Normalized count and needs
            roster_features.append(current / max(1, max_count))
            need = max(0, max_count - current)  # Calculate actual need
            roster_features.append(need / max(1, max_count))  # Normalize need
            
            # Add extra emphasis on unfilled required positions
            if position in ['QB', 'RB', 'WR', 'TE'] and current == 0 and max_count > 0:
                # Add a special feature that signals "no players at this required position"
                roster_features.append(1.0)
            else:
                roster_features.append(0.0)
        
        # Player pool features with projection model insights
        pool_features = []
        # Calculate stats for available players by position
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
            position_players = [p for p in self.available_players if p.position == position]
            
            # Number of players available normalized by typical roster needs
            typical_needs = {'QB': 1, 'RB': 2, 'WR': 3, 'TE': 1, 'K': 1, 'DST': 1}
            normalized_count = len(position_players) / (self.league_size * typical_needs.get(position, 1))
            pool_features.append(min(1.0, normalized_count))
            
            if position_players:
                # Get top projected players
                top_players = sorted(position_players, key=lambda p: p.projected_points, reverse=True)[:5]
                
                # Average points of top 5 (or fewer) players
                avg_points = sum(p.projected_points for p in top_players) / len(top_players)
                
                # Normalize by position-specific baseline
                position_baselines = {'QB': 20, 'RB': 15, 'WR': 15, 'TE': 10, 'K': 5, 'DST': 8}
                normalized_points = avg_points / position_baselines.get(position, 15)
                pool_features.append(min(1.0, normalized_points))
                
                # Quality depth - standard deviation of top players
                if len(top_players) > 1:
                    points = [p.projected_points for p in top_players]
                    std_dev = np.std(points)
                    normalized_std = std_dev / position_baselines.get(position, 15)
                    pool_features.append(min(1.0, normalized_std))
                else:
                    pool_features.append(0.0)
                
                # VBD features - get average VBD of top players
                vbd_values = [getattr(p, 'vbd', 0) for p in top_players]
                avg_vbd = sum(vbd_values) / len(vbd_values) if vbd_values else 0
                max_vbd_baseline = 10.0  # Approximate max expected VBD
                normalized_vbd = min(1.0, max(0, avg_vbd) / max_vbd_baseline)
                pool_features.append(normalized_vbd)
            else:
                # No players of this position available
                pool_features.extend([0.0, 0.0, 0.0])
        
        # Add best available player overall features
        if self.valid_players:
            best_player = max(self.valid_players, key=lambda p: getattr(p, 'vbd', p.projected_points))
            
            # VBD of best player
            best_vbd = getattr(best_player, 'vbd', 0)
            pool_features.append(min(1.0, max(0, best_vbd) / 20.0))  # Normalize assuming 20 is very high VBD
            
            # Position of best player - one-hot encoding
            for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
                pool_features.append(1.0 if best_player.position == position else 0.0)
        else:
            # No valid players
            pool_features.append(0.0)  # No VBD
            pool_features.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # No position
        
        # ADP value available - find best value players
        if self.valid_players:
            adp_values = []
            for player in self.valid_players:
                if hasattr(player, 'adp') and player.adp:
                    adp_diff = max(0, player.adp - self.overall_pick)
                    adp_values.append(adp_diff)
            
            if adp_values:
                max_adp_value = max(adp_values)
                normalized_adp = min(1.0, max_adp_value / 50)  # Normalize with a cap at 50 picks difference
                pool_features.append(normalized_adp)
            else:
                pool_features.append(0.0)
        else:
            pool_features.append(0.0)
        
        # Combine all features
        feature_vector = draft_features + roster_features + pool_features
        
        return np.array(feature_vector, dtype=np.float32)



    def get_action_features(self, player_idx) -> np.ndarray:
        """
        Generate rich feature vector for a player using projection models
        
        Returns:
        --------
        np.ndarray
            Feature vector for the player
        """
        if player_idx >= len(self.valid_players):
            return np.zeros(50, dtype=np.float32)  # Default feature vector
        
        player = self.valid_players[player_idx]
        position = player.position.lower()
        
        # Try using cached model first
        if position in DraftState._model_cache:
            model_data = DraftState._model_cache[position]
            logger.debug(f"Using cached model for {player.name} ({position})")
        # Then try projection models
        elif hasattr(self, 'projection_models') and self.projection_models and position in self.projection_models:
            model_data = self.projection_models[position]
            # Cache for future use
            DraftState._model_cache[position] = model_data
            logger.debug(f"Using projection model for {player.name} ({position}) with {len(model_data.get('features', []))} features")
        # If no model is available, use default features
        else:
            logger.debug(f"No model available for {player.name} ({position})")
            position_encoding = [0, 0, 0, 0]
            position_map = {'qb': 0, 'rb': 1, 'wr': 2, 'te': 3}
            if position in position_map:
                position_encoding[position_map[position]] = 1
            
            position_baselines = {'qb': 20, 'rb': 15, 'wr': 15, 'te': 10}
            baseline = position_baselines.get(position, 15)
            
            normalized_points = player.projected_points / baseline
            uncertainty = (player.projection_high - player.projection_low) / baseline
            upside = (player.ceiling_projection - player.projected_points) / baseline
            
            default_features = np.array(position_encoding + [normalized_points, uncertainty, upside] + [0] * 43, dtype=np.float32)
            return default_features
        
        # Use pre-trained projection model
        model = model_data.get('model')
        if self.use_top_n_features > 0 and 'feature_importances' in model_data and model_data['feature_importances'] is not None:
            # Use only top N features
            top_features = model_data['feature_importances'].nlargest(self.use_top_n_features, 'importance')['feature'].tolist()
            logger.debug(f"Using top {self.use_top_n_features} features for {position}: {top_features}")
        else:
            # Use all features
            top_features = model_data.get('features', [])
        
        try:
            # Create DataFrame with player information
            player_data = {
                'name': player.name,
                'position': player.position,
                'team': player.team,
                'projected_points': player.projected_points,
                'projection_low': player.projection_low,
                'projection_high': player.projection_high,
                'ceiling_projection': player.ceiling_projection
            }
            
            # Add default values for any missing required features
            for feature in top_features:  # Now uses top_features instead of all features
                if feature not in player_data:
                    player_data[feature] = 0.0
            
            # Convert to DataFrame
            df = pd.DataFrame([player_data])
            
            # Select features used in model training
            X = df[top_features].fillna(0) 
            
            # Make prediction
            y_pred = model.predict(X)
            
            # Create a rich feature vector combining model prediction and original player data
            feature_vector = [
                # Model's projected points (first feature)
                y_pred[0],
                
                # Original player projection details
                player.projected_points,
                player.projection_low,
                player.projection_high,
                player.ceiling_projection,
                
                # Position one-hot encoding
                1 if player.position.lower() == 'qb' else 0,
                1 if player.position.lower() == 'rb' else 0,
                1 if player.position.lower() == 'wr' else 0,
                1 if player.position.lower() == 'te' else 0,
                1 if player.position.lower() == 'k' else 0,
                1 if player.position.lower() == 'dst' else 0,
                
                # Additional features from the projection model
                *X.values[0]  # Unpack all features from the projection model
            ]
            
            # Pad or truncate to ensure consistent vector size
            feature_vector = np.array(feature_vector[:50], dtype=np.float32)
            if len(feature_vector) < 50:
                feature_vector = np.pad(feature_vector, (0, 50 - len(feature_vector)), mode='constant')
            
            # Check for NaN or inf values
            if np.isnan(feature_vector).any() or np.isinf(feature_vector).any():
                logger.warning(f"Invalid values in feature vector for {player.name}, using defaults")
                return np.zeros(50, dtype=np.float32)
            
            # Check for extreme values that could destabilize training
            if np.abs(feature_vector).max() > 100:
                logger.warning(f"Extreme values in feature vector for {player.name}, clipping")
                feature_vector = np.clip(feature_vector, -100, 100)
                
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error using projection model for {player.name}: {str(e)}")
            # Fall back to default features
            return np.zeros(50, dtype=np.float32)

class PPOMemory:
    """Memory buffer for PPO training"""
    
    def __init__(self, batch_size=32):
        self.states = []
        self.actions = []
        self.action_features = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
        self.batch_size = batch_size
    
    def store(self, state, action, action_features, probs, vals, reward, done):
        """
        Store a transition in memory
        
        Parameters:
        -----------
        state : np.ndarray
            State vector
        action : int
            Action taken
        action_features : list
            Features for all valid actions
        probs : float
            Probability of taking the action
        vals : float
            Value estimate
        reward : float
            Reward received
        done : bool
            Whether this is a terminal state
        """
        self.states.append(state)
        self.actions.append(action)
        self.action_features.append(action_features)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)
    
    def clear(self):
        """Clear memory"""
        self.states = []
        self.actions = []
        self.action_features = []
        self.probs = []
        self.vals = []
        self.rewards = []
        self.dones = []
    
    def generate_batches(self):
        """
        Generate batch indices for training
        
        Returns:
        --------
        list
            List of batches of indices
        """
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return batches

class PPODrafter:
    """PPO-based Reinforcement Learning model for fantasy football drafting"""
    
    def __init__(self, state_dim, action_feature_dim=7, action_dim=256, 
                 lr_actor=0.0003, lr_critic=0.0003, gamma=0.99, 
                 gae_lambda=0.95, policy_clip=0.2, batch_size=32, 
                 n_epochs=10, entropy_coef=0.01, use_top_n_features = 0):
        """
        Initialize the PPO drafter
        
        Parameters:
        -----------
        state_dim : int
            Dimension of state vector
        action_feature_dim : int
            Dimension of action feature vector
        action_dim : int
            Maximum number of actions (players to choose from)
        lr_actor : float
            Learning rate for actor network
        lr_critic : float
            Learning rate for critic network
        gamma : float
            Discount factor
        gae_lambda : float
            GAE lambda parameter
        policy_clip : float
            PPO clipping parameter
        batch_size : int
            Batch size for training
        n_epochs : int
            Number of epochs per update
        entropy_coef : float
            Entropy coefficient for exploration
        """
        self.state_dim = state_dim
        self.action_feature_dim = action_feature_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.entropy_coef = entropy_coef
        
        # Build actor network (policy)
        self.actor = self._build_actor_network(lr_actor)
        
        # Build critic network (value function)
        self.critic = self._build_critic_network(lr_critic)
        
        # Initialize memory
        self.memory = PPOMemory(batch_size)
        
        # Track metrics
        self.rewards_history = []
        self.win_rates = []
        self.episodes = 0
        self.use_top_n_features = use_top_n_features
    
    def _build_actor_network(self, learning_rate):
        """Build actor network with a simpler and more effective architecture"""
        # Input for state
        state_input = Input(shape=(self.state_dim,), name='state_input')
        
        # Input for action features
        action_features_input = Input(shape=(self.action_dim, self.action_feature_dim), 
                                    name='action_features_input')
        
        # Process state input - wider network
        x = Dense(256, activation='relu')(state_input)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        state_features = Dense(64, activation='relu')(x)
        
        # Process each action using a 1D Conv to better capture feature relationships
        action_conv = tf.keras.layers.TimeDistributed(
            Dense(32, activation='relu')
        )(action_features_input)
        
        action_conv = tf.keras.layers.TimeDistributed(
            Dense(32, activation='relu')
        )(action_conv)
        
        # Expand state features for broadcasting
        state_features_expanded = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=1)
        )(state_features)
        
        state_features_tiled = tf.keras.layers.Lambda(
            lambda x, dim=self.action_dim: tf.tile(x, [1, dim, 1])
        )(state_features_expanded)
        
        # Combine state and action features
        combined = tf.keras.layers.Concatenate(axis=2)([state_features_tiled, action_conv])
        
        # Process combined features
        combined_processed = tf.keras.layers.TimeDistributed(
            Dense(32, activation='relu')
        )(combined)
        
        # Output a single value per action
        logits = tf.keras.layers.TimeDistributed(
            Dense(1, activation=None)
        )(combined_processed)
        
        # Reshape to remove extra dimension
        logits = tf.keras.layers.Reshape((self.action_dim,))(logits)
        
        # Add softmax activation
        probs = tf.keras.layers.Activation('softmax')(logits)
        
        # Create model
        model = Model(inputs=[state_input, action_features_input], 
                    outputs=probs)
        
        # Use a higher learning rate
        model.compile(optimizer=Adam(learning_rate=learning_rate*5))
        
        return model
    
    def _build_critic_network(self, learning_rate):
        """Build critic network (value function)"""
        state_input = Input(shape=(self.state_dim,))
        
        x = Dense(128, activation='relu')(state_input)
        x = Dropout(0.2)(x)
        x = Dense(128, activation='relu')(x)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        
        value = Dense(1, activation=None)(x)
        
        model = Model(inputs=state_input, outputs=value)
        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                     loss='mse')
        
        return model
    
    def select_action(self, state, training=True):
        """
        Select an action (player) based on the current state
        
        Parameters:
        -----------
        state : DraftState
            Current draft state
        training : bool
            Whether we're in training mode
                
        Returns:
        --------
        (int, float, float, np.ndarray)
            Selected action index, action probability, state value, and action features
        """
        # Get valid players from the state
        valid_players = state.valid_players
        
        if not valid_players:
            return None, 0, 0, None
        
        # If there are more valid players than our action dimension,
        # we need to select a subset of them
        if len(valid_players) > self.action_dim:
            # Sort players by projected points and take the top ones
            sorted_players = sorted(valid_players, key=lambda p: getattr(p, 'vbd', p.projected_points), reverse=True)
            valid_players = sorted_players[:self.action_dim]
            
            # Update the state's valid_players for this inference
            # Make a shallow copy of the state and modify it
            state = copy.copy(state)
            state.valid_players = valid_players
        
        # For early exploration in training
        if training and self.episodes < 20 and random.random() < 0.8:
            # Pure exploration in early episodes
            action = random.randint(0, len(valid_players) - 1)
            dummy_features = np.zeros((self.action_dim, self.action_feature_dim))
            return action, 1.0, 0.0, dummy_features
        
        # Get state features and action features
        state_features = state.to_feature_vector()
        state_features = np.expand_dims(state_features, axis=0)
        
        # Create action features array
        action_features = np.zeros((1, self.action_dim, self.action_feature_dim))
        
        # Fill only valid players (which is now guaranteed to be <= self.action_dim)
        for i in range(len(valid_players)):
            action_features[0, i] = state.get_action_features(i)
        
        # Get action probabilities
        try:
            probs = self.actor.predict([state_features, action_features], verbose=0)[0]
            
            # Only consider valid players
            masked_probs = np.zeros_like(probs)
            masked_probs[:len(valid_players)] = probs[:len(valid_players)]
            
            # Normalize 
            sum_probs = np.sum(masked_probs)
            if sum_probs > 0:
                masked_probs = masked_probs / sum_probs
            else:
                # Fallback to uniform distribution
                masked_probs[:len(valid_players)] = 1.0 / len(valid_players)
            
            # Get state value from critic
            value = self.critic.predict(state_features, verbose=0)[0][0]
            
            if training:
                # Sample action with additional exploration
                if random.random() < 0.3:  # 30% chance for pure exploration
                    action = random.randint(0, len(valid_players) - 1)
                else:
                    # Important fix: ensure we're only selecting from valid players
                    valid_indices = np.arange(len(valid_players))
                    valid_probs = masked_probs[:len(valid_players)]
                    # Renormalize valid probs
                    valid_probs = valid_probs / valid_probs.sum() if valid_probs.sum() > 0 else np.ones(len(valid_players))/len(valid_players)
                    action = np.random.choice(valid_indices, p=valid_probs)
            else:
                # During evaluation, choose best action from valid players
                action = np.argmax(masked_probs[:len(valid_players)])
            
            return action, masked_probs[action], value, action_features[0]
        except Exception as e:
            logger.error(f"Error in select_action: {str(e)}")
            # Fallback to random
            action = random.randint(0, len(valid_players) - 1)
            dummy_features = np.zeros((self.action_dim, self.action_feature_dim))
            return action, 1.0, 0.0, dummy_features
    
    def store_transition(self, state, action, action_features, probs, vals, reward, done):
        """Store transition in memory"""
        self.memory.store(state, action, action_features, probs, vals, reward, done)
    
    def update_policy(self):
        """Update policy using PPO"""
        # Return early if not enough samples
        if len(self.memory.states) == 0:
            return 0, 0
        
        # Get values from memory
        states = np.array(self.memory.states)
        actions = np.array(self.memory.actions)
        action_features_list = self.memory.action_features
        old_probs = np.array(self.memory.probs)
        values = np.array(self.memory.vals)
        rewards = np.array(self.memory.rewards)
        dones = np.array(self.memory.dones)
        
        # Calculate advantages using GAE
        advantages = np.zeros(len(rewards))
        gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # For last step, use reward only
                delta = rewards[t] - values[t]
            else:
                # For other steps, use TD error
                delta = rewards[t] + self.gamma * values[t+1] * (1-dones[t]) - values[t]
            
            gae = delta + self.gamma * self.gae_lambda * (1-dones[t]) * gae
            advantages[t] = gae
        
        # Calculate returns
        returns = advantages + values
        
        # Normalize advantages
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-10)
        
        # Update policy for n_epochs
        actor_losses = []
        critic_losses = []
        
        for _ in range(self.n_epochs):
            # Generate batches
            batches = self.memory.generate_batches()
            
            for batch in batches:
                # Extract batch data
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_old_probs = old_probs[batch]
                batch_advantages = advantages[batch]
                batch_returns = returns[batch]
                
                # Extract action features for this batch
                batch_action_features = [action_features_list[i] for i in batch]
                
                # Prepare action features array
                action_features_array = np.zeros((len(batch), self.action_dim, self.action_feature_dim))
                for i, action_features in enumerate(batch_action_features):
                    action_count = min(len(action_features), self.action_dim)
                    for j in range(action_count):
                        action_features_array[i, j] = action_features[j]
                
                with tf.GradientTape(persistent=True) as tape:
                    # Actor forward pass
                    probs = self.actor([batch_states, action_features_array], training=True)
                    
                    # Extract probabilities for taken actions
                    actions_one_hot = tf.one_hot(batch_actions, self.action_dim)
                    action_probs = tf.reduce_sum(probs * actions_one_hot, axis=1)
                    
                    # Calculate ratio
                    ratio = action_probs / (batch_old_probs + 1e-10)
                    
                    # Calculate surrogate losses
                    surr1 = ratio * batch_advantages
                    surr2 = tf.clip_by_value(ratio, 1-self.policy_clip, 1+self.policy_clip) * batch_advantages
                    
                    # Calculate entropy bonus
                    entropy = -tf.reduce_sum(probs * tf.math.log(probs + 1e-10), axis=1)
                    
                    # Actor loss
                    actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2) + self.entropy_coef * entropy)
                    
                    # Critic forward pass
                    values = self.critic(batch_states, training=True)
                    values = tf.squeeze(values)
                    
                    # Critic loss
                    critic_loss = tf.reduce_mean(tf.square(batch_returns - values))
                
                # Calculate gradients
                actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
                critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
                
                # Apply gradients
                self.actor.optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
                self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
                
                # Track losses
                actor_losses.append(actor_loss.numpy())
                critic_losses.append(critic_loss.numpy())
        
        # Clear memory
        self.memory.clear()
        
        return np.mean(actor_losses), np.mean(critic_losses)
    
    
    def train(self, draft_simulator: DraftSimulator, season_simulator: SeasonSimulator,
            num_episodes: int = 100, eval_interval: int = 10, 
            save_interval: int = 1, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Train the RL model through multiple draft-lineup evaluations
        """
        logger.info(f"Training RL drafter for {num_episodes} episodes")
        
        # Keep track of best model
        best_reward = -float('inf')
        best_weights = None
        episode_rewards = []  # Track all episode rewards
        
        # Training metrics tracking
        self.position_distributions = []  # Track positions drafted
        self.value_metrics = []  # Track value metrics
        self.rank_history = []  # Track rankings
        
        # Store original parameters to recreate simulator each episode
        original_players = copy.deepcopy(draft_simulator.players)
        league_size = draft_simulator.league_size
        roster_limits = draft_simulator.roster_limits.copy()
        num_rounds = draft_simulator.num_rounds
        scoring_settings = draft_simulator.scoring_settings.copy()
        
        # Store original strategies to restore them
        original_strategies = ["VBD", "ESPN", "ZeroRB", "HeroRB", "TwoRB", "BestAvailable", "PPO"]
        
        # Save initial model before training if path is provided
        if save_path:
            self.save_model(os.path.join(save_path, "ppo_drafter_initial"), episode=0, is_initial=True)
            logger.info("Saved initial model before training")
        
        # Import LineupEvaluator
        from src.models.lineup_evaluator import LineupEvaluator
        
        # Training loop
        for episode in range(1, num_episodes + 1):
            self.episodes = episode
            rl_draft_position = random.randint(1, league_size)  # Randomly select draft position
            logger.info(f"Episode {episode}/{num_episodes}")
            logger.info(f"Episode {episode}: RL team drafting from position {rl_draft_position}")
            
            # Completely reinitialize the draft simulator for each episode
            fresh_players = copy.deepcopy(original_players)
            draft_simulator = DraftSimulator(
                players=fresh_players,
                league_size=league_size,
                roster_limits=roster_limits,
                num_rounds=num_rounds,
                scoring_settings=scoring_settings
            )
            
            # Make sure we have the projection models in the simulator
            if hasattr(draft_simulator, 'projection_models') and draft_simulator.projection_models is None:
                logger.warning("Projection models not set in draft simulator, loading them")
                projection_loader = ProjectionModelLoader()
                draft_simulator.projection_models = projection_loader.models
            
            # First, ensure ALL teams are using their default strategies (not PPO)
            for i, team in enumerate(draft_simulator.teams):
                # Assign strategies in a cyclic manner using original_strategies, skipping PPO
                valid_strategies = [s for s in original_strategies if s != "PPO"]
                strategy_idx = i % len(valid_strategies)
                team.strategy = valid_strategies[strategy_idx]
            
            # Then, explicitly set ONLY ONE team to use PPO strategy based on draft position
            rl_team = None
            for team in draft_simulator.teams:
                if team.draft_position == rl_draft_position:
                    team.strategy = "PPO"
                    rl_team = team
                    break
            
            if not rl_team:
                # If no team found with that position (shouldn't happen), choose a random one
                rl_team = random.choice(draft_simulator.teams)
                rl_team.strategy = "PPO"
                logger.info(f"Set {rl_team.name} to use PPO strategy")
            
            # Double-check that there's only one PPO team (debugging)
            ppo_teams = [team for team in draft_simulator.teams if team.strategy == "PPO"]
            if len(ppo_teams) != 1:
                logger.error(f"Error: Found {len(ppo_teams)} PPO teams instead of 1!")
                # Fix by keeping only the first PPO team
                for i, team in enumerate(ppo_teams):
                    if i > 0:
                        team.strategy = "VBD"  # Reset extra PPO teams
            
            
            season_simulator = SeasonSimulator(
                        teams=draft_simulator.teams,
                        num_regular_weeks=season_simulator.num_regular_weeks,
                        num_playoff_teams=season_simulator.num_playoff_teams,
                        num_playoff_weeks=season_simulator.num_playoff_weeks,
                        randomness=season_simulator.randomness
                        )
            # Store the model reference in the simulator
            draft_simulator.rl_model = self
            
            # Track this episode's decisions and rewards
            episode_reward = 0
            draft_history = []
            
            # Current state
            current_round = 1
            current_pick = 1
            
            # Run until all rounds are complete
            while current_round <= draft_simulator.num_rounds:
                # Get the team picking - FIXED calculation
                team_idx = (current_pick - 1) % draft_simulator.league_size
                if current_round % 2 == 0:  # Even round, reverse order
                    team_idx = draft_simulator.league_size - 1 - team_idx
                
                team = draft_simulator.teams[team_idx]
                
                # Additional check to make sure we're consistent about draft positions
                if team.draft_position != team_idx + 1:
                    logger.warning(f"Draft position mismatch: team.draft_position={team.draft_position}, but team_idx={team_idx}")

                # If this is the RL team, use our model
                if team.strategy == "PPO":
                    # Get available players
                    available_players = [p for p in draft_simulator.players if not p.is_drafted]
                    
                    # Create state with projection models
                    state = DraftState(
                        team=team,
                        available_players=available_players,
                        round_num=current_round,
                        overall_pick=current_pick,
                        league_size=draft_simulator.league_size,
                        roster_limits=draft_simulator.roster_limits,
                        max_rounds=draft_simulator.num_rounds,
                        projection_models=draft_simulator.projection_models,
                        use_top_n_features=self.use_top_n_features
                    )
                    
                    needs_te = not any(p.position == "TE" for p in team.roster)
                    te_draft_round = current_round >= 6 and current_round <= 12  # Good rounds to draft a TE
                    available_tes = [p for p in available_players if p.position == "TE" and not p.is_drafted]

                    if needs_te and te_draft_round and available_tes and random.random() < 0.35:  # 65% chance to force TE pick
                        # Sort TEs by projected points
                        available_tes.sort(key=lambda p: p.projected_points, reverse=True)
                        te_player = available_tes[0]
                        
                        # Add the TE to the team
                        team.add_player(te_player, current_round, current_pick)
                        logger.info(f"RL team forced to draft TE: {te_player.name} ({te_player.position}) - Round {current_round}, Pick {current_pick}")
                        
                        # Create a simulated action and store in draft history
                        simulated_state = DraftState(
                            team=team,
                            available_players=available_players,
                            round_num=current_round,
                            overall_pick=current_pick,
                            league_size=draft_simulator.league_size,
                            roster_limits=draft_simulator.roster_limits,
                            max_rounds=draft_simulator.num_rounds,
                            use_top_n_features=self.use_top_n_features
                        )
                        
                        # Find equivalent action index
                        simulated_action = next((i for i, p in enumerate(simulated_state.valid_players) 
                                            if p.name == te_player.name), 0)
                        
                        # Add to draft history with high probability and positive reward bias
                        draft_history.append((simulated_state, simulated_action, te_player, 0.9, 0.0, 
                                            np.zeros((self.action_dim, self.action_feature_dim))))
                        # Increment pick counters
                        current_pick += 1
                        if current_pick > current_round * draft_simulator.league_size:
                            current_round += 1
                        # Skip to next pick since we've already made this pick
                        continue
                    # Select action - Note we now get precomputed action features
                    action, prob, value, precomputed_features = self.select_action(state, training=True)
                    
                    # Execute action
                    if action is not None and action < len(state.valid_players):
                        player = state.valid_players[action]
                        team.add_player(player, current_round, current_pick)
                        draft_history.append((state, action, player, prob, value, precomputed_features))
                        logger.info(f"RL team drafted: {player.name} ({player.position}) - Round {current_round}, Pick {current_pick}")
                    else:
                        logger.warning(f"Invalid action: {action}, valid players: {len(state.valid_players)}")
                        # Fallback to best available if action is invalid
                        available_players = [p for p in draft_simulator.players if not p.is_drafted]
                        valid_positions = [pos for pos in draft_simulator.roster_limits.keys() if team.can_draft_position(pos)]
                        valid_players = [p for p in available_players if p.position in valid_positions]
                        
                        if valid_players:
                            # Sort by projected points
                            valid_players.sort(key=lambda p: p.projected_points, reverse=True)
                            player = valid_players[0]
                            team.add_player(player, current_round, current_pick)
                            logger.info(f"RL team fallback draft: {player.name} ({player.position})")
                
                # Otherwise, use the team's strategy
                else:
                    # Make the pick (using draft_simulator methods)
                    picked_player = draft_simulator._make_pick(team, current_round, current_pick, ppo_model=self)
                    
                    if not picked_player:
                        logger.warning(f"Team {team.name} could not make a valid pick!")
                
                # Move to next pick
                current_pick += 1
                if current_pick > current_round * draft_simulator.league_size:
                    current_round += 1
                    
                if current_pick > draft_simulator.num_rounds * draft_simulator.league_size:
                    logger.info(f"Reached maximum picks ({current_pick-1}), ending draft")
                    break
                    
            # Validate team positions after draft
            for team in draft_simulator.teams:
                self.validate_team_positions(team, roster_limits, 'after_draft')
            # Use the LineupEvaluator instead of SeasonSimulator
            # lineup_evaluator = LineupEvaluator(
            #     teams=draft_simulator.teams,
            #     num_weeks=17,  # NFL season
            #     randomness=0.2,
            #     injury_chance=0.05
            # )
            # lineup_results = lineup_evaluator.evaluate_teams()
            
            # # Use the SeasonEvaluator with lineup results
            # evaluation = SeasonEvaluator(draft_simulator.teams, lineup_results)
            
            season_results = season_simulator.simulate_season()
            evaluation = SeasonEvaluator(draft_simulator.teams, season_results)
            
            
            # Calculate reward for RL team
            # Calculate reward for RL team
            rl_metrics = None

            # First check if PPO strategy metrics exist
            if "PPO" in evaluation.metrics:
                for metrics in evaluation.metrics["PPO"]["teams"]:
                    if metrics["team_name"] == rl_team.name:
                        rl_metrics = metrics
                        break

            # If not found by strategy, try finding by team name directly
            if not rl_metrics:
                # Log this to understand what's happening
                logger.info(f"Searching for RL team '{rl_team.name}' in all strategy metrics")
                for strategy, strategy_metrics in evaluation.metrics.items():
                    for metrics in strategy_metrics.get("teams", []):
                        if metrics["team_name"] == rl_team.name:
                            rl_metrics = metrics
                            logger.info(f"Found RL team under strategy: {strategy}")
                            break
                    if rl_metrics:
                        break

            # If still not found, try looking directly in standings
            if not rl_metrics:
                logger.info(f"Looking for RL team in standings directly")
                
                # Create basic metrics from standings if possible
                # if "standings" in lineup_results:
                #     for standing in lineup_results["standings"]:
                if "standings" in season_results:
                    for standing in season_results["standings"]:
                        if standing.get("team") == rl_team.name:
                            logger.info(f"Found RL team in standings directly")
                            # Construct minimal metrics
                            rl_metrics = {
                                "team_name": rl_team.name,
                                "rank": standing.get("rank", 10),  # Default to last place if missing
                                "wins": standing.get("wins", 0),
                                "points_for": standing.get("points_for", 0),
                                "playoff_result": "Champion" if standing.get("rank", 10) == 1 else 
                                                "Runner-up" if standing.get("rank", 10) == 2 else
                                                "Third Place" if standing.get("rank", 10) == 3 else
                                                "Playoff Qualification" if standing.get("rank", 10) <= 6 else
                                                "Missed Playoffs"
                            }
                            break

            if rl_metrics:
                # Enhanced reward function with draft quality components
                
                # 1. Base reward from performance evaluation
                base_reward = (
                    -3.0 * rl_metrics["rank"] +  # Lower rank is better (rank 1 = first place)
                    2.0 * rl_metrics["wins"] +  # More wins is better
                    0.01 * rl_metrics["points_for"] +  # More points is better
                    15.0 * (1 if rl_metrics.get("playoff_result") == "Champion" else 0) +
                    7.0 * (1 if rl_metrics.get("playoff_result") in ["Runner-up", "Third Place"] else 0) +
                    3.0 * (1 if rl_metrics.get("playoff_result") == "Playoff Qualification" else 0)
                )

                # 2. Draft quality reward
                draft_quality_reward = 0.0
                vbd_sum = 0.0
                position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "K": 0, "DST": 0}

                for player in rl_team.roster:
                    # Track positions drafted
                    if player.position in position_counts:
                        position_counts[player.position] += 1
                    
                    # Reward for drafting high-value players
                    vbd = getattr(player, 'vbd', 0)
                    vbd_sum += max(0, vbd)  # Only count positive VBD

                # Scale VBD reward
                draft_quality_reward += vbd_sum * 0.1

                # 3. Roster balance reward
                starter_points = rl_team.get_starting_lineup_points()
                total_points = rl_team.get_total_projected_points()
                # Higher ratio means more efficient roster construction
                roster_efficiency = starter_points / max(1, total_points)
                roster_balance_reward = roster_efficiency * 5.0  # Scale appropriately

                # 4. Position requirements penalty
                position_requirements = {
                    "QB": 1,  # Need at least 1 QB
                    "RB": 4,  # Need at least 2 RBs
                    "WR": 4,  # Need at least 2 WRs
                    "TE": 1  # Need at least 1 TE
                }

                position_penalty = 0
                for pos, requirement in position_requirements.items():
                    # Check if position requirements are met
                    if position_counts.get(pos, 0) < requirement:
                        # Heavy penalty for each missing required position
                        position_penalty += 10.0 * (requirement - position_counts.get(pos, 0))
                        logger.warning(f"Position requirement not met: {pos} - needed {requirement}, got {position_counts.get(pos, 0)}")

                # Combined reward with position penalty
                reward = base_reward + draft_quality_reward + roster_balance_reward - position_penalty

                # Add a baseline adjustment so rewards aren't always negative
                reward += 10.0

                # Log the reward components
                logger.info(f"  Reward components - Base: {base_reward:.2f}, Draft Quality: {draft_quality_reward:.2f}")
                logger.info(f"  Balance: {roster_balance_reward:.2f}, Position Penalty: {position_penalty:.2f}")
                logger.info(f"  Final Reward: {reward:.2f}")
                # Track episode reward
                episode_reward = reward
                episode_rewards.append(reward)
                self.rewards_history.append(reward)  # Add to main history
                
                # Track position distribution for learning verification
                self.position_distributions.append(position_counts)
                
                # Track value metrics
                self.value_metrics.append({
                    "vbd_sum": vbd_sum,
                    "roster_efficiency": roster_efficiency,
                    "starter_points": starter_points,
                    "total_points": total_points
                })
                
                # Track rank history
                self.rank_history.append(rl_metrics["rank"])
                
                # Check if this is the best model so far
                if reward > best_reward:
                    best_reward = reward
                    # Save both weights and a copy of the model
                    if hasattr(self.actor, 'get_weights'):
                        best_weights = [w.copy() for w in self.actor.get_weights()]
                    
                    if save_path:
                        self.save_model(os.path.join(save_path, "ppo_drafter_best"))
                        logger.info(f"New best model with reward: {reward:.2f}")
                
                # Process draft history for learning
                if draft_history:
                    # Distribute rewards across draft picks with diminishing returns
                    # Earlier picks are more important than later ones
                    num_picks = len(draft_history)
                    
                    for i, (state, action, player, prob, value, precomputed_features) in enumerate(draft_history):
                        # Calculate pick-specific reward based on position
                        # Earlier picks get more reward weight
                        pick_weight = 1.0 - 0.5 * (i / num_picks)
                        action_reward = reward * pick_weight
                        
                        # For the first few episodes, add an auxiliary reward based on player quality
                        if episode <= 10:  # First 10 episodes get this guidance
                            # Add reward based on player quality (VBD)
                            vbd_value = getattr(player, 'vbd', 0)
                            if vbd_value > 0:
                                position_factor = {'QB': 0.8, 'RB': 1.2, 'WR': 1.0, 'TE': 0.9}.get(player.position, 1.0)
                                aux_reward = vbd_value * position_factor * 0.5
                                action_reward += aux_reward
                        
                        # Determine if this is the last action in the sequence
                        done = (i == num_picks - 1)
                        
                        # Store in PPO memory for learning - using precomputed features
                        self.memory.store(
                            state.to_feature_vector(),
                            action,
                            precomputed_features,  # Use precomputed features from earlier
                            prob,
                            value,
                            action_reward,
                            done
                        )
                    
                    # Perform PPO policy updates if we have enough samples
                    if len(self.memory.states) >= self.batch_size:
                        # Perform multiple updates for more efficient learning
                        total_actor_loss = 0
                        total_critic_loss = 0
                        update_iterations = 2  # Number of updates to perform
                        
                        for update_iter in range(update_iterations):
                            actor_loss, critic_loss = self.update_policy()
                            total_actor_loss += actor_loss
                            total_critic_loss += critic_loss
                        
                        # Log average losses
                        avg_actor_loss = total_actor_loss / update_iterations
                        avg_critic_loss = total_critic_loss / update_iterations
                        logger.info(f"  Updated model {update_iterations} times - Avg Actor loss: {avg_actor_loss:.4f}, Avg Critic loss: {avg_critic_loss:.4f}")
                    else:
                        logger.info(f"  Not enough samples for update ({len(self.memory.states)}/{self.batch_size})")
                else:
                    logger.warning("No draft history for RL team!")
            else:
                logger.warning("Could not find RL team metrics for reward calculation!")
            
            # Save model at defined intervals
            if save_path and episode % save_interval == 0:
                self.save_model(os.path.join(save_path, f"ppo_drafter_episode_{episode}"), episode=episode)
                logger.info(f"Saved model at episode {episode}")
            
            # Evaluate performance periodically
            if episode % eval_interval == 0:
                # Get win rates for each strategy
                win_rates = {}
                for strategy, metrics in evaluation.metrics.items():
                    # Calculate win rate safely
                    wins = metrics.get("avg_wins", 0)
                    losses = metrics.get("avg_losses", 0) 
                    total_games = wins + losses
                    win_rate = wins / max(total_games, 1)  # Avoid division by zero
                    win_rates[strategy] = win_rate
                
                self.win_rates.append(win_rates)
                
                # Calculate current win percentage for RL vs baseline
                logger.info(f"Win rates after episode {episode}:")
                for strategy, win_rate in win_rates.items():
                    logger.info(f"  {strategy}: Win Rate = {win_rate:.3f}")
                self.update_training_plots(save_path, episode)
                # Log learning verification metrics
                avg_rank = sum(self.rank_history[-eval_interval:]) / min(eval_interval, len(self.rank_history[-eval_interval:]))
                avg_vbd = sum(metric["vbd_sum"] for metric in self.value_metrics[-eval_interval:]) / min(eval_interval, len(self.value_metrics[-eval_interval:]))
                avg_efficiency = sum(metric["roster_efficiency"] for metric in self.value_metrics[-eval_interval:]) / min(eval_interval, len(self.value_metrics[-eval_interval:]))
                
                logger.info(f"Learning metrics (last {eval_interval} episodes):")
                logger.info(f"  Avg Rank: {avg_rank:.2f}")
                logger.info(f"  Avg VBD: {avg_vbd:.2f}")
                logger.info(f"  Avg Roster Efficiency: {avg_efficiency:.2f}")
                
                # Show position distribution
                pos_dist = {}
                for counts in self.position_distributions[-eval_interval:]:
                    for pos, count in counts.items():
                        pos_dist[pos] = pos_dist.get(pos, 0) + count
                
                total_players = sum(pos_dist.values())
                if total_players > 0:
                    logger.info("  Position distribution:")
                    for pos, count in sorted(pos_dist.items()):
                        logger.info(f"    {pos}: {count} ({count/total_players*100:.1f}%)")

        # Restore best model if we found one
        if best_weights is not None and hasattr(self.actor, 'set_weights'):
            self.actor.set_weights(best_weights)
            self.target_network.set_weights(best_weights)
            logger.info("Restored best model weights from training")
        
        # Save final model
        if save_path:
            self.save_model(os.path.join(save_path, f"ppo_drafter_final"), is_final=True)
            logger.info("Saved final model")
        
        # Return training results
        return {
            "rewards_history": self.rewards_history,
            "win_rates": self.win_rates,
            "best_reward": best_reward,
            "final_epsilon": self.epsilon if hasattr(self, 'epsilon') else None,
            "episodes": self.episodes,
            "episode_rewards": episode_rewards,
            "position_distributions": self.position_distributions,
            "value_metrics": self.value_metrics,
            "rank_history": self.rank_history
        }
    
    
    # def save_model(self, path, episode=None, is_initial=False, is_final=False):
    #     """
    #     Save the PPO model
        
    #     Parameters:
    #     -----------
    #     path : str
    #         Path to save the model
    #     episode : int, optional
    #         Current episode number for naming
    #     is_initial : bool, optional
    #         Whether this is the initial model before training
    #     is_final : bool, optional
    #         Whether this is the final model after training
    #     """
    #     # Save actor weights instead of the full model
    #     self.actor.save_weights(f"{path}_actor.weights.h5")
        
    #     # Save critic weights
    #     self.critic.save_weights(f"{path}_critic.weights.h5")
        
    #     # Save training history and metadata
    #     history = {
    #         "rewards_history": self.rewards_history,
    #         "win_rates": self.win_rates,
    #         "episodes": self.episodes,
    #         "state_dim": self.state_dim,
    #         "action_dim": self.action_dim,
    #         "action_feature_dim": self.action_feature_dim
    #     }
        
    #     with open(f"{path}_history.json", "w") as f:
    #         # Convert numpy arrays to lists for JSON serialization
    #         for key, value in history.items():
    #             if isinstance(value, np.ndarray):
    #                 history[key] = value.tolist()
            
    #         json.dump(history, f, indent=2)
        
    #     logger.info(f"Model weights saved to {path}")
    
    def save_model(self, path, episode=None, is_initial=False, is_final=False):
        """
        Save the PPO model
        
        Parameters:
        -----------
        path : str
            Path to save the model
        episode : int, optional
            Current episode number for naming
        is_initial : bool, optional
            Whether this is the initial model before training
        is_final : bool, optional
            Whether this is the final model after training
        """
        # Save actor weights instead of the full model
        self.actor.save_weights(f"{path}_actor.weights.h5")
        
        # Save critic weights
        self.critic.save_weights(f"{path}_critic.weights.h5")
        
        # Save training history and metadata with enhanced metrics
        history = {
            "rewards_history": self.rewards_history,
            "win_rates": self.win_rates,
            "episodes": self.episodes,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_feature_dim": self.action_feature_dim
        }
        
        # Add enhanced training metrics if available
        if hasattr(self, 'position_distributions'):
            history["position_distributions"] = self.position_distributions
        
        if hasattr(self, 'value_metrics'):
            history["value_metrics"] = self.value_metrics
        
        if hasattr(self, 'rank_history'):
            history["rank_history"] = self.rank_history
        
        with open(f"{path}_history.json", "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            for key, value in history.items():
                if isinstance(value, np.ndarray):
                    history[key] = value.tolist()
            
            json.dump(history, f, indent=2)
        
        logger.info(f"Model weights saved to {path}")
    
    @classmethod
    def load_model(cls, path, use_top_n_features=0):
        """
        Load a saved PPO model
        
        Parameters:
        -----------
        path : str
            Path to the saved model
            
        Returns:
        --------
        PPODrafter
            Loaded model
        """
        # Load history and metadata
        try:
            with open(f"{path}_history.json", "r") as f:
                history = json.load(f)
            
            state_dim = history.get("state_dim", 20)
            action_dim = history.get("action_dim", 64)
            action_feature_dim = history.get("action_feature_dim", 7)
        except FileNotFoundError:
            logger.warning(f"History file not found, using default dimensions")
            state_dim = 20
            action_dim = 256
            action_feature_dim = 7
        
        # Create instance
        instance = cls(state_dim=state_dim, action_dim=action_dim, 
                    action_feature_dim=action_feature_dim, use_top_n_features = use_top_n_features)
        
        # Initialize the actor and critic networks
        # This will create the networks with the right architecture
        # Create a dummy state to initialize the networks
        dummy_state = np.zeros((1, state_dim))
        dummy_action_features = np.zeros((1, action_dim, action_feature_dim))
        
        # Forward pass to build the networks
        instance.actor([dummy_state, dummy_action_features])
        instance.critic(dummy_state)
        
        # Load weights if available
        try:
            instance.actor.load_weights(f"{path}_actor.weights.h5")
            logger.info(f"Actor weights loaded from {path}_actor.weights.h5")
        except:
            logger.error(f"Failed to load actor weights from {path}_actor.weights.h5")
        
        try:
            instance.critic.load_weights(f"{path}_critic_weights.h5")
            logger.info(f"Critic weights loaded from {path}_critic.weights.h5")
        except:
            logger.error(f"Failed to load critic weights from {path}_critic.weights.h5")
        
        # Load history if available
        if history:
            instance.rewards_history = history.get("rewards_history", [])
            instance.win_rates = history.get("win_rates", [])
            instance.episodes = history.get("episodes", 0)
        
        return instance
    
    
    def generate_learning_visualizations(ppo_model, output_dir=None):
        """
        Generate visualizations to verify model learning
        
        Parameters:
        -----------
        ppo_model : PPODrafter
            Trained PPO model
        output_dir : str, optional
            Directory to save visualizations
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Check if we have necessary data
        if not hasattr(ppo_model, 'rewards_history') or not ppo_model.rewards_history:
            logger.warning("No reward history available for visualization")
            return
        
        # Create plots using matplotlib
        plt.figure(figsize=(12, 8))
        
        # 1. Reward Progression
        plt.subplot(2, 2, 1)
        plt.plot(ppo_model.rewards_history)
        plt.title('Reward Progression')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        
        # 2. Rank History (if available)
        if hasattr(ppo_model, 'rank_history') and ppo_model.rank_history:
            plt.subplot(2, 2, 2)
            plt.plot(ppo_model.rank_history)
            plt.title('Team Ranking Progression')
            plt.xlabel('Episode')
            plt.ylabel('Rank (lower is better)')
            plt.grid(True)
            # Use reversed y-axis so lower rank (better) is at the top
            plt.gca().invert_yaxis()
        
        # 3. Position Distribution Over Time (if available)
        if hasattr(ppo_model, 'position_distributions') and ppo_model.position_distributions:
            plt.subplot(2, 2, 3)
            
            # Extract position counts for each episode
            positions = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
            position_data = {pos: [] for pos in positions}
            
            for dist in ppo_model.position_distributions:
                for pos in positions:
                    position_data[pos].append(dist.get(pos, 0))
            
            # Plot counts by position
            for pos in positions:
                if any(position_data[pos]):  # Only plot if position has data
                    plt.plot(position_data[pos], label=pos)
            
            plt.title('Position Distribution')
            plt.xlabel('Episode')
            plt.ylabel('Count')
            plt.legend()
            plt.grid(True)
        
        # 4. Value Metrics (if available)
        if hasattr(ppo_model, 'value_metrics') and ppo_model.value_metrics:
            plt.subplot(2, 2, 4)
            
            # Extract metrics
            vbd_values = [metric.get('vbd_sum', 0) for metric in ppo_model.value_metrics]
            efficiency_values = [metric.get('roster_efficiency', 0) for metric in ppo_model.value_metrics]
            
            # Plot two metrics on the same axis
            plt.plot(vbd_values, label='Total VBD')
            plt.plot(efficiency_values, label='Roster Efficiency')
            
            plt.title('Draft Value Metrics')
            plt.xlabel('Episode')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True)
        
        plt.tight_layout()
        
        # Save the combined plot
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'learning_verification.png'))
            logger.info(f"Learning verification plots saved to {os.path.join(output_dir, 'learning_verification.png')}")
        
        # Create a separate plot for win rates over time
        if hasattr(ppo_model, 'win_rates') and ppo_model.win_rates:
            plt.figure(figsize=(10, 6))
            
            # Extract strategies
            strategies = list(ppo_model.win_rates[0].keys())
            episodes = list(range(len(ppo_model.win_rates) * ppo_model.eval_interval if hasattr(ppo_model, 'eval_interval') else 10, 
                                (len(ppo_model.win_rates) + 1) * ppo_model.eval_interval if hasattr(ppo_model, 'eval_interval') else 10 * (len(ppo_model.win_rates) + 1), 
                                ppo_model.eval_interval if hasattr(ppo_model, 'eval_interval') else 10))
            
            # Plot win rate for each strategy
            for strategy in strategies:
                win_rates = [wr.get(strategy, 0) for wr in ppo_model.win_rates]
                plt.plot(episodes, win_rates, label=strategy)
            
            plt.title('Win Rates by Strategy')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate')
            plt.legend()
            plt.grid(True)
            
            # Save the win rates plot
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'win_rates.png'))
                logger.info(f"Win rates plot saved to {os.path.join(output_dir, 'win_rates.png')}")

    plt.close('all')  # Close all figures to free memory
    
    
    def validate_team_positions(self, team, roster_limits, stage='after_draft'):
        """
        Validate that a team's roster meets position limits
        
        Parameters:
        -----------
        team : Team
            Team to validate
        roster_limits : dict
            Dictionary of position limits
        stage : str
            Stage of validation (for logging)
            
        Returns:
        --------
        bool
            True if valid, False otherwise
        """
        logger.info(f"Validating team {team.name} positions at {stage}")
        
        # Check each position
        position_issues = []
        for position, limit in roster_limits.items():
            # Ignore bench and IR positions for this check
            if position in ['BE', 'IR']:
                continue
            
            current_count = len(team.roster_by_position.get(position, []))
            logger.info(f"  {position}: {current_count}/{limit}")
            
            if current_count > limit:
                position_issues.append(f"{position}: {current_count}/{limit}")
        
        # Log overall roster
        total_players = sum(len(players) for position, players in team.roster_by_position.items())
        logger.info(f"  Total players: {total_players}")
        
        # Report any issues
        if position_issues:
            logger.warning(f"Position limit issues for {team.name}: {', '.join(position_issues)}")
            return False
        
        return True
    
    

    def update_training_plots(self, ppo_models_dir, episode):
        """Update training plots during training"""
        # Reward history plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.rewards_history)
        plt.title(f'PPO Training Progress (Episode {episode})')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(os.path.join(ppo_models_dir, 'training_progress_current.png'))
        plt.close()
        
        # Win rates plot if available
        if self.win_rates:
            plt.figure(figsize=(12, 6))
            eval_interval = episode // len(self.win_rates)
            strategies = list(self.win_rates[0].keys())
            
            for strategy in strategies:
                win_rates = [wr.get(strategy, 0) for wr in self.win_rates]
                plt.plot(
                    range(eval_interval, episode + 1, eval_interval),
                    win_rates,
                    label=strategy
                )
            
            plt.title(f'Win Rates by Strategy (Episode {episode})')
            plt.xlabel('Episode')
            plt.ylabel('Win Rate')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(ppo_models_dir, 'win_rates_current.png'))
            plt.close()


    