import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input, Lambda, Concatenate
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
import logging
import random
import copy
from collections import deque
import os

from .draft_simulator import DraftSimulator, Team, Player
from .season_simulator import SeasonSimulator, SeasonEvaluator
from src.models.ppo_drafter import DraftState

logger = logging.getLogger(__name__)

class HierarchicalPPODrafter:
    """PPO-based Reinforcement Learning model with hierarchical policy for fantasy football drafting"""
    
    def __init__(self, state_dim, action_feature_dim=50, action_dim=256, 
                 lr_meta=0.0003, lr_sub=0.0003, lr_critic=0.0003, 
                 gamma=0.99, gae_lambda=0.95, policy_clip=0.2, 
                 batch_size=32, n_epochs=10, entropy_coef=0.01,
                 use_top_n_features=0, curriculum_enabled=True,
                 opponent_modeling_enabled=True):
        """
        Initialize the hierarchical PPO drafter
        
        Parameters:
        -----------
        state_dim : int
            Dimension of state vector
        action_feature_dim : int
            Dimension of action feature vector
        action_dim : int
            Maximum number of actions (players to choose from)
        lr_meta : float
            Learning rate for meta policy network
        lr_sub : float
            Learning rate for sub policy networks
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
        use_top_n_features : int
            Number of top features to use from projection models
        curriculum_enabled : bool
            Whether to use curriculum learning
        opponent_modeling_enabled : bool
            Whether to use opponent modeling
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
        self.use_top_n_features = use_top_n_features
        self.curriculum_enabled = curriculum_enabled
        self.opponent_modeling_enabled = opponent_modeling_enabled
        
        # Define positions for sub-policies
        self.positions = ["QB", "RB", "WR", "TE", "K", "DST"]
        
        # Build meta-policy network (position selection)
        self.meta_policy = self._build_meta_policy(lr_meta)
        
        # Build sub-policy networks (player selection for each position)
        self.sub_policies = {}
        for position in self.positions:
            self.sub_policies[position] = self._build_sub_policy(position, lr_sub)
        
        # Build critic network (value function)
        self.critic = self._build_critic_network(lr_critic)
        
        # Initialize memory
        self.memory = HierarchicalPPOMemory(batch_size)
        
        # Track metrics
        self.rewards_history = []
        self.win_rates = []
        self.episodes = 0
        
        # Curriculum learning attributes
        self.curriculum_phase = 1  # Start at phase 1
        self.phase_episode_count = 0
        self.phase_rewards = []
        self.phase_thresholds = {
            1: 0.7,  # 70% valid rosters
            2: 100.0,  # Minimum total projected points
            3: 120.0,  # Minimum starter points
        }
        self.phase_durations = {
            1: 200,  # Episodes in phase 1
            2: 500,  # Episodes in phase 2
            3: 700,  # Episodes in phase 3
            4: float('inf')  # Phase 4 continues until end
        }
        self.curriculum_rewards_history = {phase: [] for phase in range(1, 5)}
        
        # Position requirements for valid roster
        self.position_requirements = {
            "QB": 1,  # Need at least 1 QB
            "RB": 3,  # Need at least 2 RBs
            "WR": 4,  # Need at least 2 WRs
            "TE": 1,  # Need at least 1 TE
        }
        
        # Advanced curriculum features
        self.reward_stats = {phase: {"mean": 0, "std": 1} for phase in range(1, 5)}
        self.phase_transition_episodes = []
        self.curriculum_temperature = 0.0  # Starts at 0, increases over time
        self.reward_mix_weights = {  # How much to include previous phase rewards
            1: [1.0, 0.0, 0.0, 0.0],
            2: [0.3, 0.7, 0.0, 0.0],
            3: [0.1, 0.3, 0.6, 0.0],
            4: [0.05, 0.15, 0.3, 0.5]
        }
        
        # Opponent modeling statistics
        self.opponent_model_stats = {
            "position_accuracy": [],
            "run_detection_rate": [],
            "value_cliff_usage": []
        }
        
        # Position priority weights for adaptive adjustment
        self.position_priority_weights = {
            "QB": 1.0,
            "RB": 1.0,
            "WR": 1.0,
            "TE": 1.0,
            "K": 0.5,
            "DST": 0.5
        }
        
        # Initialize exponential moving averages for adaptive position weights
        self.position_ema = {pos: 1.0 for pos in self.position_priority_weights}
        self.ema_alpha = 0.1  # EMA update rate
    
    def _build_meta_policy(self, learning_rate):
        """
        Build meta-policy network for position selection
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for optimizer
            
        Returns:
        --------
        Model
            Tensorflow model for meta-policy
        """
        # Input for state
        state_input = Input(shape=(self.state_dim,), name='meta_state_input')
        
        # Process state input
        x = Dense(128, activation='relu')(state_input)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        
        # Output position probabilities (one for each position)
        position_logits = Dense(len(self.positions), activation=None)(x)
        position_probs = tf.keras.layers.Activation('softmax')(position_logits)
        
        # Create model
        model = Model(inputs=state_input, outputs=position_probs)
        model.compile(optimizer=Adam(learning_rate=learning_rate))
        
        return model
    
    def _build_sub_policy(self, position, learning_rate):
        """
        Build sub-policy network for player selection within a position
        
        Parameters:
        -----------
        position : str
            Position this sub-policy handles
        learning_rate : float
            Learning rate for optimizer
            
        Returns:
        --------
        Model
            Tensorflow model for sub-policy
        """
        # Input for state
        state_input = Input(shape=(self.state_dim,), name=f'sub_state_input_{position}')
        
        # Input for position-specific players' features
        player_features_input = Input(shape=(self.action_dim, self.action_feature_dim), 
                                     name=f'player_features_input_{position}')
        
        # Process state input
        x = Dense(128, activation='relu')(state_input)
        x = Dropout(0.2)(x)
        state_features = Dense(64, activation='relu')(x)
        
        # Process player features
        player_conv = tf.keras.layers.TimeDistributed(
            Dense(32, activation='relu')
        )(player_features_input)
        
        # Expand state features for broadcasting
        state_features_expanded = tf.keras.layers.Lambda(
            lambda x: tf.expand_dims(x, axis=1)
        )(state_features)
        
        state_features_tiled = tf.keras.layers.Lambda(
            lambda x, dim=self.action_dim: tf.tile(x, [1, dim, 1])
        )(state_features_expanded)
        
        # Combine state and player features
        combined = tf.keras.layers.Concatenate(axis=2)([state_features_tiled, player_conv])
        
        # Process combined features
        combined_processed = tf.keras.layers.TimeDistributed(
            Dense(32, activation='relu')
        )(combined)
        
        # Output a single value per player
        logits = tf.keras.layers.TimeDistributed(
            Dense(1, activation=None)
        )(combined_processed)
        
        # Reshape to remove extra dimension
        logits = tf.keras.layers.Reshape((self.action_dim,))(logits)
        
        # Add softmax activation
        probs = tf.keras.layers.Activation('softmax')(logits)
        
        # Create model
        model = Model(inputs=[state_input, player_features_input], outputs=probs)
        model.compile(optimizer=Adam(learning_rate=learning_rate))
        
        return model
    
    def _build_critic_network(self, learning_rate):
        """
        Build critic network for value function
        
        Parameters:
        -----------
        learning_rate : float
            Learning rate for optimizer
            
        Returns:
        --------
        Model
            Tensorflow model for critic
        """
        state_input = Input(shape=(self.state_dim,))
        
        x = Dense(128, activation='relu')(state_input)
        x = Dropout(0.2)(x)
        x = Dense(64, activation='relu')(x)
        
        value = Dense(1, activation=None)(x)
        
        model = Model(inputs=state_input, outputs=value)
        model.compile(optimizer=Adam(learning_rate=learning_rate), 
                     loss='mse')
        
        return model
    
    def select_action(self, state, training=True):
        """
        Select an action (player) using hierarchical policy
        
        Parameters:
        -----------
        state : DraftState
            Current draft state
        training : bool
            Whether we're in training mode
                
        Returns:
        --------
        tuple
            Selected action info including position, player index, probabilities, value, and features
        """
        # Get valid players from the state
        valid_players = state.valid_players
        
        if not valid_players:
            return None, None, 0, 0, None
        
        # Get state features
        state_features = state.to_feature_vector()
        state_features = np.expand_dims(state_features, axis=0)
        
        # Step 1: Use meta-policy to select a position
        position_probs = self.meta_policy.predict(state_features, verbose=0)[0]
        
        # Filter to only consider positions with available players
        valid_positions = []
        for position in self.positions:
            position_players = [p for p in valid_players if p.position == position]
            if position_players:
                valid_positions.append(position)
        
        if not valid_positions:
            return None, None, 0, 0, None
        
        # Filter and normalize probabilities to valid positions
        valid_position_indices = [self.positions.index(pos) for pos in valid_positions]
        valid_position_probs = position_probs[valid_position_indices]
        valid_position_probs = valid_position_probs / np.sum(valid_position_probs)
        
        # Select position
        if training and random.random() < 0.2:  # Exploration during training
            position_idx = random.randint(0, len(valid_positions) - 1)
            selected_position = valid_positions[position_idx]
        else:
            position_idx = np.random.choice(len(valid_positions), p=valid_position_probs)
            selected_position = valid_positions[position_idx]
        
        # Get the position's probability
        selected_position_prob = valid_position_probs[position_idx]
        
        # Step 2: Filter players to the selected position
        position_players = [p for p in valid_players if p.position == selected_position]
        
        if not position_players:
            return None, None, 0, 0, None
        
        # Create action features array for this position
        action_features = np.zeros((1, self.action_dim, self.action_feature_dim))
        
        # Fill only valid players of the selected position
        for i, player in enumerate(position_players[:self.action_dim]):
            player_idx = valid_players.index(player)
            action_features[0, i] = state.get_action_features(player_idx)
        
        # Use the sub-policy for this position to select a player
        player_probs = self.sub_policies[selected_position].predict(
            [state_features, action_features], 
            verbose=0
        )[0]
        
        # Mask probabilities to only valid players of this position
        masked_probs = np.zeros_like(player_probs)
        masked_probs[:len(position_players)] = player_probs[:len(position_players)]
        
        # Normalize
        sum_probs = np.sum(masked_probs)
        if sum_probs > 0:
            masked_probs = masked_probs / sum_probs
        else:
            masked_probs[:len(position_players)] = 1.0 / len(position_players)
        
        # Get state value from critic
        value = self.critic.predict(state_features, verbose=0)[0][0]
        
        # Select player index
        if training and random.random() < 0.2:  # Exploration
            player_idx = random.randint(0, len(position_players) - 1)
        else:
            player_indices = np.arange(len(position_players))
            player_probs = masked_probs[:len(position_players)]
            player_idx = np.random.choice(player_indices, p=player_probs)
        
        # Convert to index in original valid_players list
        action_idx = valid_players.index(position_players[player_idx])
        action_prob = masked_probs[player_idx]
        
        return selected_position, action_idx, selected_position_prob, action_prob, value, action_features[0]
    
    def store_transition(self, state, position, action, pos_prob, action_prob, value, reward, done, action_features):
        """Store transition in memory"""
        self.memory.store(state, position, action, pos_prob, action_prob, value, reward, done, action_features)
    
    def update_policy(self):
        """Update policy using PPO with hierarchical approach"""
        # Return early if not enough samples
        if len(self.memory.states) == 0:
            return 0, 0, 0
        
        # Get values from memory
        states = np.array(self.memory.states)
        positions = self.memory.positions
        actions = np.array(self.memory.actions)
        old_pos_probs = np.array(self.memory.pos_probs)
        old_action_probs = np.array(self.memory.action_probs)
        values = np.array(self.memory.values)
        rewards = np.array(self.memory.rewards)
        dones = np.array(self.memory.dones)
        action_features_list = self.memory.action_features
        
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
        
        # Group samples by position for sub-policy updates
        position_indices = {pos: [] for pos in self.positions}
        for i, pos in enumerate(positions):
            if pos in position_indices:
                position_indices[pos].append(i)
        
        # Update metrics
        meta_actor_losses = []
        sub_actor_losses = {pos: [] for pos in self.positions}
        critic_losses = []
        
        # Update policy for n_epochs
        for _ in range(self.n_epochs):
            # Generate batches
            batches = self.memory.generate_batches()
            
            for batch in batches:
                # Extract batch data
                batch_states = states[batch]
                batch_actions = actions[batch]
                batch_positions = [positions[i] for i in batch]
                batch_old_pos_probs = old_pos_probs[batch]
                batch_old_action_probs = old_action_probs[batch]
                batch_advantages = advantages[batch]
                batch_returns = returns[batch]
                
                # Extract action features for this batch
                batch_action_features = [action_features_list[i] for i in batch]
                
                # Update critic
                with tf.GradientTape() as tape:
                    # Critic forward pass
                    batch_states_tf = tf.convert_to_tensor(batch_states, dtype=tf.float32)
                    values = self.critic(batch_states_tf, training=True)
                    values = tf.squeeze(values)
                    
                    # Critic loss
                    batch_returns_tf = tf.convert_to_tensor(batch_returns, dtype=tf.float32)
                    critic_loss = tf.reduce_mean(tf.square(batch_returns_tf - values))
                
                # Calculate critic gradients
                critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
                
                # Apply gradients
                self.critic.optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
                
                # Update meta-policy
                with tf.GradientTape() as tape:
                    # Meta-policy forward pass
                    position_probs = self.meta_policy(batch_states_tf, training=True)
                    
                    # Calculate position indices
                    position_indices = [self.positions.index(pos) for pos in batch_positions]
                    position_indices_tf = tf.convert_to_tensor(position_indices, dtype=tf.int64)
                    
                    # Extract probabilities for selected positions
                    actions_one_hot = tf.one_hot(position_indices_tf, len(self.positions))
                    position_probs_selected = tf.reduce_sum(position_probs * actions_one_hot, axis=1)
                    
                    # Calculate ratio
                    batch_old_pos_probs_tf = tf.convert_to_tensor(batch_old_pos_probs, dtype=tf.float32)
                    pos_ratio = position_probs_selected / (batch_old_pos_probs_tf + 1e-10)
                    
                    # Calculate surrogate losses
                    batch_advantages_tf = tf.convert_to_tensor(batch_advantages, dtype=tf.float32)
                    surr1 = pos_ratio * batch_advantages_tf
                    surr2 = tf.clip_by_value(pos_ratio, 1-self.policy_clip, 1+self.policy_clip) * batch_advantages_tf
                    
                    # Calculate entropy bonus
                    entropy = -tf.reduce_sum(position_probs * tf.math.log(position_probs + 1e-10), axis=1)
                    
                    # Meta-policy loss
                    meta_actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2) + self.entropy_coef * entropy)
                
                # Calculate meta-policy gradients
                meta_gradients = tape.gradient(meta_actor_loss, self.meta_policy.trainable_variables)
                
                # Apply gradients
                self.meta_policy.optimizer.apply_gradients(zip(meta_gradients, self.meta_policy.trainable_variables))
                
                # Update sub-policies for each position
                for position in self.positions:
                    # Get indices for this position
                    pos_batch_indices = [i for i, pos in enumerate(batch_positions) if pos == position]
                    
                    if len(pos_batch_indices) == 0:
                        continue  # Skip if no samples for this position
                    
                    # Extract data for this position
                    pos_states = tf.gather(batch_states_tf, pos_batch_indices)
                    pos_actions = tf.gather(batch_actions, pos_batch_indices)
                    
                    # We need to work directly with the TF tensor API since pos_actions may already be a tensor
                    if isinstance(pos_actions, tf.Tensor):
                        # Handle tensor case - just use the tensor's dtype
                        pos_actions_tf = pos_actions
                    else:
                        # Handle array case - convert to tensor and let TF choose the dtype
                        pos_actions_tf = tf.convert_to_tensor(pos_actions)
                    
                    pos_old_probs = tf.gather(batch_old_action_probs, pos_batch_indices)
                    pos_old_probs_tf = tf.convert_to_tensor(pos_old_probs, dtype=tf.float32)
                    pos_advantages = tf.gather(batch_advantages_tf, pos_batch_indices)
                    
                    # Extract action features for this position
                    pos_action_features = np.zeros((len(pos_batch_indices), self.action_dim, self.action_feature_dim))
                    for i, batch_idx in enumerate(pos_batch_indices):
                        idx = batch[batch_idx]  # Get the original index
                        if idx < len(action_features_list):
                            pos_action_features[i] = action_features_list[idx]
                    
                    # Convert to TensorFlow tensor
                    pos_action_features_tf = tf.convert_to_tensor(pos_action_features, dtype=tf.float32)
                    
                    with tf.GradientTape() as tape:
                        # Sub-policy forward pass
                        action_probs = self.sub_policies[position]([pos_states, pos_action_features_tf], training=True)
                        
                        # Extract probabilities for taken actions
                        # Use the tensor's existing dtype for one-hot encoding
                        actions_one_hot = tf.one_hot(pos_actions_tf, self.action_dim)
                        action_probs_selected = tf.reduce_sum(action_probs * actions_one_hot, axis=1)
                        
                        # Calculate ratio
                        action_ratio = action_probs_selected / (pos_old_probs_tf + 1e-10)
                        
                        # Calculate surrogate losses
                        surr1 = action_ratio * pos_advantages
                        surr2 = tf.clip_by_value(action_ratio, 1-self.policy_clip, 1+self.policy_clip) * pos_advantages
                        
                        # Calculate entropy bonus
                        entropy = -tf.reduce_sum(action_probs * tf.math.log(action_probs + 1e-10), axis=1)
                        
                        # Sub-policy loss
                        sub_actor_loss = -tf.reduce_mean(tf.minimum(surr1, surr2) + self.entropy_coef * entropy)
                    
                    # Calculate sub-policy gradients
                    sub_gradients = tape.gradient(sub_actor_loss, self.sub_policies[position].trainable_variables)
                    
                    # Apply gradients
                    self.sub_policies[position].optimizer.apply_gradients(
                        zip(sub_gradients, self.sub_policies[position].trainable_variables)
                    )
                    
                    # Track losses
                    sub_actor_losses[position].append(sub_actor_loss.numpy())
                
                # Track losses
                meta_actor_losses.append(meta_actor_loss.numpy())
                critic_losses.append(critic_loss.numpy())
        
        # Clear memory
        self.memory.clear()
        
        # Calculate average losses
        avg_meta_loss = np.mean(meta_actor_losses) if meta_actor_losses else 0
        avg_sub_losses = {pos: np.mean(losses) if losses else 0 for pos, losses in sub_actor_losses.items()}
        avg_critic_loss = np.mean(critic_losses) if critic_losses else 0
        
        # Calculate average sub-policy loss across all positions
        avg_sub_loss = np.mean([avg_sub_losses[pos] for pos in self.positions if avg_sub_losses[pos] != 0])
        
        return avg_meta_loss, avg_sub_loss, avg_critic_loss
    
    def train(self, draft_simulator, season_simulator, num_episodes=500, 
              eval_interval=10, save_interval=50, save_path=None):
        """
        Train the hierarchical RL model
        
        Parameters:
        -----------
        draft_simulator : DraftSimulator
            Draft simulator instance
        season_simulator : SeasonSimulator
            Season simulator instance
        num_episodes : int
            Number of episodes to train for
        eval_interval : int
            Number of episodes between evaluations
        save_interval : int
            Number of episodes between saving the model
        save_path : str, optional
            Path to save the model
            
        Returns:
        --------
        dict
            Training results
        """
        logger.info(f"Training hierarchical PPO drafter for {num_episodes} episodes")
        
        # Keep track of best model
        best_reward = -float('inf')
        best_weights = {
            'meta': None,
            'sub': {},
            'critic': None
        }
        
        episode_rewards = []
        
        # Training metrics tracking
        self.position_distributions = []
        self.value_metrics = []
        self.rank_history = []
        
        # Opponent modeling metrics
        if self.opponent_modeling_enabled:
            self.opponent_predictions = []
            self.value_cliff_decisions = []
            self.position_run_decisions = []
        
        # Store original parameters to recreate simulator each episode
        original_players = copy.deepcopy(draft_simulator.players)
        league_size = draft_simulator.league_size
        roster_limits = draft_simulator.roster_limits.copy()
        num_rounds = draft_simulator.num_rounds
        scoring_settings = draft_simulator.scoring_settings.copy()
        
        # Save initial model before training if path is provided
        if save_path:
            self.save_model(os.path.join(save_path, "hierarchical_ppo_initial"), episode=0, is_initial=True)
        
        # Training loop
        for episode in range(1, num_episodes + 1):
            self.episodes = episode
            rl_draft_position = random.randint(1, league_size)
            logger.info(f"Episode {episode}/{num_episodes}")
            logger.info(f"Episode {episode}: RL team drafting from position {rl_draft_position}")
            
            # Update curriculum temperature if enabled
            if self.curriculum_enabled:
                max_temp = 0.5
                self.curriculum_temperature = min(max_temp, episode / 1000)
            
            # Reinitialize the draft simulator for each episode
            fresh_players = copy.deepcopy(original_players)
            draft_simulator = DraftSimulator(
                players=fresh_players,
                league_size=league_size,
                roster_limits=roster_limits,
                num_rounds=num_rounds,
                scoring_settings=scoring_settings
            )
            
            # Set up RL team
            for i, team in enumerate(draft_simulator.teams):
                # Assign strategies in a cyclic manner, skipping PPO
                valid_strategies = ["VBD", "ESPN", "ZeroRB", "HeroRB", "TwoRB", "BestAvailable"]
                strategy_idx = i % len(valid_strategies)
                team.strategy = valid_strategies[strategy_idx]
            
            # Set ONE team to use PPO strategy based on draft position
            rl_team = None
            for team in draft_simulator.teams:
                if team.draft_position == rl_draft_position:
                    team.strategy = "PPO"
                    rl_team = team
                    break
            
            if not rl_team:
                rl_team = random.choice(draft_simulator.teams)
                rl_team.strategy = "PPO"
            
            # Reinitialize season simulator
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
            
            # Track opponent modeling decisions
            if self.opponent_modeling_enabled:
                episode_opponent_predictions = []
                episode_cliff_decisions = []
                episode_run_decisions = []
            
            # Current state
            current_round = 1
            current_pick = 1
            
            # For opponent modeling, predict the next few picks
            predicted_picks = {}
            
            # Get starter limits if available
            starter_limits = {}
            if (hasattr(draft_simulator, 'scoring_settings') and 
                draft_simulator.scoring_settings and 
                'starter_limits' in draft_simulator.scoring_settings):
                starter_limits = draft_simulator.scoring_settings['starter_limits']
            
            # Run until all rounds are complete
            while current_round <= draft_simulator.num_rounds:
                # Get the team picking
                team_idx = (current_pick - 1) % draft_simulator.league_size
                if current_round % 2 == 0:  # Even round, reverse order
                    team_idx = draft_simulator.league_size - 1 - team_idx
                
                team = draft_simulator.teams[team_idx]
                
                # If this is the RL team, use our model
                if team.strategy == "PPO":
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
                        max_rounds=draft_simulator.num_rounds,
                        projection_models=draft_simulator.projection_models,
                        use_top_n_features=self.use_top_n_features,
                        all_teams=draft_simulator.teams,
                        starter_limits=starter_limits
                    )
                    
                    # Track opponent modeling metrics if enabled
                    if self.opponent_modeling_enabled:
                        # Check if we made a previous prediction for this pick
                        if current_pick in predicted_picks:
                            predicted_position = predicted_picks[current_pick]
                            
                            # Track prediction to evaluate later
                            prediction_data = {
                                "pick": current_pick,
                                "predicted_position": predicted_position,
                                "actual_position": None  # Fill in after selection
                            }
                            episode_opponent_predictions.append(prediction_data)
                        
                        # Check for value cliffs
                        if hasattr(state, 'value_cliffs'):
                            for position in ["QB", "RB", "WR", "TE"]:
                                if position in state.value_cliffs:
                                    cliff_info = state.value_cliffs[position]
                                    
                                    if cliff_info.get('has_cliff', False) and cliff_info.get('first_cliff_position', 10) == 0:
                                        # Found a position with an immediate cliff
                                        logger.info(f"Value cliff detected for {position}, considering in pick decision")
                                        
                                        # Record the cliff decision
                                        episode_cliff_decisions.append({
                                            "pick": current_pick,
                                            "position": position,
                                            "cliff_magnitude": cliff_info.get('first_cliff_magnitude', 0)
                                        })
                        
                        # Check for position runs
                        if hasattr(state, 'position_runs'):
                            for position, run_info in state.position_runs.items():
                                if run_info.get('is_run', False):
                                    # Found a position with an active run
                                    logger.info(f"Position run detected for {position}, considering in pick decision")
                                    
                                    # Record the run decision
                                    episode_run_decisions.append({
                                        "pick": current_pick,
                                        "position": position,
                                        "run_percentage": run_info.get('run_percentage', 0)
                                    })
                    
                    # Select action using hierarchical policy
                    position, action, pos_prob, action_prob, value, precomputed_features = self.select_action(state, training=True)
                    
                    # Execute action
                    if position is not None and action is not None and action < len(state.valid_players):
                        player = state.valid_players[action]
                        team.add_player(player, current_round, current_pick)
                        draft_history.append((state, position, action, pos_prob, action_prob, value, player, precomputed_features))
                        logger.info(f"RL team drafted: {player.name} ({player.position}) - Round {current_round}, Pick {current_pick}")
                        
                        # Update any prediction data with actual position
                        for pred in episode_opponent_predictions:
                            if pred["pick"] == current_pick:
                                pred["actual_position"] = player.position
                                pred["correct"] = (pred["predicted_position"] == player.position)
                    else:
                        logger.warning(f"Invalid action: {action}, valid players: {len(state.valid_players)}")
                        # Fallback to best available
                        available_players = [p for p in draft_simulator.players if not p.is_drafted]
                        valid_positions = [pos for pos in draft_simulator.roster_limits.keys() if team.can_draft_position(pos)]
                        valid_players = [p for p in available_players if p.position in valid_positions]
                        
                        if valid_players:
                            valid_players.sort(key=lambda p: p.projected_points, reverse=True)
                            player = valid_players[0]
                            team.add_player(player, current_round, current_pick)
                            logger.info(f"RL team fallback draft: {player.name} ({player.position})")
                
                # Otherwise, use the team's strategy
                else:
                    # If opponent modeling is enabled, try to predict opponent picks
                    if self.opponent_modeling_enabled and rl_team is not None:
                        # Only predict if our team has future picks
                        rl_team_picked = any(t.strategy == "PPO" for t in draft_simulator.teams)
                        
                        if rl_team_picked:
                            # Create state to analyze opponent's needs
                            available_players = [p for p in draft_simulator.players if not p.is_drafted]
                            
                            # Create the full state with opponent modeling
                            state = DraftState(
                                team=team,
                                available_players=available_players,
                                round_num=current_round,
                                overall_pick=current_pick,
                                league_size=draft_simulator.league_size,
                                roster_limits=draft_simulator.roster_limits,
                                max_rounds=draft_simulator.num_rounds,
                                projection_models=draft_simulator.projection_models,
                                use_top_n_features=self.use_top_n_features,
                                all_teams=draft_simulator.teams,
                                starter_limits=starter_limits
                            )
                            
                            # Use opponent needs model to predict their most likely position
                            opponent_needs = state.opponent_needs.get(team.name, {})
                            
                            # Determine most likely position target
                            max_urgency = 0
                            predicted_position = None
                            
                            for position, need_info in opponent_needs.items():
                                if position in ["QB", "RB", "WR", "TE", "K", "DST"]:
                                    urgency = need_info.get('urgency', 0) * need_info.get('remaining', 0)
                                    
                                    if urgency > max_urgency:
                                        max_urgency = urgency
                                        predicted_position = position
                            
                            # Record the prediction for this pick
                            if predicted_position:
                                predicted_picks[current_pick] = predicted_position
                    
                    # Make the pick using the team's strategy
                    picked_player = draft_simulator._make_pick(team, current_round, current_pick)
                    
                    # Validate opponent pick prediction if we made one
                    if self.opponent_modeling_enabled and current_pick in predicted_picks and picked_player:
                        predicted_position = predicted_picks[current_pick]
                        actual_position = picked_player.position
                        
                        # Record prediction accuracy
                        prediction_correct = (predicted_position == actual_position)
                        
                        # Track prediction to evaluate later
                        prediction_data = {
                            "pick": current_pick,
                            "predicted_position": predicted_position,
                            "actual_position": actual_position,
                            "correct": prediction_correct
                        }
                        episode_opponent_predictions.append(prediction_data)
                
                # Move to next pick
                current_pick += 1
                if current_pick > current_round * draft_simulator.league_size:
                    current_round += 1
                    
                if current_pick > draft_simulator.num_rounds * draft_simulator.league_size:
                    logger.info(f"Reached maximum picks ({current_pick-1}), ending draft")
                    break
            
            # After draft is complete, track opponent modeling metrics
            if self.opponent_modeling_enabled:
                if episode_opponent_predictions:
                    correct_predictions = sum(1 for pred in episode_opponent_predictions if pred.get("correct", False))
                    total_predictions = len(episode_opponent_predictions)
                    prediction_accuracy = correct_predictions / max(1, total_predictions)
                    
                    # Log prediction accuracy
                    logger.info(f"Opponent pick prediction accuracy: {prediction_accuracy:.2f} ({correct_predictions}/{total_predictions})")
                    
                    # Store for global metrics
                    self.opponent_predictions.append({
                        "episode": episode,
                        "accuracy": prediction_accuracy,
                        "predictions": episode_opponent_predictions
                    })
                
                # Track cliff decisions
                if episode_cliff_decisions:
                    self.value_cliff_decisions.append({
                        "episode": episode,
                        "decisions": episode_cliff_decisions
                    })
                
                # Track run decisions
                if episode_run_decisions:
                    self.position_run_decisions.append({
                        "episode": episode,
                        "decisions": episode_run_decisions
                    })
            
            # Simulate and evaluate the season
            season_results = season_simulator.simulate_season()
            evaluation = SeasonEvaluator(draft_simulator.teams, season_results)
            
            # Calculate reward for RL team
            rl_metrics = None
            
            # First check if PPO strategy metrics exist
            if "PPO" in evaluation.metrics:
                for metrics in evaluation.metrics["PPO"]["teams"]:
                    if metrics["team_name"] == rl_team.name:
                        rl_metrics = metrics
                        break
            
            # If not found, try finding by team name directly
            if not rl_metrics:
                for strategy, strategy_metrics in evaluation.metrics.items():
                    for metrics in strategy_metrics.get("teams", []):
                        if metrics["team_name"] == rl_team.name:
                            rl_metrics = metrics
                            logger.info(f"Found RL team under strategy: {strategy}")
                            break
                    if rl_metrics:
                        break
            
            # Initialize position counts
            position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "K": 0, "DST": 0}
            vbd_sum = 0.0
            starter_points = 0.0
            total_points = 0.0
            roster_efficiency = 0.0
            
            # Count position distribution
            for player in rl_team.roster:
                if player.position in position_counts:
                    position_counts[player.position] += 1
                
                # Track VBD sum
                vbd_sum += max(0, getattr(player, 'vbd', 0))
            
            # Calculate starter and total points
            starter_points = rl_team.get_starting_lineup_points()
            total_points = rl_team.get_total_projected_points()
            roster_efficiency = starter_points / max(1, total_points)
            
            # Update adaptive position weights if enabled
            if self.opponent_modeling_enabled:
                self._update_position_weights(rl_metrics, position_counts, starter_points)
            
            # Calculate reward based on curriculum if enabled
            if rl_metrics:
                if self.curriculum_enabled:
                    reward, reward_components = self.calculate_curriculum_reward(rl_metrics, rl_team)
                    
                    # Log the reward components
                    logger.info(f"  Reward using curriculum phase {self.curriculum_phase}: {reward:.2f}")
                    logger.info(f"  Reward components:")
                    for component, value in reward_components.items():
                        if isinstance(value, (int, float)):
                            logger.info(f"    {component}: {value:.2f}")
                    # Update curriculum phase if criteria are met
                    phase_changed = self.update_curriculum_phase()
                    if phase_changed:
                        logger.info(f"  Advanced to curriculum phase {self.curriculum_phase}")
                else:
                    # Standard reward calculation
                    reward = (
                        -3.0 * rl_metrics["rank"] +
                        2.0 * rl_metrics["wins"] +
                        0.01 * rl_metrics["points_for"] +
                        15.0 * (1 if rl_metrics.get("playoff_result") == "Champion" else 0) +
                        7.0 * (1 if rl_metrics.get("playoff_result") in ["Runner-up", "Third Place"] else 0) +
                        3.0 * (1 if rl_metrics.get("playoff_result") == "Playoff Qualification" else 0)
                    )
                    
                    # Add draft quality components
                    reward += (vbd_sum * 0.1) + (roster_efficiency * 5.0)
                    
                    # Position requirements check
                    position_penalty = 0
                    for pos, requirement in self.position_requirements.items():
                        if position_counts.get(pos, 0) < requirement:
                            position_penalty += 10.0 * (requirement - position_counts.get(pos, 0))
                    
                    reward -= position_penalty
                    reward += 10.0  # Baseline adjustment
                
                # Track episode reward
                episode_reward = reward
                episode_rewards.append(reward)
                self.rewards_history.append(reward)
                
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
                    
                    # Save meta-policy weights
                    best_weights['meta'] = [w.copy() for w in self.meta_policy.get_weights()]
                    
                    # Save sub-policy weights
                    for position in self.positions:
                        best_weights['sub'][position] = [w.copy() for w in self.sub_policies[position].get_weights()]
                    
                    # Save critic weights
                    best_weights['critic'] = [w.copy() for w in self.critic.get_weights()]
                    
                    # Save best model
                    if save_path:
                        self.save_model(os.path.join(save_path, "hierarchical_ppo_best"))
                        logger.info(f"New best model with reward: {reward:.2f}")
                
                # Process draft history for learning
                if draft_history:
                    # Distribute rewards across draft picks with diminishing returns
                    num_picks = len(draft_history)
                    
                    for i, (state, position, action, pos_prob, action_prob, value, player, precomputed_features) in enumerate(draft_history):
                        # Calculate pick-specific reward with diminishing returns
                        pick_weight = 1.0 - 0.5 * (i / num_picks)
                        action_reward = reward * pick_weight
                        
                        # Add auxiliary reward for early episodes
                        if episode <= 10:
                            vbd_value = getattr(player, 'vbd', 0)
                            if vbd_value > 0:
                                position_factor = {'QB': 0.8, 'RB': 1.2, 'WR': 1.0, 'TE': 0.9}.get(player.position, 1.0)
                                aux_reward = vbd_value * position_factor * 0.5
                                action_reward += aux_reward
                        
                        # Determine if this is the last action
                        done = (i == num_picks - 1)
                        
                        # Store transition
                        self.memory.store(
                            state.to_feature_vector(),
                            position,
                            action,
                            pos_prob,
                            action_prob,
                            value,
                            action_reward,
                            done,
                            precomputed_features
                        )
                    
                    # Perform policy updates if we have enough samples
                    if len(self.memory.states) >= self.batch_size:
                        # Perform multiple updates for more efficient learning
                        total_meta_loss = 0
                        total_sub_loss = 0
                        total_critic_loss = 0
                        update_iterations = 2
                        
                        for update_iter in range(update_iterations):
                            meta_loss, sub_loss, critic_loss = self.update_policy()
                            total_meta_loss += meta_loss
                            total_sub_loss += sub_loss
                            total_critic_loss += critic_loss
                        
                        # Log average losses
                        avg_meta_loss = total_meta_loss / update_iterations
                        avg_sub_loss = total_sub_loss / update_iterations
                        avg_critic_loss = total_critic_loss / update_iterations
                        logger.info(f"  Updated model {update_iterations} times - Meta loss: {avg_meta_loss:.4f}, Sub loss: {avg_sub_loss:.4f}, Critic loss: {avg_critic_loss:.4f}")
                    else:
                        logger.info(f"  Not enough samples for update ({len(self.memory.states)}/{self.batch_size})")
                else:
                    logger.warning("No draft history for RL team!")
            else:
                logger.warning("Could not find RL team metrics for reward calculation!")
            
            # Save model at defined intervals
            if save_path and episode % save_interval == 0:
                self.save_model(os.path.join(save_path, f"hierarchical_ppo_episode_{episode}"), episode=episode)
                logger.info(f"Saved model at episode {episode}")
            
            # Evaluate performance periodically
            if episode % eval_interval == 0:
                # Get win rates for each strategy
                win_rates = {}
                for strategy, metrics in evaluation.metrics.items():
                    wins = metrics.get("avg_wins", 0)
                    losses = metrics.get("avg_losses", 0)
                    total_games = wins + losses
                    win_rate = wins / max(total_games, 1)
                    win_rates[strategy] = win_rate
                
                self.win_rates.append(win_rates)
                
                logger.info(f"Win rates after episode {episode}:")
                for strategy, win_rate in win_rates.items():
                    logger.info(f"  {strategy}: Win Rate = {win_rate:.3f}")
                
                # Update training plots
                self.update_training_plots(save_path, episode)
                
                # Create curriculum visualization if enabled
                if self.curriculum_enabled and save_path:
                    curriculum_viz_path = os.path.join(save_path, f'curriculum_progress_ep{episode}.png')
                    self.plot_curriculum_progress(save_path=curriculum_viz_path)
                
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
        if best_weights['meta'] is not None:
            self.meta_policy.set_weights(best_weights['meta'])
            
            for position in self.positions:
                if position in best_weights['sub'] and best_weights['sub'][position] is not None:
                    self.sub_policies[position].set_weights(best_weights['sub'][position])
            
            if best_weights['critic'] is not None:
                self.critic.set_weights(best_weights['critic'])
                
            logger.info("Restored best model weights from training")
        
        # Save final model
        if save_path:
            self.save_model(os.path.join(save_path, "hierarchical_ppo_final"), is_final=True)
            logger.info("Saved final model")
        
        # Return comprehensive training results
        return {
            "rewards_history": self.rewards_history,
            "win_rates": self.win_rates,
            "best_reward": best_reward,
            "episodes": self.episodes,
            "episode_rewards": episode_rewards,
            "position_distributions": self.position_distributions,
            "value_metrics": self.value_metrics,
            "rank_history": self.rank_history,
            
            # Curriculum learning results
            "curriculum_history": self.curriculum_rewards_history if self.curriculum_enabled else None,
            "final_curriculum_phase": self.curriculum_phase if self.curriculum_enabled else None,
            
            # Opponent modeling metrics
            "opponent_predictions": self.opponent_predictions if self.opponent_modeling_enabled else None,
            "value_cliff_decisions": self.value_cliff_decisions if self.opponent_modeling_enabled else None,
            "position_run_decisions": self.position_run_decisions if self.opponent_modeling_enabled else None,
            "position_weights": self.position_priority_weights if self.opponent_modeling_enabled else None
        }
    
    def save_model(self, path, episode=None, is_initial=False, is_final=False):
        """
        Save the hierarchical PPO model
        
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
        # Save meta-policy weights
        self.meta_policy.save_weights(f"{path}_meta_policy.weights.h5")
        
        # Save sub-policy weights for each position
        for position in self.positions:
            self.sub_policies[position].save_weights(f"{path}_{position}_policy.weights.h5")
        
        # Save critic weights
        self.critic.save_weights(f"{path}_critic.weights.h5")
        
        # Save training history and metadata
        history = {
            "rewards_history": self.rewards_history,
            "win_rates": self.win_rates,
            "episodes": self.episodes,
            "state_dim": self.state_dim,
            "action_dim": self.action_dim,
            "action_feature_dim": self.action_feature_dim,
            "positions": self.positions
        }
        
        # Add enhanced training metrics if available
        if hasattr(self, 'position_distributions'):
            history["position_distributions"] = self.position_distributions
        
        if hasattr(self, 'value_metrics'):
            history["value_metrics"] = self.value_metrics
        
        if hasattr(self, 'rank_history'):
            history["rank_history"] = self.rank_history
        
        # Serialize to JSON
        import json
        with open(f"{path}_history.json", "w") as f:
            # Convert numpy arrays to lists
            for key, value in history.items():
                if isinstance(value, np.ndarray):
                    history[key] = value.tolist()
            
            json.dump(history, f, indent=2)
        
        logger.info(f"Model weights saved to {path}")
    
    @classmethod
    def load_model(cls, path):
        """
        Load a saved hierarchical PPO model
        
        Parameters:
        -----------
        path : str
            Path to the saved model
            
        Returns:
        --------
        HierarchicalPPODrafter
            Loaded model
        """
        # Load history and metadata
        try:
            import json
            with open(f"{path}_history.json", "r") as f:
                history = json.load(f)
            
            state_dim = history.get("state_dim", 100)
            action_dim = history.get("action_dim", 256)
            action_feature_dim = history.get("action_feature_dim", 50)
            positions = history.get("positions", ["QB", "RB", "WR", "TE", "K", "DST"])
        except FileNotFoundError:
            logger.warning(f"History file not found, using default dimensions")
            state_dim = 100
            action_dim = 256
            action_feature_dim = 50
            positions = ["QB", "RB", "WR", "TE", "K", "DST"]
        
        # Create instance
        instance = cls(
            state_dim=state_dim, 
            action_dim=action_dim, 
            action_feature_dim=action_feature_dim
        )
        
        # Set positions
        instance.positions = positions
        
        # Initialize networks with dummy data
        dummy_state = np.zeros((1, state_dim))
        dummy_action_features = np.zeros((1, action_dim, action_feature_dim))
        
        # Forward pass to build networks
        instance.meta_policy(dummy_state)
        for position in instance.positions:
            instance.sub_policies[position]([dummy_state, dummy_action_features])
        instance.critic(dummy_state)
        
        # Load weights if available
        try:
            instance.meta_policy.load_weights(f"{path}_meta_policy.weights.h5")
            logger.info(f"Meta-policy weights loaded")
        except:
            logger.error(f"Failed to load meta-policy weights")
        
        # Load sub-policy weights
        for position in instance.positions:
            try:
                instance.sub_policies[position].load_weights(f"{path}_{position}_policy.weights.h5")
                logger.info(f"{position} sub-policy weights loaded")
            except:
                logger.error(f"Failed to load {position} sub-policy weights")
        
        # Load critic weights
        try:
            instance.critic.load_weights(f"{path}_critic.weights.h5")
            logger.info(f"Critic weights loaded")
        except:
            logger.error(f"Failed to load critic weights")
        
        # Load history if available
        if history:
            instance.rewards_history = history.get("rewards_history", [])
            instance.win_rates = history.get("win_rates", [])
            instance.episodes = history.get("episodes", 0)
        
        return instance
    
    def calculate_curriculum_reward(self, rl_metrics, rl_team):
        """
        Calculate reward based on current curriculum phase with adaptive scaling
        
        Parameters:
        -----------
        rl_metrics : dict
            Performance metrics for the RL team
        rl_team : Team
            The RL team object with roster information
        
        Returns:
        --------
        float
            The calculated reward
        dict
            Detailed reward components
        """
        # Track positions drafted
        position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "K": 0, "DST": 0}
        for player in rl_team.roster:
            if player.position in position_counts:
                position_counts[player.position] += 1
        
        # Calculate roster completeness score
        required_positions_filled = 0
        for pos, requirement in self.position_requirements.items():
            if position_counts.get(pos, 0) >= requirement:
                required_positions_filled += 1
        
        roster_completeness = required_positions_filled / len(self.position_requirements)
        
        # Calculate Value-Based Drafting quality
        vbd_sum = 0.0
        for player in rl_team.roster:
            vbd = getattr(player, 'vbd', 0)
            vbd_sum += max(0, vbd)  # Only count positive VBD
        
        # Calculate roster efficiency
        starter_points = rl_team.get_starting_lineup_points()
        total_points = rl_team.get_total_projected_points()
        roster_efficiency = starter_points / max(1, total_points)
        
        # Phase 1 reward: Valid roster construction
        phase1_reward = 0
        for pos, requirement in self.position_requirements.items():
            if position_counts.get(pos, 0) >= requirement:
                phase1_reward += 5.0  # Bonus for each required position filled
            else:
                # Penalty for each missing required position
                phase1_reward -= 10.0 * (requirement - position_counts.get(pos, 0))
        
        # Add completion bonus if all required positions are filled
        if roster_completeness >= 1.0:
            phase1_reward += 20.0
        
        # Phase 2 reward: Total value maximization
        phase2_reward = vbd_sum * 0.1 + total_points * 0.05
        
        # Phase 3 reward: Starter optimization
        phase3_reward = starter_points * 0.1 + roster_efficiency * 10.0
        
        # Phase 4 reward: Season performance
        if rl_metrics:
            phase4_reward = (
                -3.0 * rl_metrics["rank"] +  # Lower rank is better
                2.0 * rl_metrics["wins"] +  # More wins is better
                0.01 * rl_metrics["points_for"] +  # More points is better
                15.0 * (1 if rl_metrics.get("playoff_result") == "Champion" else 0) +
                7.0 * (1 if rl_metrics.get("playoff_result") in ["Runner-up", "Third Place"] else 0) +
                3.0 * (1 if rl_metrics.get("playoff_result") == "Playoff Qualification" else 0)
            )
        else:
            phase4_reward = 0
        
        # Apply reward mixing to prevent catastrophic forgetting
        phase_rewards = [phase1_reward, phase2_reward, phase3_reward, phase4_reward]
        
        # Normalize rewards using running statistics before mixing
        normalized_rewards = []
        for phase, reward in enumerate(phase_rewards, 1):
            if len(self.curriculum_rewards_history[phase]) > 10:
                # Use running mean and std for normalization
                mean = self.reward_stats[phase]["mean"]
                std = max(1e-5, self.reward_stats[phase]["std"])  # Avoid division by zero
                normalized_rewards.append((reward - mean) / std)
            else:
                # Not enough data for normalization yet
                normalized_rewards.append(reward)
        
        # Apply reward mixing based on current phase
        mixed_reward = 0
        for i, weight in enumerate(self.reward_mix_weights[self.curriculum_phase]):
            mixed_reward += normalized_rewards[i] * weight
        
        # Store metrics for phase transition evaluation
        valid_roster_rate = roster_completeness
        
        # Add baseline adjustment 
        mixed_reward += 10.0
        
        # Store phase-specific reward for analysis
        raw_reward = phase_rewards[self.curriculum_phase - 1]  # Original non-mixed reward
        self.curriculum_rewards_history[self.curriculum_phase].append(raw_reward)
        
        # Update running statistics for reward normalization
        if len(self.curriculum_rewards_history[self.curriculum_phase]) > 1:
            rewards = self.curriculum_rewards_history[self.curriculum_phase]
            self.reward_stats[self.curriculum_phase]["mean"] = np.mean(rewards[-100:])  # Use last 100 episodes
            self.reward_stats[self.curriculum_phase]["std"] = np.std(rewards[-100:]) + 1e-5  # Add small epsilon
        
        return mixed_reward, {
            "phase1_reward": phase1_reward,
            "phase2_reward": phase2_reward,
            "phase3_reward": phase3_reward,
            "phase4_reward": phase4_reward,
            "roster_completeness": roster_completeness,
            "vbd_sum": vbd_sum,
            "roster_efficiency": roster_efficiency,
            "starter_points": starter_points,
            "total_points": total_points,
            "mixed_reward": mixed_reward
        }
    
    def update_curriculum_phase(self):
        """
        Update the curriculum phase based on progress and performance
        
        Returns:
        --------
        bool
            True if phase changed, False otherwise
        """
        if not self.curriculum_enabled:
            return False
        
        self.phase_episode_count += 1
        
        # Update curriculum temperature
        max_temp = 0.5  # Maximum temperature
        self.curriculum_temperature = min(max_temp, self.episodes / 1000)
        
        # Adjust thresholds based on temperature
        adjusted_thresholds = {
            phase: threshold * (1.0 - self.curriculum_temperature) 
            for phase, threshold in self.phase_thresholds.items()
        }
        
        # Check if we should transition based on episode count
        if self.phase_episode_count >= self.phase_durations[self.curriculum_phase]:
            old_phase = self.curriculum_phase
            self.curriculum_phase = min(4, self.curriculum_phase + 1)
            self.phase_episode_count = 0
            logger.info(f"Curriculum advanced to phase {self.curriculum_phase} based on episode count")
            self.phase_transition_episodes.append(self.episodes)
            return old_phase != self.curriculum_phase
        
        return False
    
    def _update_position_weights(self, metrics, position_counts, starter_points):
        """
        Update position priority weights based on performance
        
        Parameters:
        -----------
        metrics : dict
            Team performance metrics
        position_counts : dict
            Counts of drafted positions
        starter_points : float
            Projected starter points
        """
        # Skip if performance metrics are missing
        if not metrics:
            return
        
        # Get key performance indicators
        rank = metrics.get("rank", 0)
        wins = metrics.get("wins", 0)
        made_playoffs = metrics.get("playoff_result", "Missed Playoffs") != "Missed Playoffs"
        
        # Calculate performance score (higher is better)
        performance_score = (
            10 - rank +  # Lower rank is better
            wins * 0.5 +
            (5 if made_playoffs else 0) +
            starter_points / 20  # Scale points to similar magnitude
        )
        
        # Only update weights if we have a non-zero draft
        if sum(position_counts.values()) > 0:
            # Calculate relative position weight based on draft composition
            total_players = sum(position_counts.values())
            
            for position, count in position_counts.items():
                # Skip positions we didn't draft
                if count == 0:
                    continue
                
                # Calculate relative usage of this position
                position_ratio = count / total_players
                
                # Adjust weights based on performance
                if performance_score > 10:  # Good performance
                    # If good performance, slightly increase weight for positions we drafted more
                    weight_adjustment = position_ratio * 0.1  # Small positive adjustment
                else:  # Poor performance
                    # If poor performance, slightly decrease weight for positions we drafted more
                    weight_adjustment = -position_ratio * 0.1  # Small negative adjustment
                
                # Update the EMA for this position
                current_ema = self.position_ema.get(position, 1.0)
                new_ema = current_ema + self.ema_alpha * (1.0 + weight_adjustment - current_ema)
                self.position_ema[position] = new_ema
                
                # Update the actual weight, ensuring it stays within reasonable bounds
                self.position_priority_weights[position] = min(2.0, max(0.5, new_ema))
        
        # Log the updated weights
        logger.info("Updated position priority weights:")
        for position, weight in self.position_priority_weights.items():
            logger.info(f"  {position}: {weight:.2f}")
    
    def update_training_plots(self, output_dir, episode):
        """Update training plots during training"""
        if not output_dir:
            return
            
        import matplotlib.pyplot as plt
        
        # Reward history plot
        plt.figure(figsize=(12, 6))
        plt.plot(self.rewards_history)
        plt.title(f'Hierarchical PPO Training Progress (Episode {episode})')
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.grid(True)
        plt.savefig(os.path.join(output_dir, 'training_progress_current.png'))
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
            plt.savefig(os.path.join(output_dir, 'win_rates_current.png'))
            plt.close()
    
    def plot_curriculum_progress(self, save_path=None):
        """Generate visualization of curriculum learning progress"""
        if not self.curriculum_enabled or not hasattr(self, 'curriculum_rewards_history'):
            return
            
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(16, 12))
        
        # Plot rewards for each phase
        plt.subplot(2, 3, 1)
        for phase in range(1, 5):
            rewards = self.curriculum_rewards_history.get(phase, [])
            if rewards:
                plt.plot(rewards, label=f"Phase {phase}")
        
        plt.title('Curriculum Learning Rewards by Phase')
        plt.xlabel('Training Steps in Phase')
        plt.ylabel('Reward')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Phase duration
        plt.subplot(2, 3, 2)
        phases = list(range(1, 5))
        durations = [self.phase_durations.get(p, 0) for p in phases]
        plt.bar(phases, durations)
        plt.title('Phase Durations (Episodes)')
        plt.xlabel('Phase')
        plt.ylabel('Episodes')
        
        # Phase transitions
        if hasattr(self, 'phase_transition_episodes') and self.phase_transition_episodes:
            plt.subplot(2, 3, 3)
            transitions = self.phase_transition_episodes
            phases = list(range(2, len(transitions) + 2))  # Start from phase 2
            plt.plot(transitions, phases, 'ro-')
            plt.title('Phase Transitions')
            plt.xlabel('Episode')
            plt.ylabel('Phase')
            plt.grid(True, alpha=0.3)
        
        # Reward mix weights
        plt.subplot(2, 3, 6)
        phases = list(range(1, 5))
        components = ['Phase 1', 'Phase 2', 'Phase 3', 'Phase 4']
        
        bottom = np.zeros(4)
        for i, component in enumerate(components):
            values = [self.reward_mix_weights[phase][i] for phase in phases]
            plt.bar(phases, values, bottom=bottom, label=component)
            bottom += values
        
        plt.title('Reward Mix Weights')
        plt.xlabel('Current Phase')
        plt.ylabel('Weight')
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        plt.close()


class HierarchicalPPOMemory:
    """Memory buffer for hierarchical PPO training"""
    
    def __init__(self, batch_size=32):
        self.states = []
        self.positions = []  # Store selected positions
        self.actions = []
        self.pos_probs = []  # Store position probabilities
        self.action_probs = []  # Store action probabilities
        self.values = []
        self.rewards = []
        self.dones = []
        self.action_features = []
        self.batch_size = batch_size
    
    def store(self, state, position, action, pos_prob, action_prob, value, reward, done, action_features):
        """
        Store a transition in memory
        
        Parameters:
        -----------
        state : np.ndarray
            State vector
        position : str
            Selected position
        action : int
            Action (player) taken
        pos_prob : float
            Probability of selecting the position
        action_prob : float
            Probability of taking the action
        value : float
            Value estimate
        reward : float
            Reward received
        done : bool
            Whether this is a terminal state
        action_features : np.ndarray
            Features for actions
        """
        self.states.append(state)
        self.positions.append(position)
        self.actions.append(action)
        self.pos_probs.append(pos_prob)
        self.action_probs.append(action_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.action_features.append(action_features)
    
    def clear(self):
        """Clear memory"""
        self.states = []
        self.positions = []
        self.actions = []
        self.pos_probs = []
        self.action_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.action_features = []
    
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