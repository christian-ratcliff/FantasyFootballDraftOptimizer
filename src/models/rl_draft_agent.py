#!/usr/bin/env python3
"""
Reinforcement Learning Environment for Fantasy Football Drafting.
Agent uses probabilistic VORP-weighted selection based on risk-adjusted VORP ranking.
Opponents use need-based probabilistic selection.
Reward based on VORP.
Corrected state management and logging.
ADDED DETAILED DEBUG LOGGING in reset, step, _simulate_opponent_picks, _get_state.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import logging
import random
import time 

# --- Constants ---
ACTION_BEST_QB = 0
ACTION_BEST_RB = 1
ACTION_BEST_WR = 2
ACTION_BEST_TE = 3
ACTION_BEST_FLEX = 4
ACTION_BEST_AVAILABLE = 5
NUM_ACTIONS = 6

TOP_N_PLAYERS_STATE = 50
RECENT_PICKS_WINDOW = 12
OPPONENT_CONSIDERATION_K = 3 # Opponents consider top K overall players for probabilistic choice
OPPONENT_NEED_BONUS_FACTOR = 1.8 # Boost score significantly if player fills starting need
AGENT_CONSIDERATION_K = 5 # Agent considers top K *valid* players for its chosen action
AGENT_RISK_PENALTY_FACTOR = 0.15 # Higher value = more risk averse agent selection
VALUE_DROP_OFF_N = 20 # Compare #1 overall VORP to #N overall VORP


class FantasyDraftEnv(gym.Env):
    """
    Custom Gym environment for simulating a fantasy football draft.
    Agent selects probabilistically based on VORP within chosen action category.
    Reward function based entirely on VORP.
    Opponent logic enhanced to consider positional need.
    Corrected state management.
    Includes detailed debug logging.
    """
    metadata = {'render_modes': ['human', 'logging'], 'render_fps': 1}

    def __init__(self, projections_df, league_settings, agent_draft_pos=1):
        super().__init__()
        # Use specific logger name configured in main script
        self.rl_logger = logging.getLogger("fantasy_draft_rl")
        self.rl_logger.debug(f"Env Init Start - ID {id(self)}")
        init_start_time = time.time()


        # --- Input Validation ---
        if projections_df is None or projections_df.empty:
            self.rl_logger.error("Env Init Error: projections_df cannot be None or empty.")
            raise ValueError("projections_df cannot be None or empty for FantasyDraftEnv.")
        if league_settings is None or not league_settings:
            self.rl_logger.error("Env Init Error: league_settings cannot be None or empty.")
            raise ValueError("league_settings cannot be None or empty for FantasyDraftEnv.")
        essential_cols = ['player_id', 'name', 'position', 'projected_points', 'age',
                        'projection_low', 'projection_high', 'ceiling_projection']
        missing_essentials = [col for col in essential_cols if col not in projections_df.columns]
        if missing_essentials:
            self.rl_logger.error(f"Env Init Error: projections_df missing essential columns: {missing_essentials}")
            raise ValueError(f"projections_df missing essential columns: {missing_essentials}")

        # --- Initialize Attributes ---
        self.rl_logger.debug("Env Init: Copying projections_df...")
        copy_start_time = time.time()
        self.projections_master = projections_df.copy() # Keep original safe
        self.rl_logger.debug(f"Env Init: Projections copied in {time.time() - copy_start_time:.3f}s. Shape: {self.projections_master.shape}")
        self.league_settings = league_settings
        self.render_mode = 'logging'
        self.episode_count = 0 # Track episodes for logging

        # --- Draft Parameters ---
        try:
            self.rl_logger.debug("Env Init: Calculating draft parameters...")
            self.num_teams = league_settings['league_info']['team_count']
            self.starter_slots = league_settings.get('starter_limits', {})
            self.total_roster_size, self.total_rounds = self._calculate_draft_length()

            self.total_picks = self.num_teams * self.total_rounds
            self.rl_logger.debug(f"Env Init: Calculated total_picks: {self.total_picks} ({self.num_teams} teams * {self.total_rounds} rounds)")
            self.draft_order = self._generate_draft_order(league_settings.get('draft_settings', {}))
            if not self.draft_order or len(self.draft_order) != self.total_picks: # Check full length
                self.rl_logger.error(f"Env Init Error: Draft order generation failed or wrong length ({len(self.draft_order) if self.draft_order else 'None'} vs {self.total_picks}).")
                raise ValueError("Draft order generation failed or incorrect length.")

            if not (1 <= agent_draft_pos <= self.num_teams):
                self.rl_logger.warning(f"Invalid agent_draft_pos {agent_draft_pos}, defaulting to 1.")
                agent_draft_pos = 1
            base_order = league_settings.get('draft_settings', {}).get('draft_order')
            if not base_order or len(base_order) != self.num_teams:
                base_order = list(range(1, self.num_teams + 1)) 
            self.agent_team_id = base_order[agent_draft_pos - 1]

            self.max_pos_counts = self._get_max_pos_counts(league_settings.get('roster_settings', {}))
            self.has_op_slot = self.starter_slots.get('OP', 0) > 0
            self.rl_logger.debug("Env Init: Draft parameters calculated.")
        except KeyError as e:
            self.rl_logger.error(f"Env Init Error: Missing key in league_settings: {e}", exc_info=True)
            raise ValueError(f"Invalid league_settings structure: Missing key {e}")
        except Exception as e:
            self.rl_logger.error(f"Env Init Error during parameter setup: {e}", exc_info=True)
            raise

        # --- Calculate Combined Score & VORP Baselines ---
        self.rl_logger.debug("Env Init: Calculating VORP/Risk-Adj VORP...")
        vorp_start_time = time.time()
        score_cols = ['projected_points', 'projection_low', 'ceiling_projection', 'projection_high'] # Added proj_high
        for col in score_cols:
            self.projections_master[col] = pd.to_numeric(self.projections_master[col], errors='coerce')
        # Fill NaNs before calculation
        self.projections_master['projected_points'].fillna(0.0, inplace=True)
        self.projections_master['projection_low'].fillna(self.projections_master['projected_points'] * 0.8, inplace=True)
        self.projections_master['ceiling_projection'].fillna(self.projections_master['projected_points'] * 1.5, inplace=True)
        self.projections_master['projection_high'].fillna(self.projections_master['projected_points'] * 1.2, inplace=True)

        self.projections_master['combined_score'] = (
            0.5*self.projections_master['projected_points'] +
            0.2*self.projections_master['projection_low'] +
            0.3*self.projections_master['ceiling_projection']
        ) # Already filled NaNs above
        self.replacement_levels = self._calculate_replacement_levels()
        self.rl_logger.debug(f"Env Init: VORP Replacement Levels: {self.replacement_levels}")
        self.projections_master['vorp'] = self.projections_master.apply(self._calculate_player_vorp, axis=1)

        self.projections_master['projection_range'] = (self.projections_master['projection_high'] - self.projections_master['projection_low']).clip(lower=0)
        self.projections_master['risk_adjusted_vorp'] = (self.projections_master['vorp'] - (self.projections_master['projection_range'] * AGENT_RISK_PENALTY_FACTOR)).fillna(0.0)
        self.projections_master['risk_adjusted_vorp'] = np.maximum(self.projections_master['risk_adjusted_vorp'], self.projections_master['vorp'] * 0.1) # Floor at 10% of original VORP

        # Ensure combined_score is non-NaN for opponent logic sorting fallback
        self.projections_master['combined_score'] = self.projections_master['combined_score'].fillna(0.0)

        # Sort master list by Risk-Adjusted VORP
        self.projections_master.sort_values('risk_adjusted_vorp', ascending=False, inplace=True)
        self.rl_logger.debug(f"Env Init: VORP/Risk-Adj VORP calculated in {time.time() - vorp_start_time:.3f}s")

        self.flex_replacement_level = np.mean([ self.replacement_levels.get('RB', 0), self.replacement_levels.get('WR', 0), self.replacement_levels.get('TE', 0) ])
        self.op_replacement_level = np.mean([ self.replacement_levels.get('QB', 0), self.replacement_levels.get('RB', 0), self.replacement_levels.get('WR', 0), self.replacement_levels.get('TE', 0) ])

        # --- Logging & Spaces ---
        self.rl_logger.debug(f"RL Env Init: {self.num_teams} teams, {self.total_rounds} rounds ({self.total_roster_size} spots), {self.total_picks} total picks.")
        self.rl_logger.debug(f"RL Env Init: Agent Team ID: {self.agent_team_id} (Draft Pos: {agent_draft_pos})")
        self.rl_logger.debug(f"RL Env Init: Max Roster Counts: {self.max_pos_counts}")
        self.rl_logger.debug(f"RL Env Init: Has OP Slot: {self.has_op_slot}")

        self.max_risk_adjusted_vorp = self.projections_master['risk_adjusted_vorp'].max() if not self.projections_master.empty else 15.0
        if self.max_risk_adjusted_vorp <= 0:
            self.rl_logger.warning(f"Max Risk-Adj VORP <= 0 ({self.max_risk_adjusted_vorp}). Using fallback 15.0")
            self.max_risk_adjusted_vorp = 15.0
        self.rl_logger.debug(f"Max Risk-Adjusted VORP for state normalization: {self.max_risk_adjusted_vorp:.2f}")

        self.rl_logger.debug("Env Init: Defining action space...")
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        self.rl_logger.debug(f"Env Init: Action space defined: {self.action_space}")
        
        # + 4 (Explicit Needs: QB, RB, WR, TE)
        # + 1 (Rounds Remaining)
        # + 1 (VORP #1 Overall Available)
        # + 1 (VORP Drop-off #1 vs #N)
        # + 4 (VORP #1 Available per Position: QB, RB, WR, TE)
        self.replacement_levels = self._calculate_replacement_levels()
        self._add_vorp_columns()
        self.max_risk_adjusted_vorp = self._get_max_norm_value()
        self.max_raw_vorp = self._get_max_raw_vorp() #
        state_size = 25#1 + 4 + 4 + 4 + 1 + 4# pick_norm, topN_Pos, agent_Pos, run_Pos, avg_top5_riskadj_vorp_norm
        self.rl_logger.debug(f"Env Init: Defining observation space (Size: {state_size})...")
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(state_size,), dtype=np.float32)
        self.rl_logger.debug(f"Env Init: Observation space defined: {self.observation_space}")

        # --- Internal State (Initialize empty before reset) ---
        self.available_players_df = pd.DataFrame()
        self.current_pick_overall = 0
        self.drafted_player_ids = set()
        self.teams_rosters = {}
        self.recent_picks_pos = []

        self.rl_logger.debug(f"Env Init Complete - ID {id(self)} (Total time: {time.time() - init_start_time:.3f}s)")


    def _calculate_draft_length(self):
        """Calculate total roster size and rounds more reliably."""
        starters = self.starter_slots
        roster_limits = self.league_settings.get('roster_settings', {})
        bench_size = roster_limits.get('BE', -1)

        tracked_starters = ['QB', 'RB', 'WR', 'TE', 'OP', 'RB/WR/TE', 'K', 'D/ST']
        num_starters = sum(count for pos, count in starters.items() if pos in tracked_starters)

        if bench_size < 0:
            self.rl_logger.warning("Bench size (BE) invalid or missing. Estimating 8 bench spots.")
            # Estimate bench size if not found or invalid
            bench_size = 8 # Common default
            total_size_guess = num_starters + bench_size
            rounds_guess = total_size_guess
            self.rl_logger.warning(f"Estimated Roster Size: {total_size_guess}, Rounds: {rounds_guess}")
            return total_size_guess, rounds_guess
        else:
            total_roster_size = num_starters + bench_size
            total_rounds = total_roster_size
            if total_rounds <= 0:
                self.rl_logger.error("Calculated total_rounds <= 0 (starters + bench). Defaulting to 16.")
                total_rounds = 16
                total_roster_size = total_rounds
            return total_roster_size, total_rounds


    def _reset_internal_state(self):
        """Helper to reset internal draft state variables for a new episode."""
        self.rl_logger.debug("Resetting internal state...")
        state_reset_start = time.time()
        self.current_pick_overall = 0
        # Ensure master copy is used and includes calculated columns
        if 'risk_adjusted_vorp' not in self.projections_master.columns:
            self.rl_logger.error("INTERNAL ERROR: risk_adjusted_vorp missing from projections_master during reset!")
            # Attempt recalculation as fallback
            self.projections_master['projection_range'] = (self.projections_master['projection_high'] - self.projections_master['projection_low']).fillna(0.0).clip(lower=0)
            self.projections_master['risk_adjusted_vorp'] = (self.projections_master['vorp'] - (self.projections_master['projection_range'] * AGENT_RISK_PENALTY_FACTOR)).fillna(0.0)
            self.projections_master['risk_adjusted_vorp'] = np.maximum(self.projections_master['risk_adjusted_vorp'], self.projections_master['vorp'] * 0.1)
            self.projections_master.sort_values('risk_adjusted_vorp', ascending=False, inplace=True)

        self.available_players_df = self.projections_master.copy() # Reset from master
        self.drafted_player_ids = set()

        # Determine team IDs based on draft order if possible
        base_order = self.league_settings.get('draft_settings', {}).get('draft_order')
        if base_order and len(base_order) == self.num_teams:
             self.team_ids = list(base_order) # Use the actual team IDs from settings
        else:
             self.team_ids = list(range(1, self.num_teams + 1)) # Fallback to sequential IDs

        min_team_id = min(self.team_ids)
        max_team_id = max(self.team_ids)
        # Ensure agent_team_id is valid within the determined team IDs
        if self.agent_team_id not in self.team_ids:
            self.rl_logger.warning(f"Agent ID {self.agent_team_id} not in derived team IDs {self.team_ids}. Correcting.")
            self.agent_team_id = self.team_ids[0] # Assign to the first team ID
            self.rl_logger.info(f"Corrected agent ID to {self.agent_team_id}.")

        self.teams_rosters = {team_id: [] for team_id in self.team_ids}
        self.recent_picks_pos = []
        self.rl_logger.debug(f"Internal state reset complete (took {time.time() - state_reset_start:.3f}s). Available players: {len(self.available_players_df)}")


    def set_render_mode(self, mode):
        if mode in self.metadata['render_modes']:
            self.render_mode = mode
        else:
            self.rl_logger.warning(f"Invalid render mode '{mode}'. Using '{self.render_mode}'.")

    def _get_max_pos_counts(self, roster_settings):
        """Determine max players per position from full roster settings."""
        defaults = {'QB': 3, 'RB': 8, 'WR': 8, 'TE': 3} # Adjusted defaults slightly
        counts = {}
        starter_keys = ['QB', 'RB', 'WR', 'TE']
        for pos in starter_keys:
            # Use roster_settings limit if exists, else default, ensure minimum of 1 if they are starters
            limit = roster_settings.get(pos, defaults.get(pos, 1 if self.starter_slots.get(pos, 0) > 0 else 0))
            # Ensure limit is at least the number of starters for that position
            min_req = self.starter_slots.get(pos, 0)
            counts[pos] = max(min_req, limit if limit > 0 else (defaults.get(pos, 1) if min_req > 0 else 1)) # Ensure valid positive number
        self.rl_logger.debug(f"Determined max roster counts: {counts}")
        return counts

    def _generate_draft_order(self, draft_settings):
        """Generates the full draft order for all rounds (snake)."""
        base_order = draft_settings.get('draft_order')
        if (not base_order or not isinstance(base_order, list) or
            len(base_order) != self.num_teams or len(set(base_order)) != self.num_teams):
            self.rl_logger.warning(f"Invalid 'draft_order' in settings. Using default 1-{self.num_teams}.")
            base_order = list(range(1, self.num_teams + 1))
        full_order = []
        for i in range(self.total_rounds):
            # Correctly reverse for snake draft
            current_round_order = list(base_order) if (i + 1) % 2 == 1 else list(reversed(base_order))
            full_order.extend(current_round_order)

        # Validate final length
        if len(full_order) != self.total_picks:
            self.rl_logger.error(f"Generated draft order length ({len(full_order)}) does not match total picks ({self.total_picks}).")
            # Attempt correction or raise error
            if len(full_order) > self.total_picks:
                full_order = full_order[:self.total_picks]
                self.rl_logger.warning("Truncated generated draft order.")
            else: # Too short - cannot proceed reliably
                raise ValueError("Generated draft order is shorter than total picks required.")

        self.rl_logger.debug(f"Full draft order ({len(full_order)} picks) generated.")#: {full_order[:self.num_teams*2]}...")
        return full_order

    def _calculate_replacement_levels(self):
        """Calculate baseline projection scores for replacement players."""
        self.rl_logger.debug("Calculating VORP replacement levels...")
        levels = {}
        num_starters_per_pos = {}
        # Get base starters
        req_qb = self.starter_slots.get('QB', 1); num_starters_per_pos['QB'] = req_qb
        req_rb = self.starter_slots.get('RB', 2); num_starters_per_pos['RB'] = req_rb
        req_wr = self.starter_slots.get('WR', 2); num_starters_per_pos['WR'] = req_wr
        req_te = self.starter_slots.get('TE', 1); num_starters_per_pos['TE'] = req_te
        req_op = self.starter_slots.get('OP', 0)
        req_flex = self.starter_slots.get('RB/WR/TE', 0) # Default to 0 if not present

        # Distribute OP and Flex needs proportionally (adjust weights as needed)
        # OP Slot distribution
        if req_op > 0:
            op_weight_qb = 0.7 # Higher weight for QB in OP
            op_weight_other = (1.0 - op_weight_qb) / 3.0
            num_starters_per_pos['QB'] += req_op * op_weight_qb
            num_starters_per_pos['RB'] += req_op * op_weight_other
            num_starters_per_pos['WR'] += req_op * op_weight_other
            num_starters_per_pos['TE'] += req_op * op_weight_other
            self.rl_logger.debug(f"Distributed {req_op} OP slots.")

        # Flex Slot distribution (RB/WR heavy)
        if req_flex > 0:
            flex_weight_rb = 0.4
            flex_weight_wr = 0.5
            flex_weight_te = 0.1
            num_starters_per_pos['RB'] += req_flex * flex_weight_rb
            num_starters_per_pos['WR'] += req_flex * flex_weight_wr
            num_starters_per_pos['TE'] += req_flex * flex_weight_te
            self.rl_logger.debug(f"Distributed {req_flex} FLEX slots.")

        self.rl_logger.debug(f"Effective starters per position used for VORP baseline: {num_starters_per_pos}")

        # Calculate replacement rank and score
        for pos, effective_starters in num_starters_per_pos.items():
            # Replacement player is roughly the N*M + 1th best player, where N=teams, M=starters
            replacement_rank = int(np.ceil(self.num_teams * effective_starters)) + 1
            self.rl_logger.debug(f"Calculating VORP baseline for {pos}: Effective Starters={effective_starters:.2f}, Replacement Rank={replacement_rank}")

            # Use projections_master which has projected_points
            pos_projections = self.projections_master[self.projections_master['position'] == pos]['projected_points']
            pos_projections = pos_projections.sort_values(ascending=False).reset_index(drop=True) # Ensure sorted and clean index

            if not pos_projections.empty:
                # Get score at replacement rank, handle index out of bounds
                if replacement_rank <= len(pos_projections):
                    levels[pos] = pos_projections.iloc[replacement_rank - 1]
                else:
                    # If rank is beyond available players, use the last player's score
                    levels[pos] = pos_projections.iloc[-1]
                    self.rl_logger.warning(f"Replacement rank {replacement_rank} for {pos} exceeds available players ({len(pos_projections)}). Using last player's score.")
            else:
                levels[pos] = 0.0 # No players of this position
                self.rl_logger.warning(f"No players found for position {pos} when calculating VORP baseline.")

            levels[pos] = max(0.0, levels[pos]) # Ensure non-negative
            self.rl_logger.debug(f"VORP Baseline for {pos}: {levels[pos]:.2f}")

        return levels


    def _calculate_player_vorp(self, player_row):
        """Calculates VORP for a player based on their position."""
        position = player_row.get('position')
        # Ensure projected points is treated as float, default to 0.0 if missing/invalid
        projection = pd.to_numeric(player_row.get('projected_points'), errors='coerce')
        if pd.isna(projection): projection = 0.0

        replacement_score = self.replacement_levels.get(position, 0.0)
        return max(0.0, projection - replacement_score)

    def reset(self, seed=None, options=None):
        """Resets the environment for a new episode."""
        self.rl_logger.debug(f"======= RESETTING ENVIRONMENT (Episode {self.episode_count + 1}) =======")
        reset_start_time = time.time()
        # Seeding
        super().reset(seed=seed)
        if seed is not None:
            self.rl_logger.debug(f"Resetting with seed: {seed}")
            random.seed(seed)
            np.random.seed(seed)
            # Seed action space for reproducibility if possible 
            if hasattr(self.action_space, 'seed'):
                try:
                    self.action_space.seed(seed)
                    self.rl_logger.debug("Action space seeded.")
                except NotImplementedError:
                    self.rl_logger.debug("Action space does not support seeding.")
        else:
            self.rl_logger.debug("Resetting without specific seed.")

        self.episode_count += 1
        self.rl_logger.debug(f"===== Starting Episode {self.episode_count} =====")

        self._reset_internal_state() # Resets rosters, available players, pick counter

        self.rl_logger.debug("Reset: Simulating initial opponent picks (if any)...")
        sim_opp_start_time = time.time()
        self._simulate_opponent_picks() # Simulate picks UP TO the agent's first turn
        self.rl_logger.debug(f"Reset: Initial opponent picks simulation complete (took {time.time() - sim_opp_start_time:.3f}s). Current pick: {self.current_pick_overall + 1}")

        self.rl_logger.debug("Reset: Getting initial state...")
        get_state_start_time = time.time()
        observation = self._get_state()
        self.rl_logger.debug(f"Reset: Initial state received (took {time.time() - get_state_start_time:.3f}s).")

        self.rl_logger.debug("Reset: Getting initial info...")
        info = self._get_info()
        self.rl_logger.debug(f"Reset: Initial info received: {info}")


        # Final validation of observation state
        if np.isnan(observation).any() or np.isinf(observation).any():
            self.rl_logger.error("!!! Initial state contains NaN/Inf after reset and simulation!!! Fixing.")
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=0.0)
            observation = np.clip(observation, 0.0, 1.0) # Clip again after nan_to_num

        if observation.shape != self.observation_space.shape:
            self.rl_logger.critical(f"FATAL STATE SHAPE MISMATCH AFTER RESET! Expected {self.observation_space.shape}, Got {observation.shape}. Returning zeros.")
            # Returning zeros might mask the issue, but thats a problem for later
            observation = np.zeros_like(self.observation_space.sample(), dtype=np.float32)


        self.rl_logger.debug(f"Reset complete (Total time: {time.time() - reset_start_time:.3f}s). Agent's turn (Pick {self.current_pick_overall + 1}). State shape: {observation.shape}")
        return observation, info

    def step(self, action):
        """Execute one agent step (pick) and simulate opponent responses."""
        # --- Overall Step Timing ---
        step_start_time = time.time()
        # Initialize durations for this step
        agent_action_duration = 0.0
        opp_sim_duration = 0.0
        get_state_duration = 0.0
        df_update_duration = 0.0 # Specific timing for the DF update after agent pick
        reward = 0.0 # Initialize reward
        done = False
        truncated = False
        info = {}
        observation = np.zeros_like(self.observation_space.sample(), dtype=np.float32) # Default observation

        self.rl_logger.debug(f"--- Step Start (Pick {self.current_pick_overall + 1}, Action: {action}) ---")

        # --- Action Validation ---
        if not isinstance(action, (int, np.integer)) or not (0 <= action < NUM_ACTIONS):
            self.rl_logger.error(f"Invalid action type/value received in step: {action} (type: {type(action)}). Defaulting to BPA ({ACTION_BEST_AVAILABLE}).")
            action = int(ACTION_BEST_AVAILABLE) # Ensure it's int
        else:
            action = int(action) # Ensure standard int type

        # --- Turn Check & Draft End Check ---
        if self.current_pick_overall >= self.total_picks:
            self.rl_logger.warning(f"Step called at pick {self.current_pick_overall + 1}, but draft already ended ({self.total_picks}). Returning terminal state.")
            get_state_start_final = time.time()
            final_state = self._get_state()
            get_state_duration = time.time() - get_state_start_final # Capture duration even on early exit
            # Ensure final state is valid
            if np.isnan(final_state).any() or np.isinf(final_state).any() or final_state.shape != self.observation_space.shape:
                self.rl_logger.error(f"Invalid terminal state detected. Shape: {final_state.shape}. Returning zeros.")
                final_state = np.zeros_like(self.observation_space.sample(), dtype=np.float32)
            step_total_duration = time.time() - step_start_time
            self.rl_logger.debug(f"Step Durations (Draft End Exit): Total={step_total_duration:.4f}s | GetState={get_state_duration:.4f}s")
            return final_state, 0.0, True, False, self._get_info() # Done = True

        # Verify it's the agent's turn
        current_team_id = self.draft_order[self.current_pick_overall]
        if current_team_id != self.agent_team_id:
            self.rl_logger.error(f"State Error: Agent acting on wrong turn (Pick {self.current_pick_overall+1}, Expected Team {self.agent_team_id}, Got Team {current_team_id}). Simulating opponents to recover.")
            # Attempt recovery by letting opponents pick
            recovery_sim_start = time.time()
            try:
                # **Important**: Ensure _simulate_opponent_picks has internal timing logs too
                self._simulate_opponent_picks()
                opp_sim_duration = time.time() - recovery_sim_start # Log recovery sim time
            except Exception as sim_err:
                opp_sim_duration = time.time() - recovery_sim_start # Capture duration up to error
                self.rl_logger.error(f"Recovery simulation failed: {sim_err}. Ending episode.", exc_info=True)
                get_state_start_err = time.time()
                final_state = self._get_state() # Get last possible state
                get_state_duration = time.time() - get_state_start_err
                if np.isnan(final_state).any() or np.isinf(final_state).any() or final_state.shape != self.observation_space.shape:
                    final_state = np.zeros_like(self.observation_space.sample(), dtype=np.float32)
                step_total_duration = time.time() - step_start_time
                self.rl_logger.error(f"Step Durations (Recovery Sim Error): Total={step_total_duration:.4f}s | OppSimBlock={opp_sim_duration:.4f}s | GetState={get_state_duration:.4f}s")
                return final_state, 0.0, True, False, self._get_info() # End episode after error

            # Check if draft ended during recovery or if turn is still wrong
            if self.current_pick_overall >= self.total_picks:
                self.rl_logger.warning("Draft ended during turn error recovery.")
                get_state_start_rec_end = time.time()
                final_state = self._get_state()
                get_state_duration = time.time() - get_state_start_rec_end
                if np.isnan(final_state).any() or np.isinf(final_state).any() or final_state.shape != self.observation_space.shape:
                    final_state = np.zeros_like(self.observation_space.sample(), dtype=np.float32)
                step_total_duration = time.time() - step_start_time
                self.rl_logger.debug(f"Step Durations (Recovery Draft End): Total={step_total_duration:.4f}s | OppSimBlock={opp_sim_duration:.4f}s | GetState={get_state_duration:.4f}s")
                return final_state, 0.0, True, False, self._get_info()

            # Re-check current team ID after recovery simulation
            current_team_id = self.draft_order[self.current_pick_overall]
            if current_team_id != self.agent_team_id:
                self.rl_logger.critical(f"Recovery failed: Agent turn still skipped after simulation. Expected {self.agent_team_id}, got {current_team_id}. Raising error.")
                raise RuntimeError(f"Agent turn synchronization error. Expected {self.agent_team_id}, got {current_team_id}.")
            else:
                self.rl_logger.info(f"Recovery successful. Now agent's turn (Pick {self.current_pick_overall + 1}).")


        # --- Execute Agent Pick (Probabilistic VORP-weighted, Risk-Adjusted Ranking) ---
        self.rl_logger.debug(f"Step: Executing agent action {action}...")
        agent_action_start = time.time() # Start timing agent action logic
        picked_player = self._execute_agent_action_probabilistic(action)
        agent_action_duration = time.time() - agent_action_start # End timing agent action logic
        player_info = "N/A" # For logging

        if picked_player and isinstance(picked_player, dict) and 'player_id' in picked_player:
            player_id_to_add = picked_player['player_id']
            player_pos = picked_player.get('position', 'UNK')
            if player_id_to_add in self.drafted_player_ids:
                # Recovery logic for picking drafted player
                self.rl_logger.error(f"CRITICAL AGENT LOGIC ERROR: Agent tried to pick already drafted player {player_id_to_add} ({picked_player.get('name', '?')})!")
                bpa_recovery = self._get_available_players().head(1)
                if not bpa_recovery.empty:
                    picked_player_rec = bpa_recovery.iloc[0].to_dict() # Use different var name
                    player_id_rec = picked_player_rec['player_id']
                    player_pos_rec = picked_player_rec.get('position', 'UNK')
                    if player_id_rec in self.drafted_player_ids:
                        self.rl_logger.critical("Recovery failed: BPA is also drafted. State is corrupt.")
                        raise RuntimeError("Agent failed to pick a valid player, even with recovery.")
                    else:
                        # Overwrite original invalid pick with recovery pick
                        picked_player = picked_player_rec
                        player_id_to_add = player_id_rec
                        player_pos = player_pos_rec
                        player_info = f"{picked_player.get('name', '?')} ({player_pos} VORP:{picked_player.get('vorp',0):.2f}) [RECOVERY PICK - DRAFTED ERROR]"
                        self.rl_logger.warning("Agent recovered by picking absolute BPA after attempting to pick drafted player.")
                else: # No players left at all?
                    self.rl_logger.critical("Recovery failed: No players left to pick.")
                    raise RuntimeError("Agent failed to pick a valid player - no players left.")

            else: # Normal successful pick
                player_info = f"{picked_player.get('name', '?')} ({player_pos} VORP:{picked_player.get('vorp',0):.2f})"

            # --- Time the state update part ---
            update_start = time.time()
            self.drafted_player_ids.add(player_id_to_add)
            self.teams_rosters[current_team_id].append(picked_player)
            self.recent_picks_pos.append(player_pos)
            # --- Time the DataFrame removal specifically ---
            df_update_local_start = time.time()
            prev_len = len(self.available_players_df)
            # THE POTENTIALLY SLOW LINE:
            self.available_players_df = self.available_players_df[self.available_players_df['player_id'] != player_id_to_add].copy()
            df_update_duration = time.time() - df_update_local_start # Duration of just the df update
            update_duration = time.time() - update_start # Duration of all state updates for agent pick
            # ------------------------------------------
            self.rl_logger.debug(f"Removed player {player_id_to_add} from available. Prev len: {prev_len}, New len: {len(self.available_players_df)}")
            if len(self.available_players_df) == prev_len:
                self.rl_logger.error(f"Logic Error: Failed to remove picked player {player_id_to_add} from available_players_df!")
            # Log update times (maybe DEBUG level is better unless specifically profiling)
            self.rl_logger.debug(f"Step Update Durations (Agent Pick): TotalUpdate={update_duration:.4f}s | DF Update Only={df_update_duration:.4f}s")
            # --- End state update timing ---

        else: # Agent action + fallback failed to return a valid player dict
            self.rl_logger.error(f"Agent action {action} (including fallback) FAILED to return a valid player at pick {self.current_pick_overall + 1}.")
            player_info = "FAILED TO PICK (Internal Error)"
            self.rl_logger.critical("Ending episode due to agent's inability to select a player.")
            get_state_start_fail = time.time()
            final_state = self._get_state() # Get state before returning
            get_state_duration = time.time() - get_state_start_fail
            if np.isnan(final_state).any() or np.isinf(final_state).any() or final_state.shape != self.observation_space.shape:
                final_state = np.zeros_like(self.observation_space.sample(), dtype=np.float32)
            step_total_duration = time.time() - step_start_time
            self.rl_logger.error(f"Step Durations (Agent Pick Fail): Total={step_total_duration:.4f}s | AgentAction={agent_action_duration:.4f}s | GetState={get_state_duration:.4f}s")
            # Return terminal state immediately
            return final_state, 0.0, True, False, self._get_info()


        # Render agent action result
        self.render(pick_info=player_info)

        # --- Advance Pick & Simulate Opponents ---
        self.current_pick_overall += 1
        self.rl_logger.debug(f"Step: Agent pick complete. Advanced to pick {self.current_pick_overall + 1}. Simulating opponents...")
        # --- Time opponent simulation block ---
        opp_sim_block_start = time.time()
        if self.current_pick_overall < self.total_picks:
            try:
                # **Important**: Ensure _simulate_opponent_picks has internal timing logs too
                self._simulate_opponent_picks()
                opp_sim_duration = time.time() - opp_sim_block_start # Total time for all opponent picks this step
                self.rl_logger.debug(f"Step: Opponent simulation block complete (took {opp_sim_duration:.4f}s). Next turn is pick {self.current_pick_overall + 1}.")
            except Exception as sim_err:
                opp_sim_duration = time.time() - opp_sim_block_start # Capture duration up to error
                self.rl_logger.error(f"Opponent simulation failed after agent pick: {sim_err}. Ending episode.", exc_info=True)
                get_state_start_opp_err = time.time()
                final_state = self._get_state() # Get state before returning
                get_state_duration = time.time() - get_state_start_opp_err
                if np.isnan(final_state).any() or np.isinf(final_state).any() or final_state.shape != self.observation_space.shape:
                    final_state = np.zeros_like(self.observation_space.sample(), dtype=np.float32)
                step_total_duration = time.time() - step_start_time
                self.rl_logger.error(f"Step Durations (Opp Sim Error): Total={step_total_duration:.4f}s | AgentAction={agent_action_duration:.4f}s | OppSimBlock={opp_sim_duration:.4f}s | GetState={get_state_duration:.4f}s")
                return final_state, 0.0, True, False, self._get_info() # End episode after error
        else:
            opp_sim_duration = 0.0 # No simulation needed
            self.rl_logger.debug("Step: Draft ended immediately after agent's pick.")
        # --- End opponent simulation timing ---


        # --- Determine Done Status and Reward ---
        done = self.current_pick_overall >= self.total_picks
        # Reward calculation timing can be added inside _calculate_final_reward_vorp if needed
        reward = self._calculate_final_reward_vorp() if done else 0.0
        if done:
            # Use DEBUG or INFO based on preference for seeing end-of-episode reward
            self.rl_logger.debug(f"===== Episode {self.episode_count} Finished. Final Reward (VORP-based): {reward:.4f} =====")

        # --- Get Next State and Info ---
        self.rl_logger.debug("Step: Getting next state and info...")
        get_state_start = time.time() # Start timing get_state
        observation = self._get_state()
        info = self._get_info()
        get_state_duration = time.time() - get_state_start # End timing get_state
        truncated = False # Assuming truncation isn't based on step count within this env
        self.rl_logger.debug(f"Step: State/Info obtained (GetState took {get_state_duration:.4f}s).")

        # Final state validation
        if np.isnan(observation).any() or np.isinf(observation).any():
            self.rl_logger.error(f"!!! State contains NaN/Inf at end of step {self.current_pick_overall} !!! Fixing.")
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=0.0)
            observation = np.clip(observation, 0.0, 1.0) # Ensure bounds
        if observation.shape != self.observation_space.shape:
            self.rl_logger.critical(f"FATAL STATE SHAPE MISMATCH AT END OF STEP! Expected {self.observation_space.shape}, Got {observation.shape}. Returning zeros.")
            observation = np.zeros_like(self.observation_space.sample(), dtype=np.float32)
            done = True # Force done if state is corrupt

        # --- Final Step Duration Log ---
        step_total_duration = time.time() - step_start_time
        self.rl_logger.debug(f"--- Step End (Completed Pick {self.current_pick_overall}, Action: {action}) ---")
        # Log all collected durations at INFO level for easier performance monitoring
        self.rl_logger.debug(
            f"Step Timing Breakdown (Pick {self.current_pick_overall}): " # Log the pick number *completed* in this step
            f"Total={step_total_duration:.4f}s | "
            f"AgentAct={agent_action_duration:.4f}s | "
            f"AgentDFUpd={df_update_duration:.4f}s | " # Specific DF update time
            f"OppSim={opp_sim_duration:.4f}s | "       # Total time for all opponent picks
            f"GetState={get_state_duration:.4f}s"
        )
        # ---

        return observation, reward, done, truncated, info

    def _add_vorp_columns(self):
        self.rl_logger.debug("Env Init: Adding VORP and RiskAdjVORP columns...")
        self.projections_master['vorp'] = self.projections_master.apply(self._calculate_player_vorp, axis=1)
        self.projections_master['projection_range'] = (self.projections_master['projection_high'] - self.projections_master['projection_low']).clip(lower=0.0).fillna(0.0)
        self.projections_master['risk_adjusted_vorp'] = (self.projections_master['vorp'] - (self.projections_master['projection_range'] * AGENT_RISK_PENALTY_FACTOR)).fillna(0.0)
        self.projections_master['risk_adjusted_vorp'] = np.maximum(self.projections_master['risk_adjusted_vorp'], self.projections_master['vorp'] * 0.1)
        self.projections_master.sort_values('risk_adjusted_vorp', ascending=False, inplace=True)
        self.rl_logger.debug("Env Init: VORP columns added and DataFrame sorted.")

    def _get_max_norm_value(self):
        """Gets max risk-adjusted VORP for normalization."""
        max_val = 1.0
        if not self.projections_master.empty and 'risk_adjusted_vorp' in self.projections_master.columns:
            max_val = self.projections_master['risk_adjusted_vorp'].max()
        if pd.isna(max_val) or max_val <= 0: max_val = 15.0; self.rl_logger.warning(f"Invalid Max RiskAdjVORP ({max_val}). Using fallback 15.0")
        else: self.rl_logger.debug(f"Env Init: Max RiskAdjVORP for state normalization: {max_val:.2f}")
        return max_val

    def _get_max_raw_vorp(self):
        """Gets max raw VORP for normalization (new state features)."""
        max_val = 1.0
        if not self.projections_master.empty and 'vorp' in self.projections_master.columns:
            max_val = self.projections_master['vorp'].max()
        if pd.isna(max_val) or max_val <= 0: max_val = 15.0; self.rl_logger.warning(f"Invalid Max Raw VORP ({max_val}). Using fallback 15.0")
        else: self.rl_logger.debug(f"Env Init: Max Raw VORP for state normalization: {max_val:.2f}")
        return max_val

    def _simulate_opponent_picks(self):
        """
        Simulate picks for opponents using smarter logic until agent's turn.
        OPTIMIZED: Filters available players *before* each opponent pick call
                using an explicit copy to avoid SettingWithCopyWarning.
        Includes internal timing.
        """
        sim_loop_start_time = time.time()
        picks_simulated = 0
        total_pick_logic_time = 0.0
        total_filter_time = 0.0 # Track filtering time separately

        self.rl_logger.debug(f"SimOpp Start: Loop from pick {self.current_pick_overall + 1}.")

        # Check if the main available DataFrame exists
        if not hasattr(self, 'available_players_df') or self.available_players_df is None:
            self.rl_logger.error("SimOpp Error: self.available_players_df does not exist!")
            raise AttributeError("self.available_players_df not initialized before opponent simulation.")
        if not isinstance(self.available_players_df, pd.DataFrame):
            self.rl_logger.error(f"SimOpp Error: self.available_players_df is not a DataFrame (type: {type(self.available_players_df)})!")
            raise TypeError("self.available_players_df is not a DataFrame.")
        if 'player_id' not in self.available_players_df.columns:
            self.rl_logger.error("SimOpp Error: 'player_id' column missing from self.available_players_df!")
            raise KeyError("'player_id' column missing from available_players_df.")

        while self.current_pick_overall < self.total_picks:
            current_team_id = self.draft_order[self.current_pick_overall]
            self.rl_logger.debug(f"SimOpp Eval: Pick {self.current_pick_overall + 1}, Team {current_team_id} (Agent is {self.agent_team_id}).")

            if current_team_id == self.agent_team_id:
                self.rl_logger.debug(f"SimOpp Stop: Agent turn arriving (Pick {self.current_pick_overall + 1}).")
                break # Agent's turn

            # --- Filter available players JUST BEFORE this opponent picks ---
            filter_start = time.time()
            try:
                # Ensure types match for filtering (safer)
                master_id_type = self.available_players_df['player_id'].dtype
                ids_to_check = self.drafted_player_ids
                if self.drafted_player_ids and not isinstance(list(self.drafted_player_ids)[0], master_id_type.type):
                    try:
                        # Attempt conversion if types differ
                        ids_to_check = set(master_id_type.type(pid) for pid in self.drafted_player_ids)
                    except Exception as e:
                        self.rl_logger.warning(f"SimOpp Warn: Type conversion failed for drafted_ids. Using original set. Error: {e}")
                        ids_to_check = self.drafted_player_ids # Use original set on error

                # Create the boolean mask first
                available_for_this_pick_mask = ~self.available_players_df['player_id'].isin(ids_to_check)
                available_for_this_pick = self.available_players_df.loc[available_for_this_pick_mask].copy()
                filter_duration = time.time() - filter_start
                total_filter_time += filter_duration # Accumulate filter time

            except KeyError:
                self.rl_logger.error("SimOpp Error: 'player_id' not found in self.available_players_df during pre-filter.")
                raise RuntimeError("Missing player_id in available players DataFrame.")
            except Exception as e:
                self.rl_logger.error(f"SimOpp Error during pre-filtering: {e}", exc_info=True)
                raise RuntimeError("Failed to filter available players for opponent.")
            # ------------------------------------------------------------

            # --- Opponent's turn ---
            opponent_logic_start_time = time.time()
            # Pass the explicit COPY and the current drafted set
            opponent_pick = self._pick_opponent_smarter(
                for_team_id=current_team_id,
                current_available_df=available_for_this_pick, # Pass the copy
                drafted_ids=self.drafted_player_ids          # Pass current set
            )
            total_pick_logic_time += (time.time() - opponent_logic_start_time)
            # ----------------------
            player_info = "N/A"

            if opponent_pick and isinstance(opponent_pick, dict) and 'player_id' in opponent_pick:
                player_id_to_add = opponent_pick['player_id']
                player_pos = opponent_pick.get('position', 'UNK')

                # --- Check against drafted_ids (Safety Net - should rarely be hit now) ---
                if player_id_to_add in self.drafted_player_ids:
                    self.rl_logger.error(f"SimOpp CRITICAL: Opponent {current_team_id} picked already drafted player {player_id_to_add} DESPITE pre-filtering!")
                    # --- Recovery logic using the 'available_for_this_pick' DataFrame ---
                    bpa_recovery_candidates = available_for_this_pick # Use the pre-filtered df
                    if 'projected_points' not in bpa_recovery_candidates.columns:
                        self.rl_logger.error("SimOpp Recovery Error: Missing 'projected_points' for BPA fallback.")
                        raise RuntimeError("Cannot perform recovery pick without projected_points.")
                    bpa_recovery = bpa_recovery_candidates.sort_values('projected_points', ascending=False).head(1)
                    if not bpa_recovery.empty:
                        opponent_pick = bpa_recovery.iloc[0].to_dict()
                        player_id_to_add = opponent_pick['player_id']
                        player_pos = opponent_pick.get('position', 'UNK')
                        if player_id_to_add in self.drafted_player_ids: # Final check
                            self.rl_logger.critical(f"SimOpp Recovery Failed: BPA {player_id_to_add} also in drafted_ids set.")
                            raise RuntimeError("Opponent simulation failed recovery.")
                        else:
                            player_info = f"{opponent_pick.get('name', '?')} ({player_pos}) [RECOVERY PICK - DRAFTED ERROR]"
                            self.rl_logger.warning(f"Opponent {current_team_id} recovered picking BPA {player_id_to_add} after filter failure.")
                    else:
                        self.rl_logger.critical(f"SimOpp Recovery Failed: No players left in pre-filtered list for Opponent {current_team_id}.")
                        raise RuntimeError("Opponent simulation recovery failed - no players left.")
                    # --- End Recovery ---
                else: # Normal pick
                    player_info = f"{opponent_pick.get('name', '?')} ({player_pos})"

                # --- Update state (Only add to set/list - FAST) ---
                self.drafted_player_ids.add(player_id_to_add)
                self.teams_rosters[current_team_id].append(opponent_pick)
                self.recent_picks_pos.append(player_pos)
                # --- NO DataFrame modification here ---

            else: # Opponent failed to make a selection
                self.rl_logger.error(f"Opponent {current_team_id} FAILED to make a pick at overall pick {self.current_pick_overall + 1}. available_for_this_pick size: {len(available_for_this_pick)}")
                player_info = "FAILED TO PICK (Opponent Internal Error)"
                # Added detail to error message
                raise RuntimeError(f"Opponent {current_team_id} failed to select a player when {len(available_for_this_pick)} players were available to them.")

            self.render(opponent_pick=player_info, team_id=current_team_id)
            self.current_pick_overall += 1
            picks_simulated += 1

            if self.current_pick_overall >= self.total_picks:
                self.rl_logger.debug("SimOpp: Draft ended during opponent simulation loop.")
                break

        sim_loop_total_time = time.time() - sim_loop_start_time
        avg_pick_logic_time = (total_pick_logic_time / picks_simulated) if picks_simulated > 0 else 0
        avg_filter_time = (total_filter_time / picks_simulated) if picks_simulated > 0 else 0

        # Log summary timing for the entire simulation block
        self.rl_logger.debug(
            f"SimOpp Finish: Loop done. Simulated {picks_simulated} picks. "
            f"TotalSimTime={sim_loop_total_time:.4f}s | "
            f"AvgFilter={avg_filter_time:.6f}s | "     
            f"AvgPickLogic={avg_pick_logic_time:.6f}s. "
            f"Next pick: {self.current_pick_overall + 1}"
        )

    def _get_available_players(self):
        """
        Returns the current DataFrame of available players,
        filtering the master list based on the drafted_player_ids set.
        OPTIMIZED: Filters master list instead of modifying a shared DF.
        """
        if not hasattr(self, 'projections_master') or self.projections_master.empty:
            self.rl_logger.error("INTERNAL ERROR: projections_master missing or empty in _get_available_players!")
            return pd.DataFrame()
        if not hasattr(self, 'drafted_player_ids'):
            self.rl_logger.error("INTERNAL ERROR: drafted_player_ids set missing in _get_available_players!")
            # Attempt to rebuild drafted_player_ids from rosters as a fallback
            self.drafted_player_ids = set()
            for team_id, roster in self.teams_rosters.items():
                for player in roster:
                    if 'player_id' in player:
                        self.drafted_player_ids.add(player['player_id'])
            self.rl_logger.warning("Rebuilt drafted_player_ids set from rosters.")

        try:
            # Convert drafted_ids to the same type as the DataFrame column if necessary (safer)
            master_id_type = self.projections_master['player_id'].dtype
            drafted_ids_list = list(self.drafted_player_ids) # Convert set to list for type checking/conversion
            if drafted_ids_list and not isinstance(drafted_ids_list[0], master_id_type.type):
                # Attempt conversion if types differ
                try:
                    converted_ids = set(master_id_type.type(pid) for pid in drafted_ids_list)
                    ids_to_check = converted_ids
                    self.rl_logger.debug(f"Converted drafted_ids for filtering: {len(ids_to_check)} ids.")
                except Exception as e:
                    self.rl_logger.error(f"Failed to convert drafted_ids type ({type(drafted_ids_list[0])}) to master type ({master_id_type}): {e}. Using original set.")
                    ids_to_check = self.drafted_player_ids # Use original set on error
            else:
                ids_to_check = self.drafted_player_ids # Types match or set is empty

            # Perform the efficient filtering using isin on the set/converted set
            available_mask = ~self.projections_master['player_id'].isin(ids_to_check)
            current_available = self.projections_master.loc[available_mask].copy() # Copy AFTER filtering
        except KeyError:
            self.rl_logger.error("INTERNAL ERROR: 'player_id' column not found in projections_master during filtering!")
            return pd.DataFrame()
        except Exception as filter_err:
            self.rl_logger.error(f"Error filtering available players: {filter_err}", exc_info=True)
            return pd.DataFrame()


        # Sort the result before returning (e.g., by risk-adjusted VORP for state)
        # Ensure the sort column exists
        sort_col = 'risk_adjusted_vorp'
        fallback_sort_col = 'projected_points'

        if sort_col in current_available.columns:
            # self.rl_logger.debug(f"_get_available_players: Returning {len(current_available)} players sorted by {sort_col}.")
            return current_available.sort_values(sort_col, ascending=False)
        elif fallback_sort_col in current_available.columns:
            self.rl_logger.warning(f"Sort column '{sort_col}' not found, sorting by '{fallback_sort_col}' instead.")
            return current_available.sort_values(fallback_sort_col, ascending=False)
        else:
            self.rl_logger.error(f"Neither '{sort_col}' nor '{fallback_sort_col}' found for sorting available players!")
            return current_available # Return unsorted

    def _execute_agent_action_probabilistic(self, action, force_fallback=False):
        """
        Selects agent pick probabilistically based on VORP, ensuring player is available.
        Uses self.drafted_player_ids for availability checks.
        """
        exec_start_time = time.time()
        self.rl_logger.debug(f"ExecAgentAction: Start Action={action}, Fallback={force_fallback}")

        agent_roster = self.teams_rosters.get(self.agent_team_id, [])
        agent_pos_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
        for p in agent_roster:
            pos = p.get('position')
            if pos and pos in agent_pos_counts:
                agent_pos_counts[pos] += 1
        self.rl_logger.debug(f"ExecAgentAction: Agent current counts: {agent_pos_counts}")

        # Get currently available players (already filtered based on drafted_player_ids by the function)
        # This list is sorted by risk_adjusted_vorp
        current_available = self._get_available_players()
        if current_available.empty:
            self.rl_logger.warning(f"ExecAgentAction: No players available at all for action {action}.")
            return None

        self.rl_logger.debug(f"ExecAgentAction: Top 5 Available (RiskAdjVORP) after filtering drafted:\n{current_available[['name','position','risk_adjusted_vorp']].head().to_string(index=False)}")

        candidates_df = pd.DataFrame()
        action_desc = "UNKNOWN"

        # 1. Determine Candidate Pool based on action category (from the already filtered available list)
        if action == ACTION_BEST_QB: candidates_df = current_available[current_available['position'] == 'QB'].copy(); action_desc = "Best QB"
        elif action == ACTION_BEST_RB: candidates_df = current_available[current_available['position'] == 'RB'].copy(); action_desc = "Best RB"
        elif action == ACTION_BEST_WR: candidates_df = current_available[current_available['position'] == 'WR'].copy(); action_desc = "Best WR"
        elif action == ACTION_BEST_TE: candidates_df = current_available[current_available['position'] == 'TE'].copy(); action_desc = "Best TE"
        elif action == ACTION_BEST_FLEX:
            flex_pos = ['RB','WR','TE'];
            if self.has_op_slot: flex_pos.append('QB')
            candidates_df = current_available[current_available['position'].isin(flex_pos)].copy(); action_desc = "Best FLEX"
        elif action == ACTION_BEST_AVAILABLE: candidates_df = current_available.copy(); action_desc = "Best Available (BPA)"
        else: self.rl_logger.error(f"ExecAgentAction: Unexpected agent action index: {action}"); candidates_df = current_available.copy(); action_desc = "BPA (Error Fallback)"

        if candidates_df.empty:
            self.rl_logger.debug(f"ExecAgentAction: Action '{action_desc}' yielded no initial candidates from available list.")
            if not force_fallback:
                self.rl_logger.debug("ExecAgentAction: Trying BPA fallback because initial action had no candidates.")
                # Call recursively with BPA action
                return self._execute_agent_action_probabilistic(ACTION_BEST_AVAILABLE, force_fallback=True)
            else:
                self.rl_logger.warning(f"ExecAgentAction: BPA Fallback failed: No candidates available at all.")
                return None # No players left for BPA either

        # 2. Filter Candidates by Roster Limits (Top K based on RiskAdjVORP)
        # candidates_df is already sorted by risk_adjusted_vorp from current_available
        self.rl_logger.debug(f"ExecAgentAction: Filtering top {len(candidates_df)} '{action_desc}' candidates by roster limits ({self.max_pos_counts}). Considering top {AGENT_CONSIDERATION_K} valid.")
        valid_candidates_list = []
        considered_count = 0
        for _, player_row in candidates_df.iterrows():
            considered_count += 1
            player_id = player_row['player_id']
            player_pos = player_row['position']

            # --- Roster Limit Check ---
            fits_limit = True
            current_count = agent_pos_counts.get(player_pos, 0)
            max_count = self.max_pos_counts.get(player_pos, 99) # Default high if not tracked

            if player_pos in self.max_pos_counts: # Check if the position IS tracked by limits
                if current_count >= max_count:
                    fits_limit = False
                    self.rl_logger.debug(f"ExecAgentAction: Candidate {player_row.get('name','?')} ({player_pos}) skipped, limit reached ({current_count} >= {max_count})")
            elif action in [ACTION_BEST_QB, ACTION_BEST_RB, ACTION_BEST_WR, ACTION_BEST_TE]: # If specific pos action but candidate doesn't match
                fits_limit = False # Ensure K/DEF etc. aren't picked for specific position actions
                self.rl_logger.debug(f"ExecAgentAction: Candidate {player_row.get('name','?')} ({player_pos}) skipped, non-matching position for specific action {action_desc}")

            if fits_limit:
                valid_candidates_list.append(player_row.to_dict())
                self.rl_logger.debug(f"ExecAgentAction: Added valid candidate {player_row.get('name','?')} ({player_pos}, ID: {player_id})")

            # Stop looking once we have enough valid options OR we've checked enough top players
            if len(valid_candidates_list) >= AGENT_CONSIDERATION_K:
                self.rl_logger.debug(f"ExecAgentAction: Reached consideration limit ({AGENT_CONSIDERATION_K}) for valid candidates.")
                break
            if considered_count >= TOP_N_PLAYERS_STATE: # Safety break
                self.rl_logger.debug(f"ExecAgentAction: Checked top {TOP_N_PLAYERS_STATE} potential candidates, stopping search.")
                break


        # 3. Handle No Valid Candidates / Fallback
        if not valid_candidates_list:
            self.rl_logger.debug(f"ExecAgentAction: Action '{action_desc}' yielded no candidates fitting limits within search window.")
            if not force_fallback:
                self.rl_logger.debug("ExecAgentAction: Trying BPA fallback because no valid candidates found for original action.")
                # Call recursively with BPA action
                return self._execute_agent_action_probabilistic(ACTION_BEST_AVAILABLE, force_fallback=True)
            else:
                self.rl_logger.warning(f"ExecAgentAction: BPA Fallback also failed: No player fits limits among available players.")
                # Last resort: find *any* available player respecting limits
                self.rl_logger.debug("ExecAgentAction: Last resort - checking ALL available players for ANY valid limit fit.")
                # Iterate through the FULL available list (already filtered for drafted)
                for _, player_row in current_available.iterrows():
                    pos = player_row['position']; fits_limit_fallback = True
                    if pos in self.max_pos_counts and agent_pos_counts.get(pos, 0) >= self.max_pos_counts[pos]: fits_limit_fallback = False
                    if fits_limit_fallback:
                            self.rl_logger.warning(f"ExecAgentAction: Last resort pick: {player_row.get('name','?')} ({pos}).")
                            return player_row.to_dict() # Return the first player that fits limits
                self.rl_logger.error("ExecAgentAction: LAST RESORT FAILED - No player fits limits among all available.")
                return None


        # 4. Probabilistic Choice based on VORP (from the valid, limit-fitting candidates)
        self.rl_logger.debug(f"ExecAgentAction: Performing probabilistic choice among {len(valid_candidates_list)} valid candidates.")
        candidate_final_df = pd.DataFrame(valid_candidates_list)
        # Ensure 'vorp' exists, default to 0 if not (should be present from projections)
        if 'vorp' not in candidate_final_df.columns: candidate_final_df['vorp'] = 0.0
        candidate_final_df['vorp'] = pd.to_numeric(candidate_final_df['vorp'], errors='coerce').fillna(0.0)

        self.rl_logger.debug(f"ExecAgentAction: Candidate Pool for Prob Choice:\n{candidate_final_df[['name','position','vorp','risk_adjusted_vorp']].round(2).to_string(index=False)}")

        weights = candidate_final_df['vorp'].clip(lower=0.01).values # Use VORP for weighting, clip low

        chosen_player = None; log_msg = ""
        if np.sum(weights) <= 0.01 * len(weights): # If sum of weights is near zero
            # Pick the one with the highest *risk-adjusted* VORP (already sorted)
            chosen_player = candidate_final_df.iloc[0].to_dict(); log_msg = "picking highest RiskAdjVORP (all candidates ~0 VORP)"
            self.rl_logger.debug("ExecAgentAction: All candidates had near-zero VORP, selecting top by RiskAdjVORP.")
        elif len(candidate_final_df) == 1:
            chosen_player = candidate_final_df.iloc[0].to_dict(); log_msg = "picking only valid candidate"
            self.rl_logger.debug("ExecAgentAction: Only one valid candidate found.")
        else:
            probabilities = weights / np.sum(weights)
            probabilities = np.nan_to_num(probabilities, nan=0.0); probabilities[probabilities < 0] = 0
            if np.sum(probabilities) == 0:
                probabilities = np.ones(len(candidate_final_df)) / len(candidate_final_df)
                self.rl_logger.warning("ExecAgentAction: All probabilities became zero. Using equal probability.")
            else:
                probabilities /= np.sum(probabilities)

            try:
                chosen_index = np.random.choice(len(candidate_final_df), p=probabilities)
                chosen_player = candidate_final_df.iloc[chosen_index].to_dict(); log_msg = f"prob VORP choice from {len(candidate_final_df)}"
                self.rl_logger.debug(f"ExecAgentAction: Chose index {chosen_index} with probabilities {np.round(probabilities, 3)}")
            except ValueError as e:
                self.rl_logger.error(f"ExecAgentAction: Prob choice ValueError: {e}. Prob sum: {np.sum(probabilities)}. Picking highest RiskAdjVORP.")
                chosen_player = candidate_final_df.iloc[0].to_dict(); log_msg = "picking highest RiskAdjVORP (prob error)"

        self.rl_logger.debug(f"Agent action '{action_desc}' -> {log_msg}. Picked: {chosen_player.get('name', '?')} ({chosen_player.get('position','?')}) VORP: {chosen_player.get('vorp',0):.2f} RiskAdjVORP: {chosen_player.get('risk_adjusted_vorp',0):.2f}")
        self.rl_logger.debug(f"ExecAgentAction finished (took {time.time() - exec_start_time:.3f}s)")
        return chosen_player

    def _pick_opponent_smarter(self, for_team_id, current_available_df, drafted_ids):
        """
        Opponent strategy: Considers need and value from the passed available players,
        picks probabilistically, ensuring player is not in the passed drafted_ids set.
        OPTIMIZED: Uses passed dataframes/sets, nlargest for candidate selection.

        Args:
            for_team_id: The ID of the team making the pick.
            current_available_df (pd.DataFrame): DataFrame of players currently available
                                                (already filtered for previously drafted players
                                                before this simulation block started).
            drafted_ids (set): Set of player IDs already drafted in the entire draft.
        """
        opp_pick_start_time = time.time()
        self.rl_logger.debug(f"PickOpponent Start: Team {for_team_id}. Available passed: {len(current_available_df)}. Drafted passed: {len(drafted_ids)}.")

        # --- Input Validation ---
        if current_available_df.empty:
            self.rl_logger.warning(f"PickOpponent Warn: Passed available DF is empty for Team {for_team_id}.")
            return None
        # Ensure required columns exist in the passed DataFrame
        required_cols = ['player_id', 'position', 'combined_score', 'projected_points', 'name']


        # --- Get Opponent Roster Info ---
        opp_roster = self.teams_rosters.get(for_team_id, [])
        opp_pos_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
        for player_dict in opp_roster:
            pos = player_dict.get('position')
            if pos and pos in opp_pos_counts:
                opp_pos_counts[pos] += 1
        self.rl_logger.debug(f"PickOpponent: Team {for_team_id} counts: {opp_pos_counts}") 

        # --- Calculate Needs ---
        # Optimize dictionary creation
        req_starters = {'QB': self.starter_slots.get('QB', 1), 'RB': self.starter_slots.get('RB', 2), 
                    'WR': self.starter_slots.get('WR', 2), 'TE': self.starter_slots.get('TE', 1)}
        needs = {pos: max(0, req - opp_pos_counts.get(pos, 0)) for pos, req in req_starters.items()}
        
        total_starters_count = sum(v for k, v in self.starter_slots.items() if k in ['QB','RB','WR','TE','OP','RB/WR/TE'])
        starters_filled_count = sum(opp_pos_counts.get(p, 0) for p in req_starters)
        num_flex_op_needed = max(0, total_starters_count - starters_filled_count)
        self.rl_logger.debug(f"PickOpponent: Team {for_team_id} Needs: {needs}, Flex/OP Needed: {num_flex_op_needed}")

        # --- Select Initial Candidate Pool using nlargest  ---
        consider_k_initial = OPPONENT_CONSIDERATION_K * 2
        numeric_df = current_available_df.copy()
        numeric_df['combined_score'] = pd.to_numeric(numeric_df['combined_score'], errors='coerce').fillna(0.0)
        
        # Get top K based on combined_score
        top_k_overall_candidates = numeric_df.nlargest(consider_k_initial, 'combined_score')

        if top_k_overall_candidates.empty:
            self.rl_logger.warning(f"PickOpponent Warn: No candidates found for Team {for_team_id} using nlargest from available list.")
            # Fallback: try getting the single best from the original available df
            fallback_best = numeric_df.nlargest(1, 'combined_score')
            if fallback_best.empty:
                self.rl_logger.error(f"PickOpponent Error: Absolutely no players available for Team {for_team_id}.")
                return None
            else:
                top_k_overall_candidates = fallback_best # Use the single best as the pool

        # --- Convert dataframe to more efficient list of dicts for processing ---
        candidates = top_k_overall_candidates.to_dict('records')
        
        # --- Apply Need Bonus to the Candidate Pool ---
        # Pre-compute position sets for faster lookups
        is_flex_positions = {'RB', 'WR', 'TE'}
        is_op_positions = {'QB', 'RB', 'WR', 'TE'} if self.has_op_slot else set()
        
        temp_needs = needs.copy()
        temp_flex_op_filled = 0
        
        for i, player in enumerate(candidates):
            player_pos = player['position']
            base_score = player['combined_score']
            bonus = 1.0  # Reset bonus each iteration
            
            # Check starter need first
            if player_pos in temp_needs and temp_needs[player_pos] > 0:
                bonus = OPPONENT_NEED_BONUS_FACTOR
                temp_needs[player_pos] -= 1  # Decrement temporary need count
            # Then check flex/op need
            elif (num_flex_op_needed - temp_flex_op_filled) > 0:
                is_flex = player_pos in is_flex_positions
                is_op = player_pos in is_op_positions
                
                if is_op and player_pos == 'QB': 
                    bonus = 1.0+(OPPONENT_NEED_BONUS_FACTOR-1.0)*0.7
                    temp_flex_op_filled += 1
                elif is_op: 
                    bonus = 1.0+(OPPONENT_NEED_BONUS_FACTOR-1.0)*0.6
                    temp_flex_op_filled += 1
                elif is_flex: 
                    bonus = 1.0+(OPPONENT_NEED_BONUS_FACTOR-1.0)*0.5
                    temp_flex_op_filled += 1

            # Update adjusted score in candidate dict
            player['adjusted_score'] = base_score * bonus
            candidates[i] = player

        # --- Sort Candidates by Adjusted Score ---
        candidates.sort(key=lambda x: x.get('adjusted_score', 0), reverse=True)
        
        # --- Filter candidates by limits/availability (no iterrows) ---
        valid_candidates = []
        considered_count = 0
        
        for player in candidates:
            considered_count += 1
            player_id = player['player_id']
            player_pos = player['position']

            # --- Availability Check using PASSED drafted_ids set ---
            if player_id in drafted_ids:
                # This check should ideally not be hit often if the input current_available_df was correct
                self.rl_logger.warning(f"PickOpponent Warn: Candidate {player.get('name','?')} ({player_id}) found in passed drafted_ids set. Skipping.")
                continue

            # --- Roster Limit Check ---
            current_count = opp_pos_counts.get(player_pos, 0)
            max_count = self.max_pos_counts.get(player_pos, 99) # Default high if not tracked

            if player_pos in self.max_pos_counts and current_count >= max_count:
                continue  # Skip if position limit exceeded

            # Valid candidate
            valid_candidates.append(player)

            # --- Stop once enough valid candidates are found or sufficient checked ---
            if len(valid_candidates) >= OPPONENT_CONSIDERATION_K:
                break
            # Safety break
            if considered_count >= len(candidates) or considered_count >= OPPONENT_CONSIDERATION_K * 5:
                break

        # --- Probabilistic Choice / Fallback ---
        if not valid_candidates:
            self.rl_logger.debug(f"PickOpponent Warn: Team {for_team_id} - No valid candidates found after bonus & limits. Falling back to BPA respecting limits from PASSED available list.")
            # Sort by combined_score once for fallback
            bpa_fallback_df = numeric_df.sort_values('combined_score', ascending=False)
            
            for _, player_row in bpa_fallback_df.iterrows():
                # --- Availability Check using PASSED drafted_ids set ---
                if player_row['player_id'] in drafted_ids:
                    continue
                
                fallback_pos = player_row['position']
                current_count_fb = opp_pos_counts.get(fallback_pos, 0)
                max_count_fb = self.max_pos_counts.get(fallback_pos, 99)
                
                if fallback_pos in self.max_pos_counts and current_count_fb >= max_count_fb:
                    continue  # Skip if limit exceeded

                # Valid fallback found
                self.rl_logger.debug(f"PickOpponent Info: Team {for_team_id} selecting BPA fallback: {player_row.get('name', '?')} ({fallback_pos}).")
                pick_duration = time.time() - opp_pick_start_time
                self.rl_logger.debug(f"PickOpponent Finish (Fallback BPA): Team {for_team_id}. Took {pick_duration:.6f}s.")
                return player_row.to_dict() # Return the first BPA that fits limits and is available
            
            # Fallback failed
            self.rl_logger.error(f"PickOpponent Error: Team {for_team_id} BPA fallback FAILED - no available player fits limits.")
            return None

        # --- Optimized probabilistic choice among valid candidates ---
        # Use numpy arrays for faster calculation
        projected_points = np.array([float(c.get('projected_points', 0.0)) for c in valid_candidates])
        projected_points = np.clip(projected_points, 0.1, None)  # Clip to avoid zeros

        chosen_player = None
        log_msg = ""
        
        if np.sum(projected_points) <= 0.1 * len(projected_points): # If sum is near zero
            chosen_player = valid_candidates[0]
            log_msg = "picking highest AdjScore (all candidates ~0 proj pts)"
        elif len(valid_candidates) == 1:
            chosen_player = valid_candidates[0]
            log_msg = "picking only valid candidate"
        else:
            # Optimized probability calculation
            with np.errstate(divide='ignore', invalid='ignore'):
                probabilities = projected_points / np.sum(projected_points)
            
            # Handle edge cases
            probabilities = np.nan_to_num(probabilities, nan=0.0)
            probabilities = np.maximum(0, probabilities)
            
            if np.sum(probabilities) == 0:
                probabilities = np.ones(len(valid_candidates)) / len(valid_candidates)
                self.rl_logger.warning(f"PickOpponent Warn: Team {for_team_id} probabilities zero, using uniform.")
            else:
                probabilities /= np.sum(probabilities)

            try:
                idx = np.random.choice(len(valid_candidates), p=probabilities)
                chosen_player = valid_candidates[idx]
                log_msg = f"prob choice from {len(valid_candidates)}"
            except ValueError as e:
                self.rl_logger.error(f"PickOpponent Error: Team {for_team_id} prob choice ValueError: {e}. Prob sum: {np.sum(probabilities)}. Picking highest AdjScore.")
                chosen_player = valid_candidates[0]
                log_msg = "picking highest AdjScore (prob error)"

        pick_duration = time.time() - opp_pick_start_time
        # Log the final pick outcome
        self.rl_logger.debug(f"PickOpponent Finish: Team {for_team_id} {log_msg}. Picked: {chosen_player.get('name', '?')} ({chosen_player.get('position','?')}). Took {pick_duration:.6f}s.")
        return chosen_player
    
    def _get_state(self):
            """Constructs the NEW expanded and normalized state vector."""
            # get_state_start = time.time()
            # self.rl_logger.debug(f"GetState: Start (Pick {self.current_pick_overall + 1})")
            try:
                state = np.zeros(self.observation_space.shape, dtype=np.float32)
                current_available = self.available_players_df.sort_values('risk_adjusted_vorp', ascending=False) # Ensure sorted locally
                idx = 0 # Index for state vector

                # --- Original Features ---
                # 1. Pick Normalization
                state[idx] = min(1.0, self.current_pick_overall / max(1, self.total_picks -1)); idx += 1

                # 2. Top N Available per Position %
                top_overall_df = current_available.head(TOP_N_PLAYERS_STATE)
                total_in_top_n = max(1, len(top_overall_df)) # Avoid division by zero
                for pos in ['QB','RB','WR','TE']:
                    count = 0
                    if not top_overall_df.empty and 'position' in top_overall_df.columns:
                        count = len(top_overall_df[top_overall_df['position'] == pos])
                    state[idx] = count / total_in_top_n; idx += 1

                # 3. Agent Roster Counts Normalization
                agent_roster = self.teams_rosters.get(self.agent_team_id, [])
                counts = {'QB':0,'RB':0,'WR':0,'TE':0}
                for p in agent_roster: pos = p.get('position');
                if pos in counts: counts[pos]+=1
                for pos in ['QB','RB','WR','TE']:
                    max_c = max(1, self.max_pos_counts.get(pos, 1))
                    state[idx] = min(1.0, counts.get(pos, 0) / max_c); idx += 1

                # 4. Recent Positional Runs Normalization
                recent = self.recent_picks_pos[-RECENT_PICKS_WINDOW:]
                runs = {'QB':0,'RB':0,'WR':0,'TE':0}
                for pos in recent:
                    if pos in runs: runs[pos] += 1
                norm_recent = max(1, len(recent)) # Normalize by actual number of recent picks if < window size
                for pos in ['QB','RB','WR','TE']:
                    state[idx] = runs.get(pos, 0) / norm_recent; idx += 1

                # --- NEW Features ---
                # 5. Explicit Needs (Normalized by required starters)
                needs = {}
                for pos in ['QB', 'RB', 'WR', 'TE']:
                    req = self.starter_slots.get(pos, 0) # Get required starters
                    needs[pos] = max(0, req - counts.get(pos, 0))
                    state[idx] = min(1.0, needs[pos] / max(1, req)) if req > 0 else 0.0 # Normalize need
                    #self.rl_logger.debug(f"GetState: state[{idx}] (need_{pos}) = {state[idx]:.4f}")
                    idx += 1

                # 6. Rounds Remaining (Normalized)
                current_round = (self.current_pick_overall // self.num_teams) + 1
                rounds_remaining = max(0, self.total_rounds - current_round)
                state[idx] = rounds_remaining / max(1, self.total_rounds -1); idx += 1
                #self.rl_logger.debug(f"GetState: state[{idx-1}] (rounds_rem) = {state[idx-1]:.4f}")


                # 7. VORP #1 Overall Available (Normalized by max RAW VORP)
                vorp1 = 0.0
                if not current_available.empty and 'vorp' in current_available.columns:
                    # Get top player by RAW vorp, not risk-adjusted, for value signal
                    top_raw_vorp_player = current_available.sort_values('vorp', ascending=False).iloc[0]
                    vorp1 = top_raw_vorp_player['vorp']
                state[idx] = max(0.0, vorp1 / max(1.0, self.max_raw_vorp)); idx += 1
                #self.rl_logger.debug(f"GetState: state[{idx-1}] (vorp1_overall) = {state[idx-1]:.4f}")


                # 8. VORP Drop-off #1 vs #N (Normalized difference)
                vorpN = 0.0; vorp_drop = 0.0
                if len(current_available) >= VALUE_DROP_OFF_N and 'vorp' in current_available.columns:
                    # Use raw VORP for drop-off calculation
                    sorted_by_vorp = current_available.sort_values('vorp', ascending=False)
                    vorp1_drop = sorted_by_vorp.iloc[0]['vorp'] # Already calculated vorp1 above effectively
                    vorpN = sorted_by_vorp.iloc[VALUE_DROP_OFF_N - 1]['vorp']
                    vorp_drop = max(0.0, vorp1_drop - vorpN) # Difference, ensure non-negative
                state[idx] = vorp_drop / max(1.0, self.max_raw_vorp); idx += 1 # Normalize by max possible VORP
                #self.rl_logger.debug(f"GetState: state[{idx-1}] (vorp_drop1-{VALUE_DROP_OFF_N}) = {state[idx-1]:.4f}")


                # 9. VORP #1 Available per Position (Normalized by max RAW VORP)
                for pos in ['QB','RB','WR','TE']:
                    vorp1_pos = 0.0
                    pos_available = current_available[current_available['position'] == pos]
                    if not pos_available.empty and 'vorp' in pos_available.columns:
                        # Use raw VORP
                        vorp1_pos = pos_available.sort_values('vorp', ascending=False).iloc[0]['vorp']
                    state[idx] = max(0.0, vorp1_pos / max(1.0, self.max_raw_vorp)); idx += 1
                    #self.rl_logger.debug(f"GetState: state[{idx-1}] (vorp1_{pos}) = {state[idx-1]:.4f}")

                # --- OLD Feature (Avg Top 5 RiskAdj VORP) --- Now at end
                # 10. Average Risk-Adjusted VORP of Top 5 Available (Normalized by max RiskAdj VORP)
                top_5 = current_available.head(5) # Already sorted by risk_adjusted_vorp
                top_5_avg_value = 0.0
                if not top_5.empty and 'risk_adjusted_vorp' in top_5.columns:
                    top_5_numeric = pd.to_numeric(top_5['risk_adjusted_vorp'], errors='coerce')
                    top_5_avg_value = top_5_numeric.mean();
                    if pd.isna(top_5_avg_value): top_5_avg_value = 0.0
                norm_max_risk_adj = max(1.0, self.max_risk_adjusted_vorp)
                state[idx] = max(0.0, top_5_avg_value / norm_max_risk_adj); idx += 1
                #self.rl_logger.debug(f"GetState: state[{idx-1}] (avg_top5_riskadj_vorp) = {state[idx-1]:.4f}")

                # --- Final State Validation ---
                if idx != len(state): self.rl_logger.error(f"GetState: Size mismatch! Expected {len(state)}, filled {idx}.")
                state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
                state = np.clip(state, 0.0, 1.0)

                #self.rl_logger.debug(f"GetState finished ({(time.time() - get_state_start):.4f}s). State: {np.round(state, 3)}")
                return state

            except Exception as e:
                self.rl_logger.critical(f"CRITICAL ERROR in _get_state: {e}", exc_info=True)
                return np.zeros(self.observation_space.shape, dtype=np.float32)

    def _get_info(self):
        """Returns auxiliary information about the environment state."""
        info_start_time = time.time()
        current_available = self._get_available_players(); avail_counts = {}
        if not current_available.empty and 'position' in current_available.columns:
            avail_counts = {p: len(current_available[current_available['position'] == p]) for p in ['QB', 'RB', 'WR', 'TE']}
        else: avail_counts = {p: 0 for p in ['QB', 'RB', 'WR', 'TE']}

        top_5_info = []
        if not current_available.empty:
            cols=['name','position','projected_points','vorp', 'risk_adjusted_vorp']
            cols_avail=[c for c in cols if c in current_available.columns]
            if cols_avail: # Check if any columns to display exist
                try:
                    top_5_info = current_available[cols_avail].head(5).round(2).to_dict('records')
                except Exception as e:
                    self.rl_logger.warning(f"Error creating top_5_info dict: {e}")

        info_dict = {"current_pick": self.current_pick_overall + 1,
                    "round": (self.current_pick_overall // self.num_teams) + 1,
                    "pick_in_round": (self.current_pick_overall % self.num_teams) + 1,
                    "agent_roster_size": len(self.teams_rosters.get(self.agent_team_id, [])),
                    "players_drafted": len(self.drafted_player_ids),
                    "available_player_count": len(current_available),
                    "available_counts_pos": avail_counts,
                    "top_5_available": top_5_info}
        self.rl_logger.debug(f"GetInfo finished (took {time.time() - info_start_time:.3f}s)")
        return info_dict

    def _calculate_final_reward_vorp(self):
        """
        Calculates episodic reward based on VORP of starters + weighted VORP of bench,
        INCLUDING an explicit penalty for unfilled required starting slots.
        """
        reward_start_time = time.time()
        self.rl_logger.debug("Calculating final reward (VORP-based with penalty)...")
        agent_roster_list = self.teams_rosters.get(self.agent_team_id, [])

        # Handle empty roster case
        if not agent_roster_list:
            self.rl_logger.warning("Agent roster empty during final reward calculation.")
            return 0.0

        try:
            # Ensure roster is a DataFrame and VORP is calculated/present
            agent_roster_df = pd.DataFrame(agent_roster_list)
            if 'vorp' not in agent_roster_df.columns or agent_roster_df['vorp'].isnull().any():
                self.rl_logger.debug("Recalculating VORP for final reward.")
                # Ensure projected_points exists and is numeric before calculating VORP
                if 'projected_points' not in agent_roster_df.columns:
                    self.rl_logger.error("Reward Calc Error: Roster DF missing 'projected_points' for VORP recalc.")
                    # Attempt to merge from master list as a fallback (less ideal)
                    if 'player_id' in agent_roster_df.columns and hasattr(self, 'projections_master'):
                        agent_roster_df = pd.merge(
                            agent_roster_df.drop(columns=['projected_points'], errors='ignore'),
                            self.projections_master[['player_id', 'projected_points']],
                            on='player_id',
                            how='left'
                        )
                        agent_roster_df['projected_points'] = pd.to_numeric(agent_roster_df['projected_points'], errors='coerce').fillna(0.0)
                    else:
                        raise ValueError("Missing projected_points and cannot merge.")

                agent_roster_df['vorp'] = agent_roster_df.apply(self._calculate_player_vorp, axis=1)

            # Ensure essential columns exist and VORP is numeric
            if 'player_id' not in agent_roster_df.columns or 'position' not in agent_roster_df.columns:
                self.rl_logger.error("Reward Calc Error: Roster DF missing player_id or position.")
                raise ValueError("Missing ID/Pos in roster df for reward calc")
            agent_roster_df['vorp'] = pd.to_numeric(agent_roster_df['vorp'], errors='coerce').fillna(0.0)

            self.rl_logger.debug(f"Agent Roster for Reward Calc (Size: {len(agent_roster_df)}):\n{agent_roster_df[['name','position','vorp']].round(2).to_string(index=False)}")

        except Exception as e:
            self.rl_logger.error(f"Reward Calc DF Creation/VORP Error: {e}", exc_info=True)
            return 0.0

        # --- Determine Starters based on VORP within required slots ---
        agent_roster_df = agent_roster_df.sort_values('vorp', ascending=False)
        starters_vorp_sum = 0.0
        players_used = set()
        filled_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0, 'OP': 0, 'FLEX': 0} # Track filled slots

        # Define required starter counts from league settings
        req_counts = {
            'QB': self.starter_slots.get('QB', 1),
            'RB': self.starter_slots.get('RB', 2),
            'WR': self.starter_slots.get('WR', 2),
            'TE': self.starter_slots.get('TE', 1),
            'OP': self.starter_slots.get('OP', 0),
            'FLEX': self.starter_slots.get('RB/WR/TE', 0) # Use the specific flex key from settings
        }
        self.rl_logger.debug(f"Reward Calc: Required Starter Counts: {req_counts}")

        # 1. Fill required position slots first (QB, RB, WR, TE)
        for pos, req_count in [('QB', req_counts['QB']), ('RB', req_counts['RB']), ('WR', req_counts['WR']), ('TE', req_counts['TE'])]:
            # Select top VORP players for the position, excluding those already used
            starters = agent_roster_df[
                (agent_roster_df['position'] == pos) &
                (~agent_roster_df['player_id'].isin(players_used))
            ].head(req_count)

            filled_count = len(starters)
            filled_counts[pos] = filled_count # Update filled count

            # Log warning if required slots couldn't be filled
            if filled_count < req_count:
                self.rl_logger.warning(f"Reward Calc Warning: Could only fill {filled_count}/{req_count} required {pos} slots.")

            starters_vorp_sum += starters['vorp'].sum()
            players_used.update(starters['player_id'])
            self.rl_logger.debug(f"Reward Calc: Filled {pos} slots ({filled_count}/{req_count}). VORP added: {starters['vorp'].sum():.2f}. Players used: {len(players_used)}")

        # 2. Fill OP (Superflex) slots (QB > RB > WR > TE based on VORP)
        if req_counts['OP'] > 0:
            op_eligible_pos = ['QB','RB','WR','TE']
            # Select top VORP players from eligible positions, excluding those already used
            op_starters = agent_roster_df[
                agent_roster_df['position'].isin(op_eligible_pos) &
                (~agent_roster_df['player_id'].isin(players_used))
            ].head(req_counts['OP'])

            filled_count = len(op_starters)
            filled_counts['OP'] = filled_count # Update filled count

            if filled_count < req_counts['OP']:
                self.rl_logger.warning(f"Reward Calc Warning: Could only fill {filled_count}/{req_counts['OP']} OP slots.")

            starters_vorp_sum += op_starters['vorp'].sum()
            players_used.update(op_starters['player_id'])
            self.rl_logger.debug(f"Reward Calc: Filled OP slots ({filled_count}/{req_counts['OP']}). VORP added: {op_starters['vorp'].sum():.2f}. Players used: {len(players_used)}")

        # 3. Fill FLEX slots (RB > WR > TE based on VORP)
        if req_counts['FLEX'] > 0:
            flex_eligible_pos = ['RB','WR','TE']
            # Select top VORP players from eligible positions, excluding those already used
            flex_starters = agent_roster_df[
                agent_roster_df['position'].isin(flex_eligible_pos) &
                (~agent_roster_df['player_id'].isin(players_used))
            ].head(req_counts['FLEX'])

            filled_count = len(flex_starters)
            filled_counts['FLEX'] = filled_count # Update filled count

            if filled_count < req_counts['FLEX']:
                self.rl_logger.warning(f"Reward Calc Warning: Could only fill {filled_count}/{req_counts['FLEX']} FLEX slots.")

            starters_vorp_sum += flex_starters['vorp'].sum()
            players_used.update(flex_starters['player_id'])
            self.rl_logger.debug(f"Reward Calc: Filled FLEX slots ({filled_count}/{req_counts['FLEX']}). VORP added: {flex_starters['vorp'].sum():.2f}. Players used: {len(players_used)}")

        # 4. Calculate Bench VORP
        bench_players = agent_roster_df[~agent_roster_df['player_id'].isin(players_used)]
        bench_vorp_sum = bench_players['vorp'].sum()
        bench_weight = 0.15 # Weight bench VORP less
        base_reward_value = starters_vorp_sum + bench_weight * bench_vorp_sum

        # --- 5. Calculate Penalty for Unfilled Required Slots ---
        unfilled_penalty = 0.0

        penalties = {
            'QB': 20.0, # Higher penalty for QB
            'TE': 15.0, # Higher penalty for TE
            'RB': 10.0, # Original/Standard penalty for RB
            'WR': 10.0  # Original/Standard penalty for WR
        }
        # Only penalize for core required positions (QB, RB, WR, TE)
        required_core_starters = {'QB': req_counts['QB'], 'RB': req_counts['RB'], 'WR': req_counts['WR'], 'TE': req_counts['TE']}

        for pos, required in required_core_starters.items():
            penalty_per_slot = penalties.get(pos, 10.0)
            filled = filled_counts.get(pos, 0) # Get filled count for this position
            if filled < required:
                unfilled_count = required - filled
                current_penalty = unfilled_count * penalty_per_slot
                unfilled_penalty += current_penalty
                self.rl_logger.warning(f"Reward Penalty Applied: Unfilled {pos} slots: {unfilled_count}. Adding penalty: {current_penalty:.2f}")

        # --- 6. Calculate Final Reward ---
        final_reward_value = base_reward_value - unfilled_penalty

        # Ensure the final reward is a finite number
        if not np.isfinite(final_reward_value):
            self.rl_logger.error(f"Non-finite VORP reward calculated: {final_reward_value}. Base: {base_reward_value}, Penalty: {unfilled_penalty}. Returning 0.0.")
            return 0.0

        # Log final calculation details
        self.rl_logger.debug(f"Reward Calc (VORP): Starters Sum={starters_vorp_sum:.2f} ({len(players_used)} used), Bench Sum={bench_vorp_sum:.2f} ({len(bench_players)} players), Penalty={unfilled_penalty:.2f}, Final Reward={final_reward_value:.4f}")
        self.rl_logger.debug(f"Reward calculation finished (took {time.time() - reward_start_time:.3f}s)")

        return final_reward_value


    def render(self, pick_info=None, opponent_pick=None, team_id=None):
        """Log draft picks based on render mode."""
        if self.render_mode not in ['logging', 'human']: return

        log_pick_num = self.current_pick_overall # The pick NUMBER just completed or being processed
        if log_pick_num >= len(self.draft_order): # Handle potential index error if called after draft ends
            current_team_id_for_log = "N/A"; round_num, pick_in_round = "End", "End"
        else:
            pick_index_just_made = log_pick_num # Refers to the index in draft_order
            round_num = (pick_index_just_made // self.num_teams) + 1
            pick_in_round = (pick_index_just_made % self.num_teams) + 1
            # Use provided team_id if available (from opponent sim), else derive from order
            current_team_id_for_log = team_id if team_id is not None else self.draft_order[pick_index_just_made]

        prefix = f"Pick {pick_index_just_made+1:3d} (R{round_num:2d}.{pick_in_round:2d}) Team {current_team_id_for_log:>2}"

        # Determine if it was the agent's pick based on team ID
        is_agent_pick = (current_team_id_for_log == self.agent_team_id)

        if is_agent_pick and pick_info:
            log_message = f"*** {prefix} (Agent) drafts: {pick_info} ***"
            self.rl_logger.debug(log_message)
            if self.render_mode == 'human': print(log_message) # Also print for human mode
        elif not is_agent_pick and opponent_pick:
            log_message = f"    {prefix}         drafts: {opponent_pick}"
            # Log opponent picks at INFO level if verbose, otherwise DEBUG
            if self.rl_logger.isEnabledFor(logging.DEBUG):
                self.rl_logger.debug(log_message)
            elif self.rl_logger.isEnabledFor(logging.INFO): # Or higher (Warning etc)
                self.rl_logger.debug(log_message) # Log opponent picks if logger set to INFO
            if self.render_mode == 'human': print(log_message)


    def close(self):
        """Clean up any resources."""
        self.rl_logger.info("Closing FantasyDraftEnv.")
        pass


