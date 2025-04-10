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
import time # For timing logs

# Configure logging for the environment
# logger = logging.getLogger(__name__) # Use specific logger below
# Basic handler if run standalone
# if not logger.hasHandlers():
#      handler = logging.StreamHandler()
#      formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#      handler.setFormatter(formatter)
#      logger.addHandler(handler)
#      logger.setLevel(logging.INFO) # Set default level if run standalone


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
OPPONENT_CONSIDERATION_K = 5 # Opponents consider top K overall players for probabilistic choice
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
                 base_order = list(range(1, self.num_teams + 1)) # Use default if invalid
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
             # Seed action space for reproducibility if possible (depends on space type)
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
             # Returning zeros might mask the issue, consider raising an error?
             observation = np.zeros_like(self.observation_space.sample(), dtype=np.float32)


        self.rl_logger.debug(f"Reset complete (Total time: {time.time() - reset_start_time:.3f}s). Agent's turn (Pick {self.current_pick_overall + 1}). State shape: {observation.shape}")
        return observation, info

    def step(self, action):
        """Execute one agent step (pick) and simulate opponent responses."""
        step_start_time = time.time()
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
            final_state = self._get_state()
            # Ensure final state is valid
            if np.isnan(final_state).any() or np.isinf(final_state).any() or final_state.shape != self.observation_space.shape:
                self.rl_logger.error(f"Invalid terminal state detected. Shape: {final_state.shape}. Returning zeros.")
                final_state = np.zeros_like(self.observation_space.sample(), dtype=np.float32)
            return final_state, 0.0, True, False, self._get_info() # Done = True

        # Verify it's the agent's turn
        current_team_id = self.draft_order[self.current_pick_overall]
        if current_team_id != self.agent_team_id:
            self.rl_logger.error(f"State Error: Agent acting on wrong turn (Pick {self.current_pick_overall+1}, Expected Team {self.agent_team_id}, Got Team {current_team_id}). Simulating opponents to recover.")
            # Attempt recovery by letting opponents pick
            try:
                self._simulate_opponent_picks()
            except Exception as sim_err:
                 self.rl_logger.error(f"Recovery simulation failed: {sim_err}. Ending episode.", exc_info=True)
                 final_state = self._get_state() # Get last possible state
                 if np.isnan(final_state).any() or np.isinf(final_state).any() or final_state.shape != self.observation_space.shape:
                      final_state = np.zeros_like(self.observation_space.sample(), dtype=np.float32)
                 return final_state, 0.0, True, False, self._get_info() # End episode after error

            # Check if draft ended during recovery or if turn is still wrong
            if self.current_pick_overall >= self.total_picks:
                self.rl_logger.warning("Draft ended during turn error recovery.")
                final_state = self._get_state()
                if np.isnan(final_state).any() or np.isinf(final_state).any() or final_state.shape != self.observation_space.shape: final_state = np.zeros_like(self.observation_space.sample(), dtype=np.float32)
                return final_state, 0.0, True, False, self._get_info()
            # Re-check current team ID
            current_team_id = self.draft_order[self.current_pick_overall]
            if current_team_id != self.agent_team_id:
                self.rl_logger.critical(f"Recovery failed: Agent turn still skipped after simulation. Expected {self.agent_team_id}, got {current_team_id}. Raising error.")
                raise RuntimeError(f"Agent turn synchronization error. Expected {self.agent_team_id}, got {current_team_id}.")

        # --- Execute Agent Pick (Probabilistic VORP-weighted, Risk-Adjusted Ranking) ---
        self.rl_logger.debug(f"Step: Executing agent action {action}...")
        picked_player = self._execute_agent_action_probabilistic(action)
        player_info = "N/A" # For logging

        if picked_player and isinstance(picked_player, dict) and 'player_id' in picked_player:
            player_id_to_add = picked_player['player_id']
            player_pos = picked_player.get('position', 'UNK')
            if player_id_to_add in self.drafted_player_ids:
                 # This indicates a logic error either in available_players update or agent choice
                 self.rl_logger.error(f"CRITICAL AGENT LOGIC ERROR: Agent tried to pick already drafted player {player_id_to_add} ({picked_player.get('name', '?')})!")
                 # Attempt to pick the absolute best available as a last resort recovery
                 bpa_recovery = self._get_available_players().head(1)
                 if not bpa_recovery.empty:
                      picked_player = bpa_recovery.iloc[0].to_dict()
                      player_id_to_add = picked_player['player_id']
                      player_pos = picked_player.get('position', 'UNK')
                      if player_id_to_add in self.drafted_player_ids:
                           # If even BPA is taken, something is fundamentally broken
                           self.rl_logger.critical("Recovery failed: BPA is also drafted. State is corrupt.")
                           raise RuntimeError("Agent failed to pick a valid player, even with recovery.")
                      else:
                           player_info = f"{picked_player.get('name', '?')} ({player_pos} VORP:{picked_player.get('vorp',0):.2f}) [RECOVERY PICK - DRAFTED ERROR]"
                           self.rl_logger.warning("Agent recovered by picking absolute BPA after attempting to pick drafted player.")
                 else: # No players left at all?
                      self.rl_logger.critical("Recovery failed: No players left to pick.")
                      raise RuntimeError("Agent failed to pick a valid player - no players left.")

            else: # Normal successful pick
                 player_info = f"{picked_player.get('name', '?')} ({player_pos} VORP:{picked_player.get('vorp',0):.2f})"

            # Update state ONLY IF a valid player was determined (original or recovery)
            self.drafted_player_ids.add(player_id_to_add)
            self.teams_rosters[current_team_id].append(picked_player)
            self.recent_picks_pos.append(player_pos)
            # Update available players DF - crucial step
            prev_len = len(self.available_players_df)
            self.available_players_df = self.available_players_df[self.available_players_df['player_id'] != player_id_to_add].copy() # Use .copy()
            self.rl_logger.debug(f"Removed player {player_id_to_add} from available. Prev len: {prev_len}, New len: {len(self.available_players_df)}")
            if len(self.available_players_df) == prev_len:
                 self.rl_logger.error(f"Logic Error: Failed to remove picked player {player_id_to_add} from available_players_df!")

        else: # Agent action + fallback failed to return a valid player dict
            self.rl_logger.error(f"Agent action {action} (including fallback) FAILED to return a valid player at pick {self.current_pick_overall + 1}.")
            player_info = "FAILED TO PICK (Internal Error)"
            # Cannot proceed without a pick, treat as terminal? Or raise error?
            # For stability, let's end the episode here.
            self.rl_logger.critical("Ending episode due to agent's inability to select a player.")
            final_state = self._get_state(); reward = 0.0; done = True; truncated = False; info = self._get_info()
            return final_state, reward, done, truncated, info


        # Render agent action result
        self.render(pick_info=player_info)

        # --- Advance Pick & Simulate Opponents ---
        self.current_pick_overall += 1
        self.rl_logger.debug(f"Step: Agent pick complete. Advanced to pick {self.current_pick_overall + 1}. Simulating opponents...")
        sim_opp_start_time = time.time()
        if self.current_pick_overall < self.total_picks:
             try:
                self._simulate_opponent_picks()
                self.rl_logger.debug(f"Step: Opponent simulation complete (took {time.time() - sim_opp_start_time:.3f}s). Next turn is pick {self.current_pick_overall + 1}.")
             except Exception as sim_err:
                  self.rl_logger.error(f"Opponent simulation failed after agent pick: {sim_err}. Ending episode.", exc_info=True)
                  final_state = self._get_state(); reward = 0.0; done = True; truncated = False; info = self._get_info()
                  return final_state, reward, done, truncated, info
        else:
             self.rl_logger.debug("Step: Draft ended immediately after agent's pick.")


        # --- Determine Done Status and Reward ---
        done = self.current_pick_overall >= self.total_picks
        reward = self._calculate_final_reward_vorp() if done else 0.0 # Reward is only non-zero at the end
        if done:
             self.rl_logger.debug(f"===== Episode {self.episode_count} Finished. Final Reward (VORP-based): {reward:.4f} =====")

        # --- Get Next State and Info ---
        self.rl_logger.debug("Step: Getting next state and info...")
        get_state_start_time = time.time()
        observation = self._get_state()
        info = self._get_info()
        truncated = False # Truncation not typically used here unless step limit imposed externally
        self.rl_logger.debug(f"Step: State/Info obtained (took {time.time() - get_state_start_time:.3f}s).")

        # Final state validation
        if np.isnan(observation).any() or np.isinf(observation).any():
            self.rl_logger.error(f"!!! State contains NaN/Inf at end of step {self.current_pick_overall} !!! Fixing.")
            observation = np.nan_to_num(observation, nan=0.0, posinf=1.0, neginf=0.0)
            observation = np.clip(observation, 0.0, 1.0)
        if observation.shape != self.observation_space.shape:
             self.rl_logger.critical(f"FATAL STATE SHAPE MISMATCH AT END OF STEP! Expected {self.observation_space.shape}, Got {observation.shape}. Returning zeros.")
             observation = np.zeros_like(self.observation_space.sample(), dtype=np.float32)
             done = True # Force done if state is corrupt

        self.rl_logger.debug(f"--- Step End (Pick {self.current_pick_overall}, Action: {action}, Picked: {player_info}, Reward: {reward:.4f}, Done: {done}) --- (Total time: {time.time() - step_start_time:.3f}s)")

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
        """Simulate picks for opponents using smarter logic until agent's turn."""
        sim_loop_start_time = time.time()
        picks_simulated = 0
        self.rl_logger.debug(f"SimOpp: Starting loop from pick {self.current_pick_overall + 1}.")

        while self.current_pick_overall < self.total_picks:
            current_team_id = self.draft_order[self.current_pick_overall]
            self.rl_logger.debug(f"SimOpp: Evaluating pick {self.current_pick_overall + 1}, Team {current_team_id} (Agent is {self.agent_team_id}).")

            if current_team_id == self.agent_team_id:
                self.rl_logger.debug(f"SimOpp: Agent turn arriving (Pick {self.current_pick_overall + 1}). Stopping simulation.")
                break # Agent's turn

            # Opponent's turn
            opponent_pick_start_time = time.time()
            opponent_pick = self._pick_opponent_smarter(for_team_id=current_team_id)
            player_info = "N/A"

            if opponent_pick and isinstance(opponent_pick, dict) and 'player_id' in opponent_pick:
                player_id_to_add = opponent_pick['player_id']
                player_pos = opponent_pick.get('position', 'UNK')

                if player_id_to_add in self.drafted_player_ids:
                     self.rl_logger.error(f"SimOpp CRITICAL LOGIC ERROR: Opponent {current_team_id} tried picking already drafted player {player_id_to_add} ({opponent_pick.get('name','?')})!")
                     # Attempt recovery by picking next best available (BPA based on raw projected points as fallback)
                     bpa_recovery = self.available_players_df[~self.available_players_df['player_id'].isin(self.drafted_player_ids)].sort_values('projected_points', ascending=False).head(1)
                     if not bpa_recovery.empty:
                         opponent_pick = bpa_recovery.iloc[0].to_dict()
                         player_id_to_add = opponent_pick['player_id']
                         player_pos = opponent_pick.get('position', 'UNK')
                         if player_id_to_add in self.drafted_player_ids:
                              # If even BPA recovery is taken, state is corrupt
                              self.rl_logger.critical(f"SimOpp Recovery Failed: BPA {player_id_to_add} also drafted. State corrupt.")
                              raise RuntimeError("Opponent simulation failed to pick valid player, even with recovery.")
                         else:
                              player_info = f"{opponent_pick.get('name', '?')} ({player_pos}) [RECOVERY PICK - DRAFTED ERROR]"
                              self.rl_logger.warning(f"Opponent {current_team_id} recovered by picking BPA {player_id_to_add} after trying drafted player.")
                     else:
                         self.rl_logger.critical(f"SimOpp Recovery Failed: No players left for Opponent {current_team_id}.")
                         raise RuntimeError("Opponent simulation failed - no players left.")
                else: # Normal pick
                     player_info = f"{opponent_pick.get('name', '?')} ({player_pos})"

                # Update state (only if valid player determined)
                self.drafted_player_ids.add(player_id_to_add)
                self.teams_rosters[current_team_id].append(opponent_pick)
                self.recent_picks_pos.append(player_pos)
                # Update available players DF
                prev_len = len(self.available_players_df)
                self.available_players_df = self.available_players_df[self.available_players_df['player_id'] != player_id_to_add].copy() # Use .copy()
                self.rl_logger.debug(f"SimOpp: Removed opp pick {player_id_to_add}. Prev Avail len: {prev_len}, New Avail len: {len(self.available_players_df)}")
                if len(self.available_players_df) == prev_len:
                     self.rl_logger.error(f"SimOpp Logic Error: Failed to remove opponent picked player {player_id_to_add} from available_players_df!")

            else: # Opponent failed to make a selection
                 self.rl_logger.error(f"Opponent {current_team_id} FAILED to make a pick at overall pick {self.current_pick_overall + 1}. This should not happen if players are available.")
                 player_info = "FAILED TO PICK (Opponent Internal Error)"
                 # If an opponent fails, we probably should stop the simulation as state might be bad
                 raise RuntimeError(f"Opponent {current_team_id} failed to select a player.")

            self.render(opponent_pick=player_info, team_id=current_team_id) # Log the opponent pick
            self.current_pick_overall += 1
            picks_simulated += 1
            self.rl_logger.debug(f"SimOpp: Pick {self.current_pick_overall} simulated (Team {current_team_id}, Picked: {player_info}) in {time.time() - opponent_pick_start_time:.3f}s.")

            # Check if draft ended within the loop
            if self.current_pick_overall >= self.total_picks:
                self.rl_logger.debug("SimOpp: Draft ended during opponent simulation loop.")
                break

        self.rl_logger.debug(f"SimOpp: Loop finished. Simulated {picks_simulated} picks (Total time: {time.time() - sim_loop_start_time:.3f}s). Current pick: {self.current_pick_overall + 1}")


    # --- Agent Action Execution ---
    def _execute_agent_action_probabilistic(self, action, force_fallback=False):
        """Translates agent action into a probabilistic VORP-weighted player pick,
           considering candidates ranked by risk-adjusted VORP."""
        exec_start_time = time.time()
        self.rl_logger.debug(f"ExecAgentAction: Start Action={action}, Fallback={force_fallback}")

        agent_roster = self.teams_rosters.get(self.agent_team_id, [])
        agent_pos_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
        for p in agent_roster:
            pos = p.get('position')
            if pos and pos in agent_pos_counts:
                agent_pos_counts[pos] += 1
        self.rl_logger.debug(f"ExecAgentAction: Agent current counts: {agent_pos_counts}")


        current_available = self._get_available_players() # Sorted by risk_adjusted_vorp
        if current_available.empty:
            self.rl_logger.warning(f"ExecAgentAction: No players available for action {action}.")
            return None

        self.rl_logger.debug(f"ExecAgentAction: Top 5 Available (RiskAdjVORP): \n{current_available[['name','position','risk_adjusted_vorp']].head().to_string(index=False)}")

        candidates_df = pd.DataFrame()
        action_desc = "UNKNOWN"

        # 1. Determine Candidate Pool based on action category
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
            self.rl_logger.debug(f"ExecAgentAction: Action '{action_desc}' yielded no initial candidates.")
            if not force_fallback:
                self.rl_logger.debug("ExecAgentAction: Trying BPA fallback because initial action had no candidates.")
                return self._execute_agent_action_probabilistic(ACTION_BEST_AVAILABLE, force_fallback=True)
            else:
                 self.rl_logger.warning(f"ExecAgentAction: BPA Fallback failed: No candidates available at all.")
                 return None # No players left for BPA either

        # 2. Filter Candidates by Roster Limits (Top K based on RiskAdjVORP)
        self.rl_logger.debug(f"ExecAgentAction: Filtering top {len(candidates_df)} '{action_desc}' candidates by roster limits ({self.max_pos_counts}). Considering top {AGENT_CONSIDERATION_K} valid.")
        valid_candidates_list = []
        considered_count = 0
        for _, player_row in candidates_df.iterrows(): # Already sorted by risk_adjusted_vorp
             considered_count += 1
             pos = player_row['position']; fits_limit = True
             if pos in self.max_pos_counts:
                  if agent_pos_counts.get(pos, 0) >= self.max_pos_counts[pos]:
                       fits_limit = False
                       self.rl_logger.debug(f"ExecAgentAction: Candidate {player_row.get('name','?')} ({pos}) skipped, limit reached ({agent_pos_counts.get(pos, 0)} >= {self.max_pos_counts[pos]})")
             else: # Position not tracked by limits (e.g., K, DEF) - allow if action wasn't position-specific
                  if action not in [ACTION_BEST_QB, ACTION_BEST_RB, ACTION_BEST_WR, ACTION_BEST_TE]:
                       fits_limit = True # Allow K/DEF etc. if action is FLEX or BPA
                  else:
                       fits_limit = False # Don't allow K/DEF if specific position was requested
                       self.rl_logger.debug(f"ExecAgentAction: Candidate {player_row.get('name','?')} ({pos}) skipped, non-matching position for action {action_desc}")


             if fits_limit:
                  valid_candidates_list.append(player_row.to_dict())
                  self.rl_logger.debug(f"ExecAgentAction: Added valid candidate {player_row.get('name','?')} ({pos})")
             # Stop looking once we have enough valid options OR we've checked enough top players
             if len(valid_candidates_list) >= AGENT_CONSIDERATION_K:
                  self.rl_logger.debug(f"ExecAgentAction: Reached consideration limit ({AGENT_CONSIDERATION_K}) for valid candidates.")
                  break
             if considered_count >= TOP_N_PLAYERS_STATE: # Safety break to avoid iterating too far if limits are tight
                  self.rl_logger.debug(f"ExecAgentAction: Checked top {TOP_N_PLAYERS_STATE} candidates, stopping search.")
                  break


        # 3. Handle No Valid Candidates / Fallback
        if not valid_candidates_list:
             self.rl_logger.debug(f"ExecAgentAction: Action '{action_desc}' yielded no candidates fitting limits within search window.")
             if not force_fallback:
                 self.rl_logger.debug("ExecAgentAction: Trying BPA fallback because no valid candidates found for original action.")
                 return self._execute_agent_action_probabilistic(ACTION_BEST_AVAILABLE, force_fallback=True)
             else:
                  self.rl_logger.warning(f"ExecAgentAction: BPA Fallback also failed: No player fits limits among available players.")
                  # Last resort: find *any* available player respecting limits, even if VORP is low/0
                  self.rl_logger.debug("ExecAgentAction: Last resort - checking ALL available players for ANY valid limit fit.")
                  for _, player_row in current_available.iterrows():
                       pos = player_row['position']; fits_limit = True
                       if pos in self.max_pos_counts and agent_pos_counts.get(pos, 0) >= self.max_pos_counts[pos]: fits_limit = False
                       if fits_limit:
                            self.rl_logger.warning(f"ExecAgentAction: Last resort pick: {player_row.get('name','?')} ({pos}).")
                            return player_row.to_dict() # Return the first player that fits limits
                  self.rl_logger.error("ExecAgentAction: LAST RESORT FAILED - No player fits limits among all available.")
                  return None


        # 4. Probabilistic Choice based on VORP (from the risk-adjusted pool)
        self.rl_logger.debug(f"ExecAgentAction: Performing probabilistic choice among {len(valid_candidates_list)} valid candidates.")
        candidate_final_df = pd.DataFrame(valid_candidates_list)
        self.rl_logger.debug(f"ExecAgentAction: Candidate Pool for Prob Choice:\n{candidate_final_df[['name','position','vorp','risk_adjusted_vorp']].round(2).to_string(index=False)}")

        # Weight by VORP, clip low values to ensure non-zero probability if VORP is > 0
        # Use slightly higher clip to avoid issues with very small numbers
        weights = candidate_final_df['vorp'].clip(lower=0.01).fillna(0.01).values

        chosen_player = None; log_msg = ""
        if np.sum(weights) <= 0.01 * len(weights): # If sum of weights is near zero (all candidates have ~0 VORP)
             # Pick the one with the highest *risk-adjusted* VORP in this case (index 0 as it's pre-sorted)
             chosen_player = candidate_final_df.iloc[0].to_dict(); log_msg = "picking highest RiskAdjVORP (all candidates ~0 VORP)"
             self.rl_logger.debug("ExecAgentAction: All candidates had near-zero VORP, selecting top by RiskAdjVORP.")
        elif len(candidate_final_df) == 1:
              chosen_player = candidate_final_df.iloc[0].to_dict(); log_msg = "picking only valid candidate"
              self.rl_logger.debug("ExecAgentAction: Only one valid candidate found.")
        else:
             # Normalize weights to probabilities
             probabilities = weights / np.sum(weights)
             # Clean probabilities due to potential floating point issues
             probabilities = np.nan_to_num(probabilities, nan=0.0) # Replace NaN with 0
             probabilities[probabilities < 0] = 0 # Ensure no negative probabilities
             if np.sum(probabilities) == 0: # If all probabilities became 0 somehow
                  self.rl_logger.warning("ExecAgentAction: All probabilities became zero after cleaning. Assigning equal probability.")
                  probabilities = np.ones(len(candidate_final_df)) / len(candidate_final_df)
             else:
                  probabilities /= np.sum(probabilities) # Re-normalize

             try:
                 chosen_index = np.random.choice(len(candidate_final_df), p=probabilities)
                 chosen_player = candidate_final_df.iloc[chosen_index].to_dict(); log_msg = f"prob VORP choice from {len(candidate_final_df)}"
                 self.rl_logger.debug(f"ExecAgentAction: Chose index {chosen_index} with probabilities {np.round(probabilities, 3)}")
             except ValueError as e:
                  # This can happen if probabilities don't sum exactly to 1 due to float precision
                  self.rl_logger.error(f"ExecAgentAction: Prob choice ValueError: {e}. Prob sum: {np.sum(probabilities)}. Weights: {weights}. Probs: {probabilities}. Picking highest RiskAdjVORP.")
                  # Fallback to highest risk-adjusted VORP candidate
                  chosen_player = candidate_final_df.iloc[0].to_dict(); log_msg = "picking highest RiskAdjVORP (prob error)"

        self.rl_logger.debug(f"Agent action '{action_desc}' -> {log_msg}. Picked: {chosen_player['name']} ({chosen_player['position']}) VORP: {chosen_player['vorp']:.2f} RiskAdjVORP: {chosen_player['risk_adjusted_vorp']:.2f}")
        self.rl_logger.debug(f"ExecAgentAction finished (took {time.time() - exec_start_time:.3f}s)")
        return chosen_player

    # --- Opponent Logic ---
    def _pick_opponent_smarter(self, for_team_id, avoid_id=None):
        """Opponent strategy: Consider need and value, then pick probabilistically."""
        opp_pick_start_time = time.time()
        self.rl_logger.debug(f"PickOpponent: Start for Team {for_team_id}.")

        current_available = self.available_players_df # Already sorted by risk_adj_vorp
        if current_available.empty: self.rl_logger.warning(f"PickOpponent: No players available for Team {for_team_id}."); return None
        if avoid_id is not None:
             current_available = current_available[current_available['player_id'] != avoid_id]
             if current_available.empty: self.rl_logger.warning(f"PickOpponent: No players available after avoiding {avoid_id}."); return None

        opp_roster = self.teams_rosters.get(for_team_id, [])
        opp_pos_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
        for player_dict in opp_roster:
            pos = player_dict.get('position')
            if pos and pos in opp_pos_counts:
                opp_pos_counts[pos] += 1
        self.rl_logger.debug(f"PickOpponent: Team {for_team_id} counts: {opp_pos_counts}")


        # Needs Calc
        needs = {}; req_starters = {'QB': self.starter_slots.get('QB', 1), 'RB': self.starter_slots.get('RB', 2), 'WR': self.starter_slots.get('WR', 2), 'TE': self.starter_slots.get('TE', 1)}
        for pos_key, req in req_starters.items(): needs[pos_key] = max(0, req - opp_pos_counts.get(pos_key, 0))
        # Calculate total starters including FLEX/OP
        total_starters_count = sum(v for k, v in self.starter_slots.items() if k in ['QB','RB','WR','TE','OP','RB/WR/TE'])
        starters_filled_count = sum(opp_pos_counts.get(p, 0) for p in req_starters) # Count only base starters filled
        num_flex_op_needed = max(0, total_starters_count - starters_filled_count)
        self.rl_logger.debug(f"PickOpponent: Team {for_team_id} Needs: {needs}, Flex/OP Needed: {num_flex_op_needed}")


        # Consider Top K players based on COMBINED_SCORE (more conventional ADP proxy)
        consider_k_initial = OPPONENT_CONSIDERATION_K * 3 # Look deeper initially
        # Use combined_score sort for initial opponent pool
        top_k_overall = current_available.sort_values('combined_score', ascending=False).head(consider_k_initial).copy()
        if top_k_overall.empty:
             top_k_overall = current_available.sort_values('combined_score', ascending=False).head(1).copy(); # Fallback to absolute best
        if top_k_overall.empty: self.rl_logger.warning(f"PickOpponent: No candidates even in fallback for Team {for_team_id}."); return None
        self.rl_logger.debug(f"PickOpponent: Initial pool size based on combined_score: {len(top_k_overall)}")


        # Apply Need Bonus to combined_score
        top_k_overall['adjusted_score'] = top_k_overall['combined_score']
        temp_needs = needs.copy(); temp_flex_op_filled = 0 # Use temp counters for bonus logic
        for index, player in top_k_overall.iterrows():
            player_pos = player['position']; base_score = player['combined_score']; bonus = 1.0; reason="BPA"
            is_flex = player_pos in ['RB','WR','TE']; is_op = player_pos in ['QB','RB','WR','TE'] and self.has_op_slot

            # Check if player fills a required starter need
            if player_pos in temp_needs and temp_needs[player_pos] > 0:
                 bonus = OPPONENT_NEED_BONUS_FACTOR
                 temp_needs[player_pos] -= 1 # Decrement need temporarily for scoring logic
                 reason = f"Starter Need ({player_pos})"
            # Check if player fills a FLEX/OP need
            elif (num_flex_op_needed - temp_flex_op_filled) > 0:
                 if is_op and player_pos == 'QB': # Prioritize QB for OP slightly
                      bonus = 1.0+(OPPONENT_NEED_BONUS_FACTOR-1.0)*0.7; temp_flex_op_filled += 1; reason = "OP Need (QB)"
                 elif is_op: # Other OP eligible
                      bonus = 1.0+(OPPONENT_NEED_BONUS_FACTOR-1.0)*0.6; temp_flex_op_filled += 1; reason = f"OP Need ({player_pos})"
                 elif is_flex: # Flex eligible (and not OP or OP not needed)
                      bonus = 1.0+(OPPONENT_NEED_BONUS_FACTOR-1.0)*0.5; temp_flex_op_filled += 1; reason = f"Flex Need ({player_pos})"

            adjusted_score = base_score * bonus
            top_k_overall.loc[index, 'adjusted_score'] = adjusted_score
            # Log bonus application for debugging
            # if bonus > 1.0: self.rl_logger.debug(f"PickOpponent: Bonus Applied - Player: {player['name']}, Pos: {player_pos}, Base: {base_score:.2f}, Bonus: {bonus:.2f}, Adj: {adjusted_score:.2f}, Reason: {reason}")


        # Sort by Adjusted Score and Filter by Limit, Limit to K Valid
        top_k_overall.sort_values('adjusted_score', ascending=False, inplace=True)
        self.rl_logger.debug(f"PickOpponent: Top 5 after need bonus:\n{top_k_overall[['name','position','combined_score','adjusted_score']].head().round(2).to_string(index=False)}")

        valid_candidates = []
        considered_count = 0
        for _, player in top_k_overall.iterrows():
             considered_count += 1
             player_pos_check = player['position']; fits_limit = True
             if player_pos_check in self.max_pos_counts:
                  if opp_pos_counts.get(player_pos_check, 0) >= self.max_pos_counts[player_pos_check]:
                       fits_limit = False
                       # self.rl_logger.debug(f"PickOpponent: Candidate {player.get('name','?')} ({player_pos_check}) skipped, limit reached.")
             # else: allow positions not tracked by limits (K, DEF etc.)

             if fits_limit:
                  valid_candidates.append(player.to_dict())
             # Stop once we have enough valid candidates
             if len(valid_candidates) >= OPPONENT_CONSIDERATION_K:
                  self.rl_logger.debug(f"PickOpponent: Reached consideration limit ({OPPONENT_CONSIDERATION_K}) for valid candidates.")
                  break
             # Safety break if we iterate too far without finding enough
             if considered_count >= len(top_k_overall) or considered_count >= OPPONENT_CONSIDERATION_K * 5:
                  self.rl_logger.debug(f"PickOpponent: Stopped searching after {considered_count} candidates.")
                  break


        # Probabilistic Choice / Fallback
        if not valid_candidates:
             self.rl_logger.debug(f"PickOpponent: Team {for_team_id} - No valid candidates found after applying need bonus & limits. Falling back to BPA (Combined Score) respecting limits.")
             # Fallback uses original combined_score sort from available players
             bpa_available = current_available.sort_values('combined_score', ascending=False)
             for _, player in bpa_available.iterrows():
                 fallback_pos = player['position']
                 fits_limit = True
                 if fallback_pos in self.max_pos_counts:
                      if opp_pos_counts.get(fallback_pos, 0) >= self.max_pos_counts[fallback_pos]:
                           fits_limit = False
                 if fits_limit:
                      self.rl_logger.debug(f"PickOpponent: Team {for_team_id} selecting BPA fallback: {player.get('name', '?')} ({fallback_pos}).")
                      return player.to_dict() # Return the first BPA that fits limits
             # If loop finishes, no player fits limits even in BPA fallback
             self.rl_logger.error(f"PickOpponent: Team {for_team_id} BPA fallback FAILED - no player fits limits among all available.")
             return None # Should be rare if players are available

        # Probabilistic choice among valid candidates based on projected points (simple value signal)
        self.rl_logger.debug(f"PickOpponent: Performing probabilistic choice for Team {for_team_id} among {len(valid_candidates)} candidates.")
        candidate_df = pd.DataFrame(valid_candidates)
        self.rl_logger.debug(f"PickOpponent: Candidate Pool for Prob Choice:\n{candidate_df[['name','position','projected_points','adjusted_score']].round(2).to_string(index=False)}")

        # Use projected_points for weighting the choice
        weights = candidate_df['projected_points'].clip(lower=0.1).fillna(0.1).values

        chosen_player = None; log_msg = ""
        if np.sum(weights) <= 0.1 * len(weights): # If sum is near zero
             # Pick the one with the highest adjusted score (which includes need)
             chosen_player = candidate_df.iloc[0].to_dict(); log_msg = "picking highest AdjScore (all candidates ~0 proj pts)"
             self.rl_logger.debug("PickOpponent: All candidates had near-zero projected points, selecting top by Adjusted Score.")
        elif len(candidate_df) == 1:
              chosen_player = candidate_df.iloc[0].to_dict(); log_msg = "picking only valid candidate"
              self.rl_logger.debug("PickOpponent: Only one valid candidate found.")
        else:
             # Normalize weights to probabilities
             probabilities = weights / np.sum(weights)
             # Clean probabilities
             probabilities = np.nan_to_num(probabilities, nan=0.0); probabilities[probabilities < 0] = 0
             if np.sum(probabilities) == 0: probabilities = np.ones(len(candidate_df)) / len(candidate_df); self.rl_logger.warning("Opponent probs zero, using uniform.")
             else: probabilities /= np.sum(probabilities) # Re-normalize

             try:
                 idx = np.random.choice(len(candidate_df), p=probabilities)
                 chosen_player = candidate_df.iloc[idx].to_dict(); log_msg = f"prob choice from {len(candidate_df)}"
                 self.rl_logger.debug(f"PickOpponent: Chose index {idx} with probabilities {np.round(probabilities, 3)}")
             except ValueError as e:
                  self.rl_logger.error(f"PickOpponent Team {for_team_id} prob choice ValueError: {e}. Prob sum: {np.sum(probabilities)}. Weights: {weights}. Probs: {probabilities}. Picking highest AdjScore.")
                  chosen_player = candidate_df.iloc[0].to_dict(); log_msg = "picking highest AdjScore (prob error)"

        self.rl_logger.debug(f"PickOpponent for Team {for_team_id} {log_msg}. Picked: {chosen_player['name']} ({chosen_player['position']}). Took {time.time() - opp_pick_start_time:.3f}s.")
        return chosen_player


    # --- State and Info Getters ---
    def _get_available_players(self):
        """Returns the current DataFrame of available players, sorted by Risk-Adjusted VORP."""
        # Assuming self.available_players_df is maintained correctly
        # Sorting is done once in init/reset and maintained by removing players,
        # but we can re-sort here just to be absolutely sure for the state calculation.
        if 'risk_adjusted_vorp' in self.available_players_df.columns:
            # Create a copy before returning to avoid modifying internal state unexpectedly elsewhere
            return self.available_players_df.sort_values('risk_adjusted_vorp', ascending=False).copy()
        else:
             self.rl_logger.error("INTERNAL ERROR: risk_adjusted_vorp column missing from available_players_df! Returning unsorted.")
             return self.available_players_df.copy()


    # def _get_state(self):
    #     """Constructs the state vector, using risk-adjusted VORP for player value signal."""
    #     get_state_start = time.time()
    #     self.rl_logger.debug(f"GetState: Start (Pick {self.current_pick_overall + 1})")
    #     try:
    #         state = np.zeros(self.observation_space.shape, dtype=np.float32)
    #         current_available = self._get_available_players() # Sorted by risk_adj_vorp
    #         self.rl_logger.debug(f"GetState: Available players = {len(current_available)}")

    #         # 1. Current Pick Normalization
    #         state[0] = min(1.0, self.current_pick_overall / max(1, self.total_picks -1))
    #         self.rl_logger.debug(f"GetState: state[0] (pick_norm) = {state[0]:.4f}")

    #         # 2. Top N Available per Position (count based on current ranking)
    #         idx = 1
    #         for pos in ['QB','RB','WR','TE']:
    #             count = 0
    #             if not current_available.empty and 'position' in current_available.columns:
    #                 # Count how many of this position are in the top overall players (e.g., top 50)
    #                 # This reflects scarcity at the top
    #                 top_overall_df = current_available.head(TOP_N_PLAYERS_STATE)
    #                 count = len(top_overall_df[top_overall_df['position'] == pos])
    #             # Normalize by total number considered for state, not absolute max players
    #             state[idx] = count / TOP_N_PLAYERS_STATE
    #             self.rl_logger.debug(f"GetState: state[{idx}] (topN_{pos}) = {state[idx]:.4f} ({count}/{TOP_N_PLAYERS_STATE})")
    #             idx += 1

    #         # 3. Agent Roster Counts Normalization
    #         agent_roster = self.teams_rosters.get(self.agent_team_id, [])
    #         counts = {'QB':0,'RB':0,'WR':0,'TE':0}
    #         for p in agent_roster:
    #             pos = p.get('position')
    #             if pos in counts: counts[pos]+=1
    #         for pos in ['QB','RB','WR','TE']:
    #             # Use the calculated max_pos_counts for normalization
    #             max_c = max(1, self.max_pos_counts.get(pos, 1)) # Ensure max_c is at least 1
    #             state[idx] = min(1.0, counts.get(pos, 0) / max_c) # Clip at 1.0
    #             self.rl_logger.debug(f"GetState: state[{idx}] (agent_{pos}) = {state[idx]:.4f} ({counts.get(pos, 0)}/{max_c})")
    #             idx += 1

    #         # 4. Recent Positional Runs Normalization
    #         recent = self.recent_picks_pos[-RECENT_PICKS_WINDOW:]
    #         runs = {'QB':0,'RB':0,'WR':0,'TE':0}
    #         for pos in recent:
    #             if pos in runs: runs[pos] += 1
    #         for pos in ['QB','RB','WR','TE']:
    #             state[idx] = runs.get(pos, 0) / max(1, RECENT_PICKS_WINDOW) # Normalize by window size
    #             self.rl_logger.debug(f"GetState: state[{idx}] (run_{pos}) = {state[idx]:.4f} ({runs.get(pos, 0)}/{RECENT_PICKS_WINDOW})")
    #             idx += 1

    #         # 5. Average Risk-Adjusted VORP of Top 5 Available (Normalized)
    #         top_5 = current_available.head(5)
    #         top_5_avg_value = 0.0
    #         if not top_5.empty and 'risk_adjusted_vorp' in top_5.columns:
    #              # Ensure calculation handles potential NaNs robustly
    #              top_5_avg_value = pd.to_numeric(top_5['risk_adjusted_vorp'], errors='coerce').mean()
    #              if pd.isna(top_5_avg_value): top_5_avg_value = 0.0 # Handle case where all top 5 are NaN
    #         # Use the pre-calculated max VORP for normalization, ensure > 0
    #         norm_max = max(1.0, self.max_risk_adjusted_vorp)
    #         state[idx] = top_5_avg_value / norm_max
    #         self.rl_logger.debug(f"GetState: state[{idx}] (avg_top5_riskadj_vorp_norm) = {state[idx]:.4f} ({top_5_avg_value:.2f}/{norm_max:.2f})")
    #         idx += 1 # Increment index tracking


    #         needs = {}
    #         for pos in ['QB', 'RB', 'WR', 'TE']:
    #             req = self.starter_slots.get(pos, 0) # Get required starters
    #             needs[pos] = max(0, req - counts.get(pos, 0))
    #             # Normalize need (e.g., by max starters for that pos, or just clip)
    #             state[idx] = min(1.0, needs[pos] / max(1, req)) if req > 0 else 0.0
    #             self.rl_logger.debug(f"GetState: state[{idx}] (need_{pos}) = {state[idx]:.4f} ({needs[pos]}/{req})")
    #             idx += 1
            
            
    #         # --- Final State Validation ---
    #         if idx != len(state):
    #              self.rl_logger.error(f"GetState: State construction mismatch! Expected size {len(state)}, filled {idx} elements.")
    #              # Attempt to pad remaining with zeros, but this indicates a bug
    #              state[idx:] = 0.0


    #         if np.isnan(state).any() or np.isinf(state).any():
    #             self.rl_logger.error(f"!!! State contains NaN/Inf before final clip/fix! State: {state}")
    #             state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)

    #         state = np.clip(state, 0.0, 1.0) # Ensure all values are within the Box bounds

    #         if state.shape != self.observation_space.shape:
    #              self.rl_logger.critical(f"FATAL STATE SHAPE MISMATCH AFTER CONSTRUCTION! Expected {self.observation_space.shape}, Got {state.shape}. Returning zeros.")
    #              state = np.zeros_like(self.observation_space.sample(), dtype=np.float32)

    #         self.rl_logger.debug(f"GetState finished (took {time.time() - get_state_start:.3f}s). Final State: {np.round(state, 3)}")
    #         return state

    #     except Exception as e:
    #          self.rl_logger.error(f"CRITICAL ERROR in _get_state: {e}", exc_info=True)
    #          # Return a valid zero state to prevent crashing SB3, but log the error
    #          return np.zeros(self.observation_space.shape, dtype=np.float32)


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


    # def _calculate_final_reward_vorp(self):
    #     """Calculates episodic reward based on VORP of starters + weighted VORP of bench."""
    #     reward_start_time = time.time()
    #     self.rl_logger.debug("Calculating final reward (VORP-based)...")
    #     agent_roster_list = self.teams_rosters.get(self.agent_team_id, [])
    #     if not agent_roster_list:
    #         self.rl_logger.warning("Agent roster empty during final reward calculation."); return 0.0
    #     try:
    #         agent_roster_df = pd.DataFrame(agent_roster_list)
    #         # Recalculate VORP here to be safe, using the final roster state
    #         if 'vorp' not in agent_roster_df.columns or agent_roster_df['vorp'].isnull().any():
    #              self.rl_logger.debug("Recalculating VORP for final reward.")
    #              agent_roster_df['vorp'] = agent_roster_df.apply(self._calculate_player_vorp, axis=1)

    #         if 'player_id' not in agent_roster_df.columns or 'position' not in agent_roster_df.columns:
    #              self.rl_logger.error("Reward Calc Error: Roster DF missing player_id or position.")
    #              raise ValueError("Missing ID/Pos in roster df for reward calc")

    #         agent_roster_df['vorp'] = pd.to_numeric(agent_roster_df['vorp'], errors='coerce').fillna(0.0)
    #         self.rl_logger.debug(f"Agent Roster for Reward Calc:\n{agent_roster_df[['name','position','vorp']].round(2).to_string(index=False)}")

    #     except Exception as e:
    #          self.rl_logger.error(f"Reward Calc DF Creation/VORP Error: {e}", exc_info=True); return 0.0

    #     # Determine Starters based on VORP within required slots
    #     agent_roster_df = agent_roster_df.sort_values('vorp', ascending=False)
    #     starters_vorp_sum = 0.0; players_used = set();
    #     # Define required counts clearly
    #     req_counts = {
    #          'QB': self.starter_slots.get('QB', 1),
    #          'RB': self.starter_slots.get('RB', 2),
    #          'WR': self.starter_slots.get('WR', 2),
    #          'TE': self.starter_slots.get('TE', 1),
    #          'OP': self.starter_slots.get('OP', 0),
    #          'FLEX': self.starter_slots.get('RB/WR/TE', 0)
    #     }
    #     self.rl_logger.debug(f"Reward Calc: Required Starter Counts: {req_counts}")

    #     # Fill required position slots first
    #     for pos, req_count in [('QB', req_counts['QB']), ('RB', req_counts['RB']), ('WR', req_counts['WR']), ('TE', req_counts['TE'])]:
    #         starters = agent_roster_df[(agent_roster_df['position'] == pos) & (~agent_roster_df['player_id'].isin(players_used))].head(req_count)
    #         count_filled = len(starters)
    #         if count_filled < req_count: self.rl_logger.warning(f"Reward Calc: Could only fill {count_filled}/{req_count} required {pos} slots.")
    #         starters_vorp_sum += starters['vorp'].sum()
    #         players_used.update(starters['player_id'])
    #         self.rl_logger.debug(f"Reward Calc: Filled {pos} slots. VORP added: {starters['vorp'].sum():.2f}. Players used: {len(players_used)}")


    #     # Fill OP slots (QB > RB > WR > TE based on VORP)
    #     if req_counts['OP'] > 0:
    #         op_eligible_pos = ['QB','RB','WR','TE']
    #         op_starters = agent_roster_df[agent_roster_df['position'].isin(op_eligible_pos) & (~agent_roster_df['player_id'].isin(players_used))].head(req_counts['OP'])
    #         count_filled = len(op_starters)
    #         if count_filled < req_counts['OP']: self.rl_logger.warning(f"Reward Calc: Could only fill {count_filled}/{req_counts['OP']} OP slots.")
    #         starters_vorp_sum += op_starters['vorp'].sum()
    #         players_used.update(op_starters['player_id'])
    #         self.rl_logger.debug(f"Reward Calc: Filled OP slots. VORP added: {op_starters['vorp'].sum():.2f}. Players used: {len(players_used)}")

    #     # Fill FLEX slots (RB > WR > TE based on VORP)
    #     if req_counts['FLEX'] > 0:
    #         flex_eligible_pos = ['RB','WR','TE']
    #         flex_starters = agent_roster_df[agent_roster_df['position'].isin(flex_eligible_pos) & (~agent_roster_df['player_id'].isin(players_used))].head(req_counts['FLEX'])
    #         count_filled = len(flex_starters)
    #         if count_filled < req_counts['FLEX']: self.rl_logger.warning(f"Reward Calc: Could only fill {count_filled}/{req_counts['FLEX']} FLEX slots.")
    #         starters_vorp_sum += flex_starters['vorp'].sum()
    #         players_used.update(flex_starters['player_id'])
    #         self.rl_logger.debug(f"Reward Calc: Filled FLEX slots. VORP added: {flex_starters['vorp'].sum():.2f}. Players used: {len(players_used)}")

    #     # Calculate Bench VORP
    #     bench_players = agent_roster_df[~agent_roster_df['player_id'].isin(players_used)]
    #     bench_vorp_sum = bench_players['vorp'].sum()

    #     # Combine starter and weighted bench VORP
    #     bench_weight = 0.15 # Weight bench VORP less
    #     final_reward_value = starters_vorp_sum + bench_weight * bench_vorp_sum

    #     self.rl_logger.info(f"Reward Calc (VORP): Starters Sum={starters_vorp_sum:.2f} ({len(players_used)} players), Bench Sum={bench_vorp_sum:.2f} ({len(bench_players)} players), Final Reward={final_reward_value:.4f}")
    #     if not np.isfinite(final_reward_value):
    #          self.rl_logger.error(f"Non-finite VORP reward calculated: {final_reward_value}. Returning 0.0.")
    #          return 0.0

    #     self.rl_logger.debug(f"Reward calculation finished (took {time.time() - reward_start_time:.3f}s)")
    #     return final_reward_value

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
        # ** TUNABLE PARAMETER **: How much penalty per unfilled required starter slot?
        # Should be significant relative to average VORP scores (e.g., if avg VORP is 5, penalty of 5-10 might be good)
        penalty_per_slot = 10.0
        # Only penalize for core required positions (QB, RB, WR, TE)
        required_core_starters = {'QB': req_counts['QB'], 'RB': req_counts['RB'], 'WR': req_counts['WR'], 'TE': req_counts['TE']}

        self.rl_logger.debug(f"Reward Calc: Checking penalties (penalty_per_slot={penalty_per_slot}). Filled: {filled_counts}")
        for pos, required in required_core_starters.items():
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
            # Calculate round/pick based on the *next* pick number if logging before incrementing,
            # or the current number if logging *after* incrementing (like in _simulate_opponents).
            # Let's standardize on logging the pick *just made*.
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
        # No explicit resources to close in this version (like open files or connections)
        pass


