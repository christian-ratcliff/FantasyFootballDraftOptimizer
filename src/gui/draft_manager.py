# src/gui/draft_manager.py
"""
Manages the state and logic of an interactive fantasy football draft,
integrating with a trained RL agent for recommendations.
"""

import logging
import pandas as pd
import numpy as np
import os
import joblib # To load the model
from stable_baselines3 import PPO # To load the model class

from src.pipeline.rl_pipeline import load_league_settings, load_projections_from_csvs
from src.models.rl_draft_agent import FantasyDraftEnv, NUM_ACTIONS, ACTION_BEST_AVAILABLE, AGENT_CONSIDERATION_K, AGENT_RISK_PENALTY_FACTOR, TOP_N_PLAYERS_STATE, RECENT_PICKS_WINDOW, VALUE_DROP_OFF_N, OPPONENT_NEED_BONUS_FACTOR

# --- Helper Functions , I need to move these to a util file, it is dupicated in the rl_draft_agetn ---
def calculate_replacement_levels(projections_df, league_settings):
    """Calculates VORP replacement levels."""
    levels = {}
    num_teams = league_settings.get('league_info', {}).get('team_count', 10)
    starter_slots = league_settings.get('starter_limits', {})
    num_starters_per_pos = {}

    req_qb = starter_slots.get('QB', 1); num_starters_per_pos['QB'] = req_qb
    req_rb = starter_slots.get('RB', 2); num_starters_per_pos['RB'] = req_rb
    req_wr = starter_slots.get('WR', 2); num_starters_per_pos['WR'] = req_wr
    req_te = starter_slots.get('TE', 1); num_starters_per_pos['TE'] = req_te
    req_op = starter_slots.get('OP', 0)
    req_flex = starter_slots.get('RB/WR/TE', 0)

    # Distribute OP/Flex needs (simplified proportional approach)
    if req_op > 0:
        op_weight_qb = 0.7; op_weight_other = (1.0 - op_weight_qb) / 3.0
        num_starters_per_pos['QB'] += req_op * op_weight_qb
        for pos in ['RB', 'WR', 'TE']: num_starters_per_pos[pos] += req_op * op_weight_other
    if req_flex > 0:
        flex_weight_rb = 0.4; flex_weight_wr = 0.5; flex_weight_te = 0.1
        num_starters_per_pos['RB'] += req_flex * flex_weight_rb
        num_starters_per_pos['WR'] += req_flex * flex_weight_wr
        num_starters_per_pos['TE'] += req_flex * flex_weight_te

    for pos, effective_starters in num_starters_per_pos.items():
        replacement_rank = int(np.ceil(num_teams * effective_starters)) + 1
        pos_projections = projections_df[projections_df['position'] == pos]['projected_points']
        pos_projections = pos_projections.sort_values(ascending=False).reset_index(drop=True)
        if not pos_projections.empty:
            level_idx = min(replacement_rank - 1, len(pos_projections) - 1)
            levels[pos] = pos_projections.iloc[level_idx]
        else:
            levels[pos] = 0.0
        levels[pos] = max(0.0, levels[pos])
    return levels

def calculate_vorp(player_row, replacement_levels):
    """Calculates VORP for a player row."""
    position = player_row.get('position')
    projection = pd.to_numeric(player_row.get('projected_points'), errors='coerce')
    if pd.isna(projection):
        projection = 0.0
    replacement_score = replacement_levels.get(position, 0.0)
    return max(0.0, projection - replacement_score)

ACTION_BEST_QB = 0
ACTION_BEST_RB = 1
ACTION_BEST_WR = 2
ACTION_BEST_TE = 3
ACTION_BEST_FLEX = 4
ACTION_BEST_AVAILABLE = 5

# --- DraftManager Class ---
class DraftManager:
    def __init__(self, projections_path, league_settings_path, model_path, agent_draft_pos):
        self.logger = logging.getLogger("fantasy_football") # Use main logger
        self.logger.info("Initializing DraftManager...")

        try:
            self.league_settings = load_league_settings(league_settings_path)
            # Ensure models_dir path is correct relative to how load_projections_from_csvs expects it
            # Assuming projections are directly in the path specified
            models_dir = projections_path # Rename argument for clarity if needed
            self.projections_master = load_projections_from_csvs(models_dir, self.league_settings)
            if self.projections_master.empty:
                raise ValueError("Failed to load projections.")
            self.logger.info(f"Loaded {len(self.projections_master)} players from projections.")

            # --- Calculate VORP and Risk-Adjusted VORP ---
            self.replacement_levels = calculate_replacement_levels(self.projections_master, self.league_settings)
            self.logger.debug(f"Calculated VORP Levels: {self.replacement_levels}")

            self.projections_master['vorp'] = self.projections_master.apply(lambda row: calculate_vorp(row, self.replacement_levels), axis=1)
            self.projections_master['projection_range'] = (self.projections_master['projection_high'] - self.projections_master['projection_low']).clip(lower=0.0).fillna(0.0)
            self.projections_master['risk_adjusted_vorp'] = (self.projections_master['vorp'] - (self.projections_master['projection_range'] * AGENT_RISK_PENALTY_FACTOR)).fillna(0.0)
            self.projections_master['risk_adjusted_vorp'] = np.maximum(self.projections_master['risk_adjusted_vorp'], self.projections_master['vorp'] * 0.1)
            self.projections_master.sort_values('risk_adjusted_vorp', ascending=False, inplace=True)
            self.max_risk_adjusted_vorp = self._get_max_norm_value('risk_adjusted_vorp')
            self.max_raw_vorp = self._get_max_norm_value('vorp')
            # ---------------------------------------------

            self.num_teams = self.league_settings['league_info']['team_count']
            self.starter_slots = self.league_settings.get('starter_limits', {})
            self.max_pos_counts = self._get_max_pos_counts(self.league_settings.get('roster_settings', {}))
            self.has_op_slot = self.starter_slots.get('OP', 0) > 0

            # Calculate draft length
            roster_settings = self.league_settings.get('roster_settings', {})
            bench_size = roster_settings.get('BE', 8) # Default 8 bench
            tracked_starters = ['QB', 'RB', 'WR', 'TE', 'OP', 'RB/WR/TE', 'K', 'D/ST'] # Include K/DST if present
            num_starters = sum(count for pos, count in self.starter_slots.items() if pos in tracked_starters)
            self.total_roster_size = num_starters + bench_size
            self.total_rounds = self.total_roster_size
            self.total_picks = self.num_teams * self.total_rounds
            self.logger.info(f"Draft settings: {self.num_teams} teams, {self.total_rounds} rounds, {self.total_picks} total picks.")

            # Generate draft order
            self.draft_order = self._generate_draft_order(self.league_settings.get('draft_settings', {}))

            # Determine Agent Team ID
            base_order = self.league_settings.get('draft_settings', {}).get('draft_order')
            if not base_order or len(base_order) != self.num_teams:
                base_order = list(range(1, self.num_teams + 1))
            if not (1 <= agent_draft_pos <= self.num_teams):
                agent_draft_pos = 1 # Default
            self.agent_team_id = base_order[agent_draft_pos - 1]
            self.logger.info(f"Agent Team ID: {self.agent_team_id} (Draft Position: {agent_draft_pos})")

            # --- Load RL Model ---
            self.model = PPO.load(model_path)
            self.logger.info(f"RL Model loaded successfully from {model_path}")

            # --- Initialize Draft State ---
            self.current_pick_overall = 0
            self.available_players_df = self.projections_master.copy()
            self.drafted_player_ids = set()
            self.team_ids = list(base_order) # Use actual IDs
            self.teams_rosters = {team_id: [] for team_id in self.team_ids}
            self.recent_picks_pos = [] # For state vector

            # Define observation space size based on the RL Env logic
            # Needs careful replication of the size calculation in FantasyDraftEnv
            self.state_size = 25 # Hardcoding based on last env version, RECALCULATE if env state changes
            self.logger.info(f"DraftManager initialized. State size = {self.state_size}")

        except Exception as e:
            self.logger.error(f"Error initializing DraftManager: {e}", exc_info=True)
            raise # Re-raise the error to be caught by the GUI

    def _get_max_pos_counts(self, roster_settings):
        """Determine max players per position from full roster settings."""
        defaults = {'QB': 3, 'RB': 8, 'WR': 8, 'TE': 3}
        counts = {}
        starter_keys = ['QB', 'RB', 'WR', 'TE']
        for pos in starter_keys:
            limit = roster_settings.get(pos, defaults.get(pos, 2)) # Use roster setting if available
            min_req = self.starter_slots.get(pos, 0)
            counts[pos] = max(min_req, limit if limit > 0 else (defaults.get(pos, 1) if min_req > 0 else 1))
        return counts

    def _generate_draft_order(self, draft_settings):
        """Generates the full draft order."""
        base_order = draft_settings.get('draft_order')
        if (not base_order or not isinstance(base_order, list) or
            len(base_order) != self.num_teams or len(set(base_order)) != self.num_teams):
            self.logger.warning("Using default draft order (1-N).")
            base_order = list(range(1, self.num_teams + 1))
        full_order = []
        for i in range(self.total_rounds):
            current_round_order = list(base_order) if (i + 1) % 2 == 1 else list(reversed(base_order))
            full_order.extend(current_round_order)
        if len(full_order) != self.total_picks:
             raise ValueError("Generated draft order length mismatch.")
        return full_order

    def get_current_status(self):
        """Returns current round, pick in round, and current team ID."""
        if self.is_draft_complete():
            return self.total_rounds, self.num_teams, "Draft Complete"

        round_num = (self.current_pick_overall // self.num_teams) + 1
        pick_in_round = (self.current_pick_overall % self.num_teams) + 1
        current_team_id = self.draft_order[self.current_pick_overall]
        return round_num, pick_in_round, current_team_id

    def is_agent_turn(self):
        """Checks if it's the agent's turn."""
        if self.is_draft_complete():
            return False
        return self.draft_order[self.current_pick_overall] == self.agent_team_id

    def is_draft_complete(self):
        """Checks if the draft has finished."""
        return self.current_pick_overall >= self.total_picks

    def get_available_players(self, limit=100):
        """Returns the top available players DataFrame."""
        # Filter master list based on drafted IDs
        available_mask = ~self.projections_master['player_id'].isin(self.drafted_player_ids)
        # Return a *copy* sorted by desired column (e.g., risk-adjusted VORP)
        return self.projections_master.loc[available_mask].sort_values('risk_adjusted_vorp', ascending=False).head(limit).copy()

    def get_agent_roster(self):
        """Returns the agent's current roster as a list of dicts."""
        return self.teams_rosters.get(self.agent_team_id, [])

    def find_player(self, player_query):
        """Finds a player by name (case-insensitive) or ID in the AVAILABLE list."""
        available = self.get_available_players(limit=None) # Search all available
        if available.empty:
            return None

        # Try exact ID match first
        try:
            player_id_query = int(player_query) # Assuming IDs are integers
            match = available[available['player_id'] == player_id_query]
            if not match.empty:
                return match.iloc[0].to_dict()
        except ValueError:
            pass # Query is not an integer ID

        # Try case-insensitive name match
        query_lower = player_query.lower()
        # Exact match
        match = available[available['name'].str.lower() == query_lower]
        if not match.empty:
            if len(match) > 1:
                self.logger.warning(f"Multiple exact name matches found for '{player_query}'. Returning first.")
            return match.iloc[0].to_dict()

        # Try partial match (contains) - be careful with this
        # match = available[available['name'].str.lower().str.contains(query_lower)]
        # if not match.empty:
        #     if len(match) == 1:
        #         return match.iloc[0].to_dict()
        #     else:
        #         self.logger.warning(f"Multiple partial name matches for '{player_query}'. Requires exact match.")
        #         return None # Ambiguous

        return None # No match found

    def make_opponent_pick(self, player_dict):
        """Records an opponent's pick and updates the state."""
        if self.is_draft_complete() or self.is_agent_turn():
            self.logger.error("Cannot make opponent pick: Draft complete or agent's turn.")
            return False

        player_id = player_dict.get('player_id')
        if not player_id:
            self.logger.error("Invalid player data for opponent pick (missing ID).")
            return False

        if player_id in self.drafted_player_ids:
            self.logger.error(f"Opponent pick error: Player {player_id} already drafted.")
            return False # Should be checked by GUI calling find_player first

        current_team_id = self.draft_order[self.current_pick_overall]
        player_pos = player_dict.get('position', 'UNK')

        self.drafted_player_ids.add(player_id)
        self.teams_rosters[current_team_id].append(player_dict)
        self.recent_picks_pos.append(player_pos)
        self.current_pick_overall += 1
        self.logger.info(f"Pick {self.current_pick_overall}: Team {current_team_id} drafted {player_dict.get('name', '?')} ({player_pos})")
        return True

    def make_agent_pick(self, player_dict):
        """Records the agent's pick (chosen by user) and updates the state."""
        if self.is_draft_complete() or not self.is_agent_turn():
            self.logger.error("Cannot make agent pick: Draft complete or not agent's turn.")
            return False

        player_id = player_dict.get('player_id')
        if not player_id:
            self.logger.error("Invalid player data for agent pick (missing ID).")
            return False

        if player_id in self.drafted_player_ids:
            self.logger.error(f"Agent pick error: Player {player_id} already drafted.")
            return False # Should be checked by GUI

        current_team_id = self.agent_team_id # It must be the agent's turn
        player_pos = player_dict.get('position', 'UNK')

        self.drafted_player_ids.add(player_id)
        self.teams_rosters[current_team_id].append(player_dict)
        self.recent_picks_pos.append(player_pos)
        self.current_pick_overall += 1
        self.logger.info(f"Pick {self.current_pick_overall}: Agent Team {current_team_id} drafted {player_dict.get('name', '?')} ({player_pos})")
        return True

    def get_agent_recommendations(self, top_n=10):
        """Gets recommendations from the RL model for the agent's turn."""
        if not self.is_agent_turn():
            return []

        state_vector = self._get_state_vector()
        if state_vector is None:
            self.logger.error("Failed to get state vector for agent recommendation.")
            return []

        try:
            # Predict the *action* (category of player)
            action, _ = self.model.predict(state_vector, deterministic=True)
            action = int(action) # Ensure it's an integer
            self.logger.info(f"Agent model predicted action: {action}")

            # Translate action into a ranked list of players
            # Reuse the logic from FantasyDraftEnv._execute_agent_action_probabilistic
            # but instead of picking one, return the ranked list of valid candidates.
            recommended_players = self._get_valid_candidates_for_action(action, top_n)
            return recommended_players

        except Exception as e:
            self.logger.error(f"Error during agent prediction: {e}", exc_info=True)
            return []

    def _get_max_norm_value(self, column_name):
        """Gets max value from a column for normalization."""
        max_val = 1.0
        if not self.projections_master.empty and column_name in self.projections_master.columns:
            max_val = self.projections_master[column_name].max()
        if pd.isna(max_val) or max_val <= 0:
             max_val = 15.0 # Fallback
             self.logger.warning(f"Invalid Max Value for {column_name} ({max_val}). Using fallback {max_val}")
        return max_val

    def _get_state_vector(self):
        """Constructs the state vector for the RL model."""
        # This needs to EXACTLY match the state construction in FantasyDraftEnv._get_state
        try:
            state = np.zeros(self.state_size, dtype=np.float32)
            current_available = self.get_available_players(limit=None) # Get all available, sorted by risk_adj_vorp
            idx = 0

            # --- Replicate State Construction from FantasyDraftEnv ---
            # 1. Pick Normalization
            state[idx] = min(1.0, self.current_pick_overall / max(1, self.total_picks - 1)); idx += 1

            # 2. Top N Available per Position %
            top_overall_df = current_available.head(TOP_N_PLAYERS_STATE)
            total_in_top_n = max(1, len(top_overall_df))
            for pos in ['QB', 'RB', 'WR', 'TE']:
                count = 0
                if not top_overall_df.empty and 'position' in top_overall_df.columns:
                    count = len(top_overall_df[top_overall_df['position'] == pos])
                state[idx] = count / total_in_top_n; idx += 1

            # 3. Agent Roster Counts Normalization
            agent_roster = self.teams_rosters.get(self.agent_team_id, [])
            counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
            for p in agent_roster:
                 pos = p.get('position');
                 if pos in counts: counts[pos]+=1
            for pos in ['QB', 'RB', 'WR', 'TE']:
                max_c = max(1, self.max_pos_counts.get(pos, 1))
                state[idx] = min(1.0, counts.get(pos, 0) / max_c); idx += 1

            # 4. Recent Positional Runs Normalization
            recent = self.recent_picks_pos[-RECENT_PICKS_WINDOW:]
            runs = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
            for pos in recent:
                 if pos in runs: runs[pos] += 1
            norm_recent = max(1, len(recent))
            for pos in ['QB', 'RB', 'WR', 'TE']:
                state[idx] = runs.get(pos, 0) / norm_recent; idx += 1

            # 5. Explicit Needs (Normalized)
            needs = {}
            for pos in ['QB', 'RB', 'WR', 'TE']:
                req = self.starter_slots.get(pos, 0)
                needs[pos] = max(0, req - counts.get(pos, 0))
                state[idx] = min(1.0, needs[pos] / max(1, req)) if req > 0 else 0.0
                idx += 1

            # 6. Rounds Remaining (Normalized)
            current_round = (self.current_pick_overall // self.num_teams) + 1
            rounds_remaining = max(0, self.total_rounds - current_round)
            state[idx] = rounds_remaining / max(1, self.total_rounds - 1); idx += 1

            # 7. VORP #1 Overall Available (Normalized by max RAW VORP)
            vorp1 = 0.0
            if not current_available.empty and 'vorp' in current_available.columns:
                top_raw_vorp_player = current_available.sort_values('vorp', ascending=False).iloc[0]
                vorp1 = top_raw_vorp_player['vorp']
            state[idx] = max(0.0, vorp1 / max(1.0, self.max_raw_vorp)); idx += 1

            # 8. VORP Drop-off #1 vs #N (Normalized)
            vorpN = 0.0; vorp_drop = 0.0
            if len(current_available) >= VALUE_DROP_OFF_N and 'vorp' in current_available.columns:
                sorted_by_vorp = current_available.sort_values('vorp', ascending=False)
                vorp1_drop = sorted_by_vorp.iloc[0]['vorp']
                vorpN = sorted_by_vorp.iloc[VALUE_DROP_OFF_N - 1]['vorp']
                vorp_drop = max(0.0, vorp1_drop - vorpN)
            state[idx] = vorp_drop / max(1.0, self.max_raw_vorp); idx += 1

            # 9. VORP #1 Available per Position (Normalized by max RAW VORP)
            for pos in ['QB', 'RB', 'WR', 'TE']:
                vorp1_pos = 0.0
                pos_available = current_available[current_available['position'] == pos]
                if not pos_available.empty and 'vorp' in pos_available.columns:
                    vorp1_pos = pos_available.sort_values('vorp', ascending=False).iloc[0]['vorp']
                state[idx] = max(0.0, vorp1_pos / max(1.0, self.max_raw_vorp)); idx += 1

            # 10. Average Risk-Adjusted VORP of Top 5 Available (Normalized by max RiskAdj VORP)
            top_5 = current_available.head(5)
            top_5_avg_value = 0.0
            if not top_5.empty and 'risk_adjusted_vorp' in top_5.columns:
                 top_5_numeric = pd.to_numeric(top_5['risk_adjusted_vorp'], errors='coerce')
                 top_5_avg_value = top_5_numeric.mean();
                 if pd.isna(top_5_avg_value): top_5_avg_value = 0.0
            norm_max_risk_adj = max(1.0, self.max_risk_adjusted_vorp)
            state[idx] = max(0.0, top_5_avg_value / norm_max_risk_adj); idx += 1
            # --- End State Replication ---

            if idx != self.state_size:
                 self.logger.error(f"State vector construction size mismatch! Expected {self.state_size}, Got {idx}. Padding.")
                 # Pad remaining state with zeros if size is incorrect
                 state[idx:] = 0.0
                 if len(state) != self.state_size: # Check padding result
                      state = np.resize(state, self.state_size) # Force resize if padding failed somehow

            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
            state = np.clip(state, 0.0, 1.0)
            return state

        except Exception as e:
            self.logger.error(f"Error creating state vector: {e}", exc_info=True)
            return None

    def _get_valid_candidates_for_action(self, action, top_n):
        """Gets the top N valid player candidates for a given agent action."""
        agent_roster = self.teams_rosters.get(self.agent_team_id, [])
        agent_pos_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
        for p in agent_roster:
             pos = p.get('position');
             if pos in agent_pos_counts: agent_pos_counts[pos] += 1

        current_available = self.get_available_players(limit=None) # Get all, sorted by risk_adj_vorp
        if current_available.empty:
            return []

        candidates_df = pd.DataFrame()
        action_desc = "BPA" # Default for logging

        # 1. Determine initial pool based on action
        if action == ACTION_BEST_QB: candidates_df = current_available[current_available['position'] == 'QB'].copy(); action_desc = "Best QB"
        elif action == ACTION_BEST_RB: candidates_df = current_available[current_available['position'] == 'RB'].copy(); action_desc = "Best RB"
        elif action == ACTION_BEST_WR: candidates_df = current_available[current_available['position'] == 'WR'].copy(); action_desc = "Best WR"
        elif action == ACTION_BEST_TE: candidates_df = current_available[current_available['position'] == 'TE'].copy(); action_desc = "Best TE"
        elif action == ACTION_BEST_FLEX:
            flex_pos = ['RB','WR','TE'];
            if self.has_op_slot: flex_pos.append('QB')
            candidates_df = current_available[current_available['position'].isin(flex_pos)].copy(); action_desc = "Best FLEX"
        else: # ACTION_BEST_AVAILABLE or fallback
            candidates_df = current_available.copy(); action_desc = "Best Available (BPA)"

        if candidates_df.empty:
            self.logger.debug(f"No initial candidates for action {action_desc}. Trying BPA.")
            # Fallback to BPA if specific action yields nothing
            if action != ACTION_BEST_AVAILABLE:
                return self._get_valid_candidates_for_action(ACTION_BEST_AVAILABLE, top_n)
            else:
                return [] # No players left even for BPA

        # 2. Filter by roster limits (iterate through sorted list)
        valid_candidates_list = []
        for _, player_row in candidates_df.iterrows(): # Already sorted by risk_adj_vorp
            pos = player_row['position']
            fits_limit = True
            if pos in self.max_pos_counts:
                 if agent_pos_counts.get(pos, 0) >= self.max_pos_counts[pos]:
                      fits_limit = False
            elif action in [ACTION_BEST_QB, ACTION_BEST_RB, ACTION_BEST_WR, ACTION_BEST_TE]:
                fits_limit = False # Don't pick K/DEF for specific pos action

            if fits_limit:
                 # Add relevant info for display
                 player_info = player_row.to_dict()
                 valid_candidates_list.append(player_info)
                 if len(valid_candidates_list) >= top_n:
                     break # Stop once we have enough

        if not valid_candidates_list and action != ACTION_BEST_AVAILABLE:
            # If original action yielded no *valid* candidates, try BPA
            self.logger.debug(f"No valid candidates fitting limits for {action_desc}. Trying BPA.")
            return self._get_valid_candidates_for_action(ACTION_BEST_AVAILABLE, top_n)

        # Sort the final valid list by risk_adjusted_vorp (should already be mostly sorted)
        valid_candidates_list.sort(key=lambda p: p.get('risk_adjusted_vorp', 0), reverse=True)

        return valid_candidates_list[:top_n] # Return top N valid players
    
    

    def get_state_snapshot(self):
        """Bundles the current draft state into a serializable dictionary."""
        self.logger.info("Generating draft state snapshot...")
        state = {
            'current_pick_overall': self.current_pick_overall,
            'drafted_player_ids': list(self.drafted_player_ids), # Convert set to list for JSON/Pickle
            'teams_rosters': self.teams_rosters,
            'recent_picks_pos': self.recent_picks_pos,
            'draft_order': self.draft_order, # Include for consistency check maybe
            'total_picks': self.total_picks,
            'agent_team_id': self.agent_team_id,
            # Add other essential config pieces if they might change or affect loading
            'league_settings_checksum': hash(str(self.league_settings)), # Basic check
            'projections_checksum': pd.util.hash_pandas_object(self.projections_master).sum() # Basic check
        }
        return state

    def load_state_snapshot(self, snapshot_data):
        """Restores the draft state from a loaded dictionary."""
        self.logger.info("Loading draft state from snapshot...")
        try:
            # --- Basic Validation (Optional but Recommended) ---
            required_keys = ['current_pick_overall', 'drafted_player_ids', 'teams_rosters', 'recent_picks_pos']
            if not all(key in snapshot_data for key in required_keys):
                raise ValueError("Snapshot data missing required keys.")

            # Optional: Compare checksums or key parameters to ensure compatibility
            if 'league_settings_checksum' in snapshot_data and \
               hash(str(self.league_settings)) != snapshot_data['league_settings_checksum']:
                self.logger.warning("Loaded state league settings might differ from current settings.")
            # Add similar check for projections if desired

            if 'total_picks' in snapshot_data and self.total_picks != snapshot_data['total_picks']:
                 self.logger.warning(f"Total picks mismatch (State: {snapshot_data['total_picks']}, Current: {self.total_picks}).")
                 # Decide how to handle this - error out or proceed with caution?

            # --- Restore State Variables ---
            self.current_pick_overall = snapshot_data['current_pick_overall']
            self.drafted_player_ids = set(snapshot_data['drafted_player_ids']) # Convert list back to set
            self.teams_rosters = snapshot_data['teams_rosters']
            self.recent_picks_pos = snapshot_data['recent_picks_pos']

            # Recalculate available players based on the loaded drafted set
            # The get_available_players method now handles this filtering dynamically
            self.logger.info(f"Draft state loaded. Resuming at pick {self.current_pick_overall + 1}.")
            return True

        except Exception as e:
            self.logger.error(f"Error loading draft state snapshot: {e}", exc_info=True)
            return False

    def get_team_roster(self, team_id):
        """Gets the roster for a specific team ID."""
        return self.teams_rosters.get(team_id, [])

    # Add this helper if you implement opponent prediction
    def get_opponent_needs_and_counts(self, team_id):
        """Calculates needs and counts for a specific opponent."""
        opp_roster = self.teams_rosters.get(team_id, [])
        opp_pos_counts = {'QB': 0, 'RB': 0, 'WR': 0, 'TE': 0}
        for player_dict in opp_roster:
            pos = player_dict.get('position')
            if pos and pos in opp_pos_counts:
                opp_pos_counts[pos] += 1

        needs = {}; req_starters = {'QB': self.starter_slots.get('QB', 1), 'RB': self.starter_slots.get('RB', 2), 'WR': self.starter_slots.get('WR', 2), 'TE': self.starter_slots.get('TE', 1)}
        for pos_key, req in req_starters.items():
             needs[pos_key] = max(0, req - opp_pos_counts.get(pos_key, 0))

        total_starters_count = sum(v for k, v in self.starter_slots.items() if k in ['QB','RB','WR','TE','OP','RB/WR/TE'])
        starters_filled_count = sum(opp_pos_counts.get(p, 0) for p in req_starters)
        num_flex_op_needed = max(0, total_starters_count - starters_filled_count)

        return needs, opp_pos_counts, num_flex_op_needed

    # --- Opponent Target Prediction (Placeholder - More Complex Logic Needed) ---
    def predict_opponent_targets(self, team_id, num_targets=5):
        """
        Predicts players the specified opponent might target next.
        (Basic Implementation - Needs Refinement)
        """
        self.logger.debug(f"Predicting targets for Team {team_id}...")
        if team_id == self.agent_team_id or self.is_draft_complete():
            return []

        needs, counts, flex_op_needed = self.get_opponent_needs_and_counts(team_id)
        available_now = self.get_available_players(limit=150) # Get a larger pool

        if available_now.empty:
            return []

        # Apply need bonus (simplified logic similar to _pick_opponent_smarter)
        candidates_df = available_now.copy()
        candidates_df['adjusted_score'] = candidates_df['combined_score'] # Use combined_score as base for opponents
        temp_needs = needs.copy()
        temp_flex_filled = 0

        for index, player in candidates_df.iterrows():
            player_pos = player['position']
            base_score = player['combined_score']
            bonus = 1.0
            is_flex = player_pos in ['RB','WR','TE']
            is_op = player_pos in ['QB','RB','WR','TE'] and self.has_op_slot

            if player_pos in temp_needs and temp_needs[player_pos] > 0:
                 bonus = OPPONENT_NEED_BONUS_FACTOR # Simplified constant name
                 temp_needs[player_pos] -= 1
            elif (flex_op_needed - temp_flex_filled) > 0:
                 if is_op and player_pos == 'QB': bonus = 1.4; temp_flex_filled += 1 # Simplified bonuses
                 elif is_op: bonus = 1.3; temp_flex_filled += 1
                 elif is_flex: bonus = 1.2; temp_flex_filled += 1

            candidates_df.loc[index, 'adjusted_score'] = base_score * bonus

        # Filter by limits and sort
        candidates_df.sort_values('adjusted_score', ascending=False, inplace=True)

        potential_targets = []
        for _, player_row in candidates_df.iterrows():
            pos = player_row['position']
            fits_limit = True
            if pos in self.max_pos_counts:
                 if counts.get(pos, 0) >= self.max_pos_counts[pos]:
                      fits_limit = False

            if fits_limit:
                 potential_targets.append(player_row.to_dict())
                 if len(potential_targets) >= num_targets:
                     break

        self.logger.debug(f"Predicted {len(potential_targets)} targets for Team {team_id}.")
        return potential_targets