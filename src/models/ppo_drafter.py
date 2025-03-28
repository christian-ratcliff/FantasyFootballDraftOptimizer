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
                 projection_models=None, use_top_n_features=0,
                 all_teams=None, starter_limits=None):
        """
        Initialize draft state with opponent modeling
        
        Parameters:
        -----------
        team : Team
            The team making the current pick
        available_players : list
            List of available players
        round_num : int
            Current draft round
        overall_pick : int
            Current overall pick number
        league_size : int
            Number of teams in the league
        roster_limits : dict
            Dictionary of roster position limits
        max_rounds : int, optional
            Maximum number of rounds in the draft
        projection_models : dict, optional
            Dictionary of projection models by position
        use_top_n_features : int, optional
            Number of top features to use from projection models
        all_teams : list, optional
            List of all teams in the draft (for opponent modeling)
        starter_limits : dict, optional
            Dictionary of starter position limits
        """
        self.team = team
        self.available_players = available_players
        self.round_num = round_num
        self.overall_pick = overall_pick
        self.league_size = league_size
        self.roster_limits = roster_limits
        self.max_rounds = max_rounds
        self.use_top_n_features = use_top_n_features
        self.all_teams = all_teams if all_teams is not None else []
        self.starter_limits = starter_limits or {}
        
        # Validate and fix roster limits if needed
        self._validate_roster_limits()
        
        # Define mappings for different roster slots to actual player positions
        self.slot_to_position_map = {
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
            "BE": ["QB", "RB", "WR", "TE", "K", "DST", "DL", "LB", "DB", "DE", "DT", "CB", "S"],
            "IR": ["QB", "RB", "WR", "TE", "K", "DST", "DL", "LB", "DB", "DE", "DT", "CB", "S"]
        }
        
        # Get valid roster slots the team can still fill
        self.valid_positions = [pos for pos in self.roster_limits.keys() 
                               if team.can_draft_position(pos)]
        
        # Determine valid player positions based on available roster slots
        valid_player_positions = set()
        for slot in self.valid_positions:
            if slot in self.slot_to_position_map:
                valid_player_positions.update(self.slot_to_position_map[slot])
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
        
        # --- New opponent modeling attributes ---
        
        # Calculate picks until next turn and next two turns
        self.picks_until_next_turn = self._calculate_picks_until_next_turn()
        self.picks_until_second_turn = self._calculate_picks_until_turn(2)
        
        # Pre-calculate opponent needs and targets
        self.opponent_needs = self._calculate_opponent_needs()
        self.opponent_targets_by_position = self._identify_opponent_targets()
        
        # Detect position scarcity and runs
        self.position_scarcity = self._calculate_position_scarcity()
        self.position_runs = self._detect_position_runs()
        
        # Identify value cliffs
        self.value_cliffs = self._detect_value_cliffs()
        
    def _validate_roster_limits(self):
        """Validate and ensure consistent roster limits"""
        # If no starter limits provided, create reasonable defaults based on roster limits
        if not self.starter_limits:
            self.starter_limits = {
                "QB": 1,
                "RB": 2,
                "WR": 3,
                "TE": 1,
                "K": 1 if "K" in self.roster_limits else 0,
                "DST": 1 if "DST" in self.roster_limits else 0,
                "FLEX": 2
            }
            
            # Check if any flex-type positions exist in roster_limits
            for flex_pos in ["FLEX", "RB/WR", "WR/TE", "RB/WR/TE", "OP"]:
                if flex_pos in self.roster_limits:
                    self.starter_limits[flex_pos] = min(2, self.roster_limits.get(flex_pos, 0))
        
    def _calculate_picks_until_next_turn(self):
        """
        Calculate the number of picks until this team picks again
        
        Returns:
        --------
        int
            Number of picks until next turn
        """
        # Determine the current pick pattern (ascending or descending)
        is_ascending = (self.round_num % 2 == 1)
        
        current_position = (self.overall_pick - 1) % self.league_size
        if is_ascending:
            current_position = current_position  # 0-indexed
        else:
            current_position = self.league_size - 1 - current_position  # Reverse for even rounds
        
        team_position = self.team.draft_position - 1  # Convert to 0-indexed
        
        if current_position == team_position:
            # If it's currently this team's pick, calculate to next pick
            if is_ascending:
                # If ascending, go to the end and back to this position in descending order
                picks = 2 * (self.league_size - 1)
            else:
                # If descending, go to the start and back to this position in ascending order
                picks = 2 * team_position
            return picks
        else:
            # If it's not this team's pick, calculate picks until this team's turn
            if is_ascending:
                if team_position > current_position:
                    # Team is later in this round
                    picks = team_position - current_position
                else:
                    # Team is earlier, need to go to end and back
                    picks = (self.league_size - current_position) + (self.league_size - 1 - team_position)
            else:
                if team_position < current_position:
                    # Team is earlier in this round (when looking in reverse)
                    picks = current_position - team_position
                else:
                    # Team is later, need to go to start and back
                    picks = current_position + 1 + team_position
            return picks
    
    def _calculate_picks_until_turn(self, turn_count):
        """
        Calculate picks until a specific number of turns ahead
        
        Parameters:
        -----------
        turn_count : int
            Number of turns ahead to calculate
        
        Returns:
        --------
        int
            Number of picks until that turn
        """
        if turn_count <= 0:
            return 0
        
        # For first turn, use the specialized method
        if turn_count == 1:
            return self._calculate_picks_until_next_turn()
        
        # For multiple turns, we need to calculate based on snake draft pattern
        team_position = self.team.draft_position - 1  # Convert to 0-indexed
        current_round = self.round_num
        current_pick_in_round = (self.overall_pick - 1) % self.league_size
        
        # Initialize count with first turn
        total_picks = self._calculate_picks_until_next_turn()
        
        # Add picks for subsequent turns
        for i in range(1, turn_count):
            # In snake draft, teams pick twice every 2*league_size picks
            total_picks += 2 * self.league_size
        
        return total_picks
    
    def _calculate_opponent_needs(self):
        """
        Calculate the positional needs of each opponent team
        
        Returns:
        --------
        dict
            Dictionary mapping team names to their positional needs
        """
        if not self.all_teams:
            return {}
        
        opponent_needs = {}
        
        for team in self.all_teams:
            # Skip our own team
            if team == self.team:
                continue
            
            # Get position needs for this team
            team_needs = {}
            
            # First check starter needs
            starters_by_pos = self._calculate_starter_needs(team)
            
            # Get overall position limits
            for position, limit in self.roster_limits.items():
                # Skip bench and IR positions for this analysis
                if position in ["BE", "IR", ""]:
                    continue
                
                current_count = len(team.roster_by_position.get(position, []))
                remaining = max(0, limit - current_count)
                
                # Check if this is a standard position or flex position
                if position in ["QB", "RB", "WR", "TE", "K", "DST"]:
                    # Calculate starter need urgency
                    starter_need = starters_by_pos.get(position, 0)
                    
                    # For standard positions, calculate need level including starter urgency
                    if remaining > 0:
                        # If there's still starter needs, this is more urgent
                        if starter_need > 0:
                            urgency = 2.0  # High urgency - missing starter
                        else:
                            urgency = 0.5  # Low urgency - just depth
                        
                        team_needs[position] = {
                            'remaining': remaining,
                            'starter_need': starter_need,
                            'urgency': urgency
                        }
                else:
                    # For flex positions, determine eligible positions
                    if position in self.slot_to_position_map:
                        eligible_positions = self.slot_to_position_map[position]
                        
                        # Calculate average need across eligible positions
                        if eligible_positions and remaining > 0:
                            team_needs[position] = {
                                'remaining': remaining,
                                'eligible_positions': eligible_positions,
                                'urgency': 1.0  # Medium urgency for flex positions
                            }
            
            opponent_needs[team.name] = team_needs
        
        return opponent_needs
    
    def _calculate_starter_needs(self, team):
        """
        Calculate starter needs for a team
        
        Parameters:
        -----------
        team : Team
            Team to analyze
            
        Returns:
        --------
        dict
            Dictionary of starter needs by position
        """
        starter_needs = {}
        
        # Get current roster by position
        roster_by_pos = team.roster_by_position
        
        # Calculate needs for standard positions
        for position, limit in self.starter_limits.items():
            # Skip bench and IR positions
            if position in ["BE", "IR", ""]:
                continue
                
            current_count = len(roster_by_pos.get(position, []))
            if position in ["QB", "RB", "WR", "TE", "K", "DST"]:
                # Standard position - direct comparison
                starter_needs[position] = max(0, limit - current_count)
            elif position in self.slot_to_position_map:
                # Flex position - need to check qualified players
                eligible_positions = self.slot_to_position_map[position]
                
                # Count players eligible for this flex position
                eligible_count = sum(len(roster_by_pos.get(pos, [])) for pos in eligible_positions)
                
                # Count players already starting in their standard positions
                allocated_count = sum(min(len(roster_by_pos.get(pos, [])), 
                                         self.starter_limits.get(pos, 0)) 
                                     for pos in eligible_positions)
                
                # Available for flex = total eligible - already allocated to standard positions
                available_for_flex = max(0, eligible_count - allocated_count)
                
                # Remaining need for this flex position
                flex_need = max(0, limit - available_for_flex)
                
                # Distribute flex needs proportionally to eligible positions
                if flex_need > 0 and eligible_positions:
                    for pos in eligible_positions:
                        # Add fractional need to each position
                        weight = 1.0 / len(eligible_positions)
                        starter_needs[pos] = starter_needs.get(pos, 0) + (flex_need * weight)
        
        return starter_needs
    
    def _identify_opponent_targets(self):
        """
        Identify likely target players for each opponent in the next few picks
        
        Returns:
        --------
        dict
            Dictionary mapping positions to counts of opponents targeting that position
        """
        if not self.all_teams or not self.opponent_needs:
            return {}
        
        # How many picks ahead to consider
        picks_to_consider = min(self.league_size, 
                                self._calculate_picks_until_next_turn())
        
        # Get teams picking before our next turn
        picking_teams = self._get_teams_picking_next(picks_to_consider)
        
        # Count position targets
        position_targets = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "K": 0, "DST": 0}
        
        # Analyze each team's needs
        for team in picking_teams:
            if team == self.team:
                continue
                
            # Get this team's needs
            team_needs = self.opponent_needs.get(team.name, {})
            
            # Determine most likely position target
            max_urgency = 0
            target_position = None
            
            for position, need_info in team_needs.items():
                if position in ["QB", "RB", "WR", "TE", "K", "DST"]:
                    urgency = need_info.get('urgency', 0) * need_info.get('remaining', 0)
                    
                    if urgency > max_urgency:
                        max_urgency = urgency
                        target_position = position
                elif 'eligible_positions' in need_info:
                    # Distribute flex position needs
                    for pos in need_info['eligible_positions']:
                        # Only consider main positions
                        if pos in position_targets:
                            position_targets[pos] += (need_info.get('urgency', 0) / 
                                                    len(need_info['eligible_positions']))
            
            # Increment target count for the most needed position
            if target_position in position_targets:
                position_targets[target_position] += 1
        
        return position_targets
    
    def _get_teams_picking_next(self, pick_count):
        """
        Get the teams picking in the next N picks
        
        Parameters:
        -----------
        pick_count : int
            Number of picks to consider
            
        Returns:
        --------
        list
            List of teams picking in the specified range
        """
        if not self.all_teams:
            return []
        
        teams_picking = []
        current_round = self.round_num
        current_pick_in_round = (self.overall_pick - 1) % self.league_size
        current_overall_pick = self.overall_pick
        
        # Determine pick pattern for current round
        is_ascending = (current_round % 2 == 1)
        
        # Simulate the next picks
        for i in range(pick_count):
            next_pick = current_overall_pick + i
            next_round = (next_pick - 1) // self.league_size + 1
            next_pick_in_round = (next_pick - 1) % self.league_size
            
            # Determine if this round is ascending or descending
            next_is_ascending = (next_round % 2 == 1)
            
            # Calculate team index
            if next_is_ascending:
                team_idx = next_pick_in_round
            else:
                team_idx = self.league_size - 1 - next_pick_in_round
            
            # Get the team at this position
            for team in self.all_teams:
                if team.draft_position == team_idx + 1:  # Convert to 1-indexed
                    teams_picking.append(team)
                    break
        
        return teams_picking
    
    def _calculate_position_scarcity(self):
        """
        Calculate position scarcity metrics
        
        Returns:
        --------
        dict
            Dictionary containing scarcity metrics for each position
        """
        # Group available players by position
        players_by_pos = {}
        for player in self.available_players:
            if not player.is_drafted:
                pos = player.position
                if pos not in players_by_pos:
                    players_by_pos[pos] = []
                players_by_pos[pos].append(player)
        
        # Calculate scarcity metrics
        scarcity = {}
        for pos, players in players_by_pos.items():
            # Skip if no players at this position
            if not players:
                continue
            
            # Sort by projected points
            sorted_players = sorted(players, key=lambda p: p.projected_points, reverse=True)
            
            # Calculate metrics
            total_available = len(sorted_players)
            top_tier_count = min(5, total_available)  # Consider top 5 as top tier
            
            # Average value of top tier
            top_tier_avg = sum(p.projected_points for p in sorted_players[:top_tier_count]) / top_tier_count
            
            # Starter-quality players (above 75% of top player)
            top_value = sorted_players[0].projected_points if sorted_players else 0
            starter_threshold = 0.75 * top_value
            starter_quality_count = sum(1 for p in sorted_players if p.projected_points >= starter_threshold)
            
            # Calculate relative scarcity (fewer players = higher scarcity)
            league_starting_spots = self.starter_limits.get(pos, 0) * self.league_size
            if league_starting_spots > 0:
                relative_scarcity = starter_quality_count / league_starting_spots
            else:
                relative_scarcity = 1.0  # Default if no starter spots
            
            # Compute a scarcity score (higher = more scarce)
            scarcity_score = 1.0 - min(1.0, relative_scarcity)
            
            scarcity[pos] = {
                'total_available': total_available,
                'starter_quality_count': starter_quality_count,
                'top_tier_avg': top_tier_avg,
                'relative_scarcity': relative_scarcity,
                'scarcity_score': scarcity_score
            }
        
        return scarcity
    
    def _detect_position_runs(self):
        """
        Detect if there's been a recent run on a position
        
        Returns:
        --------
        dict
            Dictionary mapping positions to run metrics
        """
        if not self.all_teams:
            return {}
        
        # Number of recent picks to analyze
        recent_pick_count = min(self.league_size, self.overall_pick - 1)
        
        # Count positions drafted in recent picks
        position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "K": 0, "DST": 0}
        
        # Analyze recent picks if available
        for team in self.all_teams:
            # Only count teams that have already picked
            if team.draft_picks:
                # Look at most recent pick
                latest_round = max(team.draft_picks.keys())
                latest_pick = team.draft_picks[latest_round]
                
                # Check if this pick is recent enough
                if latest_round == self.round_num or (
                   latest_round == self.round_num - 1 and 
                   self.overall_pick <= self.league_size):
                    # Count this position
                    position = latest_pick.position
                    if position in position_counts:
                        position_counts[position] += 1
        
        # Calculate run metrics
        position_runs = {}
        for pos, count in position_counts.items():
            # Calculate run percentage (percentage of recent picks at this position)
            run_percentage = count / max(1, recent_pick_count)
            
            # Determine if there's a run (arbitrary threshold of 25%)
            is_run = run_percentage >= 0.25
            
            position_runs[pos] = {
                'recent_picks': count,
                'run_percentage': run_percentage,
                'is_run': is_run
            }
        
        return position_runs
    
    def _detect_value_cliffs(self):
        """
        Detect value cliffs in available players
        
        Returns:
        --------
        dict
            Dictionary mapping positions to value cliff information
        """
        # Group available players by position
        players_by_pos = {}
        for player in self.available_players:
            if not player.is_drafted:
                pos = player.position
                if pos not in players_by_pos:
                    players_by_pos[pos] = []
                players_by_pos[pos].append(player)
        
        # Calculate value cliffs
        value_cliffs = {}
        for pos, players in players_by_pos.items():
            # Skip if not enough players
            if len(players) < 3:
                continue
            
            # Sort by projected points
            sorted_players = sorted(players, key=lambda p: p.projected_points, reverse=True)
            
            # Look for significant drop-offs in value
            cliffs = []
            
            # Only check the top 15 players (or fewer if not available)
            check_count = min(15, len(sorted_players) - 1)
            
            for i in range(check_count):
                current_value = sorted_players[i].projected_points
                next_value = sorted_players[i + 1].projected_points
                
                # Calculate percentage drop
                drop_percentage = (current_value - next_value) / current_value if current_value > 0 else 0
                
                # Identify cliff if drop is significant (arbitrary threshold of 15%)
                if drop_percentage >= 0.15:
                    cliffs.append({
                        'position': i,
                        'current_value': current_value,
                        'next_value': next_value,
                        'drop_percentage': drop_percentage,
                        'player_current': sorted_players[i].name,
                        'player_next': sorted_players[i + 1].name
                    })
            
            # Store cliff information
            value_cliffs[pos] = {
                'has_cliff': len(cliffs) > 0,
                'cliffs': cliffs,
                # Store the position of the first significant cliff (if any)
                'first_cliff_position': cliffs[0]['position'] if cliffs else None,
                # Store the magnitude of the first cliff
                'first_cliff_magnitude': cliffs[0]['drop_percentage'] if cliffs else 0
            }
        
        return value_cliffs

    def to_feature_vector(self) -> np.ndarray:
        """
        Convert state to a feature vector for the RL model with enhanced opponent modeling
        
        Returns:
        --------
        np.ndarray
            Feature vector representing the state
        """
        # Basic draft state features
        draft_features = [
            self.round_num / self.max_rounds,  # Normalized round
            self.overall_pick / (self.league_size * self.max_rounds),  # Normalized pick
            self.draft_progress,  # Draft progress (0 to 1)
            self.team.draft_position / self.league_size,  # Normalized team position
            
            # Normalized picks until next turn
            self.picks_until_next_turn / (2 * self.league_size),
            
            # Normalized picks until second turn
            self.picks_until_second_turn / (3 * self.league_size),
        ]
        
        # Team composition features
        roster_features = []
        # Use all major positions including TE
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
        
        # --- Opponent modeling features ---
        opponent_features = []
        
        # Position target features (how many teams likely to draft each position)
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
            # Normalize by picks until next turn
            target_count = self.opponent_targets_by_position.get(position, 0)
            picks_until_turn = max(1, self.picks_until_next_turn)
            opponent_features.append(min(1.0, target_count / picks_until_turn))
        
        # Position scarcity features
        scarcity_features = []
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
            scarcity_info = self.position_scarcity.get(position, {})
            scarcity_score = scarcity_info.get('scarcity_score', 0.5)  # Default medium scarcity
            scarcity_features.append(scarcity_score)
        
        # Position run features - are teams going after a specific position?
        run_features = []
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
            run_info = self.position_runs.get(position, {})
            run_percentage = run_info.get('run_percentage', 0.0)
            run_features.append(run_percentage)
        
        # Value cliff features - is there a significant drop-off coming?
        cliff_features = []
        for position in ['QB', 'RB', 'WR', 'TE', 'K', 'DST']:
            cliff_info = self.value_cliffs.get(position, {})
            
            # Does position have a cliff?
            has_cliff = 1.0 if cliff_info.get('has_cliff', False) else 0.0
            cliff_features.append(has_cliff)
            
            # How soon is the cliff?
            first_cliff_pos = cliff_info.get('first_cliff_position', 10)  # Default to far away
            if first_cliff_pos == None:
                first_cliff_pos = 10
            normalized_cliff_pos = 1.0 - (min(first_cliff_pos, 10) / 10)  # Closer cliffs have higher values
            cliff_features.append(normalized_cliff_pos if has_cliff else 0.0)
            
            # How big is the cliff?
            cliff_magnitude = cliff_info.get('first_cliff_magnitude', 0.0)
            cliff_features.append(cliff_magnitude)
        
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
        feature_vector = (
            draft_features + 
            roster_features + 
            opponent_features + 
            scarcity_features + 
            run_features + 
            cliff_features + 
            pool_features
        )
        
        return np.array(feature_vector, dtype=np.float32)
   
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
    
    def __init__(self, state_dim, action_feature_dim=50, action_dim=256, 
                 lr_actor=0.0003, lr_critic=0.0003, gamma=0.99, 
                 gae_lambda=0.95, policy_clip=0.2, batch_size=32, 
                 n_epochs=10, entropy_coef=0.01, use_top_n_features=0, curriculum_enabled=True,
                 opponent_modeling_enabled=True):
        """
        Initialize the PPO drafter with opponent modeling
        
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
        self.curriculum_enabled = curriculum_enabled
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
        self.valid_roster_rate = 0.0
        self.position_requirements = {
            "QB": 1,  # Need at least 1 QB
            "RB": 3,  # Need at least 2 RBs
            "WR": 4,  # Need at least 2 WRs
            "TE": 1,  # Need at least 1 TE
        }
        self.curriculum_rewards_history = {
            1: [],
            2: [],
            3: [],
            4: []
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
        self.phase_stability_window = 20  # Episodes to consider for stability
        self.phase_performance_metrics = {phase: {} for phase in range(1, 5)}
        self.stuck_episodes = 0  # Counter for detecting if agent is stuck
        self.max_stuck_episodes = 100  # Maximum episodes before adjusting thresholds
        
        # Opponent modeling statistics
        self.opponent_model_stats = {
            "position_accuracy": [],  # Track accuracy of position predictions
            "run_detection_rate": [],  # Track detection of position runs
            "value_cliff_usage": []   # Track usage of value cliff detection
        }
        
        # Additional attributes for adaptive position weighting
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
    
    def _build_actor_network(self, learning_rate):
        """Build actor network with additional opponent modeling inputs"""
        # Input for state
        state_input = Input(shape=(self.state_dim,), name='state_input')
        
        # Input for action features
        action_features_input = Input(shape=(self.action_dim, self.action_feature_dim), 
                                    name='action_features_input')
        
        # Process state input - wider network for complex state representation
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
        """Build critic network with increased capacity for complex state"""
        state_input = Input(shape=(self.state_dim,))
        
        x = Dense(256, activation='relu')(state_input)  # Increased to 256 (from 128)
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
        Select an action (player) based on the current state with opponent modeling
        
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
        
        # Apply adaptive position weighting for players if appropriate
        if self.opponent_modeling_enabled and hasattr(self, 'position_priority_weights'):
            # Apply position priority weights during action selection
            for i, player in enumerate(valid_players):
                # Don't modify the actual player object, just adjust the selection probability
                # This will be done through modified action probabilities
                pass
        
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
            
            # Apply position priority weights if enabled
            if self.opponent_modeling_enabled and not training:
                # Apply different weights for different positions based on scarcity and needs
                for i, player in enumerate(valid_players):
                    if i < len(masked_probs):
                        position = player.position
                        
                        # Get the adaptive weight for this position
                        position_weight = self.position_priority_weights.get(position, 1.0)
                        
                        # Apply position weights for inference
                        masked_probs[i] *= position_weight
                        
                        # Apply scarcity and value cliff adjustments
                        if hasattr(state, 'position_scarcity') and position in state.position_scarcity:
                            scarcity_score = state.position_scarcity[position].get('scarcity_score', 0.5)
                            
                            # Higher scarcity = higher weight
                            scarcity_factor = 1.0 + (scarcity_score * 0.5)  # 1.0 to 1.5 range
                            masked_probs[i] *= scarcity_factor
                        
                        # Apply value cliff adjustments
                        if hasattr(state, 'value_cliffs') and position in state.value_cliffs:
                            cliff_info = state.value_cliffs[position]
                            
                            # If player is the last one before a cliff, increase priority
                            if cliff_info.get('has_cliff', False) and cliff_info.get('first_cliff_position', 10) == 0:
                                # This is the last player before a cliff
                                cliff_magnitude = cliff_info.get('first_cliff_magnitude', 0)
                                
                                # Higher magnitude cliff = higher weight
                                cliff_factor = 1.0 + (cliff_magnitude * 2.0)  # Can significantly boost priority
                                masked_probs[i] *= cliff_factor
            
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
        Train the RL model with opponent modeling enhancements
        
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
        logger.info(f"Training RL drafter for {num_episodes} episodes with opponent modeling")
        
        # Keep track of best model
        best_reward = -float('inf')
        best_weights = None
        episode_rewards = []  # Track all episode rewards
        
        # Training metrics tracking
        self.position_distributions = []  # Track positions drafted
        self.value_metrics = []  # Track value metrics
        self.rank_history = []  # Track rankings
        
        # New opponent modeling metrics
        self.opponent_predictions = []  # Track opponent pick predictions
        self.value_cliff_decisions = []  # Track value cliff-based decisions
        self.position_run_decisions = []  # Track position run-based decisions
        
        # Curriculum learning tracking
        if self.curriculum_enabled:
            logger.info(f"Using curriculum learning: starting at phase {self.curriculum_phase}")
            self.temperature_history = []
            if not hasattr(self, 'phase_transition_episodes'):
                self.phase_transition_episodes = []
        
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
        

        
        # Training loop
        for episode in range(1, num_episodes + 1):
            self.episodes = episode
            rl_draft_position = random.randint(1, league_size)  # Randomly select draft position
            logger.info(f"Episode {episode}/{num_episodes}")
            logger.info(f"Episode {episode}: RL team drafting from position {rl_draft_position}")
            
            # Update curriculum temperature if enabled
            if self.curriculum_enabled:
                # Calculate temperature (gradually increases during training)
                max_temp = 0.5  # Maximum temperature
                self.curriculum_temperature = min(max_temp, episode / 1000)  # Increases to max over 1000 episodes
                self.temperature_history.append(self.curriculum_temperature)
                
                logger.info(f"  Curriculum phase: {self.curriculum_phase}, Temperature: {self.curriculum_temperature:.2f}")
            
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
            
            # Reinitialize season simulator with fresh teams
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
            
            # Track opponent modeling decisions for this episode
            episode_opponent_predictions = []  # Track accuracy of opponent predictions
            episode_cliff_decisions = []       # Track value cliff decisions
            episode_run_decisions = []         # Track position run decisions
            
            # Current state
            current_round = 1
            current_pick = 1
            
            # For opponent modeling, predict the next few picks
            predicted_picks = {}  # {pick_number: predicted_position}
            
            # Get starter limits if available
            starter_limits = {}
            if (hasattr(draft_simulator, 'scoring_settings') and 
                draft_simulator.scoring_settings and 
                'starter_limits' in draft_simulator.scoring_settings):
                starter_limits = draft_simulator.scoring_settings['starter_limits']
            
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
                    
                    # Create enhanced state with opponent modeling
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
                        all_teams=draft_simulator.teams,  # Pass all teams for opponent modeling
                        starter_limits=starter_limits     # Pass starter limits
                    )
                    
                    # Check if we made a previous prediction for this pick
                    if current_pick in predicted_picks:
                        predicted_position = predicted_picks[current_pick]
                        actual_position = None  # We'll set this after selection
                        
                        # Track prediction to evaluate later
                        prediction_data = {
                            "pick": current_pick,
                            "predicted_position": predicted_position,
                            "actual_position": None  # Fill in after selection
                        }
                        episode_opponent_predictions.append(prediction_data)
                    
                    # Check for value cliffs
                    cliff_decision = False
                    if self.opponent_modeling_enabled and hasattr(state, 'value_cliffs'):
                        for position in ["QB", "RB", "WR", "TE"]:
                            if position in state.value_cliffs:
                                cliff_info = state.value_cliffs[position]
                                
                                if cliff_info.get('has_cliff', False) and cliff_info.get('first_cliff_position', 10) == 0:
                                    # Found a position with an immediate cliff
                                    cliff_decision = True
                                    logger.info(f"Value cliff detected for {position}, considering in pick decision")
                                    
                                    # Record the cliff decision
                                    episode_cliff_decisions.append({
                                        "pick": current_pick,
                                        "position": position,
                                        "cliff_magnitude": cliff_info.get('first_cliff_magnitude', 0)
                                    })
                    
                    # Check for position runs
                    run_decision = False
                    if self.opponent_modeling_enabled and hasattr(state, 'position_runs'):
                        for position, run_info in state.position_runs.items():
                            if run_info.get('is_run', False):
                                # Found a position with an active run
                                run_decision = True
                                logger.info(f"Position run detected for {position}, considering in pick decision")
                                
                                # Record the run decision
                                episode_run_decisions.append({
                                    "pick": current_pick,
                                    "position": position,
                                    "run_percentage": run_info.get('run_percentage', 0)
                                })
                    
                    # Check if we need a TE and handle according to current curriculum phase
                    needs_te = not any(p.position == "TE" for p in team.roster)
                    te_draft_round = current_round >= 6 and current_round <= 12  # Good rounds to draft a TE
                    available_tes = [p for p in available_players if p.position == "TE" and not p.is_drafted]

                    if needs_te and te_draft_round and available_tes and random.random() < 0.35:  # 35% chance to force TE pick
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
                        
                        # Update any prediction data with actual position
                        for pred in episode_opponent_predictions:
                            if pred["pick"] == current_pick:
                                pred["actual_position"] = player.position
                                # Mark if prediction was correct
                                pred["correct"] = (pred["predicted_position"] == player.position)
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
                    # If opponent modeling is enabled and this is a team picking before our next turn,
                    # try to predict their pick for later validation
                    if self.opponent_modeling_enabled and rl_team is not None:
                        # Only predict if our team has future picks
                        rl_team_picked = any(t.strategy == "PPO" for t in draft_simulator.teams)
                        
                        if rl_team_picked:
                            # Create state to analyze opponent's needs
                            available_players = [p for p in draft_simulator.players if not p.is_drafted]
                            
                            # Create the full state with opponent modeling
                            state = DraftState(
                                team=team,  # The opponent team
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
                                # Make the prediction
                                predicted_picks[current_pick] = predicted_position
                    
                    # Make the pick using the team's strategy
                    picked_player = draft_simulator._make_pick(team, current_round, current_pick, ppo_model=self)
                    
                    # Validate opponent pick prediction if we made one
                    if current_pick in predicted_picks and picked_player:
                        predicted_position = predicted_picks[current_pick]
                        actual_position = picked_player.position
                        
                        # Record prediction accuracy
                        prediction_correct = (predicted_position == actual_position)
                        
                        # Log the prediction result
                        logger.info(f"Opponent pick prediction: Predicted {predicted_position}, Actual {actual_position}, Correct: {prediction_correct}")
                        
                        # Track prediction to evaluate later
                        prediction_data = {
                            "pick": current_pick,
                            "predicted_position": predicted_position,
                            "actual_position": actual_position,
                            "correct": prediction_correct
                        }
                        episode_opponent_predictions.append(prediction_data)
                    
                    if not picked_player:
                        logger.warning(f"Team {team.name} could not make a valid pick!")
                
                # Move to next pick
                current_pick += 1
                if current_pick > current_round * draft_simulator.league_size:
                    current_round += 1
                    
                if current_pick > draft_simulator.num_rounds * draft_simulator.league_size:
                    logger.info(f"Reached maximum picks ({current_pick-1}), ending draft")
                    break
            
            # After draft is complete, track opponent modeling metrics
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
            
            # Validate team positions after draft
            for team in draft_simulator.teams:
                self.validate_team_positions(team, roster_limits, 'after_draft')

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

            # Initialize position counts here
            position_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "K": 0, "DST": 0}
            vbd_sum = 0.0
            starter_points = 0.0
            total_points = 0.0
            roster_efficiency = 0.0
            
            # Count position distribution regardless of curriculum mode
            for player in rl_team.roster:
                if player.position in position_counts:
                    position_counts[player.position] += 1
                
                # Track VBD sum here too
                vbd_sum += max(0, getattr(player, 'vbd', 0))
            
            # Calculate starter and total points
            starter_points = rl_team.get_starting_lineup_points()
            total_points = rl_team.get_total_projected_points()
            roster_efficiency = starter_points / max(1, total_points)  # Avoid division by zero
            
            # Update adaptive position weights
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
                        
                        # When phase changes, log the thresholds for next phase
                        next_phase = self.curriculum_phase
                        adjusted_threshold = self.phase_thresholds.get(next_phase, 0) * (1.0 - self.curriculum_temperature)
                        logger.info(f"  Next phase threshold: {adjusted_threshold:.2f} (adjusted for temperature)")
                    
                else:
                    # Use original reward calculation if curriculum not enabled
                    reward = (
                        -3.0 * rl_metrics["rank"] +  # Lower rank is better (rank 1 = first place)
                        2.0 * rl_metrics["wins"] +  # More wins is better
                        0.01 * rl_metrics["points_for"] +  # More points is better
                        15.0 * (1 if rl_metrics.get("playoff_result") == "Champion" else 0) +
                        7.0 * (1 if rl_metrics.get("playoff_result") in ["Runner-up", "Third Place"] else 0) +
                        3.0 * (1 if rl_metrics.get("playoff_result") == "Playoff Qualification" else 0)
                    )

                    # Enhanced reward with draft quality components
                    draft_quality_reward = 0.0
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

                    # Roster balance reward
                    starter_points = rl_team.get_starting_lineup_points()
                    total_points = rl_team.get_total_projected_points()
                    # Higher ratio means more efficient roster construction
                    roster_efficiency = starter_points / max(1, total_points)
                    roster_balance_reward = roster_efficiency * 5.0  # Scale appropriately

                    # Position requirements penalty
                    position_requirements = {
                        "QB": 1,  # Need at least 1 QB
                        "RB": 2,  # Need at least 2 RBs
                        "WR": 2,  # Need at least 2 WRs
                        "TE": 1   # Need at least 1 TE
                    }

                    position_penalty = 0
                    for pos, requirement in position_requirements.items():
                        # Check if position requirements are met
                        if position_counts.get(pos, 0) < requirement:
                            # Heavy penalty for each missing required position
                            position_penalty += 10.0 * (requirement - position_counts.get(pos, 0))
                            logger.warning(f"Position requirement not met: {pos} - needed {requirement}, got {position_counts.get(pos, 0)}")

                    # Combined reward with position penalty
                    reward = reward + draft_quality_reward + roster_balance_reward - position_penalty

                    # Add a baseline adjustment so rewards aren't always negative
                    reward += 10.0

                    # Log the reward components
                    logger.info(f"  Reward components - Base: {reward:.2f}, Draft Quality: {draft_quality_reward:.2f}")
                    logger.info(f"  Balance: {roster_balance_reward:.2f}, Position Penalty: {position_penalty:.2f}")
                
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
                
                # Update training plots
                self.update_training_plots(save_path, episode)
                
                # Create curriculum learning visualization if enabled
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
        if best_weights is not None and hasattr(self.actor, 'set_weights'):
            self.actor.set_weights(best_weights)
            if hasattr(self, 'target_network') and self.target_network is not None:
                self.target_network.set_weights(best_weights)
            logger.info("Restored best model weights from training")
        
        # Save final model
        if save_path:
            self.save_model(os.path.join(save_path, f"ppo_drafter_final"), is_final=True)
            logger.info("Saved final model")
        
        # Return comprehensive training results
        return {
            "rewards_history": self.rewards_history,
            "win_rates": self.win_rates,
            "best_reward": best_reward,
            "final_epsilon": self.epsilon if hasattr(self, 'epsilon') else None,
            "episodes": self.episodes,
            "episode_rewards": episode_rewards,
            "position_distributions": self.position_distributions,
            "value_metrics": self.value_metrics,
            "rank_history": self.rank_history,
            
            # Curriculum learning specific results
            "curriculum_history": self.curriculum_rewards_history if self.curriculum_enabled else None,
            "final_curriculum_phase": self.curriculum_phase if self.curriculum_enabled else None,
            "temperature_history": self.temperature_history if hasattr(self, 'temperature_history') else None,
            "phase_transition_episodes": self.phase_transition_episodes if hasattr(self, 'phase_transition_episodes') else None,
            "phase_performance_metrics": self.phase_performance_metrics if hasattr(self, 'phase_performance_metrics') else None,
            
            # Opponent modeling metrics
            "opponent_predictions": self.opponent_predictions if self.opponent_modeling_enabled else None,
            "value_cliff_decisions": self.value_cliff_decisions if self.opponent_modeling_enabled else None,
            "position_run_decisions": self.position_run_decisions if self.opponent_modeling_enabled else None,
            "position_weights": self.position_priority_weights if self.opponent_modeling_enabled else None
        }
    
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
        if ppo_models_dir != None:
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
        # Create array of individual phase rewards
        phase_rewards = [phase1_reward, phase2_reward, phase3_reward, phase4_reward]
        
        # Debug log the raw phase rewards
        logger.info(f"Raw phase rewards before mixing: {[f'{r:.2f}' for r in phase_rewards]}")
        
        # Normalize rewards using running statistics before mixing
        normalized_rewards = []
        for phase, reward in enumerate(phase_rewards, 1):
            if len(self.curriculum_rewards_history[phase]) > 10:
                # Use running mean and std for normalization
                mean = self.reward_stats[phase]["mean"]
                std = max(1e-5, self.reward_stats[phase]["std"])  # Avoid division by zero
                norm_reward = (reward - mean) / std
                normalized_rewards.append(norm_reward)
                logger.info(f"Phase {phase} normalization: raw={reward:.2f}, mean={mean:.2f}, std={std:.2f}, normalized={norm_reward:.2f}")
            else:
                # Not enough data for normalization yet
                normalized_rewards.append(reward)
                logger.info(f"Phase {phase} not normalized yet (insufficient data)")
        
        # Apply reward mixing based on current phase
        mixed_reward = 0
        mix_details = []
        
        for i, weight in enumerate(self.reward_mix_weights[self.curriculum_phase]):
            component = normalized_rewards[i] * weight
            mixed_reward += component
            mix_details.append(f"Phase {i+1}: {normalized_rewards[i]:.2f} × {weight:.2f} = {component:.2f}")
        
        logger.info(f"Reward mixing: {', '.join(mix_details)}")
        
        # Add a smaller baseline adjustment (was 10.0 before)
        # This was likely the source of the constant 50.00 reward
        baseline_adjustment = 10.0
        mixed_reward += baseline_adjustment
        logger.info(f"Added baseline adjustment of {baseline_adjustment:.1f}")
        
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
        Update the curriculum phase based on progress and performance with adaptive criteria
        
        Returns:
        --------
        bool
            True if phase changed, False otherwise
        """
        if not self.curriculum_enabled:
            return False
        
        self.phase_episode_count += 1
        
        # Update curriculum temperature (gradually increases during training)
        # This helps the agent progress if it gets stuck in early phases
        max_temp = 0.5  # Maximum temperature
        self.curriculum_temperature = min(max_temp, 
                                        self.episodes / 1000)  # Increases to max over 1000 episodes
        
        # Adjust thresholds based on temperature
        adjusted_thresholds = {
            phase: threshold * (1.0 - self.curriculum_temperature) 
            for phase, threshold in self.phase_thresholds.items()
        }
        
        # Check if agent is stuck in current phase
        if self.phase_episode_count > self.phase_durations[self.curriculum_phase] * 0.5:
            self.stuck_episodes += 1
        else:
            self.stuck_episodes = 0
        
        # If agent is stuck too long, force advancement
        if self.stuck_episodes > self.max_stuck_episodes:
            old_phase = self.curriculum_phase
            self.curriculum_phase = min(4, self.curriculum_phase + 1)
            self.phase_episode_count = 0
            self.stuck_episodes = 0
            logger.info(f"Curriculum advanced to phase {self.curriculum_phase} after being stuck")
            self.phase_transition_episodes.append(self.episodes)
            return old_phase != self.curriculum_phase
        
        # Check if we should transition based on episode count
        if self.phase_episode_count >= self.phase_durations[self.curriculum_phase]:
            old_phase = self.curriculum_phase
            self.curriculum_phase = min(4, self.curriculum_phase + 1)
            self.phase_episode_count = 0
            logger.info(f"Curriculum advanced to phase {self.curriculum_phase} based on episode count")
            self.phase_transition_episodes.append(self.episodes)
            return old_phase != self.curriculum_phase
        
        # Adaptive transition criteria - look at reward trends and stability
        rewards = self.curriculum_rewards_history[self.curriculum_phase]
        if len(rewards) >= self.phase_stability_window:
            recent_rewards = rewards[-self.phase_stability_window:]
            
            # Check if recent rewards are stable (low variance) and above threshold
            reward_mean = np.mean(recent_rewards)
            reward_std = np.std(recent_rewards)
            reward_cv = reward_std / (abs(reward_mean) + 1e-5)  # Coefficient of variation
            
            # Phase 1: Check roster completion consistently above threshold
            if self.curriculum_phase == 1:
                performance_metric = self.valid_roster_rate
                threshold = adjusted_thresholds[1]
                
                if performance_metric >= threshold and reward_cv < 0.2:  # Low variation indicates stability
                    self.curriculum_phase = 2
                    self.phase_episode_count = 0
                    logger.info(f"Curriculum advanced to phase 2 based on stable roster completion rate: {performance_metric:.2f} >= {threshold:.2f}")
                    self.phase_transition_episodes.append(self.episodes)
                    return True
            
            # Phase 2: Check total points threshold
            elif self.curriculum_phase == 2:
                performance_metric = self.phase_performance_metrics[2].get("total_points", 0)
                threshold = adjusted_thresholds[2]
                
                if performance_metric >= threshold and reward_cv < 0.25:
                    self.curriculum_phase = 3
                    self.phase_episode_count = 0
                    logger.info(f"Curriculum advanced to phase 3 based on total points: {performance_metric:.2f} >= {threshold:.2f}")
                    self.phase_transition_episodes.append(self.episodes)
                    return True
            
            # Phase 3: Check starter points and roster efficiency
            elif self.curriculum_phase == 3:
                performance_metric = self.phase_performance_metrics[3].get("starter_points", 0)
                threshold = adjusted_thresholds[3]
                
                if performance_metric >= threshold and reward_cv < 0.3:
                    self.curriculum_phase = 4
                    self.phase_episode_count = 0
                    logger.info(f"Curriculum advanced to phase 4 based on starter points: {performance_metric:.2f} >= {threshold:.2f}")
                    self.phase_transition_episodes.append(self.episodes)
                    return True
        
        return False
    
    def plot_curriculum_progress(self, save_path=None):
        """
        Generate visualization of curriculum learning progress with adaptive features
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        """
        if not self.curriculum_enabled or not hasattr(self, 'curriculum_rewards_history'):
            logger.warning("Curriculum learning is not enabled or no history available")
            return
        
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
        
        # Performance metrics by phase
        if hasattr(self, 'phase_performance_metrics'):
            plt.subplot(2, 3, 4)
            metrics = ['valid_roster_rate', 'total_points', 'starter_points']
            phases = list(range(1, 5))
            
            for metric in metrics:
                values = [self.phase_performance_metrics.get(phase, {}).get(metric, 0) for phase in phases]
                plt.plot(phases, values, 'o-', label=metric)
            
            plt.title('Performance Metrics by Phase')
            plt.xlabel('Phase')
            plt.ylabel('Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        # Curriculum temperature
        plt.subplot(2, 3, 5)
        if hasattr(self, 'temperature_history'):
            plt.plot(self.temperature_history)
        else:
            plt.plot([self.curriculum_temperature])
        plt.title('Curriculum Temperature')
        plt.xlabel('Episode')
        plt.ylabel('Temperature')
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
            logger.info(f"Curriculum learning visualization saved to {save_path}")
        
        plt.close()
        
        
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
        
        # Calculate position performance score
        position_scores = {}
        
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
    
    def generate_opponent_modeling_visualizations(self, output_dir=None):
        """
        Generate visualizations of opponent modeling performance
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save visualizations
        """
        if not self.opponent_modeling_enabled or not hasattr(self, 'opponent_predictions'):
            logger.warning("Opponent modeling is not enabled or no prediction data available")
            return
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Plot prediction accuracy over time
        plt.figure(figsize=(12, 8))
        
        # Get accuracy data
        if self.opponent_predictions:
            episodes = [data["episode"] for data in self.opponent_predictions]
            accuracy = [data["accuracy"] for data in self.opponent_predictions]
            
            plt.plot(episodes, accuracy, 'o-')
            plt.axhline(y=0.25, color='r', linestyle='--', label='Random Guess Baseline (4 positions)')
            
            plt.title('Opponent Pick Prediction Accuracy')
            plt.xlabel('Episode')
            plt.ylabel('Accuracy')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3)
            plt.legend()
            
            if output_dir:
                plt.savefig(os.path.join(output_dir, 'opponent_prediction_accuracy.png'))
                logger.info(f"Opponent prediction visualization saved to {os.path.join(output_dir, 'opponent_prediction_accuracy.png')}")
        
        plt.close()
        
        # Plot position weight adaptation
        plt.figure(figsize=(12, 8))
        
        # Plot weights for core positions
        core_positions = ["QB", "RB", "WR", "TE"]
        for position in core_positions:
            if position in self.position_priority_weights:
                plt.axhline(y=self.position_priority_weights[position], 
                          color={'QB': 'r', 'RB': 'g', 'WR': 'b', 'TE': 'y'}[position], 
                          linestyle='-', linewidth=2, 
                          label=f'{position}: {self.position_priority_weights[position]:.2f}')
        
        plt.title('Position Priority Weights')
        plt.ylabel('Weight')
        plt.ylim(0, 2)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'position_priority_weights.png'))
            logger.info(f"Position weights visualization saved to {os.path.join(output_dir, 'position_priority_weights.png')}")
        
        plt.close()
        
        # Plot value cliff and position run usage
        plt.figure(figsize=(14, 10))
        
        # Set up subplots
        plt.subplot(2, 1, 1)
        
        # Value cliff decisions by position
        position_colors = {'QB': 'red', 'RB': 'green', 'WR': 'blue', 'TE': 'orange'}
        
        if hasattr(self, 'value_cliff_decisions') and self.value_cliff_decisions:
            cliff_positions = []
            cliff_magnitudes = []
            cliff_colors = []
            
            for episode_data in self.value_cliff_decisions:
                for decision in episode_data.get("decisions", []):
                    cliff_positions.append(decision["position"])
                    cliff_magnitudes.append(decision["cliff_magnitude"])
                    cliff_colors.append(position_colors.get(decision["position"], 'gray'))
            
            if cliff_positions:
                # Plot as scatter plot
                unique_positions = sorted(set(cliff_positions))
                position_indices = {pos: i for i, pos in enumerate(unique_positions)}
                
                x_vals = [position_indices[pos] for pos in cliff_positions]
                
                plt.scatter(x_vals, cliff_magnitudes, c=cliff_colors, alpha=0.7)
                plt.title('Value Cliff Decisions by Position')
                plt.xlabel('Position')
                plt.ylabel('Cliff Magnitude')
                plt.xticks(range(len(unique_positions)), unique_positions)
                plt.grid(True, alpha=0.3)
        
        # Position run decisions
        plt.subplot(2, 1, 2)
        
        if hasattr(self, 'position_run_decisions') and self.position_run_decisions:
            run_positions = []
            run_percentages = []
            run_colors = []
            
            for episode_data in self.position_run_decisions:
                for decision in episode_data.get("decisions", []):
                    run_positions.append(decision["position"])
                    run_percentages.append(decision["run_percentage"])
                    run_colors.append(position_colors.get(decision["position"], 'gray'))
            
            if run_positions:
                # Plot as scatter plot
                unique_positions = sorted(set(run_positions))
                position_indices = {pos: i for i, pos in enumerate(unique_positions)}
                
                x_vals = [position_indices[pos] for pos in run_positions]
                
                plt.scatter(x_vals, run_percentages, c=run_colors, alpha=0.7)
                plt.title('Position Run Decisions by Position')
                plt.xlabel('Position')
                plt.ylabel('Run Percentage')
                plt.xticks(range(len(unique_positions)), unique_positions)
                plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_dir:
            plt.savefig(os.path.join(output_dir, 'positional_decisions.png'))
            logger.info(f"Positional decisions visualization saved to {os.path.join(output_dir, 'positional_decisions.png')}")
        
        plt.close()



