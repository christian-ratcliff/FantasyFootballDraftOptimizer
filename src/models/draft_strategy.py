"""
Fantasy Football Draft Simulator with Reinforcement Learning

This module provides tools to:
1. Simulate fantasy football drafts with configurable strategies
2. Simulate fantasy football seasons to evaluate draft results
3. Train a reinforcement learning agent to optimize draft decisions
4. Provide a GUI for live draft assistance

The system integrates with ESPN API for league settings and NFL data for player statistics.
"""

import numpy as np
import pandas as pd
import random
import pickle
import os
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from collections import defaultdict, Counter
from tqdm import tqdm
from typing import List, Dict, Tuple, Any, Optional, Union

# For reinforcement learning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler("fantasy_draft.log"), logging.StreamHandler()]
)
logger = logging.getLogger("fantasy_draft")

class Player:
    """
    Represents a player in the draft with their projections and actual performance
    """
    
    def __init__(self, name, position, team, projected_points, 
                 projection_low=None, projection_high=None, 
                 std_dev=None, player_id=None, adp=None, tier=None):
        """
        Initialize a player with their projections
        
        Parameters:
        -----------
        name : str
            Player name
        position : str
            Player position (QB, RB, WR, TE, K, DST)
        team : str
            NFL team
        projected_points : float
            Projected fantasy points for the season
        projection_low : float, optional
            Lower bound of projection confidence interval
        projection_high : float, optional
            Upper bound of projection confidence interval
        std_dev : float, optional
            Standard deviation of projection
        player_id : str, optional
            Unique identifier for the player
        adp : float, optional
            Average draft position (ADP)
        tier : str, optional
            Player tier (e.g., "Elite", "High Tier")
        """
        self.name = name
        self.position = position
        self.team = team
        self.projected_points = projected_points
        self.projection_low = projection_low or projected_points * 0.8
        self.projection_high = projection_high or projected_points * 1.2
        self.std_dev = std_dev or (self.projection_high - self.projection_low) / 3.92  # 95% CI
        self.player_id = player_id or name.lower().replace(' ', '_')
        self.adp = adp
        self.tier = tier
        
        # Initialize actual performance (to be set during season simulation)
        self.actual_points = None
        self.weekly_points = []
        
        # Draft status
        self.drafted = False
        self.draft_position = None
        self.drafted_by = None
    
    def __repr__(self):
        return f"Player({self.name}, {self.position}, Proj: {self.projected_points:.1f})"
    
    def generate_season_performance(self, num_weeks=17, randomness=1.0):
        """
        Generate a simulated season performance for this player
        
        Parameters:
        -----------
        num_weeks : int
            Number of weeks in the fantasy season
        randomness : float
            Factor to control variability (1.0 = normal, higher = more variability)
        
        Returns:
        --------
        list
            Weekly fantasy points
        """
        # Generate actual season-long performance as a random draw from the projection distribution
        mean = self.projected_points / num_weeks  # weekly average
        sigma = self.std_dev / np.sqrt(num_weeks) * randomness
        
        # Generate weekly scores with appropriate variance
        weekly_scores = []
        
        for week in range(num_weeks):
            # Quarterbacks tend to be more consistent than other positions
            week_variance_factor = 0.8 if self.position == 'QB' else 1.0
            
            # Receivers have higher variance
            if self.position in ['WR', 'TE']:
                week_variance_factor = 1.5
            
            # Add position-specific weekly variance
            week_sigma = sigma * week_variance_factor
            
            # Generate score for this week (ensuring non-negative)
            score = max(0, np.random.normal(mean, week_sigma))
            weekly_scores.append(score)
        
        # Store the results
        self.weekly_points = weekly_scores
        self.actual_points = sum(weekly_scores)
        
        return weekly_scores

class Team:
    """
    Represents a fantasy football team in the draft
    """
    
    def __init__(self, name, draft_position, roster_limits, scoring_settings, strategy='best_available'):
        """
        Initialize a fantasy football team
        
        Parameters:
        -----------
        name : str
            Team name
        draft_position : int
            Position in the draft order
        roster_limits : dict
            Maximum players at each position, e.g., {'QB': 3, 'RB': 6, ...}
        scoring_settings : dict
            Scoring rules for the league
        strategy : str, optional
            Draft strategy (best_available, position_priority, value_based, etc.)
        """
        self.name = name
        self.draft_position = draft_position
        self.roster_limits = roster_limits
        self.scoring_settings = scoring_settings
        self.strategy = strategy
        
        # Initialize empty roster
        self.roster = {position: [] for position in roster_limits.keys()}
        self.draft_picks = []
        
        # Initialize season performance tracking
        self.weekly_scores = []
        self.wins = 0
        self.losses = 0
        self.points_for = 0
        self.points_against = 0
        self.final_standing = None
    
    def can_draft_position(self, position):
        """Check if team can draft another player at the given position"""
        return len(self.roster[position]) < self.roster_limits[position]
    
    def draft_player(self, player, pick_number):
        """Add a player to the team's roster"""
        if not self.can_draft_position(player.position):
            raise ValueError(f"Cannot draft another {player.position}, roster full")
        
        # Add player to roster
        self.roster[player.position].append(player)
        
        # Add to draft picks
        self.draft_picks.append((pick_number, player))
        
        # Update player status
        player.drafted = True
        player.draft_position = pick_number
        player.drafted_by = self.name
        
        logger.debug(f"Team {self.name} drafted {player.name} ({player.position}) at pick {pick_number}")
    
    def get_roster_state(self):
        """Get current roster state as a feature vector for RL"""
        # Count players by position
        position_counts = {pos: len(players) for pos, players in self.roster.items()}
        
        # Calculate total projected points by position
        position_projections = {}
        for pos, players in self.roster.items():
            position_projections[pos] = sum(p.projected_points for p in players)
        
        # Calculate starters vs bench
        roster_vec = []
        for pos in sorted(self.roster_limits.keys()):
            # Add normalized count (current/max)
            roster_vec.append(position_counts[pos] / self.roster_limits[pos])
            
            # Add total projection for position
            roster_vec.append(position_projections[pos])
            
            # Add starter quality (proxy: top N players at position)
            starter_limit = min(self.roster_limits[pos], 3)  # Assume at most 3 starters per position
            starters = sorted(self.roster[pos], key=lambda p: p.projected_points, reverse=True)[:starter_limit]
            
            starter_total = sum(p.projected_points for p in starters) if starters else 0
            roster_vec.append(starter_total)
        
        return roster_vec
    
    def simulate_week(self, week, randomness=1.0):
        """
        Simulate a week's fantasy performance for the team
        
        Parameters:
        -----------
        week : int
            Week number (0-indexed)
        randomness : float
            Factor to control variability
            
        Returns:
        --------
        float
            Fantasy points scored for the week
        """
        # Get all players' weekly points
        all_weekly_points = {}
        
        # Ensure all players have simulated weekly points
        for position, players in self.roster.items():
            for player in players:
                if not player.weekly_points:
                    player.generate_season_performance(randomness=randomness)
                
                # Store player's points for the week
                all_weekly_points[player] = player.weekly_points[week]
        
        # Determine optimal starting lineup for the week
        # This can be enhanced with better positions/flex handling
        lineup_points = self._calculate_optimal_lineup(all_weekly_points)
        
        # Store the team's score for the week
        self.weekly_scores.append(lineup_points)
        
        return lineup_points
    
    def _calculate_optimal_lineup(self, player_points):
        """
        Calculate the optimal lineup given player performances
        
        Parameters:
        -----------
        player_points : dict
            Dictionary mapping Player objects to their points
            
        Returns:
        --------
        float
            Total points for optimal lineup
        """
        # Define starting positions (simplified)
        # This should be enhanced with proper flex positions, etc.
        starters = {
            'QB': 1,
            'RB': 2,
            'WR': 3,
            'TE': 1,
            'K': 1,
            'DST': 1
        }
        
        # Adjust based on league settings
        for pos in starters:
            if pos in self.roster_limits:
                starters[pos] = min(starters[pos], self.roster_limits[pos])
            else:
                starters[pos] = 0
        
        # Calculate optimal lineup
        total_points = 0
        
        # For each position, select the top N players
        for position, count in starters.items():
            # Get all players at this position and their points
            pos_players = [(p, player_points[p]) for p in self.roster.get(position, [])]
            
            # Sort by points (descending)
            pos_players.sort(key=lambda x: x[1], reverse=True)
            
            # Add top players to lineup
            top_players = pos_players[:count]
            position_points = sum(points for _, points in top_players)
            
            total_points += position_points
        
        # Add flex position logic here if needed
        
        return total_points
    
    def __repr__(self):
        roster_summary = ", ".join(f"{pos}: {len(players)}" for pos, players in self.roster.items())
        return f"Team({self.name}, Pick: {self.draft_position}, Roster: {roster_summary})"


class DraftSimulator:
    """
    Simulates a fantasy football draft with multiple teams
    """
    
    def __init__(self, players, num_teams=12, roster_limits=None, scoring_settings=None, 
                 draft_type='snake', randomize_adp=0.2):
        """
        Initialize the draft simulator
        
        Parameters:
        -----------
        players : list
            List of Player objects
        num_teams : int, optional
            Number of teams in the league
        roster_limits : dict, optional
            Maximum players at each position
        scoring_settings : dict, optional
            Scoring rules for the league
        draft_type : str, optional
            Type of draft ('snake', 'linear')
        randomize_adp : float, optional
            How much to randomize ADP (0.0 = no randomization, 1.0 = full randomization)
        """
        self.players = players.copy()  # Make a copy to avoid modifying the original
        self.num_teams = num_teams
        
        # Default roster limits if not provided
        self.roster_limits = roster_limits or {
            'QB': 3,
            'RB': 6,
            'WR': 6,
            'TE': 3,
            'K': 1,
            'DST': 1
        }
        
        # Default scoring settings if not provided
        self.scoring_settings = scoring_settings or {
            'pass_yd': 0.04,  # 1 point per 25 yards
            'pass_td': 4,
            'interception': -2,
            'rush_yd': 0.1,   # 1 point per 10 yards
            'rush_td': 6,
            'rec': 0,         # 0 for standard, 0.5 for half PPR, 1 for PPR
            'rec_yd': 0.1,    # 1 point per 10 yards
            'rec_td': 6,
            'fumble_lost': -2
        }
        
        self.draft_type = draft_type
        self.randomize_adp = randomize_adp
        
        # Calculate total roster size
        self.roster_size = sum(self.roster_limits.values())
        
        # Initialize teams
        self.teams = []
        for i in range(num_teams):
            team = Team(
                name=f"Team {i+1}",
                draft_position=i+1,
                roster_limits=self.roster_limits,
                scoring_settings=self.scoring_settings
            )
            self.teams.append(team)
        
        # Generate draft order
        self.draft_order = self._generate_draft_order()
        
        # Initialize draft state
        self.current_pick = 0
        self.drafted_players = []
        self.available_players = self.players.copy()
        
        # Reset player draft status
        for player in self.players:
            player.drafted = False
            player.draft_position = None
            player.drafted_by = None
    
    def _generate_draft_order(self):
        """Generate draft order based on draft type"""
        draft_order = []
        
        if self.draft_type == 'snake':
            for round_num in range(self.roster_size):
                # Even rounds go in reverse order
                if round_num % 2 == 0:
                    for team_idx in range(self.num_teams):
                        draft_order.append(team_idx)
                else:
                    for team_idx in range(self.num_teams - 1, -1, -1):
                        draft_order.append(team_idx)
        else:  # Linear draft
            for round_num in range(self.roster_size):
                for team_idx in range(self.num_teams):
                    draft_order.append(team_idx)
        
        return draft_order
    
    def _get_best_player(self, team, position=None, draft_pool=None, value_based=False):
        """
        Get the best available player for a team, optionally filtering by position
        
        Parameters:
        -----------
        team : Team
            Team making the pick
        position : str, optional
            Target position (filter to just this position)
        draft_pool : list, optional
            List of available players (defaults to all available players)
        value_based : bool, optional
            Whether to use value-based drafting (position scarcity)
            
        Returns:
        --------
        Player
            Best available player
        """
        available = draft_pool if draft_pool is not None else self.available_players
        
        # Filter by position if specified
        if position:
            available = [p for p in available if p.position == position]
        
        # Filter by roster limits
        available = [p for p in available if team.can_draft_position(p.position)]
        
        if not available:
            return None
        
        if value_based:
            # Value-based drafting: adjust value based on positional scarcity
            # Calculate current position values
            position_values = self._calculate_position_values()
            
            # Sort by adjusted value
            available.sort(key=lambda p: p.projected_points * position_values.get(p.position, 1.0), reverse=True)
        else:
            # Simple sorting by projected points
            available.sort(key=lambda p: p.projected_points, reverse=True)
        
        return available[0] if available else None
    
    def _calculate_position_values(self):
        """
        Calculate position scarcity values
        
        Returns:
        --------
        dict
            Multiplier values for each position
        """
        position_values = {
            'QB': 1.0,
            'RB': 1.0,
            'WR': 1.0,
            'TE': 1.0,
            'K': 1.0,
            'DST': 1.0
        }
        
        # Analyze remaining players at each position
        remaining_by_pos = defaultdict(list)
        for player in self.available_players:
            remaining_by_pos[player.position].append(player.projected_points)
        
        # Calculate positional scarcity
        for pos in remaining_by_pos:
            if not remaining_by_pos[pos]:
                continue
                
            # Count how many more players are needed at this position
            total_needed = sum(team.roster_limits[pos] - len(team.roster[pos]) 
                              for team in self.teams)
            
            # Calculate scarcity metric
            if total_needed > 0:
                # More limited positions get higher values
                scarcity = len(remaining_by_pos[pos]) / total_needed
                position_values[pos] = 1.0 + (1.0 / scarcity)
        
        return position_values
    
    def get_next_pick(self, strategy="best_available", team=None, use_randomization=True):
        """
        Determine the next pick in the draft
        
        Parameters:
        -----------
        strategy : str
            Draft strategy
        team : Team, optional
            Team making the pick (if None, use the next team in draft order)
        use_randomization : bool
            Whether to randomize selection
            
        Returns:
        --------
        tuple
            (team, player) - The team making the pick and the selected player
        """
        if self.current_pick >= len(self.draft_order):
            return None, None
        
        # Determine team making the pick
        team_idx = self.draft_order[self.current_pick]
        current_team = team if team is not None else self.teams[team_idx]
        
        # Get available players
        available = [p for p in self.players if not p.drafted]
        
        # Apply randomization to ADP if requested
        if use_randomization and self.randomize_adp > 0:
            for player in available:
                # Only randomize if player has an ADP
                if player.adp:
                    # Add noise proportional to ADP and randomization factor
                    noise = np.random.normal(0, player.adp * self.randomize_adp / 2)
                    player.tmp_projected = player.projected_points + noise
                else:
                    player.tmp_projected = player.projected_points
        else:
            # No randomization
            for player in available:
                player.tmp_projected = player.projected_points
        
        # Select player based on strategy
        selected_player = None
        
        if strategy == "best_available":
            # Sort by projected points and take the best
            available.sort(key=lambda p: p.tmp_projected, reverse=True)
            
            # Find the best player the team can draft
            for player in available:
                if current_team.can_draft_position(player.position):
                    selected_player = player
                    break
        
        elif strategy == "position_priority":
            # Define position priorities for drafting
            # Early rounds: RB, WR, TE, QB
            # Late rounds: QB, RB, WR, K, DST
            
            if self.current_pick < self.num_teams * 6:  # First 6 rounds
                priorities = ['RB', 'WR', 'TE', 'QB', 'K', 'DST']
            else:
                priorities = ['QB', 'RB', 'WR', 'TE', 'K', 'DST']
            
            # Find the best player at the highest priority position
            for position in priorities:
                if current_team.can_draft_position(position):
                    pos_players = [p for p in available if p.position == position]
                    if pos_players:
                        pos_players.sort(key=lambda p: p.tmp_projected, reverse=True)
                        selected_player = pos_players[0]
                        break
        
        elif strategy == "value_based":
            # Value-based drafting 
            selected_player = self._get_best_player(current_team, value_based=True)
        
        elif strategy == "handcuff":
            # First 10 rounds use value-based
            if self.current_pick < self.num_teams * 10:
                selected_player = self._get_best_player(current_team, value_based=True)
            else:
                # Later rounds, try to handcuff RBs
                # Get RBs already on team
                team_rbs = current_team.roster.get('RB', [])
                
                if team_rbs and current_team.can_draft_position('RB'):
                    # Get RB handcuffs - same team as existing RB
                    handcuffs = [p for p in available 
                              if p.position == 'RB' and 
                              any(p.team == rb.team for rb in team_rbs)]
                    
                    if handcuffs:
                        # Take best handcuff
                        handcuffs.sort(key=lambda p: p.tmp_projected, reverse=True)
                        selected_player = handcuffs[0]
                
                # If no handcuff available, use value-based
                if not selected_player:
                    selected_player = self._get_best_player(current_team, value_based=True)
        
        # If no player selected, try best_available as fallback
        if not selected_player:
            # Sort by projected points and take the best available player the team can draft
            available.sort(key=lambda p: p.tmp_projected, reverse=True)
            for player in available:
                if current_team.can_draft_position(player.position):
                    selected_player = player
                    break
        
        # Final safety check - if still no player, take anyone who fits
        if not selected_player and available:
            for player in available:
                if current_team.can_draft_position(player.position):
                    selected_player = player
                    break
        
        # Clean up temporary projections
        for player in available:
            if hasattr(player, 'tmp_projected'):
                delattr(player, 'tmp_projected')
        
        return current_team, selected_player
    
    def make_pick(self, team=None, player=None, strategy="best_available"):
        """
        Make the next pick in the draft
        
        Parameters:
        -----------
        team : Team, optional
            Team making the pick (if None, use the next team in draft order)
        player : Player, optional
            Player to draft (if None, auto-select based on strategy)
        strategy : str, optional
            Draft strategy if auto-selecting player
            
        Returns:
        --------
        tuple
            (team, player) - The team that made the pick and the selected player
        """
        if self.current_pick >= len(self.draft_order):
            return None, None
        
        # Determine team making the pick
        team_idx = self.draft_order[self.current_pick]
        current_team = team if team is not None else self.teams[team_idx]
        
        # Determine player to pick
        if player is None:
            current_team, selected_player = self.get_next_pick(strategy, current_team)
        else:
            selected_player = player
        
        # Make the pick
        if selected_player:
            # Ensure player isn't already drafted
            if selected_player.drafted:
                raise ValueError(f"Player {selected_player.name} is already drafted")
            
            # Ensure team can draft this position
            if not current_team.can_draft_position(selected_player.position):
                raise ValueError(f"Team {current_team.name} cannot draft another {selected_player.position}")
            
            # Make the pick
            current_team.draft_player(selected_player, self.current_pick + 1)
            
            # Update draft state
            self.drafted_players.append(selected_player)
            self.available_players = [p for p in self.available_players if p != selected_player]
            self.current_pick += 1
            
            return current_team, selected_player
        
        # No valid pick could be made
        return current_team, None
    
    def simulate_draft(self, team_strategies=None):
        """
        Simulate the entire draft
        
        Parameters:
        -----------
        team_strategies : dict, optional
            Dictionary mapping team indices to strategies
            
        Returns:
        --------
        list
            List of (team, player) tuples for each pick
        """
        # Reset draft state
        self.current_pick = 0
        self.drafted_players = []
        self.available_players = self.players.copy()
        
        # Reset player draft status
        for player in self.players:
            player.drafted = False
            player.draft_position = None
            player.drafted_by = None
        
        # Reset team rosters
        for team in self.teams:
            team.roster = {position: [] for position in self.roster_limits.keys()}
            team.draft_picks = []
        
        # Set up team strategies
        if not team_strategies:
            team_strategies = {}
        
        # Default to best_available if not specified
        default_strategy = "best_available"
        
        # Simulate all picks
        picks = []
        while self.current_pick < len(self.draft_order):
            team_idx = self.draft_order[self.current_pick]
            strategy = team_strategies.get(team_idx, default_strategy)
            team, player = self.make_pick(strategy=strategy)
            
            if player:
                picks.append((team, player))
            else:
                break  # No valid pick could be made
        
        return picks
    
    def assess_draft(self, team):
        """
        Assess the quality of a team's draft
        
        Parameters:
        -----------
        team : Team
            Team to assess
            
        Returns:
        --------
        dict
            Dictionary with draft assessment metrics
        """
        assessment = {}
        
        # Calculate total projected points
        total_projected = 0
        position_projected = {}
        
        for position, players in team.roster.items():
            position_total = sum(p.projected_points for p in players)
            position_projected[position] = position_total
            total_projected += position_total
        
        assessment['total_projected'] = total_projected
        assessment['position_projected'] = position_projected
        
        # Calculate position balance (starter quality vs depth)
        starter_projected = 0
        starter_counts = {
            'QB': 1,
            'RB': 2,
            'WR': 3,
            'TE': 1,
            'K': 1,
            'DST': 1
        }
        
        for position, count in starter_counts.items():
            if position in team.roster:
                # Sort players by projected points
                sorted_players = sorted(team.roster[position], key=lambda p: p.projected_points, reverse=True)
                
                # Take top N as starters
                starters = sorted_players[:count]
                starter_projected += sum(p.projected_points for p in starters)
        
        assessment['starter_projected'] = starter_projected
        
        # Calculate value gained vs ADP
        value_gained = 0
        for pick_num, player in team.draft_picks:
            if player.adp:
                # Positive value if drafted later than ADP
                value_gained += (player.adp - pick_num)
        
        assessment['value_gained'] = value_gained
        
        return assessment


class SeasonSimulator:
    """
    Simulates a fantasy football season and playoffs
    """
    
    def __init__(self, teams, num_weeks=14, playoff_weeks=3, playoff_teams=6, randomness=1.0):
        """
        Initialize the season simulator
        
        Parameters:
        -----------
        teams : list
            List of Team objects with drafted players
        num_weeks : int, optional
            Number of regular season weeks
        playoff_weeks : int, optional
            Number of playoff weeks
        playoff_teams : int, optional
            Number of teams that make the playoffs
        randomness : float, optional
            Factor to control player performance variability
        """
        self.teams = teams
        self.num_weeks = num_weeks
        self.playoff_weeks = playoff_weeks
        self.playoff_teams = playoff_teams
        self.randomness = randomness
        
        # Total season length
        self.total_weeks = num_weeks + playoff_weeks
        
        # Initialize schedule and results
        self.schedule = self._generate_schedule()
        self.results = []
        self.standings = []
    
    def _generate_schedule(self):
        """
        Generate a schedule for the season
        
        Returns:
        --------
        list
            List of lists containing matchups for each week
        """
        num_teams = len(self.teams)
        schedule = []
        
        if num_teams % 2 == 1:
            # Odd number of teams, one team gets a bye each week
            num_teams += 1
        
        # Create a list of team indices
        team_indices = list(range(num_teams))
        
        # For an odd number of teams, the last index represents a bye
        has_bye = len(self.teams) % 2 == 1
        
        # Generate schedule using circle method
        for week in range(self.num_weeks):
            matchups = []
            
            # Pair up teams
            for i in range(num_teams // 2):
                team1_idx = team_indices[i]
                team2_idx = team_indices[num_teams - 1 - i]
                
                # Skip matchups involving the bye team
                if has_bye and (team1_idx == num_teams - 1 or team2_idx == num_teams - 1):
                    continue
                
                # Ensure team indices are valid
                if team1_idx < len(self.teams) and team2_idx < len(self.teams):
                    matchups.append((team1_idx, team2_idx))
            
            schedule.append(matchups)
            
            # Rotate teams for next week
            team_indices = [team_indices[0]] + [team_indices[-1]] + team_indices[1:-1]
        
        return schedule
    
    def simulate_season(self):
        """
        Simulate the entire fantasy football season
        
        Returns:
        --------
        list
            Final standings with teams sorted by record and points
        """
        # Reset team stats
        for team in self.teams:
            team.weekly_scores = []
            team.wins = 0
            team.losses = 0
            team.points_for = 0
            team.points_against = 0
            team.final_standing = None
        
        # Simulate regular season weeks
        weekly_results = []
        for week in range(self.num_weeks):
            week_results = self._simulate_week(week)
            weekly_results.append(week_results)
        
        # Generate regular season standings
        reg_season_standings = self._generate_standings()
        
        # Simulate playoffs
        playoff_results = self._simulate_playoffs(reg_season_standings)
        
        # Combine results
        self.results = weekly_results + playoff_results
        
        # Final standings
        final_standings = []
        final_rank = 1
        
        # Champions and runners-up from playoffs
        for i, playoff_finish in enumerate(self.playoff_teams):
            team = reg_season_standings[playoff_finish]
            team.final_standing = final_rank
            final_standings.append(team)
            final_rank += 1
        
        # Remaining teams in order of regular season standings
        for team in reg_season_standings:
            if team not in final_standings:
                team.final_standing = final_rank
                final_standings.append(team)
                final_rank += 1
        
        self.standings = final_standings
        
        return final_standings
    
    def _simulate_week(self, week):
        """
        Simulate one week of the season
        
        Parameters:
        -----------
        week : int
            Week number (0-indexed)
            
        Returns:
        --------
        list
            List of (team1, score1, team2, score2) tuples for each matchup
        """
        # Generate weekly scores for all teams
        for team in self.teams:
            team.simulate_week(week, randomness=self.randomness)
        
        # Simulate matchups
        week_results = []
        for team1_idx, team2_idx in self.schedule[week]:
            team1 = self.teams[team1_idx]
            team2 = self.teams[team2_idx]
            
            score1 = team1.weekly_scores[week]
            score2 = team2.weekly_scores[week]
            
            # Update season stats
            team1.points_for += score1
            team1.points_against += score2
            team2.points_for += score2
            team2.points_against += score1
            
            # Determine winner
            if score1 > score2:
                team1.wins += 1
                team2.losses += 1
            elif score2 > score1:
                team2.wins += 1
                team1.losses += 1
            else:
                # In case of tie, both teams get half a win
                team1.wins += 0.5
                team2.wins += 0.5
            
            week_results.append((team1, score1, team2, score2))
        
        return week_results
    
    def _generate_standings(self):
        """
        Generate standings based on current records
        
        Returns:
        --------
        list
            Teams sorted by record and total points
        """
        # Sort teams by wins (descending) and then points (descending)
        return sorted(self.teams, key=lambda t: (t.wins, t.points_for), reverse=True)
    
    def _simulate_playoffs(self, standings):
        """
        Simulate the fantasy playoffs
        
        Parameters:
        -----------
        standings : list
            Regular season standings
            
        Returns:
        --------
        list
            List of playoff week results
        """
        # Select playoff teams
        playoff_teams = standings[:self.playoff_teams]
        
        # Playoff bracket (simplified)
        # For 6 teams: 1 & 2 get byes, 3 vs 6, 4 vs 5
        # For 4 teams: 1 vs 4, 2 vs 3
        playoff_results = []
        remaining_teams = playoff_teams.copy()
        
        # Track playoff finishes
        self.playoff_finishes = []
        
        if len(playoff_teams) == 6:
            # Week 1: Byes for 1 & 2, 3 vs 6, 4 vs 5
            if self.playoff_weeks >= 3:
                # First round (wild card)
                week_results = []
                
                # 3 vs 6
                matchup1 = self._simulate_playoff_matchup(self.num_weeks, remaining_teams[2], remaining_teams[5])
                week_results.append(matchup1)
                
                # 4 vs 5
                matchup2 = self._simulate_playoff_matchup(self.num_weeks, remaining_teams[3], remaining_teams[4])
                week_results.append(matchup2)
                
                playoff_results.append(week_results)
                
                # Update remaining teams
                remaining_teams = [
                    remaining_teams[0],  # 1 seed
                    remaining_teams[1],  # 2 seed
                    matchup1[0] if matchup1[2] > matchup1[3] else matchup1[1],  # 3/6 winner
                    matchup2[0] if matchup2[2] > matchup2[3] else matchup2[1]   # 4/5 winner
                ]
                
                # Second round (semifinals)
                week_results = []
                
                # 1 vs lowest remaining seed
                matchup1 = self._simulate_playoff_matchup(self.num_weeks + 1, remaining_teams[0], remaining_teams[3])
                week_results.append(matchup1)
                
                # 2 vs other remaining seed
                matchup2 = self._simulate_playoff_matchup(self.num_weeks + 1, remaining_teams[1], remaining_teams[2])
                week_results.append(matchup2)
                
                playoff_results.append(week_results)
                
                # Update remaining teams
                championship_teams = [
                    matchup1[0] if matchup1[2] > matchup1[3] else matchup1[1],  # First semi winner
                    matchup2[0] if matchup2[2] > matchup2[3] else matchup2[1]   # Second semi winner
                ]
                
                # Add semifinal losers to playoff finishes (tied for 3rd)
                self.playoff_finishes.append(matchup1[0] if matchup1[2] < matchup1[3] else matchup1[1])
                self.playoff_finishes.append(matchup2[0] if matchup2[2] < matchup2[3] else matchup2[1])
                
                # Championship game
                championship = self._simulate_playoff_matchup(self.num_weeks + 2, championship_teams[0], championship_teams[1])
                playoff_results.append([championship])
                
                # Add championship results to playoff finishes
                champion = championship[0] if championship[2] > championship[3] else championship[1]
                runner_up = championship[0] if championship[2] < championship[3] else championship[1]
                
                self.playoff_finishes.insert(0, champion)
                self.playoff_finishes.insert(0, runner_up)
        
        elif len(playoff_teams) == 4:
            # 4-team playoff: 1 vs 4, 2 vs 3
            if self.playoff_weeks >= 2:
                # Semifinals
                week_results = []
                
                # 1 vs 4
                matchup1 = self._simulate_playoff_matchup(self.num_weeks, remaining_teams[0], remaining_teams[3])
                week_results.append(matchup1)
                
                # 2 vs 3
                matchup2 = self._simulate_playoff_matchup(self.num_weeks, remaining_teams[1], remaining_teams[2])
                week_results.append(matchup2)
                
                playoff_results.append(week_results)
                
                # Update remaining teams
                championship_teams = [
                    matchup1[0] if matchup1[2] > matchup1[3] else matchup1[1],  # First semi winner
                    matchup2[0] if matchup2[2] > matchup2[3] else matchup2[1]   # Second semi winner
                ]
                
                # Add semifinal losers to playoff finishes (tied for 3rd)
                self.playoff_finishes.append(matchup1[0] if matchup1[2] < matchup1[3] else matchup1[1])
                self.playoff_finishes.append(matchup2[0] if matchup2[2] < matchup2[3] else matchup2[1])
                
                # Championship game
                championship = self._simulate_playoff_matchup(self.num_weeks + 1, championship_teams[0], championship_teams[1])
                playoff_results.append([championship])
                
                # Add championship results to playoff finishes
                champion = championship[0] if championship[2] > championship[3] else championship[1]
                runner_up = championship[0] if championship[2] < championship[3] else championship[1]
                
                self.playoff_finishes.insert(0, champion)
                self.playoff_finishes.insert(0, runner_up)
        
        return playoff_results
    
    def _simulate_playoff_matchup(self, week, team1, team2):
        """
        Simulate a playoff matchup
        
        Parameters:
        -----------
        week : int
            Week number (0-indexed)
        team1 : Team
            First team
        team2 : Team
            Second team
            
        Returns:
        --------
        tuple
            (team1, team2, score1, score2)
        """
        # Generate weekly scores
        score1 = team1.simulate_week(week, randomness=self.randomness)
        score2 = team2.simulate_week(week, randomness=self.randomness)
        
        return (team1, team2, score1, score2)
    
    def get_team_result(self, team):
        """
        Get a team's final result from the season
        
        Parameters:
        -----------
        team : Team
            Team to get result for
            
        Returns:
        --------
        dict
            Dictionary with team performance metrics
        """
        result = {
            'team': team,
            'record': (team.wins, team.losses),
            'win_pct': team.wins / (team.wins + team.losses) if (team.wins + team.losses) > 0 else 0,
            'points_for': team.points_for,
            'points_against': team.points_against,
            'final_standing': team.final_standing
        }
        
        # Calculate expected wins based on points scored
        total_games = team.wins + team.losses
        if total_games > 0:
            total_points = sum(t.points_for for t in self.teams)
            expected_win_pct = team.points_for / total_points * len(self.teams) / 2
            result['expected_wins'] = expected_win_pct * total_games
            result['luck_factor'] = team.wins - result['expected_wins']
        else:
            result['expected_wins'] = 0
            result['luck_factor'] = 0
        
        return result


class FantasyRL:
    """
    Reinforcement Learning agent for optimizing fantasy football draft strategy
    """
    
    def __init__(self, players, league_settings=None, num_teams=12, draft_type='snake',
                 learning_rate=0.001, hidden_layer_sizes=(100, 50), random_seed=42):
        """
        Initialize the RL agent
        
        Parameters:
        -----------
        players : list
            List of Player objects
        league_settings : dict, optional
            League settings (roster limits, scoring, etc.)
        num_teams : int, optional
            Number of teams in the league
        draft_type : str, optional
            Type of draft ('snake', 'linear')
        learning_rate : float, optional
            Learning rate for the RL algorithm
        hidden_layer_sizes : tuple, optional
            Hidden layer sizes for the neural network
        random_seed : int, optional
            Random seed for reproducibility
        """
        self.players = players
        self.league_settings = league_settings or {}
        self.num_teams = num_teams
        self.draft_type = draft_type
        
        # Set default roster limits if not provided
        self.roster_limits = league_settings.get('roster_limits', {
            'QB': 3,
            'RB': 6,
            'WR': 6,
            'TE': 3,
            'K': 1,
            'DST': 1
        })
        
        # Set default scoring settings if not provided
        self.scoring_settings = league_settings.get('scoring_settings', {
            'pass_yd': 0.04,  # 1 point per 25 yards
            'pass_td': 4,
            'interception': -2,
            'rush_yd': 0.1,   # 1 point per 10 yards
            'rush_td': 6,
            'rec': 0,         # 0 for standard, 0.5 for half PPR, 1 for PPR
            'rec_yd': 0.1,    # 1 point per 10 yards
            'rec_td': 6,
            'fumble_lost': -2
        })
        
        # Set up neural network for value function approximation
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Create model for value function
        self.model = MLPRegressor(
            hidden_layer_sizes=hidden_layer_sizes,
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='constant',
            learning_rate_init=learning_rate,
            max_iter=500,
            random_state=random_seed
        )
        
        # Initialize feature scaler
        self.scaler = StandardScaler()
        
        # Create experience memory for batch training
        self.experiences = []
        
        # Store previous simulations for analysis
        self.simulations = []
    
    def _build_state_features(self, team, available_players, current_pick):
        """
        Build feature vector representing current draft state
        
        Parameters:
        -----------
        team : Team
            Team making the current pick
        available_players : list
            List of available players
        current_pick : int
            Current pick number
            
        Returns:
        --------
        numpy.ndarray
            Feature vector
        """
        # Get team roster state
        roster_state = team.get_roster_state()
        
        # Calculate draft progress features
        total_picks = self.num_teams * sum(self.roster_limits.values())
        draft_progress = current_pick / total_picks
        
        round_num = current_pick // self.num_teams + 1
        pick_in_round = current_pick % self.num_teams + 1
        
        # Calculate available player features
        avail_by_pos = {}
        for position in self.roster_limits.keys():
            pos_players = [p for p in available_players if p.position == position]
            if pos_players:
                avg_proj = np.mean([p.projected_points for p in pos_players])
                max_proj = np.max([p.projected_points for p in pos_players])
                count = len(pos_players)
                
                # For select positions, also track tier distribution
                if position in ['QB', 'RB', 'WR', 'TE']:
                    high_tier_count = len([p for p in pos_players 
                                          if p.tier and p.tier in ['Elite', 'High Tier']])
                    mid_tier_count = len([p for p in pos_players 
                                         if p.tier and p.tier in ['Mid Tier']])
                    low_tier_count = len([p for p in pos_players 
                                         if p.tier and p.tier == 'Low Tier'])
                    
                    avail_by_pos[position] = [avg_proj, max_proj, count, 
                                              high_tier_count, mid_tier_count, low_tier_count]
                else:
                    avail_by_pos[position] = [avg_proj, max_proj, count, 0, 0, 0]
            else:
                avail_by_pos[position] = [0, 0, 0, 0, 0, 0]
        
        # Flatten available player features
        avail_features = []
        for position in sorted(self.roster_limits.keys()):
            avail_features.extend(avail_by_pos[position])
        
        # Build full feature vector
        features = roster_state + [draft_progress, round_num, pick_in_round] + avail_features
        
        # Convert to numpy array
        return np.array(features).reshape(1, -1)
    
    def _evaluate_draft_reward(self, team_results):
        """
        Calculate reward based on season results
        
        Parameters:
        -----------
        team_results : dict
            Dictionary with team performance metrics
            
        Returns:
        --------
        float
            Reward value
        """
        # Use a combination of:
        # 1. Final standing (higher is better)
        # 2. Win percentage (higher is better)
        # 3. Points scored (higher is better)
        # 4. Expected wins vs actual wins (luck factor)
        
        final_standing = team_results['final_standing']
        win_pct = team_results['win_pct']
        points_for = team_results['points_for']
        luck_factor = team_results['luck_factor']
        
        # Normalize the standings
        standing_reward = (self.num_teams - final_standing + 1) / self.num_teams
        
        # Championship bonus (1.5x reward if you win it all)
        championship_bonus = 1.5 if final_standing == 1 else 1.0
        
        # Calculate reward components
        standing_component = standing_reward * championship_bonus * 0.7  # Weight: 70%
        win_component = win_pct * 0.2  # Weight: 20%
        points_component = (points_for / 1500) * 0.1  # Weight: 10% (normalized by typical season total)
        
        # Adjust for luck (subtract luck factor to reward consistent strategy)
        luck_component = -abs(luck_factor) * 0.05  # Small penalty for being lucky/unlucky
        
        reward = standing_component + win_component + points_component + luck_component
        
        return reward
    
    def train(self, num_episodes=100, target_position=1, exploration_rate=0.3, randomize_opponents=True):
        """
        Train the agent through multiple draft simulations
        
        Parameters:
        -----------
        num_episodes : int, optional
            Number of draft episodes to simulate
        target_position : int, optional
            Draft position to optimize for (1-indexed)
        exploration_rate : float, optional
            Initial exploration rate (epsilon-greedy)
        randomize_opponents : bool, optional
            Whether to randomize opponent strategies
            
        Returns:
        --------
        dict
            Training statistics
        """
        # Initialize training stats
        training_stats = {
            'rewards': [],
            'final_standings': [],
            'win_pcts': [],
            'points_scored': [],
            'exploration_rates': []
        }
        
        # Initialize model with a dummy fit
        dummy_X = np.random.rand(10, len(self._build_state_features(
            Team("Dummy", 1, self.roster_limits, self.scoring_settings), 
            self.players, 0).flatten()))
        
        dummy_y = np.random.rand(10)
        self.model.fit(dummy_X, dummy_y)
        
        # Train over multiple episodes
        for episode in tqdm(range(num_episodes), desc="Training RL Agent"):
            # Decay exploration rate
            current_exploration = max(0.05, exploration_rate * (1 - episode / num_episodes))
            
            # Run one training episode
            reward, stats = self._train_episode(target_position, current_exploration, randomize_opponents)
            
            # Update stats
            training_stats['rewards'].append(reward)
            training_stats['final_standings'].append(stats['final_standing'])
            training_stats['win_pcts'].append(stats['win_pct'])
            training_stats['points_scored'].append(stats['points_for'])
            training_stats['exploration_rates'].append(current_exploration)
            
            # Log progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(training_stats['rewards'][-10:])
                avg_standing = np.mean(training_stats['final_standings'][-10:])
                
                logger.info(f"Episode {episode+1}: Avg Reward: {avg_reward:.4f}, Avg Standing: {avg_standing:.2f}")
                
            # Batch training every 10 episodes
            if (episode + 1) % 10 == 0 and self.experiences:
                self._batch_train()
                
        # Final batch training
        if self.experiences:
            self._batch_train()
        
        # Calculate training statistics
        final_stats = {
            'avg_reward': np.mean(training_stats['rewards']),
            'avg_standing': np.mean(training_stats['final_standings']),
            'avg_win_pct': np.mean(training_stats['win_pcts']),
            'avg_points': np.mean(training_stats['points_scored']),
            'episodes': num_episodes,
            'target_position': target_position
        }
        
        # Plot training progress
        self._plot_training_progress(training_stats)
        
        return final_stats
    
    def _train_episode(self, target_position, exploration_rate, randomize_opponents):
        """
        Run one training episode (draft + season simulation)
        
        Parameters:
        -----------
        target_position : int
            Draft position to optimize for (1-indexed)
        exploration_rate : float
            Current exploration rate
        randomize_opponents : bool
            Whether to randomize opponent strategies
            
        Returns:
        --------
        tuple
            (reward, team_results) - Episode reward and team results
        """
        # Initialize draft simulator
        simulator = DraftSimulator(
            players=self.players,
            num_teams=self.num_teams,
            roster_limits=self.roster_limits,
            scoring_settings=self.scoring_settings,
            draft_type=self.draft_type
        )
        
        # Set up team strategies
        team_strategies = {}
        if randomize_opponents:
            # Randomly assign strategies to opponents
            possible_strategies = ['best_available', 'position_priority', 'value_based', 'handcuff']
            
            for team_idx in range(self.num_teams):
                if team_idx != target_position - 1:  # Skip the target team
                    team_strategies[team_idx] = random.choice(possible_strategies)
        
        # Episode memory (state, action, reward, next_state)
        episode_memory = []
        
        # Reset draft state
        simulator.current_pick = 0
        simulator.drafted_players = []
        simulator.available_players = simulator.players.copy()
        
        # Reset player draft status
        for player in simulator.players:
            player.drafted = False
            player.draft_position = None
            player.drafted_by = None
        
        # Reset team rosters
        for team in simulator.teams:
            team.roster = {position: [] for position in simulator.roster_limits.keys()}
            team.draft_picks = []
        
        # Get target team
        target_team = simulator.teams[target_position - 1]
        
        # Simulate draft
        while simulator.current_pick < len(simulator.draft_order):
            team_idx = simulator.draft_order[simulator.current_pick]
            current_team = simulator.teams[team_idx]
            
            if team_idx == target_position - 1:
                # Target team uses RL policy
                state = self._build_state_features(
                    current_team,
                    simulator.available_players,
                    simulator.current_pick
                )
                
                # Choose action with epsilon-greedy policy
                if np.random.random() < exploration_rate:
                    # Exploration: random action
                    available = [p for p in simulator.available_players if not p.drafted]
                    available = [p for p in available if current_team.can_draft_position(p.position)]
                    
                    if available:
                        # Pick one at random
                        selected_player = random.choice(available)
                    else:
                        # No valid pick
                        break
                else:
                    # Exploitation: use model
                    selected_player = self._select_best_player(current_team, simulator)
                
                # Make the pick
                _, player = simulator.make_pick(current_team, selected_player)
                
                if player:
                    # Store this state-action pair
                    action_idx = simulator.players.index(player)
                    episode_memory.append((state, action_idx, simulator.current_pick))
            else:
                # Other teams use assigned strategy
                strategy = team_strategies.get(team_idx, 'best_available')
                _, _ = simulator.make_pick(strategy=strategy)
        
        # Simulate season
        season = SeasonSimulator(
            teams=simulator.teams,
            num_weeks=13,
            playoff_weeks=3,
            playoff_teams=min(6, self.num_teams // 2)
        )
        
        final_standings = season.simulate_season()
        
        # Get target team's results
        team_result = season.get_team_result(target_team)
        
        # Calculate reward
        reward = self._evaluate_draft_reward(team_result)
        
        # Update episode memory with rewards
        for state, action_idx, pick_num in episode_memory:
            # Calculate pick-specific reward
            # Picks earlier in the draft have more impact, weight accordingly
            pick_weight = 1.0 - pick_num / (simulator.num_teams * sum(simulator.roster_limits.values()))
            pick_reward = reward * (0.8 + pick_weight * 0.2)
            
            # Add to experience memory
            self.experiences.append((state, action_idx, pick_reward))
        
        # Store simulation for analysis
        self.simulations.append({
            'draft': simulator.draft_order,
            'team_picks': target_team.draft_picks,
            'results': team_result
        })
        
        return reward, team_result
    
    def _select_best_player(self, team, simulator):
        """
        Select the best player according to the learned value function
        
        Parameters:
        -----------
        team : Team
            Team making the pick
        simulator : DraftSimulator
            Current draft simulator state
            
        Returns:
        --------
        Player
            Best player to draft
        """
        # Get available players that can be drafted
        available = [p for p in simulator.available_players if not p.drafted]
        available = [p for p in available if team.can_draft_position(p.position)]
        
        if not available:
            return None
        
        # Get current state
        current_state = self._build_state_features(
            team,
            simulator.available_players,
            simulator.current_pick
        )
        
        # Evaluate each possible action
        player_values = []
        
        for player in available:
            # Simulate drafting this player
            team_copy = Team(
                team.name,
                team.draft_position,
                team.roster_limits,
                team.scoring_settings,
                team.strategy
            )
            
            # Copy current roster
            for position, players in team.roster.items():
                team_copy.roster[position] = players.copy()
            
            # Add player to roster
            team_copy.roster[player.position].append(player)
            
            # Get next state
            next_available = [p for p in simulator.available_players if p != player and not p.drafted]
            
            next_state = self._build_state_features(
                team_copy,
                next_available,
                simulator.current_pick + 1
            )
            
            # Get predicted value
            value = self.model.predict(next_state)[0]
            
            player_values.append((player, value))
        
        # Select player with highest value
        player_values.sort(key=lambda x: x[1], reverse=True)
        
        return player_values[0][0] if player_values else None
    
    def _batch_train(self):
        """
        Train the model using batch learning from experience memory
        """
        if not self.experiences:
            return
        
        # Prepare training data
        states = []
        rewards = []
        
        for state, _, reward in self.experiences:
            states.append(state.flatten())
            rewards.append(reward)
        
        # Convert to numpy arrays
        X = np.array(states)
        y = np.array(rewards)
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.partial_fit(X_scaled, y)
        
        # Clear experiences to avoid reusing them
        self.experiences = []
    
    def _plot_training_progress(self, stats):
        """
        Plot training progress
        
        Parameters:
        -----------
        stats : dict
            Training statistics
        """
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), dpi=100)
        
        # Plot rewards
        axes[0, 0].plot(stats['rewards'], 'b-')
        axes[0, 0].set_title('Reward per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # Add smoothed reward line
        if len(stats['rewards']) > 10:
            window_size = min(10, len(stats['rewards']) // 5)
            smoothed = np.convolve(stats['rewards'], np.ones(window_size)/window_size, mode='valid')
            axes[0, 0].plot(range(window_size-1, len(stats['rewards'])), smoothed, 'r-', linewidth=2)
        
        # Plot final standings (lower is better)
        axes[0, 1].plot(stats['final_standings'], 'g-')
        axes[0, 1].set_title('Final Standing per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Final Standing')
        axes[0, 1].invert_yaxis()  # Lower values are better
        
        # Add smoothed standing line
        if len(stats['final_standings']) > 10:
            window_size = min(10, len(stats['final_standings']) // 5)
            smoothed = np.convolve(stats['final_standings'], np.ones(window_size)/window_size, mode='valid')
            axes[0, 1].plot(range(window_size-1, len(stats['final_standings'])), smoothed, 'r-', linewidth=2)
        
        # Plot win percentage
        axes[1, 0].plot(stats['win_pcts'], 'm-')
        axes[1, 0].set_title('Win Percentage per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Win Percentage')
        
        # Plot exploration rate
        axes[1, 1].plot(stats['exploration_rates'], 'k-')
        axes[1, 1].set_title('Exploration Rate')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        
        # Layout and save
        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_draft_patterns(self):
        """
        Analyze draft patterns from simulations
        
        Returns:
        --------
        dict
            Analysis results
        """
        if not self.simulations:
            return {}
        
        # Count positions drafted by round
        position_by_round = defaultdict(lambda: defaultdict(int))
        
        # Count successful patterns
        pick_success = defaultdict(list)
        
        # Analyze each simulation
        for sim in self.simulations:
            team_picks = sim['team_picks']
            results = sim['results']
            
            # Consider successful if team finished in top 3
            successful = results['final_standing'] <= 3
            
            for pick_num, player in team_picks:
                round_num = (pick_num - 1) // self.num_teams + 1
                position_by_round[round_num][player.position] += 1
                
                # Count pick success
                pick_success[(round_num, player.position)].append(1 if successful else 0)
        
        # Calculate success rates
        success_rates = {}
        for (round_num, position), results in pick_success.items():
            if results:
                success_rates[(round_num, position)] = sum(results) / len(results)
        
        # Top performer analysis
        if self.simulations:
            # Sort by final standing (best first)
            top_results = sorted(self.simulations, key=lambda x: x['results']['final_standing'])
            
            # Extract top 5 drafts
            top_drafts = []
            for i in range(min(5, len(top_results))):
                sim = top_results[i]
                draft_summary = {
                    'final_standing': sim['results']['final_standing'],
                    'win_pct': sim['results']['win_pct'],
                    'points_for': sim['results']['points_for'],
                    'picks': [(pick_num, player.name, player.position) for pick_num, player in sim['team_picks']]
                }
                top_drafts.append(draft_summary)
        
        return {
            'position_by_round': dict(position_by_round),
            'success_rates': success_rates,
            'top_drafts': top_drafts
        }
    
    def save_model(self, filename="rl_draft_model.pkl"):
        """
        Save the trained model to a file
        
        Parameters:
        -----------
        filename : str, optional
            Filename to save the model
        """
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'roster_limits': self.roster_limits,
            'scoring_settings': self.scoring_settings,
            'num_teams': self.num_teams,
            'draft_type': self.draft_type,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Model saved to {filename}")
    
    @classmethod
    def load_model(cls, filename="rl_draft_model.pkl", players=None):
        """
        Load a trained model from a file
        
        Parameters:
        -----------
        filename : str, optional
            Filename to load the model from
        players : list, optional
            List of Player objects
            
        Returns:
        --------
        FantasyRL
            Loaded model
        """
        with open(filename, 'rb') as f:
            model_data = pickle.load(f)
        
        # Create instance
        instance = cls(
            players=players or [],
            league_settings={
                'roster_limits': model_data['roster_limits'],
                'scoring_settings': model_data['scoring_settings']
            },
            num_teams=model_data['num_teams'],
            draft_type=model_data['draft_type']
        )
        
        # Load model components
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        
        logger.info(f"Model loaded from {filename} (saved on {model_data['timestamp']})")
        
        return instance
    
    def get_draft_recommendation(self, team, available_players, current_pick):
        """
        Get draft recommendation for a team
        
        Parameters:
        -----------
        team : Team
            Team making the pick
        available_players : list
            List of available players
        current_pick : int
            Current pick number
            
        Returns:
        --------
        list
            Sorted list of (player, value) tuples
        """
        # Get available players that can be drafted
        available = [p for p in available_players if not p.drafted]
        available = [p for p in available if team.can_draft_position(p.position)]
        
        if not available:
            return []
        
        # Get current state
        current_state = self._build_state_features(
            team,
            available_players,
            current_pick
        )
        
        # Evaluate each possible action
        player_values = []
        
        for player in available:
            # Simulate drafting this player
            team_copy = Team(
                team.name,
                team.draft_position,
                team.roster_limits,
                team.scoring_settings,
                team.strategy
            )
            
            # Copy current roster
            for position, players in team.roster.items():
                team_copy.roster[position] = players.copy()
            
            # Add player to roster
            team_copy.roster[player.position].append(player)
            
            # Get next state
            next_available = [p for p in available if p != player]
            
            next_state = self._build_state_features(
                team_copy,
                next_available,
                current_pick + 1
            )
            
            # Get predicted value
            scaled_state = self.scaler.transform(next_state)
            value = self.model.predict(scaled_state)[0]
            
            player_values.append((player, value))
        
        # Sort by value (descending)
        player_values.sort(key=lambda x: x[1], reverse=True)
        
        return player_values


class DraftAssistantGUI:
    """
    GUI for assisting with live drafts
    
    Note: This is a minimal implementation. For a more advanced GUI,
    consider using libraries like tkinter, PyQt, or Streamlit.
    """
    
    def __init__(self, rl_agent, all_players, num_teams=12, roster_limits=None, 
                 scoring_settings=None, draft_type='snake'):
        """
        Initialize the draft assistant
        
        Parameters:
        -----------
        rl_agent : FantasyRL
            Trained RL agent
        all_players : list
            List of all Player objects
        num_teams : int, optional
            Number of teams in the league
        roster_limits : dict, optional
            Roster limits by position
        scoring_settings : dict, optional
            Scoring settings
        draft_type : str, optional
            Type of draft ('snake', 'linear')
        """
        self.rl_agent = rl_agent
        self.all_players = all_players
        self.num_teams = num_teams
        
        # Default roster limits if not provided
        self.roster_limits = roster_limits or {
            'QB': 3,
            'RB': 6,
            'WR': 6,
            'TE': 3,
            'K': 1,
            'DST': 1
        }
        
        # Default scoring settings if not provided
        self.scoring_settings = scoring_settings or {
            'pass_yd': 0.04,  # 1 point per 25 yards
            'pass_td': 4,
            'interception': -2,
            'rush_yd': 0.1,   # 1 point per 10 yards
            'rush_td': 6,
            'rec': 0,         # 0 for standard, 0.5 for half PPR, 1 for PPR
            'rec_yd': 0.1,    # 1 point per 10 yards
            'rec_td': 6,
            'fumble_lost': -2
        }
        
        self.draft_type = draft_type
        
        # Initialize draft state
        self.my_team = None
        self.current_pick = 0
        self.available_players = all_players.copy()
        self.drafted_players = []
        
        # Generate draft order
        self.draft_order = self._generate_draft_order()
    
    def _generate_draft_order(self):
        """Generate draft order based on draft type"""
        draft_order = []
        total_roster_spots = sum(self.roster_limits.values())
        
        if self.draft_type == 'snake':
            for round_num in range(total_roster_spots):
                # Even rounds go in reverse order
                if round_num % 2 == 0:
                    for team_idx in range(self.num_teams):
                        draft_order.append(team_idx)
                else:
                    for team_idx in range(self.num_teams - 1, -1, -1):
                        draft_order.append(team_idx)
        else:  # Linear draft
            for round_num in range(total_roster_spots):
                for team_idx in range(self.num_teams):
                    draft_order.append(team_idx)
        
        return draft_order
    
    def initialize_draft(self, my_position):
        """
        Initialize the draft with my team at the specified position
        
        Parameters:
        -----------
        my_position : int
            My draft position (1-indexed)
        """
        # Create my team
        self.my_team = Team(
            name=f"My Team",
            draft_position=my_position,
            roster_limits=self.roster_limits,
            scoring_settings=self.scoring_settings
        )
        
        # Initialize draft state
        self.current_pick = 0
        self.available_players = self.all_players.copy()
        self.drafted_players = []
        
        # Reset player draft status
        for player in self.all_players:
            player.drafted = False
            player.draft_position = None
            player.drafted_by = None
        
        # My team index in the draft order
        self.my_team_idx = my_position - 1
        
        print(f"Draft initialized with your team drafting at position {my_position}")
        print(f"Available roster spots: {self.roster_limits}")
        
        # Show initial recommendations
        self.show_recommendations()
    
    def draft_player(self, player_name):
        """
        Record a drafted player
        
        Parameters:
        -----------
        player_name : str
            Name of the drafted player
            
        Returns:
        --------
        bool
            True if player was found and drafted, False otherwise
        """
        # Find player by name
        found_player = None
        for player in self.available_players:
            if player.name.lower() == player_name.lower():
                found_player = player
                break
        
        if not found_player:
            print(f"Player '{player_name}' not found in available players")
            return False
        
        # Determine team making the pick
        team_idx = self.draft_order[self.current_pick]
        
        # If it's my team, add to my roster
        if team_idx == self.my_team_idx:
            # Check if my team can draft this position
            if not self.my_team.can_draft_position(found_player.position):
                print(f"Your team cannot draft another {found_player.position}, roster full")
                return False
            
            # Add to my roster
            self.my_team.draft_player(found_player, self.current_pick + 1)
            print(f"You drafted {found_player.name} ({found_player.position})")
        else:
            # Just mark as drafted by another team
            found_player.drafted = True
            found_player.draft_position = self.current_pick + 1
            found_player.drafted_by = f"Team {team_idx + 1}"
            
            print(f"Team {team_idx + 1} drafted {found_player.name} ({found_player.position})")
        
        # Update draft state
        self.drafted_players.append(found_player)
        self.available_players = [p for p in self.available_players if p != found_player]
        self.current_pick += 1
        
        # Show updated recommendations if next pick is mine
        if self.current_pick < len(self.draft_order):
            next_team_idx = self.draft_order[self.current_pick]
            if next_team_idx == self.my_team_idx:
                print("\nYour pick is next!")
                self.show_recommendations()
            else:
                picks_until_mine = self._picks_until_my_turn()
                if picks_until_mine is not None:
                    print(f"\n{picks_until_mine} picks until your next selection")
        
        return True
    
    def _picks_until_my_turn(self):
        """Calculate picks until my next turn"""
        if self.current_pick >= len(self.draft_order):
            return None
        
        for i in range(self.current_pick, len(self.draft_order)):
            if self.draft_order[i] == self.my_team_idx:
                return i - self.current_pick
        
        return None
    
    def show_recommendations(self, top_n=10):
        """
        Show draft recommendations
        
        Parameters:
        -----------
        top_n : int, optional
            Number of recommendations to show
        """
        # Get recommendations from RL agent
        recommendations = self.rl_agent.get_draft_recommendation(
            self.my_team,
            self.available_players,
            self.current_pick
        )
        
        # Calculate need-based recommendations
        need_based = self._get_need_based_recommendations(top_n=top_n)
        
        # Show current roster
        print("\n=== Your Current Roster ===")
        for position, players in self.my_team.roster.items():
            player_names = [p.name for p in players]
            print(f"{position} ({len(players)}/{self.roster_limits[position]}): {', '.join(player_names) if player_names else 'None'}")
        
        # Show RL recommendations
        print("\n=== RL Agent Recommendations ===")
        for i, (player, value) in enumerate(recommendations[:top_n]):
            print(f"{i+1}. {player.name} ({player.position}) - Projected: {player.projected_points:.1f}, Value: {value:.4f}")
        
        # Show need-based recommendations
        print("\n=== Need-Based Recommendations ===")
        for i, player in enumerate(need_based):
            print(f"{i+1}. {player.name} ({player.position}) - Projected: {player.projected_points:.1f}")
        
        # Show current pick info
        if self.current_pick < len(self.draft_order):
            team_idx = self.draft_order[self.current_pick]
            round_num = self.current_pick // self.num_teams + 1
            pick_in_round = self.current_pick % self.num_teams + 1
            
            print(f"\nCurrent Pick: Round {round_num}, Pick {pick_in_round} (Overall: {self.current_pick + 1})")
            print(f"Team on the clock: {'Your Team' if team_idx == self.my_team_idx else f'Team {team_idx + 1}'}")
    
    def _get_need_based_recommendations(self, top_n=10):
        """
        Get need-based recommendations
        
        Parameters:
        -----------
        top_n : int, optional
            Number of recommendations to show
            
        Returns:
        --------
        list
            List of recommended players
        """
        # Calculate position needs
        needs = {}
        for position, limit in self.roster_limits.items():
            current = len(self.my_team.roster.get(position, []))
            needs[position] = limit - current
        
        # Sort positions by need (most needed first)
        sorted_needs = sorted(needs.items(), key=lambda x: (x[1], x[0]), reverse=True)
        
        # Get recommendations for each position
        recommendations = []
        
        for position, need in sorted_needs:
            if need <= 0:
                continue  # Skip positions we don't need
            
            # Get available players at this position
            pos_players = [p for p in self.available_players 
                          if p.position == position and not p.drafted]
            
            # Sort by projected points
            pos_players.sort(key=lambda p: p.projected_points, reverse=True)
            
            # Add top players to recommendations
            recommendations.extend(pos_players[:need])
        
        # Sort overall by projected points and return top N
        recommendations.sort(key=lambda p: p.projected_points, reverse=True)
        
        return recommendations[:top_n]
    
    def show_draft_board(self):
        """
        Display the current draft board
        """
        print("\n=== Draft Board ===")
        
        # Get all drafted players
        drafted = sorted(self.drafted_players, key=lambda p: p.draft_position)
        
        # Organize by round and pick
        rounds = {}
        for player in drafted:
            round_num = (player.draft_position - 1) // self.num_teams + 1
            pick_in_round = (player.draft_position - 1) % self.num_teams + 1
            
            if round_num not in rounds:
                rounds[round_num] = {}
            
            rounds[round_num][pick_in_round] = player
        
        # Display each round
        for round_num in sorted(rounds.keys()):
            print(f"\nRound {round_num}:")
            for pick in range(1, self.num_teams + 1):
                player = rounds[round_num].get(pick)
                if player:
                    team_name = "Your Team" if player.drafted_by == "My Team" else player.drafted_by
                    print(f"  Pick {pick}: {player.name} ({player.position}) - {team_name}")
                else:
                    print(f"  Pick {pick}: Not selected yet")
    
    def search_players(self, query, position=None):
        """
        Search for players by name
        
        Parameters:
        -----------
        query : str
            Search query
        position : str, optional
            Filter by position
            
        Returns:
        --------
        list
            List of matching players
        """
        query = query.lower()
        
        # Filter available players
        results = []
        for player in self.all_players:
            if query in player.name.lower():
                if position is None or player.position == position:
                    status = "Available"
                    if player.drafted:
                        status = f"Drafted by {player.drafted_by} at pick {player.draft_position}"
                    
                    results.append((player, status))
        
        # Print results
        print(f"\nSearch results for '{query}'{f' (Position: {position})' if position else ''}:")
        
        if not results:
            print("No matching players found")
        else:
            for i, (player, status) in enumerate(results):
                print(f"{i+1}. {player.name} ({player.position}) - Projected: {player.projected_points:.1f} - {status}")
        
        return results
    
    def run_command(self, command):
        """
        Run a user command
        
        Parameters:
        -----------
        command : str
            User command
            
        Returns:
        --------
        bool
            True to continue, False to exit
        """
        command = command.strip().lower()
        
        if command == 'exit' or command == 'quit':
            return False
        
        elif command == 'help':
            self._show_help()
        
        elif command.startswith('draft '):
            player_name = command[6:].strip()
            self.draft_player(player_name)
        
        elif command == 'rec' or command == 'recommendations':
            self.show_recommendations()
        
        elif command.startswith('rec '):
            try:
                top_n = int(command[4:].strip())
                self.show_recommendations(top_n=top_n)
            except ValueError:
                print("Invalid number. Usage: rec [number]")
        
        elif command == 'board' or command == 'draft board':
            self.show_draft_board()
        
        elif command.startswith('search '):
            query = command[7:].strip()
            
            # Check for position filter
            position = None
            if ':' in query:
                parts = query.split(':')
                query = parts[0].strip()
                position = parts[1].strip().upper()
            
            self.search_players(query, position)
        
        elif command == 'roster':
            print("\n=== Your Current Roster ===")
            for position, players in self.my_team.roster.items():
                player_names = [p.name for p in players]
                print(f"{position} ({len(players)}/{self.roster_limits[position]}): {', '.join(player_names) if player_names else 'None'}")
        
        elif command == 'next':
            # Find next player to be drafted
            team_idx = self.draft_order[self.current_pick]
            print(f"Team {team_idx + 1} is on the clock (Pick {self.current_pick + 1})")
            
            # If it's my turn, show recommendations
            if team_idx == self.my_team_idx:
                print("It's your pick!")
                self.show_recommendations()
        
        elif command.startswith('pick '):
            try:
                pick_num = int(command[5:].strip())
                if 1 <= pick_num <= len(self.draft_order):
                    team_idx = self.draft_order[pick_num - 1]
                    round_num = (pick_num - 1) // self.num_teams + 1
                    pick_in_round = (pick_num - 1) % self.num_teams + 1
                    
                    print(f"Pick {pick_num}: Round {round_num}, Pick {pick_in_round}")
                    print(f"Team on the clock: {'Your Team' if team_idx == self.my_team_idx else f'Team {team_idx + 1}'}")
                else:
                    print(f"Pick number must be between 1 and {len(self.draft_order)}")
            except ValueError:
                print("Invalid pick number. Usage: pick [number]")
        
        else:
            print(f"Unknown command: {command}")
            print("Type 'help' for a list of commands")
        
        return True
    
    def _show_help(self):
        """Show help message"""
        print("\n=== Draft Assistant Commands ===")
        print("draft [player name] - Draft a player")
        print("rec - Show recommendations")
        print("rec [number] - Show N recommendations")
        print("search [query] - Search for players")
        print("search [query]:[position] - Search for players by position")
        print("board - Show draft board")
        print("roster - Show your roster")
        print("next - Show who's on the clock")
        print("pick [number] - Show info about a specific pick")
        print("help - Show this help message")
        print("exit/quit - Exit the program")
    
    def run_interactive(self):
        """
        Run the draft assistant interactively
        """
        print("\n=== Fantasy Football Draft Assistant ===")
        
        # Get draft position
        while True:
            try:
                position = int(input(f"Enter your draft position (1-{self.num_teams}): "))
                if 1 <= position <= self.num_teams:
                    break
                print(f"Draft position must be between 1 and {self.num_teams}")
            except ValueError:
                print("Please enter a valid number")
        
        # Initialize draft
        self.initialize_draft(position)
        
        # Show help
        self._show_help()
        
        # Main command loop
        while True:
            command = input("\nEnter command: ")
            if not self.run_command(command):
                break
        
        print("Exiting draft assistant")
