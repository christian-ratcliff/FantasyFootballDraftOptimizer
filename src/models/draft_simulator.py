import random
import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)

class Player:
    """Represents a fantasy football player in the draft"""
    
    def __init__(self, name: str, position: str, team: str, projected_points: float, 
                 adp: float = None, risk: float = None, tier: str = None, 
                 projection_low: float = None, projection_high: float = None,
                 ceiling_projection: float = None, **kwargs):
        """
        Initialize a player
        
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
        adp : float, optional
            Average Draft Position
        risk : float, optional
            Risk level (0-1, higher = riskier)
        tier : str, optional
            Player tier (e.g., "Elite", "High Tier", etc.)
        projection_low : float, optional
            Lower bound of projection range
        projection_high : float, optional
            Upper bound of projection range
        ceiling_projection : float, optional
            Ceiling projection (high-end outcome)
        **kwargs : additional player attributes
        """
        self.name = name
        self.position = position
        self.team = team
        self.projected_points = projected_points
        self.adp = adp if adp is not None else float('inf')
        self.risk = risk if risk is not None else 0.5
        self.tier = tier if tier is not None else "Unknown"
        self.projection_low = projection_low if projection_low is not None else projected_points * 0.8
        self.projection_high = projection_high if projection_high is not None else projected_points * 1.2
        self.ceiling_projection = ceiling_projection if ceiling_projection is not None else projected_points * 1.4
        self.is_drafted = False
        self.drafted_round = None
        self.drafted_pick = None
        self.drafted_team = None
        
        # Store additional attributes
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def __str__(self):
        return f"{self.name} ({self.position}-{self.team}) - {self.projected_points:.1f} pts"
    
    def __repr__(self):
        return self.__str__()

class Team:
    """Represents a fantasy football team in the draft"""
    
    def __init__(self, name: str, draft_position: int, roster_limits: Dict[str, int], 
                 strategy: str = "VBD", strategy_params: Dict = None, league_size: int = 10,
                 scoring_settings: Dict = None):
        """
        Initialize a team
        
        Parameters:
        -----------
        name : str
            Team name
        draft_position : int
            Draft position (1-based)
        roster_limits : Dict[str, int]
            Maximum players by position (e.g., {"QB": 2, "RB": 5, ...})
        strategy : str, optional
            Draft strategy to use
        strategy_params : Dict, optional
            Parameters for the draft strategy
        league_size : int, optional
            Number of teams in the league
        scoring_settings : Dict, optional
            Scoring settings for the league
        """
        self.name = name
        self.draft_position = draft_position
        self.roster_limits = roster_limits
        self.strategy = strategy
        self.strategy_params = strategy_params or {}
        self.league_size = league_size
        self.scoring_settings = scoring_settings or {}
        
        # Initialize roster
        self.roster: List[Player] = []
        
        # Track roster by position
        self.roster_by_position = defaultdict(list)
        
        # Track draft picks by round
        self.draft_picks = {}
    
    def add_player(self, player: Player, round_num: int, pick_num: int) -> None:
        """
        Add a player to the team's roster
        
        Parameters:
        -----------
        player : Player
            Player to draft
        round_num : int
            Draft round
        pick_num : int
            Overall pick number
        """
        player.is_drafted = True
        player.drafted_round = round_num
        player.drafted_pick = pick_num
        player.drafted_team = self
        
        self.roster.append(player)
        
        # Add to position-specific roster
        position = player.position
        self.roster_by_position[position].append(player)
        
        self.draft_picks[round_num] = player
    
    def can_draft_position(self, position: str) -> bool:
        """
        Check if the team can draft a player at the specified position
        
        Parameters:
        -----------
        position : str
            Player position or roster slot
                
        Returns:
        --------
        bool
            True if the team can draft a player at the position, False otherwise
        """
        # Define position mappings for flexible slots
        flexible_slots = {
            "FLEX": ["RB", "WR", "TE"],
            "RB/WR": ["RB", "WR"],
            "WR/TE": ["WR", "TE"],
            "RB/WR/TE": ["RB", "WR", "TE"],
            "OP": ["QB", "RB", "WR", "TE"],
            "DP": ["DL", "LB", "DB", "DE", "DT", "CB", "S"],
            "BE": ["QB", "RB", "WR", "TE", "K", "DST", "DL", "LB", "DB", "DE", "DT", "CB", "S"],
            "IR": ["QB", "RB", "WR", "TE", "K", "DST", "DL", "LB", "DB", "DE", "DT", "CB", "S"],
        }
        
        # For standard positions, check direct limits
        if position in ["QB", "RB", "WR", "TE", "K", "DST", "DL", "LB", "DB"]:
            current_count = len(self.roster_by_position.get(position, []))
            max_count = self.roster_limits.get(position, 0)
            
            if current_count >= max_count:
                return False
            
            # Also check if we have room for more players overall
            current_total = sum(len(players) for pos, players in self.roster_by_position.items())
            max_total = sum(count for pos, count in self.roster_limits.items())
            
            return current_total < max_total
        
        # Handle special cases for bench and IR
        if position == "BE":
            # Check if total roster size is below maximum
            be_count = len(self.roster_by_position.get("BE", []))
            max_be = self.roster_limits.get("BE", 0)
            
            if be_count >= max_be:
                return False
            
            # Check if we have room for more players overall
            current_total = sum(len(players) for pos, players in self.roster_by_position.items())
            max_total = sum(count for pos, count in self.roster_limits.items() 
                        if pos not in ["IR", ""])
            
            return current_total < max_total
        
        if position == "IR":
            # Check if IR spots are available
            ir_count = len(self.roster_by_position.get("IR", []))
            max_ir = self.roster_limits.get("IR", 0)
            return ir_count < max_ir
        
        # Handle flexible position slots
        if position in flexible_slots:
            current_count = len(self.roster_by_position.get(position, []))
            max_count = self.roster_limits.get(position, 0)
            
            # Check if this flexible slot is already full
            if current_count >= max_count:
                return False
                
            # For flexible slots, check if any eligible positions can still be drafted
            for eligible_pos in flexible_slots[position]:
                # For each eligible position, check if we need more players at that position
                current_pos_count = len(self.roster_by_position.get(eligible_pos, []))
                max_pos_count = self.roster_limits.get(eligible_pos, float('inf'))
                
                # If we have room for more of this position, it's valid
                if current_pos_count < max_pos_count:
                    # Also check if we have room for more players overall
                    current_total = sum(len(players) for pos, players in self.roster_by_position.items())
                    max_total = sum(count for pos, count in self.roster_limits.items() 
                                if pos not in ["IR", ""])
                    
                    return current_total < max_total
                    
            # If all eligible positions are full, we can't draft for this slot
            return False
        
        # For unknown positions, assume we can draft if we haven't met the overall roster limit
        current_total = sum(len(players) for pos, players in self.roster_by_position.items())
        max_total = sum(count for pos, count in self.roster_limits.items() 
                    if pos not in ["IR", ""])
        
        return current_total < max_total
    
    def get_position_needs(self) -> Dict[str, int]:
        """
        Get the number of players needed by position
        
        Returns:
        --------
        Dict[str, int]
            Dictionary mapping positions to number of players needed
        """
        needs = {}
        
        for position, limit in self.roster_limits.items():
            current = len(self.roster_by_position[position])
            needs[position] = max(0, limit - current)
        
        return needs
    
    def get_total_projected_points(self) -> float:
        """
        Get total projected points for all players on the roster
        
        Returns:
        --------
        float
            Total projected points
        """
        return sum(player.projected_points for player in self.roster)
    
    def get_optimal_starters(self) -> Dict[str, List[Player]]:
        """
        Get the optimal starting lineup based on projected points
        
        Returns:
        --------
        Dict[str, List[Player]]
            Dictionary mapping positions to list of starting players
        """
        # Define starting lineup slots
        starter_limits = {
            "QB": 1,
            "RB": 2,
            "WR": 2,
            "TE": 1,
            "FLEX": 1,  # RB/WR/TE
            "K": 1,
            "DST": 1
        }
        
        # Allow customizing starter limits from scoring settings if available
        if self.scoring_settings and "starter_limits" in self.scoring_settings:
            starter_limits.update(self.scoring_settings["starter_limits"])
        
        starters = defaultdict(list)
        
        # First, fill required positions
        for position in ["QB", "RB", "WR", "TE", "K", "DST"]:
            if position in starter_limits and starter_limits[position] > 0:
                # Sort players by projected points
                sorted_players = sorted(
                    self.roster_by_position[position], 
                    key=lambda p: p.projected_points, 
                    reverse=True
                )
                
                # Add top N players as starters
                for i in range(min(starter_limits[position], len(sorted_players))):
                    starters[position].append(sorted_players[i])
        
        # Handle FLEX position (RB/WR/TE)
        if "FLEX" in starter_limits and starter_limits["FLEX"] > 0:
            # Get remaining players not already in starting lineup
            remaining_flex = []
            
            for position in ["RB", "WR", "TE"]:
                for player in self.roster_by_position[position]:
                    if player not in starters[position]:
                        remaining_flex.append(player)
            
            # Sort by projected points
            remaining_flex.sort(key=lambda p: p.projected_points, reverse=True)
            
            # Add top N players as FLEX starters
            for i in range(min(starter_limits["FLEX"], len(remaining_flex))):
                starters["FLEX"].append(remaining_flex[i])
        
        return starters
    
    def get_starting_lineup_points(self) -> float:
        """
        Get total projected points for the optimal starting lineup
        
        Returns:
        --------
        float
            Total projected points for starters
        """
        starters = self.get_optimal_starters()
        
        return sum(
            player.projected_points 
            for position in starters 
            for player in starters[position]
        )

class DraftSimulator:
    """Simulate a fantasy football draft with various strategies"""
    
    def __init__(self, players: List[Player], league_size: int = 10, 
                roster_limits: Dict[str, int] = None, num_rounds: int = 18,
                scoring_settings: Dict = None, user_pick: int = None, projection_models=None):
        """
        Initialize the draft simulator
        
        Parameters:
        -----------
        players : List[Player]
            List of available players
        league_size : int, optional
            Number of teams in the league
        roster_limits : Dict[str, int], optional
            Maximum players by position (e.g., {"QB": 2, "RB": 5, ...})
        num_rounds : int, optional
            Number of draft rounds
        scoring_settings : Dict, optional
            Scoring settings for the league
        user_pick : int, optional
            Draft position of the user (1-based, None for all AI)
        """
        self.players = players.copy()
        self.league_size = league_size
        self.num_rounds = num_rounds
        self.user_pick = user_pick
        
        # Default roster limits if none provided
        if roster_limits is None:
            self.roster_limits = {
                "QB": 2,
                "RB": 5,
                "WR": 5,
                "TE": 2,
                "K": 1,
                "DST": 1
            }
        else:
            self.roster_limits = roster_limits
        
        # Default scoring settings if none provided
        if scoring_settings is None:
            self.scoring_settings = {
                "passing_yards": 0.04,
                "passing_td": 4,
                "interception": -2,
                "rushing_yards": 0.1,
                "rushing_td": 6,
                "receiving_yards": 0.1,
                "receiving_td": 6,
                "reception": 0.5,
                "fumble_lost": -2
            }
        else:
            self.scoring_settings = scoring_settings
        
        # Initialize teams
        self.teams: List[Team] = []
        
        # Keep track of the draft
        self.current_round = 0
        self.current_pick = 0
        self.draft_history = []
        
        # Calculate baseline values for VBD
        self._calculate_baselines()
        
        # Initialize the teams with different strategies
        self._initialize_teams()
        
        self.projection_models = projection_models or {}
    
    def _calculate_baselines(self) -> None:
        """
        Calculate baseline values for Value-Based Drafting (VBD)
        """
        # Group players by position
        players_by_position = defaultdict(list)
        
        for player in self.players:
            players_by_position[player.position].append(player)
        
        # Sort by projected points
        for position in players_by_position:
            players_by_position[position].sort(key=lambda p: p.projected_points, reverse=True)
        
        # Define baseline indices by position (typically the last starter at each position)
        baseline_indices = {
            "QB": self.league_size,  # Last starting QB
            "RB": self.league_size * 2 + 2,  # Last starting RB including 2 FLEX
            "WR": self.league_size * 2 + 2,  # Last starting WR including 2 FLEX
            "TE": self.league_size,  # Last starting TE
            "K": self.league_size,  # Last starting K
            "DST": self.league_size  # Last starting DST
        }
        
        # Store baseline values
        self.baseline_values = {}
        
        for position, index in baseline_indices.items():
            pos_players = players_by_position[position]
            if index < len(pos_players):
                self.baseline_values[position] = pos_players[index].projected_points
            else:
                # If not enough players, use the last player's points
                self.baseline_values[position] = pos_players[-1].projected_points if pos_players else 0
        
        # Calculate VBD for each player
        for player in self.players:
            baseline = self.baseline_values.get(player.position, 0)
            player.vbd = player.projected_points - baseline
    
    def _initialize_teams(self) -> None:
        """
        Initialize the teams with different strategies
        """
        # Define the strategies to use
        strategies = [
            "VBD",  # Value-Based Drafting
            "ESPN",  # ESPN Autopick Algorithm
            "ZeroRB",  # 0RB Strategy
            "HeroRB",  # Hero RB Strategy
            "TwoRB",  # 2RB Strategy
            "BestAvailable",  # Simple best available
            "PPO"  # PPO (Proximal Policy Optimization) Strategy - will be handled specially
        ]
        
        # Create teams
        for i in range(self.league_size):
            # Assign draft position (1-indexed)
            draft_position = i + 1
            
            # Determine if this is the user's team
            is_user = (self.user_pick is not None and draft_position == self.user_pick)
            
            # Assign team name
            if is_user:
                team_name = "User Team"
                strategy = "User"  # User will make picks manually
            else:
                team_name = f"Team {draft_position}"
                # Cycle through strategies
                strategy = strategies[i % len(strategies)]
            
            # Create the team
            team = Team(
                name=team_name,
                draft_position=draft_position,
                roster_limits=self.roster_limits.copy(),
                strategy=strategy,
                league_size=self.league_size,
                scoring_settings=self.scoring_settings
            )
            
            self.teams.append(team)
    
    def run_draft(self, ppo_model=None) -> Tuple[List[Team], List[Dict]]:
        """
        Run the draft simulation with a proper snake draft order
        
        Parameters:
        -----------
        ppo_model : PPODrafter, optional
            PPO model to use for teams with PPO strategy
            
        Returns:
        --------
        Tuple[List[Team], List[Dict]]
            List of teams after the draft and draft history
        """
        logger.info(f"Starting draft simulation with {self.league_size} teams and {self.num_rounds} rounds")
        # Reset draft state
        self.current_round = 1
        self.current_pick = 1
        self.draft_history = []
        
        logger.info(f"Draft Simulator Details:")
        logger.info(f"League Size: {self.league_size}")
        
        for team in self.teams:
            logger.info(f"Team: {team.name}, Draft Position: {team.draft_position}, Strategy: {team.strategy}")
        
        # Reset all players' draft status
        for player in self.players:
            player.is_drafted = False
            player.drafted_round = None
            player.drafted_pick = None
            player.drafted_team = None
        
        # Run the draft
        for round_num in range(1, self.num_rounds + 1):
            self.current_round = round_num
            logger.info(f"Starting Round {round_num}")
            
            # Determine pick order (true snake draft)
            if round_num % 2 == 1:
                # Odd rounds: 1 to N (ascending)
                pick_order = list(range(self.league_size))
            else:
                # Even rounds: N to 1 (descending)
                pick_order = list(range(self.league_size - 1, -1, -1))
            
            # Make picks for this round
            for i, team_idx in enumerate(pick_order):
                team = self.teams[team_idx]
                overall_pick = (round_num - 1) * self.league_size + (team.draft_position if round_num % 2 == 1 else (self.league_size - team.draft_position + 1))

                self.current_pick = overall_pick
                logger.info(f"Round {round_num}, Team: {team.name}, Draft Position: {team.draft_position}, Overall Pick: {overall_pick}")
                # Make a pick
                picked_player = self._make_pick(team, round_num, overall_pick, ppo_model)
                
                if picked_player:
                    logger.info(f"Pick {overall_pick} (Round {round_num}.{i+1}): {team.name} selects {picked_player}")
                    
                    # Add to draft history
                    self.draft_history.append({
                        "round": round_num,
                        "pick": i + 1,
                        "overall_pick": overall_pick,
                        "team": team.name,
                        "player": picked_player.name,
                        "position": picked_player.position,
                        "projected_points": picked_player.projected_points
                    })
                else:
                    logger.warning(f"Team {team.name} could not make a valid pick!")
        
        logger.info("Draft completed")
        
        # Log team summaries
        for team in self.teams:
            total_points = team.get_total_projected_points()
            starting_points = team.get_starting_lineup_points()
            
            logger.info(f"{team.name} - Total: {total_points:.1f} pts, Starters: {starting_points:.1f} pts")
            
            # Log roster by position
            for position in sorted(team.roster_by_position.keys()):
                players = team.roster_by_position[position]
                logger.info(f"  {position}: {[player.name for player in players]}")
        
        return self.teams, self.draft_history
    
    def _make_pick(self, team: Team, round_num: int, overall_pick: int, ppo_model=None) -> Optional[Player]:
        """
        Make a draft pick for a team with improved position limit validation
        
        Parameters:
        -----------
        team : Team
            Team making the pick
        round_num : int
            Draft round
        overall_pick : int
            Overall pick number
        ppo_model : PPODrafter, optional
            PPO model to use for PPO strategy
                
        Returns:
        --------
        Optional[Player]
            The player picked, or None if no valid pick
        """
        # Get available players
        available_players = [p for p in self.players if not p.is_drafted]
        
        # Get valid positions and valid players
        valid_positions = [pos for pos in self.roster_limits.keys() if team.can_draft_position(pos)]
        
        # Determine valid player positions based on available roster slots
        valid_player_positions = set()
        
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
            "OP": ["QB", "RB", "WR", "TE"],
            "DL": ["DL", "DE", "DT"],
            "LB": ["LB"],
            "DB": ["DB", "CB", "S"],
            "DP": ["DL", "DE", "DT", "LB", "DB", "CB", "S"],
            "BE": ["QB", "RB", "WR", "TE", "K", "DST", "DL", "LB", "DB", "DE", "DT", "CB", "S"],
            "IR": ["QB", "RB", "WR", "TE", "K", "DST", "DL", "LB", "DB", "DE", "DT", "CB", "S"]
        }
        
        for slot in valid_positions:
            if slot in slot_to_position_map:
                valid_player_positions.update(slot_to_position_map[slot])
            else:
                # For any unrecognized slot type, assume it can hold any position
                valid_player_positions.update(["QB", "RB", "WR", "TE", "K", "DST"])
        
        # Check current position counts against limits
        position_counts = {
            "QB": len(team.roster_by_position.get("QB", [])),
            "RB": len(team.roster_by_position.get("RB", [])),
            "WR": len(team.roster_by_position.get("WR", [])),
            "TE": len(team.roster_by_position.get("TE", [])),
            "K": len(team.roster_by_position.get("K", [])), 
            "DST": len(team.roster_by_position.get("DST", []))
        }
        
        # Remove positions that have reached their limits
        for pos, count in position_counts.items():
            if pos in self.roster_limits and count >= self.roster_limits.get(pos, 0):
                if pos in valid_player_positions:
                    valid_player_positions.remove(pos)
        
        # Filter to valid players
        valid_players = [p for p in available_players if p.position in valid_player_positions]
        
        if not valid_players:
            logger.warning(f"Team {team.name} has no valid players to draft. Details:")
            logger.warning(f"  Valid positions: {valid_positions}")
            logger.warning(f"  Valid player positions: {valid_player_positions}")
            logger.warning(f"  Current roster: {[(pos, len(team.roster_by_position[pos])) for pos in sorted(team.roster_by_position.keys()) if len(team.roster_by_position[pos]) > 0]}")
            logger.warning(f"  Position counts: {position_counts}")
            logger.warning(f"  Available players by position: {[(pos, len([p for p in available_players if p.position == pos])) for pos in sorted(set([p.position for p in available_players]))]}")
            
            # Try to find a workaround - check if bench spots are available
            if "BE" in valid_positions:
                # Just pick best available player regardless of position as a bench player
                logger.info(f"Attempting to draft a bench player for {team.name}")
                if available_players:
                    # Sort by projected points
                    available_players.sort(key=lambda p: p.projected_points, reverse=True)
                    return available_players[0]
            
            return None
        
        # Check team's strategy
        if team.strategy == "User" and self.user_pick is not None:
            # User team - would prompt for input in an interactive setting
            # For simulation purposes, default to VBD
            player = self._pick_vbd(team, valid_players)
        
        elif team.strategy == "VBD":
            player = self._pick_vbd(team, valid_players)
        
        elif team.strategy == "ESPN":
            player = self._pick_espn(team, valid_players)
        
        elif team.strategy == "ZeroRB":
            player = self._pick_zero_rb(team, valid_players, round_num)
        
        elif team.strategy == "HeroRB":
            player = self._pick_hero_rb(team, valid_players, round_num)
        
        elif team.strategy == "TwoRB":
            player = self._pick_two_rb(team, valid_players, round_num)
        
        elif team.strategy == "BestAvailable":
            player = self._pick_best_available(team, valid_players)
        
        elif team.strategy == "PPO":
            # Use PPO model if available
            if ppo_model is not None:
                try:
                    # Import locally to avoid circular imports
                    from src.models.ppo_drafter import DraftState
                    
                    # Create state with the already filtered valid players
                    state = DraftState(
                        team=team,
                        available_players=valid_players,  # Use pre-filtered valid players
                        round_num=round_num,
                        overall_pick=overall_pick,
                        league_size=self.league_size,
                        roster_limits=self.roster_limits,
                        max_rounds=self.num_rounds,
                    )
                    
                    # Get action from PPO model
                    action, _, _ = ppo_model.select_action(state, training=False)
                    
                    # Return the selected player if valid
                    if action is not None and action < len(state.valid_players):
                        return state.valid_players[action]
                    
                except ImportError as e:
                    logger.warning(f"Couldn't import DraftState: {e}")
                    logger.warning("Falling back to VBD strategy")
                
                # Fall back to VBD if model returns invalid action or DraftState import fails
                player = self._pick_vbd(team, valid_players)
            else:
                # Fallback to VBD if no PPO model
                logger.warning(f"No PPO model provided for team {team.name}, falling back to VBD")
                player = self._pick_vbd(team, valid_players)
        else:
            # Default to VBD
            player = self._pick_vbd(team, valid_players)
        
        # If strategy couldn't pick a player, try best available as fallback
        if not player and valid_players:
            logger.warning(f"Strategy {team.strategy} couldn't find a valid pick, falling back to best available.")
            player = self._pick_best_available(team, valid_players)
        
        # Final validation check
        if player and not team.can_draft_position(player.position):
            logger.warning(f"Player {player.name} ({player.position}) selected by strategy {team.strategy} would violate position limits for team {team.name}")
            # Find a player at a position we can draft
            for pos in valid_positions:
                pos_players = [p for p in valid_players if p.position == pos]
                if pos_players:
                    player = max(pos_players, key=lambda p: p.projected_points)
                    logger.info(f"Replacing with {player.name} ({player.position})")
                    break
        
        # Add player to team
        if player:
            team.add_player(player, round_num, overall_pick)
        else:
            logger.warning(f"Team {team.name} could not make a valid pick!")
        
        return player

    def _pick_vbd(self, team: Team, available_players: List[Player]) -> Optional[Player]:
        """
        Select a player using Value-Based Drafting
        
        Parameters:
        -----------
        team : Team
            Team making the pick
        available_players : List[Player]
            List of available players
            
        Returns:
        --------
        Optional[Player]
            Selected player, or None if no valid pick
        """
        # Filter to positions we need
        valid_positions = [pos for pos in self.roster_limits.keys() if team.can_draft_position(pos)]
        valid_players = [p for p in available_players if p.position in valid_positions]
        
        if not valid_players:
            return None
        
        # Sort by VBD
        valid_players.sort(key=lambda p: getattr(p, 'vbd', p.projected_points), reverse=True)
        
        if len(valid_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
            return random.choice(valid_players[:3])
        
        # Check if we're extremely weak at a position and need to prioritize it
        needs = team.get_position_needs()
        
        # Calculate draft progress (0 to 1)
        draft_progress = (self.current_round - 1) / self.num_rounds
        
        # If we're in the later rounds and missing starters at a position, prioritize that position
        if draft_progress > 0.5:
            critical_positions = [pos for pos, need in needs.items() if need > 0 and len(team.roster_by_position[pos]) == 0]
            
            if critical_positions:
                # Try to fill critical positions first
                for pos in critical_positions:
                    pos_players = [p for p in valid_players if p.position == pos]
                    if pos_players:
                        return pos_players[0]
        
        # Use VBD by default
        return valid_players[0]
    
    def _pick_espn(self, team: Team, available_players: List[Player]) -> Optional[Player]:
        """
        Select a player using a simplified version of ESPN's autopick algorithm
        
        Parameters:
        -----------
        team : Team
            Team making the pick
        available_players : List[Player]
            List of available players
            
        Returns:
        --------
        Optional[Player]
            Selected player, or None if no valid pick
        """
        # Filter to positions we need
        valid_positions = [pos for pos in self.roster_limits.keys() if team.can_draft_position(pos)]
        valid_players = [p for p in available_players if p.position in valid_positions]
        
        if not valid_players:
            return None
        
        # ESPN algorithm prioritizes filling starting lineup first
        # It considers ADP, projections, and positional needs
        
        # Get position needs for starting lineup
        starter_needs = {
            "QB": 1 - len(team.roster_by_position["QB"]),
            "RB": 2 - len(team.roster_by_position["RB"]),
            "WR": 2 - len(team.roster_by_position["WR"]),
            "TE": 1 - len(team.roster_by_position["TE"]),
            "K": 1 - len(team.roster_by_position["K"]),
            "DST": 1 - len(team.roster_by_position["DST"])
        }
        
        # Filter to only needed starter positions
        needed_positions = [pos for pos, need in starter_needs.items() if need > 0]
        
        # If we still need starters, prioritize those positions
        if needed_positions:
            starter_players = [p for p in valid_players if p.position in needed_positions]
            
            if starter_players:
                # Sort by a combination of ADP and projections
                for player in starter_players:
                    # ESPN uses a proprietary formula, but we'll approximate with this:
                    player.espn_score = player.projected_points / 10 - (player.adp / 200)
                
                starter_players.sort(key=lambda p: getattr(p, 'espn_score', 0), reverse=True)
                return starter_players[0]
        
        # If starters are filled, pick best player available by projections
        # but with some bias towards positions with fewer backups
        for player in valid_players:
            position_count = len(team.roster_by_position[player.position])
            # Slight penalty for positions we already have a lot of players at
            player.espn_score = player.projected_points * (1 - position_count * 0.05)
        
        valid_players.sort(key=lambda p: getattr(p, 'espn_score', 0), reverse=True)
        if len(valid_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
            return random.choice(valid_players[:3])
        return valid_players[0]
    
    def _pick_zero_rb(self, team: Team, available_players: List[Player], round_num: int) -> Optional[Player]:
        """
        Select a player using Zero RB strategy
        
        Parameters:
        -----------
        team : Team
            Team making the pick
        available_players : List[Player]
            List of available players
        round_num : int
            Current draft round
            
        Returns:
        --------
        Optional[Player]
            Selected player, or None if no valid pick
        """
        # Filter to positions we need
        valid_positions = [pos for pos in self.roster_limits.keys() if team.can_draft_position(pos)]
        valid_players = [p for p in available_players if p.position in valid_positions]
        
        if not valid_players:
            return None
        
        # Zero RB strategy: avoid RBs in early rounds (1-4), then aggressively target them later
        if round_num <= 4:
            # Prioritize WR, TE, QB in early rounds
            early_positions = ["WR", "TE", "QB"]
            early_players = [p for p in valid_players if p.position in early_positions]
            
            if early_players:
                # Sort by VBD
                early_players.sort(key=lambda p: getattr(p, 'vbd', p.projected_points), reverse=True)
                if len(early_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
                    return random.choice(early_players[:3])
                return early_players[0]
        
        elif round_num <= 7:
            # Rounds 5-7: Start targeting high-upside RBs, but still prefer WR
            rb_count = len(team.roster_by_position["RB"])
            
            if rb_count == 0:
                # Need at least one RB
                rb_players = [p for p in valid_players if p.position == "RB"]
                
                if rb_players:
                    # Sort by ceiling projection to get high-upside RBs
                    rb_players.sort(key=lambda p: getattr(p, 'ceiling_projection', p.projected_points * 1.4), reverse=True)
                    if len(rb_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
                        return random.choice(rb_players[:3])
                    return rb_players[0]
            
            # Otherwise, continue with WR focus
            wr_players = [p for p in valid_players if p.position == "WR"]
            
            if wr_players:
                wr_players.sort(key=lambda p: getattr(p, 'vbd', p.projected_points), reverse=True)
                if len(wr_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
                    return random.choice(wr_players[:3])
                return wr_players[0]
        
        # After round 7, aggressively target RBs
        elif round_num <= 11 and team.can_draft_position("RB"):
            rb_players = [p for p in valid_players if p.position == "RB"]
            
            if rb_players:
                # Focus on high-upside RBs
                rb_players.sort(key=lambda p: getattr(p, 'ceiling_projection', p.projected_points * 1.4), reverse=True)
                if len(rb_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
                    return random.choice(rb_players[:3])
                return rb_players[0]
        
        # Default to VBD for other rounds
        valid_players.sort(key=lambda p: getattr(p, 'vbd', p.projected_points), reverse=True)
        if len(valid_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
            return random.choice(valid_players[:3])
        return valid_players[0]
    
    def _pick_hero_rb(self, team: Team, available_players: List[Player], round_num: int) -> Optional[Player]:
        """
        Select a player using Hero RB strategy
        
        Parameters:
        -----------
        team : Team
            Team making the pick
        available_players : List[Player]
            List of available players
        round_num : int
            Current draft round
            
        Returns:
        --------
        Optional[Player]
            Selected player, or None if no valid pick
        """
        # Filter to positions we need
        valid_positions = [pos for pos in self.roster_limits.keys() if team.can_draft_position(pos)]
        valid_players = [p for p in available_players if p.position in valid_positions]
        
        if not valid_players:
            return None
        
        # Hero RB strategy: One elite RB in round 1-2, then wait until later
        if round_num <= 2:
            # If we don't have an RB yet, prioritize getting one elite RB
            rb_count = len(team.roster_by_position["RB"])
            
            if rb_count == 0:
                rb_players = [p for p in valid_players if p.position == "RB"]
                
                if rb_players:
                    # Sort by VBD for the one hero RB
                    rb_players.sort(key=lambda p: getattr(p, 'vbd', p.projected_points), reverse=True)
                    if len(rb_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
                        return random.choice(rb_players[:3])
                    return rb_players[0]
            
            # If we already have our hero RB, focus on WR/TE
            early_positions = ["WR", "TE"]
            early_players = [p for p in valid_players if p.position in early_positions]
            
            if early_players:
                early_players.sort(key=lambda p: getattr(p, 'vbd', p.projected_points), reverse=True)
                if len(early_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
                    return random.choice(early_players[:3])
                return early_players[0]
        
        elif round_num <= 5:
            # Rounds 3-5: Focus on WR/TE/QB
            middle_positions = ["WR", "TE", "QB"]
            middle_players = [p for p in valid_players if p.position in middle_positions]
            
            if middle_players:
                middle_players.sort(key=lambda p: getattr(p, 'vbd', p.projected_points), reverse=True)
                if len(middle_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
                    return random.choice(middle_players[:3])
                return middle_players[0]
        
        # After round 5, start looking for RB2
        elif round_num <= 9 and team.can_draft_position("RB"):
            rb_count = len(team.roster_by_position["RB"])
            
            if rb_count <= 1:  # We need an RB2
                rb_players = [p for p in valid_players if p.position == "RB"]
                
                if rb_players:
                    rb_players.sort(key=lambda p: getattr(p, 'vbd', p.projected_points), reverse=True)
                    if len(rb_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
                        return random.choice(rb_players[:3])
                    return rb_players[0]
        
        # Default to VBD for other rounds
        valid_players.sort(key=lambda p: getattr(p, 'vbd', p.projected_points), reverse=True)
        if len(valid_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
            return random.choice(valid_players[:3])
        return valid_players[0]
    
    def _pick_two_rb(self, team: Team, available_players: List[Player], round_num: int) -> Optional[Player]:
        """
        Select a player using 2RB strategy
        
        Parameters:
        -----------
        team : Team
            Team making the pick
        available_players : List[Player]
            List of available players
        round_num : int
            Current draft round
            
        Returns:
        --------
        Optional[Player]
            Selected player, or None if no valid pick
        """
        # Filter to positions we need
        valid_positions = [pos for pos in self.roster_limits.keys() if team.can_draft_position(pos)]
        valid_players = [p for p in available_players if p.position in valid_positions]
        
        if not valid_players:
            return None
        
        # 2RB strategy: First 2 RBs in rounds 1-3, then pivot to WR
        if round_num <= 3:
            # If we don't have enough RBs yet, prioritize getting them
            rb_count = len(team.roster_by_position["RB"])
            
            if rb_count < 2:
                rb_players = [p for p in valid_players if p.position == "RB"]
                
                if rb_players:
                    rb_players.sort(key=lambda p: getattr(p, 'vbd', p.projected_points), reverse=True)
                    if len(rb_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
                        return random.choice(rb_players[:3])
                    return rb_players[0]
            
            # If we already have our 2 RBs, look for WR/TE
            early_positions = ["WR", "TE"]
            early_players = [p for p in valid_players if p.position in early_positions]
            
            if early_players:
                early_players.sort(key=lambda p: getattr(p, 'vbd', p.projected_points), reverse=True)
                if len(early_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
                    return random.choice(early_players[:3])
                return early_players[0]
        
        elif round_num <= 7:
            # Rounds 4-7: Focus on WR/TE/QB
            middle_positions = ["WR", "TE", "QB"]
            middle_players = [p for p in valid_players if p.position in middle_positions]
            
            if middle_players:
                middle_players.sort(key=lambda p: getattr(p, 'vbd', p.projected_points), reverse=True)
                if len(middle_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
                    return random.choice(middle_players[:3])
                return middle_players[0]
        
        # Default to VBD for other rounds
        valid_players.sort(key=lambda p: getattr(p, 'vbd', p.projected_points), reverse=True)
        if len(valid_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
            return random.choice(valid_players[:3])
        return valid_players[0]
    
    def _pick_best_available(self, team: Team, available_players: List[Player]) -> Optional[Player]:
        """
        Select the best available player by projected points
        
        Parameters:
        -----------
        team : Team
            Team making the pick
        available_players : List[Player]
            List of available players
            
        Returns:
        --------
        Optional[Player]
            Selected player, or None if no valid pick
        """
        # Filter to positions we need
        valid_positions = [pos for pos in self.roster_limits.keys() if team.can_draft_position(pos)]
        valid_players = [p for p in available_players if p.position in valid_positions]
        
        if not valid_players:
            return None
        
        # Sort by projected points
        valid_players.sort(key=lambda p: p.projected_points, reverse=True)
        if len(valid_players) >= 3 and random.random() < 0.2:  # 20% chance to pick randomly from top 3
            return random.choice(valid_players[:3])
        
        return valid_players[0]
    
    def _pick_ppo(self, team: Team, available_players: List[Player], round_num: int, overall_pick: int, ppo_model) -> Optional[Player]:
        """
        Select a player using the PPO model
        
        Parameters:
        -----------
        team : Team
            Team making the pick
        available_players : List[Player]
            List of available players
        round_num : int
            Current draft round
        overall_pick : int
            Overall pick number
        ppo_model : PPODrafter
            PPO model to use for selection
            
        Returns:
        --------
        Optional[Player]
            Selected player, or None if no valid pick
        """
        # Import DraftState to avoid circular imports
        from src.models.ppo_drafter import DraftState
        
        # Filter to positions we need
        valid_positions = [pos for pos in self.roster_limits.keys() if team.can_draft_position(pos)]
        valid_players = [p for p in available_players if p.position in valid_positions]
        
        if not valid_players:
            return None
            
        # Create a state for the PPO model
        state = DraftState(
            team=team,
            available_players=available_players,
            round_num=round_num,
            overall_pick=overall_pick,
            league_size=self.league_size,
            roster_limits=self.roster_limits,
            max_rounds=self.num_rounds,
        )
        
        # Get action from PPO model
        action, _, _ = ppo_model.select_action(state, training=False)
        
        # Return the selected player if valid
        if action is not None and action < len(state.valid_players):
            return state.valid_players[action]
        
        # Fall back to VBD if model returns invalid action
        logger.warning(f"PPO model returned invalid action: {action}, falling back to VBD")
        return self._pick_vbd(team, available_players)
    
    def create_draft_report(self, output_path: str = None) -> pd.DataFrame:
        """
        Create a report of the draft results
        
        Parameters:
        -----------
        output_path : str, optional
            Path to save the report CSV
            
        Returns:
        --------
        pd.DataFrame
            Draft results as a DataFrame
        """
        if not self.draft_history:
            logger.warning("No draft history. Run the draft first.")
            return pd.DataFrame()
        
        # Create DataFrame from draft history
        df = pd.DataFrame(self.draft_history)
        
        # Add team strategy
        df["team_strategy"] = df["team"].map({team.name: team.strategy for team in self.teams})
        
        # Get team results
        team_results = []
        for team in self.teams:
            team_results.append({
                "team": team.name,
                "strategy": team.strategy,
                "total_projected_points": team.get_total_projected_points(),
                "starting_lineup_points": team.get_starting_lineup_points(),
                "qb_count": len(team.roster_by_position["QB"]),
                "rb_count": len(team.roster_by_position["RB"]),
                "wr_count": len(team.roster_by_position["WR"]),
                "te_count": len(team.roster_by_position["TE"]),
                "k_count": len(team.roster_by_position["K"]),
                "dst_count": len(team.roster_by_position["DST"])
            })
        
        team_df = pd.DataFrame(team_results)
        
        # Save to CSV if output path provided
        if output_path:
            df.to_csv(output_path, index=False)
            team_df.to_csv(output_path.replace(".csv", "_teams.csv"), index=False)
            logger.info(f"Draft report saved to {output_path}")
        
        return df, team_df
    
    @staticmethod
    def load_players_from_projections(projections: Dict[str, pd.DataFrame], vbd_baseline: Dict = None) -> List[Player]:
        """
        Load players from projection DataFrames
        
        Parameters:
        -----------
        projections : Dict[str, pd.DataFrame]
            Dictionary of position-specific projection DataFrames
        vbd_baseline : Dict, optional
            Dictionary of VBD baseline values by position
            
        Returns:
        --------
        List[Player]
            List of Player objects
        """
        players = []
        
        # Process each position
        for position_key, df in projections.items():
            if df.empty:
                continue
            
            # Get position from key (e.g., "qb" -> "QB")
            position = position_key.upper()
            
            # Ensure required columns are present
            required_cols = ["name", "projected_points"]
            if not all(col in df.columns for col in required_cols):
                logger.warning(f"Missing required columns in {position} projections. Skipping.")
                continue
            
            # Process each player
            for _, row in df.iterrows():
                # Get required values with defaults
                name = row["name"]
                projected_points = row["projected_points"] if not pd.isna(row["projected_points"]) else 0
                
                # Get optional values with defaults
                team = row.get("team", "Unknown")
                projection_low = row.get("projection_low", projected_points * 0.8)
                projection_high = row.get("projection_high", projected_points * 1.2)
                ceiling_projection = row.get("ceiling_projection", projected_points * 1.4)
                tier = row.get("projection_tier", "Unknown")
                
                # Create Player object
                player = Player(
                    name=name,
                    position=position,
                    team=team,
                    projected_points=projected_points,
                    projection_low=projection_low,
                    projection_high=projection_high,
                    ceiling_projection=ceiling_projection,
                    tier=tier,
                    # Store additional attributes
                    **{k: v for k, v in row.items() if k not in ["name", "position", "team", "projected_points", 
                                                                "projection_low", "projection_high", 
                                                                "ceiling_projection", "tier"]}
                )
                
                # Calculate VBD if baseline provided
                if vbd_baseline and position in vbd_baseline:
                    player.vbd = projected_points - vbd_baseline[position]
                
                players.append(player)
        
        return players