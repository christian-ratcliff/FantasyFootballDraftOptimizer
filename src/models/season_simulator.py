"""
Season Simulator for Fantasy Football

This module simulates an entire fantasy football season using teams from a draft simulation.
It evaluates team performance and calculates the final standings.
"""
import numpy as np
import pandas as pd
import logging
import random
from typing import List, Dict, Tuple, Optional, Any, Union
from dataclasses import dataclass
from collections import defaultdict
import copy

# Import the Team class from draft_simulator
from .draft_simulator import Team, Player

# Set up logging
logger = logging.getLogger(__name__)

class FantasyGame:
    """Represents a weekly fantasy matchup between two teams"""
    
    def __init__(self, home_team: Team, away_team: Team, week: int):
        """
        Initialize a fantasy matchup
        
        Parameters:
        -----------
        home_team : Team
            Home team in the matchup
        away_team : Team
            Away team in the matchup
        week : int
            Week number
        """
        self.home_team = home_team
        self.away_team = away_team
        self.week = week
        self.home_score = 0
        self.away_score = 0
        self.winner = None
        self.loser = None
        self.is_tie = False
    
    def simulate(self, player_performances: Dict[str, float], randomness: float = 0.2) -> None:
        """
        Simulate the matchup
        
        Parameters:
        -----------
        player_performances : Dict[str, float]
            Dictionary mapping player names to their score for this week
        randomness : float, optional
            Amount of randomness to apply (0.0 = deterministic, higher = more random)
        """
        # Simulate home team score
        self.home_score = self._simulate_team_score(self.home_team, player_performances, randomness)
        
        # Simulate away team score
        self.away_score = self._simulate_team_score(self.away_team, player_performances, randomness)
        
        # Determine winner
        if self.home_score > self.away_score:
            self.winner = self.home_team
            self.loser = self.away_team
            self.is_tie = False
        elif self.away_score > self.home_score:
            self.winner = self.away_team
            self.loser = self.home_team
            self.is_tie = False
        else:
            self.is_tie = True
            self.winner = None
            self.loser = None
    
    def _simulate_team_score(self, team: Team, player_performances: Dict[str, float], randomness: float) -> float:
        """
        Simulate a team's score for the week
        
        Parameters:
        -----------
        team : Team
            Team to simulate
        player_performances : Dict[str, float]
            Dictionary mapping player names to their score for this week
        randomness : float
            Amount of randomness to apply
            
        Returns:
        --------
        float
            Simulated team score
        """
        # Get optimal starters
        starters = team.get_optimal_starters()
        
        # Calculate total score from starters
        total_score = 0.0
        
        for position, players in starters.items():
            for player in players:
                # Get base score from player performance
                base_score = player_performances.get(player.name, player.projected_points / 17)
                
                # Apply randomness
                if randomness > 0:
                    # Calculate standard deviation based on player's projection range
                    std_dev = ((player.projection_high - player.projection_low) / 17) / 2
                    std_dev = max(std_dev, base_score * randomness)  # Ensure minimum randomness
                    
                    # Generate random score
                    player_score = random.gauss(base_score, std_dev)
                    
                    # Ensure score is non-negative
                    player_score = max(0, player_score)
                else:
                    player_score = base_score
                
                total_score += player_score
        
        return total_score

class SeasonSimulator:
    """Simulates a complete fantasy football season"""
    
    def __init__(self, teams: List[Team], num_regular_weeks: int = 14, 
                 num_playoff_teams: int = 6, num_playoff_weeks: int = 3,
                 randomness: float = 0.2):
        """
        Initialize the season simulator
        
        Parameters:
        -----------
        teams : List[Team]
            List of fantasy teams
        num_regular_weeks : int, optional
            Number of regular season weeks
        num_playoff_teams : int, optional
            Number of teams that make the playoffs
        num_playoff_weeks : int, optional
            Number of playoff weeks
        randomness : float, optional
            Amount of randomness in weekly scoring (0.0 = deterministic, higher = more random)
        """
        self.teams = teams
        self.num_teams = len(teams)
        self.num_regular_weeks = num_regular_weeks
        self.num_playoff_teams = num_playoff_teams
        self.num_playoff_weeks = num_playoff_weeks
        self.randomness = randomness
        
        # Season variables
        self.current_week = 0
        self.schedule = []
        self.weekly_results = []
        self.player_performances = {}
        
        # Team records and standings
        self.records = {
            team.name: {"wins": 0, "losses": 0, "ties": 0, "points_for": 0, "points_against": 0}
            for team in teams
        }
        
        # Generate season schedule
        self._generate_schedule()
    
    def _generate_schedule(self) -> None:
        """
        Generate a complete season schedule
        """
        schedule = []
        
        # Only implemented for even number of teams currently
        if self.num_teams % 2 != 0:
            raise ValueError("Number of teams must be even")
        
        # Use a "circle" scheduling algorithm
        # This is a common algorithm for round-robin tournaments
        team_indices = list(range(self.num_teams))
        
        for week in range(self.num_regular_weeks):
            week_matchups = []
            
            # Create this week's matchups
            for i in range(self.num_teams // 2):
                home_idx = team_indices[i]
                away_idx = team_indices[self.num_teams - 1 - i]
                
                # Alternate home/away each week for fairness
                if week % 2 == 0:
                    matchup = (home_idx, away_idx)
                else:
                    matchup = (away_idx, home_idx)
                
                week_matchups.append(matchup)
            
            schedule.append(week_matchups)
            
            # Rotate teams (keep first team fixed, rotate others)
            team_indices = [team_indices[0]] + [team_indices[-1]] + team_indices[1:-1]
        
        self.schedule = schedule
    
    def _generate_player_performances(self) -> None:
        """
        Generate weekly performance data for all players
        """
        player_performances = {}
        
        # Collect all players from all teams
        all_players = []
        for team in self.teams:
            all_players.extend(team.roster)
        
        # Generate weekly performances for each player
        for week in range(1, self.num_regular_weeks + self.num_playoff_weeks + 1):
            week_performances = {}
            
            for player in all_players:
                # Base weekly score = season projection / games
                base_score = player.projected_points / 17  # Assuming 17 NFL games
                
                # Apply natural week-to-week variance
                std_dev = (player.projection_high - player.projection_low) / 17
                std_dev = max(std_dev, base_score * 0.3)  # Ensure minimum variance
                
                # Generate performance
                performance = random.gauss(base_score, std_dev)
                performance = max(0, performance)  # No negative scores
                
                # Store performance
                week_performances[player.name] = performance
            
            player_performances[week] = week_performances
        
        self.player_performances = player_performances
    
    def simulate_season(self) -> Dict[str, Any]:
        """
        Simulate the entire fantasy season
        
        Returns:
        --------
        Dict[str, Any]
            Season results including standings, playoff outcomes, and team stats
        """
        # Reset simulation state
        self.current_week = 0
        self.weekly_results = []
        
        # Reset team records
        self.records = {
            team.name: {"wins": 0, "losses": 0, "ties": 0, "points_for": 0, "points_against": 0}
            for team in self.teams
        }
        
        # Generate player performances for entire season
        self._generate_player_performances()
        
        # Simulate regular season
        logger.info("Simulating regular season...")
        regular_season_results = self.simulate_regular_season()
        
        # Get playoff teams
        playoff_teams = self._determine_playoff_teams()
        logger.info(f"Playoff teams: {[team.name for team in playoff_teams]}")
        
        # Simulate playoffs
        logger.info("Simulating playoffs...")
        playoff_results = self.simulate_playoffs(playoff_teams)
        
        # Combine results
        season_results = {
            "regular_season": regular_season_results,
            "playoffs": playoff_results,
            "champion": playoff_results.get("champion"),
            "runner_up": playoff_results.get("runner_up"),
            "standings": self._get_final_standings(),
            "team_records": self.records,
        }
        
        return season_results
    
    def simulate_regular_season(self) -> Dict[str, Any]:
        """
        Simulate the regular season
        
        Returns:
        --------
        Dict[str, Any]
            Regular season results
        """
        weekly_results = []
        
        # Simulate each week
        for week in range(1, self.num_regular_weeks + 1):
            self.current_week = week
            logger.info(f"Simulating Week {week}...")
            
            # Get matchups for this week
            matchups = self.schedule[week - 1]
            
            # Simulate each matchup
            week_results = []
            for home_idx, away_idx in matchups:
                home_team = self.teams[home_idx]
                away_team = self.teams[away_idx]
                
                # Create game
                game = FantasyGame(home_team, away_team, week)
                
                # Simulate game
                game.simulate(self.player_performances[week], self.randomness)
                
                # Update records
                self._update_records(game)
                
                # Store results
                week_results.append({
                    "home_team": home_team.name,
                    "away_team": away_team.name,
                    "home_score": game.home_score,
                    "away_score": game.away_score,
                    "winner": game.winner.name if game.winner else None,
                    "loser": game.loser.name if game.loser else None,
                    "is_tie": game.is_tie
                })
            
            weekly_results.append({
                "week": week,
                "matchups": week_results
            })
        
        # Calculate current standings
        standings = self._calculate_standings()
        
        regular_season_results = {
            "weekly_results": weekly_results,
            "standings": standings
        }
        
        return regular_season_results
    
    def _update_records(self, game: FantasyGame) -> None:
        """
        Update team records after a game
        
        Parameters:
        -----------
        game : FantasyGame
            Completed game
        """
        home_team = game.home_team
        away_team = game.away_team
        
        # Update points
        self.records[home_team.name]["points_for"] += game.home_score
        self.records[home_team.name]["points_against"] += game.away_score
        self.records[away_team.name]["points_for"] += game.away_score
        self.records[away_team.name]["points_against"] += game.home_score
        
        # Update win/loss/tie records
        if game.is_tie:
            self.records[home_team.name]["ties"] += 1
            self.records[away_team.name]["ties"] += 1
        else:
            winner = game.winner
            loser = game.loser
            
            self.records[winner.name]["wins"] += 1
            self.records[loser.name]["losses"] += 1
    
    def _calculate_standings(self) -> List[Dict[str, Any]]:
        """
        Calculate current standings
        
        Returns:
        --------
        List[Dict[str, Any]]
            Standings as a sorted list of team records
        """
        standings = []
        
        for team in self.teams:
            record = self.records[team.name]
            total_games = record["wins"] + record["losses"] + record["ties"]
            
            # Calculate win percentage
            if total_games > 0:
                win_pct = (record["wins"] + 0.5 * record["ties"]) / total_games
            else:
                win_pct = 0.0
            
            # Add to standings
            standings.append({
                "team": team.name,
                "strategy": team.strategy,
                "wins": record["wins"],
                "losses": record["losses"],
                "ties": record["ties"],
                "win_pct": win_pct,
                "points_for": record["points_for"],
                "points_against": record["points_against"],
                "point_differential": record["points_for"] - record["points_against"]
            })
        
        # Sort by win percentage, then points for
        standings.sort(key=lambda x: (x["win_pct"], x["points_for"]), reverse=True)
        
        # Add rank
        for i, s in enumerate(standings):
            s["rank"] = i + 1
        
        return standings
    
    def _determine_playoff_teams(self) -> List[Team]:
        """
        Determine which teams make the playoffs
        
        Returns:
        --------
        List[Team]
            List of playoff teams
        """
        # Get standings
        standings = self._calculate_standings()
        
        # Get top teams
        playoff_team_names = [s["team"] for s in standings[:self.num_playoff_teams]]
        
        # Get team objects
        playoff_teams = [team for team in self.teams if team.name in playoff_team_names]
        
        return playoff_teams
    
    def simulate_playoffs(self, playoff_teams: List[Team]) -> Dict[str, Any]:
        """
        Simulate the playoffs
        
        Parameters:
        -----------
        playoff_teams : List[Team]
            List of teams that made the playoffs
            
        Returns:
        --------
        Dict[str, Any]
            Playoff results
        """
        # Make sure we have the right number of playoff teams
        if len(playoff_teams) != self.num_playoff_teams:
            raise ValueError(f"Expected {self.num_playoff_teams} playoff teams, got {len(playoff_teams)}")
        
        # Set up bracket based on number of playoff teams
        if self.num_playoff_teams in (4, 6, 8):
            # Standard elimination bracket with byes for top seeds if needed
            if self.num_playoff_teams == 4:
                # 4-team bracket: 1v4, 2v3
                round_1_matchups = [
                    (playoff_teams[0], playoff_teams[3]),
                    (playoff_teams[1], playoff_teams[2])
                ]
                byes = []
            elif self.num_playoff_teams == 6:
                # 6-team bracket: 1&2 get byes, 3v6, 4v5
                round_1_matchups = [
                    (playoff_teams[2], playoff_teams[5]),
                    (playoff_teams[3], playoff_teams[4])
                ]
                byes = [playoff_teams[0], playoff_teams[1]]
            elif self.num_playoff_teams == 8:
                # 8-team bracket: 1v8, 2v7, 3v6, 4v5
                round_1_matchups = [
                    (playoff_teams[0], playoff_teams[7]),
                    (playoff_teams[1], playoff_teams[6]),
                    (playoff_teams[2], playoff_teams[5]),
                    (playoff_teams[3], playoff_teams[4])
                ]
                byes = []
            
            # Simulate playoff rounds
            playoff_results = self._simulate_elimination_bracket(round_1_matchups, byes)
            
            return playoff_results
        else:
            raise ValueError(f"Unsupported number of playoff teams: {self.num_playoff_teams}")
    
    def _simulate_elimination_bracket(self, 
                                    initial_matchups: List[Tuple[Team, Team]],
                                    byes: List[Team] = None) -> Dict[str, Any]:
        """
        Simulate an elimination bracket
        
        Parameters:
        -----------
        initial_matchups : List[Tuple[Team, Team]]
            List of first-round matchups
        byes : List[Team], optional
            Teams with first-round byes
                
        Returns:
        --------
        Dict[str, Any]
            Bracket results
        """
        if byes is None:
            byes = []
        
        # Keep track of all results by round
        results_by_round = []
        
        # Start week counter after regular season
        week = self.num_regular_weeks + 1
        
        # Round 1
        logger.info(f"Simulating Playoff Round 1 (Week {week})...")
        round_1_results = []
        round_1_winners = []
        
        for matchup in initial_matchups:
            home_team, away_team = matchup
            
            # Create game
            game = FantasyGame(home_team, away_team, week)
            
            # Simulate game
            game.simulate(self.player_performances[week], self.randomness)
            
            # Store results
            round_1_results.append({
                "home_team": home_team.name,
                "away_team": away_team.name,
                "home_score": game.home_score,
                "away_score": game.away_score,
                "winner": game.winner.name if game.winner else None,
                "loser": game.loser.name if game.loser else None
            })
            
            # Advance winner
            if game.winner:
                round_1_winners.append(game.winner)
        
        # Add bye teams to winners
        round_1_winners.extend(byes)
        
        # Store round results
        results_by_round.append({
            "round": 1,
            "week": week,
            "matchups": round_1_results,
            "byes": [team.name for team in byes]
        })
        
        # Round 2 (semi-finals)
        week += 1
        logger.info(f"Simulating Playoff Semi-Finals (Week {week})...")
        
        # Handle case when we don't have enough teams for semi-finals
        if len(round_1_winners) < 4 and len(round_1_winners) > 0:
            logger.warning(f"Expected 4 teams for semi-finals, got {len(round_1_winners)}. Adding dummy teams to complete bracket.")
            # Add dummy teams if needed to complete the bracket
            while len(round_1_winners) < 4:
                # Create a copy of the first team as a dummy (will likely lose)
                dummy_team = copy.deepcopy(round_1_winners[0])
                dummy_team.name = f"{dummy_team.name}_dummy_{len(round_1_winners)}"
                round_1_winners.append(dummy_team)
        
        # Make sure we have the right number of teams
        if len(round_1_winners) == 4:
            # Set up semi-final matchups
            # Top remaining seed plays lowest remaining seed
            round_1_winners.sort(key=lambda t: next((s["rank"] for s in self._calculate_standings() if s["team"] == t.name), 999))
            semi_matchups = [
                (round_1_winners[0], round_1_winners[3]),
                (round_1_winners[1], round_1_winners[2])
            ]
            
            semi_results = []
            semi_winners = []
            semi_losers = []  # For 3rd place game
            
            for matchup in semi_matchups:
                home_team, away_team = matchup
                
                # Create game
                game = FantasyGame(home_team, away_team, week)
                
                # Simulate game
                game.simulate(self.player_performances[week], self.randomness)
                
                # Store results
                semi_results.append({
                    "home_team": home_team.name,
                    "away_team": away_team.name,
                    "home_score": game.home_score,
                    "away_score": game.away_score,
                    "winner": game.winner.name if game.winner else None,
                    "loser": game.loser.name if game.loser else None
                })
                
                # Advance winner
                if game.winner:
                    semi_winners.append(game.winner)
                    semi_losers.append(game.loser)
            
            # Store round results
            results_by_round.append({
                "round": 2,
                "week": week,
                "matchups": semi_results
            })
            
            # Round 3 (finals)
            week += 1
            logger.info(f"Simulating Playoff Finals (Week {week})...")
            
            # Championship game
            if len(semi_winners) == 2:
                championship_matchup = (semi_winners[0], semi_winners[1])
                championship_game = FantasyGame(championship_matchup[0], championship_matchup[1], week)
                championship_game.simulate(self.player_performances[week], self.randomness)
                
                championship_result = {
                    "home_team": championship_matchup[0].name,
                    "away_team": championship_matchup[1].name,
                    "home_score": championship_game.home_score,
                    "away_score": championship_game.away_score,
                    "winner": championship_game.winner.name if championship_game.winner else None,
                    "loser": championship_game.loser.name if championship_game.loser else None
                }
                
                # 3rd place game
                if len(semi_losers) == 2:
                    third_place_matchup = (semi_losers[0], semi_losers[1])
                    third_place_game = FantasyGame(third_place_matchup[0], third_place_matchup[1], week)
                    third_place_game.simulate(self.player_performances[week], self.randomness)
                    
                    third_place_result = {
                        "home_team": third_place_matchup[0].name,
                        "away_team": third_place_matchup[1].name,
                        "home_score": third_place_game.home_score,
                        "away_score": third_place_game.away_score,
                        "winner": third_place_game.winner.name if third_place_game.winner else None,
                        "loser": third_place_game.loser.name if third_place_game.loser else None
                    }
                else:
                    third_place_result = None
                
                # Store round results
                finals_results = [championship_result]
                if third_place_result:
                    finals_results.append(third_place_result)
                
                results_by_round.append({
                    "round": 3,
                    "week": week,
                    "matchups": finals_results
                })
                
                # Determine champion and other placements
                champion = championship_game.winner
                runner_up = championship_game.loser
                
                third_place = None
                fourth_place = None
                if third_place_result:
                    third_place_game_winner = next((team for team in self.teams if team.name == third_place_result["winner"]), None)
                    third_place_game_loser = next((team for team in self.teams if team.name == third_place_result["loser"]), None)
                    third_place = third_place_game_winner
                    fourth_place = third_place_game_loser
                
                # Final bracket results
                bracket_results = {
                    "rounds": results_by_round,
                    "champion": champion.name if champion else None,
                    "runner_up": runner_up.name if runner_up else None,
                    "third_place": third_place.name if third_place else None,
                    "fourth_place": fourth_place.name if fourth_place else None
                }
                
                return bracket_results
            
            else:
                logger.error(f"Expected 2 semi-final winners, got {len(semi_winners)}")
                return {"rounds": results_by_round}
        
        else:
            logger.error(f"Expected 4 teams for semi-finals, got {len(round_1_winners)}")
            return {"rounds": results_by_round}
    
    def _get_final_standings(self) -> List[Dict[str, Any]]:
        """
        Get final season standings including playoffs
        
        Returns:
        --------
        List[Dict[str, Any]]
            Final standings
        """
        # Start with regular season standings
        standings = self._calculate_standings()
        
        # Will be updated with playoff results later when we integrate with playoffs
        
        return standings

class SeasonEvaluator:
    """Evaluates results of a fantasy football season"""
    
    def __init__(self, draft_teams: List[Team], season_results: Dict[str, Any], 
                 baseline_strategy: str = "VBD"):
        """
        Initialize the evaluator
        
        Parameters:
        -----------
        draft_teams : List[Team]
            Teams from the draft simulator
        season_results : Dict[str, Any]
            Results from the season simulator
        baseline_strategy : str, optional
            Strategy to use as baseline for comparison
        """
        self.draft_teams = draft_teams
        self.season_results = season_results
        self.baseline_strategy = baseline_strategy
        
        # Calculate metrics
        self.metrics = self.calculate_metrics()
    
    def calculate_metrics(self) -> Dict[str, Dict[str, Any]]:
        """
        Calculate performance metrics for each team/strategy
        
        Returns:
        --------
        Dict[str, Dict[str, Any]]
            Metrics by strategy
        """
        # Get final standings
        standings = self.season_results["standings"]
        
        # Get team records
        team_records = self.season_results["team_records"]
        
        # Group metrics by strategy
        metrics_by_strategy = defaultdict(list)
        
        # Process each team
        for team in self.draft_teams:
            # Get team's standing
            team_standing = next((s for s in standings if s["team"] == team.name), None)
            
            if not team_standing:
                logger.warning(f"Could not find standing for team {team.name}")
                continue
            
            # Calculate metrics
            metrics = {
                "team_name": team.name,
                "strategy": team.strategy,
                "rank": team_standing["rank"],
                "wins": team_standing["wins"],
                "losses": team_standing["losses"],  # Make sure this is captured
                "win_percentage": team_standing["win_pct"],
                "points_for": team_standing["points_for"],
                "points_against": team_standing["points_against"],
                "draft_value": team.get_total_projected_points(),
                "starter_value": team.get_starting_lineup_points(),
            }
            
            # Add playoff result if applicable
            if "playoffs" in self.season_results and "champion" in self.season_results["playoffs"]:
                if team.name == self.season_results["playoffs"]["champion"]:
                    metrics["playoff_result"] = "Champion"
                elif team.name == self.season_results["playoffs"]["runner_up"]:
                    metrics["playoff_result"] = "Runner-up"
                elif team.name == self.season_results["playoffs"].get("third_place"):
                    metrics["playoff_result"] = "Third Place"
                elif team.name == self.season_results["playoffs"].get("fourth_place"):
                    metrics["playoff_result"] = "Fourth Place"
                else:
                    # Check if team was in playoffs
                    in_playoffs = False
                    for round_data in self.season_results["playoffs"].get("rounds", []):
                        for matchup in round_data.get("matchups", []):
                            if team.name in (matchup.get("home_team"), matchup.get("away_team")):
                                in_playoffs = True
                                break
                        if in_playoffs:
                            break
                    
                    if in_playoffs:
                        metrics["playoff_result"] = "Playoff Qualification"
                    else:
                        metrics["playoff_result"] = "Missed Playoffs"
            
            # Add to strategy group
            metrics_by_strategy[team.strategy].append(metrics)
        
        # Aggregate metrics by strategy
        strategy_metrics = {}
        
        for strategy, team_metrics in metrics_by_strategy.items():
            # Number of teams with this strategy
            num_teams = len(team_metrics)
            
            # Calculate aggregate metrics
            strategy_metrics[strategy] = {
                "num_teams": num_teams,
                "avg_rank": sum(m["rank"] for m in team_metrics) / num_teams,
                "avg_wins": sum(m["wins"] for m in team_metrics) / num_teams,
                "avg_losses": sum(m["losses"] for m in team_metrics) / num_teams,  # Add this line
                "avg_points_for": sum(m["points_for"] for m in team_metrics) / num_teams,
                "avg_draft_value": sum(m["draft_value"] for m in team_metrics) / num_teams,
                "avg_starter_value": sum(m["starter_value"] for m in team_metrics) / num_teams,
                "championship_rate": sum(1 for m in team_metrics if m.get("playoff_result") == "Champion") / num_teams,
                "playoff_rate": sum(1 for m in team_metrics if m.get("playoff_result") in 
                                ["Champion", "Runner-up", "Third Place", "Fourth Place", "Playoff Qualification"]) / num_teams,
                "teams": team_metrics
            }
        
        # Calculate Value Over Replacement (VOR) and Points Above Expectation (PAE)
        if self.baseline_strategy in strategy_metrics:
            baseline = strategy_metrics[self.baseline_strategy]
            
            for strategy, metrics in strategy_metrics.items():
                metrics["vor_rank"] = baseline["avg_rank"] - metrics["avg_rank"]
                metrics["vor_wins"] = metrics["avg_wins"] - baseline["avg_wins"]
                metrics["vor_points"] = metrics["avg_points_for"] - baseline["avg_points_for"]
                metrics["pae"] = metrics["avg_points_for"] - metrics["avg_draft_value"]
        
        return strategy_metrics
    
    def get_best_strategy(self) -> str:
        """
        Determine the best overall strategy based on metrics
        
        Returns:
        --------
        str
            Name of the best strategy
        """
        if not self.metrics:
            return None
        
        # Score each strategy on multiple dimensions
        strategy_scores = {}
        
        for strategy, metrics in self.metrics.items():
            # Calculate a score combining multiple metrics
            # Lower rank is better, higher values for others are better
            score = (
                -1.0 * metrics["avg_rank"] +  # Negative because lower rank is better
                3.0 * metrics["avg_wins"] +
                0.05 * metrics["avg_points_for"] +
                5.0 * metrics["championship_rate"] +
                2.0 * metrics["playoff_rate"]
            )
            
            strategy_scores[strategy] = score
        
        # Return strategy with highest score
        return max(strategy_scores.items(), key=lambda x: x[1])[0]
    
    def generate_report(self, include_team_details: bool = True) -> Dict[str, Any]:
        """
        Generate a comprehensive report of season results
        
        Parameters:
        -----------
        include_team_details : bool, optional
            Whether to include detailed team metrics
            
        Returns:
        --------
        Dict[str, Any]
            Report data
        """
        report = {
            "strategy_metrics": self.metrics,
            "best_strategy": self.get_best_strategy(),
            "standings": self.season_results["standings"],
            "champion": self.season_results.get("playoffs", {}).get("champion"),
            "runner_up": self.season_results.get("playoffs", {}).get("runner_up")
        }
        
        if not include_team_details:
            # Remove team details to save space
            for strategy, metrics in report["strategy_metrics"].items():
                if "teams" in metrics:
                    del metrics["teams"]
        
        return report
    
    def save_report(self, output_path: str) -> None:
        """
        Save the evaluation report to a file
        
        Parameters:
        -----------
        output_path : str
            Path to save the report
        """
        report = self.generate_report()
        
        try:
            import json
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Report saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving report: {e}")