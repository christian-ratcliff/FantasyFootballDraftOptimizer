import random
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)
class LineupEvaluator:
    """Evaluates team performance based on optimal lineup with realistic randomness"""
    
    def __init__(self, teams, num_weeks=17, randomness=0.2, injury_chance=0.05):
        """
        Initialize the lineup evaluator
        
        Parameters:
        -----------
        teams : List[Team]
            List of fantasy teams
        num_weeks : int, optional
            Number of weeks to simulate (typically NFL season length)
        randomness : float, optional
            Amount of randomness in player performance (0.0 = deterministic, higher = more random)
        injury_chance : float, optional
            Probability of a player getting injured for a game
        """
        self.teams = teams
        self.num_weeks = num_weeks
        self.randomness = randomness
        self.injury_chance = injury_chance
    
    def evaluate_teams(self):
        """
        Evaluate all teams based on optimal lineups with randomness
        
        Returns:
        --------
        Dict[str, Any]
            Dictionary with team evaluations and rankings
        """
        team_scores = []
        
        for team in self.teams:
            # Calculate optimal lineup points with randomness
            total_points, weekly_points = self._calculate_team_points(team)
            
            # Store results in format that SeasonEvaluator expects
            team_scores.append({
                "team": team,
                "name": team.name,
                "strategy": team.strategy,
                "total_points": total_points,
                "weekly_points": weekly_points,
                "average_points": total_points / self.num_weeks if self.num_weeks > 0 else 0
            })
        
        # Sort by total points (descending)
        team_scores.sort(key=lambda x: x["total_points"], reverse=True)
        
        # Add rankings
        for i, team_score in enumerate(team_scores):
            team_score["rank"] = i + 1
        
        # Create standings in the format that SeasonEvaluator expects
        standings = []
        for ts in team_scores:
            # This is the key fix: create standings entries that match what SeasonEvaluator expects
            standings.append({
                "team": ts["team"].name,  # Use name string instead of team object
                "rank": ts["rank"],
                "wins": self.num_weeks - ts["rank"],  # Approximate wins based on rank
                "losses": ts["rank"] - 1,  # Approximate losses based on rank
                "point_differential": ts["total_points"],
                "points_for": ts["total_points"],
                "points_against": sum(other["total_points"] for other in team_scores if other["name"] != ts["name"]) / (len(team_scores) - 1),
                "strategy": ts["team"].strategy,  # Include strategy to help with finding PPO team
                "win_pct": (self.num_weeks - ts["rank"]) / self.num_weeks if self.num_weeks > 0 else 0
            })
        
        # Convert to final format (similar to SeasonSimulator output)
        results = {
            "standings": standings,  # Use the properly formatted standings
            "team_records": {ts["name"]: {
                "wins": self.num_weeks - ts["rank"],
                "losses": ts["rank"] - 1,
                "ties": 0,  # No ties in our simplified model
                "points_for": ts["total_points"],
                "points_against": sum(other["total_points"] for other in team_scores if other["name"] != ts["name"]) / (len(team_scores) - 1)
            } for ts in team_scores}
        }
        
        # Simplified playoff results based on rankings
        playoffs = {}
        if len(team_scores) >= 4:
            playoffs["champion"] = team_scores[0]["name"]
            playoffs["runner_up"] = team_scores[1]["name"]
            if len(team_scores) >= 6:
                playoffs["third_place"] = team_scores[2]["name"]
                playoffs["fourth_place"] = team_scores[3]["name"]
        
        results["playoffs"] = playoffs
        
        return results
    
    def _calculate_team_points(self, team):
        """
        Calculate points for a team across simulated weeks
        
        Parameters:
        -----------
        team : Team
            Fantasy team to evaluate
            
        Returns:
        --------
        Tuple[float, List[float]]
            Total points and list of weekly points
        """
        weekly_points = []
        
        # Get optimal lineup
        starters = team.get_optimal_starters()
        
        # For each week
        for week in range(self.num_weeks):
            week_points = 0
            
            # Calculate points for each position group
            for position, players in starters.items():
                for player in players:
                    # Check for injury
                    if random.random() < self.injury_chance:
                        # Player is injured this week
                        continue
                    
                    # Base points from projection
                    base_points = player.projected_points / self.num_weeks
                    
                    # Apply randomness based on player's range
                    if hasattr(player, 'projection_high') and hasattr(player, 'projection_low'):
                        std_dev = (player.projection_high - player.projection_low) / (2 * self.num_weeks)
                    else:
                        # Default if range not available
                        std_dev = base_points * 0.3
                    
                    # Ensure minimum variation
                    std_dev = max(std_dev, base_points * self.randomness)
                    
                    # Generate performance
                    performance = random.gauss(base_points, std_dev)
                    performance = max(0, performance)  # No negative points
                    
                    week_points += performance
            
            weekly_points.append(week_points)
        
        # Calculate total points
        total_points = sum(weekly_points)
        
        return total_points, weekly_points