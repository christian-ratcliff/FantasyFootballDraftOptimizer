#!/usr/bin/env python3
"""
League Config Generator

This script generates a config file for the Fantasy Football Draft Optimizer
by connecting to your ESPN fantasy football league and extracting the settings.
"""

import os
import json
import argparse
from espn_api.football import League

def generate_config(league_id, year, espn_s2=None, swid=None, output_path=None):
    """
    Generate a comprehensive config file by connecting to ESPN API and extracting all league settings
    
    Parameters:
    -----------
    league_id : int
        ESPN league ID
    year : int
        Season year
    espn_s2 : str, optional
        ESPN S2 cookie for private leagues
    swid : str, optional
        SWID cookie for private leagues
    output_path : str, optional
        Path to save the config file (default: configs/league_settings.json)
    
    Returns:
    --------
    bool
        True if config was successfully generated, False otherwise
    """
    print(f"Connecting to ESPN league {league_id} for year {year}...")
    
    try:
        # Initialize connection to ESPN league
        league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
        
        # Extract relevant league settings
        settings = league.settings
        
        # Extract scoring settings - get EVERYTHING
        scoring_dict = {}
        for item in settings.scoring_format:
            scoring_dict[item['abbr']] = item['points']
        
        # Calculate bench spots
        total_roster_size = 18  # Total roster size
        bench_spots = total_roster_size - 10
        # Extract roster settings
        roster_dict = {
            "QB": 3,  # Maximum 3 QBs
            "RB": 10,  # Effectively unlimited
            "WR": 10,  # Effectively unlimited
            "TE": 2,  # Effectively unlimited
            "BE": bench_spots,  # Calculated bench spots
            "IR": 1    # One IR spot
        }
        
        # Get draft details including draft type, time per pick, etc.
        draft_settings = {
            "keeper_count": settings.keeper_count,
            "draft_order": [],
            "draft_type": "snake"  # default, would need to get actual value from API
        }
        
        # Try to get draft order if draft has occurred
        if hasattr(league, 'draft') and league.draft:
            draft_order = []
            for pick in league.draft:
                if pick.round_num == 1:
                    draft_order.append(pick.team.team_id)
            draft_settings["draft_order"] = draft_order
        
        # Build config object with comprehensive settings
        config = {
            "league_id": league_id,
            "year": year,
            "espn_s2": espn_s2,
            "swid": swid,
            "league_info": {
                "name": settings.name,
                "team_count": settings.team_count,
                "playoff_teams": settings.playoff_team_count,
                "nfl_games_per_player": 17,  # Set to current NFL season length
                "current_matchup_period": league.currentMatchupPeriod,
                "current_scoring_period": league.scoringPeriodId,
                "first_scoring_period": league.firstScoringPeriod,
                "final_scoring_period": league.finalScoringPeriod,
                "current_week": league.current_week
            },
            "schedule_settings": {
                "regular_season_weeks": settings.reg_season_count,
                "matchup_periods": settings.matchup_periods,
                "playoff_matchup_period_length": settings.playoff_matchup_period_length,
                "playoff_seed_tie_rule": settings.playoff_seed_tie_rule
            },
            "trade_settings": {
                "deadline": settings.trade_deadline,
                "veto_votes_required": settings.veto_votes_required
            },
            "acquisition_settings": {
                "faab_enabled": settings.faab,
                "acquisition_budget": settings.acquisition_budget
            },
            "draft_settings": draft_settings,
            "division_settings": {
                "division_map": settings.division_map
            },
            "tiebreaker_settings": {
                "regular_season_tiebreaker": settings.tie_rule,
                "playoff_tiebreaker": settings.playoff_tie_rule
            },
            "scoring_settings": scoring_dict,
            "roster_settings": roster_dict,
            "scoring_type": settings.scoring_type
        }
        
        # Extract starter limits for simulation
        # First try to determine from actual settings, then fall back to reasonable defaults
        starter_limits = settings.position_slot_counts
        
        # # Standard positions
        # for pos in ["QB", "RB", "WR", "TE", "D/ST", "K", "FLEX"]:
        #     if pos in roster_dict:
        #         starter_limits[pos] = roster_dict[pos]
        
        # # Handle special FLEX spots
        # for flex_pos in ["RB/WR", "WR/TE", "RB/WR/TE", "OP"]:
        #     if flex_pos in roster_dict:
        #         starter_limits[flex_pos] = roster_dict[flex_pos]
        
        # # Handle defensive positions for IDP leagues
        # for def_pos in ["DL", "LB", "DB", "DE", "DT", "CB", "S", "DP"]:
        #     if def_pos in roster_dict:
        #         starter_limits[def_pos] = roster_dict[def_pos]
        
        # Add starter limits to config
        config["starter_limits"] = starter_limits
        
        # Add bench and IR spots
        if "BE" in roster_dict:
            config["bench_spots"] = roster_dict["BE"]
        if "IR" in roster_dict:
            config["ir_spots"] = roster_dict["IR"]
        
        # Add additional team data for simulation and GUI
        team_data = []
        for team in league.teams:
            team_info = {
                "team_id": team.team_id,
                "team_abbrev": team.team_abbrev,
                "team_name": team.team_name,
                "owner_name": team.owners[0] if team.owners else None,
                "division_id": team.division_id,
                "division_name": team.division_name,
                "wins": team.wins,
                "losses": team.losses,
                "ties": team.ties,
                "points_for": team.points_for,
                "points_against": team.points_against,
                "acquisition_budget_spent": team.acquisition_budget_spent,
                "waiver_rank": team.waiver_rank,
                "standing": team.standing,
                "logo_url": team.logo_url
            }
            team_data.append(team_info)
        
        config["teams"] = team_data
        
        # Create a player map to help with draft assistant
        player_map = {}
        if hasattr(league, 'player_map'):
            # Convert player map to a more usable format
            for player_id, player_name in league.player_map.items():
                if isinstance(player_id, int):  # Only take ID -> name mappings
                    player_map[str(player_id)] = player_name
        
        config["player_map"] = player_map
        
        # Get schedule information
        schedule_data = []
        try:
            matchups = league.scoreboard()
            for matchup in matchups:
                if hasattr(matchup, 'home_team') and hasattr(matchup, 'away_team'):
                    schedule_data.append({
                        'home_team_id': matchup.home_team.team_id if matchup.home_team else None,
                        'away_team_id': matchup.away_team.team_id if matchup.away_team else None,
                        'home_score': matchup.home_score,
                        'away_score': matchup.away_score,
                        'matchup_type': matchup.matchup_type,
                        'is_playoff': matchup.is_playoff
                    })
        except Exception as e:
            print(f"Error fetching schedule: {e}")
        
        config["current_schedule"] = schedule_data
        
        # Set default output path if not provided
        if output_path is None:
            output_path = "configs/league_settings.json"
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save config to file
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"Config file successfully generated and saved to {output_path}")
        print(f"League: {settings.name}")
        print(f"Teams: {settings.team_count}")
        
        return True
    
    except Exception as e:
        print(f"Error connecting to ESPN league: {e}")
        print("\nPossible issues:")
        print("1. Invalid league ID or year")
        print("2. Private league that requires cookies")
        print("3. ESPN API changes or rate limiting")
        
        return False
    
    
def get_espn_cookies():
    """
    Guide the user through getting ESPN cookies for private leagues
    
    Returns:
    --------
    tuple
        (espn_s2, swid) cookies
    """
    print("\n=== ESPN Cookie Instructions ===")
    print("For private leagues, you need to provide your ESPN cookies (espn_s2 and SWID).")
    print("To get these cookies:")
    print("1. Log in to your ESPN Fantasy Football account in a web browser")
    print("2. Open your browser's developer tools (F12 or right-click > Inspect)")
    print("3. Go to the Application/Storage tab")
    print("4. Look for Cookies in the sidebar and click on the ESPN website")
    print("5. Find cookies named 'espn_s2' and 'SWID'")
    print("6. Copy the values and paste them below")
    
    espn_s2 = input("\nEnter your espn_s2 cookie (or press Enter to skip): ").strip()
    swid = input("Enter your SWID cookie (or press Enter to skip): ").strip()
    
    if not espn_s2:
        espn_s2 = None
    if not swid:
        swid = None
    
    return espn_s2, swid

def main():
    """Main function to run the config generator interactively"""
    parser = argparse.ArgumentParser(description='Generate config file for Fantasy Football Optimizer')
    parser.add_argument('--league_id', type=int, help='ESPN league ID')
    parser.add_argument('--year', type=int, help='Season year')
    parser.add_argument('--espn_s2', help='ESPN S2 cookie for private leagues')
    parser.add_argument('--swid', help='SWID cookie for private leagues')
    parser.add_argument('--output', help='Path to save config file')
    
    args = parser.parse_args()
    
    # If command line args not provided, prompt user interactively
    if not args.league_id:
        try:
            league_id = int(input("Enter your ESPN league ID: "))
        except ValueError:
            print("Error: League ID must be a number")
            return
    else:
        league_id = args.league_id
    
    if not args.year:
        try:
            year = int(input(f"Enter the season year [default: 2024]: ") or "2024")
        except ValueError:
            print("Error: Year must be a number")
            return
    else:
        year = args.year
    
    # Get cookies if not provided
    if not args.espn_s2 or not args.swid:
        is_private = input("Is this a private league? (y/n): ").lower().startswith('y')
        if is_private:
            espn_s2, swid = get_espn_cookies()
        else:
            espn_s2, swid = None, None
    else:
        espn_s2, swid = args.espn_s2, args.swid
    
    # Get output path
    output_path = args.output
    
    # Generate config
    success = generate_config(league_id, year, espn_s2, swid, output_path)
    
    if success:
        print("\nNow you can run the main analysis script:")
        print("python main.py")

if __name__ == "__main__":
    print("===== Fantasy Football League Config Generator =====\n")
    main()