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
    Generate a config file by connecting to ESPN API and extracting league settings
    
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
        
        # Extract scoring settings
        scoring_dict = {}
        for item in settings.scoring_format:
            scoring_dict[item['abbr']] = item['points']
        
        # Extract roster settings
        roster_dict = settings.position_slot_counts
        
        # Build config object
        config = {
            "league_id": league_id,
            "year": year,
            "espn_s2": espn_s2,
            "swid": swid,
            "league_info": {
                "name": settings.name,
                "team_count": settings.team_count,
                "playoff_teams": settings.playoff_team_count
            },
            "scoring_settings": scoring_dict,
            "roster_settings": roster_dict
        }
        
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