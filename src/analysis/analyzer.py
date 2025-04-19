import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from espn_api.football import League
import nfl_data_py as nfl
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set plot style
plt.style.use('fivethirtyeight')
sns.set_palette('colorblind')

class FantasyFootballAnalyzer:
    """
    A class to analyze fantasy football data and provide insights for draft strategy
    """
    
    def __init__(self, league_id, year, espn_s2=None, swid=None):
        """
        Initialize the analyzer with league information
        
        Parameters:
        -----------
        league_id : int
            ESPN league ID
        year : int
            Season year
        espn_s2 : str, optional
            ESPN S2 cookie for private leagues
        swid : str, optional
            ESPN SWID cookie for private leagues
        """
        self.league_id = league_id
        self.year = year
        self.espn_s2 = espn_s2
        self.swid = swid
        
        # Load league data
        self.league = League(league_id=league_id, year=year, espn_s2=espn_s2, swid=swid)
        
        # Store league settings
        self.settings = self.league.settings
        self.scoring_format = self.extract_scoring_format()
        self.roster_positions = self.settings.position_slot_counts
        
        # Initialize data containers
        self.historical_data = None
        self.current_season_data = None
        self.player_projections = None
    
    def extract_scoring_format(self):
        """Extract and return scoring format from league settings"""
        scoring_dict = {}
        for item in self.settings.scoring_format:
            scoring_dict[item['abbr']] = item['points']
        return scoring_dict
    
    def get_historical_data(self, years=3):
        """
        Get historical player data for the specified number of years
        
        Parameters:
        -----------
        years : int, optional
            Number of past years to retrieve data for
        """
        year_list = list(range(self.year - years, self.year))
        
        # Get seasonal stats
        print(f"Loading seasonal data for years: {year_list}")
        seasonal_data = nfl.import_seasonal_data(year_list)
        
        # Get weekly stats
        print(f"Loading weekly data for years: {year_list}")
        weekly_data = nfl.import_weekly_data(year_list)
        
        # Get player IDs for mapping
        print("Loading player ID mapping data")
        player_ids = nfl.import_ids()
        
        # Merge player names from ID mapping
        seasonal_data = pd.merge(
            seasonal_data,
            player_ids[['gsis_id', 'name', 'position']],
            left_on='player_id',
            right_on='gsis_id',
            how='left'
        )
        
        # Store the data
        self.historical_data = {
            'seasonal': seasonal_data,
            'weekly': weekly_data,
            'player_ids': player_ids
        }
        
        print(f"Loaded data for {len(seasonal_data['name'].unique())} players")
        return self.historical_data
    
    def get_current_season_data(self):
        """Get data for the current season"""
        # Get current season data
        print(f"Loading current season data for year: {self.year}")
        current_season_data = {}
        
        # Get seasonal stats if available
        try:
            seasonal_data = nfl.import_seasonal_data([self.year])
            
            # Get player IDs for mapping if not already loaded
            if self.historical_data is None or 'player_ids' not in self.historical_data:
                player_ids = nfl.import_ids()
            else:
                player_ids = self.historical_data['player_ids']
            
            # Merge player names from ID mapping
            seasonal_data = pd.merge(
                seasonal_data,
                player_ids[['gsis_id', 'name', 'position']],
                left_on='player_id',
                right_on='gsis_id',
                how='left'
            )
            
            current_season_data['seasonal'] = seasonal_data
            print(f"Loaded seasonal data with {len(seasonal_data)} rows")
        except Exception as e:
            print(f"Could not load seasonal data: {e}")
        
        # Get NGS data
        try:
            passing_data = nfl.import_ngs_data('passing', [self.year])
            rushing_data = nfl.import_ngs_data('rushing', [self.year])
            receiving_data = nfl.import_ngs_data('receiving', [self.year])
            
            # Add position data to NGS data if player_ids are available
            if 'player_ids' in self.historical_data:
                player_ids = self.historical_data['player_ids']
                for dataset in [passing_data, rushing_data, receiving_data]:
                    if 'player_gsis_id' in dataset.columns:
                        dataset = pd.merge(
                            dataset,
                            player_ids[['gsis_id', 'position']],
                            left_on='player_gsis_id',
                            right_on='gsis_id',
                            how='left'
                        )
                        # Add player_id column for consistency
                        dataset['player_id'] = dataset['player_gsis_id']
            
            current_season_data['ngs'] = {
                'passing': passing_data,
                'rushing': rushing_data,
                'receiving': receiving_data
            }
            print(f"Loaded NGS data")
        except Exception as e:
            print(f"Could not load NGS data: {e}")
        
        # Store the data
        self.current_season_data = current_season_data
        
        return self.current_season_data
    
    def explore_league_settings(self):
        """Explore and print league settings"""
        print("=== League Settings ===")
        print(f"League Name: {self.settings.name}")
        print(f"League Size: {self.settings.team_count} teams")
        print(f"Playoff Teams: {self.settings.playoff_team_count}")
        
        print("\n=== Roster Settings ===")
        for position, count in self.roster_positions.items():
            if count > 0:
                print(f"{position}: {count}")
        
        print("\n=== Scoring Format ===")
        for abbr, points in self.scoring_format.items():
            if points != 0:
                print(f"{abbr}: {points}")
    
    def explore_player_data(self, position=None):
        """
        Explore player data and create visualizations
        
        Parameters:
        -----------
        position : str, optional
            Filter to specific position (QB, RB, WR, TE)
        """
        if self.historical_data is None:
            print("No historical data loaded. Run get_historical_data() first.")
            return
        
        seasonal_data = self.historical_data['seasonal']
        
        if position:
            # Filter by position directly
            if 'position' in seasonal_data.columns:
                seasonal_data = seasonal_data[seasonal_data['position'] == position]
                print(f"Filtered to {len(seasonal_data)} {position} players")
            else:
                print("Position column not found in seasonal data.")
                return
        
        # Create visualizations
        plt.figure(figsize=(12, 8))
        
        # Key stats based on position
        if position == 'QB':
            key_stats = ['passing_yards', 'passing_tds', 'interceptions', 'rushing_yards', 'rushing_tds']
        elif position == 'RB':
            key_stats = ['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds']
        elif position == 'WR' or position == 'TE':
            key_stats = ['receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_air_yards']
        else:
            key_stats = ['fantasy_points', 'fantasy_points_ppr']
        
        # Plot distributions of key stats
        for i, stat in enumerate(key_stats):
            if stat in seasonal_data.columns:
                plt.subplot(len(key_stats), 1, i+1)
                sns.histplot(seasonal_data[stat].dropna(), kde=True)
                plt.title(f'Distribution of {stat}')
                plt.tight_layout()
        
        plt.show()
        
        # correlations
        if len(key_stats) > 1:
            corr_data = seasonal_data[key_stats].corr()
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_data, annot=True, cmap='coolwarm')
            plt.title(f'Correlation Matrix for {position if position else "All Players"}')
            plt.tight_layout()
            plt.show()
    
    def calculate_fantasy_points(self, data, scoring_format=None):
        """
        Calculate fantasy points based on league scoring settings
        
        Parameters:
        -----------
        data : DataFrame
            Player statistics
        scoring_format : dict, optional
            Custom scoring format. If None, use league's scoring
            
        Returns:
        --------
        DataFrame
            Original data with fantasy points added
        """
        if scoring_format is None:
            scoring_format = self.scoring_format
        
        # avoid modifying original
        result = data.copy()
        
        # Standard scoring mappings - extend this as needed based on league settings
        result['calculated_points'] = 0
        
        # Passing points
        if 'PY' in scoring_format and 'passing_yards' in result.columns:
            result['calculated_points'] += result['passing_yards'] * scoring_format['PY']
        
        if 'PTD' in scoring_format and 'passing_tds' in result.columns:
            result['calculated_points'] += result['passing_tds'] * scoring_format['PTD']
        
        if 'INT' in scoring_format and 'interceptions' in result.columns:
            result['calculated_points'] += result['interceptions'] * scoring_format['INT']
        
        # Rushing points
        if 'RY' in scoring_format and 'rushing_yards' in result.columns:
            result['calculated_points'] += result['rushing_yards'] * scoring_format['RY']
        
        if 'RTD' in scoring_format and 'rushing_tds' in result.columns:
            result['calculated_points'] += result['rushing_tds'] * scoring_format['RTD']
        
        # Receiving points
        if 'REY' in scoring_format and 'receiving_yards' in result.columns:
            result['calculated_points'] += result['receiving_yards'] * scoring_format['REY']
        
        if 'RETD' in scoring_format and 'receiving_tds' in result.columns:
            result['calculated_points'] += result['receiving_tds'] * scoring_format['RETD']
        
        if 'REC' in scoring_format and 'receptions' in result.columns:
            result['calculated_points'] += result['receptions'] * scoring_format['REC']
        
        return result
    
    def preprocess_data(self, target_year=None):
        """
        Preprocess data for modeling
        
        Parameters:
        -----------
        target_year : int, optional
            Year to use as target for prediction
            
        Returns:
        --------
        dict
            Preprocessed data ready for modeling
        """
        if self.historical_data is None:
            print("No historical data loaded. Run get_historical_data() first.")
            return
        
        seasonal_data = self.historical_data['seasonal']
        
        # Calculate fantasy points
        seasonal_data = self.calculate_fantasy_points(seasonal_data)
        
        # Filter relevant columns for modeling
        features = ['season', 'player_id', 'games', 'passing_yards', 'passing_tds', 'interceptions',
                   'rushing_yards', 'rushing_tds', 'receptions', 'targets', 'receiving_yards', 'receiving_tds',
                   'fantasy_points', 'fantasy_points_ppr', 'calculated_points']
        
        # Add position column if it exists
        if 'position' in seasonal_data.columns:
            features.append('position')
        
        # Filter to columns that actually exist in the dataframe
        features = [col for col in features if col in seasonal_data.columns]
        
        model_data = seasonal_data[features].copy()
        
        # Handle missing values
        model_data = model_data.fillna(0)
        
        # Check if we're doing actual train/test split
        if target_year:
            # We're using a target year as test data, the rest as training
            train_data = model_data[model_data['season'] < target_year]
            test_data = model_data[model_data['season'] == target_year]
            
            # Only proceed if we have both train and test data
            if len(train_data) == 0 or len(test_data) == 0:
                print("Insufficient data for split based on target year")
                return model_data
        else:
            # Use the most recent season in the data as the test set
            max_season = model_data['season'].max()
            train_data = model_data[model_data['season'] < max_season]
            test_data = model_data[model_data['season'] == max_season]
            
            # Only proceed if we have both train and test data
            if len(train_data) == 0 or len(test_data) == 0:
                print("Insufficient data for train/test split")
                return model_data
        
        # Log data shapes and column presence
        print(f"Train data shape: {train_data.shape}, Position column exists: {'position' in train_data.columns}")
        print(f"Test data shape: {test_data.shape}, Position column exists: {'position' in test_data.columns}")
        
        # Create per-game features
        for df in [train_data, test_data]:
            # Create features for modeling - normalize by games played
            for col in ['passing_yards', 'passing_tds', 'interceptions',
                        'rushing_yards', 'rushing_tds', 'receptions', 'targets', 
                        'receiving_yards', 'receiving_tds']:
                if col in df.columns:
                    df[f'{col}_per_game'] = df[col] / df['games'].clip(lower=1)
            
            # Add "points per game" features
            if 'fantasy_points' in df.columns:
                df['fantasy_points_per_game'] = df['fantasy_points'] / df['games'].clip(lower=1)
            
            if 'calculated_points' in df.columns:
                df['calculated_points_per_game'] = df['calculated_points'] / df['games'].clip(lower=1)
        
        # Store preprocessed data
        preprocessed_data = {
            'train': train_data,
            'test': test_data
        }
        
        return preprocessed_data
    
    def save_data(self, filename='fantasy_football_data.pkl'):
        """
        Save all data to a pickle file
        
        Parameters:
        -----------
        filename : str, optional
            Filename to save data to
        """
        data_to_save = {
            'league_settings': {
                'scoring_format': self.scoring_format,
                'roster_positions': self.roster_positions
            },
            'historical_data': self.historical_data,
            'current_season_data': self.current_season_data
        }
        
        pd.to_pickle(data_to_save, filename)
        print(f"Data saved to {filename}")
    
    def load_data(self, filename='fantasy_football_data.pkl'):
        """
        Load data from a pickle file
        
        Parameters:
        -----------
        filename : str, optional
            Filename to load data from
        """
        try:
            data = pd.read_pickle(filename)
            
            self.scoring_format = data['league_settings']['scoring_format']
            self.roster_positions = data['league_settings']['roster_positions']
            self.historical_data = data['historical_data']
            self.current_season_data = data['current_season_data']
            
            print(f"Data loaded from {filename}")
        except FileNotFoundError:
            print(f"File {filename} not found")
        except Exception as e:
            print(f"Error loading data: {e}")