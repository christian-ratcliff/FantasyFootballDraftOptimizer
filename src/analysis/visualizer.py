"""
Modified module for creating visualizations of fantasy football data
with improved organization by position and analysis type
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FantasyDataVisualizer:
    """
    Class for creating visualizations of fantasy football data
    """
    
    def __init__(self, data_dict, feature_sets=None, output_dir="data/outputs"):
        """
        Initialize with data dictionary and output directory
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary containing various dataframes of player data
        feature_sets : dict
            Dictionary containing engineered feature sets
        output_dir : str
            Directory to save visualizations
        """
        self.data_dict = data_dict
        self.feature_sets = feature_sets or {}
        self.output_dir = output_dir
        
        # Create organized visualization directories
        self._create_visualization_dirs()
        
        # Set visualization style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_context("notebook", font_scale=1.1)
        
        # Use a nice color palette (colorblind-friendly)
        self.colors = sns.color_palette("colorblind")
        self.positions_palette = {
            'QB': self.colors[0],
            'RB': self.colors[1],
            'WR': self.colors[2],
            'TE': self.colors[3]
        }
        
        # Custom color maps for heatmaps and clusters
        self.cmap_red_blue = LinearSegmentedColormap.from_list("red_blue", 
                                                            ["#ef8a62", "#f7f7f7", "#67a9cf"])
        
        # Fixed colors for clusters 
        self.cluster_colors = {
            'Elite': '#1f77b4',       # blue
            'High Tier': '#2ca02c',   # green
            'Mid Tier': '#ff7f0e',    # orange
            'Low Tier': '#d62728',    # red
            'Bottom Tier': '#9467bd'  # purple
        }
        
        # Default cluster tiers (in order of quality)
        self.cluster_tiers = ['Elite', 'High Tier', 'Mid Tier', 'Low Tier', 'Bottom Tier']
        
        logger.info(f"Initialized data visualizer with {len(data_dict)} datasets")
    
    def _create_visualization_dirs(self):
        """Create organized visualization directory structure"""
        # Positions
        positions = ['qb', 'rb', 'wr', 'te', 'overall']
        
        # Analysis categories
        categories = [
            'correlations',
            'clusters',
            'feature_importance',
            'time_trends',
            'distributions',
            'league_settings'
        ]
        
        # Create directories for each position and category
        for position in positions:
            for category in categories:
                dir_path = os.path.join(self.output_dir, position, category)
                os.makedirs(dir_path, exist_ok=True)
        
        # Map of analysis types to directories
        self.viz_dirs = {
            'eda': 'basic_stats',
            'players': 'players',
            'positions': 'position_analysis',
            'trends': 'time_trends',
            'clusters': 'clusters',
            'advanced': 'advanced_analytics',
            'league': 'league_settings'
        }
        
        logger.info(f"Created organized visualization directory structure")
    
    def explore_league_settings(self, league_data):
        """
        Visualize league settings
        
        Parameters:
        -----------
        league_data : dict
            Dictionary containing league settings
        """
        if not league_data:
            logger.warning("No league data provided")
            return
        
        # Extract settings
        scoring_settings = league_data.get('scoring_settings', {})
        roster_settings = league_data.get('roster_settings', {})
        league_info = league_data.get('league_info', {})
        
        # Output directory
        league_dir = os.path.join(self.output_dir, 'overall', 'league_settings')
        
        if not scoring_settings and not roster_settings:
            logger.warning("No league settings to visualize")
            return
        
        # Create scoring settings visualization
        if scoring_settings:
            logger.info("Creating scoring settings visualization")
            
            # Filter to non-zero scoring settings
            scoring_values = [(abbr, points) for abbr, points in scoring_settings.items() if points != 0]
            
            if not scoring_values:
                logger.warning("No non-zero scoring settings found")
            else:
                # Sort by absolute value for better visualization
                scoring_values.sort(key=lambda x: abs(x[1]), reverse=True)
                
                # Take top 20 for readability
                if len(scoring_values) > 20:
                    logger.info(f"Limiting scoring visualization to top 20 of {len(scoring_values)} settings")
                    scoring_values = scoring_values[:20]
                
                # Create figure
                plt.figure(figsize=(12, 8))
                
                # Create horizontal bar chart
                abbrs, points = zip(*scoring_values)
                bars = plt.barh(abbrs, points)
                
                # Add color based on positive/negative
                for i, bar in enumerate(bars):
                    if points[i] > 0:
                        bar.set_color('#2ca02c')  # Green for positive
                    else:
                        bar.set_color('#d62728')  # Red for negative
                
                # Add values to the end of each bar
                for i, (abbr, value) in enumerate(zip(abbrs, points)):
                    plt.text(value + (0.01 * max(points) if value >= 0 else -0.01 * max(points)),
                        i, f' {value}', va='center')
                
                # Add title and labels
                league_name = league_info.get('name', 'League')
                plt.title(f'Scoring Settings for {league_name}', fontsize=16, pad=20)
                plt.xlabel('Points', fontsize=12)
                plt.tight_layout()
                
                # Save figure
                plt.savefig(os.path.join(league_dir, 'scoring_settings.png'))
                plt.close()
        
        # Create roster settings visualization
        if roster_settings:
            logger.info("Creating roster settings visualization")
            
            # Filter to non-zero roster positions
            roster_values = [(pos, count) for pos, count in roster_settings.items() if count > 0]
            
            if not roster_values:
                logger.warning("No roster positions found")
            else:
                # Sort by count and position hierarchy
                position_order = {
                    'QB': 1, 'RB': 2, 'WR': 3, 'TE': 4, 'FLEX': 5, 'K': 6, 'D/ST': 7,
                    'DL': 8, 'LB': 9, 'DB': 10, 'IDP': 11, 'BE': 12, 'IR': 13
                }
                
                # Sort by position hierarchy, then by count
                roster_values.sort(key=lambda x: (position_order.get(x[0], 100), -x[1]))
                
                # Create figure
                plt.figure(figsize=(10, 6))
                
                # Create horizontal bar chart
                positions, counts = zip(*roster_values)
                bars = plt.barh(positions, counts)
                
                # Add color based on position group
                for i, bar in enumerate(bars):
                    pos = positions[i]
                    if pos in ['QB', 'RB', 'WR', 'TE', 'FLEX']:
                        bar.set_color(self.positions_palette.get(pos, '#1f77b4'))  # Use position palette if available
                    elif pos in ['K', 'D/ST']:
                        bar.set_color('#9467bd')  # Purple for K and D/ST
                    elif pos in ['BE', 'IR']:
                        bar.set_color('#7f7f7f')  # Gray for bench
                    else:
                        bar.set_color('#8c564b')  # Brown for IDP
                
                # Add values to the end of each bar
                for i, (pos, count) in enumerate(zip(positions, counts)):
                    plt.text(count + 0.1, i, f' {count}', va='center')
                
                # Add title and labels
                league_name = league_info.get('name', 'League')
                plt.title(f'Roster Settings for {league_name}', fontsize=16, pad=20)
                plt.xlabel('Number of Players', fontsize=12)
                plt.tight_layout()
                
                # Save figure
                plt.savefig(os.path.join(league_dir, 'roster_settings.png'))
                plt.close()
        
        # Create league summary table
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('tight')
        ax.axis('off')
        
        # Gather league info
        league_name = league_info.get('name', 'Unknown League')
        team_count = league_info.get('team_count', 'Unknown')
        playoff_teams = league_info.get('playoff_teams', 'Unknown')
        
        # Create a summary table
        table_data = [
            ['League Name', league_name],
            ['Number of Teams', team_count],
            ['Playoff Teams', playoff_teams],
            ['QB Roster Spots', roster_settings.get('QB', 0)],
            ['RB Roster Spots', roster_settings.get('RB', 0)],
            ['WR Roster Spots', roster_settings.get('WR', 0)],
            ['TE Roster Spots', roster_settings.get('TE', 0)],
            ['FLEX Roster Spots', roster_settings.get('FLEX', 0)],
            ['Bench Spots', roster_settings.get('BE', 0)],
            ['Passing TD Points', scoring_settings.get('PTD', 0)],
            ['Rushing TD Points', scoring_settings.get('RTD', 0)],
            ['Receiving TD Points', scoring_settings.get('RETD', 0)],
            ['PPR Points', scoring_settings.get('REC', 0)],
            ['Interception Points', scoring_settings.get('INT', 0)]
        ]
        
        # Create the table
        table = ax.table(cellText=table_data, loc='center', cellLoc='left', colWidths=[0.4, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.2)
        
        # Style the table
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(fontproperties=dict(weight='bold'))
            if col == 0:
                cell.set_text_props(fontproperties=dict(weight='semibold'))
        
        plt.title('League Summary', fontsize=16, pad=20)
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(league_dir, 'league_summary.png'))
        plt.close()
        
        logger.info("League settings visualizations created")
    
    def explore_data_distributions(self):
        """
        Create visualizations of key data distributions
        """
        logger.info("Creating data distribution visualizations")
        
        # Use seasonal data for distribution analysis
        seasonal = self.data_dict.get('seasonal', pd.DataFrame())
        if seasonal.empty:
            logger.warning("No seasonal data found for distribution analysis")
            return
        
        # Check if position column exists
        if 'position' not in seasonal.columns:
            logger.warning("No position column in seasonal data")
            return
        
        # Create violin plots of fantasy points by position
        self._plot_fantasy_points_by_position(seasonal)
        
        # Create histograms of age distributions by position
        if 'age' in seasonal.columns:
            self._plot_age_distribution(seasonal)
        
        # Create position-specific distribution plots
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_data = seasonal[seasonal['position'] == position].copy()
            if not pos_data.empty:
                self._plot_position_distributions(pos_data, position)
        
        # Create distribution of games played
        if 'games' in seasonal.columns:
            self._plot_games_distribution(seasonal)
        
        logger.info("Data distribution visualizations created")
    
    def _plot_fantasy_points_by_position(self, data):
        """
        Plot fantasy points distribution by position
        
        Parameters:
        -----------
        data : DataFrame
            Seasonal data
        """
        if 'fantasy_points' not in data.columns:
            logger.warning("No fantasy_points column in data")
            return
        
        # Output directory
        dist_dir = os.path.join(self.output_dir, 'overall', 'distributions')
        
        # Create violin plot of fantasy points by position
        plt.figure(figsize=(12, 8))
        
        # Filter to main positions and non-zero fantasy points
        pos_filter = data['position'].isin(['QB', 'RB', 'WR', 'TE'])
        point_filter = data['fantasy_points'] > 0
        
        # Apply filters
        filtered_data = data[pos_filter & point_filter].copy()
        
        if filtered_data.empty:
            logger.warning("No data after filtering for fantasy points distribution")
            return
        
        # Create violin plot
        ax = sns.violinplot(x='position', y='fantasy_points', hue='position', data=filtered_data,
                palette=self.positions_palette, inner='quartile', legend=False)
        
        # Add title and labels
        plt.title('Fantasy Points Distribution by Position', fontsize=16, pad=20)
        plt.xlabel('Position', fontsize=14)
        plt.ylabel('Fantasy Points', fontsize=14)
        
        # Add position averages as text
        for i, pos in enumerate(['QB', 'RB', 'WR', 'TE']):
            if pos in filtered_data['position'].values:
                avg = filtered_data[filtered_data['position'] == pos]['fantasy_points'].mean()
                ax.text(i, avg + 2, f"Avg: {avg:.1f}", ha='center', 
                    color='black', fontweight='bold', 
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(dist_dir, 'fantasy_points_violin.png'))
        plt.close()
        
        # Create a box plot version
        plt.figure(figsize=(12, 8))
        
        # Create box plot
        ax = sns.boxplot(x='position', y='fantasy_points', data=filtered_data,
                    palette=self.positions_palette, showfliers=False)
        
        # Add swarm plot for individual points
        sns.swarmplot(x='position', y='fantasy_points', data=filtered_data, 
                    color='black', alpha=0.5, size=2)
        
        # Add title and labels
        plt.title('Fantasy Points Distribution by Position (Box Plot)', fontsize=16, pad=20)
        plt.xlabel('Position', fontsize=14)
        plt.ylabel('Fantasy Points', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(dist_dir, 'fantasy_points_box.png'))
        plt.close()
        
        # Create a KDE plot version
        plt.figure(figsize=(12, 8))
        
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos in filtered_data['position'].values:
                pos_data = filtered_data[filtered_data['position'] == pos]
                sns.kdeplot(pos_data['fantasy_points'], label=pos, 
                        color=self.positions_palette.get(pos), 
                        fill=True, alpha=0.3)
        
        # Add title and labels
        plt.title('Fantasy Points Density by Position', fontsize=16, pad=20)
        plt.xlabel('Fantasy Points', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.legend(title='Position')
        
        plt.tight_layout()
        plt.savefig(os.path.join(dist_dir, 'fantasy_points_density.png'))
        plt.close()
        
        logger.info("Fantasy points distribution visualizations created")
    
    def _plot_age_distribution(self, data):
        """
        Plot age distribution by position
        
        Parameters:
        -----------
        data : DataFrame
            Seasonal data
        """
        # Filter to valid ages and main positions
        age_filter = (data['age'] > 0) & (data['age'] < 50)  # Filter out unrealistic ages
        pos_filter = data['position'].isin(['QB', 'RB', 'WR', 'TE'])
        
        # Apply filters
        filtered_data = data[age_filter & pos_filter].copy()
        
        if filtered_data.empty:
            logger.warning("No data after filtering for age distribution")
            return
        
        # Output directory
        dist_dir = os.path.join(self.output_dir, 'overall', 'distributions')
        
        # Create histogram
        plt.figure(figsize=(12, 8))
        
        # Create histogram for each position
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos in filtered_data['position'].values:
                pos_data = filtered_data[filtered_data['position'] == pos]
                sns.histplot(pos_data['age'], label=pos, 
                        color=self.positions_palette.get(pos), 
                        alpha=0.7, kde=True)
        
        # Add title and labels
        plt.title('Age Distribution by Position', fontsize=16, pad=20)
        plt.xlabel('Age', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.legend(title='Position')
        
        # Set x-axis limits to focus on relevant age range
        plt.xlim(20, 40)
        
        plt.tight_layout()
        plt.savefig(os.path.join(dist_dir, 'age_distribution.png'))
        plt.close()
        
        # Create boxplot version
        plt.figure(figsize=(12, 8))
        
        # Create box plot
        ax = sns.boxplot(x='position', y='age', data=filtered_data,
                    palette=self.positions_palette, showfliers=False)
        
        # Add swarm plot for individual points
        sns.swarmplot(x='position', y='age', data=filtered_data, 
                    color='black', alpha=0.5, size=4)
        
        # Add title and labels
        plt.title('Age Distribution by Position (Box Plot)', fontsize=16, pad=20)
        plt.xlabel('Position', fontsize=14)
        plt.ylabel('Age', fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(dist_dir, 'age_box.png'))
        plt.close()
        
        # Also save position-specific age distributions
        for pos in ['QB', 'RB', 'WR', 'TE']:
            pos_data = filtered_data[filtered_data['position'] == pos]
            if not pos_data.empty:
                pos_dir = os.path.join(self.output_dir, pos.lower(), 'distributions')
                
                plt.figure(figsize=(10, 6))
                sns.histplot(pos_data['age'], kde=True, color=self.positions_palette.get(pos))
                plt.title(f'Age Distribution for {pos}', fontsize=16, pad=20)
                plt.xlabel('Age', fontsize=14)
                plt.ylabel('Count', fontsize=14)
                plt.xlim(20, 40)
                plt.tight_layout()
                plt.savefig(os.path.join(pos_dir, 'age_distribution.png'))
                plt.close()
        
        logger.info("Age distribution visualizations created")
    
    def _plot_position_distributions(self, pos_data, position):
        """
        Plot key stat distributions for a specific position
        
        Parameters:
        -----------
        pos_data : DataFrame
            Position-specific data
        position : str
            Position name
        """
        # Output directory
        pos_dir = os.path.join(self.output_dir, position.lower(), 'distributions')
        
        # Select key stats based on position
        if position == 'QB':
            key_stats = ['passing_yards', 'passing_tds', 'interceptions', 'rushing_yards']
            if 'fantasy_points_per_game' in pos_data.columns:
                key_stats.append('fantasy_points_per_game')
            if 'completion_percentage' in pos_data.columns:
                key_stats.append('completion_percentage')
        elif position == 'RB':
            key_stats = ['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards']
            if 'fantasy_points_per_game' in pos_data.columns:
                key_stats.append('fantasy_points_per_game')
            if 'yards_per_carry' in pos_data.columns:
                key_stats.append('yards_per_carry')
        elif position in ['WR', 'TE']:
            key_stats = ['receiving_yards', 'receiving_tds', 'receptions', 'targets']
            if 'fantasy_points_per_game' in pos_data.columns:
                key_stats.append('fantasy_points_per_game')
            if 'yards_per_reception' in pos_data.columns:
                key_stats.append('yards_per_reception')
        
        # Filter to stats that exist in the data
        available_stats = [stat for stat in key_stats if stat in pos_data.columns]
        
        if not available_stats:
            logger.warning(f"No key stats available for {position}")
            return
        
        # Create grid of histograms
        n_stats = len(available_stats)
        n_cols = min(2, n_stats)
        n_rows = (n_stats + n_cols - 1) // n_cols
        
        plt.figure(figsize=(12, 4 * n_rows))
        
        for i, stat in enumerate(available_stats):
            plt.subplot(n_rows, n_cols, i + 1)
            
            # Filter out zeros for better visualization
            stat_data = pos_data[pos_data[stat] > 0][stat]
            
            if len(stat_data) > 0:
                sns.histplot(stat_data, kde=True, color=self.positions_palette.get(position))
                
                # Add stat average as vertical line
                avg = stat_data.mean()
                plt.axvline(avg, color='red', linestyle='--', 
                        label=f'Avg: {avg:.1f}')
                
                # Add title and labels
                plt.title(f'{stat.replace("_", " ").title()} Distribution')
                plt.xlabel(stat.replace('_', ' ').title())
                plt.ylabel('Count')
                plt.legend()
            else:
                plt.text(0.5, 0.5, f'No data for {stat}', 
                    ha='center', va='center', 
                    transform=plt.gca().transAxes,
                    fontsize=14)
        
        plt.tight_layout()
        plt.savefig(os.path.join(pos_dir, 'stat_distributions.png'))
        plt.close()
        
        # Create individual distribution plots
        for stat in available_stats:
            # Filter out zeros for better visualization
            stat_data = pos_data[pos_data[stat] > 0][stat]
            
            if len(stat_data) > 0:
                plt.figure(figsize=(10, 6))
                sns.histplot(stat_data, kde=True, color=self.positions_palette.get(position))
                
                # Add stat average as vertical line
                avg = stat_data.mean()
                plt.axvline(avg, color='red', linestyle='--', 
                        label=f'Avg: {avg:.1f}')
                
                # Add title and labels
                plt.title(f'{stat.replace("_", " ").title()} Distribution for {position}')
                plt.xlabel(stat.replace('_', ' ').title())
                plt.ylabel('Count')
                plt.legend()
                
                plt.tight_layout()
                plt.savefig(os.path.join(pos_dir, f'{stat}_distribution.png'))
                plt.close()
        
        logger.info(f"{position} distribution visualizations created")
    
    def _plot_games_distribution(self, data):
        """
        Plot games played distribution
        
        Parameters:
        -----------
        data : DataFrame
            Seasonal data
        """
        # Filter to non-zero games and main positions
        games_filter = data['games'] > 0
        pos_filter = data['position'].isin(['QB', 'RB', 'WR', 'TE'])
        
        # Apply filters
        filtered_data = data[games_filter & pos_filter].copy()
        
        if filtered_data.empty:
            logger.warning("No data after filtering for games distribution")
            return
        
        # Output directory
        overall_dir = os.path.join(self.output_dir, 'overall', 'distributions')
        
        # Create histogram
        plt.figure(figsize=(12, 8))
        
        # Create histogram for each position
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos in filtered_data['position'].values:
                pos_data = filtered_data[filtered_data['position'] == pos]
                sns.histplot(pos_data['games'], label=pos, 
                        color=self.positions_palette.get(pos), 
                        alpha=0.7, kde=True, discrete=True)
        
        # Add title and labels
        plt.title('Games Played Distribution by Position', fontsize=16, pad=20)
        plt.xlabel('Games Played', fontsize=14)
        plt.ylabel('Count', fontsize=14)
        plt.legend(title='Position')
        
        plt.tight_layout()
        plt.savefig(os.path.join(overall_dir, 'games_distribution.png'))
        plt.close()
        
        # Create durability visualization if seasons_in_league exists
        if 'seasons_in_league' in filtered_data.columns:
            # Group by position and seasons_in_league
            durability = filtered_data.groupby(['position', 'seasons_in_league'])['games'].mean().reset_index()
            
            # Create line plot
            plt.figure(figsize=(12, 8))
            
            for pos in ['QB', 'RB', 'WR', 'TE']:
                if pos in durability['position'].values:
                    pos_data = durability[durability['position'] == pos]
                    
                    # Sort by seasons_in_league
                    pos_data = pos_data.sort_values('seasons_in_league')
                    
                    # Only plot if we have enough data points
                    if len(pos_data) > 1:
                        plt.plot(pos_data['seasons_in_league'], pos_data['games'], 
                            marker='o', label=pos, 
                            color=self.positions_palette.get(pos))
            
            # Add title and labels
            plt.title('Average Games Played by Career Length', fontsize=16, pad=20)
            plt.xlabel('Seasons in League', fontsize=14)
            plt.ylabel('Average Games Played', fontsize=14)
            plt.legend(title='Position')
            
            # Set y-axis to start at 0
            plt.ylim(bottom=0)
            
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(overall_dir, 'career_durability.png'))
            plt.close()
        
        logger.info("Games played distribution visualizations created")
    
    def explore_performance_trends(self):
        """
        Create visualizations of performance trends over time
        """
        logger.info("Creating performance trend visualizations")
        
        # Use seasonal data for trend analysis
        seasonal = self.data_dict.get('seasonal', pd.DataFrame())
        if seasonal.empty:
            logger.warning("No seasonal data found for trend analysis")
            return
        
        # Check for required columns
        if 'position' not in seasonal.columns or 'season' not in seasonal.columns:
            logger.warning("Missing position or season columns for trend analysis")
            return
        
        # Plot fantasy points trends by position and season
        if 'fantasy_points' in seasonal.columns:
            self._plot_fantasy_points_trends(seasonal)
        
        # Plot position-specific trends
        for position in ['QB', 'RB', 'WR', 'TE']:
            pos_data = seasonal[seasonal['position'] == position].copy()
            if not pos_data.empty:
                self._plot_position_trends(pos_data, position)
        
        # Plot age trends if available
        if 'age' in seasonal.columns and 'fantasy_points_per_game' in seasonal.columns:
            self._plot_age_trends(seasonal)
        
        logger.info("Performance trend visualizations created")
    
    def _plot_fantasy_points_trends(self, data):
        """
        Plot fantasy points trends by position and season
        
        Parameters:
        -----------
        data : DataFrame
            Seasonal data
        """
        # Group by position and season
        trend_data = data.groupby(['position', 'season']).agg({
            'fantasy_points': ['mean', 'median', 'std'],
            'player_id': 'count'
        }).reset_index()
        
        # Flatten column names
        trend_data.columns = ['position', 'season', 'avg_points', 'median_points', 'std_points', 'player_count']
        
        # Filter to main positions
        trend_data = trend_data[trend_data['position'].isin(['QB', 'RB', 'WR', 'TE'])]
        
        if trend_data.empty:
            logger.warning("No trend data after filtering")
            return
        
        # Output directory
        trend_dir = os.path.join(self.output_dir, 'overall', 'time_trends')
        
        # Create line plot for average fantasy points
        plt.figure(figsize=(12, 8))
        
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos in trend_data['position'].values:
                pos_data = trend_data[trend_data['position'] == pos]
                plt.plot(pos_data['season'], pos_data['avg_points'], 
                    marker='o', label=pos, 
                    color=self.positions_palette.get(pos))
        
        # Add title and labels
        plt.title('Average Fantasy Points by Position Over Time', fontsize=16, pad=20)
        plt.xlabel('Season', fontsize=14)
        plt.ylabel('Average Fantasy Points', fontsize=14)
        plt.legend(title='Position')
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(trend_dir, 'fantasy_points_trend.png'))
        plt.close()
        
        # Create line plot for median fantasy points
        plt.figure(figsize=(12, 8))
        
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos in trend_data['position'].values:
                pos_data = trend_data[trend_data['position'] == pos]
                plt.plot(pos_data['season'], pos_data['median_points'], 
                    marker='o', label=pos, 
                    color=self.positions_palette.get(pos))
        
        # Add title and labels
        plt.title('Median Fantasy Points by Position Over Time', fontsize=16, pad=20)
        plt.xlabel('Season', fontsize=14)
        plt.ylabel('Median Fantasy Points', fontsize=14)
        plt.legend(title='Position')
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(trend_dir, 'fantasy_points_median_trend.png'))
        plt.close()
        
        # Create line plot for player count
        plt.figure(figsize=(12, 8))
        
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos in trend_data['position'].values:
                pos_data = trend_data[trend_data['position'] == pos]
                plt.plot(pos_data['season'], pos_data['player_count'], 
                    marker='o', label=pos, 
                    color=self.positions_palette.get(pos))
        
        # Add title and labels
        plt.title('Player Count by Position Over Time', fontsize=16, pad=20)
        plt.xlabel('Season', fontsize=14)
        plt.ylabel('Number of Players', fontsize=14)
        plt.legend(title='Position')
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(trend_dir, 'player_count_trend.png'))
        plt.close()
        
        # Also save position-specific trends
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos in trend_data['position'].values:
                pos_dir = os.path.join(self.output_dir, pos.lower(), 'time_trends')
                pos_data = trend_data[trend_data['position'] == pos]
                
                plt.figure(figsize=(10, 6))
                plt.plot(pos_data['season'], pos_data['avg_points'], marker='o', label='Average', 
                    color=self.positions_palette.get(pos))
                plt.plot(pos_data['season'], pos_data['median_points'], marker='s', linestyle='--', label='Median', 
                    color=self.positions_palette.get(pos), alpha=0.7)
                plt.title(f'Fantasy Points Trend for {pos}', fontsize=16, pad=20)
                plt.xlabel('Season', fontsize=14)
                plt.ylabel('Fantasy Points', fontsize=14)
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(pos_dir, 'fantasy_points_trend.png'))
                plt.close()
        
        logger.info("Fantasy points trend visualizations created")
    
    def _plot_position_trends(self, pos_data, position):
        """
        Plot position-specific stat trends
        
        Parameters:
        -----------
        pos_data : DataFrame
            Position-specific data
        position : str
            Position name
        """
        # Output directory
        pos_dir = os.path.join(self.output_dir, position.lower(), 'time_trends')
        
        # Select key stats based on position
        if position == 'QB':
            key_stats = [
                ('passing_yards', 'Passing Yards'),
                ('passing_tds', 'Passing TDs'),
                ('interceptions', 'Interceptions'),
                ('rushing_yards', 'Rushing Yards')
            ]
            if 'completion_percentage' in pos_data.columns:
                key_stats.append(('completion_percentage', 'Completion Percentage'))
        elif position == 'RB':
            key_stats = [
                ('rushing_yards', 'Rushing Yards'),
                ('rushing_tds', 'Rushing TDs'),
                ('receptions', 'Receptions'),
                ('receiving_yards', 'Receiving Yards')
            ]
            if 'yards_per_carry' in pos_data.columns:
                key_stats.append(('yards_per_carry', 'Yards Per Carry'))
        elif position in ['WR', 'TE']:
            key_stats = [
                ('receiving_yards', 'Receiving Yards'),
                ('receiving_tds', 'Receiving TDs'),
                ('receptions', 'Receptions'),
                ('targets', 'Targets')
            ]
            if 'yards_per_reception' in pos_data.columns:
                key_stats.append(('yards_per_reception', 'Yards Per Reception'))
            if 'reception_rate' in pos_data.columns:
                key_stats.append(('reception_rate', 'Reception Rate'))
        
        # Filter to stats that exist in the data
        available_stats = [(col, label) for col, label in key_stats if col in pos_data.columns]
        
        if not available_stats:
            logger.warning(f"No key stats available for {position} trends")
            return
        
        # Create multiple trend plots
        for stat_col, stat_label in available_stats:
            # Group by season and calculate statistics
            trend_data = pos_data.groupby('season').agg({
                stat_col: ['mean', 'median', 'std']
            }).reset_index()
            
            # Flatten column names
            trend_data.columns = ['season', 'avg', 'median', 'std']
            
            if trend_data.empty:
                logger.warning(f"No trend data for {stat_label} ({position})")
                continue
            
            # Create figure
            plt.figure(figsize=(12, 8))
            
            # Plot average
            plt.plot(trend_data['season'], trend_data['avg'], 
                marker='o', label='Average', color=self.positions_palette.get(position))
            
            # Add confidence interval
            plt.fill_between(
                trend_data['season'],
                trend_data['avg'] - trend_data['std'],
                trend_data['avg'] + trend_data['std'],
                alpha=0.2, color=self.positions_palette.get(position),
                label='±1 Std Dev'
            )
            
            # Add median
            plt.plot(trend_data['season'], trend_data['median'], 
                   marker='s', linestyle='--', label='Median', 
                color=self.positions_palette.get(position), alpha=0.7)
            
            # Add title and labels
            plt.title(f'{stat_label} Trends for {position}', fontsize=16, pad=20)
            plt.xlabel('Season', fontsize=14)
            plt.ylabel(stat_label, fontsize=14)
            plt.legend()
            
            # Add grid
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Create safe filename
            safe_filename = stat_col.lower().replace(' ', '_').replace('/', '_')
            plt.savefig(os.path.join(pos_dir, f'{safe_filename}_trend.png'))
            plt.close()
        
        logger.info(f"{position} trend visualizations created")
    
    def _plot_age_trends(self, data):
        """
        Plot fantasy performance by age
        
        Parameters:
        -----------
        data : DataFrame
            Seasonal data
        """
        # Filter to valid ages and main positions
        age_filter = (data['age'] > 0) & (data['age'] < 45)
        pos_filter = data['position'].isin(['QB', 'RB', 'WR', 'TE'])
        
        # Apply filters
        filtered_data = data[age_filter & pos_filter].copy()
        
        if filtered_data.empty:
            logger.warning("No data after filtering for age trends")
            return
        
        # Group by position and age
        age_trend = filtered_data.groupby(['position', 'age']).agg({
            'fantasy_points_per_game': ['mean', 'median', 'std', 'count']
        }).reset_index()
        
        # Flatten column names
        age_trend.columns = ['position', 'age', 'avg_points', 'median_points', 'std_points', 'player_count']
        
        # Filter out ages with few players
        age_trend = age_trend[age_trend['player_count'] >= 5]
        
        if age_trend.empty:
            logger.warning("No age trend data after filtering")
            return
        
        # Output directory
        overall_dir = os.path.join(self.output_dir, 'overall', 'time_trends')
        
        # Create line plot
        plt.figure(figsize=(12, 8))
        
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos in age_trend['position'].values:
                pos_data = age_trend[age_trend['position'] == pos]
                
                # Sort by age
                pos_data = pos_data.sort_values('age')
                
                plt.plot(pos_data['age'], pos_data['avg_points'], 
                    marker='o', label=pos, 
                    color=self.positions_palette.get(pos))
                
                # Add confidence interval
                plt.fill_between(
                    pos_data['age'],
                    pos_data['avg_points'] - pos_data['std_points'] / np.sqrt(pos_data['player_count']),
                    pos_data['avg_points'] + pos_data['std_points'] / np.sqrt(pos_data['player_count']),
                    alpha=0.2, color=self.positions_palette.get(pos)
                )
        
        # Add title and labels
        plt.title('Fantasy Points per Game by Age and Position', fontsize=16, pad=20)
        plt.xlabel('Age', fontsize=14)
        plt.ylabel('Average Fantasy Points per Game', fontsize=14)
        plt.legend(title='Position')
        
        # Set y-axis to start at 0
        plt.ylim(bottom=0)
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(overall_dir, 'fantasy_points_by_age.png'))
        plt.close()
        
        # Also save position-specific age trends
        for pos in ['QB', 'RB', 'WR', 'TE']:
            if pos in age_trend['position'].values:
                pos_dir = os.path.join(self.output_dir, pos.lower(), 'time_trends')
                pos_data = age_trend[age_trend['position'] == pos]
                
                # Sort by age
                pos_data = pos_data.sort_values('age')
                
                plt.figure(figsize=(10, 6))
                plt.plot(pos_data['age'], pos_data['avg_points'], marker='o', 
                    color=self.positions_palette.get(pos))
                
                # Add confidence interval
                plt.fill_between(
                    pos_data['age'],
                    pos_data['avg_points'] - pos_data['std_points'] / np.sqrt(pos_data['player_count']),
                    pos_data['avg_points'] + pos_data['std_points'] / np.sqrt(pos_data['player_count']),
                    alpha=0.2, color=self.positions_palette.get(pos)
                )
                
                plt.title(f'Fantasy Points by Age for {pos}', fontsize=16, pad=20)
                plt.xlabel('Age', fontsize=14)
                plt.ylabel('Fantasy Points per Game', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(pos_dir, 'fantasy_points_by_age.png'))
                plt.close()
        
        logger.info("Age trend visualizations created")
    
    def visualize_clusters(self, feature_sets, cluster_models):
        """
        Visualize player clusters
        
        Parameters:
        -----------
        feature_sets : dict
            Dictionary of feature sets
        cluster_models : dict
            Dictionary of cluster models
        """
        if not feature_sets or not cluster_models:
            logger.warning("No feature sets or cluster models provided for visualization")
            return
        
        logger.info("Creating cluster visualizations")
        
        # Create clusters visualizations for each position
        for position in ['qb', 'rb', 'wr', 'te']:
            # Check if we have the required data
            train_key = f"{position}_train"
            model_key = position
            
            if train_key not in feature_sets or model_key not in cluster_models:
                logger.warning(f"Missing data for {position} cluster visualization")
                continue
            
            # Get training data with cluster assignments
            train_data = feature_sets[train_key]
            
            if 'cluster' not in train_data.columns:
                logger.warning(f"No cluster assignments in {position} training data")
                continue
            
            # Get cluster model
            model_dict = cluster_models[model_key]
            
            # Create output directory
            cluster_dir = os.path.join(self.output_dir, position, 'clusters')
            
            # Create cluster scatter plot
            self._plot_clusters(train_data, position, model_dict, cluster_dir)
            
            # Create radar chart for cluster characteristics
            self._create_cluster_radar_chart(train_data, position, model_dict, cluster_dir)
            
            # Create top players by cluster table
            self._create_top_players_table(train_data, position, cluster_dir)
            
            # Create cluster stat comparison
            self._create_cluster_stats_comparison(train_data, position, cluster_dir)
        
        logger.info("Cluster visualizations created")
    
    def _plot_clusters(self, data, position, model_dict, output_dir):
        """
        Create scatter plot of player clusters
        
        Parameters:
        -----------
        data : DataFrame
            Position data with cluster assignments
        position : str
            Position name
        model_dict : dict
            Dictionary containing cluster model components
        output_dir : str
            Output directory for visualizations
        """
        # Check if we have the necessary components
        if 'features' not in model_dict:
            logger.warning(f"No features defined for {position} cluster model")
            return
        
        features = model_dict['features']
        
        if len(features) < 2:
            logger.warning(f"Need at least 2 features for {position} cluster visualization")
            return
        
        # Get cluster colors and tiers
        if 'tier' in data.columns:
            # Use tier-based colors
            data['color'] = data['tier'].map(self.cluster_colors)
            data['color'].fillna('#999999', inplace=True)  # Gray for unknown tiers
        else:
            # Default to cluster-based coloring
            n_clusters = data['cluster'].nunique()
            colors = sns.color_palette("colorblind", n_clusters)
            color_map = {i: colors[i] for i in range(n_clusters)}
            data['color'] = data['cluster'].map(color_map)
        
        # Prepare data for visualization
        # If we have many features, use PCA for dimensionality reduction
        if len(features) > 2:
            # Filter to just the features used for clustering
            X = data[features].fillna(0)
            
            # Scale the data
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Apply PCA for 2D visualization
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_scaled)
            
            # Add PCA components to data
            data['pca1'] = X_pca[:, 0]
            data['pca2'] = X_pca[:, 1]
            
            # Create scatter plot
            plt.figure(figsize=(14, 10))
            
            # Plot points by tier/cluster
            for name, group in data.groupby('tier' if 'tier' in data.columns else 'cluster'):
                plt.scatter(
                    group['pca1'], group['pca2'],
                    label=name,
                    color=group['color'].iloc[0],
                    s=80, alpha=0.8, edgecolor='white', linewidth=0.5
                )
            
            # Add title and labels
            plt.title(f"{position.upper()} Player Clusters", fontsize=16, pad=20)
            plt.xlabel(f"Principal Component 1", fontsize=14)
            plt.ylabel(f"Principal Component 2", fontsize=14)
            
            # Add explained variance info
            explained_var = pca.explained_variance_ratio_
            plt.figtext(0.5, 0.01, 
                    f"Explained variance: PC1 = {explained_var[0]:.2%}, PC2 = {explained_var[1]:.2%}",
                    ha='center', fontsize=12)
        else:
            # Use the first two features directly
            plt.figure(figsize=(14, 10))
            
            # Plot points by tier/cluster
            for name, group in data.groupby('tier' if 'tier' in data.columns else 'cluster'):
                plt.scatter(
                    group[features[0]], group[features[1]],
                    label=name,
                    color=group['color'].iloc[0],
                    s=80, alpha=0.8, edgecolor='white', linewidth=0.5
                )
            
            # Add title and labels
            plt.title(f"{position.upper()} Player Clusters", fontsize=16, pad=20)
            plt.xlabel(features[0].replace('_', ' ').title(), fontsize=14)
            plt.ylabel(features[1].replace('_', ' ').title(), fontsize=14)
        
        # Add legend
        plt.legend(title="Tiers" if 'tier' in data.columns else "Clusters")
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add annotations for top players in each cluster
        if 'name' in data.columns and 'fantasy_points_per_game' in data.columns:
            groups = data.groupby('tier' if 'tier' in data.columns else 'cluster')
            
            for name, group in groups:
                # Get top 3 players by fantasy points
                top_players = group.nlargest(3, 'fantasy_points_per_game')
                
                for _, player in top_players.iterrows():
                    if 'pca1' in data.columns and 'pca2' in data.columns:
                        x, y = player['pca1'], player['pca2']
                    else:
                        x, y = player[features[0]], player[features[1]]
                    
                    # Add player name annotation
                    plt.annotate(
                        player['name'],
                        (x, y),
                        xytext=(5, 5),
                        textcoords='offset points',
                        fontsize=9,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray")
                    )
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'clusters.png'))
        plt.close()
        
        logger.info(f"{position} cluster scatter plot created")
    
    def _create_cluster_radar_chart(self, data, position, model_dict, output_dir):
        """
        Create radar chart showing cluster characteristics
        
        Parameters:
        -----------
        data : DataFrame
            Position data with cluster assignments
        position : str
            Position name
        model_dict : dict
            Dictionary containing cluster model components
        output_dir : str
            Output directory for visualizations
        """
        # Check if we have the necessary components
        if 'features' not in model_dict:
            logger.warning(f"No features defined for {position} cluster model")
            return
        
        features = model_dict['features']
        
        if len(features) < 3:
            logger.warning(f"Need at least 3 features for {position} radar chart")
            return
        
        # Calculate mean values by cluster/tier
        group_col = 'tier' if 'tier' in data.columns else 'cluster'
        
        # Calculate z-scores for each feature
        for feature in features:
            mean = data[feature].mean()
            std = data[feature].std()
            data[f"{feature}_z"] = (data[feature] - mean) / std if std > 0 else 0
        
        # Get z-score feature names
        z_features = [f"{feature}_z" for feature in features]
        
        # Group by cluster/tier and calculate means
        radar_data = data.groupby(group_col)[z_features].mean()
        
        # Rename columns back to original feature names
        radar_data.columns = [col[:-2] for col in radar_data.columns]
        
        # Create radar chart
        # Number of features
        N = len(features)
        
        # Create angle list for radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
        
        # Create pretty feature labels
        feature_labels = [f.replace('_', ' ').title() for f in features]
        
        # Add each cluster/tier to radar chart
        for tier, values in radar_data.iterrows():
            # Get values as list and close the loop
            values_list = values.values.tolist()
            values_list += values_list[:1]
            
            # Get color based on tier/cluster
            color = self.cluster_colors.get(tier, '#1f77b4') if group_col == 'tier' else None
            
            # Plot values
            ax.plot(angles, values_list, linewidth=2, linestyle='solid', label=tier, color=color)
            ax.fill(angles, values_list, alpha=0.1, color=color)
        
        # Set feature labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_labels)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Add title
        plt.title(f"{position.upper()} Cluster Characteristics", fontsize=16, pad=20)
        
        # Add reference lines
        plt.yticks([-2, -1, 0, 1, 2], ["-2σ", "-1σ", "Mean", "+1σ", "+2σ"])
        plt.ylim(-2.5, 2.5)
        
        plt.tight_layout()
        
        # Save figure
        plt.savefig(os.path.join(output_dir, f'radar_chart.png'))
        plt.close()
        
        logger.info(f"{position} cluster radar chart created")
    
    def _create_top_players_table(self, data, position, output_dir):
        """
        Create table of top players by cluster
        
        Parameters:
        -----------
        data : DataFrame
            Position data with cluster assignments
        position : str
            Position name
        output_dir : str
            Output directory for visualizations
        """
        # Check if we have required columns
        if 'name' not in data.columns or 'fantasy_points_per_game' not in data.columns:
            logger.warning(f"Missing required columns for {position} top players table")
            return
        
        # Group by cluster/tier
        group_col = 'tier' if 'tier' in data.columns else 'cluster'
        
        # Create dataframe for output
        top_players = []
        
        # Get top players for each cluster/tier
        for group_name, group_data in data.groupby(group_col):
            # Get top 5 players by fantasy points per game
            group_top = group_data.nlargest(5, 'fantasy_points_per_game')
            
            # Add cluster/tier info
            group_top = group_top.copy()
            group_top['tier_name'] = group_name
            
            # Select relevant columns
            cols = ['tier_name', 'name', 'fantasy_points_per_game', 'age']
            if 'season' in group_top.columns:
                cols.append('season')
            
            # Only include columns that exist
            cols = [col for col in cols if col in group_top.columns]
            
            top_players.append(group_top[cols])
        
        # Combine all clusters/tiers
        if top_players:
            top_df = pd.concat(top_players)
            
            # Save to CSV
            top_df.to_csv(os.path.join(output_dir, f'top_players_by_cluster.csv'), index=False)
            
            logger.info(f"{position} top players table created")
        else:
            logger.warning(f"No top players found for {position} clusters")
    
    def _create_cluster_stats_comparison(self, data, position, output_dir):
        """
        Create statistical comparison of clusters
        
        Parameters:
        -----------
        data : DataFrame
            Position data with cluster assignments
        position : str
            Position name
        output_dir : str
            Output directory for visualizations
        """
        # Group by cluster/tier
        group_col = 'tier' if 'tier' in data.columns else 'cluster'
        
        # Select relevant stats based on position
        if position == 'qb':
            stats = [
                'fantasy_points_per_game', 'passing_yards_per_game', 'passing_tds_per_game',
                'interceptions', 'completion_percentage', 'rushing_yards_per_game'
            ]
        elif position == 'rb':
            stats = [
                'fantasy_points_per_game', 'rushing_yards_per_game', 'rushing_tds_per_game',
                'receptions', 'receiving_yards_per_game', 'touches_per_game'
            ]
        elif position in ['wr', 'te']:
            stats = [
                'fantasy_points_per_game', 'receiving_yards_per_game', 'receiving_tds_per_game',
                'receptions', 'targets', 'yards_per_reception'
            ]
        else:
            stats = ['fantasy_points_per_game']
        
        # Filter to stats that exist in the data
        available_stats = [stat for stat in stats if stat in data.columns]
        
        if not available_stats:
            logger.warning(f"No relevant stats available for {position} cluster comparison")
            return
        
        # Calculate stats by cluster/tier
        cluster_stats = data.groupby(group_col)[available_stats].agg(['mean', 'min', 'max', 'count'])
        
        # Flatten column names
        cluster_stats.columns = ['_'.join(col) for col in cluster_stats.columns]
        
        # Save to CSV
        cluster_stats.reset_index().to_csv(
            os.path.join(output_dir, f'cluster_stats.csv'), 
            index=False
        )
        
        logger.info(f"{position} cluster stats comparison created")
    
    def run_all_visualizations(self, league_data=None):
        """
        Run all visualizations
        
        Parameters:
        -----------
        league_data : dict, optional
            League settings data
        """
        logger.info("Running all visualizations")
        
        # Visualize league settings if provided
        if league_data:
            self.explore_league_settings(league_data)
        
        # Explore data distributions
        self.explore_data_distributions()
        
        # Explore performance trends
        self.explore_performance_trends()
        
        # Visualize clusters if feature sets and cluster models are available
        if hasattr(self, 'feature_sets') and hasattr(self, 'cluster_models'):
            self.visualize_clusters(self.feature_sets, self.cluster_models)
        
        logger.info("All visualizations completed")
        
        
