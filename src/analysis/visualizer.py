import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FuncFormatter
import logging

logger = logging.getLogger(__name__)

class FantasyDataVisualizer:
    """
    Class for creating visualizations of fantasy football data
    """
    
    def __init__(self, data_dict, output_dir="data/outputs", current_year=2024):
        """
        Initialize with data dictionary and output directory
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary containing various dataframes of player data
        output_dir : str, optional
            Directory to save visualizations
        current_year : int, optional
            Current year for filtering active players
        """
        self.data_dict = data_dict
        self.output_dir = output_dir
        self.current_year = current_year
        
        # Create organized visualization directories
        self.viz_dirs = {
            'players': os.path.join(output_dir, 'player_analysis'),
            'positions': os.path.join(output_dir, 'position_analysis'),
            'trends': os.path.join(output_dir, 'trends_analysis'),
            'clusters': os.path.join(output_dir, 'clustering'),
            'advanced': os.path.join(output_dir, 'advanced_analytics')
        }
        
        for dir_path in self.viz_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
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
        
        # Fixed colors for clusters (consistent across visualizations)
        self.cluster_colors = {
            'Elite': '#1f77b4',       # blue
            'High Tier': '#2ca02c',   # green
            'Mid Tier': '#ff7f0e',    # orange
            'Low Tier': '#d62728',    # red
            'Bottom Tier': '#9467bd'  # purple
        }
        
        # Default cluster tiers (in order of quality)
        self.cluster_tiers = ['Elite', 'High Tier', 'Mid Tier', 'Low Tier', 'Bottom Tier']
    
    def _format_column_name(self, col_name):
        """Format column name for display on plots"""
        return " ".join(word.capitalize() for word in col_name.replace('_', ' ').split())
    
    def _add_plot_styling(self, ax, title, xlabel=None, ylabel=None, legend_title=None):
        """Add consistent styling to plot axes"""
        # Set title with nice font
        ax.set_title(title, fontsize=14, pad=15, fontweight='bold')
        
        # Set axis labels if provided
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
            
        # Style the grid
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Style the legend if it exists
        if legend_title and ax.get_legend():
            ax.legend(title=legend_title, frameon=True, facecolor='white', 
                     framealpha=0.9, edgecolor='lightgray')
            
        # Style the spines
        for spine in ax.spines.values():
            spine.set_color('lightgray')
            
    def _format_thousands(self, x, pos):
        """Format large numbers with K for thousands"""
        if x >= 1000:
            return f'{x/1000:.1f}K'
        return f'{x:.0f}'
    
    def _filter_active_players(self, df):
        """
        Filter to only active players based on latest season rosters
        
        Parameters:
        -----------
        df : DataFrame
            DataFrame to filter
            
        Returns:
        --------
        DataFrame
            Filtered DataFrame with only active players
        """
        # Active players should be in the current year's rosters
        if 'rosters' in self.data_dict and not self.data_dict['rosters'].empty:
            current_rosters = self.data_dict['rosters']
            if 'season' in current_rosters.columns:
                current_rosters = current_rosters[current_rosters['season'] == self.current_year]
            
            active_ids = set(current_rosters['player_id'].unique())
            return df[df['player_id'].isin(active_ids)]
        
        # Fallback - if we don't have roster data, use the latest season data
        if 'season' in df.columns:
            latest_season = df['season'].max()
            latest_df = df[df['season'] == latest_season]
            return latest_df
        
        return df
    
    def plot_point_distributions_by_position(self, data_key='all_seasonal', filter_by_top_clusters=True):
        """
        Create visualizations of fantasy points by position
        
        Parameters:
        -----------
        data_key : str, optional
            Key for the dataframe to use from data_dict
        filter_by_top_clusters : bool, optional
            Whether to filter by top 3 clusters
        """
        if data_key not in self.data_dict or self.data_dict[data_key].empty:
            logger.warning(f"Data key {data_key} not found or empty in data_dict.")
            return
        
        df = self.data_dict[data_key].copy()
        
        # Check if required columns exist
        if 'fantasy_points' not in df.columns:
            logger.warning("Required column 'fantasy_points' not found.")
            return
        
        # Check for position column
        if 'position' not in df.columns:
            logger.warning("Required column 'position' not found. Attempting to add from player_ids.")
            # Try to add position from player_ids
            if 'player_ids' in self.data_dict and 'player_id' in df.columns:
                player_ids = self.data_dict['player_ids']
                if 'gsis_id' in player_ids.columns and 'position' in player_ids.columns:
                    position_map = dict(zip(player_ids['gsis_id'], player_ids['position']))
                    df['position'] = df['player_id'].map(position_map)
                    logger.info(f"Added position column from player_ids. Position coverage: {df['position'].notna().mean():.1%}")
            
            # If still no position column, abort
            if 'position' not in df.columns:
                logger.warning("Cannot add position column. Aborting visualization.")
                return
        
        # Filter to active players
        df = self._filter_active_players(df)
        
        # Filter by top clusters if available and requested
        if filter_by_top_clusters and 'top_cluster' in df.columns:
            # Fix: Handle NaN values in top_cluster by filling with False
            df = df[df['top_cluster'].fillna(False)]
            logger.info(f"Filtered to {len(df)} players in top clusters")
        elif filter_by_top_clusters:
            logger.info("top_cluster column not found, skipping cluster filtering")
        
        # Filter to main positions
        positions = ['QB', 'RB', 'WR', 'TE']
        df_filtered = df[df['position'].isin(positions)].copy()
        
        if df_filtered.empty:
            logger.warning("No data remains after filtering.")
            return
            
        # Add a column for pretty labels
        df_filtered['Position'] = df_filtered['position']
        df_filtered['Fantasy Points'] = df_filtered['fantasy_points']
        
        # Add season label if available
        season_label = ""
        if 'season' in df_filtered.columns:
            season_label = f" ({df_filtered['season'].min()}-{df_filtered['season'].max()})"
        
        # Violin Plot
        plt.figure(figsize=(12, 8), dpi=120)
        ax = sns.violinplot(x='Position', y='Fantasy Points', data=df_filtered,
                         palette=self.positions_palette, inner='quartile')
        
        self._add_plot_styling(ax, f"Distribution of Fantasy Points by Position{season_label}", 
                             ylabel="Fantasy Points")
        
        # Add position averages as text
        for i, pos in enumerate(positions):
            if pos in df_filtered['position'].values:
                avg = df_filtered[df_filtered['position'] == pos]['fantasy_points'].mean()
                ax.text(i, avg + 2, f"Avg: {avg:.1f}", ha='center', 
                       color='black', fontweight='bold', 
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor='lightgray'))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dirs['positions'], 'points_violin.png'))
        plt.close()
        
        # Box Plot
        plt.figure(figsize=(12, 8), dpi=120)
        ax = sns.boxplot(x='Position', y='Fantasy Points', data=df_filtered,
                      palette=self.positions_palette, showfliers=False)
        
        # Add individual points as a swarm for more clarity
        sns.swarmplot(x='Position', y='Fantasy Points', data=df_filtered, 
                     color='black', alpha=0.5, size=4)
        
        title_suffix = " (Top 3 Clusters)" if filter_by_top_clusters and 'top_cluster' in df.columns else ""
        self._add_plot_styling(ax, f"Fantasy Points Distribution by Position{season_label}{title_suffix}", 
                             ylabel="Fantasy Points")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dirs['positions'], 'points_box.png'))
        plt.close()
        
        # Kernel Density Estimation Plot
        plt.figure(figsize=(12, 8), dpi=120)
        for position in positions:
            if position in df_filtered['position'].values:
                position_data = df_filtered[df_filtered['position'] == position]['Fantasy Points']
                sns.kdeplot(position_data, label=position, color=self.positions_palette[position],
                          fill=True, alpha=0.3)
        
        plt.title(f"Fantasy Points Density by Position{season_label}{title_suffix}", 
                 fontsize=14, pad=15, fontweight='bold')
        plt.xlabel("Fantasy Points", fontsize=12, labelpad=10)
        plt.ylabel("Density", fontsize=12, labelpad=10)
        plt.legend(title="Position", frameon=True)
        plt.grid(alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dirs['positions'], 'points_density.png'))
        plt.close()
    
    def plot_positional_stats(self, data_key='all_seasonal', position='QB', filter_by_top_clusters=True):
        """
        Create visualization of correlations between stats for a position
        
        Parameters:
        -----------
        data_key : str, optional
            Key for the dataframe to use from data_dict
        position : str
            Position to analyze
        filter_by_top_clusters : bool, optional
            Whether to filter by top 3 clusters
        """
        if data_key not in self.data_dict or self.data_dict[data_key].empty:
            logger.warning(f"Data key {data_key} not found or empty in data_dict.")
            return
        
        df = self.data_dict[data_key].copy()
        
        # Check if position column exists
        if 'position' not in df.columns:
            logger.warning("Position column not found in data. Attempting to add from player_ids.")
            # Try to add position from player_ids
            if 'player_ids' in self.data_dict and 'player_id' in df.columns:
                player_ids = self.data_dict['player_ids']
                if 'gsis_id' in player_ids.columns and 'position' in player_ids.columns:
                    position_map = dict(zip(player_ids['gsis_id'], player_ids['position']))
                    df['position'] = df['player_id'].map(position_map)
                    logger.info(f"Added position column from player_ids. Position coverage: {df['position'].notna().mean():.1%}")
            
            # If still no position column, abort
            if 'position' not in df.columns:
                logger.warning("Cannot add position column. Aborting visualization.")
                return
        
        # Filter to active players
        df = self._filter_active_players(df)
        
        # Filter by top clusters if available and requested
        if filter_by_top_clusters and 'top_cluster' in df.columns:
            # Fix: Handle NaN values in top_cluster by filling with False
            df = df[df['top_cluster'].fillna(False)]
            logger.info(f"Filtered to {len(df)} players in top clusters")
        elif filter_by_top_clusters:
            logger.info("top_cluster column not found, skipping cluster filtering")
        
        # Filter by position
        pos_df = df[df['position'] == position].copy()
        
        if pos_df.empty:
            logger.warning(f"No {position} players found after filtering.")
            return
        
        # Define metrics that directly relate to fantasy points (to be excluded from correlation)
        fantasy_points_related = [
            'fantasy_points', 'fantasy_points_ppr', 'calculated_points', 
            'points', 'points_', '_points'
        ]
        
        # Select relevant stats based on position - excluding fantasy point metrics
        stats_to_exclude = []
        for exclude_pattern in fantasy_points_related:
            stats_to_exclude.extend([col for col in pos_df.columns if exclude_pattern in col])
        
        if position == 'QB':
            base_stats = ['passing_yards', 'passing_tds', 'interceptions', 'rushing_yards', 'rushing_tds', 'sacks']
            derived_stats = ['completion_percentage', 'yards_per_attempt', 'td_percentage', 'int_percentage']
        elif position == 'RB':
            base_stats = ['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds']
            derived_stats = ['rushing_efficiency', 'yards_per_reception', 'total_yards_per_touch']
        elif position in ['WR', 'TE']:
            base_stats = ['receptions', 'targets', 'receiving_yards', 'receiving_tds', 'receiving_air_yards']
            derived_stats = ['yards_per_reception', 'reception_ratio', 'yards_per_target']
        else:
            base_stats = []
            derived_stats = []
        
        # Add fantasy points for correlation target
        stats = base_stats + derived_stats + ['fantasy_points']
        
        # Check which stats are available
        available_stats = [stat for stat in stats if stat in pos_df.columns]
        
        if len(available_stats) <= 1:
            logger.warning(f"Not enough stats available for {position}.")
            return
        
        # Create pretty labels for the stats
        pretty_stats = {stat: self._format_column_name(stat) for stat in available_stats}
        for stat in available_stats:
            pos_df[pretty_stats[stat]] = pos_df[stat]
        
        pretty_available_stats = [pretty_stats[stat] for stat in available_stats]
        
        # Create enhanced correlation heatmap
        plt.figure(figsize=(12, 10), dpi=120)
        corr_matrix = pos_df[pretty_available_stats].corr()
        
        # Create heatmap with improved aesthetics - show all combinations
        ax = sns.heatmap(corr_matrix, annot=True, cmap=self.cmap_red_blue,
                       vmin=-1, vmax=1, fmt='.2f', linewidths=0.5, 
                       annot_kws={"size": 10})
        
        title_suffix = " (Top 3 Clusters)" if filter_by_top_clusters and 'top_cluster' in df.columns else ""
        plt.title(f"Correlation of Key Stats for {position}{title_suffix}", fontsize=16, pad=20, fontweight='bold')
        
        # Improve tick labels
        plt.xticks(rotation=45, ha='right', rotation_mode='anchor')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        filename = f'correlation_{position.lower()}'
        if filter_by_top_clusters and 'top_cluster' in df.columns:
            filename += "_top_clusters"
        
        plt.savefig(os.path.join(self.viz_dirs['positions'], f'{filename}.png'))
        plt.close()
        
        # Create pairplot for detailed relationships if there aren't too many variables
        if len(pretty_available_stats) <= 5:  # Limit to avoid too many subplots
            plt.figure(figsize=(15, 12), dpi=120)
            g = sns.pairplot(pos_df[pretty_available_stats], diag_kind='kde', 
                         plot_kws={'alpha': 0.6, 's': 60, 'edgecolor': 'white'}, 
                         diag_kws={'shade': True, 'bw_adjust': 0.8})
            
            # Add title to the figure
            g.fig.suptitle(f'Pairwise Relationships for {position} Stats{title_suffix}', 
                         fontsize=16, y=1.02, fontweight='bold')
            
            plt.tight_layout()
            
            filename = f'pairplot_{position.lower()}'
            if filter_by_top_clusters and 'top_cluster' in df.columns:
                filename += "_top_clusters"
            
            plt.savefig(os.path.join(self.viz_dirs['positions'], f'{filename}.png'))
            plt.close()
    
    def plot_performance_clusters(self, data_key='all_seasonal', position='RB', n_clusters=5):
        """
        Create visualization of player clusters using K-means
        
        Parameters:
        -----------
        data_key : str, optional
            Key for the dataframe to use from data_dict
        position : str
            Position to analyze
        n_clusters : int, optional
            Number of clusters to create
        """
        if data_key not in self.data_dict or self.data_dict[data_key].empty:
            logger.warning(f"Data key {data_key} not found or empty in data_dict.")
            return
        
        df = self.data_dict[data_key].copy()
        
        # Check if position column exists
        if 'position' not in df.columns:
            logger.warning("Position column not found in data. Attempting to add from player_ids.")
            # Try to add position from player_ids
            if 'player_ids' in self.data_dict and 'player_id' in df.columns:
                player_ids = self.data_dict['player_ids']
                if 'gsis_id' in player_ids.columns and 'position' in player_ids.columns:
                    position_map = dict(zip(player_ids['gsis_id'], player_ids['position']))
                    df['position'] = df['player_id'].map(position_map)
                    logger.info(f"Added position column from player_ids. Position coverage: {df['position'].notna().mean():.1%}")
            
            # If still no position column, abort
            if 'position' not in df.columns:
                logger.warning("Cannot add position column. Aborting visualization.")
                return
        
        # Filter to active players for clustering analysis
        df = self._filter_active_players(df)
        
        # Filter by position
        pos_df = df[df['position'] == position].copy()
        
        if pos_df.empty or len(pos_df) < n_clusters:
            logger.warning(f"Not enough {position} players for clustering.")
            return
        
        # Select relevant features for clustering based on position
        if position == 'QB':
            features = ['passing_yards', 'passing_tds', 'interceptions', 'rushing_yards', 'rushing_tds']
            if 'completion_percentage' in pos_df.columns:
                features.append('completion_percentage')
            if 'yards_per_attempt' in pos_df.columns:
                features.append('yards_per_attempt')
        elif position == 'RB':
            features = ['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds']
            if 'rushing_efficiency' in pos_df.columns:
                features.append('rushing_efficiency')
            if 'total_yards_per_touch' in pos_df.columns:
                features.append('total_yards_per_touch')
        elif position in ['WR', 'TE']:
            features = ['receptions', 'targets', 'receiving_yards', 'receiving_tds']
            if 'receiving_air_yards' in pos_df.columns:
                features.append('receiving_air_yards')
            if 'yards_per_reception' in pos_df.columns:
                features.append('yards_per_reception')
            if 'reception_ratio' in pos_df.columns:
                features.append('reception_ratio')
        else:
            features = ['fantasy_points']
        
        # Normalize by games played if available
        if 'games' in pos_df.columns:
            # Create per-game features
            for feat in features:
                if feat in pos_df.columns:
                    pos_df[f'{feat}_per_game'] = pos_df[feat] / pos_df['games'].clip(lower=1)
            
            # Use per-game features for clustering
            features = [f'{feat}_per_game' for feat in features if f'{feat}_per_game' in pos_df.columns]
        
        # Check which features are available
        available_features = [feat for feat in features if feat in pos_df.columns]
        
        if len(available_features) < 2:
            logger.warning(f"Not enough features available for {position} clustering.")
            return
        
        # Prepare data for clustering
        X = pos_df[available_features].fillna(0)
        
        # Scale the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA for visualization
        n_components = 2
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        # Apply K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Create consistent cluster naming scheme
        # Name clusters by their fantasy performance
        pos_df['cluster'] = clusters
        cluster_stats = pos_df.groupby('cluster')['fantasy_points'].mean().sort_values(ascending=False)
        
        # Map cluster numbers to tier names
        cluster_names = {}
        for i, (cluster, _) in enumerate(cluster_stats.items()):
            if i < len(self.cluster_tiers):
                cluster_names[cluster] = self.cluster_tiers[i]
            else:
                cluster_names[cluster] = f"Tier {i+1}"
        
        # Identify top 3 clusters for filtering other visualizations
        top_clusters = cluster_stats.head(3).index.tolist()
        pos_df['top_cluster'] = pos_df['cluster'].apply(lambda x: x in top_clusters)
        
        # Update the original dataframe with cluster info for this position
        cluster_cols = ['cluster', 'top_cluster']
        for col in cluster_cols:
            if col in pos_df.columns:
                df.loc[pos_df.index, col] = pos_df[col]
        
        # Update the data dictionary with the clustered data
        self.data_dict[data_key] = df
        
        # Add cluster information to dataframe for visualization
        pos_df_clustered = pos_df.copy()
        pos_df_clustered['cluster_name'] = pos_df_clustered['cluster'].map(cluster_names)
        pos_df_clustered['pca1'] = X_pca[:, 0]
        pos_df_clustered['pca2'] = X_pca[:, 1]
        
        # Create enhanced scatter plot of clusters
        plt.figure(figsize=(14, 10), dpi=120)
        
        # Use consistent colors for clusters based on tier
        # Create a color map for each cluster
        cluster_palette = {i: self.cluster_colors.get(cluster_names[i], 'gray') for i in range(n_clusters)}
        
        # Create custom scatter plot with consistent colors
        for cluster in range(n_clusters):
            cluster_data = pos_df_clustered[pos_df_clustered['cluster'] == cluster]
            plt.scatter(cluster_data['pca1'], cluster_data['pca2'], 
                       s=100, alpha=0.8, edgecolor='white', linewidth=1,
                       label=cluster_names[cluster], 
                       color=cluster_palette[cluster])
        
        # Add legend with consistent colors
        plt.legend(title="Player Tiers", loc="upper right", frameon=True)
        
        # Add names and seasons for top players in each cluster
        if 'name' in pos_df_clustered.columns and 'fantasy_points' in pos_df_clustered.columns:
            for cluster in range(n_clusters):
                # Get top 3 players in this cluster
                cluster_players = pos_df_clustered[pos_df_clustered['cluster'] == cluster]
                top_players = cluster_players.nlargest(3, 'fantasy_points')
                
                for _, player in top_players.iterrows():
                    # Include season in label if available
                    player_label = player['name']
                    if 'season' in player:
                        player_label += f" ({int(player['season'])})"
                        
                    plt.annotate(player_label, 
                               (player['pca1'], player['pca2']),
                               xytext=(5, 5), textcoords='offset points',
                               fontsize=10, fontweight='bold',
                               bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray"))
        
        plt.title(f"{position} Player Clusters Based on Performance Metrics", fontsize=16, pad=20, fontweight='bold')
        plt.xlabel("Principal Component 1", fontsize=12, labelpad=10)
        plt.ylabel("Principal Component 2", fontsize=12, labelpad=10)
        plt.grid(alpha=0.3, linestyle='--')
        
        # Add explained variance as a subtitle
        explained_variance = pca.explained_variance_ratio_
        plt.figtext(0.5, 0.01, f"PCA explained variance: {explained_variance[0]:.2%} (PC1), {explained_variance[1]:.2%} (PC2)", 
                  ha="center", fontsize=10, fontstyle='italic')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dirs['clusters'], f'clusters_{position.lower()}.png'))
        plt.close()
        
        # Create a radar chart to show the characteristics of each cluster
        self._create_cluster_radar_chart(pos_df_clustered, available_features, cluster_names, position)
        
        # Output cluster statistics as a CSV
        cluster_stats = pos_df_clustered.groupby('cluster_name')[available_features].mean()
        if 'fantasy_points' in pos_df_clustered.columns:
            points_stats = pos_df_clustered.groupby('cluster_name')['fantasy_points'].agg(['mean', 'min', 'max'])
            cluster_stats = pd.concat([cluster_stats, points_stats], axis=1)
        
        cluster_stats.to_csv(os.path.join(self.viz_dirs['clusters'], f'cluster_stats_{position.lower()}.csv'))
        
        # Create a table showing top players in each cluster
        if 'name' in pos_df_clustered.columns and 'fantasy_points' in pos_df_clustered.columns:
            top_players_by_cluster = []
            
            for cluster in range(n_clusters):
                cluster_name = cluster_names[cluster]
                cluster_players = pos_df_clustered[pos_df_clustered['cluster'] == cluster]
                
                # Add season to output if available
                if 'season' in cluster_players.columns:
                    top_5 = cluster_players.nlargest(5, 'fantasy_points')[['name', 'season', 'fantasy_points']]
                else:
                    top_5 = cluster_players.nlargest(5, 'fantasy_points')[['name', 'fantasy_points']]
                    
                top_5['cluster'] = cluster_name
                top_players_by_cluster.append(top_5)
            
            top_players_df = pd.concat(top_players_by_cluster)
            top_players_df.to_csv(os.path.join(self.viz_dirs['clusters'], f'top_players_by_cluster_{position.lower()}.csv'), index=False)
    
    def _create_cluster_radar_chart(self, clustered_df, features, cluster_names, position):
        """
        Helper method to create radar charts for clusters
        
        Parameters:
        -----------
        clustered_df : DataFrame
            DataFrame with cluster assignments
        features : list
            List of features to plot
        cluster_names : dict
            Dictionary mapping cluster numbers to names
        position : str
            Position being analyzed
        """
        # Calculate means for each cluster
        radar_df = clustered_df.groupby('cluster')[features].mean()
        
        # Scale the data for radar chart
        scaler = StandardScaler()
        radar_scaled = pd.DataFrame(
            scaler.fit_transform(radar_df),
            index=radar_df.index,
            columns=radar_df.columns
        )
        
        # Create pretty feature names
        pretty_features = [self._format_column_name(feat) for feat in features]
        
        # Number of clusters and features
        n_clusters = len(radar_df)
        N = len(features)
        
        # Create a figure for all clusters together
        fig = plt.figure(figsize=(12, 10), dpi=120)
        
        # Compute angle for each feature
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Initialize subplot
        ax = plt.subplot(111, polar=True)
        
        # Add lines and points for each cluster with better styling
        for i, cluster in enumerate(radar_scaled.index):
            # Get values for this cluster
            values = radar_scaled.loc[cluster].values.tolist()
            values += values[:1]  # Close the loop
            
            # Get consistent color from our palette
            color = self.cluster_colors.get(cluster_names[cluster], 'gray')
            
            # Plot with nice styling
            ax.plot(angles, values, linewidth=2.5, linestyle='solid', 
                   label=cluster_names[cluster], color=color)
            ax.fill(angles, values, alpha=0.25, color=color)
            
            # Add points at each feature value
            ax.scatter(angles, values, s=60, color=color, edgecolor='white', linewidth=1, zorder=10)
        
        # Set feature labels with better formatting
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(pretty_features, fontsize=11)
        
        # Improve the tick labels
        ax.set_rlabel_position(0)
        plt.yticks([-2, -1, 0, 1, 2], ["-2σ", "-1σ", "Mean", "+1σ", "+2σ"], 
                  color="grey", size=9)
        plt.ylim(-2.5, 2.5)
        
        # Add title and legend
        plt.title(f"{position} Player Type Radar Chart", fontsize=16, y=1.1, fontweight='bold')
        
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), frameon=True,
                  title="Player Types")
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.viz_dirs['clusters'], f'radar_{position.lower()}.png'))
        plt.close()
    
    def run_all_visualizations(self):
        """
        Run all visualizations in one go
        """
        # Run clustering first to establish top 3 clusters for all positions
        for position in ['QB', 'RB', 'WR', 'TE']:
            logger.info(f"Creating clusters for {position}...")
            self.plot_performance_clusters(data_key='all_seasonal', position=position, n_clusters=5)
        
        # Player position distribution plots (filtered by top 3 clusters)
        logger.info("Creating position distribution plots...")
        self.plot_point_distributions_by_position(data_key='all_seasonal', filter_by_top_clusters=True)
        
        # Position-specific correlation plots (filtered by top 3 clusters)
        for position in ['QB', 'RB', 'WR', 'TE']:
            logger.info(f"Creating correlation plots for {position}...")
            self.plot_positional_stats(data_key='all_seasonal', position=position, filter_by_top_clusters=True)