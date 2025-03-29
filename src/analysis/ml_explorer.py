"""
Machine Learning Exploratory Data Analysis for Fantasy Football
This module provides advanced data exploration and visualization capabilities
specifically designed for ML-based fantasy football analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import mutual_info_regression
import os
import logging
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.gridspec as gridspec
from matplotlib.patches import Polygon
import matplotlib.cm as cm

# Configure logging
logger = logging.getLogger(__name__)

class MLExplorer:
    """
    Class for advanced exploratory data analysis and visualization for ML modeling
    """
    
    def __init__(self, data_dict, feature_sets, output_dir='data/outputs'):
        """
        Initialize with data and output directory
        
        Parameters:
        -----------
        data_dict : dict
            Dictionary containing processed data frames
        feature_sets : dict
            Dictionary containing engineered feature sets
        output_dir : str
            Directory to save visualizations
        """
        self.data_dict = data_dict
        self.feature_sets = feature_sets
        self.output_dir = output_dir
        
        # Define position colors for consistency
        self.position_colors = {
            'QB': '#1f77b4',  # blue
            'RB': '#ff7f0e',  # orange
            'WR': '#2ca02c',  # green
            'TE': '#d62728'   # red
        }
        
        logger.info(f"Initialized ML Explorer with {len(data_dict)} datasets and {len(feature_sets)} feature sets")
    
    def create_correlation_matrices(self):
        """
        Create correlation matrices for each position
        """
        logger.info("Creating correlation matrices")
        
        for position in ['qb', 'rb', 'wr', 'te']:
            # Check if we have data for this position
            train_key = f"{position}_train"
            if train_key not in self.feature_sets or self.feature_sets[train_key].empty:
                logger.warning(f"No {position} training data available")
                continue
            
            # Get training data
            train_data = self.feature_sets[train_key].copy()
            
            # Select relevant features for correlation analysis
            features = self._select_ml_features(train_data, position)
            
            if len(features) < 2:
                logger.warning(f"Not enough features for {position} correlation analysis")
                continue
            
            # Create correlation matrix
            corr_data = train_data[features].corr()
            
            # Create output directory
            corr_dir = os.path.join(self.output_dir, position, 'correlations')
            os.makedirs(corr_dir, exist_ok=True)
            
            # Save full correlation matrix to CSV
            corr_data.to_csv(os.path.join(corr_dir, f'full_correlation_matrix.csv'))
            
            # Create a clean version of the correlation matrix for the top correlations
            # Get the absolute correlation values
            corr_abs = corr_data.abs()
            
            # Get the upper triangle (excluding the diagonal)
            upper = corr_abs.where(np.triu(np.ones(corr_abs.shape), k=1).astype(bool))
            
            # Find top 10 correlations and their pairs
            top_corr_values = upper.unstack().sort_values(ascending=False)[:10]
            top_corr_pairs = [(i, j, corr_data.loc[i, j]) for i, j in top_corr_values.index]
            
            # Create a more focused correlation heatmap for the top correlations
            plt.figure(figsize=(12, 10))
            
            # Create a DataFrame with just the top correlations
            top_corr_df = pd.DataFrame({
                'Feature 1': [pair[0] for pair in top_corr_pairs],
                'Feature 2': [pair[1] for pair in top_corr_pairs],
                'Correlation': [pair[2] for pair in top_corr_pairs]
            })
            
            # Save top correlations to CSV
            top_corr_df.to_csv(os.path.join(corr_dir, f'top_correlations.csv'), index=False)
            
            # Create a horizontal bar chart for the top correlations
            plt.figure(figsize=(12, 8))
            bars = plt.barh(
                [f"{pair[0]} — {pair[1]}" for pair in top_corr_pairs],
                [abs(pair[2]) for pair in top_corr_pairs],
                color=[plt.cm.RdBu_r(0.5 * (1 + pair[2])) for pair in top_corr_pairs]
            )
            
            # Add correlation values to the bars
            for i, bar in enumerate(bars):
                plt.text(
                    bar.get_width() + 0.01,
                    bar.get_y() + bar.get_height()/2,
                    f"{top_corr_pairs[i][2]:.2f}",
                    va='center'
                )
            
            # Add title and labels
            plt.title(f'Top 10 Feature Correlations for {position.upper()}', fontsize=16, pad=20)
            plt.xlabel('Absolute Correlation Value', fontsize=12)
            plt.xlim(0, 1)
            
            # Adjust layout
            plt.tight_layout()
            
            # Save figure
            file_path = os.path.join(corr_dir, f'top_correlations_chart.png')
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            # Create a focused heatmap just for the top 10 correlations
            # Identify unique features in the top correlations
            unique_features = set()
            for f1, f2, _ in top_corr_pairs:
                unique_features.add(f1)
                unique_features.add(f2)
            unique_features = list(unique_features)
            
            # Create a smaller correlation matrix with just these features
            top_features_corr = corr_data.loc[unique_features, unique_features]
            
            # Create heatmap
            plt.figure(figsize=(12, 10))
            cmap = sns.diverging_palette(230, 20, as_cmap=True)
            
            sns.heatmap(top_features_corr, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f", annot_kws={"size": 10})
            
            # Add title
            plt.title(f'Correlation Matrix of Top Features for {position.upper()}', fontsize=16, pad=20)
            
            # Adjust labels for better readability
            plt.tight_layout()
            
            # Save figure
            file_path = os.path.join(corr_dir, f'top_features_correlation_matrix.png')
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Created correlation visualizations for {position}")
            
            # Still create the full correlation matrix, but save it separately for reference
            plt.figure(figsize=(16, 14))
            mask = np.triu(np.ones_like(corr_data, dtype=bool))
            
            sns.heatmap(corr_data, mask=mask, cmap=cmap, vmax=1, vmin=-1, center=0,
                    square=True, linewidths=.5, annot=True, fmt=".2f", annot_kws={"size": 7})
            
            # Add title
            plt.title(f'Full Feature Correlation Matrix for {position.upper()}', fontsize=16, pad=20)
            
            # Adjust labels for better readability
            plt.tight_layout()
            
            # Save figure
            file_path = os.path.join(corr_dir, f'full_correlation_matrix.png')
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            # Create hierarchical clustering of features
            self._create_feature_clustering(corr_data, position, corr_dir)
    
    def create_feature_importance_plots(self, target='fantasy_points_per_game'):
        """
        Create feature importance plots using random forest
        
        Parameters:
        -----------
        target : str
            Target variable to predict
        """
        logger.info(f"Creating feature importance plots for target: {target}")
        
        for position in ['qb', 'rb', 'wr', 'te']:
            # Check if we have data for this position
            train_key = f"{position}_train"
            if train_key not in self.feature_sets or self.feature_sets[train_key].empty:
                logger.warning(f"No {position} training data available")
                continue
            
            # Get training data
            train_data = self.feature_sets[train_key].copy()
            
            # Check if target exists
            if target not in train_data.columns:
                logger.warning(f"Target {target} not found in {position} data")
                continue
            
            # Select relevant features for importance analysis
            features = self._select_ml_features(train_data, position)
            
            if len(features) < 2:
                logger.warning(f"Not enough features for {position} importance analysis")
                continue
            
            # Create output directory
            importance_dir = os.path.join(self.output_dir, position, 'feature_importance')
            os.makedirs(importance_dir, exist_ok=True)
            
            try:
                # Prepare data
                X = train_data[features].fillna(0)
                y = train_data[target].fillna(0)
                
                # Train random forest for feature importance
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                # Get feature importances
                importances = rf.feature_importances_
                
                # Sort features by importance
                indices = np.argsort(importances)[::-1]
                
                # Limit to top 20 features for readability
                top_n = min(20, len(features))
                indices = indices[:top_n]
                
                # Create bar plot
                plt.figure(figsize=(14, 10))
                
                # Create bars with position-specific color
                bars = plt.barh(range(top_n), importances[indices], align='center',
                             color=self.position_colors.get(position.upper(), '#1f77b4'))
                
                # Add feature names as y-tick labels
                plt.yticks(range(top_n), [features[i] for i in indices])
                
                # Add labels and title
                plt.xlabel('Feature Importance')
                plt.title(f'Top {top_n} Features for Predicting {target} - {position.upper()}', fontsize=16, pad=20)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save figure
                file_path = os.path.join(importance_dir, f'random_forest_importance.png')
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close()
                
                logger.info(f"Created feature importance plot for {position}")
                
                # Calculate mutual information for additional perspective
                try:
                    mi_scores = mutual_info_regression(X, y)
                    mi_indices = np.argsort(mi_scores)[::-1][:top_n]
                    
                    # Create bar plot for mutual information
                    plt.figure(figsize=(14, 10))
                    
                    # Create bars
                    bars = plt.barh(range(top_n), mi_scores[mi_indices], align='center',
                                color=self.position_colors.get(position.upper(), '#1f77b4'))
                    
                    # Add feature names as y-tick labels
                    plt.yticks(range(top_n), [features[i] for i in mi_indices])
                    
                    # Add labels and title
                    plt.xlabel('Mutual Information Score')
                    plt.title(f'Top {top_n} Features by Mutual Information - {position.upper()}', fontsize=16, pad=20)
                    
                    # Adjust layout
                    plt.tight_layout()
                    
                    # Save figure
                    file_path = os.path.join(importance_dir, f'mutual_info.png')
                    plt.savefig(file_path, dpi=300, bbox_inches="tight")
                    plt.close()
                    
                    logger.info(f"Created mutual information plot for {position}")
                except Exception as e:
                    logger.error(f"Error creating mutual information plot for {position}: {e}")
            
            except Exception as e:
                logger.error(f"Error creating feature importance plot for {position}: {e}")
    
    def create_pair_plots(self):
        """
        Create pairwise relationship plots for key features
        """
        logger.info("Creating pair plots for key features")
        
        for position in ['qb', 'rb', 'wr', 'te']:
            # Check if we have data for this position
            train_key = f"{position}_train"
            if train_key not in self.feature_sets or self.feature_sets[train_key].empty:
                logger.warning(f"No {position} training data available")
                continue
            
            # Get training data
            train_data = self.feature_sets[train_key].copy()
            
            # Select a subset of important features for pairplot (too many would be unreadable)
            key_features = self._select_key_features(train_data, position)
            
            if len(key_features) < 2:
                logger.warning(f"Not enough key features for {position} pair plot")
                continue
            
            # Create output directory
            dist_dir = os.path.join(self.output_dir, position, 'distributions')
            os.makedirs(dist_dir, exist_ok=True)
            
            try:
                # Add fantasy points for coloring
                if 'fantasy_points_per_game' in train_data.columns:
                    key_features.append('fantasy_points_per_game')
                
                # Create pairplot
                plt.figure(figsize=(16, 14))
                
                # Create subset for pair plot
                pair_data = train_data[key_features].copy()
                
                # Create pair plot with Seaborn
                g = sns.pairplot(pair_data, height=2.5, diag_kind="kde", 
                              plot_kws={'alpha': 0.6, 's': 80, 'edgecolor': 'k'},
                              diag_kws={'fill': True})
                
                # Set title
                g.fig.suptitle(f'Feature Relationships for {position.upper()}', fontsize=16, y=1.02)
                
                # Save figure
                file_path = os.path.join(dist_dir, f'pair_plot.png')
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close()
                
                logger.info(f"Created pair plot for {position}")
                
                # Create correlation scatter plots for fantasy points
                if 'fantasy_points_per_game' in train_data.columns:
                    target = 'fantasy_points_per_game'
                    features = [f for f in key_features if f != target]
                    
                    # Calculate correlations
                    correlations = []
                    for feature in features:
                        try:
                            # Pearson (linear) correlation
                            pearson, _ = pearsonr(train_data[feature], train_data[target])
                            # Spearman (rank) correlation
                            spearman, _ = spearmanr(train_data[feature], train_data[target])
                            correlations.append((feature, pearson, spearman))
                        except:
                            pass
                    
                    # Sort by absolute Pearson correlation
                    correlations.sort(key=lambda x: abs(x[1]), reverse=True)
                    
                    # Create scatter plots for top correlations
                    n_plots = min(6, len(correlations))
                    if n_plots > 0:
                        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
                        axes = axes.flatten()
                        
                        for i in range(n_plots):
                            feature, pearson, spearman = correlations[i]
                            ax = axes[i]
                            
                            # Create scatter plot
                            ax.scatter(train_data[feature], train_data[target], 
                                    alpha=0.6, s=50, edgecolor='k',
                                    color=self.position_colors.get(position.upper(), '#1f77b4'))
                            
                            # Add regression line
                            sns.regplot(x=feature, y=target, data=train_data, 
                                     scatter=False, ax=ax, color='red')
                            
                            # Add correlation info
                            ax.text(0.05, 0.95, f"Pearson: {pearson:.3f}\nSpearman: {spearman:.3f}", 
                                  transform=ax.transAxes, fontsize=12,
                                  verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.5))
                            
                            # Set labels
                            ax.set_xlabel(feature)
                            ax.set_ylabel(target)
                            
                            # Set title
                            ax.set_title(f"{feature} vs {target}")
                        
                        # Turn off any unused subplots
                        for i in range(n_plots, len(axes)):
                            axes[i].axis('off')
                        
                        # Adjust layout
                        plt.tight_layout()
                        
                        # Save figure
                        file_path = os.path.join(dist_dir, f'correlation_scatters.png')
                        plt.savefig(file_path, dpi=300, bbox_inches="tight")
                        plt.close()
                        
                        logger.info(f"Created correlation scatter plots for {position}")
            
            except Exception as e:
                logger.error(f"Error creating pair plot for {position}: {e}")
    
    def create_cluster_visualizations(self):
        """
        Create enhanced cluster visualizations
        """
        logger.info("Creating enhanced cluster visualizations")
        
        # Define a more distinct color palette
        distinct_colors = {
            'Elite': '#e41a1c',        # Bright red
            'High Tier': '#377eb8',    # Blue
            'Mid Tier': '#4daf4a',     # Green
            'Low Tier': '#ff7f00',     # Orange
            'Bottom Tier': '#984ea3'   # Purple
        }
        
        for position in ['qb', 'rb', 'wr', 'te']:
            # Check if we have data for this position - using UNFILTERED data
            train_key = f"{position}_train"  # Use unfiltered data
            if train_key not in self.feature_sets or self.feature_sets[train_key].empty:
                logger.warning(f"No {position} training data available")
                continue
            
            # Get training data
            train_data = self.feature_sets[train_key].copy()
            
            # Check if we have cluster assignments
            if 'cluster' not in train_data.columns:
                logger.warning(f"No cluster assignments in {position} data")
                continue
            
            # Create output directory
            cluster_dir = os.path.join(self.output_dir, position, 'clusters')
            os.makedirs(cluster_dir, exist_ok=True)
            
            try:
                # Get features used for clustering
                features = self._select_ml_features(train_data, position)
                
                if len(features) < 2:
                    logger.warning(f"Not enough features for {position} cluster visualization")
                    continue
                
                # Apply PCA for 2D visualization
                X = train_data[features].fillna(0)
                
                # Scale the data
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Apply PCA
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)
                
                # Add PCA components to data
                train_data['pca1'] = X_pca[:, 0]
                train_data['pca2'] = X_pca[:, 1]
                
                # Create scatter plot with clusters
                plt.figure(figsize=(14, 10))
                
                # Get unique clusters and sort by fantasy points average
                if 'tier' in train_data.columns:
                    cluster_key = 'tier'
                    cluster_groups = train_data.groupby('tier')['fantasy_points_per_game'].mean().sort_values(ascending=False)
                else:
                    cluster_key = 'cluster'
                    cluster_groups = train_data.groupby('cluster')['fantasy_points_per_game'].mean().sort_values(ascending=False)
                
                # Mark the dropped tiers (will be shown but with different markers)
                if hasattr(self, 'dropped_tiers') and position in self.dropped_tiers:
                    dropped_clusters = self.dropped_tiers[position]
                else:
                    dropped_clusters = []
                
                # Add legend entries with cluster stats
                for cluster_name, avg_points in cluster_groups.items():
                    cluster_count = train_data[train_data[cluster_key] == cluster_name].shape[0]
                    is_dropped = cluster_name in dropped_clusters
                    
                    try:
                        tier_data = train_data[train_data[cluster_key] == cluster_name]
                        
                        # Get stats for this tier
                        avg_pts = tier_data['fantasy_points_per_game'].mean()
                        min_pts = tier_data['fantasy_points_per_game'].min()
                        max_pts = tier_data['fantasy_points_per_game'].max()
                        
                        # Create legend label with stats
                        label = f"{cluster_name} (n={cluster_count}, avg={avg_pts:.1f}, range={min_pts:.1f}-{max_pts:.1f})"
                        if is_dropped:
                            label += " - DROPPED"
                        
                        # Get color based on tier/cluster
                        if cluster_key == 'tier' and cluster_name in distinct_colors:
                            color = distinct_colors[cluster_name]
                        else:
                            # Use a distinct color for each cluster number
                            color_idx = list(cluster_groups.index).index(cluster_name)
                            cmap = plt.cm.get_cmap('tab10', len(cluster_groups))
                            color = cmap(color_idx)
                        
                        # Plot points for this cluster
                        plt.scatter(
                            tier_data['pca1'], tier_data['pca2'],
                            s=80 if not is_dropped else 50,  # Smaller markers for dropped tiers
                            alpha=0.7 if not is_dropped else 0.4,  # More transparent for dropped tiers
                            label=label,
                            color=color,
                            marker='o' if not is_dropped else 'x',  # X markers for dropped tiers
                            edgecolor='w' if not is_dropped else 'black',
                            linewidth=0.5
                        )
                        
                        # Add top player names if available
                        if 'name' in tier_data.columns:
                            top_players = tier_data.nlargest(5 if not is_dropped else 2, 'fantasy_points_per_game')
                            for _, player in top_players.iterrows():
                                plt.annotate(
                                    player['name'],
                                    (player['pca1'], player['pca2']),
                                    xytext=(5, 5),
                                    textcoords='offset points',
                                    fontsize=9,
                                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec="gray")
                                )
                    except Exception as e:
                        logger.error(f"Error plotting cluster {cluster_name}: {e}")
                
                # Add title and labels
                plt.title(f"{position.upper()} Player Clusters (All Tiers)", fontsize=16, pad=20)
                plt.xlabel(f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
                plt.ylabel(f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
                
                # Add legend
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                
                # Add grid
                plt.grid(True, linestyle='--', alpha=0.7)
                
                # Adjust layout
                plt.tight_layout()
                
                # Save figure
                file_path = os.path.join(cluster_dir, f'enhanced_clusters_all.png')
                plt.savefig(file_path, dpi=300, bbox_inches="tight")
                plt.close()
                
                logger.info(f"Created enhanced cluster visualization for {position}")
                
                # Create radar chart for cluster characteristics
                self._create_radar_plot(train_data, features, cluster_key, position, cluster_dir)
                
                # Create table of top players by cluster
                self._create_top_players_table(train_data, cluster_key, position, cluster_dir)
                
            except Exception as e:
                logger.error(f"Error creating cluster visualizations for {position}: {e}")

    def _create_radar_plot(self, data, features, cluster_key, position, output_dir):
        """
        Create radar plot showing cluster characteristics
        
        Parameters:
        -----------
        data : DataFrame
            Data with cluster assignments
        features : list
            Features to include in radar plot
        cluster_key : str
            Column name for cluster assignments
        position : str
            Position name
        output_dir : str
            Directory to save the radar plot
        """
        # Calculate z-scores for each feature
        z_scores = {}
        for feature in features:
            mean = data[feature].mean()
            std = data[feature].std()
            z_scores[f"{feature}_z"] = (data[feature] - mean) / std if std > 0 else 0

        # Add all columns at once
        data = pd.concat([data, pd.DataFrame(z_scores, index=data.index)], axis=1)
        
        # Get z-score feature names
        z_features = [f"{feature}_z" for feature in features]
        
        # Group by cluster and calculate means
        cluster_means = data.groupby(cluster_key)[z_features].mean()
        
        # Rename columns back to original feature names
        cluster_means.columns = [col[:-2] for col in cluster_means.columns]
        
        # Limit to 8 features for readability
        if len(features) > 8:
            # Find most variable features across clusters
            feature_variance = cluster_means.var().sort_values(ascending=False)
            top_features = feature_variance.index[:8].tolist()
            cluster_means = cluster_means[top_features]
        
        # Number of features
        N = len(cluster_means.columns)
        
        # Create angle list for radar chart
        angles = [n / float(N) * 2 * np.pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(polar=True))
        
        # Get feature labels (clean up names)
        feature_labels = [f.replace('_', ' ').title() for f in cluster_means.columns]
        
        # Get a color palette for clusters
        n_clusters = len(cluster_means)
        cluster_palette = sns.color_palette("viridis", n_clusters)
        
        # Add each cluster to radar chart
        for i, (cluster_name, values) in enumerate(cluster_means.iterrows()):
            values_list = values.values.tolist()
            values_list += values_list[:1]  # Close the loop
            
            # Get color for this cluster
            color = cluster_palette[i]
            
            # Plot values
            ax.plot(angles, values_list, linewidth=2, linestyle='solid', label=cluster_name, color=color)
            ax.fill(angles, values_list, alpha=0.1, color=color)
        
        # Add feature labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(feature_labels)
        
        # Add reference circles
        ax.set_yticks([-2, -1, 0, 1, 2])
        ax.set_yticklabels(['-2σ', '-1σ', 'Mean', '+1σ', '+2σ'])
        ax.set_ylim(-2.5, 2.5)
        
        # Add gridlines
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # Add title
        plt.title(f"{position.upper()} Cluster Characteristics", fontsize=16, pad=20)
        
        # Add legend
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        # Save figure
        file_path = os.path.join(output_dir, f'radar_chart.png')
        plt.savefig(file_path, dpi=300, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Created radar chart for {position} clusters")
    
    
    def _create_feature_clustering(self, corr_data, position, output_dir):
        """Create hierarchical clustering visualization of features"""
        try:
            plt.figure(figsize=(16, 10))
            
            # Replace NaN values with 0
            corr_values = corr_data.fillna(0).values
            # Replace infinite values with 0
            corr_values = np.nan_to_num(corr_values, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Calculate linkage with cleaned data
            corr_linkage = linkage(corr_values, method='ward', metric='euclidean')
            
            # Create dendrogram
            dendrogram(corr_linkage, labels=corr_data.columns, leaf_rotation=90)
            
            # Add title and explanation
            plt.title(f'Feature Clustering for {position.upper()}', fontsize=16, pad=20)
            plt.figtext(0.5, 0.01, 
                    "Features connected at lower heights are more closely correlated. "
                    "This clustering helps identify groups of similar features.",
                    ha='center', fontsize=12)
            
            # Adjust layout
            plt.tight_layout(rect=[0, 0.05, 1, 1])  # Make room for the explanation text
            
            # Save figure
            file_path = os.path.join(output_dir, f'feature_clusters.png')
            plt.savefig(file_path, dpi=300, bbox_inches="tight")
            plt.close()
            
            logger.info(f"Created feature clustering for {position}")
        except Exception as e:
            logger.error(f"Error creating feature clustering for {position}: {e}")

    def _create_top_players_table(self, data, cluster_key, position, output_dir):
        """
        Create CSV table with top players by cluster
        
        Parameters:
        -----------
        data : DataFrame
            Data with cluster assignments
        cluster_key : str
            Column name for cluster assignments
        position : str
            Position name
        output_dir : str
            Directory to save the table
        """
        if 'name' not in data.columns or 'fantasy_points_per_game' not in data.columns:
            logger.warning(f"Cannot create top players table for {position} - missing required columns")
            return
            
        # Create a list to hold player data
        top_players_data = []
        
        # Get unique clusters
        clusters = data[cluster_key].unique()
        
        # For each cluster, get top players
        for cluster in clusters:
            cluster_data = data[data[cluster_key] == cluster]
            
            # Get top 10 players by fantasy points per game
            top_players = cluster_data.nlargest(10, 'fantasy_points_per_game')
            
            # Add cluster info
            top_players = top_players.copy()
            top_players['cluster_name'] = cluster
            
            # Select relevant columns
            cols = ['name', 'cluster_name', 'fantasy_points_per_game', 'age']
            if 'season' in top_players.columns:
                cols.append('season')
            
            # Only include columns that exist
            available_cols = [col for col in cols if col in top_players.columns]
            
            # Add to combined list
            top_players_data.append(top_players[available_cols])
        
        # Combine all clusters
        if top_players_data:
            top_df = pd.concat(top_players_data)
            
            # Save to CSV
            file_path = os.path.join(output_dir, f'top_players_by_cluster.csv')
            top_df.to_csv(file_path, index=False)
            
            logger.info(f"Created top players table for {position}")
            
            # Also create a simple text file for easy reading
            file_path = os.path.join(output_dir, f'top_players_by_cluster.txt')
            with open(file_path, 'w') as f:
                f.write(f"Top Players by Cluster for {position.upper()}\n")
                f.write("="*50 + "\n\n")
                
                for cluster in clusters:
                    cluster_data = data[data[cluster_key] == cluster]
                    f.write(f"Cluster: {cluster}\n")
                    f.write("-"*50 + "\n")
                    
                    # Calculate cluster stats
                    avg_pts = cluster_data['fantasy_points_per_game'].mean()
                    f.write(f"Average Fantasy Points: {avg_pts:.2f}\n\n")
                    
                    # List top 10 players
                    top_players = cluster_data.nlargest(10, 'fantasy_points_per_game')
                    f.write("Top 10 Players:\n")
                    
                    for i, (_, player) in enumerate(top_players.iterrows(), 1):
                        f.write(f"{i}. {player['name']} - {player['fantasy_points_per_game']:.2f} pts/game\n")
                    
                    f.write("\n\n")
            
            logger.info(f"Created readable top players text file for {position}")
    
    def _select_ml_features(self, data, position):
        """
        Select features for ML analysis
        
        Parameters:
        -----------
        data : DataFrame
            Position data
        position : str
            Position name
            
        Returns:
        --------
        list
            List of selected feature names
        """
        # Exclude non-feature columns
        exclude_cols = [
            'player_id', 'name', 'team', 'season', 'cluster', 'tier',
            'pca1', 'pca2', 'gsis_id', 'position', 'fantasy_points',
            'fantasy_points_per_game', 'fantasy_points_ppr'
        ]
        
        # Get all numeric columns
        numeric_cols = data.select_dtypes(include=['number']).columns.tolist()
        
        # Filter out excluded columns
        features = [col for col in numeric_cols if col not in exclude_cols and not col.startswith('pca')]
        
        # Remove highly correlated features
        if len(features) > 2:
            # Calculate correlation matrix
            corr_matrix = data[features].corr().abs()
            
            # Find pairs with correlation > 0.95
            upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
            to_drop = [column for column in upper.columns if any(upper[column] > 0.95)]
            
            # Remove highly correlated features
            features = [f for f in features if f not in to_drop]
        
        return features
    
    def _select_key_features(self, data, position):
        """
        Select a subset of key features for visualization
        
        Parameters:
        -----------
        data : DataFrame
            Position data
        position : str
            Position name
            
        Returns:
        --------
        list
            List of key feature names
        """
        # Position-specific key features
        if position == 'qb':
            key_features = [
                'passing_yards_per_game', 'passing_tds_per_game', 
                'rushing_yards_per_game', 'fantasy_points_per_game',
                'completion_percentage', 'adjusted_yards_per_attempt'
            ]
        elif position == 'rb':
            key_features = [
                'rushing_yards_per_game', 'receiving_yards_per_game', 
                'total_tds', 'touches_per_game', 'fantasy_points_per_game',
                'yards_per_carry', 'fantasy_points_per_touch'
            ]
        elif position in ['wr', 'te']:
            key_features = [
                'receiving_yards_per_game', 'targets', 'receptions',
                'yards_per_reception', 'fantasy_points_per_game',
                'air_yards_share', 'racr'
            ]
        else:
            key_features = ['fantasy_points_per_game']
        
        # Filter to features that actually exist in the data
        available_features = [f for f in key_features if f in data.columns]
        
        # If we have few features, add more
        if len(available_features) < 3 and 'fantasy_points_per_game' in data.columns:
            # Find features most correlated with fantasy points
            numeric_cols = data.select_dtypes(include=['number']).columns
            correlations = []
            
            for col in numeric_cols:
                if col != 'fantasy_points_per_game' and col not in available_features:
                    correlation = data[col].corr(data['fantasy_points_per_game'])
                    if not pd.isna(correlation):
                        correlations.append((col, abs(correlation)))
            
            # Sort by correlation and add top features
            correlations.sort(key=lambda x: x[1], reverse=True)
            for col, _ in correlations[:5]:
                if col not in available_features:
                    available_features.append(col)
                if len(available_features) >= 6:
                    break
        
        return available_features
    
    def save_analysis_results(self):
        """
        Save all analysis results to files
        """
        logger.info("Saving analysis results to files")
        
        # Create base output directory for analysis data
        data_dir = os.path.join(self.output_dir, 'analysis_data')
        os.makedirs(data_dir, exist_ok=True)
        
        # Save correlation data
        self._save_correlation_data(data_dir)
        
        # Save feature importance data
        self._save_feature_importance_data(data_dir)
        
        # Save cluster data
        self._save_cluster_data(data_dir)
        
        logger.info("Analysis results saved to files")

    def _save_correlation_data(self, data_dir):
        """Save correlation matrices to files"""
        corr_dir = os.path.join(data_dir, 'correlations')
        os.makedirs(corr_dir, exist_ok=True)
        
        for position in ['qb', 'rb', 'wr', 'te']:
            train_key = f"{position}_train"
            if train_key not in self.feature_sets or self.feature_sets[train_key].empty:
                continue
            
            train_data = self.feature_sets[train_key]
            features = self._select_ml_features(train_data, position)
            
            if len(features) < 2:
                continue
            
            # Calculate correlation matrix
            corr_data = train_data[features].corr()
            
            # Save to CSV
            file_path = os.path.join(corr_dir, f'{position}_correlation.csv')
            corr_data.to_csv(file_path)
            
            logger.info(f"Saved {position} correlation data to {file_path}")

    def _save_feature_importance_data(self, data_dir):
        """Save feature importance rankings to files"""
        importance_dir = os.path.join(data_dir, 'feature_importance')
        os.makedirs(importance_dir, exist_ok=True)
        
        for position in ['qb', 'rb', 'wr', 'te']:
            train_key = f"{position}_train"
            if train_key not in self.feature_sets or self.feature_sets[train_key].empty:
                continue
            
            train_data = self.feature_sets[train_key]
            features = self._select_ml_features(train_data, position)
            
            if len(features) < 2 or 'fantasy_points_per_game' not in train_data.columns:
                continue
            
            try:
                # Calculate feature importance using Random Forest
                X = train_data[features].fillna(0)
                y = train_data['fantasy_points_per_game'].fillna(0)
                
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
                rf.fit(X, y)
                
                # Get feature importances
                importances = rf.feature_importances_
                
                # Create DataFrame with feature names and importances
                importance_df = pd.DataFrame({
                    'feature': features,
                    'importance': importances
                }).sort_values('importance', ascending=False)
                
                # Save to CSV
                file_path = os.path.join(importance_dir, f'{position}_feature_importance.csv')
                importance_df.to_csv(file_path, index=False)
                
                logger.info(f"Saved {position} feature importance data to {file_path}")
                
                # Also try to calculate mutual information
                try:
                    mi_scores = mutual_info_regression(X, y)
                    mi_df = pd.DataFrame({
                        'feature': features,
                        'mutual_info': mi_scores
                    }).sort_values('mutual_info', ascending=False)
                    
                    # Save to CSV
                    file_path = os.path.join(importance_dir, f'{position}_mutual_info.csv')
                    mi_df.to_csv(file_path, index=False)
                    
                    logger.info(f"Saved {position} mutual information data to {file_path}")
                except Exception as e:
                    logger.error(f"Error calculating mutual information for {position}: {e}")
                    
            except Exception as e:
                logger.error(f"Error calculating feature importance for {position}: {e}")

    def _save_cluster_data(self, data_dir):
        """Save cluster data to files"""
        cluster_dir = os.path.join(data_dir, 'clusters')
        os.makedirs(cluster_dir, exist_ok=True)
        
        for position in ['qb', 'rb', 'wr', 'te']:
            train_key = f"{position}_train"
            if train_key not in self.feature_sets or self.feature_sets[train_key].empty:
                continue
            
            train_data = self.feature_sets[train_key]
            
            if 'cluster' not in train_data.columns:
                continue
            
            # Calculate cluster statistics
            cluster_stats = train_data.groupby('cluster')['fantasy_points_per_game'].agg(
                ['mean', 'median', 'std', 'min', 'max', 'count']
            ).reset_index()
            
            # Save cluster stats to CSV
            file_path = os.path.join(cluster_dir, f'{position}_cluster_stats.csv')
            cluster_stats.to_csv(file_path, index=False)
            
            logger.info(f"Saved {position} cluster statistics to {file_path}")
            
            # Save top players per cluster
            if 'name' in train_data.columns:
                top_players = []
                for cluster in train_data['cluster'].unique():
                    cluster_data = train_data[train_data['cluster'] == cluster]
                    top_cluster_players = cluster_data.nlargest(10, 'fantasy_points_per_game')
                    top_cluster_players['cluster_rank'] = range(1, len(top_cluster_players) + 1)
                    top_players.append(top_cluster_players)
                
                if top_players:
                    top_players_df = pd.concat(top_players)
                    
                    # Select relevant columns
                    cols = ['name', 'cluster', 'fantasy_points_per_game', 'cluster_rank']
                    if 'tier' in top_players_df.columns:
                        cols.append('tier')
                    
                    # Save to CSV
                    file_path = os.path.join(cluster_dir, f'{position}_top_players_by_cluster.csv')
                    top_players_df[cols].to_csv(file_path, index=False)
                    
                    logger.info(f"Saved {position} top players by cluster to {file_path}")
    
    def run_advanced_eda(self):
        """
        Run all advanced EDA methods
        """
        logger.info("Running all advanced EDA analyses")
        
        # Create correlation matrices
        self.create_correlation_matrices()
        
        # Create feature importance plots
        self.create_feature_importance_plots()
        
        # Create pair plots
        self.create_pair_plots()
        
        # Create enhanced cluster visualizations
        self.create_cluster_visualizations()
        
        # Save analysis results to files
        self.save_analysis_results()
        
        logger.info("Advanced EDA analyses completed")