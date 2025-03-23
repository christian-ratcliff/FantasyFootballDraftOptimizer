import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel, mutual_info_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
import os
import logging
import joblib
from datetime import datetime

logger = logging.getLogger(__name__)

class FeatureEngineeringExtensions:
    """
    Extensions to the FeatureEngineering class providing additional functionality
    """
    
    def __init__(self, feature_engineering):
        """
        Initialize with a FeatureEngineering instance
        
        Parameters:
        -----------
        feature_engineering : FeatureEngineering
            An instance of the FeatureEngineering class
        """
        self.fe = feature_engineering
        self.train_data = feature_engineering.train_data
        self.test_data = feature_engineering.test_data
        self.feature_importances = None
        self.feature_correlations = None
        self.polynomial_features = None
        
        logger.info("FeatureEngineeringExtensions initialized")
    
    def analyze_feature_importance(self, target='fantasy_points_per_game', n_estimators=100, max_features=0.5, position=None):
        """
        Analyze feature importance using Random Forest - with robust error handling
        
        Parameters:
        -----------
        target : str, optional
            Target variable for importance analysis
        n_estimators : int, optional
            Number of trees in the Random Forest
        max_features : float or int, optional
            Maximum number of features to consider for each split
        position : str, optional
            Only analyze data for a specific position
                
        Returns:
        --------
        self : FeatureEngineeringExtensions
            Returns self for method chaining
        """
        df = self.train_data
        
        # Filter by position if specified
        if position and 'position' in df.columns:
            df = df[df['position'] == position].copy()
            logger.info(f"Analyzing feature importance for {target} (position: {position})")
        else:
            logger.info(f"Analyzing feature importance for {target} (all positions)")
        
        if len(df) == 0:
            logger.warning(f"No data for position {position}, creating empty feature importance DataFrame")
            self.feature_importances = pd.DataFrame(columns=['Feature', 'Importance', 'Mutual_Info', 'Composite_Score'])
            return self
        
        if target not in df.columns:
            logger.warning(f"Target {target} not found in data, skipping feature importance analysis")
            self.feature_importances = pd.DataFrame(columns=['Feature', 'Importance', 'Mutual_Info', 'Composite_Score'])
            return self
        
        # These are columns we should always exclude as they're not predictive
        always_exclude = [
            'player_id', 'name', 'gsis_id', 'team_id',
            # Fantasy point columns (to avoid leakage with our target)
            'fantasy_points', 'calculated_points', 'fantasy_points_ppr'
        ]
        
        # Get all columns
        all_columns = df.columns.tolist()
        
        # Remove always excluded features
        features_to_use = [col for col in all_columns if col not in always_exclude]
        
        # Remove the target itself to avoid leakage
        if target in features_to_use:
            features_to_use.remove(target)
        
        # Exclude additional columns that match the target to avoid data leakage
        if target == 'fantasy_points_per_game':
            features_to_use = [col for col in features_to_use if 'fantasy_points' not in col.lower()]
        
        logger.info(f"Using {len(features_to_use)} features for importance analysis")
        
        # Handle case where not enough features are available
        if len(features_to_use) < 5:
            logger.warning(f"Not enough features available for analysis (only {len(features_to_use)})")
            # Create dummy importance dataframe
            self.feature_importances = pd.DataFrame({
                'Feature': features_to_use,
                'Importance': [1.0/len(features_to_use)] * len(features_to_use),
                'Mutual_Info': [1.0/len(features_to_use)] * len(features_to_use),
                'Composite_Score': [1.0/len(features_to_use)] * len(features_to_use)
            }) if features_to_use else pd.DataFrame(columns=['Feature', 'Importance', 'Mutual_Info', 'Composite_Score'])
            return self
            
        # Prepare the data for analysis - fill NA values
        X = df[features_to_use].fillna(0)
        y = df[target].fillna(0)
        
        # Check for and handle any NaN values
        if X.isna().any().any() or y.isna().any():
            logger.warning("NaN values detected after filling. Cleaning data further.")
            # Drop columns with NaNs
            nan_cols = X.columns[X.isna().any()].tolist()
            if nan_cols:
                X = X.drop(columns=nan_cols)
                features_to_use = [f for f in features_to_use if f not in nan_cols]
        
        # Handle case where not enough samples
        if len(X) < 10:
            logger.warning(f"Not enough samples for analysis (only {len(X)})")
            self.feature_importances = pd.DataFrame({
                'Feature': features_to_use,
                'Importance': [1.0/len(features_to_use)] * len(features_to_use),
                'Mutual_Info': [1.0/len(features_to_use)] * len(features_to_use),
                'Composite_Score': [1.0/len(features_to_use)] * len(features_to_use)
            }) if features_to_use else pd.DataFrame(columns=['Feature', 'Importance', 'Mutual_Info', 'Composite_Score'])
            return self
        
        # Create and fit Random Forest model
        try:
            rf = RandomForestRegressor(n_estimators=n_estimators, max_features=max_features, 
                                    n_jobs=-1, random_state=42)
            
            # Log the number of features and samples
            logger.info(f"Using {len(features_to_use)} features and {len(X)} samples for importance analysis")
            
            # Fit the model
            rf.fit(X, y)
            
            # Get feature importances
            importances = rf.feature_importances_
            
            # Create DataFrame of feature importances
            feature_importance_df = pd.DataFrame({
                'Feature': features_to_use,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            # Store feature importances
            self.feature_importances = feature_importance_df
            
            # Calculate additional importance metrics: mutual information
            try:
                from sklearn.feature_selection import mutual_info_regression
                mi_scores = mutual_info_regression(X, y, random_state=42)
                mi_df = pd.DataFrame({
                    'Feature': features_to_use,
                    'Mutual_Info': mi_scores
                }).sort_values('Mutual_Info', ascending=False)
                
                # Merge both metrics
                self.feature_importances = pd.merge(
                    feature_importance_df, 
                    mi_df, 
                    on='Feature'
                )
                
                # Calculate a composite importance score
                self.feature_importances['Composite_Score'] = (
                    self.feature_importances['Importance'].rank(pct=True) * 0.7 + 
                    self.feature_importances['Mutual_Info'].rank(pct=True) * 0.3
                )
                
                # Sort by composite score
                self.feature_importances = self.feature_importances.sort_values(
                    'Composite_Score', ascending=False
                )
                
            except Exception as e:
                logger.warning(f"Error calculating mutual information: {e}")
                self.feature_importances['Mutual_Info'] = 0
                self.feature_importances['Composite_Score'] = self.feature_importances['Importance']
            
            # Log top features
            if not self.feature_importances.empty:
                logger.info(f"Top 10 most important features for {target}:")
                for i, (_, row) in enumerate(self.feature_importances.head(min(10, len(self.feature_importances))).iterrows()):
                    logger.info(f"{i+1}. {row['Feature']} - Score: {row['Composite_Score']:.4f}")
            else:
                logger.warning("No feature importances calculated")
                
        except Exception as e:
            logger.error(f"Error during feature importance analysis: {e}")
            self.feature_importances = pd.DataFrame(columns=['Feature', 'Importance', 'Mutual_Info', 'Composite_Score'])
        
        return self
    
    def visualize_feature_importance(self, top_n=30, output_dir="data/outputs/feature_importance"):
        """
        Visualize feature importance
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top features to visualize
        output_dir : str, optional
            Directory to save visualizations
            
        Returns:
        --------
        self : FeatureEngineeringExtensions
            Returns self for method chaining
        """
        if self.feature_importances is None:
            logger.warning("No feature importance data available. Run analyze_feature_importance() first.")
            return self
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Get top N features
        top_features = self.feature_importances.head(top_n)
        
        # Create plot
        plt.figure(figsize=(12, top_n * 0.3))
        
        # Plot composite score
        ax = sns.barplot(
            x='Composite_Score', 
            y='Feature', 
            data=top_features.sort_values('Composite_Score')
        )
        
        plt.title(f"Top {top_n} Most Important Features (Composite Score)", fontsize=14)
        plt.xlabel("Importance Score", fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"top_{top_n}_features_composite.png"), dpi=300)
        plt.close()
        
        # Create separate visualizations for RF importance and MI
        plt.figure(figsize=(12, top_n * 0.3))
        ax = sns.barplot(
            x='Importance', 
            y='Feature', 
            data=top_features.sort_values('Importance')
        )
        
        plt.title(f"Top {top_n} Most Important Features (Random Forest)", fontsize=14)
        plt.xlabel("Importance Score", fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"top_{top_n}_features_rf.png"), dpi=300)
        plt.close()
        
        # Mutual Information
        plt.figure(figsize=(12, top_n * 0.3))
        ax = sns.barplot(
            x='Mutual_Info', 
            y='Feature', 
            data=top_features.sort_values('Mutual_Info')
        )
        
        plt.title(f"Top {top_n} Most Important Features (Mutual Information)", fontsize=14)
        plt.xlabel("Mutual Information Score", fontsize=12)
        plt.tight_layout()
        
        # Save plot
        plt.savefig(os.path.join(output_dir, f"top_{top_n}_features_mi.png"), dpi=300)
        plt.close()
        
        # Also save the feature importance data
        self.feature_importances.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
        
        logger.info(f"Feature importance visualizations saved to {output_dir}")
        
        return self
    
    def analyze_feature_correlations(self, method='pearson', threshold=0.85):
        """
        Analyze feature correlations and identify highly correlated features
        
        Parameters:
        -----------
        method : str, optional
            Correlation method ('pearson', 'kendall', 'spearman')
        threshold : float, optional
            Correlation threshold for identifying highly correlated features
            
        Returns:
        --------
        self : FeatureEngineeringExtensions
            Returns self for method chaining
        """
        logger.info(f"Analyzing feature correlations using {method} method")
        
        # Select numeric features for correlation analysis
        numeric_features = self.train_data.select_dtypes(include=['number']).columns
        
        # Calculate correlation matrix
        correlation_matrix = self.train_data[numeric_features].corr(method=method)
        
        # Store correlation matrix
        self.feature_correlations = correlation_matrix
        
        # Identify highly correlated feature pairs
        correlated_features = set()
        for i in range(len(correlation_matrix.columns)):
            for j in range(i):
                if abs(correlation_matrix.iloc[i, j]) > threshold:
                    colname_i = correlation_matrix.columns[i]
                    colname_j = correlation_matrix.columns[j]
                    correlated_features.add((colname_i, colname_j, correlation_matrix.iloc[i, j]))
        
        # Convert to DataFrame
        if correlated_features:
            self.highly_correlated = pd.DataFrame(list(correlated_features), 
                                                columns=['Feature_1', 'Feature_2', 'Correlation'])
            self.highly_correlated = self.highly_correlated.sort_values('Correlation', ascending=False)
            
            # Log highly correlated features
            logger.info(f"Found {len(self.highly_correlated)} highly correlated feature pairs (|corr| > {threshold})")
            for i, (_, row) in enumerate(self.highly_correlated.head(10).iterrows()):
                logger.info(f"{i+1}. {row['Feature_1']} & {row['Feature_2']} - Corr: {row['Correlation']:.4f}")
        else:
            logger.info(f"No highly correlated feature pairs found (|corr| > {threshold})")
        
        return self
    
    def visualize_feature_correlations(self, top_n=15, output_dir="data/outputs/feature_correlation", target='fantasy_points_per_game'):
        """
        Visualize feature correlations - advanced stats only
        
        Parameters:
        -----------
        top_n : int, optional
            Maximum number of features to include in correlation matrix visualization
        output_dir : str, optional
            Directory to save visualizations
        target : str, optional
            Target variable to focus correlation analysis on
                
        Returns:
        --------
        self : FeatureEngineeringExtensions
            Returns self for method chaining
        """
        if self.feature_correlations is None:
            logger.warning("No correlation data available. Run analyze_feature_correlations() first.")
            return self
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Define patterns for BASIC features to completely exclude
        basic_stat_patterns = [
            # Raw counting stats and their per-game versions
            'passing_yards', 'rushing_yards', 'receiving_yards',
            'passing_tds', 'rushing_tds', 'receiving_tds',
            'carries', 'attempts', 'targets', 'receptions', 'completions',
            'interceptions', 'fumbles', 'sacks',
            'total_yards', 'total_td', 'yards_per_game', 'td_per_game',
            
            # Fantasy point related
            'fantasy_points', 'calculated_points', '_points', 'points_',
            
            # ID and metadata columns
            'player_id', 'name', 'team_id', 'season', 'gsis_id',
            
            # Position indicators
            '_cluster_', 'cluster_', 'position_'
        ]
        
        # Define ADVANCED features to INCLUDE - these are the only ones we want to show
        advanced_features = [
            # QB metrics
            'completion_percentage', 'yards_per_attempt', 'td_percentage', 'int_percentage',
            'passing_td_to_int_ratio', 'adjusted_yards_per_attempt', 'deep_ball_percentage',
            'epa_per_pass', 'aggressive_efficacy', 'sticks_targeting', 'air_yards_differential',
            
            # RB metrics
            'rushing_efficiency', 'yards_per_reception', 'total_yards_per_touch',
            'receiving_percentage', 'rushing_first_down_rate', 'epa_per_rush',
            'rush_over_expected_efficiency', 'decision_speed',
            
            # WR/TE metrics
            'reception_ratio', 'yards_per_target', 'air_yards_percentage', 
            'yac_percentage', 'epa_per_target', 'first_down_rate', 
            'separation_score', 'yac_ability', 'air_yard_share',
            
            # Custom metrics
            'ppr_boost', 'durability_score', 'decision_speed',
            
            # Trend metrics
            '_pct_change', '_vs_3yr_avg', '_weighted_avg', '_consistency',
            
            # Always include target
            target
        ]
        
        # Get all column names
        all_columns = list(self.feature_correlations.columns)
        
        # Filter out basic stats
        excluded_features = []
        for pattern in basic_stat_patterns:
            excluded_features.extend([col for col in all_columns if pattern in col.lower()])
        
        # Find all advanced features in the correlation matrix
        available_advanced = [feat for feat in advanced_features if feat in all_columns]
        
        # Also include any features with advanced patterns not explicitly listed
        advanced_patterns = ['_efficiency', '_ratio', '_percentage', 'epa', 'separation', 
                        'cushion', 'air_yards', 'yac', 'expected', 'over_expected',
                        'durability', 'consistency', 'differential']
        
        additional_advanced = []
        for pattern in advanced_patterns:
            for col in all_columns:
                if pattern in col.lower() and col not in available_advanced and col not in excluded_features:
                    additional_advanced.append(col)
        
        # Combine and limit to top_n features
        all_advanced = available_advanced + additional_advanced
        
        # Ensure target is included if available
        if target in all_columns and target not in all_advanced:
            all_advanced = [target] + all_advanced
        
        # Select top features based on importance if available
        if hasattr(self, 'feature_importances') and self.feature_importances is not None:
            important_advanced = [f for f in self.feature_importances['Feature'].tolist() 
                                if f in all_advanced][:top_n]
            
            # If we don't have enough important advanced features, add other advanced features
            if len(important_advanced) < top_n:
                remaining = [f for f in all_advanced if f not in important_advanced]
                important_advanced.extend(remaining[:top_n - len(important_advanced)])
            
            top_features = important_advanced[:top_n]
        else:
            # Without importance data, just use the advanced features we found
            top_features = all_advanced[:top_n]
        
        # Final check - ensure we only use advanced features
        top_features = [f for f in top_features 
                    if not any(pattern in f.lower() for pattern in basic_stat_patterns)]
        
        # Create correlation matrix with selected features
        if len(top_features) > 1:
            correlation_subset = self.feature_correlations.loc[top_features, top_features]
        else:
            logger.warning("Not enough advanced features for correlation matrix. Using top performing advanced features.")
            # Fall back to using advanced patterns to find features
            candidates = [col for col in all_columns 
                        if any(pattern in col.lower() for pattern in advanced_patterns) 
                        and not any(pattern in col.lower() for pattern in basic_stat_patterns)]
            
            # Select a reasonable number
            sel_columns = candidates[:min(top_n, len(candidates))]
            correlation_subset = self.feature_correlations.loc[sel_columns, sel_columns]
        
        # Create larger visualization with better readability
        plt.figure(figsize=(24, 20))
        
        # Custom diverging colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Create better labels - shorten and format feature names
        labels = [self._format_column_name(col) for col in correlation_subset.columns]
        
        # Draw the heatmap with improved formatting
        sns.heatmap(
            correlation_subset,
            cmap=cmap,
            vmax=1.0,
            vmin=-1.0,
            center=0,
            square=True,
            linewidths=0.8,
            annot=True,
            fmt=".2f",
            cbar_kws={"shrink": .8},
            annot_kws={"size": 14},
            xticklabels=labels,
            yticklabels=labels
        )
        
        plt.title("Advanced Metrics Correlation Matrix", fontsize=20, pad=20)
        
        # Increase font size for axis labels
        plt.xticks(fontsize=14, rotation=45, ha='right', rotation_mode='anchor')
        plt.yticks(fontsize=14)
        
        plt.tight_layout()
        
        # Save visualization with "advanced" in the name
        plt.savefig(os.path.join(output_dir, f"advanced_correlation_matrix.png"), dpi=300)
        plt.close()
        
        # Create pairplot for a smaller set of key metrics if available 
        if len(correlation_subset.columns) >= 4:  # Need at least 4 variables for a meaningful pairplot
            try:
                # Select a subset of the most important variables for better visibility
                pairplot_cols = correlation_subset.columns[:min(6, len(correlation_subset.columns))]
                
                # Get the data for these columns
                plot_data = self.train_data[pairplot_cols].copy()
                
                # Create more readable column names
                plot_data.columns = [self._format_column_name(col) for col in pairplot_cols]
                
                # Create pairplot
                plt.figure(figsize=(16, 14))
                g = sns.pairplot(
                    plot_data, 
                    diag_kind='kde',
                    plot_kws={'alpha': 0.6, 's': 60, 'edgecolor': 'white'},
                    diag_kws={'shade': True, 'bw_adjust': 0.8}
                )
                
                # Add title
                g.fig.suptitle('Advanced Metrics Relationships', fontsize=20, y=1.02)
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "advanced_metrics_pairplot.png"), dpi=300)
                plt.close()
            except Exception as e:
                logger.warning(f"Error creating pairplot: {e}")
        
        # Visualize highly correlated feature pairs if available (advanced metrics only)
        if hasattr(self, 'highly_correlated') and not self.highly_correlated.empty:
            # Filter to advanced metric pairs only
            advanced_pairs = self.highly_correlated[
                (~self.highly_correlated['Feature_1'].str.contains('|'.join(basic_stat_patterns), case=False)) &
                (~self.highly_correlated['Feature_2'].str.contains('|'.join(basic_stat_patterns), case=False))
            ]
            
            if not advanced_pairs.empty:
                # Get top correlated advanced pairs
                top_advanced_pairs = advanced_pairs.head(15)  # Show top 15
                
                # Create visualization - make it larger
                plt.figure(figsize=(14, 12))
                
                # Plot absolute correlation values
                top_advanced_pairs['Abs_Correlation'] = top_advanced_pairs['Correlation'].abs()
                top_advanced_pairs = top_advanced_pairs.sort_values('Abs_Correlation')
                
                # Create better pair labels
                def format_feature_name(name):
                    """Format feature name to be more readable"""
                    # Remove common prefixes/suffixes and keep most meaningful parts
                    parts = name.split('_')
                    if len(parts) > 2:
                        # For interact features, keep first and last meaningful part
                        if 'interact' in parts:
                            meaningful = [p for p in parts if p not in ['interact', 'per', 'game']]
                            if len(meaningful) >= 2:
                                return ' '.join(word.capitalize() for word in meaningful[:2])
                    
                    # Default formatting - capitalize words
                    return ' '.join(word.capitalize() for word in name.split('_')[:2])
                
                # Create shorter, more readable pair labels
                top_advanced_pairs['Pair'] = top_advanced_pairs.apply(
                    lambda row: f"{format_feature_name(row['Feature_1'])} & {format_feature_name(row['Feature_2'])}", 
                    axis=1
                )
                
                # Create barplot with improved styling
                ax = sns.barplot(
                    x='Abs_Correlation',
                    y='Pair',
                    data=top_advanced_pairs,
                    palette='viridis'
                )
                
                # Add correlation values at the end of bars
                for i, row in enumerate(top_advanced_pairs.itertuples()):
                    ax.text(
                        row.Abs_Correlation + 0.01, 
                        i, 
                        f"{row.Abs_Correlation:.2f}", 
                        va='center',
                        fontsize=12
                    )
                
                plt.title("Top Correlated Advanced Metric Pairs", fontsize=18, pad=20)
                plt.xlabel("Absolute Correlation", fontsize=14, labelpad=10)
                plt.ylabel("Feature Pair", fontsize=14, labelpad=10)
                
                # Set x-axis to go from 0 to 1.1 to make room for labels
                plt.xlim(0, 1.1)
                
                plt.tight_layout()
                
                # Save visualization
                plt.savefig(os.path.join(output_dir, "advanced_correlated_pairs.png"), dpi=300)
                plt.close()
                
                # Save filtered correlation data
                advanced_pairs.to_csv(os.path.join(output_dir, "advanced_correlated_features.csv"), index=False)
        
        # Save filtered correlation matrix
        correlation_subset.to_csv(os.path.join(output_dir, "advanced_feature_correlations.csv"))
        
        logger.info(f"Advanced feature correlation visualizations saved to {output_dir}")
        
        return self
    
    
    
    def create_interaction_features(self, top_n=20, interaction_type='multiplication'):
        """
        Create interaction features based on top features
        
        Parameters:
        -----------
        top_n : int, optional
            Number of top features to use for creating interactions
        interaction_type : str, optional
            Type of interaction ('multiplication', 'division', 'addition', 'subtraction')
            
        Returns:
        --------
        self : FeatureEngineeringExtensions
            Returns self for method chaining
        """
        logger.info(f"Creating interaction features using {interaction_type}")
        
        # Get top features if feature importance is available
        if hasattr(self, 'feature_importances') and self.feature_importances is not None:
            top_features = self.feature_importances['Feature'].head(top_n).tolist()
        else:
            # Otherwise, use all numeric features
            numeric_features = self.fe.train_data.select_dtypes(include=['number']).columns
            top_features = [col for col in numeric_features if col not in ['player_id', 'season']]
            top_features = top_features[:min(top_n, len(top_features))]
        
        logger.info(f"Using top {len(top_features)} features for interaction")
        
        # List of features that exist in both train and test data
        valid_features = [f for f in top_features if f in self.train_data.columns and f in self.test_data.columns]
        
        # Create interaction features selectively based on position logic
        if 'position' in self.train_data.columns:
            positions = self.train_data['position'].unique()
            
            for position in positions:
                # Get position-specific features
                if position == 'QB':
                    pos_features = [f for f in valid_features if any(kw in f for kw in 
                                                                 ['passing', 'rushing', 'completion', 'attempt'])]
                elif position == 'RB':
                    pos_features = [f for f in valid_features if any(kw in f for kw in 
                                                                 ['rushing', 'receiving', 'carries', 'receptions'])]
                elif position in ['WR', 'TE']:
                    pos_features = [f for f in valid_features if any(kw in f for kw in 
                                                                 ['receiving', 'targets', 'receptions', 'air_yards'])]
                else:
                    pos_features = valid_features
                
                # Limit to reasonable number to avoid combinatorial explosion
                pos_features = pos_features[:min(10, len(pos_features))]
                
                # Count existing interactions
                prev_count = len(self.train_data.columns)
                
                # Create pairwise interactions
                for i, feat_i in enumerate(pos_features):
                    for j, feat_j in enumerate(pos_features):
                        if i < j:  # Only do pairs once
                            try:
                                # Create different types of interactions
                                if interaction_type == 'multiplication':
                                    self.train_data[f'{feat_i}_{feat_j}_interact'] = self.train_data[feat_i] * self.train_data[feat_j]
                                    self.test_data[f'{feat_i}_{feat_j}_interact'] = self.test_data[feat_i] * self.test_data[feat_j]
                                
                                elif interaction_type == 'division':
                                    # Avoid division by zero
                                    self.train_data[f'{feat_i}_by_{feat_j}'] = self.train_data[feat_i] / self.train_data[feat_j].replace(0, 1)
                                    self.test_data[f'{feat_i}_by_{feat_j}'] = self.test_data[feat_i] / self.test_data[feat_j].replace(0, 1)
                                
                                elif interaction_type == 'addition':
                                    self.train_data[f'{feat_i}_plus_{feat_j}'] = self.train_data[feat_i] + self.train_data[feat_j]
                                    self.test_data[f'{feat_i}_plus_{feat_j}'] = self.test_data[feat_i] + self.test_data[feat_j]
                                
                                elif interaction_type == 'subtraction':
                                    self.train_data[f'{feat_i}_minus_{feat_j}'] = self.train_data[feat_i] - self.train_data[feat_j]
                                    self.test_data[f'{feat_i}_minus_{feat_j}'] = self.test_data[feat_i] - self.test_data[feat_j]
                            
                            except Exception as e:
                                logger.warning(f"Error creating interaction for {feat_i} and {feat_j}: {e}")
                
                new_count = len(self.train_data.columns)
                logger.info(f"Created {new_count - prev_count} interaction features for position {position}")
        
        else:
            # If no position column, create interactions for all top features
            for i, feat_i in enumerate(valid_features):
                for j, feat_j in enumerate(valid_features):
                    if i < j:  # Only do pairs once
                        try:
                            # Create different types of interactions
                            if interaction_type == 'multiplication':
                                self.train_data[f'{feat_i}_{feat_j}_interact'] = self.train_data[feat_i] * self.train_data[feat_j]
                                self.test_data[f'{feat_i}_{feat_j}_interact'] = self.test_data[feat_i] * self.test_data[feat_j]
                            
                            elif interaction_type == 'division':
                                # Avoid division by zero
                                self.train_data[f'{feat_i}_by_{feat_j}'] = self.train_data[feat_i] / self.train_data[feat_j].replace(0, 1)
                                self.test_data[f'{feat_i}_by_{feat_j}'] = self.test_data[feat_i] / self.test_data[feat_j].replace(0, 1)
                            
                            elif interaction_type == 'addition':
                                self.train_data[f'{feat_i}_plus_{feat_j}'] = self.train_data[feat_i] + self.train_data[feat_j]
                                self.test_data[f'{feat_i}_plus_{feat_j}'] = self.test_data[feat_i] + self.test_data[feat_j]
                            
                            elif interaction_type == 'subtraction':
                                self.train_data[f'{feat_i}_minus_{feat_j}'] = self.train_data[feat_i] - self.train_data[feat_j]
                                self.test_data[f'{feat_i}_minus_{feat_j}'] = self.test_data[feat_i] - self.test_data[feat_j]
                        
                        except Exception as e:
                            logger.warning(f"Error creating interaction for {feat_i} and {feat_j}: {e}")
        
        # Update the feature engineering instance's data
        self.fe.train_data = self.train_data
        self.fe.test_data = self.test_data
        
        logger.info(f"Interaction features created successfully")
        
        return self
    
    def create_fantasy_scoring_features(self, scoring_rules):
        """
        Create features specific to fantasy scoring rules
        
        Parameters:
        -----------
        scoring_rules : dict
            Dictionary of scoring rules (e.g., {'passing_yards': 0.04, 'passing_td': 4})
            
        Returns:
        --------
        self : FeatureEngineeringExtensions
            Returns self for method chaining
        """
        logger.info("Creating fantasy scoring-specific features")
        
        # Create a "points per X" feature for each scoring category
        scoring_map = {
            'PY': 'passing_yards',
            'PTD': 'passing_tds', 
            'INT': 'interceptions',
            'RY': 'rushing_yards', 
            'RTD': 'rushing_tds',
            'REC': 'receptions',
            'REY': 'receiving_yards',
            'RETD': 'receiving_tds',
            'FUML': 'fumbles_lost',
            'PTL': 'two_pt_conversions'
        }
        
        # Iterate through scoring rules
        for category, points in scoring_rules.items():
            if category in scoring_map and scoring_map[category] in self.train_data.columns:
                stat_col = scoring_map[category]
                
                # Create points feature
                self.train_data[f'{stat_col}_points'] = self.train_data[stat_col] * points
                self.test_data[f'{stat_col}_points'] = self.test_data[stat_col] * points
                
                # Create per-game feature
                if 'games' in self.train_data.columns:
                    self.train_data[f'{stat_col}_points_per_game'] = self.train_data[f'{stat_col}_points'] / self.train_data['games'].clip(lower=1)
                    self.test_data[f'{stat_col}_points_per_game'] = self.test_data[f'{stat_col}_points'] / self.test_data['games'].clip(lower=1)
                
                logger.info(f"Created scoring feature for {category}: {points} points per {stat_col}")
        
        # Create composite scoring features
        # Pass-heavy scoring indicator
        if all(col in scoring_rules for col in ['PY', 'PTD']):
            pass_points_ratio = scoring_rules['PTD'] / (scoring_rules['RTD'] if 'RTD' in scoring_rules else 6)
            self.train_data['pass_heavy_scoring'] = pass_points_ratio > 0.5
            self.test_data['pass_heavy_scoring'] = pass_points_ratio > 0.5
        
        # PPR indicator (0 = standard, 0.5 = half, 1 = full)
        if 'REC' in scoring_rules:
            ppr_value = scoring_rules['REC']
            self.train_data['ppr_value'] = ppr_value
            self.test_data['ppr_value'] = ppr_value
            
            # PPR boost for receiving backs and slot receivers
            if 'receptions' in self.train_data.columns and 'position' in self.train_data.columns:
                self.train_data['ppr_boost'] = np.where(
                    self.train_data['position'] == 'RB',
                    self.train_data['receptions'] * ppr_value,
                    np.where(
                        self.train_data['position'].isin(['WR', 'TE']),
                        self.train_data['receptions'] * ppr_value * 0.5,  # Half effect for WR/TE
                        0
                    )
                )
                
                self.test_data['ppr_boost'] = np.where(
                    self.test_data['position'] == 'RB',
                    self.test_data['receptions'] * ppr_value,
                    np.where(
                        self.test_data['position'].isin(['WR', 'TE']),
                        self.test_data['receptions'] * ppr_value * 0.5,  # Half effect for WR/TE
                        0
                    )
                )
        
        # Create expected fantasy points based on scoring rules
        self.train_data['expected_fantasy_points'] = 0
        self.test_data['expected_fantasy_points'] = 0
        
        for category, points in scoring_rules.items():
            if category in scoring_map and scoring_map[category] in self.train_data.columns:
                stat_col = scoring_map[category]
                
                # Special handling for negative points (like interceptions)
                multiplier = -1 if category in ['INT', 'FUML'] else 1
                
                self.train_data['expected_fantasy_points'] += self.train_data[stat_col] * points * multiplier
                self.test_data['expected_fantasy_points'] += self.test_data[stat_col] * points * multiplier
        
        # Update the feature engineering instance's data
        self.fe.train_data = self.train_data
        self.fe.test_data = self.test_data
        
        logger.info("Fantasy scoring features created successfully")
        
        return self
    
    def create_advanced_ngs_features(self, ngs_data=None):
        """
        Create advanced features from Next Gen Stats data
        
        Parameters:
        -----------
        ngs_data : dict, optional
            Dictionary containing NGS data by type
            
        Returns:
        --------
        self : FeatureEngineeringExtensions
            Returns self for method chaining
        """
        logger.info("Creating advanced NGS features")
        
        if ngs_data is None or not any(not df.empty for df in ngs_data.values() if isinstance(df, pd.DataFrame)):
            logger.warning("No NGS data provided, skipping advanced NGS features")
            return self
        
        # Combine base features with NGS data
        for df_name in ['train', 'test']:
            df = getattr(self, df_name + '_data')
            
            # Check if we have player_id and position columns
            if 'player_id' not in df.columns or 'position' not in df.columns:
                logger.warning(f"player_id or position column missing in {df_name} data, skipping advanced NGS features")
                continue
            
            # Create different features based on position
            qb_indices = df[df['position'] == 'QB'].index
            rb_indices = df[df['position'] == 'RB'].index
            wr_te_indices = df[df['position'].isin(['WR', 'TE'])].index
            
            # QB advanced features
            if 'ngs_passing' in ngs_data and not ngs_data['ngs_passing'].empty:
                # Index NGS data by player GSIS ID and season
                ngs_passing = ngs_data['ngs_passing'].copy()
                
                if 'player_gsis_id' in ngs_passing.columns and 'season' in ngs_passing.columns:
                    # Extract useful metrics
                    ngs_pass_features = ['aggressiveness', 'avg_time_to_throw', 
                                       'avg_air_yards_to_sticks', 'avg_air_yards_differential',
                                       'completion_percentage_above_expectation']
                    
                    # Create new composite metrics
                    ngs_passing['aggressive_efficacy'] = ngs_passing['aggressiveness'] * ngs_passing['completion_percentage_above_expectation'].clip(lower=0)
                    ngs_passing['sticks_targeting'] = np.where(
                        ngs_passing['avg_air_yards_to_sticks'] > 0, 
                        1, 
                        np.where(
                            ngs_passing['avg_air_yards_to_sticks'] < -3,
                            -1,
                            0
                        )
                    )
                    
                    # Group by player and season
                    ngs_pass_grouped = ngs_passing.groupby(['player_gsis_id', 'season']).agg({
                        'aggressive_efficacy': 'mean',
                        'sticks_targeting': 'mean',
                        'avg_air_yards_differential': 'mean'
                    }).reset_index()
                    
                    # Merge with main dataframe for QBs
                    for idx in qb_indices:
                        player_id = df.loc[idx, 'player_id']
                        season = df.loc[idx, 'season'] if 'season' in df.columns else None
                        
                        if season:
                            # Find matching NGS data
                            ngs_match = ngs_pass_grouped[
                                (ngs_pass_grouped['player_gsis_id'] == player_id) & 
                                (ngs_pass_grouped['season'] == season)
                            ]
                            
                            if not ngs_match.empty:
                                # Add NGS features
                                df.loc[idx, 'aggressive_efficacy'] = ngs_match['aggressive_efficacy'].values[0]
                                df.loc[idx, 'sticks_targeting'] = ngs_match['sticks_targeting'].values[0]
                                df.loc[idx, 'air_yards_differential'] = ngs_match['avg_air_yards_differential'].values[0]
            
            # RB advanced features
            if 'ngs_rushing' in ngs_data and not ngs_data['ngs_rushing'].empty:
                # Index NGS data by player GSIS ID and season
                ngs_rushing = ngs_data['ngs_rushing'].copy()
                
                if 'player_gsis_id' in ngs_rushing.columns and 'season' in ngs_rushing.columns:
                    # Create new composite metrics
                    if 'rush_yards_over_expected_per_att' in ngs_rushing.columns and 'percent_attempts_gte_eight_defenders' in ngs_rushing.columns:
                        ngs_rushing['rush_over_expected_efficiency'] = ngs_rushing['rush_yards_over_expected_per_att'] * \
                                                                     (1 + ngs_rushing['percent_attempts_gte_eight_defenders'] / 100)
                    
                    # Group by player and season
                    ngs_rush_grouped = ngs_rushing.groupby(['player_gsis_id', 'season']).agg({
                        'rush_over_expected_efficiency': 'mean' if 'rush_over_expected_efficiency' in ngs_rushing.columns else 'count',
                        'efficiency': 'mean' if 'efficiency' in ngs_rushing.columns else 'count',
                        'avg_time_to_los': 'mean' if 'avg_time_to_los' in ngs_rushing.columns else 'count'
                    }).reset_index()
                    
                    # Add decision speed metric (lower time to LOS is better)
                    if 'avg_time_to_los' in ngs_rushing.columns:
                        ngs_rush_grouped['decision_speed'] = (2.5 - ngs_rush_grouped['avg_time_to_los'].clip(upper=2.5)) / 2.5 * 100
                    
                    # Merge with main dataframe for RBs
                    for idx in rb_indices:
                        player_id = df.loc[idx, 'player_id']
                        season = df.loc[idx, 'season'] if 'season' in df.columns else None
                        
                        if season:
                            # Find matching NGS data
                            ngs_match = ngs_rush_grouped[
                                (ngs_rush_grouped['player_gsis_id'] == player_id) & 
                                (ngs_rush_grouped['season'] == season)
                            ]
                            
                            if not ngs_match.empty:
                                # Add NGS features
                                for col in ['rush_over_expected_efficiency', 'efficiency', 'decision_speed']:
                                    if col in ngs_match.columns:
                                        df.loc[idx, col] = ngs_match[col].values[0]
            
            # WR/TE advanced features
            if 'ngs_receiving' in ngs_data and not ngs_data['ngs_receiving'].empty:
                # Index NGS data by player GSIS ID and season
                ngs_receiving = ngs_data['ngs_receiving'].copy()
                
                if 'player_gsis_id' in ngs_receiving.columns and 'season' in ngs_receiving.columns:
                    # Create new composite metrics
                    if 'avg_separation' in ngs_receiving.columns and 'avg_cushion' in ngs_receiving.columns:
                        ngs_receiving['separation_score'] = ngs_receiving['avg_separation'] * \
                                                          (1 + ngs_receiving['avg_cushion'] / 10)
                    
                    if 'avg_yac_above_expectation' in ngs_receiving.columns and 'avg_expected_yac' in ngs_receiving.columns:
                        ngs_receiving['yac_ability'] = ngs_receiving['avg_yac_above_expectation'] * \
                                                     (ngs_receiving['avg_expected_yac'] / 5)
                    
                    # Group by player and season
                    agg_dict = {}
                    for col in ['separation_score', 'yac_ability', 'catch_percentage', 'percent_share_of_intended_air_yards']:
                        if col in ngs_receiving.columns:
                            agg_dict[col] = 'mean'
                        else:
                            agg_dict[col] = 'count'
                    
                    ngs_rec_grouped = ngs_receiving.groupby(['player_gsis_id', 'season']).agg(agg_dict).reset_index()
                    
                    # Merge with main dataframe for WRs/TEs
                    for idx in wr_te_indices:
                        player_id = df.loc[idx, 'player_id']
                        season = df.loc[idx, 'season'] if 'season' in df.columns else None
                        
                        if season:
                            # Find matching NGS data
                            ngs_match = ngs_rec_grouped[
                                (ngs_rec_grouped['player_gsis_id'] == player_id) & 
                                (ngs_rec_grouped['season'] == season)
                            ]
                            
                            if not ngs_match.empty:
                                # Add NGS features
                                for col, new_name in [
                                    ('separation_score', 'separation_score'),
                                    ('yac_ability', 'yac_ability'),
                                    ('catch_percentage', 'reliable_hands'),
                                    ('percent_share_of_intended_air_yards', 'air_yard_share')
                                ]:
                                    if col in ngs_match.columns:
                                        df.loc[idx, new_name] = ngs_match[col].values[0]
            
            # Store updated dataframe
            setattr(self, f"{df_name}_data", df)
        
        # Update the feature engineering instance's data
        self.fe.train_data = self.train_data
        self.fe.test_data = self.test_data
        
        logger.info("Advanced NGS features created successfully")
        
        return self
    
    def remove_multicollinear_features(self, threshold=0.95):
        """
        Remove highly multicollinear features
        
        Parameters:
        -----------
        threshold : float, optional
            Correlation threshold for considering features as multicollinear
            
        Returns:
        --------
        self : FeatureEngineeringExtensions
            Returns self for method chaining
        """
        logger.info(f"Removing multicollinear features with threshold {threshold}")
        
        # Run correlation analysis if not already done
        if not hasattr(self, 'feature_correlations') or self.feature_correlations is None:
            self.analyze_feature_correlations(threshold=threshold)
        
        # If no correlation analysis, return
        if not hasattr(self, 'highly_correlated') or self.highly_correlated is None:
            logger.warning("No correlation data available. Run analyze_feature_correlations() first.")
            return self
        
        # Get features to remove based on correlations
        features_to_remove = []
        
        if not self.highly_correlated.empty:
            # For each highly correlated pair, keep the one with higher importance
            for _, row in self.highly_correlated.iterrows():
                feat1 = row['Feature_1']
                feat2 = row['Feature_2']
                
                # If we have feature importances, use them to decide
                if hasattr(self, 'feature_importances') and self.feature_importances is not None:
                    # Get importance of each feature
                    imp1 = self.feature_importances[self.feature_importances['Feature'] == feat1]['Composite_Score'].values
                    imp2 = self.feature_importances[self.feature_importances['Feature'] == feat2]['Composite_Score'].values
                    
                    # If both features have importance scores
                    if len(imp1) > 0 and len(imp2) > 0:
                        if imp1[0] < imp2[0]:
                            features_to_remove.append(feat1)
                        else:
                            features_to_remove.append(feat2)
                    else:  # Fallback: remove the second feature
                        features_to_remove.append(feat2)
                else:  # No importance info: remove the second feature
                    features_to_remove.append(feat2)
        
        # Deduplicate list
        features_to_remove = list(set(features_to_remove))
        
        # Remove features
        if features_to_remove:
            logger.info(f"Removing {len(features_to_remove)} multicollinear features")
            
            # Keep track of original shape
            train_shape_before = self.train_data.shape
            test_shape_before = self.test_data.shape
            
            # Remove features
            self.train_data = self.train_data.drop(columns=features_to_remove, errors='ignore')
            self.test_data = self.test_data.drop(columns=features_to_remove, errors='ignore')
            
            # Log shape change
            train_shape_after = self.train_data.shape
            test_shape_after = self.test_data.shape
            
            logger.info(f"Train data shape: {train_shape_before} -> {train_shape_after}")
            logger.info(f"Test data shape: {test_shape_before} -> {test_shape_after}")
            
            # Update the feature engineering instance's data
            self.fe.train_data = self.train_data
            self.fe.test_data = self.test_data
        else:
            logger.info("No multicollinear features to remove")
        
        return self
    
    def select_best_features(self, target='fantasy_points_per_game', max_features_per_position=30, return_feature_lists=False):
        """
        Select best features based on position-specific importance analysis
        
        Parameters:
        -----------
        target : str, optional
            Target variable for importance analysis
        max_features_per_position : int, optional
            Maximum number of features to retain per position
        return_feature_lists : bool, optional
            Whether to return the position-specific feature lists
            
        Returns:
        --------
        self : FeatureEngineeringExtensions
            Returns self for method chaining
        """
        logger.info(f"Selecting best features for {target} with position-specific approach")
        
        # Non-predictive metadata columns to always exclude
        metadata_columns = ['player_id', 'name', 'team', 'gsis_id']
        
        # Position-specific feature lists
        position_features = {}
        
        # Check if position column exists in the data
        if 'position' not in self.train_data.columns:
            logger.warning("Position column not found in data. Using general feature selection.")
            # Run feature importance analysis if not already done
            if not hasattr(self, 'feature_importances') or self.feature_importances is None or self.feature_importances.empty:
                self.analyze_feature_importance(target=target)
            
            # Get top features excluding metadata - handle empty feature_importances
            if hasattr(self, 'feature_importances') and self.feature_importances is not None and not self.feature_importances.empty:
                features_to_keep = [f for f in self.feature_importances['Feature'].tolist() 
                                if f not in metadata_columns][:max_features_per_position]
            else:
                logger.warning("No feature importance data available, using fallback feature list")
                # Fallback to using all numerical columns
                features_to_keep = [col for col in self.train_data.select_dtypes(include=['number']).columns 
                                if col not in metadata_columns][:max_features_per_position]
                
            features_to_keep.append('position')  # Keep position for filtering/analysis
            
            # Add season for tracking
            if 'season' in self.train_data.columns:
                features_to_keep.append('season')
            
            # Always include the target
            if target in self.train_data.columns and target not in features_to_keep:
                features_to_keep.append(target)
            
            position_features['ALL'] = features_to_keep
        else:
            # Get unique positions
            positions = self.train_data['position'].unique()
            
            # Standard positions in fantasy football
            standard_positions = ['QB', 'RB', 'WR', 'TE']
            
            # Filter to standard positions that exist in the data
            positions = [pos for pos in positions if pos in standard_positions]
            
            # Add 'ALL' to positions to also get general features
            positions = positions + ['ALL']
            
            for position in positions:
                logger.info(f"Selecting features for position: {position}")
                
                if position == 'ALL':
                    position_data = self.train_data.copy()
                else:
                    position_data = self.train_data[self.train_data['position'] == position].copy()
                
                if len(position_data) == 0:
                    logger.warning(f"No data for position {position}, skipping")
                    continue
                
                # Define position-specific key metrics
                position_key_metrics = {
                    'QB': [
                        # Passing efficiency
                        'completion_percentage', 'yards_per_attempt', 'adjusted_yards_per_attempt',
                        'td_percentage', 'int_percentage', 'passing_td_to_int_ratio',
                        # Advanced metrics
                        'epa_per_pass', 'aggressive_efficacy', 'air_yards_differential',
                        # Volume metrics
                        'passing_yards_per_game', 'passing_tds_per_game',
                        # Rushing contribution
                        'rushing_yards_per_game', 'rushing_efficiency',
                        # Consistency/trend
                        'durability_score', 'fantasy_points_consistency'
                    ],
                    'RB': [
                        # Rushing metrics
                        'rushing_efficiency', 'rushing_yards_per_game', 'rushing_tds_per_game',
                        'rushing_first_down_rate', 'rush_over_expected_efficiency',
                        # Receiving contribution
                        'receiving_percentage', 'receptions_per_game', 'receiving_yards_per_game',
                        'yards_per_reception', 'receiving_rb_ratio',
                        # Advanced metrics
                        'epa_per_rush', 'total_yards_per_touch', 'decision_speed',
                        # PPR specific
                        'ppr_boost',
                        # Consistency/trend
                        'durability_score', 'fantasy_points_consistency'
                    ],
                    'WR': [
                        # Volume metrics
                        'targets_per_game', 'receptions_per_game', 'receiving_yards_per_game',
                        # Efficiency
                        'reception_ratio', 'yards_per_target', 'yards_per_reception',
                        # Advanced metrics
                        'air_yards_percentage', 'yac_percentage', 'air_yard_share',
                        'separation_score', 'yac_ability', 'epa_per_target',
                        'first_down_rate', 'first_down_per_target',
                        # PPR specific
                        'ppr_boost',
                        # Consistency/trend
                        'durability_score', 'fantasy_points_consistency'
                    ],
                    'TE': [
                        # Volume metrics
                        'targets_per_game', 'receptions_per_game', 'receiving_yards_per_game',
                        # Efficiency
                        'reception_ratio', 'yards_per_target', 'yards_per_reception',
                        # Advanced metrics
                        'air_yards_percentage', 'yac_percentage', 'epa_per_target',
                        'first_down_rate', 'separation_score',
                        # PPR specific
                        'ppr_boost',
                        # Consistency/trend
                        'durability_score', 'fantasy_points_consistency'
                    ],
                    'ALL': [
                        # General fantasy metrics
                        'fantasy_points_per_game', 'fantasy_points_consistency',
                        # Performance trends
                        'durability_score',
                        # Position-adjusted metrics
                        'position_adjusted_age', 'peak_season',
                        # Cluster information
                        'cluster_rank'
                    ]
                }
                
                # Get key metrics for this position
                position_metrics = position_key_metrics.get(position, [])
                
                # Get metrics that exist in the data
                available_metrics = [m for m in position_metrics if m in position_data.columns]
                
                # If we have very few metrics, add raw stats
                if len(available_metrics) < 5:
                    logger.warning(f"Few position-specific metrics available for {position}, adding raw stats")
                    if position == 'QB':
                        raw_stats = ['passing_yards', 'passing_tds', 'interceptions', 'rushing_yards', 'rushing_tds']
                    elif position == 'RB':
                        raw_stats = ['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards', 'receiving_tds']
                    elif position in ['WR', 'TE']:
                        raw_stats = ['receptions', 'targets', 'receiving_yards', 'receiving_tds']
                    else:
                        raw_stats = []
                    
                    # Add raw stats that exist in the data
                    available_metrics.extend([s for s in raw_stats if s in position_data.columns])
                
                # If we don't have feature importance data yet, analyze it for this position
                if position != 'ALL':  # Skip for ALL to avoid duplicating analysis
                    try:
                        # Create temporary object for position-specific analysis to avoid overwriting main
                        position_feature_eng = FeatureEngineeringExtensions(self.fe)
                        position_feature_eng.train_data = position_data
                        position_feature_eng.test_data = self.test_data[self.test_data['position'] == position] if position != 'ALL' else self.test_data
                        
                        # Analyze feature importance for this position
                        position_feature_eng.analyze_feature_importance(target=target)
                        
                        # Get importance data
                        position_importance = position_feature_eng.feature_importances
                        
                        # Add top important features from analysis
                        if position_importance is not None and not position_importance.empty:
                            # Get features not already in available_metrics
                            additional_features = [f for f in position_importance['Feature'].tolist() 
                                            if f not in available_metrics 
                                            and f not in metadata_columns][:max_features_per_position]
                            available_metrics.extend(additional_features)
                    except Exception as e:
                        logger.warning(f"Error analyzing feature importance for {position}: {e}")
                
                # Add trend features if available
                trend_features = []
                for base_metric in available_metrics:
                    for suffix in ['_prev_season', '_change', '_pct_change', '_3yr_avg', 
                                '_vs_3yr_avg', '_weighted_avg', '_vs_weighted_avg']:
                        trend_col = f"{base_metric}{suffix}"
                        if trend_col in position_data.columns:
                            trend_features.append(trend_col)
                
                # Add up to 10 trend features (avoiding too many)
                available_metrics.extend(trend_features[:10])
                
                # Add interactions between top metrics if available
                interaction_features = []
                for col in position_data.columns:
                    if '_interact' in col:
                        # Check if this is an interaction of metrics we care about
                        base_metrics = col.split('_interact')[0].split('_')
                        if any(metric in available_metrics for metric in base_metrics):
                            interaction_features.append(col)
                
                # Add up to 5 interaction features
                available_metrics.extend(interaction_features[:5])
                
                # Always include these columns
                essential_columns = ['position']
                if 'season' in position_data.columns:
                    essential_columns.append('season')
                if target in position_data.columns and target not in available_metrics:
                    essential_columns.append(target)
                
                # Add essential columns
                available_metrics.extend(essential_columns)
                
                # If we still don't have enough features, add more numeric columns for this position
                if len(available_metrics) < 5:
                    logger.warning(f"Not enough features for {position}, adding numeric columns")
                    numeric_cols = position_data.select_dtypes(include=['number']).columns
                    # Take numeric columns not in metadata and not already in available_metrics
                    more_numeric = [col for col in numeric_cols 
                                if col not in metadata_columns 
                                and col not in available_metrics][:max_features_per_position]
                    available_metrics.extend(more_numeric)
                
                # Remove duplicates and limit to max_features_per_position
                available_metrics = list(dict.fromkeys(available_metrics))[:max_features_per_position + len(essential_columns)]
                
                # Store metrics for this position
                position_features[position] = available_metrics
                
                logger.info(f"Selected {len(available_metrics)} features for {position}")
        
        # Now combine all position-specific features to create the final feature list
        all_features = set()
        for features in position_features.values():
            all_features.update(features)
        
        # Convert to list
        features_to_keep = list(all_features)
        
        # Make sure these are actually in both datasets
        features_to_keep = [f for f in features_to_keep 
                        if f in self.train_data.columns 
                        and f in self.test_data.columns]
        
        logger.info(f"Final feature set includes {len(features_to_keep)} features")
        
        # Create filtered datasets
        filtered_train = self.train_data[features_to_keep].copy()
        filtered_test = self.test_data[features_to_keep].copy()
        
        # Store selected features
        self.selected_features = features_to_keep
        
        # Update DataFrames
        self.train_data = filtered_train
        self.test_data = filtered_test
        
        # Update the feature engineering instance's data
        self.fe.train_data = self.train_data
        self.fe.test_data = self.test_data
        
        if return_feature_lists:
            return self, position_features
        
        return self
    
    def save_feature_engineered_data(self, output_dir="data/processed", prefix="feature_engineered"):
        """
        Save feature engineered data
        
        Parameters:
        -----------
        output_dir : str, optional
            Directory to save data
        prefix : str, optional
            Prefix for filenames
            
        Returns:
        --------
        self : FeatureEngineeringExtensions
            Returns self for method chaining
        """
        logger.info(f"Saving feature engineered data to {output_dir}")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save data
        self.train_data.to_csv(os.path.join(output_dir, f"{prefix}_train.csv"), index=False)
        self.test_data.to_csv(os.path.join(output_dir, f"{prefix}_test.csv"), index=False)
        
        # Save feature importance if available
        if hasattr(self, 'feature_importances') and self.feature_importances is not None:
            self.feature_importances.to_csv(os.path.join(output_dir, f"{prefix}_importance.csv"), index=False)
        
        # Save feature correlations if available
        if hasattr(self, 'feature_correlations') and self.feature_correlations is not None:
            self.feature_correlations.to_csv(os.path.join(output_dir, f"{prefix}_correlations.csv"))
        
        # Save selected features if available
        if hasattr(self, 'selected_features') and self.selected_features is not None:
            with open(os.path.join(output_dir, f"{prefix}_selected_features.txt"), 'w') as f:
                f.write('\n'.join(self.selected_features))
        
        # Save models and transformers
        if hasattr(self, 'polynomial_features') and self.polynomial_features is not None:
            joblib.dump(self.polynomial_features['transformer'], 
                      os.path.join(output_dir, f"{prefix}_poly_transformer.pkl"))
        
        logger.info("Feature engineered data saved successfully")
        
        return self
    
    def get_processed_data(self, selected_only=True):
        """
        Return the processed data
        
        Parameters:
        -----------
        selected_only : bool, optional
            If True, return only the selected features
            
        Returns:
        --------
        dict
            Dictionary containing processed 'train' and 'test' dataframes
        """
        # Update the original FeatureEngineering instance's data
        self.fe.train_data = self.train_data
        self.fe.test_data = self.test_data
        
        if hasattr(self, 'selected_features') and selected_only:
            return {
                'train': self.train_data[self.selected_features],
                'test': self.test_data[self.selected_features]
            }
        else:
            return {
                'train': self.train_data,
                'test': self.test_data
            }
            
    def create_advanced_ngs_features(self, ngs_data=None):
        """
        Create advanced features from Next Gen Stats data
        
        Parameters:
        -----------
        ngs_data : dict, optional
            Dictionary containing NGS data by type
            
        Returns:
        --------
        self : FeatureEngineeringExtensions
            Returns self for method chaining
        """
        logger.info("Creating advanced NGS features")
        
        if ngs_data is None or not any(not df.empty for df in ngs_data.values() if isinstance(df, pd.DataFrame)):
            logger.warning("No NGS data provided, skipping advanced NGS features")
            return self
        
        # Combine base features with NGS data
        for df_name in ['train', 'test']:
            df = getattr(self, df_name + '_data')
            
            # Check if we have player_id and position columns
            if 'player_id' not in df.columns or 'position' not in df.columns:
                logger.warning(f"player_id or position column missing in {df_name} data, skipping advanced NGS features")
                continue
            
            # Create different features based on position
            qb_indices = df[df['position'] == 'QB'].index
            rb_indices = df[df['position'] == 'RB'].index
            wr_te_indices = df[df['position'].isin(['WR', 'TE'])].index
            
            # QB advanced features
            if 'ngs_passing' in ngs_data and not ngs_data['ngs_passing'].empty:
                # Index NGS data by player GSIS ID and season
                ngs_passing = ngs_data['ngs_passing'].copy()
                
                if 'player_gsis_id' in ngs_passing.columns and 'season' in ngs_passing.columns:
                    # Extract useful metrics
                    ngs_pass_features = ['aggressiveness', 'avg_time_to_throw', 
                                    'avg_air_yards_to_sticks', 'avg_air_yards_differential',
                                    'completion_percentage_above_expectation']
                    
                    # Create new composite metrics
                    ngs_passing['aggressive_efficacy'] = ngs_passing['aggressiveness'] * ngs_passing['completion_percentage_above_expectation'].clip(lower=0)
                    ngs_passing['sticks_targeting'] = np.where(
                        ngs_passing['avg_air_yards_to_sticks'] > 0, 
                        1, 
                        np.where(
                            ngs_passing['avg_air_yards_to_sticks'] < -3,
                            -1,
                            0
                        )
                    )
                    
                    # Group by player and season
                    ngs_pass_grouped = ngs_passing.groupby(['player_gsis_id', 'season']).agg({
                        'aggressive_efficacy': 'mean',
                        'sticks_targeting': 'mean',
                        'avg_air_yards_differential': 'mean'
                    }).reset_index()
                    
                    # Merge with main dataframe for QBs
                    for idx in qb_indices:
                        player_id = df.loc[idx, 'player_id']
                        season = df.loc[idx, 'season'] if 'season' in df.columns else None
                        
                        if season:
                            # Find matching NGS data
                            ngs_match = ngs_pass_grouped[
                                (ngs_pass_grouped['player_gsis_id'] == player_id) & 
                                (ngs_pass_grouped['season'] == season)
                            ]
                            
                            if not ngs_match.empty:
                                # Add NGS features
                                df.loc[idx, 'aggressive_efficacy'] = ngs_match['aggressive_efficacy'].values[0]
                                df.loc[idx, 'sticks_targeting'] = ngs_match['sticks_targeting'].values[0]
                                df.loc[idx, 'air_yards_differential'] = ngs_match['avg_air_yards_differential'].values[0]
            
            # RB advanced features
            if 'ngs_rushing' in ngs_data and not ngs_data['ngs_rushing'].empty:
                # Index NGS data by player GSIS ID and season
                ngs_rushing = ngs_data['ngs_rushing'].copy()
                
                if 'player_gsis_id' in ngs_rushing.columns and 'season' in ngs_rushing.columns:
                    # Create new composite metrics
                    if 'rush_yards_over_expected_per_att' in ngs_rushing.columns and 'percent_attempts_gte_eight_defenders' in ngs_rushing.columns:
                        ngs_rushing['rush_over_expected_efficiency'] = ngs_rushing['rush_yards_over_expected_per_att'] * \
                                                                    (1 + ngs_rushing['percent_attempts_gte_eight_defenders'] / 100)
                    
                    # Group by player and season
                    ngs_rush_grouped = ngs_rushing.groupby(['player_gsis_id', 'season']).agg({
                        'rush_over_expected_efficiency': 'mean' if 'rush_over_expected_efficiency' in ngs_rushing.columns else 'count',
                        'efficiency': 'mean' if 'efficiency' in ngs_rushing.columns else 'count',
                        'avg_time_to_los': 'mean' if 'avg_time_to_los' in ngs_rushing.columns else 'count'
                    }).reset_index()
                    
                    # Add decision speed metric (lower time to LOS is better)
                    if 'avg_time_to_los' in ngs_rushing.columns:
                        ngs_rush_grouped['decision_speed'] = (2.5 - ngs_rush_grouped['avg_time_to_los'].clip(upper=2.5)) / 2.5 * 100
                    
                    # Merge with main dataframe for RBs
                    for idx in rb_indices:
                        player_id = df.loc[idx, 'player_id']
                        season = df.loc[idx, 'season'] if 'season' in df.columns else None
                        
                        if season:
                            # Find matching NGS data
                            ngs_match = ngs_rush_grouped[
                                (ngs_rush_grouped['player_gsis_id'] == player_id) & 
                                (ngs_rush_grouped['season'] == season)
                            ]
                            
                            if not ngs_match.empty:
                                # Add NGS features
                                for col in ['rush_over_expected_efficiency', 'efficiency', 'decision_speed']:
                                    if col in ngs_match.columns:
                                        df.loc[idx, col] = ngs_match[col].values[0]
            
            # WR/TE advanced features
            if 'ngs_receiving' in ngs_data and not ngs_data['ngs_receiving'].empty:
                # Index NGS data by player GSIS ID and season
                ngs_receiving = ngs_data['ngs_receiving'].copy()
                
                if 'player_gsis_id' in ngs_receiving.columns and 'season' in ngs_receiving.columns:
                    # Create new composite metrics
                    if 'avg_separation' in ngs_receiving.columns and 'avg_cushion' in ngs_receiving.columns:
                        ngs_receiving['separation_score'] = ngs_receiving['avg_separation'] * \
                                                        (1 + ngs_receiving['avg_cushion'] / 10)
                    
                    if 'avg_yac_above_expectation' in ngs_receiving.columns and 'avg_expected_yac' in ngs_receiving.columns:
                        ngs_receiving['yac_ability'] = ngs_receiving['avg_yac_above_expectation'] * \
                                                    (ngs_receiving['avg_expected_yac'] / 5)
                    
                    # Group by player and season
                    agg_dict = {}
                    for col in ['separation_score', 'yac_ability', 'catch_percentage', 'percent_share_of_intended_air_yards']:
                        if col in ngs_receiving.columns:
                            agg_dict[col] = 'mean'
                        else:
                            agg_dict[col] = 'count'
                    
                    ngs_rec_grouped = ngs_receiving.groupby(['player_gsis_id', 'season']).agg(agg_dict).reset_index()
                    
                    # Merge with main dataframe for WRs/TEs
                    for idx in wr_te_indices:
                        player_id = df.loc[idx, 'player_id']
                        season = df.loc[idx, 'season'] if 'season' in df.columns else None
                        
                        if season:
                            # Find matching NGS data
                            ngs_match = ngs_rec_grouped[
                                (ngs_rec_grouped['player_gsis_id'] == player_id) & 
                                (ngs_rec_grouped['season'] == season)
                            ]
                            
                            if not ngs_match.empty:
                                # Add NGS features
                                for col, new_name in [
                                    ('separation_score', 'separation_score'),
                                    ('yac_ability', 'yac_ability'),
                                    ('catch_percentage', 'reliable_hands'),
                                    ('percent_share_of_intended_air_yards', 'air_yard_share')
                                ]:
                                    if col in ngs_match.columns:
                                        df.loc[idx, new_name] = ngs_match[col].values[0]
            
            # Store updated dataframe
            setattr(self, f"{df_name}_data", df)
        
        # Update the feature engineering instance's data
        self.fe.train_data = self.train_data
        self.fe.test_data = self.test_data
        
        logger.info("Advanced NGS features created successfully")
        
        return self
    
    def _format_column_name(self, col_name):
        """Format column name for display on plots"""
        # Shorten column name for better display
        if len(col_name) > 25:
            parts = col_name.split('_')
            if len(parts) > 2:
                # Keep first and last parts if name is very long
                col_name = f"{parts[0]}_{parts[-1]}"
        
        return " ".join(word.capitalize() for word in col_name.replace('_', ' ').split())