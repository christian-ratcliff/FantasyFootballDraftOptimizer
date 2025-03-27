import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime


# Setup logging
logger = logging.getLogger(__name__)

class PlayerProjectionModel:
    """
    Model that uses time series approach for player projections
    """
    
    def __init__(self, feature_sets, output_dir='data/models', use_filtered=False):
        """
        Initialize the projection model with our feature sets
        
        Parameters:
        -----------
        feature_sets : dict
            Dictionary of feature sets created by FantasyFeatureEngineering
        output_dir : str
            Directory to save model files
        use_filtered : bool
            Whether to use filtered feature sets
        """
        if not use_filtered:
            self.feature_sets = {k: v for k, v in feature_sets.items() if '_filtered' not in k}
        else:
            self.feature_sets = feature_sets
        self.output_dir = output_dir
        self.models = {}
        self.feature_importances = {}
        self.target = 'fantasy_points_per_game'
        self.validation_metrics = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if we have the necessary datasets
        self._validate_feature_sets()
    
    def _validate_feature_sets(self):
        """Validate that we have the necessary feature sets"""
        required_keys = [f"{pos}_train" for pos in ['qb', 'rb', 'wr', 'te']]
        missing_keys = [key for key in required_keys if key not in self.feature_sets]
        
        if missing_keys:
            logger.warning(f"Missing required feature sets: {missing_keys}")
            logger.warning("Make sure you've processed data before using this model")
    
    def _get_available_years(self, position):
        """Get all available years for a position"""
        train_key = f"{position}_train"
        
        if train_key not in self.feature_sets or self.feature_sets[train_key].empty:
            logger.warning(f"No training data for {position}")
            return []
        
        # Get training data
        train_data = self.feature_sets[train_key]
        
        # Check if we have seasons information
        if 'season' not in train_data.columns:
            logger.warning(f"No season information in {position} data")
            return []
        
        # Get all seasons
        all_seasons = sorted(train_data['season'].unique())
        return all_seasons
    
    def _filter_meaningful_samples(self, train_data, position):
        """Filter data to only include meaningful sample sizes"""
        if position == 'qb':
            # Filter QBs with meaningful sample size
            if 'attempts' in train_data.columns:
                min_attempts = 100
                train_data = train_data[train_data['attempts'] >= min_attempts].copy()
                logger.info(f"Filtered QBs to those with at least {min_attempts} attempts ({len(train_data)} players)")
            
            # Filter for starter status
            if 'attempts' in train_data.columns and 'games' in train_data.columns:
                train_data['attempts_per_game'] = train_data['attempts'] / train_data['games'].clip(lower=1)
                # Consider QBs who averaged at least 15 attempts per game as starters
                train_data = train_data[train_data['attempts_per_game'] >= 15].copy()
                logger.info(f"Filtered to QBs with starter-level attempts ({len(train_data)} players)")
        
        elif position == 'rb':
            # Filter RBs with meaningful touches
            if 'carries' in train_data.columns:
                min_carries = 50
                train_data = train_data[train_data['carries'] >= min_carries].copy()
                logger.info(f"Filtered RBs to those with at least {min_carries} carries ({len(train_data)} players)")
        
        elif position in ['wr', 'te']:
            # Filter WRs/TEs with meaningful targets
            if 'targets' in train_data.columns:
                min_targets = 30
                train_data = train_data[train_data['targets'] >= min_targets].copy()
                logger.info(f"Filtered {position.upper()}s to those with at least {min_targets} targets ({len(train_data)} players)")
        
        return train_data
    
    def _select_features(self, train_data):
        """Select relevant features for modeling"""
        exclude_cols = [
            'player_id', 'name', 'team', 'season', 'cluster', 'tier', 'pca1', 'pca2', 
            'gsis_id', 'position', 'projected_points', self.target, 'ppr_sh', 'fantasy_points', 'ceiling_factor',
            'fantasy_points_ppr',
        ]
        
        numeric_cols = train_data.select_dtypes(include=['number']).columns.tolist()
        features = [col for col in numeric_cols if col not in exclude_cols]
        
        return features
    
    def train_with_time_series_validation(self, position, model_type='random_forest', hyperparams=None):
        """
        Train model with time series validation approach
        
        Parameters:
        -----------
        position : str
            Position to train model for ('qb', 'rb', 'wr', 'te')
        model_type : str
            Type of model ('random_forest' or 'gradient_boosting')
        hyperparams : dict, optional
            Model hyperparameters (if not provided, defaults will be used)
            
        Returns:
        --------
        dict
            Validation metrics
        """
        # Get all available years
        all_years = self._get_available_years(position)
        
        if len(all_years) < 2:
            logger.warning(f"Need at least 2 years of data for time series validation. Found: {len(all_years)}")
            return None
        
        # Get training data
        train_key = f"{position}_train"
        train_data_full = self.feature_sets[train_key].copy()
        
        # Apply meaningful sample filtering
        train_data_full = self._filter_meaningful_samples(train_data_full, position)
        
        # Select features 
        features = self._select_features(train_data_full)
        
        if len(features) < 3:
            logger.warning(f"Not enough features for {position}, need at least 3")
            return None
        
        # Create model based on type and provided hyperparameters
        if model_type == 'random_forest':
            # Default RF hyperparams
            rf_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'n_jobs': -1,
                'random_state': 42
            }
            
            # Update with user-provided hyperparams
            if hyperparams:
                rf_params.update(hyperparams)
                
            model = RandomForestRegressor(**rf_params)
            
        elif model_type == 'gradient_boosting':
            # Default GB hyperparams
            gb_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': 42
            }
            
            # Update with user-provided hyperparams
            if hyperparams:
                gb_params.update(hyperparams)
                
            model = GradientBoostingRegressor(**gb_params)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        # Time series validation
        logger.info(f"Running temporal validation for {position}...")
        validation_results = []
        
        # For each test year (starting from the second year)
        for i in range(1, len(all_years)):
            test_year = all_years[i]
            train_years = all_years[:i]
            
            # Create train/test split based on years
            train_mask = train_data_full['season'].isin(train_years)
            test_mask = train_data_full['season'] == test_year
            
            # Get X and y for train and test
            X_train = train_data_full.loc[train_mask, features].fillna(0)
            y_train = train_data_full.loc[train_mask, self.target].fillna(0)
            
            X_test = train_data_full.loc[test_mask, features].fillna(0)
            y_test = train_data_full.loc[test_mask, self.target].fillna(0)
            
            # Skip if we have too little data
            if len(X_train) < 10 or len(X_test) < 5:
                logger.warning(f"Skipping year {test_year} due to insufficient data")
                continue
            
            # Create recency weights to emphasize recent seasons within training data
            sample_weights = None
            if len(train_years) > 1:
                season_weights = {}
                # Apply exponential weighting to seasons - more recent years get higher weights
                for idx, season in enumerate(train_years):
                    weight = 2.0 ** (idx / (len(train_years) - 1))
                    season_weights[season] = weight
                
                # Apply weights to each sample based on season
                sample_weights = train_data_full.loc[train_mask, 'season'].map(season_weights).values
            
            # Fit model on training data
            model.fit(X_train, y_train, sample_weight=sample_weights)
            
            # Evaluate on train set
            train_pred = model.predict(X_train)
            train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
            
            # Evaluate on test set (next year)
            test_pred = model.predict(X_test)
            test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
            
            # Calculate gap
            gap = test_rmse - train_rmse
            
            # Log result
            logger.info(f"Years {train_years} → {test_year}: Train RMSE = {train_rmse:.2f}, Test RMSE = {test_rmse:.2f}, Gap = {gap:.2f}")
            
            # Store result
            validation_results.append({
                'train_years': train_years,
                'test_year': test_year,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'gap': gap,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            })
        
        # Train final model on all data
        X_all = train_data_full[features].fillna(0)
        y_all = train_data_full[self.target].fillna(0)
        
        # Create recency weights for all data
        all_weights = None
        if len(all_years) > 1:
            season_weights = {}
            for idx, season in enumerate(all_years):
                weight = 2.0 ** (idx / (len(all_years) - 1))
                season_weights[season] = weight
            
            all_weights = train_data_full['season'].map(season_weights).values
        
        # Fit final model on all data
        logger.info(f"Training final {position} model on all {len(all_years)} years of data ({len(X_all)} samples)")
        model.fit(X_all, y_all, sample_weight=all_weights)
        
        # Calculate metrics on all data (recognizing this is optimistic)
        all_pred = model.predict(X_all)
        
        metrics = {
            'mse': mean_squared_error(y_all, all_pred),
            'rmse': np.sqrt(mean_squared_error(y_all, all_pred)),
            'mae': mean_absolute_error(y_all, all_pred),
            'r2': r2_score(y_all, all_pred),
            'temporal_validation': validation_results
        }
        
        # Store model and validation metrics
        self.models[position] = {
            'model': model,
            'features': features,
            'model_type': model_type,
            'training_samples': len(X_all),
            'validation_metrics': metrics
        }
        
        # Store feature importances
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': features,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            self.feature_importances[position] = importance_df
            
            # Log top features
            logger.info(f"Top 5 important features for {position}:")
            for i, row in importance_df.head(5).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save the model
        self.save_model(position)
        
        # Log overall metrics
        logger.info(f"Final {position} model metrics:")
        logger.info(f"  MSE: {metrics['mse']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def train_all_positions(self, model_type='random_forest', hyperparams=None):
        """
        Train models for all positions using time series validation
        
        Parameters:
        -----------
        model_type : str
            Type of model ('random_forest' or 'gradient_boosting')
        hyperparams : dict, optional
            Model hyperparameters
            
        Returns:
        --------
        dict
            Dictionary of validation metrics by position
        """
        all_metrics = {}
        
        for position in ['qb', 'rb', 'wr', 'te']:
            metrics = self.train_with_time_series_validation(
                position, 
                model_type=model_type,
                hyperparams=hyperparams
            )
            
            if metrics:
                all_metrics[position] = metrics
        
        return all_metrics

    def project_players(self, position, data, use_do_not_draft=True):
        """
        Generate projections for players with improved confidence intervals and ceiling projections
        
        Parameters:
        -----------
        position : str
            Position to project ('qb', 'rb', 'wr', 'te')
        data : DataFrame
            Data containing players to project
        use_do_not_draft : bool
            Whether to zero out projections for players flagged as "do not draft"
            
        Returns:
        --------
        DataFrame
            DataFrame with multiple projection metrics added
        """
        # Check if model exists
        if position not in self.models:
            logger.warning(f"No model found for {position}. Train the model first.")
            return data
        
        # Get model info
        model_info = self.models[position]
        model = model_info['model']
        features = model_info['features']
        model_type = model_info['model_type']
        
        # Create a copy of data
        prediction_data = data.copy()
        
        # Check for missing features
        missing_features = [f for f in features if f not in prediction_data.columns]
        if missing_features:
            logger.warning(f"Missing features for {position}: {missing_features}")
            # Use only available features
            available_features = [f for f in features if f in prediction_data.columns]
            if len(available_features) < 3:
                logger.warning(f"Not enough features for {position} projection, need at least 3")
                # Handle insufficient features
                prediction_data['projected_points'] = np.nan
                prediction_data['projection_low'] = np.nan
                prediction_data['projection_high'] = np.nan
                prediction_data['ceiling_projection'] = np.nan
                return prediction_data
            
            X = prediction_data[available_features].fillna(0)
        else:
            # All features available
            X = prediction_data[features].fillna(0)
        
        # Make baseline projections
        y_pred = model.predict(X)
        
        # Add predictions to data
        prediction_data['projected_points'] = y_pred
        
        # Generate rich confidence intervals
        if model_type == 'random_forest':
            # For RF, we use the variance of predictions across trees for better uncertainty estimates
            trees = model.estimators_
            tree_preds = np.array([tree.predict(X) for tree in trees])
            y_std = np.std(tree_preds, axis=0)
            
            # Calculate confidence interval
            prediction_data['projection_low'] = y_pred - 1.28 * y_std
            prediction_data['projection_high'] = y_pred + 1.28 * y_std
            
            # Calculate skew in predictions to create asymmetric intervals
            skewness = np.zeros_like(y_pred)
            for i in range(len(y_pred)):
                if y_std[i] > 0:
                    skewness[i] = np.mean(((tree_preds[:, i] - y_pred[i]) / y_std[i]) ** 3)
            
            # Adjust confidence intervals based on skewness
            skew_adjustment = np.clip(skewness * 0.2, -0.5, 0.5)
            prediction_data['projection_low'] = y_pred - (1.28 - skew_adjustment) * y_std
            prediction_data['projection_high'] = y_pred + (1.28 + skew_adjustment) * y_std
            
            # Breakout potential adjustment
            if 'breakout_probability' in prediction_data.columns:
                breakout_boost = prediction_data['breakout_probability'] / 100 * 0.5
                prediction_data['projection_high'] += y_pred * breakout_boost
        else:
            # For other models, use position-specific uncertainty
            position_uncertainty = {
                'qb': 0.15,
                'rb': 0.20,
                'wr': 0.25,
                'te': 0.30
            }
            
            uncertainty = position_uncertainty.get(position, 0.20)
            prediction_data['projection_low'] = y_pred * (1 - uncertainty)
            prediction_data['projection_high'] = y_pred * (1 + uncertainty)
        
        # Ensure projections are not negative
        prediction_data['projection_low'] = prediction_data['projection_low'].clip(lower=0)
        
        # Ceiling projections
        if 'ceiling_factor' in prediction_data.columns:
            prediction_data['ceiling_projection'] = y_pred * prediction_data['ceiling_factor']
        else:
            # Default ceiling factors by position
            ceiling_factors = {
                'qb': 1.4,
                'rb': 1.5,
                'wr': 1.7,
                'te': 1.6
            }
            prediction_data['ceiling_projection'] = y_pred * ceiling_factors.get(position, 1.5)
        
        # Apply "do not draft" flags to zero out projections
        if use_do_not_draft and 'do_not_draft' in prediction_data.columns:
            do_not_draft_mask = prediction_data['do_not_draft'] == 1
            if do_not_draft_mask.any():
                count = do_not_draft_mask.sum()
                logger.info(f"Zeroing out projections for {count} {position.upper()}s flagged as 'do not draft'")
                prediction_data.loc[do_not_draft_mask, 'projected_points'] = 0
                prediction_data.loc[do_not_draft_mask, 'projection_low'] = 0
                prediction_data.loc[do_not_draft_mask, 'projection_high'] = 0
                prediction_data.loc[do_not_draft_mask, 'ceiling_projection'] = 0
        
        # Add projection tiers
        prediction_data = self._add_projection_tiers(prediction_data)
        
        return prediction_data
    
    def _add_projection_tiers(self, data):
        """Add projection tiers based on projected points"""
        if 'projected_points' not in data.columns:
            return data
        
        # Copy data to avoid modifying the original
        result = data.copy()
        
        try:
            # Create temporary column for tiering
            result['projected_points_for_tiers'] = result['projected_points'].clip(lower=0.1)
            unique_values = result['projected_points_for_tiers'].nunique()
            
            if unique_values >= 5:
                # Enough unique values for 5 bins
                try:
                    result['projection_tier'] = pd.qcut(
                        result['projected_points_for_tiers'], 
                        5, 
                        labels=['Tier 5', 'Tier 4', 'Tier 3', 'Tier 2', 'Tier 1'],
                        duplicates='drop'
                    )
                except ValueError as e:
                    logger.warning(f"Error using qcut: {e}. Falling back to manual tiering.")
                    unique_values = 1  # Force manual tiering
            
            if unique_values < 5:
                # Not enough unique values, use manual assignment
                logger.info(f"Only {unique_values} unique projection values, using manual tiering")
                
                # Create simple tier boundaries based on fixed point values
                result['projection_tier'] = 'Tier 3'  # Default tier
                
                max_val = result['projected_points'].max()
                min_val = max(result['projected_points'].min(), 0.1)
                
                # Calculate simple percentile-based cutoffs
                if max_val > min_val:
                    cutoffs = [
                        min_val,
                        min_val + (max_val - min_val) * 0.2,
                        min_val + (max_val - min_val) * 0.4,
                        min_val + (max_val - min_val) * 0.6,
                        min_val + (max_val - min_val) * 0.8,
                        max_val
                    ]
                    
                    # Make sure all cutoffs are unique
                    cutoffs = sorted(list(set(cutoffs)))
                    
                    if len(cutoffs) > 1:
                        # Ensure we have matching number of labels and bins
                        tier_labels = ['Tier 5', 'Tier 4', 'Tier 3', 'Tier 2', 'Tier 1'][:len(cutoffs)-1]
                        
                        # Manual assignment based on thresholds
                        for i in range(len(tier_labels)):
                            lower = cutoffs[i]
                            upper = cutoffs[i+1]
                            mask = (result['projected_points'] >= lower)
                            if i < len(tier_labels) - 1:
                                mask &= (result['projected_points'] < upper)
                            else:
                                mask &= (result['projected_points'] <= upper)
                            
                            result.loc[mask, 'projection_tier'] = tier_labels[i]
        except Exception as e:
            logger.warning(f"Could not create projection tiers: {e}")
            result['projection_tier'] = 'Tier 3'  # Default tier
        
        # Clean up temporary columns
        if 'projected_points_for_tiers' in result.columns:
            result = result.drop(columns=['projected_points_for_tiers'])
        
        return result
    
    def generate_full_projections(self, projection_data=None, use_do_not_draft=True):
        """
        Generate projections for all positions
        
        Parameters:
        -----------
        projection_data : dict, optional
            Dictionary of projection data by position
        use_do_not_draft : bool, optional
            Whether to apply the do_not_draft flag (default: True)
            
        Returns:
        --------
        dict
            Dictionary of projections by position
        """
        projections = {}
        
        for position in ['qb', 'rb', 'wr', 'te']:
            # Get projection data
            if projection_data and position in projection_data:
                data = projection_data[position]
            elif f"{position}_projection" in self.feature_sets:
                data = self.feature_sets[f"{position}_projection"]
            else:
                logger.warning(f"No projection data for {position}")
                continue
            
            # Generate projections
            proj_data = self.project_players(position, data, use_do_not_draft=use_do_not_draft)
            projections[position] = proj_data
            
            # Log projection stats
            if 'projected_points' in proj_data.columns:
                mean_proj = proj_data['projected_points'].mean()
                min_proj = proj_data['projected_points'].min()
                max_proj = proj_data['projected_points'].max()
                logger.info(f"{position} projections: mean={mean_proj:.2f}, min={min_proj:.2f}, max={max_proj:.2f}")
                
                # Save top players to CSV
                if 'name' in proj_data.columns:
                    top_players = proj_data.sort_values('projected_points', ascending=False).head(20)
                    top_players_path = os.path.join(self.output_dir, f"top_{position}_projections.csv")
                    cols = ['name', 'projected_points', 'projection_low', 'projection_high']
                    if 'age' in top_players.columns:
                        cols.append('age')
                    top_players[cols].to_csv(top_players_path, index=False)
                    logger.info(f"Saved top {position} projections to {top_players_path}")
        
        return projections
    
    def save_model(self, position):
        """
        Save the trained model to disk
        
        Parameters:
        -----------
        position : str
            Position model to save
                
        Returns:
        --------
        str
            Path to saved model file
        """
        if position not in self.models:
            logger.warning(f"No model found for {position}. Train the model first.")
            return None
        
        # Create filename
        filename = f"{position}_model.joblib"
        
        # Create full path
        model_path = os.path.join(self.output_dir, filename)
        
        # Get model info
        model_info = self.models[position]
        
        # Add timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create data to save
        save_data = {
            'model': model_info['model'],
            'features': model_info['features'],
            'model_type': model_info['model_type'],
            'position': position,
            'training_samples': model_info['training_samples'],
            'validation_metrics': model_info.get('validation_metrics', {}),
            'feature_importances': self.feature_importances.get(position),
            'last_updated': timestamp
        }
        
        # Save to disk
        joblib.dump(save_data, model_path)
        logger.info(f"Saved {position} model to {model_path}")
        
        return model_path
    
    @classmethod
    def load_model(cls, model_path, feature_sets=None):
        """
        Load a saved model
        
        Parameters:
        -----------
        model_path : str
            Path to saved model file
        feature_sets : dict, optional
            Dictionary of feature sets
            
        Returns:
        --------
        TimeSeriesPlayerProjectionModel
            Loaded model
        """
        # Load model data
        model_data = joblib.load(model_path)
        
        # Create new instance
        instance = cls(feature_sets=feature_sets or {})
        
        # Get position and add to models dict
        position = model_data['position']
        instance.models[position] = {
            'model': model_data['model'],
            'features': model_data['features'],
            'model_type': model_data['model_type'],
            'training_samples': model_data['training_samples'],
            'validation_metrics': model_data.get('validation_metrics', {})
        }
        
        # Add feature importances
        if 'feature_importances' in model_data and model_data['feature_importances'] is not None:
            instance.feature_importances[position] = model_data['feature_importances']
        
        logger.info(f"Loaded {position} model from {model_path}")
        
        return instance




class ProjectionModelLoader:
    """
    Centralized loader for projection models to avoid repeated disk reads
    """
    def __init__(self, models_dir='data/models'):
        """
        Initialize and load all projection models
        
        Parameters:
        -----------
        models_dir : str
            Directory containing joblib model files
        """
        self.models_dir = models_dir
        self.models = self._load_all_models()
    
    def _load_all_models(self):
        """
        Load projection models for all positions
        
        Returns:
        --------
        dict
            Dictionary of loaded projection models by position
        """
        projection_models = {}
        positions = ['qb', 'rb', 'wr', 'te']
        
        for position in positions:
            model_path = os.path.join(self.models_dir, f'{position}_model.joblib')
            try:
                model_data = joblib.load(model_path)
                projection_models[position] = model_data
                logger.info(f"Loaded projection model for {position}")
                
                # Log model details
                self._log_model_details(position, model_data)
                
            except Exception as e:
                logger.warning(f"Could not load {position} projection model: {e}")
        
        return projection_models
    
    def _log_model_details(self, position, model_data):
        """
        Log detailed information about the loaded model
        
        Parameters:
        -----------
        position : str
            Player position
        model_data : dict
            Loaded model data
        """
        logger.info(f"\n=== {position.upper()} Projection Model Details ===")
        
        # Model type
        logger.info(f"Model Type: {model_data.get('model_type', 'Unknown')}")
        
        # Training samples
        logger.info(f"Training Samples: {model_data.get('training_samples', 'N/A')}")
        
        # Features
        features = model_data.get('features', [])
        logger.info(f"Number of Features: {len(features)}")
        
        # Validation Metrics
        metrics = model_data.get('validation_metrics', {})
        if metrics:
            logger.info("Validation Metrics:")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
        
        # Feature Importances
        feature_importances = model_data.get('feature_importances')
        if feature_importances is not None:
            logger.info("\nTop 5 Most Important Features:")
            top_features = feature_importances.head(5)
            for _, row in top_features.iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        logger.info("=" * 50)
    
    def get_model(self, position):
        """
        Retrieve a specific position's model
        
        Parameters:
        -----------
        position : str
            Player position (lowercase)
        
        Returns:
        --------
        dict or None
            Loaded model data for the position
        """
        return self.models.get(position.lower())
