"""
Projection models that use our existing filtered feature sets
"""

import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

class PlayerProjectionModel:
    """
    Model that uses our pre-processed feature sets to create player projections
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
        
        # Check if we have the necessary filtered datasets
        self._validate_feature_sets()
    
    def _validate_feature_sets(self):
        """Validate that we have the necessary feature sets"""
        required_keys = [f"{pos}_train_filtered" for pos in ['qb', 'rb', 'wr', 'te']]
        missing_keys = [key for key in required_keys if key not in self.feature_sets]
        
        if missing_keys:
            logger.warning(f"Missing required feature sets: {missing_keys}")
            logger.warning("Make sure you've run clustering and filtering before using this model")
    
    def create_train_validation_split(self, position, validation_seasons=None):
        """
        Create proper train/validation split using historical seasons
        
        Parameters:
        -----------
        position : str
            Position to create split for ('qb', 'rb', 'wr', 'te')
        validation_seasons : int, list, optional
            Season(s) to use as validation (can be a single year or list of years)
                
        Returns:
        --------
        tuple
            (X_train, X_val, y_train, y_val, features) - Train/validation data and feature list
        """
        # Use unfiltered data instead of filtered
        train_key = f"{position}_train"
        
        # Check if we have data
        if train_key not in self.feature_sets or self.feature_sets[train_key].empty:
            logger.warning(f"No training data for {position}")
            return None, None, None, None, None
        
        # Get training data
        train_data = self.feature_sets[train_key].copy()
        
        # Apply meaningful sample filtering instead of tier filtering
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
        
        # Check if we have seasons information
        if 'season' not in train_data.columns:
            logger.warning(f"No season information in {position} data, cannot create time-based split")
            return None, None, None, None, None
        
        # Convert validation_seasons to list if it's a single integer
        if validation_seasons is not None:
            if isinstance(validation_seasons, int):
                validation_seasons = [validation_seasons]
        else:
            # If validation_seasons is None, use the most recent season as default
            seasons = sorted(train_data['season'].unique())
            if len(seasons) < 2:
                logger.warning(f"Need at least 2 seasons for validation split, only found {len(seasons)}")
                return None, None, None, None, None
            validation_seasons = [seasons[-1]]
        
        # Get training seasons (all seasons except validation ones)
        all_seasons = sorted(train_data['season'].unique())
        training_seasons = [s for s in all_seasons if s not in validation_seasons]
        
        logger.info(f"Using {validation_seasons} as validation season(s) and {training_seasons} for training")
        
        # Create train/validation split
        train_mask = train_data['season'].isin(training_seasons)
        val_mask = train_data['season'].isin(validation_seasons)
        
        # Select features and target
        exclude_cols = [
            'player_id', 'name', 'team', 'season', 'cluster', 'tier', 'pca1', 'pca2', 
            'gsis_id', 'position', 'projected_points', self.target, 'ppr_sh', 'fantasy_points', 'ceiling_factor',
            'fantasy_points_ppr',
            # 'attempts_per_game'  # Exclude derived columns we just created
        ]
        
        numeric_cols = train_data.select_dtypes(include=['number']).columns.tolist()
        features = [col for col in numeric_cols if col not in exclude_cols]
        
        # We need at least some features
        if len(features) < 3:
            logger.warning(f"Not enough features for {position}, need at least 3")
            return None, None, None, None, None
        
        # Create X and y for train and validation
        X_train = train_data.loc[train_mask, features].fillna(0)
        y_train = train_data.loc[train_mask, self.target].fillna(0)
        
        X_val = train_data.loc[val_mask, features].fillna(0)
        y_val = train_data.loc[val_mask, self.target].fillna(0)
        
        logger.info(f"Created train/validation split for {position}")
        logger.info(f"  Training samples: {len(X_train)}")
        logger.info(f"  Validation samples: {len(X_val)}")
        logger.info(f"  Features: {len(features)}")
        
        return X_train, X_val, y_train, y_val, features, train_data, train_mask

    def train_with_validation(self, position, model_type='random_forest', validation_season=None, 
                            hyperparams=None, perform_cv=True, evaluate_overfit=True):
        """
        Train model with proper validation
        
        Parameters:
        -----------
        position : str
            Position to train model for ('qb', 'rb', 'wr', 'te')
        model_type : str
            Type of model ('random_forest' or 'gradient_boosting')
        validation_season : int, list, optional
            Season(s) to use for validation - can be a single year or list of years
        hyperparams : dict, optional
            Model hyperparameters (if not provided, defaults will be used)
        perform_cv : bool
            Whether to perform cross-validation (slower but more robust)
        evaluate_overfit : bool
            Whether to run overfitting detection analysis
            
        Returns:
        --------
        dict
            Validation metrics
        """
        # Create train/validation split
        result = self.create_train_validation_split(position, validation_season)
        
        if result is None or len(result) < 5:
            return None
            
        X_train, X_val, y_train, y_val, features, train_data, train_mask = result
        
        if X_train is None or X_val is None:
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
        
        # Create recency weights to emphasize recent seasons
        sample_weights = None
        if 'season' in train_data.columns:
            # Get all training seasons
            all_seasons = sorted(train_data.loc[train_mask, 'season'].unique())
            season_weights = {}
            
            # Apply exponential weighting to seasons
            for i, season in enumerate(all_seasons):
                # Most recent training season gets highest weight
                weight = 4.0 ** (i / (len(all_seasons) - 1)) if len(all_seasons) > 1 else 1.0
                season_weights[season] = weight
            
            # Apply weights to each sample based on season
            sample_weights = train_data.loc[train_mask, 'season'].map(season_weights).values
        
        # Perform cross-validation if requested (only on training data)
        if perform_cv:
            logger.info(f"Performing time-series cross-validation for {position}")
            
            # Get seasons from training data
            available_seasons = sorted(train_data.loc[train_mask, 'season'].unique())
            
            # Use TimeSeriesSplit with number of splits based on available seasons
            n_splits = min(len(available_seasons) - 1, 3)
            if n_splits < 2:
                logger.warning(f"Not enough seasons for cross-validation, using single train/val split")
            else:
                # Run cross-validation
                cv_scores = []
                
                # For each possible split point
                for split_point in range(1, len(available_seasons)):
                    # Get training and validation seasons
                    cv_train_seasons = available_seasons[:split_point]
                    cv_val_season = available_seasons[split_point]
                    
                    logger.info(f"CV fold: train on {cv_train_seasons}, validate on {cv_val_season}")
                    
                    # Get indices for this split
                    cv_train_mask = train_data.loc[train_mask, 'season'].isin(cv_train_seasons)
                    cv_val_mask = train_data.loc[train_mask, 'season'] == cv_val_season
                    
                    # Get feature values
                    cv_train_X = X_train[cv_train_mask]
                    cv_train_y = y_train[cv_train_mask]
                    cv_val_X = X_train[cv_val_mask]
                    cv_val_y = y_train[cv_val_mask]
                    
                    # Skip if we don't have enough data
                    if len(cv_train_X) < 10 or len(cv_val_X) < 5:
                        logger.warning(f"Skipping fold due to insufficient data")
                        continue
                    
                    # Get sample weights for this CV fold
                    cv_weights = None
                    if sample_weights is not None:
                        cv_weights = sample_weights[cv_train_mask]
                    
                    # Fit model on training fold with weights
                    model.fit(cv_train_X, cv_train_y, sample_weight=cv_weights)
                    
                    # Evaluate on validation fold
                    cv_pred = model.predict(cv_val_X)
                    cv_rmse = np.sqrt(mean_squared_error(cv_val_y, cv_pred))
                    cv_scores.append(cv_rmse)
                
                # Log CV results
                if cv_scores:
                    logger.info(f"CV RMSE scores: {cv_scores}")
                    logger.info(f"Mean CV RMSE: {np.mean(cv_scores):.4f}")
                else:
                    logger.warning("No valid CV folds")
        
        # Fit model on all training data with sample weights
        logger.info(f"Fitting {model_type} model for {position} on {len(X_train)} samples with recency weighting")
        model.fit(X_train, y_train, sample_weight=sample_weights)
        
        # Evaluate on validation set
        val_pred = model.predict(X_val)
        val_metrics = {
            'mse': mean_squared_error(y_val, val_pred),
            'rmse': np.sqrt(mean_squared_error(y_val, val_pred)),
            'mae': mean_absolute_error(y_val, val_pred),
            'r2': r2_score(y_val, val_pred)
        }
        
        logger.info(f"Validation metrics for {position} model:")
        logger.info(f"  MSE: {val_metrics['mse']:.4f}")
        logger.info(f"  RMSE: {val_metrics['rmse']:.4f}")
        logger.info(f"  MAE: {val_metrics['mae']:.4f}")
        logger.info(f"  R²: {val_metrics['r2']:.4f}")
        
        # Store model and validation metrics
        self.models[position] = {
            'model': model,
            'features': features,
            'model_type': model_type,
            'training_samples': len(X_train),
            'validation_samples': len(X_val),
            'validation_metrics': val_metrics
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
        
        # Save model metadata
        self.validation_metrics[position] = val_metrics
        
        # Save the model
        self.save_model(position)
        
        # Run overfitting detection if requested
        if evaluate_overfit:
            logger.info(f"Running overfitting detection for {position} model...")
            overfitting_results = self.evaluate_overfitting(position)
            # Store overfitting results with the model
            if overfitting_results:
                self.models[position]['overfitting_analysis'] = overfitting_results
        
        return val_metrics
    
    
    def train_all_positions(self, model_type='random_forest', validation_season=None, 
                            hyperparams=None, perform_cv=True, evaluate_overfit=True):
            """
            Train models for all positions
            
            Parameters:
            -----------
            model_type : str
                Type of model ('random_forest' or 'gradient_boosting')
            validation_season : int, list, optional
                Season(s) to use for validation - can be a single year or list of years
            hyperparams : dict, optional
                Model hyperparameters
            perform_cv : bool
                Whether to perform cross-validation
            evaluate_overfit : bool
                Whether to run overfitting detection analysis
                
            Returns:
            --------
            dict
                Dictionary of validation metrics by position
            """
            all_metrics = {}
            
            for position in ['qb', 'rb', 'wr', 'te']:
                metrics = self.train_with_validation(
                    position, 
                    model_type=model_type,
                    validation_season=validation_season,
                    hyperparams=hyperparams,
                    perform_cv=perform_cv,
                    evaluate_overfit=evaluate_overfit
                )
                
                if metrics:
                    all_metrics[position] = metrics
            
            return all_metrics

    def train_position(self, position, model_type='random_forest'):
        """
        Train a model for a specific position (legacy method for compatibility)
        
        Parameters:
        -----------
        position : str
            Position to train model for ('qb', 'rb', 'wr', 'te')
        model_type : str
            Type of model to use ('random_forest' or 'gradient_boosting')
            
        Returns:
        --------
        self : ProjectionModel
            Returns self for method chaining
        """
        logger.warning("train_position is deprecated, use train_with_validation instead")
        self.train_with_validation(position, model_type=model_type, perform_cv=False)
        return self
    
    
    def evaluate_overfitting(self, position):
        """
        Run comprehensive overfitting detection for a specific position model
        
        Parameters:
        -----------
        position : str
            Position to evaluate ('qb', 'rb', 'wr', 'te')
            
        Returns:
        --------
        dict
            Dictionary with all evaluation results
        """
        from src.models.model_evaluation import (
            evaluate_model_fit, plot_learning_curves, analyze_feature_importance,
            temporal_validation, cross_validation_analysis
        )
        
        # Check if model exists
        if position not in self.models:
            logger.warning(f"No model found for {position}. Train the model first.")
            return None
        
        # Get model info
        model_info = self.models[position]
        model = model_info['model']
        features = model_info['features']
        
        # Create output directory for evaluation results
        eval_dir = os.path.join(self.output_dir, 'evaluation', position)
        os.makedirs(eval_dir, exist_ok=True)
        
        # We need to get training and validation data
        # Create proper train/validation split using previous method
        result = self.create_train_validation_split(position)
        
        if result is None or len(result) < 5:
            logger.warning(f"Could not create train/validation split for {position}")
            return None
            
        X_train, X_val, y_train, y_val, features, train_data, train_mask = result
        
        if X_train is None or X_val is None:
            logger.warning(f"Invalid train/validation data for {position}")
            return None
        
        # Initialize results dictionary
        results = {}
        
        # 1. Basic train/validation comparison
        logger.info(f"Evaluating train/validation fit for {position}...")
        model_fit = evaluate_model_fit(model, X_train, y_train, X_val, y_val)
        results['model_fit'] = model_fit
        
        # 2. Learning curves
        logger.info(f"Generating learning curves for {position}...")
        learning_curve_path = os.path.join(eval_dir, 'learning_curve.png')
        _, learning_curve_data = plot_learning_curves(
            model, X_train, y_train, 
            cv=5, 
            output_path=learning_curve_path
        )
        results['learning_curve'] = learning_curve_data
        
        # 3. Feature importance analysis
        logger.info(f"Analyzing feature importance for {position}...")
        importance_path = os.path.join(eval_dir, 'feature_importance.png')
        _, importance_df = analyze_feature_importance(
            model, features, 
            output_path=importance_path,
            top_n=10
        )
        if importance_df is not None:
            # Save feature importance to CSV
            importance_csv = os.path.join(eval_dir, 'feature_importance.csv')
            importance_df.to_csv(importance_csv, index=False)
            results['feature_importance'] = importance_df.to_dict(orient='records')
        
        # 4. Temporal validation (across seasons)
        if 'season' in train_data.columns:
            logger.info(f"Running temporal validation for {position}...")
            seasons = sorted(train_data['season'].unique())
            
            if len(seasons) >= 3:  # Need at least 3 seasons for meaningful validation
                temporal_path = os.path.join(eval_dir, 'temporal_validation.png')
                _, temporal_df = temporal_validation(
                    model, train_data, seasons, features,
                    target='fantasy_points_per_game',
                    output_path=temporal_path
                )
                
                if temporal_df is not None:
                    # Save temporal validation to CSV
                    temporal_csv = os.path.join(eval_dir, 'temporal_validation.csv')
                    temporal_df.to_csv(temporal_csv, index=False)
                    results['temporal_validation'] = temporal_df.to_dict(orient='records')
        
        # 5. Cross-validation analysis
        logger.info(f"Running cross-validation analysis for {position}...")
        cv_path = os.path.join(eval_dir, 'cross_validation.png')
        _, cv_results = cross_validation_analysis(
            model, X_train, y_train, 
            cv=5,
            output_path=cv_path
        )
        results['cross_validation'] = cv_results
        
        # Save the combined results
        results_path = os.path.join(eval_dir, 'overfitting_evaluation.json')
        import json
        
        # Convert numpy arrays and other non-serializable types
        def json_serialize(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.int64):
                return int(obj)
            if isinstance(obj, np.float64):
                return float(obj)
            raise TypeError(f"Type {type(obj)} not serializable")
        
        try:
            with open(results_path, 'w') as f:
                json.dump(results, f, default=json_serialize, indent=2)
            logger.info(f"Saved overfitting evaluation results to {results_path}")
        except Exception as e:
            logger.error(f"Error saving overfitting evaluation results: {e}")
        
        # Log overall assessment
        self._log_overfitting_assessment(results)
        
        return results

    def _log_overfitting_assessment(self, results):
        """Log an overall assessment of overfitting based on evaluation results"""
        if not results or 'model_fit' not in results:
            return
        
        # Extract key metrics
        model_fit = results['model_fit']
        r2_gap = model_fit['r2_gap']
        rmse_gap = model_fit['rmse_gap']
        
        # Create assessment
        overfitting_signals = []
        
        # Check train/validation gap
        if r2_gap > 0.2:
            overfitting_signals.append(f"Large R² gap between train and validation ({r2_gap:.2f})")
        
        # Check learning curve if available
        if 'learning_curve' in results:
            lc = results['learning_curve']
            if lc['final_gap'] < -1.0:  # Negative gap means validation error > training error
                overfitting_signals.append(f"Large final gap in learning curve ({lc['final_gap']:.2f})")
        
        # Check cross-validation results if available
        if 'cross_validation' in results:
            cv = results['cross_validation']
            r2_ratio = cv['test_r2_std'] / max(abs(cv['test_r2_mean']), 0.01)
            if r2_ratio > 0.2:
                overfitting_signals.append(f"High variance in CV R² scores (std/mean: {r2_ratio:.2f})")
        
        # Log assessment
        if overfitting_signals:
            logger.warning("OVERFITTING ASSESSMENT: Potential overfitting detected")
            for signal in overfitting_signals:
                logger.warning(f"  - {signal}")
        else:
            logger.info("OVERFITTING ASSESSMENT: No strong signals of overfitting detected")
    
    def project_players(self, position, data, use_do_not_draft=True):
        """
        Generate projections for players with improved confidence intervals and ceiling projections
        
        Parameters:
        -----------
        position : str
            Position to project ('qb', 'rb', 'wr', 'te')
        data : DataFrame
            Data containing players to project
            
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
        
        # ENHANCED: Generate rich confidence intervals
        if model_type == 'random_forest':
            # For RF, we use the variance of predictions across trees for better uncertainty estimates
            trees = model.estimators_
            tree_preds = np.array([tree.predict(X) for tree in trees])
            y_std = np.std(tree_preds, axis=0)
            
            # Calculate both symmetric and asymmetric intervals
            
            # 1. Basic confidence interval (symmetric)
            prediction_data['projection_low'] = y_pred - 1.28 * y_std
            prediction_data['projection_high'] = y_pred + 1.28 * y_std
            
            # 2. Calculate skew in predictions to create asymmetric intervals
            # Higher skew = more upside potential
            skewness = np.zeros_like(y_pred)
            for i in range(len(y_pred)):
                if y_std[i] > 0:  # Avoid division by zero
                    skewness[i] = np.mean(((tree_preds[:, i] - y_pred[i]) / y_std[i]) ** 3)
            
            # Adjust confidence intervals based on skewness
            # Positive skew = more upside (higher ceiling)
            skew_adjustment = np.clip(skewness * 0.2, -0.5, 0.5)
            prediction_data['projection_low'] = y_pred - (1.28 - skew_adjustment) * y_std
            prediction_data['projection_high'] = y_pred + (1.28 + skew_adjustment) * y_std
            
            # 3. Breakout potential adjustment
            if 'breakout_probability' in prediction_data.columns:
                # Scale the ceiling higher for breakout candidates
                breakout_boost = prediction_data['breakout_probability'] / 100 * 0.5
                prediction_data['projection_high'] += y_pred * breakout_boost
        else:
            # For other models, use position-specific uncertainty
            position_uncertainty = {
                'qb': 0.15,  # 15% uncertainty for QBs
                'rb': 0.20,  # 20% uncertainty for RBs
                'wr': 0.25,  # 25% uncertainty for WRs
                'te': 0.30   # 30% uncertainty for TEs
            }
            
            uncertainty = position_uncertainty.get(position, 0.20)
            prediction_data['projection_low'] = y_pred * (1 - uncertainty)
            prediction_data['projection_high'] = y_pred * (1 + uncertainty)
        
        # Ensure projections are not negative
        prediction_data['projection_low'] = prediction_data['projection_low'].clip(lower=0)
        
        # ENHANCED: Ceiling projections
        # If we have ceiling_factor from feature engineering, use it
        if 'ceiling_factor' in prediction_data.columns:
            prediction_data['ceiling_projection'] = y_pred * prediction_data['ceiling_factor']
        else:
            # Default ceiling factors by position
            ceiling_factors = {
                'qb': 1.4,  # QBs can have huge ceiling games
                'rb': 1.5,  # RBs have high TD variance 
                'wr': 1.7,  # WRs have highest week-to-week variance
                'te': 1.6   # TEs have high variance too
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
        
        # Apply specialized adjustments
        prediction_data = self._apply_specialized_adjustments(position, prediction_data)
        
        # Add projection tiers based on projected points
        try:
            # First determine how many unique values we have
            prediction_data['projected_points_for_tiers'] = prediction_data['projected_points'].clip(lower=0.1)
            unique_values = prediction_data['projected_points_for_tiers'].nunique()
            
            if unique_values >= 5:
                # We have enough unique values for 5 bins
                try:
                    prediction_data['projection_tier'] = pd.qcut(
                        prediction_data['projected_points_for_tiers'], 
                        5, 
                        labels=['Tier 5', 'Tier 4', 'Tier 3', 'Tier 2', 'Tier 1'],
                        duplicates='drop'
                    )
                except ValueError as e:
                    logger.warning(f"Error using qcut: {e}. Falling back to manual tiering.")
                    # If qcut fails, fall back to manual tiering
                    unique_values = 1  # Force manual tiering
            
            if unique_values < 5:
                # Not enough unique values, use manual assignment
                logger.info(f"Only {unique_values} unique projection values, using manual tiering")
                
                # Create simple tier boundaries based on fixed point values
                prediction_data['projection_tier'] = 'Tier 3'  # Default tier
                
                max_val = prediction_data['projected_points'].max()
                min_val = max(prediction_data['projected_points'].min(), 0.1)
                
                # Only continue if we have a valid range
                if max_val > min_val:
                    # Calculate simple percentile-based cutoffs
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
                    
                    # Only proceed with tiering if we have enough unique cutoffs
                    if len(cutoffs) > 1:
                        # Ensure we have matching number of labels and bins
                        tier_labels = ['Tier 5', 'Tier 4', 'Tier 3', 'Tier 2', 'Tier 1'][:len(cutoffs)-1]
                        
                        # Manual assignment based on thresholds
                        for i in range(len(tier_labels)):
                            lower = cutoffs[i]
                            upper = cutoffs[i+1]
                            mask = (prediction_data['projected_points'] >= lower)
                            if i < len(tier_labels) - 1:
                                mask &= (prediction_data['projected_points'] < upper)
                            else:
                                mask &= (prediction_data['projected_points'] <= upper)
                            
                            prediction_data.loc[mask, 'projection_tier'] = tier_labels[i]
        except Exception as e:
            logger.warning(f"Could not create projection tiers: {e}")
            # Fallback to simple manual assignment
            prediction_data['projection_tier'] = 'Tier 3'  # Default tier

        # Clean up temporary columns
        if 'projected_points_for_tiers' in prediction_data.columns:
            prediction_data = prediction_data.drop(columns=['projected_points_for_tiers'])

        # Add breakout tier for players with high ceiling relative to baseline
        try:
            # Calculate ceiling to base ratio
            prediction_data['ceiling_to_base_ratio'] = prediction_data['ceiling_projection'] / prediction_data['projected_points'].clip(lower=0.1)
            
            # Clip values to reasonable range and create a temporary column for tiering
            prediction_data['ceiling_ratio_for_tiers'] = prediction_data['ceiling_to_base_ratio'].clip(lower=1.0, upper=2.0)
            
            # First determine how many unique values we have
            unique_values = prediction_data['ceiling_ratio_for_tiers'].nunique()
            
            if unique_values >= 5:
                # We have enough unique values for 5 bins
                try:
                    prediction_data['breakout_tier'] = pd.qcut(
                        prediction_data['ceiling_ratio_for_tiers'],
                        5, 
                        labels=['Low Ceiling', 'Below Avg Ceiling', 'Average Ceiling', 'High Ceiling', 'Breakout Potential'],
                        duplicates='drop'
                    )
                except ValueError as e:
                    logger.warning(f"Error using qcut for breakout tiers: {e}. Falling back to manual tiering.")
                    # If qcut fails, fall back to manual tiering
                    unique_values = 1  # Force manual tiering
            
            if unique_values < 5:
                # Not enough unique values, use manual assignment
                logger.info(f"Only {unique_values} unique ceiling ratio values, using manual tiering")
                
                # Create simple tier assignments based on fixed ratio values
                prediction_data['breakout_tier'] = 'Average Ceiling'  # Default tier
                
                # Define breakout tier thresholds
                ratio_thresholds = {
                    'Low Ceiling': 1.0,
                    'Below Avg Ceiling': 1.2,
                    'Average Ceiling': 1.4,
                    'High Ceiling': 1.6,
                    'Breakout Potential': 1.8
                }
                
                # Apply thresholds
                for tier, threshold in ratio_thresholds.items():
                    if tier == 'Low Ceiling':
                        prediction_data.loc[prediction_data['ceiling_ratio_for_tiers'] < threshold, 'breakout_tier'] = tier
                    elif tier == 'Breakout Potential':
                        prediction_data.loc[prediction_data['ceiling_ratio_for_tiers'] >= threshold, 'breakout_tier'] = tier
                    else:
                        next_tier = list(ratio_thresholds.keys())[list(ratio_thresholds.keys()).index(tier) + 1]
                        next_threshold = ratio_thresholds[next_tier]
                        prediction_data.loc[(prediction_data['ceiling_ratio_for_tiers'] >= threshold) & 
                                        (prediction_data['ceiling_ratio_for_tiers'] < next_threshold), 'breakout_tier'] = tier
                    
        except Exception as e:
            logger.warning(f"Could not create breakout tiers: {e}")
            # Fallback to simple assignment
            prediction_data['breakout_tier'] = 'Average Ceiling'  # Default tier

        # Clean up temporary columns
        if 'ceiling_ratio_for_tiers' in prediction_data.columns:
            prediction_data = prediction_data.drop(columns=['ceiling_ratio_for_tiers'])
        
        return prediction_data

    def _apply_specialized_adjustments(self, position, prediction_data):
        """
        Apply specialized adjustments that aren't redundant with feature engineering
        
        Parameters:
        -----------
        position : str
            Position to adjust ('qb', 'rb', 'wr', 'te')
        prediction_data : DataFrame
            Data with raw projections
            
        Returns:
        --------
        DataFrame
            Data with adjusted projections
        """
        # Only include adjustments that are NOT redundant with feature engineering
        
        if position == 'qb':
            # TD regression adjustment - not captured by features
            if all(col in prediction_data.columns for col in ['passing_tds_per_game', 'passing_yards_per_game']):
                # Calculate TD rate (TDs per 300 yards)
                prediction_data['td_rate'] = prediction_data['passing_tds_per_game'] / (prediction_data['passing_yards_per_game'] / 300 + 0.001)
                
                # Identify outliers (very high or low TD rates)
                high_td_rate = prediction_data['td_rate'] > 2.0  # Above average TD rate
                low_td_rate = prediction_data['td_rate'] < 1.0   # Below average TD rate
                
                # Apply regression adjustments
                prediction_data.loc[high_td_rate, 'projected_points'] *= 0.95  # Regress high TD rates downward
                prediction_data.loc[low_td_rate, 'projected_points'] *= 1.05   # Regress low TD rates upward
                
                # Clean up temporary column
                prediction_data = prediction_data.drop(columns=['td_rate'])
        
        elif position == 'rb':
            # Volume-based adjustment for projection confidence - not a feature adjustment
            if 'touches_per_game' in prediction_data.columns:
                # High-volume RBs are more reliable
                high_volume = prediction_data['touches_per_game'] > 18
                # Low-volume RBs are less reliable
                low_volume = prediction_data['touches_per_game'] < 10
                
                # Apply volume adjustments to confidence intervals only
                if 'projection_low' in prediction_data.columns and 'projection_high' in prediction_data.columns:
                    # Narrower intervals for high-volume backs (more certain floor)
                    prediction_data.loc[high_volume, 'projection_low'] *= 1.05
                    
                    # Wider intervals for low-volume backs (more uncertain)
                    prediction_data.loc[low_volume, 'projection_low'] *= 0.90
                    prediction_data.loc[low_volume, 'projection_high'] *= 1.05
        
        elif position == 'wr':
            # Target share consistency adjustment - affects confidence not central projection
            if 'target_share' in prediction_data.columns and 'projection_low' in prediction_data.columns:
                # High target share WRs are more reliable floor
                high_share = prediction_data['target_share'] > 0.25
                
                # Apply minor adjustments to confidence intervals
                prediction_data.loc[high_share, 'projection_low'] *= 1.03  # Slight boost to floor
        
        elif position == 'te':
            # No specialized TE adjustments needed that aren't already in features
            pass
        
        return prediction_data
    
    def evaluate_position(self, position, data=None):
        """
        Evaluate model performance for a position
        
        Parameters:
        -----------
        position : str
            Position to evaluate ('qb', 'rb', 'wr', 'te')
        data : DataFrame, optional
            Data to evaluate on (default uses training data)
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        # Check if model exists
        if position not in self.models:
            logger.warning(f"No model found for {position}. Train the model first.")
            return None
        
        # Get model info
        model_info = self.models[position]
        model = model_info['model']
        features = model_info['features']
        
        # Use provided data or get training data
        if data is not None:
            eval_data = data
        else:
            train_key = f"{position}_train_filtered"
            if train_key not in self.feature_sets or self.feature_sets[train_key].empty:
                logger.warning(f"No evaluation data available for {position}")
                return None
            eval_data = self.feature_sets[train_key].copy()
        
        # Check if target exists
        if self.target not in eval_data.columns:
            logger.warning(f"Target column {self.target} not found in evaluation data")
            return None
        
        # Check if we have the required features
        missing_features = [f for f in features if f not in eval_data.columns]
        if missing_features:
            logger.warning(f"Missing features in evaluation data: {missing_features}")
            # Use only available features
            features = [f for f in features if f in eval_data.columns]
            if len(features) < 3:
                logger.warning("Not enough features for evaluation")
                return None
        
        # Create X and y
        X = eval_data[features].fillna(0)
        y = eval_data[self.target].fillna(0)
        
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        logger.info(f"Evaluation metrics for {position} model:")
        logger.info(f"  MSE: {metrics['mse']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  R²: {metrics['r2']:.4f}")
        
        return metrics
    
    def save_model(self, position):
            """
            Save the trained model to disk (overwriting existing model)
            
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
            
            # Create fixed filename without timestamp for overwriting
            filename = f"{position}_model.joblib"
            
            # Create full path
            model_path = os.path.join(self.output_dir, filename)
            
            # Get model info
            model_info = self.models[position]
            
            # Add timestamp for tracking when this model was last updated
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create data to save
            save_data = {
                'model': model_info['model'],
                'features': model_info['features'],
                'model_type': model_info['model_type'],
                'position': position,
                'training_samples': model_info['training_samples'],
                'validation_samples': model_info.get('validation_samples', 0),
                'validation_metrics': model_info.get('validation_metrics', {}),
                'feature_importances': self.feature_importances.get(position),
                'last_updated': timestamp
            }
            
            # Save to disk (overwriting existing file)
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
        ProjectionModel
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
            'validation_samples': model_data.get('validation_samples', 0),
            'validation_metrics': model_data.get('validation_metrics', {})
        }
        
        # Add feature importances
        if 'feature_importances' in model_data and model_data['feature_importances'] is not None:
            instance.feature_importances[position] = model_data['feature_importances']
        
        logger.info(f"Loaded {position} model from {model_path}")
        
        return instance
    
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
            elif f"{position}_projection_filtered" in self.feature_sets:
                data = self.feature_sets[f"{position}_projection_filtered"]
            else:
                logger.warning(f"No projection data for {position}")
                continue
            
            # Generate projections with the flag parameter
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