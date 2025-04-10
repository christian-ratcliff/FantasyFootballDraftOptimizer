import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
import joblib
import os
from datetime import datetime

# Setup logging
logger = logging.getLogger(__name__)

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
try:
    import optuna
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class PlayerProjectionModel:
    """
    Model that uses enhanced approach for player projections with hierarchical modeling,
    hyperparameter tuning, and formal feature selection techniques
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
        self.hierarchical_models = {}  # For storing hierarchical models
        self.feature_importances = {}
        self.selected_features = {}  # Store selected features by position
        self.target = 'fantasy_points_per_game'
        self.validation_metrics = {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Check if we have the necessary datasets
        self._validate_feature_sets()
    
    def _validate_feature_sets(self):
        """Validate necessary feature sets for training."""
        required_keys = [f"{pos}_train" for pos in ['qb', 'rb', 'wr', 'te']]
        missing_keys = [key for key in required_keys if key not in self.feature_sets or self.feature_sets[key].empty]
        if missing_keys: logger.warning(f"Missing/empty required training sets: {missing_keys} in {list(self.feature_sets.keys())}")
        else: logger.debug("Required training feature sets found.")
        required_proj_keys = [f"{pos}_projection" for pos in ['qb', 'rb', 'wr', 'te']]
        missing_proj_keys = [key for key in required_proj_keys if key not in self.feature_sets or self.feature_sets[key].empty]
        if missing_proj_keys: logger.debug(f"Note: Projection feature sets missing/empty: {missing_proj_keys}")
    
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
            'fantasy_points_ppr', 'fantasy_points_per_game', 'fantasy_points_per_attempt'
        ]
        
        numeric_cols = train_data.select_dtypes(include=['number']).columns.tolist()
        features = [col for col in numeric_cols if col not in exclude_cols]
        
        return features
    

    def select_features_with_method(self, position, method='shap', top_n=20):
        """
        Select features using various methods
        
        Parameters:
        -----------
        position : str
            Position to select features for ('qb', 'rb', 'wr', 'te')
        method : str
            Feature selection method ('importance', 'rfe', 'lasso', 'shap')
        top_n : int
            Number of top features to select
            
        Returns:
        --------
        list
            List of selected feature names
        """
        # Get training data
        train_key = f"{position}_train"
        
        if train_key not in self.feature_sets or self.feature_sets[train_key].empty:
            logger.warning(f"No {position} training data available")
            return []
        
        # Get training data
        train_data = self.feature_sets[train_key].copy()
        
        # Apply meaningful sample filtering
        train_data = self._filter_meaningful_samples(train_data, position)
        
        # Get all potential features
        all_features = self._select_features(train_data)
        
        if len(all_features) < 3:
            logger.warning(f"Not enough features for {position}, need at least 3")
            return all_features
        
        # Get target variable
        target = self.target
        
        if target not in train_data.columns:
            logger.warning(f"Target {target} not found in {position} data")
            return all_features
        
        # Prepare data
        X = train_data[all_features].fillna(0)
        y = train_data[target].fillna(0)
        
        selected_features = []
        
        # Apply the selected method
        if method == 'importance':
            # Use Random Forest for feature importance
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Get feature importances
            importances = model.feature_importances_
            
            # Create DataFrame with feature names and importances
            importance_df = pd.DataFrame({
                'feature': all_features,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Select top features
            selected_features = importance_df.head(top_n)['feature'].tolist()
            
            # Log top features
            logger.info(f"Top {top_n} {position} features by importance:")
            for i, row in importance_df.head(top_n).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        elif method == 'rfe':
            # Use Recursive Feature Elimination
            # Base estimator
            estimator = RandomForestRegressor(n_estimators=50, random_state=42)
            
            # RFE with cross-validation
            selector = RFECV(estimator, step=1, cv=3, scoring='neg_mean_squared_error', min_features_to_select=10)
            
            try:
                selector.fit(X, y)
                
                # Get selected features
                selected_mask = selector.support_
                selected_features = [all_features[i] for i in range(len(all_features)) if selected_mask[i]]
                
                logger.info(f"Selected {len(selected_features)} features with RFE")
                
                # If we still want to limit to top_n features
                if len(selected_features) > top_n:
                    # Train a model on selected features to get their importances
                    X_selected = X[selected_features]
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X_selected, y)
                    
                    # Get feature importances
                    importances = model.feature_importances_
                    
                    # Create DataFrame with feature names and importances
                    importance_df = pd.DataFrame({
                        'feature': selected_features,
                        'importance': importances
                    }).sort_values('importance', ascending=False)
                    
                    # Select top features
                    selected_features = importance_df.head(top_n)['feature'].tolist()
            except Exception as e:
                logger.warning(f"RFE feature selection failed: {e}")
                # Fall back to importance-based selection
                selected_features = self.select_features_with_method(position, 'importance', top_n)
        
        elif method == 'lasso':
            # Use Lasso regularization for feature selection
            from sklearn.linear_model import LassoCV
            
            # Normalize features for Lasso
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Use LassoCV to find optimal alpha
            lasso = LassoCV(cv=3, random_state=42, max_iter=10000)
            
            try:
                lasso.fit(X_scaled, y)
                
                # Get feature coefficients
                coefficients = np.abs(lasso.coef_)
                
                # Create DataFrame with feature names and coefficients
                coef_df = pd.DataFrame({
                    'feature': all_features,
                    'coefficient': coefficients
                }).sort_values('coefficient', ascending=False)
                
                # Select features with non-zero coefficients
                non_zero_features = coef_df[coef_df['coefficient'] > 0]['feature'].tolist()
                
                # Select top features
                selected_features = non_zero_features[:top_n]
                
                logger.info(f"Selected {len(selected_features)} features with Lasso (alpha={lasso.alpha_:.6f})")
            except Exception as e:
                logger.warning(f"Lasso feature selection failed: {e}")
                # Fall back to importance-based selection
                selected_features = self.select_features_with_method(position, 'importance', top_n)
        
        elif method == 'shap':
            # Use SHAP values for feature selection
            try:
                import shap
                
                # Train a model
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                model.fit(X, y)
                
                # Calculate SHAP values
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(X)
                
                # Calculate mean absolute SHAP values for each feature
                mean_shap = np.abs(shap_values).mean(axis=0)
                # Create DataFrame with feature names and SHAP values
                shap_df = pd.DataFrame({
                    'feature': all_features,
                    'shap_value': mean_shap
                }).sort_values('shap_value', ascending=False)
                
                # Select top features
                selected_features = shap_df.head(top_n)['feature'].tolist()
                
                logger.info(f"Selected {len(selected_features)} features with SHAP values")
            except ImportError:
                logger.warning("SHAP library not installed. Falling back to importance-based selection.")
                selected_features = self.select_features_with_method(position, 'importance', top_n)
            except Exception as e:
                logger.warning(f"SHAP feature selection failed: {e}")
                selected_features = self.select_features_with_method(position, 'importance', top_n)
        
        else:
            logger.warning(f"Unknown feature selection method: {method}")
            # Use all features
            selected_features = all_features
        
        return selected_features

    def _create_model(self, model_type, hyperparams):
        """Create a model based on specified type and hyperparameters"""
        if model_type == 'random_forest':
            # Default RF hyperparams
            rf_params = {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 5,  
                'min_samples_leaf': 2,   
                'max_features': 'sqrt',
                'n_jobs': -1,
                'random_state': 42
            }
            
            if hyperparams:
                rf_params.update(hyperparams)
                
            return RandomForestRegressor(**rf_params)
            
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
            
            if hyperparams:
                gb_params.update(hyperparams)
                
            return GradientBoostingRegressor(**gb_params)
            
        elif model_type == 'xgboost':
            # Default XGBoost hyperparams
            xgb_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'min_child_weight': 1,
                'subsample': 0.8,
                'colsample_bytree': 0.8,  # Column subsampling
                'reg_alpha': 0.1,         # L1 regularization
                'reg_lambda': 1.0,        # L2 regularization
                'min_child_weight': 3,    # Controls overfitting
                'objective': 'reg:squarederror',
                'random_state': 42
            }
            
            if hyperparams:
                xgb_params.update(hyperparams)
                
            try:
                import xgboost as xgb
                return xgb.XGBRegressor(**xgb_params)
            except ImportError:
                logger.warning("XGBoost not installed. Using RandomForest instead.")
                return RandomForestRegressor(n_estimators=100, random_state=42)
                
        elif model_type == 'lightgbm':
            # Default LightGBM hyperparams
            lgb_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'num_leaves': 31,
                'min_data_in_leaf': 20,
                'objective': 'regression',
                'random_state': 42
            }
            
            if hyperparams:
                lgb_params.update(hyperparams)
                
            try:
                import lightgbm as lgb
                return lgb.LGBMRegressor(**lgb_params)
            except ImportError:
                logger.warning("LightGBM not installed. Using RandomForest instead.")
                return RandomForestRegressor(n_estimators=100, random_state=42)
                
        else:
            logger.warning(f"Unknown model type: {model_type}. Using RandomForest instead.")
            return RandomForestRegressor(n_estimators=100, random_state=42)

    def train_with_time_series_validation(self, position, model_type='random_forest', hyperparams=None, selected_features=None):
        """
        Train model with time series validation approach
        
        Parameters:
        -----------
        position : str
            Position to train model for ('qb', 'rb', 'wr', 'te')
        model_type : str
            Type of model ('random_forest', 'gradient_boosting', 'xgboost', 'lightgbm')
        hyperparams : dict, optional
            Model hyperparameters
        selected_features : list, optional
            List of pre-selected features to use
            
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
        if selected_features is None:
            # Use feature selection if not provided
            features = self._select_features(train_data_full)
            
            # Apply feature selection
            try:
                # Use more sophisticated feature selection
                selected_features = self.select_features_with_method(position, method='importance', top_n=20)
                if len(selected_features) < 3:
                    logger.warning(f"Feature selection returned too few features for {position}. Using all features.")
                    selected_features = features
            except Exception as e:
                logger.warning(f"Feature selection failed: {e}. Using all features.")
                selected_features = features
        
        logger.info(f"Using {len(selected_features)} features for {position} model")
        self.selected_features[position] = selected_features  # Store for later use
        
        if len(selected_features) < 3:
            logger.warning(f"Not enough features for {position}, need at least 3")
            return None
        
        # Create model
        model = self._create_model(model_type, hyperparams)
        
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
            X_train = train_data_full.loc[train_mask, selected_features].fillna(0)
            y_train = train_data_full.loc[train_mask, self.target].fillna(0)
            
            X_test = train_data_full.loc[test_mask, selected_features].fillna(0)
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
            test_mae = mean_absolute_error(y_test, test_pred)
            
            # Additional metrics
            test_r2 = r2_score(y_test, test_pred)
            
            # Calculate gap
            gap = test_rmse - train_rmse
            
            # Log result
            logger.info(f"Years {train_years} → {test_year}: Train RMSE = {train_rmse:.2f}, Test RMSE = {test_rmse:.2f}, Test MAE = {test_mae:.2f}, R² = {test_r2:.2f}")
            
            # Store result
            validation_results.append({
                'train_years': train_years,
                'test_year': test_year,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'test_mae': test_mae,
                'test_r2': test_r2,
                'gap': gap,
                'train_samples': len(X_train),
                'test_samples': len(X_test)
            })
        
        # Train final model on all data
        X_all = train_data_full[selected_features].fillna(0)
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
            'features': selected_features,
            'model_type': model_type,
            'training_samples': len(X_all),
            'validation_metrics': metrics
        }
        
        # Store feature importances
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': selected_features,
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

    def find_optimal_hyperparameters(self, position, model_type='xgboost', n_trials=50):
        """
        Find optimal hyperparameters using Optuna
        
        Parameters:
        -----------
        position : str
            Position to optimize for ('qb', 'rb', 'wr', 'te')
        model_type : str
            Type of model ('random_forest', 'gradient_boosting', 'xgboost', 'lightgbm')
        n_trials : int
            Number of optimization trials
            
        Returns:
        --------
        dict
            Best hyperparameters
        """
        try:
            import optuna
        except ImportError:
            logger.warning("Optuna not installed. Using default hyperparameters.")
            return None
        
        # Get training data
        train_key = f"{position}_train"
        
        if train_key not in self.feature_sets or self.feature_sets[train_key].empty:
            logger.warning(f"No {position} training data available")
            return None
        
        # Get training data
        train_data = self.feature_sets[train_key].copy()
        
        # Apply meaningful sample filtering
        train_data = self._filter_meaningful_samples(train_data, position)
        
        # Get all available years for time series validation
        all_years = sorted(train_data['season'].unique())
        
        if len(all_years) < 3:
            logger.warning(f"Need at least 3 years of data for hyperparameter tuning. Found: {len(all_years)}")
            return None
        
        # Define objective function for Optuna
        def objective(trial):
            # Define hyperparameters to tune based on model type
            if model_type == 'random_forest':
                hyperparams = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'max_depth': trial.suggest_int('max_depth', 3, 20),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
            elif model_type == 'gradient_boosting':
                hyperparams = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 200),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 2, 10),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10)
                }
            elif model_type == 'xgboost':
                hyperparams = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 2, 12),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                    'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0)
                }
            elif model_type == 'lightgbm':
                hyperparams = {
                    'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                    'max_depth': trial.suggest_int('max_depth', 3, 12),
                    'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50)
                }
            else:
                logger.warning(f"Unknown model type: {model_type}. Using default hyperparameters.")
                return None
            
            # Use feature selection for consistency
            if position in self.selected_features:
                selected_features = self.selected_features[position]
            else:
                selected_features = self.select_features_with_method(position, method='importance', top_n=20)
                self.selected_features[position] = selected_features
            
            # Use time series validation to evaluate this parameter set
            cv_scores = []
            
            # For each test year (using the last 3 years as test)
            for i in range(max(1, len(all_years)-3), len(all_years)):
                test_year = all_years[i]
                train_years = all_years[:i]
                
                # Create train/test split based on years
                train_mask = train_data['season'].isin(train_years)
                test_mask = train_data['season'] == test_year
                
                # Get X and y for train and test
                X_train = train_data.loc[train_mask, selected_features].fillna(0)
                y_train = train_data.loc[train_mask, self.target].fillna(0)
                
                X_test = train_data.loc[test_mask, selected_features].fillna(0)
                y_test = train_data.loc[test_mask, self.target].fillna(0)
                
                # Skip if we have too little data
                if len(X_train) < 10 or len(X_test) < 5:
                    continue
                
                # Create model with trial hyperparameters
                model = self._create_model(model_type, hyperparams)
                
                # Fit model
                model.fit(X_train, y_train)
                
                # Evaluate on test set
                test_pred = model.predict(X_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                cv_scores.append(test_rmse)
            
            # Return average RMSE across folds
            if not cv_scores:
                return float('inf')  # Penalty for invalid configurations
            
            return np.mean(cv_scores)
        
        # Create Optuna study
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials, timeout=600)  # 10 minute timeout
        
        # Get best hyperparameters
        best_params = study.best_params
        logger.info(f"Best hyperparameters for {position} {model_type}: {best_params}")
        logger.info(f"Best RMSE: {study.best_value:.4f}")
        
        return best_params

    def _get_hierarchical_components(self, position):
        """
        Get hierarchical components for a position with emphasis on NGS metrics
        
        Parameters:
        -----------
        position : str
            Position ('qb', 'rb', 'wr', 'te')
            
        Returns:
        --------
        list
            List of component metrics to model
        """
        if position == 'qb':
            return [
                # Volume components
                {'name': 'attempts_per_game', 'type': 'volume'},
                # NGS components (primary)
                {'name': 'ngs_pass_completion_percentage_above_expectation', 'type': 'ngs'},
                {'name': 'ngs_pass_avg_time_to_throw', 'type': 'ngs'},
                {'name': 'ngs_pass_avg_air_yards_differential', 'type': 'ngs'},
                {'name': 'ngs_pass_aggressiveness', 'type': 'ngs'},
                # Traditional metrics (secondary)
                {'name': 'completion_percentage', 'type': 'efficiency'},
                {'name': 'yards_per_attempt', 'type': 'efficiency'},
                {'name': 'td_percentage', 'type': 'efficiency'},
            ]
        elif position == 'rb':
            return [
                # Volume components
                {'name': 'carries_per_game', 'type': 'volume'},
                {'name': 'targets_per_game', 'type': 'volume'},
                # NGS components (primary)
                {'name': 'ngs_rush_rush_yards_over_expected_per_att', 'type': 'ngs'},
                {'name': 'ngs_rush_efficiency', 'type': 'ngs'},
                {'name': 'ngs_rush_percent_attempts_gte_eight_defenders', 'type': 'ngs'},
                # Traditional metrics (secondary)
                {'name': 'yards_per_carry', 'type': 'efficiency'},
                {'name': 'reception_rate', 'type': 'efficiency'},
            ]
        elif position in ['wr', 'te']:
            return [
                # Volume components
                {'name': 'targets_per_game', 'type': 'volume'},
                # NGS components (primary)
                {'name': 'ngs_rec_avg_separation', 'type': 'ngs'},
                {'name': 'ngs_rec_avg_cushion', 'type': 'ngs'},
                {'name': 'ngs_rec_avg_yac_above_expectation', 'type': 'ngs'},
                {'name': 'ngs_rec_percent_share_of_intended_air_yards', 'type': 'ngs'},
                # Traditional metrics (secondary)
                {'name': 'reception_rate', 'type': 'efficiency'},
                {'name': 'yards_per_reception', 'type': 'efficiency'},
            ]
        else:
            return []

    def train_all_positions(self, model_type='xgboost', hyperparams=None, optimize_hyperparams=False):
        """
        Train models for all positions using time series validation
        
        Parameters:
        -----------
        model_type : str
            Type of model ('random_forest', 'gradient_boosting', 'xgboost', 'lightgbm')
        hyperparams : dict, optional
            Model hyperparameters
        optimize_hyperparams : bool
            Whether to use Optuna to find optimal hyperparameters
            
        Returns:
        --------
        dict
            Dictionary of validation metrics by position
        """
        all_metrics = {}
        
        for position in ['qb', 'rb', 'wr', 'te']:
            # Find optimal hyperparameters if requested
            position_hyperparams = hyperparams
            if optimize_hyperparams:
                try:
                    opt_hyperparams = self.find_optimal_hyperparameters(position, model_type)
                    if opt_hyperparams:
                        position_hyperparams = opt_hyperparams
                        logger.info(f"Using optimized hyperparameters for {position}")
                except Exception as e:
                    logger.warning(f"Hyperparameter optimization failed: {e}")
            
            # Perform feature selection
            selected_features = self.select_features_with_method(position, method='shap', top_n=20)
            
            # Train model with time series validation
            metrics = self.train_with_time_series_validation(
                position, 
                model_type=model_type,
                hyperparams=position_hyperparams,
                selected_features=selected_features
            )
            
            if metrics:
                all_metrics[position] = metrics
        
        return all_metrics

    def select_features_with_ngs_priority(self, position, method='importance', top_n=20):
        """Feature selection that prioritizes NGS metrics first"""
        # Get all potential features
        train_data = self.feature_sets[f"{position}_train"].copy()
        train_data = self._filter_meaningful_samples(train_data, position)
        all_features = self._select_features(train_data)
        
        # Find all NGS features in the data
        ngs_prefix = f"ngs_{position.lower()}_" if position in ['rb', 'wr', 'te'] else 'ngs_pass_'
        ngs_features = [f for f in all_features if f.startswith(ngs_prefix)]
        
        # Define critical NGS metrics by position
        critical_ngs = []
        if position == 'qb':
            critical_ngs = ['completion_percentage_above_expectation', 'avg_time_to_throw']
        elif position == 'rb':
            critical_ngs = ['rush_yards_over_expected_per_att', 'efficiency']
        elif position in ['wr', 'te']:
            critical_ngs = ['avg_separation', 'avg_yac_above_expectation', 'percent_share_of_intended_air_yards']
        
        # Ensure critical NGS metrics are included
        priority_features = [f for f in ngs_features if any(metric in f for metric in critical_ngs)]
        
        # Use standard feature selection for remaining slots
        remaining_slots = max(0, top_n - len(priority_features))
        other_features = self.select_features_with_method(position, method, remaining_slots)
        
        # Remove any duplicates
        other_features = [f for f in other_features if f not in priority_features]
        
        final_features = priority_features + other_features
        logger.info(f"Selected {len(final_features)} features for {position} with {len(priority_features)} NGS features prioritized")
        
        return final_features
    
    def train_hierarchical_model(self, position, model_type='xgboost', hyperparams=None):
        """
        Train hierarchical model for a position
        
        Parameters:
        -----------
        position : str
            Position to train for ('qb', 'rb', 'wr', 'te')
        model_type : str
            Type of model ('random_forest', 'gradient_boosting', 'xgboost', 'lightgbm')
        hyperparams : dict, optional
            Model hyperparameters
            
        Returns:
        --------
        dict
            Dictionary of component models and metrics
        """
        logger.info(f"Training hierarchical model for {position}...")
        
        # Get hierarchical components for this position
        components = self._get_hierarchical_components(position)
        
        if not components:
            logger.warning(f"No hierarchical components defined for {position}")
            return None
        
        # Get training data
        train_key = f"{position}_train"
        
        if train_key not in self.feature_sets or self.feature_sets[train_key].empty:
            logger.warning(f"No {position} training data available")
            return None
        
        # Get training data
        train_data = self.feature_sets[train_key].copy()
        
        # Apply meaningful sample filtering
        train_data = self._filter_meaningful_samples(train_data, position)
        
        # Calculate any missing component metrics if needed
        train_data = self._prepare_component_metrics(train_data, position, components)
        
        # Train a model for each component
        component_models = {}
        
        for component in components:
            component_name = component['name']
            
            # Skip if component not in data
            if component_name not in train_data.columns:
                logger.warning(f"Component {component_name} not found in {position} data")
                continue
            
            logger.info(f"Training model for {position} {component_name}...")
            
            # Select features based on component type
            if component['type'] == 'volume':
                # For volume metrics, focus on usage, team context, and career stage
                feature_types = ['usage', 'team', 'career']
            else:
                # For efficiency metrics, focus on skill, physical traits, and experience
                feature_types = ['skill', 'physical', 'experience']
            
            # Select features for this component
            selected_features = self.select_features_with_method(position, method='importance', top_n=15)
            
            # Create model
            model = self._create_model(model_type, hyperparams)
            
            # Get all available years
            all_years = sorted(train_data['season'].unique())
            
            if len(all_years) < 2:
                logger.warning(f"Need at least 2 years of data for {component_name} model. Found: {len(all_years)}")
                continue
            
            # Time series validation for this component
            validation_results = []
            
            # For each test year (starting from the second year)
            for i in range(1, len(all_years)):
                test_year = all_years[i]
                train_years = all_years[:i]
                
                # Create train/test split based on years
                train_mask = train_data['season'].isin(train_years)
                test_mask = train_data['season'] == test_year
                
                # Get X and y for train and test
                X_train = train_data.loc[train_mask, selected_features].fillna(0)
                y_train = train_data.loc[train_mask, component_name].fillna(0)
                
                X_test = train_data.loc[test_mask, selected_features].fillna(0)
                y_test = train_data.loc[test_mask, component_name].fillna(0)
                
                # Skip if we have too little data
                if len(X_train) < 10 or len(X_test) < 5:
                    logger.warning(f"Skipping year {test_year} for {component_name} due to insufficient data")
                    continue
                
                # Fit model on training data
                model.fit(X_train, y_train)
                
                # Evaluate on test set
                test_pred = model.predict(X_test)
                test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
                
                # Store validation result
                validation_results.append({
                    'test_year': test_year,
                    'test_rmse': test_rmse,
                    'test_samples': len(X_test)
                })
            
            # Train final model on all data
            X_all = train_data[selected_features].fillna(0)
            y_all = train_data[component_name].fillna(0)
            
            # Fit final model
            model.fit(X_all, y_all)
            
            # Calculate metrics on all data
            all_pred = model.predict(X_all)
            
            metrics = {
                'rmse': np.sqrt(mean_squared_error(y_all, all_pred)),
                'mae': mean_absolute_error(y_all, all_pred),
                'r2': r2_score(y_all, all_pred)
            }
            
            # Store model info
            component_models[component_name] = {
                'model': model,
                'features': selected_features,
                'metrics': metrics,
                'validation_results': validation_results
            }
            
            logger.info(f"Completed {position} {component_name} model. RMSE: {metrics['rmse']:.4f}, R²: {metrics['r2']:.4f}")
        
        # Store hierarchical models
        self.hierarchical_models[position] = {
            'components': components,
            'models': component_models
        }
        
        logger.info(f"Completed hierarchical modeling for {position}")
        
        return component_models

    def _prepare_component_metrics(self, data, position, components):
        """
        Calculate any missing component metrics
        
        Parameters:
        -----------
        data : DataFrame
            Player data
        position : str
            Position ('qb', 'rb', 'wr', 'te')
        components : list
            List of component metrics
            
        Returns:
        --------
        DataFrame
            Data with component metrics added
        """
        result = data.copy()
        
        for component in components:
            component_name = component['name']
            
            if component_name in result.columns:
                # Already exists
                continue
            
            # Calculate missing components based on position
            if position == 'qb':
                if component_name == 'attempts_per_game' and 'attempts' in result.columns and 'games' in result.columns:
                    result[component_name] = result['attempts'] / result['games'].clip(lower=1)
                elif component_name == 'completion_percentage' and 'completions' in result.columns and 'attempts' in result.columns:
                    result[component_name] = (result['completions'] / result['attempts'].clip(lower=1)) * 100
                elif component_name == 'yards_per_attempt' and 'passing_yards' in result.columns and 'attempts' in result.columns:
                    result[component_name] = result['passing_yards'] / result['attempts'].clip(lower=1)
                elif component_name == 'td_percentage' and 'passing_tds' in result.columns and 'attempts' in result.columns:
                    result[component_name] = (result['passing_tds'] / result['attempts'].clip(lower=1)) * 100
                elif component_name == 'rushing_yards_per_game' and 'rushing_yards' in result.columns and 'games' in result.columns:
                    result[component_name] = result['rushing_yards'] / result['games'].clip(lower=1)
            
            elif position == 'rb':
                if component_name == 'carries_per_game' and 'carries' in result.columns and 'games' in result.columns:
                    result[component_name] = result['carries'] / result['games'].clip(lower=1)
                elif component_name == 'targets_per_game' and 'targets' in result.columns and 'games' in result.columns:
                    result[component_name] = result['targets'] / result['games'].clip(lower=1)
                elif component_name == 'yards_per_carry' and 'rushing_yards' in result.columns and 'carries' in result.columns:
                    result[component_name] = result['rushing_yards'] / result['carries'].clip(lower=1)
                elif component_name == 'reception_rate' and 'receptions' in result.columns and 'targets' in result.columns:
                    result[component_name] = result['receptions'] / result['targets'].clip(lower=1)
                elif component_name == 'yards_per_reception' and 'receiving_yards' in result.columns and 'receptions' in result.columns:
                    result[component_name] = result['receiving_yards'] / result['receptions'].clip(lower=1)
                elif component_name == 'rushing_td_rate' and 'rushing_tds' in result.columns and 'carries' in result.columns:
                    result[component_name] = result['rushing_tds'] / result['carries'].clip(lower=1) * 100
                elif component_name == 'receiving_td_rate' and 'receiving_tds' in result.columns and 'receptions' in result.columns:
                    result[component_name] = result['receiving_tds'] / result['receptions'].clip(lower=1) * 100
            
            elif position in ['wr', 'te']:
                if component_name == 'targets_per_game' and 'targets' in result.columns and 'games' in result.columns:
                    result[component_name] = result['targets'] / result['games'].clip(lower=1)
                elif component_name == 'reception_rate' and 'receptions' in result.columns and 'targets' in result.columns:
                    result[component_name] = result['receptions'] / result['targets'].clip(lower=1)
                elif component_name == 'yards_per_reception' and 'receiving_yards' in result.columns and 'receptions' in result.columns:
                    result[component_name] = result['receiving_yards'] / result['receptions'].clip(lower=1)
                elif component_name == 'receiving_td_rate' and 'receiving_tds' in result.columns and 'receptions' in result.columns:
                    result[component_name] = result['receiving_tds'] / result['receptions'].clip(lower=1) * 100
        
        return result

    def _project_with_hierarchical_model(self, data, position):
        """
        Generate projections using hierarchical models with NGS emphasis
        
        Parameters:
        -----------
        data : DataFrame
            Data containing players to project
        position : str
            Position to project ('qb', 'rb', 'wr', 'te')
                
        Returns:
        --------
        DataFrame
            DataFrame with component and combined projections
        """
        # Check if hierarchical models exist
        if position not in self.hierarchical_models:
            logger.warning(f"No hierarchical models found for {position}")
            return data
        
        # Get component models and info
        hierarchical_info = self.hierarchical_models[position]
        components = hierarchical_info['components']
        component_models = hierarchical_info['models']
        
        # Create a copy of data
        prediction_data = data.copy()
        
        # Calculate any missing component metrics
        prediction_data = self._prepare_component_metrics(prediction_data, position, components)
        
        # Project each component
        for component_name, model_info in component_models.items():
            model = model_info['model']
            features = model_info['features']
            
            # Check for missing features
            missing_features = [f for f in features if f not in prediction_data.columns]
            if missing_features:
                logger.warning(f"Missing features for {component_name}: {missing_features}")
                # Use only available features
                available_features = [f for f in features if f in prediction_data.columns]
                if len(available_features) < 3:
                    logger.warning(f"Not enough features for {component_name} projection")
                    prediction_data[f'projected_{component_name}'] = np.nan
                    continue
                
                X = prediction_data[available_features].fillna(0)
            else:
                # All features available
                X = prediction_data[features].fillna(0)
            
            # Make predictions for this component
            try:
                component_pred = model.predict(X)
                prediction_data[f'projected_{component_name}'] = component_pred
            except Exception as e:
                logger.error(f"Error predicting {component_name}: {e}")
                prediction_data[f'projected_{component_name}'] = np.nan
        
        # Combine component projections into fantasy points based on position
        if position == 'qb':
            # NGS-based components get higher weight than traditional stats
            ngs_weight = 1.3  # 30% more weight to NGS metrics
            
            # Pass CPOE is more predictive than completion percentage
            if 'projected_ngs_pass_completion_percentage_above_expectation' in prediction_data.columns and 'projected_completion_percentage' in prediction_data.columns:
                cpoe = prediction_data['projected_ngs_pass_completion_percentage_above_expectation']
                # Adjust completion percentage based on CPOE (which is more stable year-to-year)
                cpoe_factor = 1.0 + cpoe * 0.01
                prediction_data['adjusted_completion_percentage'] = prediction_data['projected_completion_percentage'] * cpoe_factor
            else:
                prediction_data['adjusted_completion_percentage'] = prediction_data.get('projected_completion_percentage', 0)
            
            # Use NGS air yards differential to adjust yards per attempt
            if 'projected_ngs_pass_avg_air_yards_differential' in prediction_data.columns and 'projected_yards_per_attempt' in prediction_data.columns:
                air_diff = prediction_data['projected_ngs_pass_avg_air_yards_differential']
                ya_factor = 1.0 + air_diff * 0.05
                prediction_data['adjusted_yards_per_attempt'] = prediction_data['projected_yards_per_attempt'] * ya_factor
            else:
                prediction_data['adjusted_yards_per_attempt'] = prediction_data.get('projected_yards_per_attempt', 0)
            
            # Adjust TD percentage based on aggressiveness
            if 'projected_ngs_pass_aggressiveness' in prediction_data.columns and 'projected_td_percentage' in prediction_data.columns:
                agg = prediction_data['projected_ngs_pass_aggressiveness']
                agg_factor = 1.0 + (agg - 15) * 0.01  # Baseline around 15% aggressiveness
                prediction_data['adjusted_td_percentage'] = prediction_data['projected_td_percentage'] * agg_factor
            else:
                prediction_data['adjusted_td_percentage'] = prediction_data.get('projected_td_percentage', 0)
            
            # Calculate passing and rushing yards with NGS-adjusted stats
            if 'projected_attempts_per_game' in prediction_data.columns:
                prediction_data['projected_completions_per_game'] = (prediction_data['projected_attempts_per_game'] * 
                                                                prediction_data['adjusted_completion_percentage'] / 100)
                prediction_data['projected_passing_yards_per_game'] = (prediction_data['projected_attempts_per_game'] * 
                                                                prediction_data['adjusted_yards_per_attempt'])
                prediction_data['projected_passing_tds_per_game'] = (prediction_data['projected_attempts_per_game'] * 
                                                                prediction_data['adjusted_td_percentage'] / 100)
            
            # Combine into fantasy points (based on typical scoring)
            try:
                # Passing: 0.04 pts per yard, 4 pts per TD
                # Rushing: 0.1 pts per yard, 6 pts per TD
                prediction_data['projected_fantasy_pts_from_passing'] = (
                    prediction_data['projected_passing_yards_per_game'] * 0.04 +
                    prediction_data['projected_passing_tds_per_game'] * 4
                )
                
                prediction_data['projected_fantasy_pts_from_rushing'] = (
                    prediction_data.get('projected_rushing_yards_per_game', 0) * 0.1 +
                    prediction_data.get('projected_rushing_tds', 0) * 6 / prediction_data['games'].clip(lower=1)
                )
                
                # Combine components
                prediction_data['projected_points'] = (
                    prediction_data['projected_fantasy_pts_from_passing'] +
                    prediction_data['projected_fantasy_pts_from_rushing']
                )
            except Exception as e:
                logger.error(f"Error combining QB projections: {e}")
        
        elif position == 'rb':
            # NGS-based components get higher weight for RBs
            
            # Rush yards over expected is more predictive than simple yards per carry
            if 'projected_ngs_rush_rush_yards_over_expected_per_att' in prediction_data.columns and 'projected_yards_per_carry' in prediction_data.columns:
                ryoe = prediction_data['projected_ngs_rush_rush_yards_over_expected_per_att']
                # Adjust yards per carry based on RYOE (which measures true skill)
                ryoe_factor = 1.0 + ryoe * 0.1
                prediction_data['adjusted_yards_per_carry'] = prediction_data['projected_yards_per_carry'] * ryoe_factor
            else:
                prediction_data['adjusted_yards_per_carry'] = prediction_data.get('projected_yards_per_carry', 0)
            
            # Adjust based on defenders in box
            if 'projected_ngs_rush_percent_attempts_gte_eight_defenders' in prediction_data.columns:
                def_pct = prediction_data['projected_ngs_rush_percent_attempts_gte_eight_defenders']
                # Higher % means tougher running conditions
                box_factor = 1.0 - (def_pct * 0.002)  # Small penalty for facing stacked boxes
                prediction_data['adjusted_yards_per_carry'] *= box_factor
            
            # Use NGS efficiency for TD rate adjustment
            if 'projected_ngs_rush_efficiency' in prediction_data.columns and 'projected_rushing_td_rate' in prediction_data.columns:
                efficiency = prediction_data['projected_ngs_rush_efficiency']
                eff_factor = 1.0 + (efficiency - prediction_data['projected_ngs_rush_efficiency'].mean()) * 0.1  # Adjust based on deviation from average
                prediction_data['adjusted_rushing_td_rate'] = prediction_data['projected_rushing_td_rate'] * eff_factor
            else:
                prediction_data['adjusted_rushing_td_rate'] = prediction_data.get('projected_rushing_td_rate', 0)
            
            # Calculate rushing stats per game with NGS adjustments
            if 'projected_carries_per_game' in prediction_data.columns:
                prediction_data['projected_rushing_yards_per_game'] = (prediction_data['projected_carries_per_game'] * 
                                                                prediction_data['adjusted_yards_per_carry'])
                prediction_data['projected_rushing_tds_per_game'] = (prediction_data['projected_carries_per_game'] * 
                                                                prediction_data['adjusted_rushing_td_rate'] / 100)
            
            # Calculate receiving stats per game (standard)
            if all(col in prediction_data.columns for col in ['projected_targets_per_game', 'projected_reception_rate', 'projected_yards_per_reception']):
                prediction_data['projected_receptions_per_game'] = (prediction_data['projected_targets_per_game'] * 
                                                            prediction_data['projected_reception_rate'])
                prediction_data['projected_receiving_yards_per_game'] = (prediction_data['projected_receptions_per_game'] * 
                                                                    prediction_data['projected_yards_per_reception'])
            
            # Calculate receiving TDs
            if all(col in prediction_data.columns for col in ['projected_receptions_per_game', 'projected_receiving_td_rate']):
                prediction_data['projected_receiving_tds_per_game'] = (prediction_data['projected_receptions_per_game'] * 
                                                                prediction_data['projected_receiving_td_rate'] / 100)
            
            # Combine into fantasy points (based on typical scoring)
            try:
                # Rushing: 0.1 pts per yard, 6 pts per TD
                # Receiving: 0.1 pts per yard, 6 pts per TD, 0.5 pts per reception (half PPR)
                prediction_data['projected_fantasy_pts_from_rushing'] = (
                    prediction_data['projected_rushing_yards_per_game'] * 0.1 +
                    prediction_data['projected_rushing_tds_per_game'] * 6
                )
                
                prediction_data['projected_fantasy_pts_from_receiving'] = (
                    prediction_data.get('projected_receiving_yards_per_game', 0) * 0.1 +
                    prediction_data.get('projected_receiving_tds_per_game', 0) * 6 +
                    prediction_data.get('projected_receptions_per_game', 0) * 0.5  # Half PPR
                )
                
                # Combine components
                prediction_data['projected_points'] = (
                    prediction_data['projected_fantasy_pts_from_rushing'] +
                    prediction_data['projected_fantasy_pts_from_receiving']
                )
            except Exception as e:
                logger.error(f"Error combining RB projections: {e}")
        
        elif position in ['wr', 'te']:
            # NGS is especially valuable for WR/TE projections
            
            # Separation is highly predictive of success
            if 'projected_ngs_rec_avg_separation' in prediction_data.columns and 'projected_reception_rate' in prediction_data.columns:
                sep = prediction_data['projected_ngs_rec_avg_separation']
                # Better separation = better catch rate
                sep_factor = 1.0 + (sep - 2.8) * 0.05  # Baseline around 2.8 yards of separation
                prediction_data['adjusted_reception_rate'] = prediction_data['projected_reception_rate'] * sep_factor
            else:
                prediction_data['adjusted_reception_rate'] = prediction_data.get('projected_reception_rate', 0)
            
            # YAC above expectation helps adjust yards per reception
            if 'projected_ngs_rec_avg_yac_above_expectation' in prediction_data.columns and 'projected_yards_per_reception' in prediction_data.columns:
                yac_above = prediction_data['projected_ngs_rec_avg_yac_above_expectation']
                # Better YAC = more yards per reception
                yac_factor = 1.0 + yac_above * 0.08
                prediction_data['adjusted_yards_per_reception'] = prediction_data['projected_yards_per_reception'] * yac_factor
            else:
                prediction_data['adjusted_yards_per_reception'] = prediction_data.get('projected_yards_per_reception', 0)
            
            # Air yards share helps project true opportunity
            if 'projected_ngs_rec_percent_share_of_intended_air_yards' in prediction_data.columns and 'projected_targets_per_game' in prediction_data.columns:
                air_share = prediction_data['projected_ngs_rec_percent_share_of_intended_air_yards'] / 100  # Convert to decimal
                # Higher air yards share = more valuable targets
                prediction_data['air_yards_adjustment'] = 1.0 + (air_share - 0.15) * 1.0  # Baseline around 15% share
            else:
                prediction_data['air_yards_adjustment'] = 1.0
            
            # Calculate receiving stats with NGS adjustments
            if all(col in prediction_data.columns for col in ['projected_targets_per_game', 'adjusted_reception_rate', 'adjusted_yards_per_reception']):
                prediction_data['projected_receptions_per_game'] = (prediction_data['projected_targets_per_game'] * 
                                                            prediction_data['adjusted_reception_rate'])
                
                prediction_data['projected_receiving_yards_per_game'] = (prediction_data['projected_receptions_per_game'] * 
                                                                    prediction_data['adjusted_yards_per_reception'])
                
                # Apply air yards adjustment to yards (more air yards = more yards)
                prediction_data['projected_receiving_yards_per_game'] *= prediction_data['air_yards_adjustment']
            
            # Calculate receiving TDs with air yards adjustment (deep targets generate more TDs)
            if all(col in prediction_data.columns for col in ['projected_receptions_per_game', 'projected_receiving_td_rate', 'air_yards_adjustment']):
                td_adjustment = prediction_data['air_yards_adjustment'] ** 1.2  # Exponential effect on TDs
                prediction_data['adjusted_receiving_td_rate'] = prediction_data['projected_receiving_td_rate'] * td_adjustment
                prediction_data['projected_receiving_tds_per_game'] = (prediction_data['projected_receptions_per_game'] * 
                                                                prediction_data['adjusted_receiving_td_rate'] / 100)
            
            # Combine into fantasy points (based on typical scoring)
            try:
                # Receiving: 0.1 pts per yard, 6 pts per TD, 0.5 pts per reception (half PPR)
                prediction_data['projected_points'] = (
                    prediction_data['projected_receiving_yards_per_game'] * 0.1 +
                    prediction_data['projected_receiving_tds_per_game'] * 6 +
                    prediction_data['projected_receptions_per_game'] * 0.5  # Half PPR
                )
            except Exception as e:
                logger.error(f"Error combining WR/TE projections: {e}")
        
        # Fill in projection range using uncertainty in component models
        prediction_data = self._add_projection_range(prediction_data, position)
        
        return prediction_data
    
    def _add_projection_range(self, data, position):
        """Add projection range based on component uncertainties"""
        # Calculate confidence intervals based on position
        position_uncertainty = {
            'qb': 0.15,
            'rb': 0.20,
            'wr': 0.25,
            'te': 0.30
        }
        
        # Create copy of data
        result = data.copy()
        
        # Add projection range
        if 'projected_points' in result.columns:
            uncertainty = position_uncertainty.get(position, 0.20)
            result['projection_low'] = result['projected_points'] * (1 - uncertainty)
            result['projection_high'] = result['projected_points'] * (1 + uncertainty)
            
            # Ensure projections are not negative
            result['projection_low'] = result['projection_low'].clip(lower=0)
        
        # Calculate ceiling based on component maximums
        try:
            # Get component confidence intervals
            comp_cols = [col for col in result.columns if col.startswith('projected_') 
                        and not col in ['projected_points', 'projection_low', 'projection_high']]
            
            if comp_cols:
                # Calculate upside scenario based on components
                for col in comp_cols:
                    result[f'{col}_ceiling'] = result[col] * 1.2  # 20% upside for each component
                
                # Recalculate fantasy points with ceiling components
                if position == 'qb':
                    if all(col in result.columns for col in ['projected_passing_yards_per_game_ceiling', 'projected_passing_tds_per_game_ceiling',
                                                         'projected_rushing_yards_per_game_ceiling']):
                        result['ceiling_projection'] = (
                            result['projected_passing_yards_per_game_ceiling'] * 0.04 +
                            result['projected_passing_tds_per_game_ceiling'] * 4 +
                            result['projected_rushing_yards_per_game_ceiling'] * 0.1 +
                            result['projected_rushing_tds'] * 6 / result['games'].clip(lower=1) * 1.2
                        )
                elif position == 'rb':
                    if all(col in result.columns for col in ['projected_rushing_yards_per_game_ceiling', 'projected_rushing_tds_per_game_ceiling',
                                                         'projected_receiving_yards_per_game_ceiling', 'projected_receiving_tds_per_game_ceiling',
                                                         'projected_receptions_per_game_ceiling']):
                        result['ceiling_projection'] = (
                            result['projected_rushing_yards_per_game_ceiling'] * 0.1 +
                            result['projected_rushing_tds_per_game_ceiling'] * 6 +
                            result['projected_receiving_yards_per_game_ceiling'] * 0.1 +
                            result['projected_receiving_tds_per_game_ceiling'] * 6 +
                            result['projected_receptions_per_game_ceiling'] * 0.5
                        )
                elif position in ['wr', 'te']:
                    if all(col in result.columns for col in ['projected_receiving_yards_per_game_ceiling', 'projected_receiving_tds_per_game_ceiling',
                                                         'projected_receptions_per_game_ceiling']):
                        result['ceiling_projection'] = (
                            result['projected_receiving_yards_per_game_ceiling'] * 0.1 +
                            result['projected_receiving_tds_per_game_ceiling'] * 6 +
                            result['projected_receptions_per_game_ceiling'] * 0.5
                        )
            else:
                # Fall back to standard ceiling calculation
                result['ceiling_projection'] = result['projected_points'] * 1.5
        except Exception as e:
            logger.error(f"Error calculating ceiling projection: {e}")
            result['ceiling_projection'] = result['projected_points'] * 1.5
        
        # Add projection tiers
        result = self._add_projection_tiers(result)
        
        return result
    
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

    def train_all_hierarchical_models(self, model_type='xgboost', hyperparams=None):
        """
        Train hierarchical models for all positions
        
        Parameters:
        -----------
        model_type : str
            Type of model ('random_forest', 'gradient_boosting', 'xgboost', 'lightgbm')
        hyperparams : dict, optional
            Model hyperparameters
            
        Returns:
        --------
        dict
            Dictionary of hierarchical models by position
        """
        all_models = {}
        
        for position in ['qb', 'rb', 'wr', 'te']:
            models = self.train_hierarchical_model(position, model_type, hyperparams)
            
            if models:
                all_models[position] = models
        
        return all_models

    def generate_full_projections(self, projection_data=None, use_do_not_draft=True, use_hierarchical=True):
        """
        Generate projections for all positions
        
        Parameters:
        -----------
        projection_data : dict, optional
            Dictionary of projection data by position
        use_do_not_draft : bool, optional
            Whether to apply the do_not_draft flag (default: True)
        use_hierarchical : bool, optional
            Whether to use hierarchical models (default: True)
            
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
            
            # Generate projections using hierarchical or direct method
            if use_hierarchical and position in self.hierarchical_models:
                logger.info(f"Using hierarchical projections for {position}")
                proj_data = self._project_with_hierarchical_model(data, position)
            else:
                logger.info(f"Using direct projections for {position}")
                proj_data = self.project_players(position, data, use_do_not_draft=use_do_not_draft)
            
            # Apply "do not draft" flags after projection
            if use_do_not_draft and 'do_not_draft' in proj_data.columns:
                do_not_draft_mask = proj_data['do_not_draft'] == 1
                if do_not_draft_mask.any():
                    count = do_not_draft_mask.sum()
                    logger.info(f"Zeroing out projections for {count} {position.upper()}s flagged as 'do not draft'")
                    proj_data.loc[do_not_draft_mask, 'projected_points'] = 0
                    proj_data.loc[do_not_draft_mask, 'projection_low'] = 0
                    proj_data.loc[do_not_draft_mask, 'projection_high'] = 0
                    proj_data.loc[do_not_draft_mask, 'ceiling_projection'] = 0
            
            projections[position] = proj_data
            
            # Log projection stats
            if 'projected_points' in proj_data.columns:
                mean_proj = proj_data['projected_points'].mean()
                min_proj = proj_data['projected_points'].min()
                max_proj = proj_data['projected_points'].max()
                logger.info(f"{position} projections: mean={mean_proj:.2f}, min={min_proj:.2f}, max={max_proj:.2f}")
                
                # Save top players to CSV
                if 'name' in proj_data.columns:
                    top_players = proj_data.sort_values('projected_points', ascending=False)
                    top_players_path = os.path.join(self.output_dir, f"top_{position}_projections.csv")
                    cols = ['player_id', 'name', 'position', 'projected_points', 'projection_low', 'projection_high', 'ceiling_projection']
                    if 'age' in top_players.columns:
                        cols.append('age')
                    component_cols = [col for col in top_players.columns if col.startswith('projected_') and col not in cols]
                    cols.extend(component_cols[:5])  # Add top component predictions
                    
                    # Create a copy of the dataframe for export
                    export_players = top_players[cols].copy()
                    # Modify player_id to skip first three characters
                    export_players['player_id'] = export_players['player_id'].str[3:]
                    # Save the modified dataframe
                    export_players.to_csv(top_players_path, index=False)
                    
                    logger.info(f"Saved top {position} projections to {top_players_path}")
        
        return projections

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
        
        # Add hierarchical models if available
        if position in self.hierarchical_models:
            save_data['hierarchical_models'] = self.hierarchical_models[position]
        
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
        PlayerProjectionModel
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
        
        # Add hierarchical models if available
        if 'hierarchical_models' in model_data:
            instance.hierarchical_models[position] = model_data['hierarchical_models']
        
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
        self.hierarchical_models = {}
        self._load_hierarchical_components()
    
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
                
                # Store hierarchical models separately if available
                if 'hierarchical_models' in model_data:
                    self.hierarchical_models[position] = model_data['hierarchical_models']
                    logger.info(f"Loaded hierarchical models for {position} with {len(model_data['hierarchical_models'].get('models', {}))} component models")
                
            except Exception as e:
                logger.warning(f"Could not load {position} projection model: {e}")
        
        return projection_models
    
    def _load_hierarchical_components(self):
        """Load hierarchical component definitions for each position"""
        # Default component definitions if not loaded from models
        self.component_definitions = {
            'qb': [
                {'name': 'attempts_per_game', 'type': 'volume'},
                {'name': 'completion_percentage', 'type': 'efficiency'},
                {'name': 'yards_per_attempt', 'type': 'efficiency'},
                {'name': 'td_percentage', 'type': 'efficiency'},
                {'name': 'rushing_yards_per_game', 'type': 'volume'},
                {'name': 'rushing_tds', 'type': 'volume'}
            ],
            'rb': [
                {'name': 'carries_per_game', 'type': 'volume'},
                {'name': 'targets_per_game', 'type': 'volume'},
                {'name': 'yards_per_carry', 'type': 'efficiency'},
                {'name': 'reception_rate', 'type': 'efficiency'},
                {'name': 'yards_per_reception', 'type': 'efficiency'},
                {'name': 'rushing_td_rate', 'type': 'efficiency'},
                {'name': 'receiving_td_rate', 'type': 'efficiency'}
            ],
            'wr': [
                {'name': 'targets_per_game', 'type': 'volume'},
                {'name': 'reception_rate', 'type': 'efficiency'},
                {'name': 'yards_per_reception', 'type': 'efficiency'},
                {'name': 'receiving_td_rate', 'type': 'efficiency'}
            ],
            'te': [
                {'name': 'targets_per_game', 'type': 'volume'},
                {'name': 'reception_rate', 'type': 'efficiency'},
                {'name': 'yards_per_reception', 'type': 'efficiency'},
                {'name': 'receiving_td_rate', 'type': 'efficiency'}
            ]
        }
        
        # Override with actual component definitions from loaded models
        for position, hierarchical_model in self.hierarchical_models.items():
            if 'components' in hierarchical_model:
                self.component_definitions[position] = hierarchical_model['components']
                logger.info(f"Using loaded component definitions for {position}")
    
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
        
        # Hierarchical Models
        if 'hierarchical_models' in model_data:
            hierarchical = model_data['hierarchical_models']
            component_models = hierarchical.get('models', {})
            logger.info(f"\nHierarchical Models: {len(component_models)} component models")
            
            if component_models:
                logger.info("Component Models:")
                for component, comp_data in component_models.items():
                    metrics = comp_data.get('metrics', {})
                    if metrics:
                        logger.info(f"  {component}: RMSE={metrics.get('rmse', 'N/A'):.4f}, R²={metrics.get('r2', 'N/A'):.4f}")
        
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
    
    def get_hierarchical_model(self, position):
        """
        Retrieve a specific position's hierarchical model
        
        Parameters:
        -----------
        position : str
            Player position (lowercase)
        
        Returns:
        --------
        dict or None
            Loaded hierarchical model data for the position
        """
        return self.hierarchical_models.get(position.lower())
    
    def project_player_stats(self, player_data, position):
        """
        Project player stats using hierarchical or direct models
        
        Parameters:
        -----------
        player_data : dict
            Dictionary of player stats
        position : str
            Player position
            
        Returns:
        --------
        dict
            Dictionary with projected stats
        """
        position = position.lower()
        
        # First try hierarchical projection
        if self.get_hierarchical_model(position):
            return self._project_with_hierarchical_model(player_data, position)
        
        # Fall back to direct projection
        direct_model = self.get_model(position)
        if direct_model:
            return self._project_with_direct_model(player_data, position)
        
        # No model available
        return {"projected_points": 0}
    
    def _project_with_hierarchical_model(self, player_data, position):
        """Project using hierarchical component models"""
        hierarchical_model = self.get_hierarchical_model(position)
        if not hierarchical_model or 'models' not in hierarchical_model:
            return {"projected_points": 0}
        
        # Get component models
        component_models = hierarchical_model['models']
        components = hierarchical_model.get('components', self.component_definitions.get(position, []))
        
        # Project each component
        projections = {}
        for component in components:
            component_name = component['name']
            if component_name in component_models:
                comp_model = component_models[component_name]
                model = comp_model.get('model')
                features = comp_model.get('features', [])
                
                # Create feature vector
                if not features:
                    continue
                    
                # Check if we have enough features
                available_features = [f for f in features if f in player_data]
                if len(available_features) < 3:
                    continue
                
                try:
                    # Get feature values
                    X = np.array([[player_data.get(f, 0) for f in features]])
                    
                    # Make prediction
                    pred = model.predict(X)[0]
                    
                    # Store prediction
                    projections[f'projected_{component_name}'] = pred
                except Exception as e:
                    logger.error(f"Error projecting {component_name}: {e}")
        
        # Convert component projections to fantasy points based on position
        points = 0
        
        try:
            if position == 'qb':
                # Passing yards: 0.04 pts per yard
                if 'projected_attempts_per_game' in projections and 'projected_yards_per_attempt' in projections:
                    passing_yards = projections['projected_attempts_per_game'] * projections['projected_yards_per_attempt']
                    points += passing_yards * 0.04
                
                # Passing TDs: 4 pts per TD
                if 'projected_attempts_per_game' in projections and 'projected_td_percentage' in projections:
                    passing_tds = projections['projected_attempts_per_game'] * projections['projected_td_percentage'] / 100
                    points += passing_tds * 4
                
                # Rushing: 0.1 pts per yard, 6 pts per TD
                if 'projected_rushing_yards_per_game' in projections:
                    points += projections['projected_rushing_yards_per_game'] * 0.1
                
                if 'projected_rushing_tds' in projections:
                    points += projections['projected_rushing_tds'] * 6 / player_data.get('games', 16)
            
            elif position == 'rb':
                # Rushing: 0.1 pts per yard, 6 pts per TD
                if 'projected_carries_per_game' in projections and 'projected_yards_per_carry' in projections:
                    rushing_yards = projections['projected_carries_per_game'] * projections['projected_yards_per_carry']
                    points += rushing_yards * 0.1
                
                if 'projected_carries_per_game' in projections and 'projected_rushing_td_rate' in projections:
                    rushing_tds = projections['projected_carries_per_game'] * projections['projected_rushing_td_rate'] / 100
                    points += rushing_tds * 6
                
                # Receiving: 0.1 pts per yard, 6 pts per TD, 0.5 pts per reception (half PPR)
                if all(k in projections for k in ['projected_targets_per_game', 'projected_reception_rate', 'projected_yards_per_reception']):
                    receptions = projections['projected_targets_per_game'] * projections['projected_reception_rate']
                    receiving_yards = receptions * projections['projected_yards_per_reception']
                    
                    points += receptions * 0.5  # Half PPR
                    points += receiving_yards * 0.1
                
                if 'projected_receiving_td_rate' in projections and 'projected_targets_per_game' in projections and 'projected_reception_rate' in projections:
                    receptions = projections['projected_targets_per_game'] * projections['projected_reception_rate']
                    receiving_tds = receptions * projections['projected_receiving_td_rate'] / 100
                    points += receiving_tds * 6
            
            elif position in ['wr', 'te']:
                # Receiving: 0.1 pts per yard, 6 pts per TD, 0.5 pts per reception (half PPR)
                if all(k in projections for k in ['projected_targets_per_game', 'projected_reception_rate', 'projected_yards_per_reception']):
                    receptions = projections['projected_targets_per_game'] * projections['projected_reception_rate']
                    receiving_yards = receptions * projections['projected_yards_per_reception']
                    
                    points += receptions * 0.5  # Half PPR
                    points += receiving_yards * 0.1
                
                if 'projected_receiving_td_rate' in projections and 'projected_targets_per_game' in projections and 'projected_reception_rate' in projections:
                    receptions = projections['projected_targets_per_game'] * projections['projected_reception_rate']
                    receiving_tds = receptions * projections['projected_receiving_td_rate'] / 100
                    points += receiving_tds * 6
            
            # Add all projections plus final points
            result = projections.copy()
            result['projected_points'] = points
            
            # Add uncertainty range
            uncertainty = {'qb': 0.15, 'rb': 0.20, 'wr': 0.25, 'te': 0.30}.get(position, 0.20)
            result['projection_low'] = max(0, points * (1 - uncertainty))
            result['projection_high'] = points * (1 + uncertainty)
            
            # Add ceiling projection with increased component values
            ceiling_factor = {'qb': 1.4, 'rb': 1.5, 'wr': 1.7, 'te': 1.6}.get(position, 1.5)
            result['ceiling_projection'] = points * ceiling_factor
            
            return result
        
        except Exception as e:
            logger.error(f"Error combining projections: {e}")
            return {"projected_points": 0}
    
    def _project_with_direct_model(self, player_data, position):
        """Project using direct model"""
        model_data = self.get_model(position)
        if not model_data or 'model' not in model_data:
            return {"projected_points": 0}
        
        model = model_data['model']
        features = model_data.get('features', [])
        
        # Check if we have enough features
        available_features = [f for f in features if f in player_data]
        if len(available_features) < 3:
            return {"projected_points": 0}
        
        try:
            # Get feature values
            X = np.array([[player_data.get(f, 0) for f in features]])
            
            # Make prediction
            points = model.predict(X)[0]
            
            # Add uncertainty range
            uncertainty = {'qb': 0.15, 'rb': 0.20, 'wr': 0.25, 'te': 0.30}.get(position, 0.20)
            
            return {
                'projected_points': points,
                'projection_low': max(0, points * (1 - uncertainty)),
                'projection_high': points * (1 + uncertainty),
                'ceiling_projection': points * {'qb': 1.4, 'rb': 1.5, 'wr': 1.7, 'te': 1.6}.get(position, 1.5)
            }
        except Exception as e:
            logger.error(f"Error making direct projection: {e}")
            return {"projected_points": 0}