# src/models/model_evaluation.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import learning_curve, cross_validate
import os
import logging

logger = logging.getLogger(__name__)

def evaluate_model_fit(model, X_train, y_train, X_val, y_val):
    """Evaluate potential overfitting by comparing train and validation metrics"""
    # Get training performance
    train_pred = model.predict(X_train)
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    train_r2 = r2_score(y_train, train_pred)
    
    # Get validation performance
    val_pred = model.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    val_r2 = r2_score(y_val, val_pred)
    
    # Calculate the gap (indicator of overfitting)
    rmse_gap = train_rmse - val_rmse
    r2_gap = train_r2 - val_r2
    
    logger.info(f"Training RMSE: {train_rmse:.2f}, R²: {train_r2:.2f}")
    logger.info(f"Validation RMSE: {val_rmse:.2f}, R²: {val_r2:.2f}")
    logger.info(f"Gap (train-val) RMSE: {rmse_gap:.2f}, R²: {r2_gap:.2f}")
    
    # Rule of thumb: if training performance is significantly better than validation,
    # you likely have overfitting
    if train_r2 > val_r2 + 0.1:
        logger.warning("Model may be overfitting (R² gap > 0.1)")
    
    return {
        'train_rmse': train_rmse,
        'train_r2': train_r2,
        'val_rmse': val_rmse,
        'val_r2': val_r2,
        'rmse_gap': rmse_gap,
        'r2_gap': r2_gap
    }

def plot_learning_curves(model, X, y, cv=5, output_path=None):
    """Plot learning curves to detect overfitting"""
    # Calculate learning curve data
    train_sizes, train_scores, val_scores = learning_curve(
        model, X, y, cv=cv, scoring='neg_mean_squared_error',
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    # Convert MSE to RMSE
    train_rmse = np.sqrt(-train_scores).mean(axis=1)
    val_rmse = np.sqrt(-val_scores).mean(axis=1)
    
    # Plot curves
    plt.figure(figsize=(12, 8))
    plt.plot(train_sizes, train_rmse, 'o-', color='blue', label='Training RMSE')
    plt.plot(train_sizes, val_rmse, 'o-', color='red', label='Validation RMSE')
    plt.title('Learning Curves')
    plt.xlabel('Training Set Size')
    plt.ylabel('RMSE')
    plt.legend()
    plt.grid(True)
    
    # Examine gap at full training size
    gap = val_rmse[-1] - train_rmse[-1]
    logger.info(f"Final gap between validation and training RMSE: {gap:.2f}")
    
    # Save the plot if a path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Learning curve plot saved to {output_path}")
    
    return plt, {
        'train_sizes': train_sizes,
        'train_rmse': train_rmse,
        'val_rmse': val_rmse,
        'final_gap': gap
    }

def analyze_feature_importance(model, feature_names, output_path=None):
    """Analyze feature importances to identify potential overfitting signals"""
    if not hasattr(model, 'feature_importances_'):
        logger.warning("Model doesn't have feature_importances_ attribute")
        return None, None
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Plot feature importances
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importances')
    plt.bar(range(len(indices)), importances[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    
    # Log top and bottom features
    logger.info("Top 10 features:")
    for i in range(min(10, len(feature_names))):
        logger.info(f"{feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Check for distribution of importance
    # If just a few features have most importance, model likely isn't overfitting
    # If many minor features have significant importance, possible overfitting
    importance_threshold = 0.01
    minor_features = sum(i > importance_threshold for i in importances)
    logger.info(f"Features with importance > {importance_threshold}: {minor_features} out of {len(feature_names)}")
    
    # Save the plot if a path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {output_path}")
    
    # Create a DataFrame with importance results
    importance_df = pd.DataFrame({
        'feature': [feature_names[i] for i in indices],
        'importance': importances[indices]
    })
    
    return plt, importance_df

def temporal_validation(model, data, years, feature_cols, target='fantasy_points_per_game', output_path=None):
    """Validate model across multiple seasons to detect overfitting to specific years"""
    results = []
    
    for test_year in years[1:]:  # Skip first year as we need training data
        # Use all previous years for training
        train_years = [y for y in years if y < test_year]
        
        # Skip if no training years
        if not train_years:
            continue
        
        # Split data
        train_mask = data['season'].isin(train_years)
        test_mask = data['season'] == test_year
        
        # Skip if not enough data for either split
        if train_mask.sum() < 10 or test_mask.sum() < 5:
            logger.warning(f"Insufficient data for temporal validation with test year {test_year}")
            continue
        
        train_data = data[train_mask]
        test_data = data[test_mask]
        
        # Extract X and y - use only features that exist in the data
        valid_features = [c for c in feature_cols if c in train_data.columns]
        X_train = train_data[valid_features]
        y_train = train_data[target]
        X_test = test_data[valid_features]
        y_test = test_data[target]
        
        # Fit model
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        
        results.append({
            'train_years': train_years,
            'test_year': test_year,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'gap': test_rmse - train_rmse,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        })
        
        logger.info(f"Years {train_years} → {test_year}: Train RMSE = {train_rmse:.2f}, Test RMSE = {test_rmse:.2f}, Gap = {test_rmse - train_rmse:.2f}")
    
    # Analyze consistency across years
    if results:
        gaps = [r['gap'] for r in results]
        avg_gap = sum(gaps) / len(gaps)
        std_gap = np.std(gaps)
        
        logger.info(f"\nAverage gap: {avg_gap:.2f}, Std Dev: {std_gap:.2f}")
        logger.info("Consistent gaps across years suggest stable model (less overfitting)")
        logger.info("Widely varying gaps suggest overfitting to specific seasons")
        
        # Create gap visualization
        plt.figure(figsize=(12, 6))
        test_years = [r['test_year'] for r in results]
        train_rmse = [r['train_rmse'] for r in results]
        test_rmse = [r['test_rmse'] for r in results]
        
        x = np.arange(len(test_years))
        width = 0.35
        
        plt.bar(x - width/2, train_rmse, width, label='Train RMSE')
        plt.bar(x + width/2, test_rmse, width, label='Test RMSE')
        
        plt.xlabel('Test Year')
        plt.ylabel('RMSE')
        plt.title('Model Performance Across Years')
        plt.xticks(x, test_years)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save the plot if a path is provided
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Temporal validation plot saved to {output_path}")
        
        # Create results DataFrame
        results_df = pd.DataFrame(results)
        
        return plt, results_df
    else:
        logger.warning("No valid results from temporal validation")
        return None, None

def cross_validation_analysis(model, X, y, cv=5, output_path=None):
    """Analyze cross-validation scores to detect overfitting"""
    # Run cross-validation
    cv_results = cross_validate(
        model, X, y, 
        cv=cv, 
        scoring=['neg_mean_squared_error', 'r2'],
        return_train_score=True
    )
    
    # Extract and convert scores
    train_rmse = np.sqrt(-cv_results['train_neg_mean_squared_error'])
    test_rmse = np.sqrt(-cv_results['test_neg_mean_squared_error'])
    train_r2 = cv_results['train_r2']
    test_r2 = cv_results['test_r2']
    
    # Calculate statistics
    train_rmse_mean, train_rmse_std = train_rmse.mean(), train_rmse.std()
    test_rmse_mean, test_rmse_std = test_rmse.mean(), test_rmse.std()
    train_r2_mean, train_r2_std = train_r2.mean(), train_r2.std()
    test_r2_mean, test_r2_std = test_r2.mean(), test_r2.std()
    
    logger.info(f"Train RMSE: {train_rmse_mean:.2f} ± {train_rmse_std:.2f}")
    logger.info(f"Test RMSE: {test_rmse_mean:.2f} ± {test_rmse_std:.2f}")
    logger.info(f"Train R²: {train_r2_mean:.2f} ± {train_r2_std:.2f}")
    logger.info(f"Test R²: {test_r2_mean:.2f} ± {test_r2_std:.2f}")
    
    # High variance in test scores can indicate overfitting
    if test_rmse_std / test_rmse_mean > 0.2:
        logger.warning("High variance in test scores may indicate overfitting")
    
    # Create boxplot
    plt.figure(figsize=(10, 6))
    data = [train_rmse, test_rmse]
    plt.boxplot(data, labels=['Train RMSE', 'Test RMSE'])
    plt.title('Cross-Validation Score Distribution')
    plt.grid(True, alpha=0.3)
    
    # Add a second plot for R²
    plt.figure(figsize=(10, 6))
    data_r2 = [train_r2, test_r2]
    plt.boxplot(data_r2, labels=['Train R²', 'Test R²'])
    plt.title('Cross-Validation R² Distribution')
    plt.grid(True, alpha=0.3)
    
    # Save the plots if a path is provided
    if output_path:
        # Save RMSE plot
        rmse_path = output_path.replace('.png', '_rmse.png')
        plt.figure(1)
        plt.savefig(rmse_path, dpi=300, bbox_inches='tight')
        logger.info(f"CV RMSE plot saved to {rmse_path}")
        
        # Save R² plot
        r2_path = output_path.replace('.png', '_r2.png')
        plt.figure(2)
        plt.savefig(r2_path, dpi=300, bbox_inches='tight')
        logger.info(f"CV R² plot saved to {r2_path}")
    
    return plt, {
        'train_rmse_mean': train_rmse_mean,
        'train_rmse_std': train_rmse_std,
        'test_rmse_mean': test_rmse_mean,
        'test_rmse_std': test_rmse_std,
        'train_r2_mean': train_r2_mean,
        'train_r2_std': train_r2_std,
        'test_r2_mean': test_r2_mean,
        'test_r2_std': test_r2_std,
        'train_rmse': train_rmse.tolist(),
        'test_rmse': test_rmse.tolist(),
        'train_r2': train_r2.tolist(),
        'test_r2': test_r2.tolist()
    }