#!/usr/bin/env python3
"""
Fantasy Football Player Projection System

This script orchestrates the player projection pipeline using
a modular approach with configuration from YAML.
"""


import sys
import yaml
from datetime import datetime
from src.pipeline.projection_pipeline import ProjectionPipeline


def load_config(config_path='config.yaml'):
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        sys.exit(1)

def setup_logging(config):
    """Set up logging based on configuration - redirecting all loggers"""
    import logging
    
    # Get log settings from config
    log_level_name = config.get('logging', {}).get('level', 'INFO')
    log_level = getattr(logging, log_level_name, logging.INFO)
    log_file = config.get('logging', {}).get('file', 'fantasy_football.log')
    
    # Create handlers
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)  # File gets everything at configured level
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.WARNING)  # Console only gets warnings and above
    
    # Create formatters
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # Configure root logger (affects all loggers)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)  # Ensure we capture all logs at INFO level or above
    
    # Remove any existing handlers to avoid duplicates
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Add the handlers to the root logger
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    # Configure Optuna logger specifically if we need to
    import optuna
    optuna_logger = logging.getLogger("optuna")
    
    # Remove all handlers from Optuna logger to prevent direct console output
    for handler in optuna_logger.handlers[:]:
        optuna_logger.removeHandler(handler)
    
    # Optuna logger will inherit from root logger, ensuring it logs to our file but respects console level
    
    # Create our application logger
    logger = logging.getLogger("fantasy_football")
    logger.info(f"Logging initialized at level {log_level_name}")
    
    return logger


def main():
    """Main entry point for the fantasy football projection system"""
    # Load configuration
    config = load_config()
    
    # Setup logging
    logger = setup_logging(config)
    logger.info("Starting Fantasy Football Player Projection System")
    
    # Create and run the projection pipeline
    pipeline = ProjectionPipeline(config)
    projections = pipeline.run()
    
    # Display summary of results
    print_summary(pipeline)
    
    logger.info("Fantasy Football Player Projection System completed successfully")
    return 0

def print_summary(pipeline):
    """Print a human-readable summary of the projection results"""
    summary = pipeline.get_summary()
    
    print("\n========== Fantasy Football Projections Summary ==========")
    print(f"League: {summary['league']}")
    print(f"Teams: {summary['teams']}")
    print(f"Projection year: {summary['projection_year']}")
    print(f"Used hierarchical projections: {summary['used_hierarchical']}")
    print(f"Used NGS data: {summary['used_ngs']}")
    
    # Display top players by position
    for position in ['qb', 'rb', 'wr', 'te']:
        if position in summary['top_players']:
            print(f"\nTop 5 projected {position.upper()} players:")
            for player in summary['top_players'][position]:
                print(f"  {player['name']}: {player['points']:.2f} pts/game")
    
    print(f"\nProjections completed successfully!")
    print(f"Output data and projections saved to the output directory")

if __name__ == "__main__":
    sys.exit(main())