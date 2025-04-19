# Fantasy Football Draft Optimization: Hybrid ML/RL Approach

## Overview

This project implements a sophisticated system for optimizing fantasy football draft strategy. It combines machine learning techniques for player performance projection with reinforcement learning to train an agent capable of making strategic draft decisions within a simulated environment. The goal is to move beyond simple heuristics and static rankings to build consistently competitive fantasy teams tailored to specific league rules.

## Features

*   **Data Pipeline:** Ingests historical NFL player statistics (including Next Gen Stats via `nfl-data-py`) and league-specific settings (via `espn-api`).
*   **Player Projection:** Utilizes supervised learning models (XGBoost recommended) trained on engineered features to forecast player performance (Fantasy Points Per Game) with uncertainty estimates (low/high/ceiling). Includes temporal validation and Optuna hyperparameter optimization.
*   **Feature Engineering:** Creates advanced features capturing player efficiency, usage, career trajectory, risk factors, and NGS insights.
*   **Player Clustering:** Groups players into tiers using K-Means based on performance profiles to aid analysis and optional filtering.
*   **Reinforcement Learning Agent:** Trains a Proximal Policy Optimization (PPO) agent within a custom Gym environment (`FantasyDraftEnv`) to learn an optimal drafting policy based on maximizing team Value Over Replacement Player (VORP).
*   **Draft Simulation:** The `FantasyDraftEnv` simulates the draft process, including need-based heuristic opponents.
*   **Configuration Driven:** Uses `config.yaml` to control data sources, modeling choices, RL training parameters, HPO settings, and caching.
*   **Visualization & Analysis:** Generates plots for EDA, feature importance, model evaluation, and RL training progress.

## Project Structure

```
.
├── configs/
│ └── league_settings.json # Auto-generated/User-provided league rules
├── data/
│ ├── raw/ # Cached raw data (CSVs)
│ ├── processed/ # Cached processed data & feature sets (CSVs)
│ ├── models/ # Saved projection models (.joblib), HPO DBs (.db)
│ └── outputs/ # Generated plots, evaluation results, logs
├── src/
│ ├── analysis/
│ │ ├── ml_explorer.py # Advanced ML-focused EDA/Viz
│ │ ├── tensorboard_parser.py # Parses RL logs for plotting
│ │ └── visualizer.py # General data visualization
│ │ └── analyzer.py # General data analysis
│ ├── data/
│ │ └── loader.py # Data loading and initial processing
│ ├── features/
│ │ └── engineering.py # Feature engineering and clustering
│ ├── models/
│ │ ├── projections.py # Supervised learning projection models
│ │ ├── rl_draft_agent.py # Gym Environment for RL Draft
│ │ └── callback.py # Custom SB3 callbacks (e.g., Optuna)
│ └── pipeline/
│ ├── projection_pipeline.py # Orchestrates projection generation
│ └── rl_pipeline.py # Orchestrates RL agent training & HPO
├── generate_config.py # Script to fetch ESPN league settings
├── main.py # Main entry point to run pipelines
├── config.yaml # Central configuration file
├── requirements.txt # Python dependencies
├── README.md # This file
```
## Setup Instructions

1.  **Clone Repository:**
    ```bash
    git clone <repository_url>
    cd <repository_directory>
    ```
2.  **Python Environment:** Use Python 3.10. Create and activate a virtual environment:
    ```bash
    python3.10 -m venv .venv
    source .venv/bin/activate  # Linux/macOS
    # .venv\Scripts\activate    # Windows
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    *Note: If you encounter issues with specific libraries (especially ML/RL ones), consult their documentation for system-specific prerequisites.*
4.  **Get ESPN Credentials (Private Leagues Only):**
    *   If your target ESPN league is private, you need your `espn_s2` and `swid` cookies.
    *   Log in to your ESPN Fantasy Football league in your browser.
    *   Open Developer Tools (F12 or right-click -> Inspect).
    *   Go to the "Application" (Chrome/Edge) or "Storage" (Firefox) tab.
    *   Find "Cookies" in the sidebar and select `https://fantasy.espn.com`.
    *   Locate the cookies named `espn_s2` and `SWID`. Copy their respective "Value" fields.
    *   Paste these values into the `league.espn_s2` and `league.swid` fields in `config.yaml` **OR** be prepared to enter them when running `generate_config.py`. **Keep these confidential!**


## Configuration (`config.yaml`)

This file controls the entire process. Key sections:

*   **`league`**: Your ESPN league ID, the target season year, and optional private league credentials.
*   **`data`**: Start year for historical data, whether to include NGS.
*   **`clustering`**: Number of clusters (tiers), how many bottom tiers to filter out, whether to use the filtered data for training/projection.
*   **`projections`**: Target year for projections, model type (xgboost recommended), whether to run Optuna HPO for projections, feature selection method.
*   **`caching`**: Flags to control using cached data for raw, processed, and feature sets. Saves time on subsequent runs.
*   **`models_loading`**: Set `use_existing_projection_models: true` to load previously trained models (`*.joblib` in `data/models/`) and skip projection training. Features must still be generated or cached.
*   **`rl_training`**:
    *   `enabled`: Set to `true` to run the RL training pipeline.
    *   `agent_draft_position`: Simulates the agent drafting from this slot (1-based).
    *   `final_training_timesteps`: Total steps for the final RL training run.
    *   `ppo_hyperparams`: Default/best hyperparameters for the PPO agent.
    *   `optuna`: Settings for hyperparameter optimization (enable, trials, parallel jobs, search space).
*   **`paths`**: Locations for data, models, outputs, and logs.
*   **`logging`**: Logging level and file output.

## Usage Workflow

1. **Set Environment Variables**:
    * In order to get the parallelization to work, you must export the following variables, **AFTER** you activate the environment
    ```bash
    export OMP_NUM_THREADS=1                                 
    export MKL_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1
    export VECLIB_MAXIMUM_THREADS=1
    export NUMEXPR_NUM_THREADS=1
    ```

2.  **Generate League Configuration:** (Required unless `configs/league_settings.json` already exists and is correct)
    *   Run the config generator script. It will prompt for League ID, Year, and credentials if needed (or use values from `config.yaml`).
    ```bash
    python generate_config.py
    ```
    *   This creates/updates `configs/league_settings.json`. Verify its contents.

3.  **Customize `config.yaml`:**
    *   Ensure `league.id` and `league.year` match your target league and season.
    *   Set `projections.projection_year` to the season you want projections *for*.
    *   Configure `data.start_year` for historical context.
    *   Decide on caching (`caching.*`), model loading (`models_loading.use_existing_projection_models`), and RL training (`rl_training.enabled`, `optuna.enabled`).
    *   Adjust model types, feature selection, HPO settings as desired.

4.  **Run the Main Pipeline:**
    ```bash
    python main.py
    ```
    *   This script reads `config.yaml` and executes the configured steps:
        *   Loads/processes data (using cache if enabled).
        *   Performs feature engineering (using cache if enabled).
        *   Trains/loads projection models.
        *   Generates player projections.
        *   Optionally runs Optuna HPO for the RL agent.
        *   Optionally trains the final RL agent.
        *   Generates visualizations and evaluation results (if enabled).

5.  **Review Outputs:**
    *   **Projections:** Final projections saved as `data/models/player_projections.pkl` (a dictionary of DataFrames, one per position) and also as individual CSVs `data/models/top_{pos}_projections.csv`.
    *   **Models:** Trained projection models saved as `data/models/{pos}_model.joblib`. RL model saved as `data/models/ppo_fantasy_draft_agent.zip` (or custom name). RL checkpoints in `data/models/rl_checkpoints/`.
    *   **Visualizations:** Plots saved in `data/outputs/` categorized by position and analysis type (e.g., `data/outputs/rb/correlations/`, `data/outputs/evaluation/`).
    *   **Logs:** Main log file (`fantasy_football.log`) and potentially RL worker logs (`data/outputs/PID_logs/`) are created. TensorBoard logs in `ppo_fantasydraft_tensorboard/`.

## Key Components Explained

*   **Player Projections:** Uses XGBoost (or others) to predict next-season performance based on historical stats, engineered features, and NGS data. Temporal validation ensures robustness. Output includes point projections and uncertainty ranges.
*   **RL Environment (`FantasyDraftEnv`):** Simulates a snake draft. The state includes draft progress, agent roster status, positional needs, market signals (positional runs), and top player values. The agent receives a reward based on the final team's VORP.
*   **RL Agent (PPO):** Learns a policy to choose draft actions (Best QB, ..., BPA) to maximize the VORP reward. Trained over millions of simulated drafts. Optuna helps find optimal PPO hyperparameters.

## Dependencies

See `requirements.txt`. Key libraries include:

*   `pandas`, `numpy`
*   `scikit-learn`
*   `xgboost`, `lightgbm` (Optional but recommended)
*   `stable-baselines3[extra]` (RL)
*   `optuna` (Optional for HPO)
*   `matplotlib`, `seaborn` (Plotting)
*   `nfl-data-py` (NFL Data)
*   `espn-api` (ESPN League Data)
*   `gymnasium` (RL Environment Base)
*   `tbparse` (TensorBoard Log Parsing)
*   `joblib`, `pickle`, `pyyaml` (Serialization/Config)

---