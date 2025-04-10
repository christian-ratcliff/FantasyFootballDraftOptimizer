#src/analysis/tensorboard_parser.py

import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # For checking finite values

# Try importing tbparse, handle if not installed
try:
    import tbparse
    TBPARSE_AVAILABLE = True
except ImportError:
    TBPARSE_AVAILABLE = False
    logging.getLogger("fantasy_football").error("tbparse not installed. Run 'pip install tbparse' to enable TensorBoard log parsing.")

logger = logging.getLogger("fantasy_football") # Use main logger or specific analysis logger

def process_and_plot_tb_data(log_dir, output_csv_path="training_data.csv", output_plot_path="training_plots_combined.png"):
    """
    Reads TensorBoard event files, saves data to CSV, and creates custom plots.

    Args:
        log_dir (str): Path to the TensorBoard log directory
                       (e.g., './ppo_fantasydraft_tensorboard/PPO_DraftAgent_1').
        output_csv_path (str): Path to save the extracted data as CSV.
        output_plot_path (str): Path to save the combined plot image.
    """
    if not TBPARSE_AVAILABLE:
        logger.error("tbparse library is required but not installed. Skipping TensorBoard data processing.")
        return

    if not os.path.isdir(log_dir):
        logger.error(f"TensorBoard log directory not found: {log_dir}. Skipping processing.")
        return

    logger.info(f"Processing TensorBoard logs from: {log_dir}")

    try:
        # Initialize the reader. Point it to the specific run directory if multiple exist.
        # If unsure, point to the parent directory, tbparse can handle subdirs.
        reader = tbparse.SummaryReader(log_dir, extra_columns={'dir_name'}) # dir_name helps if you have multiple runs logged
        df = reader.scalars # Get scalar data as DataFrame

        if df.empty:
            logger.warning(f"No scalar data found in TensorBoard logs at: {log_dir}")
            return

        # Rename columns for clarity
        df.rename(columns={'step': 'Timesteps', 'tag': 'Metric', 'value': 'Value'}, inplace=True)

        # --- Data Cleaning & Filtering ---
        # Remove non-finite values that might cause plotting issues
        initial_rows = len(df)
        df = df[np.isfinite(df['Value'])]
        if len(df) < initial_rows:
            logger.warning(f"Removed {initial_rows - len(df)} non-finite rows from TensorBoard data.")


        metrics_to_keep = [
                    'rollout/ep_rew_mean',      
                    'train/loss',
                    'train/value_loss',
                    'train/policy_gradient_loss',
                    'train/entropy_loss',
                    'train/approx_kl',
                    'train/clip_fraction',
                    'train/explained_variance',
                ]
        # Optional: Filter specific metrics if needed
        df = df[df['Metric'].isin(metrics_to_keep)].copy()

        # --- Save to CSV ---
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
            df.to_csv(output_csv_path, index=False)
            logger.info(f"Saved extracted TensorBoard data to: {output_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save TensorBoard data to CSV: {e}", exc_info=True)


        # --- Custom Plotting ---
        metrics = sorted(df['Metric'].unique()) # Get unique metrics sorted alphabetically
        if not metrics:
            logger.warning("No metrics found in DataFrame after processing.")
            return

        logger.info(f"Generating custom plot for metrics: {metrics}")

        ncols = 3 # Adjust layout as needed
        nrows = (len(metrics) + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 4 * nrows), squeeze=False)
        axes = axes.flatten()

        for i, metric_name in enumerate(metrics):
            if i >= len(axes):
                logger.warning(f"Exceeded number of axes ({len(axes)}). Skipping plot for {metric_name}.")
                break # Stop if we run out of plot axes

            ax = axes[i]
            metric_df = df[df['Metric'] == metric_name]

            if not metric_df.empty:
                # Clean title
                clean_title = metric_name.replace('rollout/', '').replace('train/', '').replace('time/', '')

                ax.plot(metric_df['Timesteps'], metric_df['Value'], label=clean_title, linewidth=1.5)

                # Add rolling average
                window_size = max(5, len(metric_df) // 20) # Dynamic window
                if len(metric_df) >= window_size:
                    try:
                         # Use forward-fill then back-fill for NaNs potentially introduced by rolling
                        rolling_avg = metric_df['Value'].rolling(window=window_size, min_periods=1, center=True).mean().fillna(method='ffill').fillna(method='bfill')
                        ax.plot(metric_df['Timesteps'], rolling_avg, linestyle='--', alpha=0.7, label='Avg', linewidth=1.0)
                    except Exception as roll_err:
                         logger.warning(f"Could not calc rolling avg for {metric_name}: {roll_err}")

                # Apply log scale for specific loss metrics
                if clean_title in ['loss', 'value_loss']:
                    # Check if there are positive values before setting log scale
                    if (metric_df['Value'] > 1e-8).any():
                        ax.set_yscale('log')
                    else:
                        logger.warning(f"Cannot use log scale for {clean_title}, no positive values > 1e-8.")

                ax.set_title(clean_title, fontsize=10)
                ax.set_xlabel("Timesteps", fontsize=8)
                ax.set_ylabel("Value", fontsize=8)
                ax.tick_params(axis='both', which='major', labelsize=8)
                ax.grid(True, linestyle='--', alpha=0.6)
                ax.legend(loc='best', fontsize='x-small')
            else:
                logger.warning(f"No data points found for metric '{metric_name}'.")
                ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes, fontsize=9, color='grey')


        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle("RL Agent Training Metrics (from TensorBoard)", fontsize=14)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save the combined plot
        try:
            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_plot_path), exist_ok=True)
            fig.savefig(output_plot_path, dpi=200)
            logger.info(f"Saved combined training plot to: {output_plot_path}")
        except Exception as e:
            logger.error(f"Failed to save combined plot: {e}", exc_info=True)
        finally:
            plt.close(fig) # Ensure figure is closed

    except Exception as e:
        logger.error(f"Error processing TensorBoard logs in {log_dir}: {e}", exc_info=True)