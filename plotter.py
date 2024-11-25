import os
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp

# Ensure the output directory exists
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Save the plot
def save_plot(fig, filename, folder="plots"):
    ensure_dir(folder)
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {filepath}")

def plot_prediction_trajectories(predictions, horizon, folder="plots"):
    """
    Plots the trajectory of predictions over time for each forecaster.

    Args:
        predictions: 3D array of shape (n_forecasters, horizon, features).
        horizon: Number of time steps for predictions.
        folder: Directory to save the plot.
    """
    if predictions.ndim != 3:
        raise ValueError("Invalid predictions shape. Expected a 3D array.")

    # Aggregate across features (axis 2) to reduce to 2D
    aggregated_predictions = jnp.mean(predictions, axis=2)  # Shape: (n_forecasters, horizon)

    fig, ax = plt.subplots(figsize=(8, 5))
    for i, prediction in enumerate(aggregated_predictions):
        ax.plot(range(1, horizon + 1), prediction, label=f"Forecaster {i + 1}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Predictions")
    ax.set_title("Prediction Trajectories")
    ax.legend()
    save_plot(fig, "prediction_trajectories.png", folder)


def plot_uncertainty(mean_prediction, std_dev, horizon, folder="plots"):
    """
    Plots mean predictions with uncertainty bounds.

    Args:
        mean_prediction: Mean predictions (2D array).
        std_dev: Standard deviations (2D array).
        horizon: Number of time steps to forecast.
        folder: Directory to save the plot.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    time_steps = range(1, horizon + 1)

    # Handle each feature (column) separately
    for feature_idx in range(mean_prediction.shape[1]):
        mean = mean_prediction[:, feature_idx]
        std = std_dev[:, feature_idx]
        
        ax.plot(time_steps, mean, label=f"Mean Prediction (Feature {feature_idx + 1})", linewidth=2)
        ax.fill_between(
            time_steps,
            mean - std,
            mean + std,
            alpha=0.3,
            label=f"Uncertainty Bounds (Feature {feature_idx + 1})"
        )
    
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Prediction")
    ax.set_title("Prediction with Uncertainty Bounds")
    ax.legend()
    save_plot(fig, "uncertainty_bounds.png", folder)


def plot_prediction_distribution(predictions, folder="plots"):
    """
    Plots the distribution of predictions as a boxplot.

    Args:
        predictions: 3D array of shape (n_forecasters, horizon, features).
        folder: Directory to save the plot.
    """
    print(f"Initial predictions shape: {predictions.shape}")

    if predictions.ndim != 3:
        raise ValueError("Invalid predictions shape. Expected a 3D array.")

    # Aggregate predictions across features (axis=2) to reduce to 2D
    aggregated_predictions = jnp.mean(predictions, axis=2)  # Shape: (n_forecasters, horizon)
    print(f"Aggregated predictions shape (after mean along features): {aggregated_predictions.shape}")

    # Prepare data for boxplot: Each time step gets all forecaster predictions
    boxplot_data = [aggregated_predictions[:, t] for t in range(aggregated_predictions.shape[1])]
    print(f"Boxplot data length: {len(boxplot_data)}, Each entry length: {[len(entry) for entry in boxplot_data]}")

    # Plot the boxplot
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(boxplot_data, positions=range(1, len(boxplot_data) + 1))  # One boxplot per time step
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Predictions")
    ax.set_title("Prediction Distribution")
    save_plot(fig, "prediction_distribution.png", folder)


def plot_training_loss(loss_history, folder="plots"):
    """
    Plots the training loss for each forecaster over epochs.

    Args:
        loss_history: A list of lists containing loss values per epoch for each forecaster.
        folder: Directory to save the plot.
    """
    if not loss_history or not isinstance(loss_history[0], list):
        raise ValueError("Invalid loss_history format. Expected a list of lists.")
    
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, losses in enumerate(loss_history):
        ax.plot(losses, label=f"Forecaster {i + 1}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Forecaster Training Loss")
    ax.legend()
    save_plot(fig, "training_loss.png", folder)


def plot_prediction_heatmap(predictions, folder="plots"):
    """
    Plots a heatmap of predictions aggregated across features.

    Args:
        predictions: 3D array of predictions (n_forecasters, horizon, features).
        folder: Directory to save the plot.
    """
    if predictions.ndim != 3:
        raise ValueError("Invalid predictions shape. Expected a 3D array.")

    # Aggregate across features (axis 2) to reduce to 2D
    aggregated_predictions = jnp.mean(predictions, axis=2)  # Shape: (n_forecasters, horizon)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    cax = ax.imshow(aggregated_predictions.T, aspect='auto', cmap='viridis')  # Transpose for time on x-axis
    fig.colorbar(cax, ax=ax)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Forecaster Index")
    ax.set_title("Heatmap of Predictions (Aggregated Across Features)")
    save_plot(fig, "prediction_heatmap.png", folder)
