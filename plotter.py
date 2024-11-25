import os
import matplotlib.pyplot as plt
import numpy as np
import jax.numpy as jnp


def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def save_plot(fig, filename, folder="plots"):
    ensure_dir(folder)
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved plot: {filepath}")


def plot_random_forecasters(predictions, horizon, num_random_forecasters=4):
    """
    Plots trajectories for a random subset of forecasters.

    Args:
        predictions (array-like): Array of predictions (shape: num_forecasters x horizon).
        horizon (int): Number of time steps.
        num_random_forecasters (int): Number of random forecasters to plot.
    """
    num_forecasters = predictions.shape[0]

    # Randomly select forecasters
    random_indices = np.random.choice(num_forecasters, num_random_forecasters, replace=False)
    print(f"Selected forecaster indices: {random_indices}")

    plt.figure(figsize=(10, 6))
    for idx in random_indices:
        plt.plot(range(1, horizon + 1), predictions[idx], label=f"Forecaster {idx+1}")

    plt.title("Randomly Selected Forecaster Trajectories")
    plt.xlabel("Time Steps")
    plt.ylabel("Predictions")
    plt.legend(loc="upper left", ncol=2)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/random_forecasters.png")


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
    Plot the distribution of predictions over time steps using box plots.
    Args:
        predictions: A 3D numpy array of shape (num_forecasters, time_steps, features).
        folder: Directory to save the plot.
    """

    # Aggregate data along the forecaster dimension
    boxplot_data = predictions.mean(axis=-1).T  # Shape: (time_steps, num_forecasters)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.boxplot(
        boxplot_data,
        positions=range(1, boxplot_data.shape[0] + 1),
        widths=0.5,
        patch_artist=True,
        boxprops=dict(facecolor="lightblue", color="darkblue"),
        medianprops=dict(color="red", linewidth=2),
        whiskerprops=dict(color="darkblue"),
        capprops=dict(color="darkblue"),
        flierprops=dict(marker="o", markerfacecolor="orange", markersize=5, linestyle="none"),
    )

    ax.yaxis.grid(True, linestyle="--", alpha=0.7)

    ax.set_xlabel("Time Steps", fontsize=12)
    ax.set_ylabel("Predictions", fontsize=12)
    ax.set_title("Prediction Distribution", fontsize=14)
    ax.set_xticks(range(1, boxplot_data.shape[0] + 1))
    ensure_dir(folder)
    save_plot(fig, "prediction_distribution.png", folder)


def plot_random_forecasters_loss(loss_history, random_indices, num_epochs):
    """
    Plots loss history for a given subset of forecasters.

    Args:
        loss_history (list of lists): Loss history for all forecasters.
        random_indices (list or array): Indices of randomly selected forecasters.
        num_epochs (int): Number of epochs.
    """
    plt.figure(figsize=(10, 6))
    for idx in random_indices:
        plt.plot(range(1, num_epochs + 1), loss_history[idx], label=f"Forecaster {idx+1}")

    plt.title("Loss History for Randomly Selected Forecasters")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("plots/random_forecasters_loss.png")


def plot_quantiles(predictions, horizon, folder="plots"):
    """
    Plot the 5th, 50th (median), and 95th percentiles for predictions over time.

    Args:
        predictions: A 3D numpy array of shape (num_forecasters, time_steps, features).
        horizon: Number of time steps to forecast.
        folder: Directory to save the plot.
    """
    # Compute quantiles across the forecaster dimension
    quantiles_5th = np.percentile(predictions, 5, axis=0)  # Shape: (time_steps, features)
    quantiles_50th = np.percentile(predictions, 50, axis=0)  # Median
    quantiles_95th = np.percentile(predictions, 95, axis=0)  # 95th percentile

    time_steps = range(1, horizon + 1)
    fig, ax = plt.subplots(figsize=(10, 6))

    for feature_idx in range(quantiles_5th.shape[1]):
        ax.plot(time_steps, quantiles_50th[:, feature_idx], label=f"Median (Feature {feature_idx + 1})", linewidth=2)
        ax.fill_between(
            time_steps,
            quantiles_5th[:, feature_idx],
            quantiles_95th[:, feature_idx],
            alpha=0.3,
            label=f"5th-95th Percentile Range (Feature {feature_idx + 1})"
        )

    ax.set_xlabel("Time Steps", fontsize=12)
    ax.set_ylabel("Prediction", fontsize=12)
    ax.set_title("Prediction Quantiles Over Time", fontsize=14)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    save_plot(fig, "prediction_quantiles.png", folder)