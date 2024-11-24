import os
import matplotlib.pyplot as plt
import numpy as np

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

# 1. Prediction Trajectory Over Time
def plot_prediction_trajectories(predictions, horizon, folder="plots"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, prediction in enumerate(predictions):
        ax.plot(range(1, horizon + 1), prediction, label=f"Forecaster {i + 1}")
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Predictions")
    ax.set_title("Prediction Trajectories")
    ax.legend()
    save_plot(fig, "prediction_trajectories.png", folder)

# 2. Uncertainty Visualization
def plot_uncertainty(mean_prediction, std_dev, horizon, folder="plots"):
    fig, ax = plt.subplots(figsize=(8, 5))
    time_steps = range(1, horizon + 1)
    ax.plot(time_steps, mean_prediction, label="Mean Prediction", linewidth=2)
    ax.fill_between(
        time_steps,
        mean_prediction - std_dev,
        mean_prediction + std_dev,
        alpha=0.3,
        label="Uncertainty Bounds"
    )
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Prediction")
    ax.set_title("Prediction with Uncertainty Bounds")
    ax.legend()
    save_plot(fig, "uncertainty_bounds.png", folder)

# 3. Prediction Distribution
def plot_prediction_distribution(predictions, folder="plots"):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(predictions, positions=range(1, predictions.shape[0] + 1))
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Predictions")
    ax.set_title("Prediction Distribution")
    save_plot(fig, "prediction_distribution.png", folder)

# 4. Forecaster Training Convergence
def plot_training_loss(loss_history, folder="plots"):
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, losses in enumerate(loss_history):
        ax.plot(losses, label=f"Forecaster {i + 1}")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    ax.set_title("Forecaster Training Loss")
    ax.legend()
    save_plot(fig, "training_loss.png", folder)

# 5. Heatmap of Predictions
def plot_prediction_heatmap(predictions, folder="plots"):
    fig, ax = plt.subplots(figsize=(8, 5))
    cax = ax.imshow(predictions, aspect='auto', cmap='viridis')
    fig.colorbar(cax, ax=ax)
    ax.set_xlabel("Time Steps")
    ax.set_ylabel("Forecaster Index")
    ax.set_title("Heatmap of Predictions")
    save_plot(fig, "prediction_heatmap.png", folder)