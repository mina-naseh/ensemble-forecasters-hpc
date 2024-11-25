import jax
import jax.numpy as jnp
from mpi4py import MPI
from forecaster import training_loop, forecast, forecast_1step_with_loss
import pandas as pd
import numpy as np
import os


def ensure_dir(directory):
    """Ensure the directory exists."""
    if not os.path.exists(directory):
        os.makedirs(directory)


def create_ensemble(num_forecasters: int, W: jnp.array, b: jnp.array, noise_std: float):
    """
    Create an ensemble of forecasters with different initializations.

    Args:
        num_forecasters: Number of forecasters in the ensemble.
        W: Initial weight matrix.
        b: Initial bias vector.
        noise_std: Standard deviation for noise to initialize forecasters.

    Returns:
        A list of (W, b) tuples for each forecaster.
    """
    ensemble = []
    for i in range(num_forecasters):
        key = jax.random.PRNGKey(i)  # Unique random seed for each forecaster
        W_noise = jax.random.normal(key, W.shape) * noise_std
        b_noise = jax.random.normal(key, b.shape) * noise_std

        W_init = W + W_noise
        b_init = b + b_noise

        ensemble.append((W_init, b_init))
    return ensemble


def train_ensemble_mpi(ensemble, X, y, grad, num_epochs, track_loss=False):
    """
    Train the ensemble of forecasters using MPI for parallelism.

    Args:
        ensemble: The list of forecaster parameters (W, b).
        X: Input data.
        y: Target data.
        grad: Gradient computation function.
        num_epochs: Number of epochs for training.
        track_loss: Whether to track and return loss history for each forecaster.

    Returns:
        Trained ensemble (on rank 0) and optional loss history.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Dynamic forecaster distribution
    local_ensemble = [
        ensemble[i] for i in range(len(ensemble)) if i % size == rank
    ]
    print(f"Rank {rank}: Local ensemble size = {len(local_ensemble)}")

    local_trained = []
    local_loss_history = [] if track_loss else None

    # Train forecasters in the local ensemble
    for W, b in local_ensemble:
        if track_loss:
            losses = []
            for epoch in range(num_epochs):
                delta = grad((W, b), X, y)
                W -= 0.1 * delta[0]
                b -= 0.1 * delta[1]
                loss = forecast_1step_with_loss((W, b), X, y)
                losses.append(float(loss))
            local_trained.append((W, b))
            local_loss_history.append(losses)
        else:
            W, b = training_loop(grad, num_epochs, W, b, X, y)
            local_trained.append((W, b))

    # Gather trained forecasters and loss history on rank 0
    gathered_trained = comm.gather(local_trained, root=0)
    gathered_loss_history = comm.gather(local_loss_history, root=0) if track_loss else None

    if rank == 0:
        # Combine results from all ranks
        trained_ensemble = [forecaster for rank_trained in gathered_trained for forecaster in rank_trained]
        if track_loss:
            loss_history = [losses for rank_losses in gathered_loss_history for losses in rank_losses]
            return trained_ensemble, loss_history
        return trained_ensemble, None
    return None, None


def aggregate_forecasts_mpi(ensemble, X, horizon):
    """
    Aggregate predictions from the ensemble using MPI.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if ensemble is None:
        raise ValueError(f"Rank {rank}: Received None for ensemble. Ensure training and broadcast are correct.")

    local_ensemble = [ensemble[i] for i in range(len(ensemble)) if i % size == rank]
    local_predictions = []

    for idx, (W, b) in enumerate(local_ensemble):
        y_predicted = forecast(horizon, X, W, b)
        print(f"Rank {rank}: Forecaster {idx} generated prediction with shape {y_predicted.shape}.")
        local_predictions.append(y_predicted)

    # Gather predictions on rank 0
    print(f"Rank {rank}: Entering comm.gather with {len(local_predictions)} predictions.")
    try:
        gathered_predictions = comm.gather(local_predictions, root=0)
    except Exception as e:
        raise RuntimeError(f"Rank {rank}: Error during comm.gather - {e}")

    print(f"Rank {rank}: Finished comm.gather.")

    if rank == 0:
        # Combine results from all ranks
        if not gathered_predictions:
            raise ValueError("Rank 0: Gathered predictions are empty or None.")
        all_predictions = [pred for rank_preds in gathered_predictions for pred in rank_preds]
        return all_predictions
    return None


def debug_and_profile(rank, stage, start_time, end_time, additional_info=""):
    """
    Helper function to log timing and debugging information.

    Args:
        rank: Rank of the MPI process.
        stage: Stage of execution (e.g., "Training", "Aggregation").
        start_time: Start time of the stage.
        end_time: End time of the stage.
        additional_info: Additional context or debugging info.
    """
    elapsed_time = end_time - start_time
    print(f"Rank {rank}: {stage} completed in {elapsed_time:.4f} seconds. {additional_info}")


def export_statistics_to_csv(predictions, loss_history, horizon, stats_folder="stats"):
    """
    Export prediction statistics and loss statistics to separate CSV files in the specified folder.

    Args:
        predictions: A 3D numpy array of shape (num_forecasters, time_steps, features).
        loss_history: A 2D list or array of shape (num_forecasters, num_epochs) containing the loss history.
        horizon: Number of time steps to forecast.
        stats_folder: Folder where statistics files will be saved.
    """
    ensure_dir(stats_folder)

    output_file_predictions = os.path.join(stats_folder, "prediction_stats.csv")
    output_file_loss = os.path.join(stats_folder, "loss_stats.csv")

    num_forecasters, time_steps, features = predictions.shape

    mean_predictions = predictions.mean(axis=0)
    std_predictions = predictions.std(axis=0)  
    quantiles_5th = np.percentile(predictions, 5, axis=0)
    quantiles_50th = np.percentile(predictions, 50, axis=0)
    quantiles_95th = np.percentile(predictions, 95, axis=0)

    prediction_data = {
        "Time Step": [],
        "Feature": [],
        "Mean Prediction": [],
        "Standard Deviation": [],
        "5th Percentile": [],
        "Median (50th Percentile)": [],
        "95th Percentile": []
    }

    for t in range(time_steps):
        for f in range(features):
            prediction_data["Time Step"].append(t + 1)
            prediction_data["Feature"].append(f + 1)
            prediction_data["Mean Prediction"].append(mean_predictions[t, f])
            prediction_data["Standard Deviation"].append(std_predictions[t, f])
            prediction_data["5th Percentile"].append(quantiles_5th[t, f])
            prediction_data["Median (50th Percentile)"].append(quantiles_50th[t, f])
            prediction_data["95th Percentile"].append(quantiles_95th[t, f])

    df_predictions = pd.DataFrame(prediction_data)
    df_predictions.to_csv(output_file_predictions, index=False)
    print(f"Prediction statistics exported to {output_file_predictions}")

    # Compute loss statistics across forecasters
    loss_array = np.array(loss_history)  # Convert to array for easier computation
    loss_mean = loss_array.mean(axis=0)  # Mean loss per epoch
    loss_std = loss_array.std(axis=0)    # Std deviation loss per epoch
    loss_min = loss_array.min(axis=0)    # Minimum loss per epoch
    loss_max = loss_array.max(axis=0)    # Maximum loss per epoch

    loss_data = {
        "Epoch": list(range(1, loss_array.shape[1] + 1)),
        "Mean Loss": list(loss_mean),
        "Std Dev Loss": list(loss_std),
        "Min Loss": list(loss_min),
        "Max Loss": list(loss_max),
    }

    df_loss = pd.DataFrame(loss_data)
    df_loss.to_csv(output_file_loss, index=False)
    print(f"Loss statistics exported to {output_file_loss}")