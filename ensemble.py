import jax
import jax.numpy as jnp
from mpi4py import MPI
from forecaster import training_loop, forecast, forecast_1step_with_loss

# Initialize an ensemble of forecasters
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

# Train each forecaster in the ensemble using MPI
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

    # Distribute forecasters among MPI ranks
    local_ensemble = ensemble[rank::size]  # Each rank gets a subset of forecasters
    print(f"Rank {rank}: Local ensemble size = {len(local_ensemble)}")

    local_trained = []
    local_loss_history = [] if track_loss else None

    # Train forecasters in the local ensemble
    for W, b in local_ensemble:
        if track_loss:
            # Track loss history for this forecaster
            losses = []
            for epoch in range(num_epochs):
                delta = grad((W, b), X, y)
                W -= 0.1 * delta[0]
                b -= 0.1 * delta[1]
                loss = forecast_1step_with_loss((W, b), X, y)
                losses.append(float(loss))  # Ensure loss is a Python scalar for easier handling
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

# Aggregate predictions from the ensemble using MPI
def aggregate_forecasts_mpi(ensemble, X, horizon):
    """
    Aggregate predictions from the ensemble using MPI.

    Args:
        ensemble: Trained ensemble of forecasters.
        X: Input data.
        horizon: Number of time steps to forecast.

    Returns:
        All predictions from the ensemble (on rank 0).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribute forecasters among MPI ranks
    local_ensemble = ensemble[rank::size]  # Each rank gets a subset of forecasters
    local_predictions = []

    # Generate predictions locally
    for W, b in local_ensemble:
        y_predicted = forecast(horizon, X, W, b)
        print(f"Rank {rank}: Prediction: {y_predicted}")
        local_predictions.append(y_predicted)

    # Gather predictions on rank 0
    print(f"Rank {rank}: Entering comm.gather with {len(local_predictions)} predictions.")
    gathered_predictions = comm.gather(local_predictions, root=0)       
    print(f"Rank {rank}: Finished comm.gather.")


    if rank == 0:
        # Combine results from all ranks
        all_predictions = [pred for rank_preds in gathered_predictions for pred in rank_preds]
        return all_predictions
    return None

# Debugging and Profiling Utilities
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

