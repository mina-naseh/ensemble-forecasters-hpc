import jax
import jax.numpy as jnp
from mpi4py import MPI
from forecaster import training_loop, forecast, forecast_1step_with_loss


# Initialize an ensemble of forecasters
def create_ensemble(num_forecasters: int, W: jnp.array, b: jnp.array, noise_std: float):
    ensemble = []
    for i in range(num_forecasters):
        key = jax.random.PRNGKey(i)  # Unique random seed for each forecaster
        W_noise = jax.random.normal(key, W.shape) * noise_std
        b_noise = jax.random.normal(key, b.shape) * noise_std

        W_init = W + W_noise
        b_init = b + b_noise

        ensemble.append((W_init, b_init))
    return ensemble

# # Train each forecaster in the ensemble
# def train_ensemble(ensemble, X, y, grad, num_epochs):
#     trained_ensemble = []
#     for W, b in ensemble:
#         W_trained, b_trained = training_loop(grad, num_epochs, W, b, X, y)
#         trained_ensemble.append((W_trained, b_trained))
#     return trained_ensemble

# Train each forecaster in the ensemble using MPI
def train_ensemble_mpi(ensemble, X, y, grad, num_epochs, track_loss=False):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribute forecasters among MPI ranks
    local_ensemble = ensemble[rank::size]  # Each rank gets a subset of forecasters
    local_trained = []
    local_loss_history = [] if track_loss else None

    for W, b in local_ensemble:
        if track_loss:
            # Track loss history for this forecaster
            losses = []
            for epoch in range(num_epochs):
                delta = grad((W, b), X, y)
                W -= 0.1 * delta[0]
                b -= 0.1 * delta[1]
                loss = forecast_1step_with_loss((W, b), X, y)
                losses.append(loss)
            local_trained.append((W, b))
            local_loss_history.append(losses)
        else:
            W, b = training_loop(grad, num_epochs, W, b, X, y)
            local_trained.append((W, b))

    # Gather trained forecasters and loss history on rank 0
    gathered_trained = comm.gather(local_trained, root=0)
    gathered_loss_history = comm.gather(local_loss_history, root=0) if track_loss else None

    if rank == 0:
        trained_ensemble = [forecaster for rank_trained in gathered_trained for forecaster in rank_trained]
        if track_loss:
            loss_history = [losses for rank_losses in gathered_loss_history for losses in rank_losses]
            return trained_ensemble, loss_history
        return trained_ensemble, None
    return None, None

# # Aggregate predictions from the ensemble
# def aggregate_forecasts(ensemble, X, horizon):
#     predictions = []
#     for W, b in ensemble:
#         y_predicted = forecast(horizon, X, W, b)
#         predictions.append(y_predicted)
#     return predictions

# Aggregate predictions from the ensemble using MPI
def aggregate_forecasts_mpi(ensemble, X, horizon):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Distribute forecasters among MPI ranks
    local_ensemble = ensemble[rank::size]  # Each rank gets a subset of forecasters
    local_predictions = []

    for W, b in local_ensemble:
        y_predicted = forecast(horizon, X, W, b)
        local_predictions.append(y_predicted)

    # Gather predictions on rank 0
    gathered_predictions = comm.gather(local_predictions, root=0)

    # Combine all predictions on rank 0
    if rank == 0:
        all_predictions = [pred for rank_preds in gathered_predictions for pred in rank_preds]
        return all_predictions
    return None