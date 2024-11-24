from mpi4py import MPI
import jax
import jax.numpy as jnp
from forecaster import forecast_1step_with_loss
from ensemble import create_ensemble, train_ensemble_mpi, aggregate_forecasts_mpi
from config import X, y, W, b, num_forecasters, noise_std, num_epochs, horizon
from plotter import (
    plot_prediction_trajectories,
    plot_uncertainty,
    plot_prediction_distribution,
    plot_training_loss,
    plot_prediction_heatmap
)
import os

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        print("Initializing ensemble forecasting with MPI...\n")

    # Gradient computation
    grad = jax.grad(forecast_1step_with_loss)

    # Create or broadcast the ensemble
    if rank == 0:
        print("Creating ensemble...")
        ensemble = create_ensemble(num_forecasters, W, b, noise_std)
        print(f"Ensemble created with {num_forecasters} forecasters.")
    else:
        ensemble = None

    ensemble = comm.bcast(ensemble, root=0)
    if rank == 0:
        print("Ensemble broadcasted to all ranks.\n")

    # Train ensemble using MPI
    if rank == 0:
        print("Starting training...")
    trained_ensemble, loss_history = train_ensemble_mpi(
        ensemble, X, y, grad, num_epochs, track_loss=True
    )
    if rank == 0:
        print("Training completed.\n")

    # Make predictions and aggregate results
    if rank == 0:
        print("Aggregating predictions...")
        predictions = aggregate_forecasts_mpi(trained_ensemble, X, horizon)
        predictions = jnp.array(predictions)

        # Calculate mean and standard deviation
        mean_prediction = jnp.mean(predictions, axis=0)
        std_dev = jnp.std(predictions, axis=0)

        print("Predictions aggregated.")
        print(f"Mean prediction: {mean_prediction}")
        print(f"Standard deviation: {std_dev}\n")

        # Ensure the `plots` folder exists
        os.makedirs("plots", exist_ok=True)

        # Create plots
        print("Generating plots...")
        plot_prediction_trajectories(predictions, horizon)
        plot_uncertainty(mean_prediction, std_dev, horizon)
        plot_prediction_distribution(predictions)
        plot_prediction_heatmap(predictions)

        # Plot training loss if loss history is tracked
        if loss_history is not None:
            plot_training_loss(loss_history)

        print("All plots saved in the 'plots/' directory.")

if __name__ == "__main__":
    main()
