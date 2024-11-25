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
import time

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    start_time = time.time()

    # Initialization messages
    if rank == 0:
        print("Rank 0: Initializing ensemble forecasting with MPI...")
        print(f"Total MPI processes: {size}\n")
    else:
        print(f"Rank {rank}: Waiting for broadcast...\n")
    
    comm.Barrier()  # Synchronize before starting gradient computation

    # Gradient computation
    grad = jax.grad(forecast_1step_with_loss)

    comm.Barrier()  # Synchronize before creating or broadcasting the ensemble
    if rank == 0:
        print("Rank 0: Creating ensemble...")
        ensemble = create_ensemble(num_forecasters, W, b, noise_std)
        print(f"Rank 0: Ensemble created with {num_forecasters} forecasters.")
    else:
        ensemble = None

    # Broadcast the ensemble to all ranks
    ensemble = comm.bcast(ensemble, root=0)
    print(f"Rank {rank}: Received ensemble broadcast.")

    comm.Barrier()  # Synchronize before training
    # Train ensemble using MPI
    if rank == 0:
        print("Rank 0: Starting training...")
    
    trained_ensemble, loss_history = train_ensemble_mpi(
        ensemble, X, y, grad, num_epochs, track_loss=True
    )

    if rank == 0:
        print("Rank 0: Training completed.\n")
    comm.Barrier()  # Synchronize before aggregation

    # Broadcast the trained ensemble to all ranks
    trained_ensemble = comm.bcast(trained_ensemble, root=0)
    print(f"Rank {rank}: Received trained ensemble broadcast.")

    # Aggregate predictions
    print(f"Rank {rank}: Starting prediction aggregation...")
    predictions = aggregate_forecasts_mpi(trained_ensemble, X, horizon)
    
    if rank == 0:
        print("Rank 0: Aggregation completed.")
        if predictions is not None:
            predictions = jnp.array(predictions)
            print(f"Rank 0: Predictions: {predictions}")

            # Calculate statistics
            mean_prediction = jnp.mean(predictions, axis=0)
            std_dev = jnp.std(predictions, axis=0)
            print(f"Rank 0: Mean prediction: {mean_prediction}")
            print(f"Rank 0: Standard deviation: {std_dev}\n")

            # Create plots
            print("Rank 0: Generating plots...")
            os.makedirs("plots", exist_ok=True)
            plot_prediction_trajectories(predictions, horizon)
            plot_uncertainty(mean_prediction, std_dev, horizon)
            plot_prediction_distribution(predictions)
            plot_prediction_heatmap(predictions)

            # Plot training loss if loss history is available
            if loss_history is not None:
                print(f"Loss history: {loss_history}")
                plot_training_loss(loss_history)

            print("Rank 0: All plots saved in the 'plots/' directory.")
    
    comm.Barrier()  # Synchronize before ending
    end_time = time.time()
    print(f"Rank {rank}: Total execution time: {end_time - start_time:.4f} seconds.")

if __name__ == "__main__":
    main()
