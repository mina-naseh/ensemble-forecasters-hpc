from mpi4py import MPI
import jax
import jax.numpy as jnp
from forecaster import forecast_1step_with_loss
from ensemble import create_ensemble, train_ensemble_mpi, aggregate_forecasts_mpi, export_statistics_to_csv
from config import X, y, W, b, num_forecasters, noise_std, num_epochs, horizon
from plotter import (
    plot_random_forecasters,
    plot_uncertainty,
    plot_random_forecasters_loss,
    plot_quantiles,
)
import os
import time
import numpy as np


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

    comm.Barrier()  # Synchronize before creating the ensemble
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

            mean_prediction = jnp.mean(predictions, axis=0)
            std_dev = jnp.std(predictions, axis=0)
            print(f"Rank 0: Mean prediction: {mean_prediction}")
            print(f"Rank 0: Standard deviation: {std_dev}\n")

            print("Rank 0: Generating plots...")
            os.makedirs("plots", exist_ok=True)
            plot_uncertainty(mean_prediction, std_dev, horizon)
            plot_quantiles(predictions, horizon)

            if loss_history is not None:
                num_random_forecasters = 4
                random_indices = np.random.choice(num_forecasters, num_random_forecasters, replace=False)

                print(f"Randomly selected forecaster indices: {random_indices}")

                plot_random_forecasters(predictions, horizon, num_random_forecasters=len(random_indices))
                plot_random_forecasters_loss(loss_history, random_indices, num_epochs)
                export_statistics_to_csv(
                    predictions=predictions,
                    loss_history=loss_history,
                    horizon=5,
                    stats_folder="stats"
                )


            print("Rank 0: All plots saved in the 'plots/' directory.")
    
    comm.Barrier()  # Synchronize before ending
    end_time = time.time()
    print(f"Rank {rank}: Total execution time: {end_time - start_time:.4f} seconds.")

if __name__ == "__main__":
    main()