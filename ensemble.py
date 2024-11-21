import jax
import jax.numpy as jnp
from forecaster import forecast, training_loop

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

# Train each forecaster in the ensemble
def train_ensemble(ensemble, X, y, grad, num_epochs):
    trained_ensemble = []
    for W, b in ensemble:
        W_trained, b_trained = training_loop(grad, num_epochs, W, b, X, y)
        trained_ensemble.append((W_trained, b_trained))
    return trained_ensemble

# Aggregate predictions from the ensemble
def aggregate_forecasts(ensemble, X, horizon):
    predictions = []
    for W, b in ensemble:
        y_predicted = forecast(horizon, X, W, b)
        predictions.append(y_predicted)
    return predictions
