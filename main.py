import jax
import jax.numpy as jnp
from forecaster import forecast_1step_with_loss
from ensemble import create_ensemble, train_ensemble, aggregate_forecasts

# Initialize example data and parameters
X = jnp.array([[0.1, 0.4], [0.1, 0.5], [0.1, 0.6]])  # Input example
y = jnp.array([[0.1, 0.7]])  # Expected output
W = jnp.array([[0., 1., 0., 1., 0., 1.], [0., 1., 0, 1., 0., 1.]])  # Neural network weights
b = jnp.array([0.1])  # Bias

num_forecasters = 3
noise_std = 0.1
num_epochs = 20
horizon = 5

grad = jax.grad(forecast_1step_with_loss)

ensemble = create_ensemble(num_forecasters, W, b, noise_std)

trained_ensemble = train_ensemble(ensemble, X, y, grad, num_epochs)

predictions = aggregate_forecasts(trained_ensemble, X, horizon)

print(f"Predictions from ensemble: {predictions}")

predictions = jnp.array(predictions)
mean_prediction = jnp.mean(predictions, axis=0)
std_dev = jnp.std(predictions, axis=0)

print(f"Mean prediction: {mean_prediction}")
print(f"Standard deviation: {std_dev}")
