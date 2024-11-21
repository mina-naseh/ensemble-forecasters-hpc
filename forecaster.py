import jax
import jax.numpy as jnp

# Forecasting one step ahead
def forecast_1step(X: jnp.array, W: jnp.array, b: jnp.array) -> jnp.array:
    X_flatten = X.flatten()
    y_next = jnp.dot(W, X_flatten) + b
    return y_next

# Forecasting for a given horizon
def forecast(horizon: int, X: jnp.array, W: jnp.array, b: jnp.array) -> jnp.array:
    result = []

    for t in range(horizon):
        X_flatten = X.flatten()
        y_next = forecast_1step(X_flatten, W, b)

        # Update X by shifting and adding the new prediction
        X = jnp.roll(X, shift=-1, axis=0)
        X = X.at[-1].set(y_next)

        result.append(y_next)

    return jnp.array(result)

# Loss function for a single step forecast
def forecast_1step_with_loss(params: tuple, X: jnp.array, y: jnp.array) -> float:
    W, b = params
    y_next = forecast_1step(X, W, b)
    return jnp.sum((y_next - y) ** 2)

# Training loop for optimizing weights and biases
def training_loop(grad: callable, num_epochs: int, W: jnp.array, b: jnp.array, 
                  X: jnp.array, y: jnp.array) -> tuple:
    for i in range(num_epochs):
        delta = grad((W, b), X, y)
        W -= 0.1 * delta[0]
        b -= 0.1 * delta[1]
    return W, b
