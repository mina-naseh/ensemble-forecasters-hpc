import jax.numpy as jnp

# parameters for runs
num_forecasters = 4
noise_std = 0.01
num_epochs = 2
horizon = 5

# Dataset
X = jnp.array([[0.1, 0.4], [0.1, 0.5], [0.1, 0.6]])
y = jnp.array([[0.1, 0.7]])

# Initial weights (W) and biases (b)
W = jnp.array([[0., 1., 0., 1., 0., 1.], [0., 1., 0., 1., 0., 1.]])
b = jnp.array([0.1])
