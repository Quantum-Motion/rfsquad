from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from rfsquad.noise.interface import INoiseModel

"""
This module implements a simple and efficient Gaussian noise model, `GaussianNoise`, which generates and applies Gaussian
(normal distribution) noise to a system state. The model is designed to simulate the impact of random fluctuations in quantum dot arrays.

Class:
- **GaussianNoise**: Main noise model class for generating and applying Gaussian noise.

Core Concepts:
- **Gaussian Distribution**:
   - Noise is sampled from a normal distribution defined by the mean and standard deviation.
- **Random Noise Application**:
   - Noise is scaled by the amplitude and added to the system state.

"""


class GaussianNoise(INoiseModel):
    def __init__(
        self, amplitude: float, mean: float = 0.0, std: float = 1.0, random_seed=42
    ):
        """
        Create a Gaussian noise model.

        Args:
            amplitude (float): Noise level
            mean (float, optional): Mean of the noise. Defaults to 0.0.
            std (float, optional): Standard deviation of the noise. Defaults to 1.0.
            random_seed (int, optional): Random seed used with numpy random generator. Defaults to 42.
        """
        self.mean = mean
        self.std = std
        self.amplitude = amplitude
        self.random_seed = random_seed

    def apply(self, state: jnp.ndarray, sweep: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Apply Gaussian noise to the state.

        Args:
            state (jnp.ndarray): State to apply noise to
            sweep (dict[str, jnp.ndarray]): Sweep parameters

        Returns:
            jnp.ndarray: Noisy state
        """
        rng = np.random.default_rng(self.random_seed)
        return state + self.amplitude * rng.normal(self.mean, self.std, state.shape)
