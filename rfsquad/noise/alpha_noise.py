from __future__ import annotations

import jax.numpy as jnp  # type: ignore
import numpy as np

from rfsquad.noise.interface import INoiseModel

"""
This module implements a noise model class, `AlphaNoise`, which generates and applies 1/f^α noise to a system state,
simulating the effects of noise in quantum dot or other systems. This customizable noise model supports various types
of noise, including pink noise (1/f) and brownian noise (1/f^2).
Class:
- **AlphaNoise**: Main noise model class implementing 1/f^α noise generation and application.

Core Concepts:
- **Fourier Transform**:
   - Noise is generated in the frequency domain and converted to the time domain for application.
- **Frequency Scaling**:
   - The noise spectrum is scaled by 1/f^α, where α determines the type of noise.

"""


class AlphaNoise(INoiseModel):
    def __init__(
        self,
        amplitude: float,
        time: float,
        alpha: float = 1,
        fmin: float = 0,
        random_seed: int = 42,
        sweep_gate_order: list[str] = None,  # type: ignore
    ) -> None:
        """
        Create an 1/f noise model with an optional alpha argument for the exponent creating 1/f**alpha noise.

            Use alpha = 1 for pink (1/f) noise.
            Use alpha = 2 for brownian (1/f**2) noise.

        This function is borrowed from the colorednoise package, https://github.com/felixpatzelt/colorednoise, which is based on the algorithm in:
            Timmer, J. and Koenig, M.: On generating power law noise. Astron. Astrophys. 300, 707-710 (1995)

        Args:
            amplitude (float): Noise level
            time (float): Time duration of the simulation in seconds.
            alpha (float): Exponent for 1/f noise.
            fmin (float): Low frequency cutoff. Must in the [0, 0.5] range. Defaults to 0.
            random_seed (int): Random seed used with numpy random generator. Defaults to 42.
            sweep_gate_order (list[str]): Order in which the gates were swept (slow to fast). Defaults to None, which uses the order from the Device.

        """
        self.amplitude = amplitude
        self.time = time
        self.random_seed = random_seed
        self.alpha = alpha
        self.fmin = fmin
        self.sweep_gate_order = sweep_gate_order

    def apply(self, state: jnp.ndarray, sweep: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Apply Alpha (1/f**alpha) noise to the state.

        Args:
            state (jnp.ndarray): State to apply noise to
            sweep (dict[str, jnp.ndarray]): Sweep parameters

        Returns:
            jnp.ndarray: Noisy state
        """

        # this function is copied from the colorednoise package
        # https://github.com/felixpatzelt/colorednoise
        # copied so that it's made jax compatible and to avoid the dependency on colorednoise

        # TODO check that sweep and state have compatible shapes

        size = [state.reshape(-1).shape[0]]

        # The number of samples in each time series
        samples = size[-1]

        # Calculate Frequencies (we assume a sample rate of one)
        # Use fft functions for real output (-> hermitian spectrum)
        sample_spacing = self.time / samples
        f = jnp.fft.rfftfreq(samples, sample_spacing)

        # Validate / normalise fmin
        if 0 <= self.fmin <= 0.5:
            fmin = max(self.fmin, 1.0 / samples)  # Low frequency cutoff
        else:
            raise ValueError("fmin must be chosen between 0 and 0.5.")

        # Build scaling factors for all frequencies
        # kept it np so it's easier to index and add to
        s_scale = np.array(f)
        ix = jnp.sum(s_scale < fmin)  # Index of the cutoff
        if ix and ix < len(s_scale):
            s_scale[:ix] = s_scale[ix]
        s_scale = s_scale ** (-self.alpha / 2.0)

        # Calculate theoretical output standard deviation from scaling
        w = s_scale[1:].copy()
        w[-1] *= (1 + (samples % 2)) / 2.0  # correct f = +-0.5
        sigma = 2 * jnp.sqrt(jnp.sum(w**2)) / samples

        # Adjust size to generate one Fourier component per frequency
        size[-1] = len(f)

        # Add empty dimension(s) to broadcast s_scale along last
        # dimension of generated random power + phase (below)
        dims_to_add = len(size) - 1
        s_scale = s_scale[(None,) * dims_to_add + (...,)]

        # prepare random number generator
        rng = np.random.default_rng(self.random_seed)

        # Generate scaled random power + phase
        sr = rng.normal(scale=s_scale, size=size)
        si = rng.normal(scale=s_scale, size=size)

        # If the signal length is even, frequencies +/- 0.5 are equal
        # so the coefficient must be real.
        if not (samples % 2):
            si[..., -1] = 0
            sr[..., -1] *= jnp.sqrt(2)  # Fix magnitude

        # Regardless of signal length, the DC component must be real
        si[..., 0] = 0
        sr[..., 0] *= jnp.sqrt(2)  # Fix magnitude

        # Combine power + corrected phase to Fourier components
        s = sr + 1j * si

        # Transform to real time series & scale to unit variance
        y = jnp.fft.irfft(s, n=samples, axis=-1) / sigma

        y = y.reshape(state.shape)
        y = y.transpose(self._get_order(sweep))

        return state + self.amplitude * y

    def _get_order(self, sweep: dict[str, jnp.ndarray]) -> tuple[int]:
        """
        Helper to get the order in which the gates were swept

        Args:
            sweep (dict[str, jnp.ndarray]): Sweep parameters

        Returns:
            tuple[int]: Order in which the gates were swept
        """
        if self.sweep_gate_order is None:
            return tuple(range(len(sweep.keys())))

        else:
            return tuple(
                [list(sweep.keys()).index(gate) for gate in self.sweep_gate_order]
            )  # type: ignore
