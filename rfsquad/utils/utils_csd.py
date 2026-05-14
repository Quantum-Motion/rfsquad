from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp

if TYPE_CHECKING:
    pass


"""
This module provides utility functions for handling and processing charge stability diagram (CSD) data in quantum dot
simulations. These utilities support operations such as reshaping and masking data, computing probabilities, and loading
CSD objects from file.

Functions:
1. **reshape_and_mask_data**: Reshapes Fock state data to align with raw energy data and applies masking to handle negative Fock states by setting corresponding energies to infinity.
2. **_compute_probabilities**: Computes the probabilities of a quantum dot array being in specific Fock states using Boltzmann statistics.

"""


def reshape_and_mask_data(
    raw_data: jnp.ndarray, fock_states: jnp.ndarray
) -> jnp.ndarray:
    """
    Reshape fock states to align with the raw data and apply masking to get rid of
    energies corresponding to negative Fock states.

    Args:
        raw_data (ndarray): Raw data with shape (sweep1, sweep2, ..., sweepN, config).
        fock_states (ndarray): Fock states with shape (config, num_dots, sweep points).

    Returns:
        ndarray: Masked data with energies set to infinity where fock states negatice.
    """
    # Step 1: Reshape the fock states to align with the raw data
    raw_data_shape = raw_data.shape  # (sweep1, sweep2, ...., sweepN, config)
    fs_reshaped = fock_states.reshape(
        fock_states.shape[0], fock_states.shape[1], *raw_data_shape[:-1]
    )  # shape (config, num_dots, sweep1, ... sweepN)

    # Step 2: Create a mask where the fock states are negative
    negative_mask = fs_reshaped < 0  # shape (config, num_dots, sweep1, ... sweepN)

    # Step 3: Summing over the number of dots in the mask to detect any negatives
    mask_neg = jnp.any(negative_mask, axis=1)  # shape (config, sweep1, ... sweepN)

    # Step 4: Apply mask to the raw data to eliminate energies corresponding to negative fock states
    data = raw_data * (
        1 - jnp.moveaxis(mask_neg, 0, -1)
    )  # shape (config, sweep1, ... sweepN)

    # Step 5: Set energy equal to zero to infinity to correspond to negative fock state positions
    data = jnp.where(data == 0, jnp.inf, data)

    return data


@jax.jit
def _compute_probabilities(
    i: jnp.ndarray,
    raw_data: jnp.ndarray,
    temperature: float,
    kB_eV: float,
    fock_states: jnp.ndarray,
    logZ: jnp.ndarray,
) -> jnp.ndarray:
    """
    Helper function for computing probabilities (p) of the quantum dot array being in a particular fock state.

    Args:
        i (jnp.ndarray): Index of the dot
        raw_data (jnp.ndarray): Raw data
        temperature (float): Temperature in Kelvin
        kB_eV (float): Boltzmann constant in eV
        fock_states (np.ndarray): Fock states
        logZ (np.ndarray): Log of the partition function

    Returns:
        jnp.ndarray: Probabilities
    """
    log_term = -raw_data / (temperature * kB_eV) + jnp.log(fock_states[..., i])
    return jnp.exp(logsumexp(log_term, axis=-1) - logZ)
