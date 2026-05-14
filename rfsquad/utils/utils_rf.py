from __future__ import annotations

from typing import TYPE_CHECKING

import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax.scipy.special import logsumexp  # type: ignore
from scipy.constants import physical_constants  # type: ignore

from rfsquad.utils.utils_ci import _cim_einsum

if TYPE_CHECKING:
    from digitaltwin.core import Device

"""
This module provides functions for the implementation for simulating quantum dot systems for the RFSolver, using eigenvalue calculations to determine energy levels
and the RF calculation to simulate the RF Measurement.

These functions are used for the RFSolver.

List of Functions:
1. **_compute_n**: Computes average occupancies for each dot using RF.
2. **_cim_with_rf**: Calculates CIM probabilities incorporating rate functions.
3. **_outer_jvp**: Computes RF signals with automatic differentiation using JAX.
4. **_inner_jvp**: Inner function for calculating probabilities over sensor gate voltages.
"""


@jax.jit
def _compute_n(i, raw_data, temperature, kB_eV, fock_states, logZ) -> jnp.ndarray:
    """
    Function for computing average occupancies n_i at each dot.

    Args:
        i (jnp.ndarray): Index of the dot
        raw_data (jnp.ndarray): Raw data
        temperature (float): Temperature in Kelvin
        kB_eV (float): Boltzmann constant in eV
        fock_states (jnp.ndarray): Fock states Shape: (config, num dots, sweep points)
        logZ (jnp.ndarray): Log of the partition function

    Returns:
        jnp.ndarray: Average occupancies
    """

    transposed_fock_states = jnp.transpose(
        fock_states, (2, 0, 1)
    )  # Shape: (sweep points, config, num dots)
    log_term = -raw_data / (temperature * kB_eV) + jnp.log(
        transposed_fock_states[..., i]
    )
    # prevent nans for the idiotic gradient by making neg inf most negative number in f32
    log_term = jnp.nan_to_num(log_term)
    return jnp.exp(logsumexp(log_term, axis=-1) - logZ)  # Shape: (sweep points, dots)


def _cim_with_rf(
    pairwise: jnp.ndarray,
    fock_states: jnp.ndarray,
    temperature: float,
    device: Device,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Function calculating the CIM probabilities from RF.

    Args:
        pairwise (jnp.ndarray): Pairwise voltages
        fock_states (jnp.ndarray): Fock states
        c_dd_inv (jnp.ndarray): Inverted capacitance matrix
        c_dg (jnp.ndarray): Dot <-> gate capacitance matrix
        temperature (float): Temperature in Kelvin
        num_dots (int): Number of dots

    Returns:
        jnp.ndarray: Probabilities
        jnp.ndarray: Energies
        jnp.ndarray: fock states

    """

    energies = (_cim_einsum(pairwise.T, fock_states, device.C_dd_inv, device.C_dg),)[0]
    lever_arm_matrix = device.lever_arm_matrix

    kB_eV = physical_constants["Boltzmann constant in eV/K"][0]

    #####################################################################################
    # Now we alter the fock states and any ones with a negative value we set all to zero.
    # This is necessary for the calculation for RF.
    #####################################################################################

    # Step 1: Determine which of the fock states have negative values and create a mask
    negative_check = (fock_states < 0).any(
        axis=1, keepdims=True
    )  # Shape: (config, 1, sweep_points)

    # Step 2: Broadcast this mask along the second axis to match fock_states shape
    # This will set all entries along the second axis to 0 if there's any negative value
    fock_states = jnp.where(negative_check, 0, fock_states)

    # Here we adjust the eigenvalues and set the ones that correspond to negative fock
    # states = jnp.inf. This means that when we take the exponential, they will zero.
    energies = jnp.where(
        negative_check.squeeze(axis=1).T, jnp.inf, energies
    )  # Set energies to inf where Fock state is negative

    logZ = logsumexp(
        -energies / (kB_eV * temperature), axis=-1
    )  # Shape: (sweep points)

    # Now the fock states are adjusted so that all fock states with even one negative value are all zero and their
    # corresponding energies are set to inf. This means that they will not effect the _compute_n function.
    prob = jax.vmap(_compute_n, in_axes=(0, None, None, None, None, None))(
        jnp.arange(device.num_dots), energies, temperature, kB_eV, fock_states, logZ
    )  # Shape: (sweep points, dots)

    # Note that now we are also returning the lever arm matrix - this is needed for the voltage dependant capacitance matrices, and one should note that this is also voltage dependant.
    return prob, lever_arm_matrix, energies, fock_states


def _outer_jvp(
    sweep: dict,
    fock_states: jnp.ndarray,
    device: Device,
    sensor_gate: str,
    virtual_gate_matrix: jnp.ndarray,
    temperature: float,
    reshape: tuple,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Outer function of the jvp process actually containing the inner function that is differentiated; and the differentiation itself.

    Args:
        sweep (dict): Sweep points
        fock_states (jnp.ndarray): Fock states
        device (Device): Device object
        sensor_gate (str): Sensor gate
        virtual_gate_matrix (jnp.ndarray): Virtual gate matrix
        temperature (float): Temperature in Kelvin
        reshape (tuple): Reshape tuple

    Returns:
        jnp.ndarray: RF signal, shape: (sweep points)
        jnp.ndarray: dynamic_tc_matrix, shape: (sweep points, num dots, num dots)
        jnp.ndarray: energies, shape: (sweep points, num dots)
        jnp.ndarray: eigenvectors, shape: (sweep points, num dots, num dots)
        jnp.ndarray: fock_states, shape: (num dots, sweep points)

    """
    sensor_gate_idx = device.gate_names.index(sensor_gate)

    # if sensor gate in sweep; use that
    if sensor_gate in sweep:
        sensor_voltage_array = sweep[sensor_gate]
    else:
        # else use gate_voltage-threshold_voltage
        sensor_voltage_array = jnp.array(
            [device.gate_voltages[sensor_gate] - device.threshold_voltages[sensor_gate]]
        )

    def _inner_jvp(
        sensor_voltage_array: jnp.ndarray,
    ) -> tuple[jnp.ndarray, list]:
        """
        Inner function to defined whatever the jacobian is taken over

        Args:
            sensor_voltage_array (jnp.ndarray): Sensor gate voltage array

        Returns:
            jnp.ndarray: probabilities of occupation for each dot in the array
            tuple: auxiliary terms
        """
        # make list for the gates for meshgrid
        combo_list = []
        for name in device.gate_names:
            if name == sensor_gate:
                combo_list.append(
                    sensor_voltage_array - device.threshold_voltages[name]
                )
            elif name in sweep:
                combo_list.append(sweep[name] - device.threshold_voltages[name])
            else:
                combo_list.append(
                    jnp.array(
                        [device.gate_voltages[name] - device.threshold_voltages[name]]
                    )
                )

        combine = jnp.array(jnp.meshgrid(*combo_list, indexing="ij")).reshape(
            device.num_gates, -1
        )

        if virtual_gate_matrix is not None:
            combine = virtual_gate_matrix @ combine

        # Now cim returns also the lever arm matrices, as these could be voltage dependant
        rf_measurement, lever_arm_matrix, energies, new_fock_states = _cim_with_rf(
            combine.T,
            fock_states,
            temperature,
            device=device,
        )
        dynamic_tc_matrix = None
        eigenvectors = jnp.tile(
            jnp.eye(fock_states.shape[0]), (fock_states.shape[2], 1, 1)
        )
        # Note now that the lever arm matrix is also dependent on the voltages, and we thus have this as an
        # output which we can extract using the has_aux functionality.
        return rf_measurement, [
            lever_arm_matrix,
            dynamic_tc_matrix,
            energies,
            eigenvectors,
            new_fock_states,
        ]

    tangent = jnp.ones(sensor_voltage_array.shape)
    # currently this produces nans for some reason
    # i believe they come from solving the eigenvalues

    # has_aux is now true to return the lever arm matrix. Shape: (sweep points, num_dots, num_gates)
    _, vjp_result, aux = jax.jvp(
        _inner_jvp, (sensor_voltage_array,), (tangent,), has_aux=True
    )

    # Use unpacking to make the code cleaner :)
    lever_arm_matrix, *oth_aux = aux

    reshape = (*jax.tree.map(lambda x: x.shape[0], jax.tree.leaves(sweep)),)

    # Need to change the lever arm matrix to be voltage dependant
    # multiply by lever arm matrix - usual old way
    total = jnp.einsum(
        "i,...i -> ...",
        lever_arm_matrix[:, sensor_gate_idx],
        vjp_result.T,
    )
    return (
        jnp.squeeze(total.reshape(reshape)),
        *oth_aux,
    )
