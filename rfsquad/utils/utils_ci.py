from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import jax  # type: ignore
import jax.numpy as jnp  # type: ignore
from jax import jit, vmap
from jaxopt import BoxCDQP

"""
This module provides functions for the implementation for simulating quantum dot systems for the CISolver, including tools
for calculating energy configurations, and generating Fock states
matrices.

Note that these functions are used in all solvers, including RFSolver.

List of Functions:
1. **_cim_term1**: Computes the first term of the capacitance interaction model (CIM) energy.
2. **_cim_term2**: Computes the second term of the CIM energy.
3. **_cim_term3**: Computes the third term of the CIM energy.
4. **_cim_einsum**: Calculates the total CIM energy using precomputed terms.
5. **compute_energies**: Selects between static and dynamic methods to compute CIM energies.
6. **_create_pairwise**: Generates pairwise gate voltage configurations.
7. **_create_virtual**: Computes virtual gate voltages using a virtual gate matrix.
8. **_create_virtual_gate_matrix**: Constructs a virtual gate matrix from the lever arm matrix.
9. **_create_virtual_helper**: Applies the virtual gate matrix to pairwise voltages.
10. **_create_single_fock_reduced**: Creates reduced Fock states for a single voltage configuration.
11. **_create_fock_reduced**: Generates reduced Fock states for all voltage configurations.
12. **_create_fock_full**: Produces full Fock states for a specified range of charge states.
"""


@jax.jit
def _cim_term1(C_dd_inv: jnp.ndarray, fock_states: jnp.ndarray) -> jnp.ndarray:
    """Generate the first term of the CIM energy calculation.

    Args:
        C_dd_inv (jnp.ndarray): Inverse of the dot <-> dot capacitance matrix, shape [dots, dots]
        fock_states (jnp.ndarray): Fock states describing the charge configuration, shape [states, dots, sweep points]

    Returns:
        jnp.ndarray: Array of shape [sweep_points, states] containing the first term of the CIM energy calculation
    """
    return 0.5 * jnp.einsum(
        "iks, kj, ijs -> si", fock_states, C_dd_inv, fock_states, optimize="optimal"
    )


@jax.jit
def _cim_term2(
    C_dd_inv: jnp.ndarray, C_dg_V: jnp.ndarray, fock_states: jnp.ndarray
) -> jnp.ndarray:
    """Generate the second term of the CIM energy calculation.

    Args:
        C_dd_inv (jnp.ndarray): Inverse of the dot <-> dot capacitance matrix, shape [dots, dots]
        C_dg_V (jnp.ndarray): C_dg @ gate_voltages, shape [dots, sweep points]
        fock_states (jnp.ndarray): Fock states describing the charge configuration, shape [states, dots, sweep points]

    Returns:
        jnp.ndarray: Array of shape [sweep points, states] containing the second term of the CIM energy calculation
    """
    return jnp.einsum(
        "ji,jk,lki->il", C_dg_V, C_dd_inv, fock_states, optimize="optimal"
    )


@jax.jit
def _cim_term3(C_dd_inv: jnp.ndarray, C_dg_V: jnp.ndarray) -> jnp.ndarray:
    """Generate the third term of the CIM energy calculation.

    Args:
        C_dd_inv (jnp.ndarray): Inverse of the dot <-> dot capacitance matrix, shape [dots, dots]
        C_dg_V (jnp.ndarray): C_dg @ gate_voltages, shape [dots, sweep points]

    Returns:
        jnp.ndarray: Array of shape [sweep points] containing the third term of the CIM energy calculation
    """
    return 0.5 * jnp.einsum("ji,jk,ki->i", C_dg_V, C_dd_inv, C_dg_V, optimize="optimal")


@jax.jit
def _cim_einsum(
    pairwise: jnp.ndarray,
    fock_states: jnp.ndarray,
    C_dd_inv: jnp.ndarray,
    C_dg: jnp.ndarray,
) -> jnp.ndarray:
    """Calculate the energy of the CIM model using einsum and functions for the three terms.

    Args:
        pairwise (jnp.ndarray): Voltage pairs, expected in the shape [gates, sweep points]
        fock_states (jnp.ndarray): Fock states, expected in the shape [states, dots, sweep points]
        C_dd_inv (jnp.ndarray): Inverted capacitance matrix, expected in the shape [dots, dots]
        C_dg (jnp.ndarray): Dot <-> gate capacitance matrix, expected in the shape [dots, gates]

    Returns:
        jnp.ndarray: Array of shape [sweep points, states] containing the energy each state at each point
    """
    C_dg_V = C_dg @ pairwise
    term1 = _cim_term1(C_dd_inv, fock_states)  # Shape: (sweep points, config )
    term2 = _cim_term2(C_dd_inv, C_dg_V, fock_states)  # Shape: (sweep points, config )
    term3 = _cim_term3(C_dd_inv, C_dg_V)[:, None]  # Shape: (sweep points, 1)
    return term1 - term2 + term3


def compute_energies(pairwise, fock_states, C_dd_inv, C_dg):
    """
    Computes energies using the standard method.

    Args:
        pairwise (jnp.ndarray):
            Array representing pairwise input data, typically used in the computation.
        fock_states (jnp.ndarray):
            Array of Fock states involved in the computation.
        C_dd_inv (jnp.ndarray):
            Inverted  dot dot capacitance matrix, used for energy calculations.
        C_dg (jnp.ndarray):
            Dot gate capacitance matrix defining the coupling between gates and dots.

    Returns:
        jnp.ndarray:
            The computed energies based on the specified method and input parameters.
    """

    return (_cim_einsum(pairwise.T, fock_states, C_dd_inv, C_dg),)[0]


# GATE VOLTAGES JIT FUNCTIONS
@jit
def _create_pairwise(
    sweep: dict[str, jnp.ndarray],
    threshold_voltages: dict[str, float],
    gate_voltages: dict[str, float],
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Helper function that creates pairwise (or more than pairwise) gate voltages

    Args:
        sweep (dict): Voltage arrays of gates being swept in the simulation.
        threshold_voltages (dict[str, float]):  Dictionary of threshold voltages for each gate in the device.
        gate_voltages (dict[str, float]): Dictionary of static / initial gate voltages for each gate in the device.

    Returns:
        tuple[jnp.ndarray, dict[str, jnp.ndarray]]: tuple of pairwise gate voltage array and the dictionary containing the same array but transposed
    """
    # SWEEP VOLTAGES
    # subtract threshold voltage from sweep, but also only keep the keys that are in the sweep
    tv = {key: threshold_voltages[key] for key in sweep if key in threshold_voltages}
    subtracted_sweep = jax.tree.map(lambda x, y: x - y, sweep, tv)
    # meshes the voltages
    # then zips it with the keys to create a dict then maps that to flatten
    # so if 100 points in 2d sweep shape in meshgrid is a list of 2 100x100 arrays
    # then zipped with the keys and flattened it gets into a dict of 2 10000 arrays

    sweep_dict = jax.tree.map(
        lambda v: v.flatten(),
        dict(
            zip(
                subtracted_sweep.keys(),
                jnp.meshgrid(*subtracted_sweep.values(), indexing="ij"),
            )
        ),
    )

    # STATIC VOLTAGES
    # calculates the stupid difference between the gate voltages and the threshold voltages
    # and makes a dict of that where values have the same shape as the sweep ones
    static_dict = jax.tree.map(
        lambda x, y: jnp.full_like(jax.tree.leaves(sweep_dict)[0], x - y),
        gate_voltages,
        threshold_voltages,
    )

    # can easily merge those two because static has keys in the correct order, values are same shape
    # and using c = static | swept will make a union of all of static ones + change the swept ones (hopefully)
    sorted_dict = static_dict | sweep_dict

    # stack them to make them of a (total_points, num_gates) shape
    pairwise = jnp.stack(jax.tree.leaves(sorted_dict), axis=-1)

    return pairwise, sweep


@jit
def _create_virtual(
    sweep: dict[str, jnp.ndarray],
    threshold_voltages: dict[str, float],
    gate_voltages: dict[str, float],
    virtual_gate_matrix: jnp.ndarray,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """
    Helper creating pairwise gate voltages

    Args:
        sweep (dict): Voltage arrays of gates being swept in the simulation.
        threshold_voltages (dict[str, float]):  Dictionary of threshold voltages for each gate in the device.
        gate_voltages (dict[str, float]): Dictionary of static / initial gate voltages for each gate in the device.
        virtual_gate_matrix (jnp.ndarray):

    Returns:
        tuple[jnp.ndarray, dict[str, jnp.ndarray]]: tuple of pairwise gate voltage array and the dictionary containing the same array but transposed
    """
    pairwise, _ = _create_pairwise(
        sweep=sweep,
        threshold_voltages=threshold_voltages,
        gate_voltages=gate_voltages,
    )

    # lax cond whether virtual gate matrix is none
    # this is evaluated before
    # need to transpose pairwise back to the original shape
    new_pairwise_untransposed = virtual_gate_matrix @ pairwise.T

    # create axes with ALL pairwise gate voltages, then only keep the ones in the sweep
    all_axes = dict(zip(gate_voltages.keys(), new_pairwise_untransposed))

    # dictionary of swept keys so it maps itself correctly - need
    # to iterate over keys but also need them as inputs to lambda
    # (which are the values/leafs from the pytree)
    dict_of_keys = dict(zip(sweep.keys(), sweep.keys()))

    # the input here could be a list but then it would make a list
    # and you'd need to combine it back into a dictionary and
    # that's effort
    axes_for_csd = jax.tree.map(
        lambda x: jnp.linspace(
            jnp.min(all_axes[x]), jnp.max(all_axes[x]), sweep[x].shape[0]
        ),
        dict_of_keys,
    )

    return new_pairwise_untransposed.T, axes_for_csd


def _create_virtual_gate_matrix(
    cdd_inv: jnp.ndarray, cgd: jnp.ndarray, rtol: float = 1e-15
) -> jnp.ndarray:
    """
    Computes the optimal virtual gate matrix with row-wise normalization.

    Args:
        cdd_inv (jnp.ndarray): The inverse of the dot-to-dot capacitance matrix.
        cgd (jnp.ndarray): The dot-to-gate capacitance matrix.
        rtol (float, optional): The relative tolerance parameter for the pseudo-inverse.
            Defaults to 1e-15.

    Returns:
        jnp.ndarray: The optimal virtual gate matrix with rows normalized by their maximum
        absolute values.
    """
    n_dot = cdd_inv.shape[0]
    n_gate = cgd.shape[1]
    virtual_gate_matrix = jnp.linalg.pinv(cdd_inv @ cgd, rtol=rtol)

    # Normalize each row by its maximum absolute value
    row_max_values = jnp.max(jnp.abs(virtual_gate_matrix), axis=0, keepdims=True)
    vgm = virtual_gate_matrix / row_max_values
    if n_dot < n_gate:
        vgm = jnp.pad(vgm, ((0, 0), (0, n_gate - n_dot)), mode="constant")

    return vgm


def _create_virtual_helper(
    pairwise: jnp.ndarray, virtual_gate_matrix: jnp.ndarray
) -> jnp.ndarray:
    """
    Helper for virtualisation when lever arm matrix is used in lieu of virtual gate matrix.
    Only applies the virtual gate matrix to the first num_dots gates (which should match the number and position of plunger gates).

    Args:
        pairwise (jnp.ndarray): Pairwise gate voltages, should be of shape (n_points, n_gates) because it's transposed in the code here again to (n_gates, n_points)
        virtual_gate_matrix (jnp.ndarray): Virtual gate matrix

    Returns:
        jnp.ndarray: Pairwise gate voltages in the (n_gates, n_points) shape
    """
    # only the first n columns of the lever arm matrix are used
    # because we only consider plunger gates, which should be the first n gates

    num_plunger_gates = virtual_gate_matrix.shape[0]
    gate_voltages = virtual_gate_matrix @ pairwise.T[:num_plunger_gates]

    # add the other gates back in
    gate_voltages = jnp.vstack(
        (
            gate_voltages,
            pairwise.T[num_plunger_gates:],
        )
    )

    return gate_voltages


# FOCK SPACE JIT FUNCTIONS
### these should be evaluated at simulation runtime because it changes based on the voltages provided if using the reduced fock space


@partial(jit, static_argnums=(3))
def _create_single_fock_reduced(
    C_dd_inv: jnp.ndarray,
    C_dg: jnp.ndarray,
    voltage_config: jnp.ndarray,
    max_fock_depth: int,
) -> jnp.ndarray:
    """
    Computes the reduced Fock states for a given voltage configuration in a quantum dot system. This function performs
    a constrained dual quadratic programming (CDQP) optimization using `jaxopt.BoxCDQP` to find integer states
    corresponding to a voltage configuration. The integer states are expanded by a range defined by `k`, resulting
    in a set of Fock states.

    The main goal is to solve for integer electron occupation states within a specified range and subject to given
    constraints on a quantum dot system.

    The optimization problem is defined as:

        minimize     0.5 * x^T * C_dd_inv * x + c^T * x
        subject to   l <= x <= u

    where:
        - `C_dd_inv` is the inverse capacitance matrix (dot-dot coupling).
        - `C_dg` is the capacitance matrix (dot-gate coupling).
        - `voltage_config` represents the external voltage configuration vector.

    Using the solution, the function expands around the optimized solution to generate all possible states within a
    range of ±max_fock_depth for each electron occupation state.

    Args:
        C_dd_inv (jnp.ndarray): Inverse of the dot-dot capacitance matrix, of shape `(num_dots, num_dots)`.
        C_dg (jnp.ndarray): Dot-gate capacitance matrix, of shape `(num_dots, num_gates)`.
        voltage_config (jnp.ndarray): Voltage configuration for a single voltage sweep, of shape `(num_gates,)`.

    Returns:
        jnp.ndarray: Array of reduced Fock states, containing all integer configurations within the range, of shape
                     `(2*max_fock_depth + 1)**num_dots`.

    Mathematical Details:
        The function performs the following steps:

        1. Define Constraints:
            - `l = 0`: Lower bounds for electron occupation states.
            - `u = ∞`: Upper bounds for electron occupation states.

        2. Compute Linear Term for Optimization:
            - `c = -C_dd_inv @ C_dg @ voltage_config`

        3. Run CDQP Optimization: Solves for the continuous `x` vector that minimizes the quadratic objective,
           providing a preliminary solution in floating-point values.

        4. Generate Integer Range: Using the solution `sol`, we expand it to an integer grid of possible Fock
           states with range ±k:
           - `sol_floor = floor(sol) - k`
           - `sol_ceil = ceil(sol) + k`

        5. Create All Combinations: Generate all integer Fock states in the range `[sol_floor, sol_ceil]` for
           each dimension. The resulting Fock states are all possible combinations of these integer values.

    """
    # Note: I am changing the shape here to num_dots=C_dd_inv.shape[0] instead of voltage_config.shape -> this must be taking into account when numdots !=numgates
    num_dots = C_dd_inv.shape[0]

    # Note: I am changing the shape here to num_dots=C_dd_inv.shape[0] instead of voltage_config.shape -> this must be taking into account when numdots !=numgates
    num_dots = C_dd_inv.shape[0]

    l = jnp.zeros(num_dots)  # noqa: E741
    u = jnp.inf * jnp.ones(num_dots)

    c = -C_dd_inv @ C_dg @ voltage_config

    qp = BoxCDQP()
    init = jnp.zeros(num_dots)
    init = jnp.zeros(num_dots)
    sol = qp.run(init, params_obj=(C_dd_inv, c), params_ineq=(l, u)).params

    sol_floor = jnp.floor(sol).astype(int) - max_fock_depth

    N = sol.shape[0]

    masks = jnp.array(jax.numpy.indices((2 * max_fock_depth + 1,) * N).reshape(N, -1).T)
    combinations = vmap(lambda mask: sol_floor + mask)(masks)

    return combinations


@partial(jit, static_argnums=(3))
def _create_fock_reduced(
    C_dd_inv: jnp.ndarray, C_dg: jnp.ndarray, pairwise: jnp.ndarray, max_fock_depth: int
) -> jnp.ndarray:
    """Create reduced fock states for entire voltage sweeps

    Args:
        C_dd_inv (jnp.ndarray): Inverse of dot dot capacitance matrix
        C_dg (jnp.ndarray): dot gate capacitance matrix
        pairwise (jnp.ndarray): Array of voltages swept. Size (sweep points, num_gates)
        max_fock_depth (int): Range of fock states to consider

    Returns:
        jnp.ndarray: reduced fock states for all voltage configurations
    """

    # NOTE: vmap is used to vectorize the computation over the first axis of the pairwise array.
    # This is required as jaxopt function can only return scalar, not vector.
    vmap_fock = vmap(
        _create_single_fock_reduced, in_axes=(None, None, 0, None), out_axes=-1
    )
    fock_results = vmap_fock(C_dd_inv, C_dg, pairwise, max_fock_depth)
    return fock_results


@partial(jit, static_argnums=(0, 1))
def _create_fock_full(
    max_charge_states: int, num_dots: int, pairwise: jnp.ndarray
) -> jnp.ndarray:
    """
    Create a list of available Fock states for M quantum dots with a maximum occupancy of N (electrons) for each dot.

    Args:
        max_charge_states (int): Maximum number of charge states.
        num_dots (int): Number of quantum dots.
        pairwise (jnp.ndarray): Array of voltages swept. Size (sweep points, num_gates)

    Returns:
        jnp.ndarray: Fock states of shape ((max_charge_states + 1)**num_dots , num_dots, sweep points).
    """
    args = jnp.zeros((num_dots, max_charge_states + 1)) + jnp.arange(
        0, max_charge_states + 1
    )
    number_of_configurations = (max_charge_states + 1) ** num_dots
    fock_states = jnp.stack(jnp.meshgrid(*args, indexing="ij"), axis=-1).reshape(
        number_of_configurations, num_dots
    )
    # Here we broadcast to add an extra dimension, which will represent the voltage dependance.
    # In this case, the fock states are the same for each voltage configuration, so we can just repeat the calculated full fock states.
    fock_states_broadcast = jnp.broadcast_to(
        fock_states[..., None], (number_of_configurations, num_dots, pairwise.shape[0])
    )
    return fock_states_broadcast
