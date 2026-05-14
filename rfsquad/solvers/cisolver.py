from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp  # type: ignore

from rfsquad.csd import ChargeStabilityDiagram
from rfsquad.solvers.interface import (
    ISolver,
)
from rfsquad.utils.utils_ci import compute_energies

if TYPE_CHECKING:
    from rfsquad.core import Device

"""
cisolver.py

This module defines the CISolver class, which implements the constant interaction model solver
for the charge stability diagram of quantum dot devices.

Classes:
    CISolver: Constant interaction model solver for the charge stability diagram.
Example:
    from rfsquad.core import Device
    from rfsquad.solvers.cisolver import CISolver

    # Create a Device object
    device = Device(C_dd=jnp.array([[10, -1], [-1, 10]]), C_dg=jnp.array([[1.0, 0.1], [0, 1.0]]))

    # Initialize the CISolver with the device
    solver = CISolver(device=device)

    # Define the gate voltages to sweep over
    sweep = {"gate_1": jnp.linspace(-1, 1, 100), "gate_2": jnp.linspace(-1, 1, 100)}

    # Use the solver to compute the charge stability diagram
    csd = solver.run(sweep=sweep)
"""


## CONSTANT INTERACTION MODEL - NEW AND IMPROVED
class CISolver(ISolver):
    """
    Constant interaction model solver for the charge stability diagram.
    """

    def __init__(
        self,
        device: Device,
        use_virtual_gates: bool = False,
        reduce_fock_space: bool = False,
        max_charge_carriers: int = 1,
        max_fock_depth: int = 1,
    ):
        """
        Args:
            device (Device): Device object containing the device parameters.
            use_virtual_gates (bool, optional): Whether to use virtual gates. Defaults to False.
            reduce_fock_space (bool, optional): Whether to reduce the fock space. Defaults to False.
            max_charge_carriers (int, optional): Maximum number of charge carriers. Defaults to 1.
            max_fock_depth (int, optional): Determines number of fock states to keep when reducing the fock space. Defaults to 1. Number of focks states kept = 2 * max_fock_depth + 1.
        """

        super().__init__(
            device,
            use_virtual_gates,
            reduce_fock_space,
            max_charge_carriers,
            max_fock_depth,
        )

    def run(
        self, sweep: dict[str, jnp.ndarray], update_solver: dict = None
    ) -> ChargeStabilityDiagram:  # type: ignore
        """
        Runs the solver to generate a charge stability diagram.

        Args:
            sweep (dict[str, jnp.ndarray]): Dictionary containing the sweep points, can be any number of gates.
            update_solver (dict, optional): Dictionary containing any updates to the solver parameters
                (reduce_fock states, use_virtual_gates, max_charge_carriers). Defaults to None.

        Returns:
            ChargeStabilityDiagram: Charge stability diagram object.
        """
        # 0. set the environment parameters and validate them
        # 1. create mesh for sweep + axes for CSD
        # 2. create fock states
        self._setup_run(sweep=sweep, update_solver=update_solver)

        # Note that the capacitance matrices have been defined in the setup function. If they were not defined, then the
        # capacitance functions have already set to their defaults in the set up function.

        energies = compute_energies(
            self.pairwise,
            self.fock_states,
            self.C_dd_inv,
            self.C_dg,
        )

        energies = jnp.reshape(energies, self.reshape)

        csd = ChargeStabilityDiagram(
            sweep=self.axes_for_csd,
            raw_data=energies,
            solver="cim",
            virtual_gates=self.use_virtual_gates,
            fock_states=self.fock_states,
        )

        return csd
