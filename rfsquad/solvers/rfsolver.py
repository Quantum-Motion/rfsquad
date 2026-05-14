from __future__ import annotations

from typing import TYPE_CHECKING

from rfsquad.csd import ChargeStabilityDiagram, RFDiagram
from rfsquad.solvers.interface import ISolver
from rfsquad.utils.utils_ci import _create_virtual, _create_virtual_gate_matrix
from rfsquad.utils.utils_rf import _outer_jvp

if TYPE_CHECKING:
    from rfsquad.core import Device

"""
rfsolver.py

This module defines the RFSolver class, which implements the constant interaction model solver
for the charge stability diagram of quantum dot devices and simulates the RF readout of the charge stability map.

Unlike the CISolver, this method also requires the temperature and the dot used
as a sensor dot in the measurement process.

Classes:
    RFSolver: Solver for the charge stability diagram and simulates

Example:
    from rfsquad.core import Device
    from rfsquad.solvers.rfsolver import RFSolver

    # Create a Device object
    device = Device(C_dd=jnp.array([[10, -1], [-1, 10]]), C_dg=jnp.array([[1.0, 0.1], [0, 1.0]]))

    # Initialize the RFSolver with the device
    solver = RFSolver(device=device)

    # Define the gate voltages to sweep over
    sweep = {"gate_1": jnp.linspace(-1, 1, 100), "gate_2": jnp.linspace(-1, 1, 100)}

    # Use the solver to compute the charge stability diagram
    csd = solver.run(sweep=sweep , temperature = 0.5, sensor_dot = "dot_1")
"""


class RFSolver(ISolver):
    """
    Solver for the RF-simulated charge stability diagram.
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
            use_capacitance_functions (bool, optional): Boolean to determine whether to vary capacitance matrices using the provided capacitance functions. Defaults to False.
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

        self._needs_eigen = False

    def run(  # type: ignore[override]
        self,
        sweep: dict,
        sensor_gate: str,
        temperature: float,
        update_solver: dict = None,
    ) -> RFDiagram:  # type: ignore
        """
        Runs the solver to generate a charge stability diagram.

        Args:
            sweep (dict[str, jnp.ndarray]): Dictionary containing the sweep points, can be any number of gates.
            sensor_gate (str): Name of the sensor gate.
            temperature (float): Temperature in Kelvin.
            update_solver (dict, optional): Dictionary containing any updates to the solver parameters
                (reduce_fock states, use_virtual_gates, max_charge_carriers). Defaults to None.

        Returns:
            RFDiagram: RF-simulated charge stability diagram object.
        """
        # 0. set the environment parameters and validate them
        self._setup_run(sweep=sweep, update_solver=update_solver)

        # we just want vgm here because we need it later on in the outer_jvp function
        virtual_gate_matrix = None
        if self.use_virtual_gates:
            if self.device.virtual_gate_matrix is not None:
                virtual_gate_matrix = self.device.virtual_gate_matrix
            else:
                virtual_gate_matrix = _create_virtual_gate_matrix(
                    cdd_inv=self.device.C_dd_inv,
                    cgd=self.device.C_dg,  # type: ignore
                )

            _, self.axes_for_csd = _create_virtual(
                sweep=sweep,
                threshold_voltages=self.device.threshold_voltages,
                gate_voltages=self.device.gate_voltages,
                virtual_gate_matrix=virtual_gate_matrix,
            )

        # 3. actually calculate the rf_measurement
        # this function has another function within it that calculates the pairwise voltages from sensor gate voltages and all others; and then does all of the normal calculations for the energies
        (
            self.rf_measurement,
            dynamic_tc_matrix,
            energies,
            eigenvectors,
            new_fock_states,
        ) = _outer_jvp(
            sweep,
            self.fock_states,
            self.device,
            sensor_gate,
            virtual_gate_matrix,  # type: ignore
            temperature,
            self.reshape,
        )

        # 4. assign the energies to a charge stability diagram and return it
        csd = ChargeStabilityDiagram(
            sweep=self.axes_for_csd,
            raw_data=self.rf_measurement,
            solver="rf",
            virtual_gates=self.use_virtual_gates,
            fock_states=self.fock_states,
        )

        rf = RFDiagram(
            csd,
            temperature=temperature,
            sensor_gate=sensor_gate,
        )
        return rf
