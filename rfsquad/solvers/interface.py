from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax  # type: ignore
import jax.numpy as jnp  # type: ignore

if TYPE_CHECKING:
    from rfsquad.core import Device

from rfsquad.core import Device
from rfsquad.csd import ChargeStabilityDiagram
from rfsquad.utils.globals import DEVICE_ALLOWED, DEVICE_REQUIRED, SOLVER_ALLOWED
from rfsquad.utils.io import HDF5, Json, PathLike, Toml, _from_file_helper
from rfsquad.utils.utils_ci import (
    _create_fock_full,
    _create_fock_reduced,
    _create_pairwise,
    _create_virtual,
    _create_virtual_gate_matrix,
)

"""
This module implements the `ISolver` abstract base class, which defines the interface for solvers that compute charge
stability diagrams (CSD) in quantum dot simulations. The `ISolver` class provides a framework for configuring and
executing simulations, managing solver parameters, and integrating with quantum device models.

Class:
- **ISolver**: Abstract base class defining the interface for solvers.

"""


@dataclass
class ISolver(ABC):
    """
    Abstract interface class for charge stability diagrams
    """

    device: Device
    use_virtual_gates: bool = False
    reduce_fock_space: bool = False
    max_charge_carriers: int = 1
    max_fock_depth: int = 1

    @abstractmethod
    def run(self, sweep: dict, update_solver: dict = None) -> ChargeStabilityDiagram:  # type: ignore
        pass

    def to_file(self, filename: PathLike):
        """
        Save the Solver object to a file.

        Args:
            filename (PathLike): Path toml or hdf5 file to save the object to.
        """

        match pathlib.Path(filename).suffix:
            case ".toml":
                Toml.dump(filename=filename, objects=[self])
            case ".h5" | ".hdf5":
                HDF5.dump(filename=filename, objects=[self])
            case ".json":
                Json.dump(filename=filename, objects=[self])
            case _:
                raise ValueError(
                    "File extension not supported. Please use either .toml, .json or .h5/.hdf5."
                )

    @classmethod
    def from_file(cls, filename: PathLike):
        """
        Load a Solver object from a file.

        Must contain Device object in the representation. Device must contain C_dd and C_dg.

        There is no validation whether the same Solver class is being loaded - might be added in the future.

        Args:
            filename (PathLike): Path to toml / hdf5 / json file to load the solver from.

        Returns:
            Solver object
        """
        file_obj = _from_file_helper(filename)

        # check device loaded correctly
        file_obj._check_data_format(
            kind="Device",
            required_keys=DEVICE_REQUIRED,
            allowed_keys=DEVICE_ALLOWED,
        )
        # TODO maybe also check if there is a solver name key and warn if class.name doesn't match
        if "name" in file_obj.data["Solver"]:
            _ = file_obj.data["Solver"].pop("name")

        # no required env keys any more - optional validation at instantiation
        file_obj._check_data_format(
            kind="Solver",
            required_keys=[],
            allowed_keys=SOLVER_ALLOWED,
        )
        # create new objects
        dev = Device.from_file(filename)
        inst = cls(dev)

        # add the other params if they exist
        # this function expects a dict so there's no need to unpack
        inst._update_solver(file_obj.data["Solver"])

        return inst

    def __repr__(self):
        s = f"{self.__class__.__name__} using:\n{self.device}\n \nand"
        s += "\nSolver parameters: \n"
        s += f"Max charge carriers: {self.max_charge_carriers}\n"
        s += f"Reduced fock states: {self.reduce_fock_space}\n"
        s += f"Use virtual gates: {self.use_virtual_gates}\n"
        s += (
            f"The max_fock_depth value in this solver ({self.max_fock_depth}) determines the "
            f"truncation level of the Fock space and balances simulation speed and accuracy. "
        )
        return s

    def _validate_solver(self):
        if self.max_charge_carriers is not None and self.max_charge_carriers <= 0:
            raise ValueError("Max charge carriers must be greater than 0")

        if not self.reduce_fock_space and self.max_charge_carriers is None:
            raise ValueError(
                "If reduce_fock_space is False, max_charge_carriers must be specified"
            )

        # Validate self.max_fock_depth
        if not isinstance(self.max_fock_depth, int):
            raise TypeError("max_fock_depth must be an integer")
        if self.max_fock_depth <= 0:
            raise ValueError("max_fock_depth must be greater than 0")

    def _update_solver(self, update_solver: dict):
        """
        Update the solver parameters
        """
        if update_solver is not None:
            for key, value in update_solver.items():
                match key.lower():
                    case "max_charge_carriers":
                        self.max_charge_carriers = value
                    case "reduce_fock_space":
                        self.reduce_fock_space = value
                    case "max_fock_depth":
                        self.max_fock_depth = value
                    case "use_virtual_gates":
                        self.use_virtual_gates = value
                    case _:
                        raise KeyError(f"Invalid solver parameter: {key}")

    def _setup_run(self, sweep: dict[str, jnp.ndarray], update_solver: dict) -> None:  # type: ignore
        """
        Refactored the run setup from the run methods of child classes to avoid code duplication.

        Args:
            sweep (dict[str, jnp.ndarray]): Dictionary of gate voltages being swept in the simulation.
            update_solver (dict): Dictionary of solver parameters to update.

        Returns:
            None

        """
        # 0. set the environment parameters and validate them
        self._update_solver(update_solver)
        self._validate_solver()

        # 1. create mesh for sweep + axes for CSD
        virtual_gate_matrix = _create_virtual_gate_matrix(
            cdd_inv=self.device.C_dd_inv,
            cgd=self.device.C_dg,  # type: ignore
        )

        if self.device.virtual_gate_matrix is not None:
            virtual_gate_matrix = self.device.virtual_gate_matrix

        self.pairwise, self.axes_for_csd = jax.lax.cond(
            self.use_virtual_gates,
            lambda: _create_virtual(
                sweep=sweep,
                threshold_voltages=self.device.threshold_voltages,
                gate_voltages=self.device.gate_voltages,
                virtual_gate_matrix=virtual_gate_matrix,
            ),
            lambda: _create_pairwise(
                sweep=sweep,
                threshold_voltages=self.device.threshold_voltages,
                gate_voltages=self.device.gate_voltages,
            ),
        )

        self.C_dd, self.C_dg, self.C_dd_inv = (
            self.device.C_dd,
            self.device.C_dg,
            self.device.C_dd_inv,
        )

        if self.reduce_fock_space:
            self.fock_states = _create_fock_reduced(
                self.device.C_dd_inv,
                self.device.C_dg,
                self.pairwise,
                self.max_fock_depth,
            )

        else:
            self.fock_states = _create_fock_full(
                self.max_charge_carriers, self.device.num_dots, self.pairwise
            )

        # reshape is sweep, sweep, ..., config
        self.reshape = (
            *jax.tree.map(lambda x: x.shape[0], jax.tree.leaves(self.axes_for_csd)),
            self.fock_states.shape[0],
        )
