from __future__ import annotations

import pathlib
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from tabulate import tabulate  # type: ignore

import rfsquad.utils.validation as dtv
from rfsquad.utils.globals import DEVICE_ALLOWED, DEVICE_REQUIRED
from rfsquad.utils.io import HDF5, Json, PathLike, Toml, _from_file_helper

"""
core.py

This module defines the core component of rf-squad for quantum dot devices.
It includes the Device class, which represents the quantum dot device system, and provides
methods for loading and saving device configurations from/to TOML and HDF5 files.

Classes:
    Device: Represents the quantum dot device system with methods for loading and saving configurations.
"""


def _make_2d_array(val: list | int | float | jnp.ndarray) -> jnp.ndarray:
    """
    Little function to convert input to 2D array. It takes a list, list of lists, individual numbers, or
    1D JAX array and converts it to a 2D JAX array.

    Args:
        val (list | int | float | jnp.ndarray): 1D value to convert to 2D array

    Returns:
        jnp.ndarray: 2D array
    """
    if isinstance(val, (int, float)):
        return jnp.array(val).reshape(1, 1)

    elif isinstance(val, list):
        # list of numbers
        if isinstance(val[0], (int, float)):
            return jnp.array(val).reshape(1, len(val))
        # list of lists, assuming with numbers
        elif isinstance(val[0], list):
            return jnp.array(val)

        else:
            raise ValueError(
                "Invalid input. Please provide a list of lists or a single number."
            )
    elif isinstance(val, (jnp.ndarray, np.ndarray)):
        if val.ndim == 1:
            return val.reshape(1, len(val))
        else:
            return val

    else:
        raise ValueError(
            "Invalid input. Please provide a list of lists or a single number."
        )


@dataclass
class Device:
    """
    Class defining the quantum dot device system

    Args:
        C_dd (jnp.ndarray | int | float): Dot-dot capacitance matrix in attofarads (aF).
        C_dg (jnp.ndarray | list): Dot-gate capacitance matrix in attofarads (aF).
        dot_names (list, optional): List of dot names. Defaults to None, which creates a list starting from "dot_1" to "dot_M".
        gate_names (list, optional): List of gate names. Defaults to None, which creates a list starting from "gate_1" to "gate_G" where G is number of gates defined in C_dg.
        virtual_gate_matrix (jnp.ndarray, optional): Matrix used to calculate virtual gate voltages. Defaults to None.
    """

    # Capacitance matrices between dot-dot and gate-dot
    C_dd: jnp.ndarray | int | float
    C_dg: jnp.ndarray | list

    # gate and dot names
    gate_names: list[str] = field(default=None)  # type: ignore
    dot_names: list[str] = field(default=None)  # type: ignore

    # virtual gate matrix
    virtual_gate_matrix: jnp.ndarray | None = field(default=None)

    def __post_init__(self):
        # Convert capacitance matrices to 2D JAX arrays
        self.C_dd = _make_2d_array(self.C_dd) * 1e-18 / 1.6e-19
        self.C_dg = _make_2d_array(self.C_dg) * 1e-18 / 1.6e-19

        self.num_dots = self.C_dd.shape[0]
        self.num_gates = self.C_dg.shape[1]

        if self.dot_names is None:
            self.dot_names = [f"dot_{i}" for i in range(1, self.num_dots + 1)]

        if self.gate_names is None:
            self.gate_names = [f"gate_{i}" for i in range(1, self.C_dg.shape[1] + 1)]

        # assign default plunger and barrier gates
        self.plunger_gates = self.gate_names[: self.num_dots]
        self.barrier_gates = self.gate_names[self.num_dots :]

        # Using OrderedDict to maintain the order of gate names consistently across all operations.
        # This ensures that the key order in `gate_voltages` and `threshold_voltages` matches
        # the order in `self.gate_names`, which is critical for downstream operations that
        # rely on a consistent ordering of keys (e.g., when merging or stacking arrays).
        # Regular dictionaries in Python 3.7+ maintain insertion order, but JAX operations,
        # especially under JIT compilation, can disrupt this order. Using OrderedDict makes the
        # ordering explicit and robust to such issues.
        self.gate_voltages = OrderedDict((gate, 0.0) for gate in self.gate_names)
        self.threshold_voltages = OrderedDict((gate, 0.0) for gate in self.gate_names)

        self.__validate_init__()

        self.C_dd_inv = jnp.linalg.inv(self.C_dd)

        self.lever_arm_matrix = _calculate_lever_arm_matrix(
            C_dd_inv=self.C_dd_inv, C_dg=self.C_dg
        )

    def add_static_voltages(
        self, gate: dict[str, float] = None, threshold: dict[str, float] = None
    ) -> None:  # type: ignore
        """
        Adds the static gate voltages and threshold voltages to the Device. If either not provided, it sets all gate voltages to 0.

        Args:
            gate (dict[str, float], optional): Dictionary of static / initial gate voltages for each gate in the device. Keys are gate names which will be mapped to gate_names, values are voltages in Volts.
            threshold (dict[str, float], optional): Dictionary of threshold voltages for each gate in the device. Keys are gate names which will be mapped to gate_names, values are voltages in Volts.
        """
        if gate is None:
            gate = {}
        if threshold is None:
            threshold = {}

        dtv.validate_threshold_gate_definitions(self.gate_names, gate, threshold)

        # Using OrderedDict to maintain the order of gate names consistently across all operations.
        # This ensures that the key order in `gate_voltages` and `threshold_voltages` matches
        # the order in `self.gate_names`, which is critical for downstream operations that
        # rely on a consistent ordering of keys (e.g., when merging or stacking arrays).
        # Regular dictionaries in Python 3.7+ maintain insertion order, but JAX operations,
        # especially under JIT compilation, can disrupt this order. Using OrderedDict makes the
        # ordering explicit and robust to such issues.

        self.gate_voltages = OrderedDict(
            {gt: gate.get(gt, 0.0) for gt in self.gate_names}
        )
        self.threshold_voltages = OrderedDict(
            {gt: threshold.get(gt, 0.0) for gt in self.gate_names}
        )

    def assign_gates(self, plunger: list, barrier: list | None = None) -> None:
        """
        Assigns the gates to the Device. By default it assigns the first N gates to the plunger gates and the rest to the barrier gates -- use this function to override this behaviour.

        Args:
            plunger (list): List of plunger gates.
            barrier (list, optional): List of barrier gates. Defaults to None.
        """
        dtv.validate_barrier_plunger_gates(self.gate_names, plunger, barrier)
        self.plunger_gates = plunger
        self.barrier_gates = barrier

    @classmethod
    def from_file(cls, filename: PathLike, verbose: bool = False) -> Device:
        """
        Loads the Device from a toml or a h5/hdf5 file. Requires the file to be formatted correctly.

        Toml file requires a [Device] table with both required Device parameters (C_dd, C_dg). The file can have additional object (i.e. Device, Solver, etc.) tables.
        HDF5 file requires a 'Device' group with both required Device parameters (C_dd, C_dg). The string inputs are automatically decoded from binary. The file can have additional object (i.e. Environment, Solver, etc.) keys.

        Usage:
            device = Device.from_file(filename)

        Args:
            filename (path-like): Path to device config toml/hdf5 file.
            verbose (bool, optional): Whether to print a summary of the loaded device. Defaults to False.

        Returns:
            Device object
        """

        file_obj = _from_file_helper(filename)

        # check all required keys are present
        file_obj._check_data_format(
            kind="Device",
            required_keys=DEVICE_REQUIRED,
            allowed_keys=DEVICE_ALLOWED,
        )

        # get plunger and barrier gates if present
        plunger = file_obj.data["Device"].pop("plunger_gates", None)
        barrier = file_obj.data["Device"].pop("barrier_gates", None)

        # get static voltages
        gv = file_obj.data["Device"].pop("gate_voltages", None)
        tv = file_obj.data["Device"].pop("threshold_voltages", None)

        # create the instance of the class
        # this no longer has plunger and barrier gates and
        # static voltages in the init
        # so had to remove them from the dict beforehand
        inst = cls(**file_obj.data["Device"])

        # assign the gates
        if plunger is not None:
            inst.assign_gates(plunger, barrier)

        # add static voltages
        inst.add_static_voltages(gv, tv)

        if verbose:
            print(inst.__repr__())

        return inst

    def to_file(self, filename: PathLike) -> None:
        """
        Convenience function for writing the Device to a toml or hdf5 file for easy future imports.

        Args:
            filename (PathLike): Path to the toml/hdf5 file.
        """
        match pathlib.Path(filename).suffix:
            case ".toml":
                Toml.dump(filename, [self])
            case ".h5" | ".hdf5":
                HDF5.dump(filename, [self])
            case ".json":
                Json.dump(filename, [self])
            case _:
                raise ValueError(
                    "File extension not supported. Please use either .toml, .json or .h5/.hdf5."
                )

    def __validate_init__(self) -> None:
        """
        Checks the input parameters by calling external validation functions.
        """
        dtv.validate_capacitance_matrix_shapes(
            self.C_dd, self.C_dg, self.num_dots, self.num_gates
        )
        dtv.validate_capacitance_matrix_elements(self.C_dd)
        dtv.validate_gate_definitions(
            self.gate_names,
            self.num_gates,
        )
        dtv.validate_virtual_gate_matrix(self.virtual_gate_matrix, self.num_gates)

    def __eq__(self, other: Any) -> bool:
        """
        Check equality between two Device instances.

        Args:
            other (Any): The object to compare against.

        Returns:
            bool: True if the objects are identical, False otherwise.
        """
        if not isinstance(other, Device):
            return False

        # Convert dataclass attributes to dictionaries for streamlined comparison
        self_dict = asdict(self)
        other_dict = asdict(other)

        for key, self_value in self_dict.items():
            other_value = other_dict[key]

            if isinstance(self_value, jnp.ndarray):
                # Compare JAX arrays
                if not jnp.array_equal(self_value, other_value):
                    return False
            else:
                # Compare primitive types
                if self_value != other_value:
                    return False

        return True

    def __repr__(self) -> str:
        C_dd_str = "Mutual (dot-dot) capacitance matrix C_dd:\n"

        C_dd_table = self.C_dd.tolist()  # type: ignore
        for i, t in enumerate(C_dd_table):
            t.insert(0, self.dot_names[i])

        C_dd_str += tabulate(
            C_dd_table, headers=self.dot_names, tablefmt="github", floatfmt=".3f"
        )

        C_dg_str = "Cross (dot-gate) capacitance matrix C_dg:\n"

        C_dg_table = self.C_dg.tolist()  # type: ignore
        for i, t in enumerate(C_dg_table):
            t.insert(0, self.dot_names[i])

        C_dg_str += tabulate(
            C_dg_table, headers=self.gate_names, tablefmt="github", floatfmt=".3f"
        )

        representation = (
            "Device instantiated with: \n"
            + "\n"
            + f"Dots: {', '.join(self.dot_names)}\n"
            + f"Gates: {', '.join(self.gate_names)}\n"
            + f"Plunger gates: {', '.join(self.plunger_gates)}\n"
        )

        if self.barrier_gates is not None and len(self.barrier_gates) > 0:
            representation += f"Barrier gates: {', '.join(self.barrier_gates)}\n"

        else:
            representation += "Barrier gates: None\n"

        part2 = C_dd_str + "\n" + "\n" + C_dg_str

        representation += part2

        if self.virtual_gate_matrix is not None:
            vgm_str = "Virtual gate matrix:\n"

            vgm_table = self.virtual_gate_matrix.tolist()
            for i, t in enumerate(vgm_table):
                t.insert(0, self.gate_names[i])

            vgm_str += tabulate(
                vgm_table, headers=self.gate_names, tablefmt="github", floatfmt=".3f"
            )
            representation += "\n \n"
            representation += vgm_str

        return representation


# AUXILIARY FUNCTIONS


@jax.jit
def _calculate_lever_arm_matrix(
    C_dd_inv: jnp.ndarray, C_dg: jnp.ndarray
) -> jnp.ndarray:
    """
    Calculate the lever arm matrix for the device

    Returns:
        jnp.ndarray: lever arm matrix
    """
    return C_dd_inv @ C_dg
