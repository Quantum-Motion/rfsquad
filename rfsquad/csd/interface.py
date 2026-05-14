from __future__ import annotations

import copy
import pathlib
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.axes import Axes as MatplotlibAxes
from matplotlib.figure import Figure as MatplotlibFigure

if TYPE_CHECKING:
    from rfsquad.solvers import ISolver

from rfsquad.noise import INoiseModel
from rfsquad.utils.globals import CSD_ALLOWED, CSD_REQUIRED, LABEL_MAP
from rfsquad.utils.io import HDF5, Json, PathLike, _from_file_helper

# ----------------
# MODULE STRUCTURE
# ----------------
# IChargeStabilityDiagram is an abstract factory class that defines the interface for the charge stability diagrams
# ChargeStabilityDiagram is the base class that implements the interface and gets returned by the solver but is useless on its own - it only holds all the raw, unprocessed data until one of the get methods is called
# EnergyDiagram, OccupancyDiagram and RFDiagram are the classes that actually process the data and plot it - they get returned by calling the get methods on the ChargeStabilityDiagram object
# you can also access the EnergyDiagram, OccupancyDiagram and RFDiagram classes directly by creating them with XXDiagram.from_file() and importing a previously saved diagram
# RF diagram has additional arguments you need to pass to it (which do not become attributes) - device, temperature and sensor dot and gate - this is because it needs to calculate the change in capacitance from the data
# ----------------


@dataclass
class IChargeStabilityDiagram(ABC):
    """
    Abstract interface class for charge stability diagrams
    """

    sweep: dict[str, jnp.ndarray]
    raw_data: jnp.ndarray
    solver: str
    virtual_gates: bool
    fock_states: jnp.ndarray
    cmap_label: str = ""

    def __post_init__(self):
        self.cmap_label = LABEL_MAP[self.solver]

    def __repr__(self) -> str:
        representation = f"{self.__class__.__name__} object\n\n"
        # sweep gates and their min max voltages
        representation += "Sweeping:\n"
        for gate, voltage in self.sweep.items():
            representation += (
                f"    {gate}: {voltage.min():.3f} to {voltage.max():.3f} V\n"
            )

        if self.virtual_gates:
            representation += "    Using virtual gates\n"

        if hasattr(self, "noise_model"):
            representation += f"\n\nUsing \n{self.noise_model}\n"  # type: ignore

        if hasattr(self, "tr_model"):
            representation += f"\n\nUsing \n{self.tr_model}\n"

        return representation

    @abstractmethod
    def _process_data(self) -> None:
        pass

    @abstractmethod
    def plot(self) -> MatplotlibFigure:
        pass

    def add_noise(self, noise_model: INoiseModel) -> None:
        """
        Adds noise models to the processed data, overwriting the data attribute and saving the model to this object

        Args:
            noise_model (INoiseModel): Noise model object
        """
        self.data = noise_model.apply(self.data, self.sweep)  # type: ignore
        self.noise_model = noise_model

    def remove_noise(self) -> None:
        """
        Removes the noise model from the data attribute by re-running post processing.
        """
        self._process_data()
        if hasattr(self, "noise_model"):
            del self.noise_model

    @classmethod
    def from_file(cls, filename: PathLike):
        """
        Load a charge stability diagram from a file.

        Only hdf5/h5 and json file formats are supported. File must contain the class name in the CSD group - this is matched and validated against the class name of the instantiated class.

        Args:
            filename (PathLike): Filename to load the charge stability diagram from.

        Returns:
            IChargeStabilityDiagram: Charge stability diagram object
        """

        from rfsquad.csd.charge_stability_diagram import ChargeStabilityDiagram

        file_obj = _from_file_helper(filename)

        # check that name in the csd matches the class name
        if file_obj.data["CSD"]["name"] != cls.__name__:  # type: ignore
            raise ValueError(
                f"Class name in the file ({file_obj.data['csd']['name']}) does not match the instantiated class name ({cls.__name__})."
            )
        # remove the name from dict so its not causing undue stress
        _ = file_obj.data["CSD"].pop("name")

        file_obj._check_data_format("CSD", CSD_REQUIRED, CSD_ALLOWED)

        # this does not seem like a good solution
        # no particular reason, just feels like i should've come up with a better inheritance structure
        match cls.__name__:
            case "ChargeStabilityDiagram":
                inst = cls(**file_obj.data["CSD"])

            case "EnergyDiagram" | "OccupancyDiagram":
                # remove data bc there's nothing to process for just CSD
                _ = file_obj.data["CSD"].pop("data")
                csd = ChargeStabilityDiagram(**file_obj.data["CSD"])
                inst = cls(csd)  # type: ignore # cls will be one of the subclasses

            case _:
                raise ValueError(f"Class name {cls.__name__} not implemented.")

        return inst

    def to_file(self, filename: PathLike, solver: ISolver = None) -> None:  # type: ignore
        """
        Save the charge stability diagram to a file. If a solver is provided it will also be saved as Device and Solver so they can be loaded back in separately.

        Args:
            filename (PathLike): Filename to save the charge stability diagram to.
            solver (ISolver, optional): Solver object. Defaults to None.
        """
        lst: list[object] = [self]
        if solver is not None:
            lst.append(solver)

        if hasattr(self, "noise_model"):
            lst.append(self.noise_model)

        match pathlib.Path(filename).suffix:
            case ".h5" | ".hdf5":
                HDF5.dump(filename=filename, objects=lst)  # type: ignore

            case ".json":
                Json.dump(filename=filename, objects=lst)  # type: ignore

            case _:
                raise ValueError(
                    "File extension not supported. Please use either .json or .h5/.hdf5."
                )

    def _plot(
        self,
        data: jnp.ndarray,
        cmap: str | mcolors.Colormap,
        figsize: tuple[float, float],
        plot_gates: list[str],
    ) -> tuple[MatplotlibFigure, MatplotlibAxes]:
        """
        Helper for plotting the charge stability diagram - picks either 1D or 2D plots based on number of gates in the sweep or total gates being plotted (so you can use it with multiple gates swept but one gate plotted)
        Returns:
            tuple[MatplotlibFigure, MatplotlibAxes]: _description_
        """
        if data.ndim > 2:
            raise ValueError("Data has more than 2 dimensions - cannot plot.")

        elif data.ndim < 1:
            raise ValueError("Data has no dimensions - cannot plot.")

        if len(self.sweep) == 1 or (plot_gates is not None and len(plot_gates) == 1):
            return self._plot_1D(data, cmap, figsize, plot_gates)  # type: ignore

        else:
            return self._plot_2D(data, cmap, figsize, plot_gates)

    def _plot_1D(
        self,
        data: jnp.ndarray,
        colour: str,
        figsize: tuple[float, float],
        plot_gates: list[str],
    ) -> tuple[MatplotlibFigure, MatplotlibAxes]:
        x_label, x = self._check_plot_gates_1D(plot_gates)
        y_label = self.cmap_label
        y = data

        c = self._check_colours_1D(cmap=colour)

        fig, axs = plt.subplots(1, 1, dpi=300, figsize=figsize, constrained_layout=True)
        axs.set_box_aspect(1)
        axs.set_xlabel(x_label)
        axs.set_ylabel(y_label)
        axs.plot(x, y, c=c, ls="", marker="o", ms=4)

        return fig, axs

    def _plot_2D(
        self,
        data: jnp.ndarray,
        cmap: str | mcolors.Colormap,
        figsize: tuple[float, float],
        plot_gates: list[str],
    ) -> tuple[MatplotlibFigure, MatplotlibAxes]:
        """
        Helper function for plotting that sets up the figure and axes

        Args:
            data (jnp.ndarray): Data to plot
            cmap (str | plt.colormap): Matplotlib colourmap
            figsize (tuple[float, float]): Figure size in inches
            plot_gates (list[str]): List of gates to plot on x and y axes

        Returns:
            fig, ax: Matplotlib figure and axis
        """

        x_label, y_label, x, y = self._check_plot_gates_2D(plot_gates=plot_gates)

        fig, axs = plt.subplots(1, 1, dpi=300, figsize=figsize, constrained_layout=True)
        axs.set_box_aspect(1)
        axs.set_xlabel(x_label)
        axs.set_ylabel(y_label)
        # data for colormesh needs to be in the shape (rows, columns)
        # but then x corresponds to columns and y to rows
        # so data needs to be transposed to be correct
        mesh = axs.pcolormesh(x, y, data.T, cmap=cmap, shading="auto")
        cbar = fig.colorbar(mesh, ax=axs, label=self.cmap_label, use_gridspec=True)  # noqa: F841

        return fig, axs

    def _check_plot_gates_1D(self, plot_gates: list[str]) -> tuple[str, jnp.ndarray]:
        if plot_gates is not None:
            if plot_gates[0] not in self.sweep.keys():
                raise KeyError(
                    "Gate selected for plotting is not defined in the sweep dictionary."
                )

            gate_x = plot_gates[0]
            x = self.sweep[gate_x]

        else:
            gate_x = list(self.sweep.keys())[0]
            x = self.sweep[gate_x]

        if self.virtual_gates:
            label_x = f"Virtual gate voltage at {gate_x} (V)"
        else:
            label_x = f"Gate voltage at {gate_x} (V)"

        return label_x, x

    def _check_colours_1D(self, cmap):
        new_cmap = cmap

        if isinstance(cmap, str):
            # is a colourmap name
            if cmap in plt.colormaps():
                # get the first colour, make it hex
                new_cmap = mcolors.rgb2hex(plt.get_cmap(cmap, 1)(0))

        return new_cmap

    def _check_plot_gates_2D(
        self, plot_gates: list[str]
    ) -> tuple[str, str, jnp.ndarray, jnp.ndarray]:
        """
        Helper function for plotting labels

        Args:
            plot_gates (list[str]): List of gates to plot on x and y axes

        Returns:
            tuple of gate names and x and y values
        """
        if plot_gates is not None:
            if len(plot_gates) != 2:
                raise ValueError("Only two gates can be plotted at any time.")

            if not all(key in self.sweep for key in plot_gates):
                raise KeyError(
                    "Not all gates selected for plotting are defined in the sweep dictionary."
                )

            gate_x = plot_gates[0]
            gate_y = plot_gates[1]
            x = self.sweep[gate_x]
            y = self.sweep[gate_y]

        else:
            gate_x = list(self.sweep.keys())[0]
            gate_y = list(self.sweep.keys())[1]
            x = self.sweep[gate_x]
            y = self.sweep[gate_y]

        if self.virtual_gates:
            label_x = f"Virtual gate voltage at {gate_x} (V)"
            label_y = f"Virtual gate voltage at {gate_y} (V)"
        else:
            label_x = f"Gate voltage at {gate_x} (V)"
            label_y = f"Gate voltage at {gate_y} (V)"

        return label_x, label_y, x, y

    def get_slice(self, slice: dict[str, float]) -> jnp.ndarray:
        """
        Get a slice of the charge stability diagram at a specific gate and voltage.

        Args:
            slice (dict[str, float]): Dictionary of gate name and voltage to slice the charge stability diagram at.

        Returns:
            jnp.ndarray: array of sliced data in the same shape, but with empty dimensions
        """
        # need to deep copy sweep and data here
        # as iterating over everything cut them out
        local_sweeps = copy.deepcopy(self.sweep)
        local_data = copy.deepcopy(self.data)  # type: ignore

        for gate, voltage in slice.items():
            # find the index in the sweep that is closest to the slice value
            idx_voltage = jnp.argmin(jnp.abs(local_sweeps[gate] - voltage))
            idx_gate = list(local_sweeps.keys()).index(gate)

            # slice the data - want to pick the axis in the self.data that corresponds to the gate we're slicing at and then the voltage?
            # np.take copies the data so memory stress may be considerable
            local_data = local_data.take(idx_voltage, axis=idx_gate)  # type: ignore
            local_sweeps.pop(gate)

        return local_data

    def _get_slice_plot(
        self, slice, x_axis=None, y_axis=None
    ) -> tuple[jnp.ndarray, list[str]]:
        if (x_axis is None) != (y_axis is None):
            raise ValueError("Both x_axis and y_axis must be provided or neither.")

        if x_axis is not None and y_axis is not None and x_axis == y_axis:
            raise ValueError("x_axis and y_axis must be different.")

        if slice is None:
            plot_gates: list[str] = list(self.sweep.keys())
            data = self.data  # type: ignore # sO dramatic

        else:
            data = self.get_slice(slice)
            plot_gates = []
            for k in self.sweep.keys():
                if k not in slice:
                    plot_gates.append(k)

        if x_axis and y_axis:
            # Here we are checking that if a user is attempting to a plot a 1d plot, the x and y axis canot be given.
            # This is because the user is attempting to plot a 1d plot with 2 axes which is not possible.
            if len(plot_gates) == 1:
                raise ValueError(
                    "Cannot supply x_axis and y_axis when less than 2 gates are swept."
                )
            if x_axis not in plot_gates or y_axis not in plot_gates:
                raise ValueError(f"x_axis and y_axis must be in {plot_gates}")
            x_idx = plot_gates.index(x_axis)
            y_idx = plot_gates.index(y_axis)
            data = jnp.transpose(data, (x_idx, y_idx))
            plot_gates = [x_axis, y_axis]

        return data, plot_gates
