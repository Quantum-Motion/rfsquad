from __future__ import annotations

from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.figure import Figure as MatplotlibFigure

if TYPE_CHECKING:
    from rfsquad.csd.charge_stability_diagram import ChargeStabilityDiagram

from rfsquad.csd.interface import IChargeStabilityDiagram
from rfsquad.utils.globals import LABEL_MAP
from rfsquad.utils.utils_csd import reshape_and_mask_data

"""
This module implements the `OccupancyDiagram` class for visualizing and analyzing the total occupancy of quantum dot systems
in a charge stability diagram. The occupancy diagram inherits from the `IChargeStabilityDiagram` interface and
specializes in processing and plotting charge occupancies derived from Fock states.

Class:
- **OccupancyDiagram**: Specialized class for processing and visualizing occupancy-related data in quantum dot systems.
"""


class OccupancyDiagram(IChargeStabilityDiagram):
    """
    Class representing the occupancy charge stability diagram object
    """

    def __init__(self, csd: ChargeStabilityDiagram):
        """
        Class representing the occupancy charge stability diagram object

        Args:
            csd (ChargeStabilityDiagram): Charge stability diagram object
        """
        super().__init__(**csd.__dict__)
        self._process_data()

    def __post_init__(self):
        self.cmap_label = LABEL_MAP["occupancy"]

    def _compute_fock_states(self, data):
        # Step 1: Determine the minimum energies
        min_energy_indices = jnp.argmin(data, axis=-1)
        min_energy_indices_reshaped = min_energy_indices.reshape(
            1, 1, -1
        )  # Shape: (1, 1, sweep points)

        # Step 2: Select the required fock states that correspond to the minimum
        selected_fock_states = jnp.take_along_axis(
            self.fock_states, min_energy_indices_reshaped, axis=0
        ).squeeze(axis=0)  # Shape: (num_dots, sweep points)

        # Sum along the first axis (num dots) to get total occupancy and reshape based on the sweep structure
        return jnp.sum(selected_fock_states, axis=0).reshape(
            *jax.tree.map(lambda x: x.shape[0], jax.tree.leaves(self.sweep))
        )  # Shape: (sweep1, sweep2, ... sweepN)

    def _process_data(self) -> None:
        # In a similar way to CI solver, we need to reshape the fock states to align with the raw data and apply masking
        # to get rid of negative fock states energies in the raw data and set these to infinity. This is then followed by selecting the fock states with the minimum energy.

        if (self.fock_states < 0).any():
            self.raw_data = reshape_and_mask_data(self.raw_data, self.fock_states)
        self.data = self._compute_fock_states(self.raw_data)

    def plot(
        self,
        cmap: str = "Greys",
        figsize: tuple[float, float] = (4, 3),
        slice: dict[str, float] = None,  # type: ignore
        x_axis: str = None,  # type: ignore
        y_axis: str = None,  # type: ignore
    ) -> MatplotlibFigure:
        """
        Plots the occupancy diagram.

        Args:
            cmap (str, optional): Matplotlib colourmap. Defaults to 'Greys'. Colour map is discretised to convey finite occupancy from CISolver.
            figsize (tuple[float, float], optional): Figure size in inches. Defaults to (4,3).
            slice (dict[str, float], optional): Slice of the n-dimensional data array in the gate voltage format. Must be specified if more than two gates are swept. Defaults to None.
            x_axis (str, optional): Gate to use for the x-axis. Defaults to None.
            y_axis (str, optional): Gate to use for the y-axis. Defaults to None.

        Raises:
            ValueError: if more than two gates are present and no slice is provided

        Returns:
            MatplotlibFigure: matplotlib figure
        """

        data, plot_gates = self._get_slice_plot(slice, x_axis, y_axis)

        # only discretise the cmap if the solver is cim
        new_cmap = self._colour_helper(plot_gates, cmap)

        fig, axs = self._plot(data, new_cmap, figsize, plot_gates)
        return fig

    def _colour_helper(self, plot_gates: list[str], cmap: str | mcolors.Colormap):
        new_cmap: str | mcolors.Colormap

        if len(self.sweep) == 1:
            # pick the first colour
            if isinstance(cmap, str):
                # is a singular hex colour
                if cmap.startswith("#"):
                    new_cmap = cmap

                else:
                    # get the first colour, make it hex
                    new_cmap = mcolors.rgb2hex(plt.get_cmap(cmap, 1)(0))

        elif plot_gates is not None and len(plot_gates) == 1:
            # also pick the first colour
            if isinstance(cmap, str):
                # is a singular hex colour
                if cmap.startswith("#"):
                    new_cmap = cmap

                else:
                    # get the first colour, make it hex
                    new_cmap = mcolors.rgb2hex(plt.get_cmap(cmap, 1)(0))

        else:
            if self.solver == "cim":
                max_n = int(jnp.max(self.data))
                new_cmap = plt.get_cmap(cmap, max_n + 1)

            else:
                new_cmap = plt.get_cmap(cmap)

        return new_cmap
