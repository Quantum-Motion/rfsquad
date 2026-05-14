from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from matplotlib.figure import Figure as MatplotlibFigure

if TYPE_CHECKING:
    from rfsquad.csd.charge_stability_diagram import ChargeStabilityDiagram

from rfsquad.csd.interface import IChargeStabilityDiagram
from rfsquad.utils.globals import LABEL_MAP
from rfsquad.utils.utils_csd import reshape_and_mask_data

"""
This module implements the `EnergyDiagram` class for visualizing and analyzing the energy landscape of quantum dot systems.
The energy diagram inherits from the `IChargeStabilityDiagram` interface and specializes in processing and plotting energy
diagrams of quantum dot arrays.

Class:
- **EnergyDiagram**: Specialized class for processing and visualizing energy-related data in quantum dot systems.
"""


class EnergyDiagram(IChargeStabilityDiagram):
    """
    Class representing the energy charge stability diagram object
    """

    def __init__(self, csd: ChargeStabilityDiagram):
        """
        Class representing the energy charge stability diagram object

        Args:
            csd (ChargeStabilityDiagram): Charge stability diagram object
        """
        super().__init__(**csd.__dict__)
        self._process_data()

    def __post_init__(self):
        self.cmap_label = LABEL_MAP["energy"]

    def _process_data(self) -> None:
        # We need to process the data taking into account the negative fock states.
        # To do this, we first determine the the indices of the negative fock states.
        # We then find their corresponding eigenvalues and set these equal to jnp.inf before determining the minimum eigeneneriges.
        # All the above is completed in the reshape and mask data function.

        ################# CURRENT SHAPES #################
        # Fock states: (config, num_dots, sweep points)
        # raw data: (sweep1, sweep2, ...., sweepN, config)
        ##################################################
        if (self.fock_states < 0).any():
            self.raw_data = reshape_and_mask_data(self.raw_data, self.fock_states)
        self.data = jnp.min(self.raw_data, axis=-1)

    def plot(
        self,
        cmap: str = "Greys",
        figsize: tuple[float, float] = (4, 3),
        slice: dict[str, float] = None,  # type: ignore
        x_axis: str = None,  # type: ignore
        y_axis: str = None,  # type: ignore
    ) -> MatplotlibFigure:
        """
        Plots the charge stability diagram.

        Args:
            cmap (str, optional): Matplotlib colourmap. Defaults to 'Greys'.
            figsize (tuple[float, float], optional): Figure size in inches. Defaults to (4,3).
            slice (dict[str, float], optional): Slice of the n-dimensional data array in the gate:voltage format. Must be specified if more than two gates are swept. Defaults to None.
            x_axis (str, optional): Gate to use for the x-axis. Defaults to None.
            y_axis (str, optional): Gate to use for the y-axis. Defaults to None.

        Raises:
            ValueError: if more than two gates are present and no slice is provided

        Returns:
            MatplotlibFigure: matplotlib figure
        """

        data, plot_gates = self._get_slice_plot(slice, x_axis, y_axis)

        fig, axs = self._plot(data, cmap, figsize, plot_gates)
        return fig
