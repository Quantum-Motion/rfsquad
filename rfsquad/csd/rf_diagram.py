from __future__ import annotations

from typing import TYPE_CHECKING

from matplotlib.figure import Figure as MatplotlibFigure

from rfsquad.csd.interface import IChargeStabilityDiagram
from rfsquad.utils.globals import LABEL_MAP

if TYPE_CHECKING:
    from rfsquad.csd.charge_stability_diagram import ChargeStabilityDiagram

"""
This module implements the `RFDiagram` class for visualizing and analysing radio-frequency (RF) charge stability diagrams
of quantum dot systems. The RF diagram inherits from the `IChargeStabilityDiagram` interface and is used to generate RF
readout data.

Class:
- **RFDiagram**: Specialized class for processing and visualizing RF-related charge stability data in quantum dot systems.
"""


class RFDiagram(IChargeStabilityDiagram):
    """
    Class representing the RF charge stability diagram object
    """

    def __init__(
        self,
        csd: ChargeStabilityDiagram,
        temperature: float,
        sensor_gate: str,
    ):
        """
        Class representing the RF charge stability diagram object

        Args:
            csd (ChargeStabilityDiagram): Charge stability diagram object
            temperature (float): Temperature in Kelvin
            sensor_gate (str): Sensor gate for the RF readout
        """
        super().__init__(**csd.__dict__)

        self.temperature = temperature
        self.sensor_gate = sensor_gate
        self._process_data()

    def __post_init__(self):
        self.cmap_label = LABEL_MAP["rf"]

    def plot(
        self,
        cmap: str = "Greys",
        figsize: tuple[float, float] = (4, 3),
        slice: dict[str, float] = None,  # type: ignore
        x_axis: str = None,  # type: ignore
        y_axis: str = None,  # type: ignore
    ) -> MatplotlibFigure:
        """
        Plots the RF charge stability diagram.

        Args:
            cmap (str, optional): Matplotlib colormap. Defaults to 'Greys'.
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

    def _process_data(self) -> None:
        match self.solver.lower():
            case "rf":
                self.data = self.raw_data

            case _:
                raise NotImplementedError(
                    f"Solver {self.solver} not implemented -- only RFSolver supported."
                )
