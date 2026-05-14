from __future__ import annotations

from typing import TYPE_CHECKING

from rfsquad.csd.energy_diagram import EnergyDiagram
from rfsquad.csd.interface import IChargeStabilityDiagram
from rfsquad.csd.occupancy_diagram import OccupancyDiagram
from rfsquad.csd.rf_diagram import RFDiagram

if TYPE_CHECKING:
    pass

"""
This module implements the `ChargeStabilityDiagram` class, which serves as the base class for managing charge stability
diagrams in quantum dot simulations in rf-squad. While this class cannot be directly plotted, it provides methods to retrieve specialized
diagrams such as energy diagrams, occupancy diagrams, and RF diagrams.

Class:
- **ChargeStabilityDiagram**: Base class for generating and managing charge stability diagrams, serving as a foundation for energy, occupancy, and RF diagrams.
"""


class ChargeStabilityDiagram(IChargeStabilityDiagram):
    """
    Class representing the base charge stability diagram object. Use to get the energy, occupancy or RF charge stability diagrams, or to save/load the diagram. Cannot be plotted directly.
    """

    def __init__(self, sweep, raw_data, solver, virtual_gates, fock_states):
        super().__init__(
            sweep=sweep,
            raw_data=raw_data,
            solver=solver,
            virtual_gates=virtual_gates,
            fock_states=fock_states,
        )

    def _process_data(self):
        """
        Not implemented for base charge stability diagram
        """
        pass

    def plot(self):
        """
        Not implemented for base charge stability diagram
        """

        raise NotImplementedError(
            "Plotting is not implemented for the base charge stability diagram - use the specialised OccupancyDiagram, EnergyDiagram or RFDiagram."
        )

    def get_energy_diagram(self) -> EnergyDiagram:
        """
        Get the energy charge stability diagram object

        Returns:
            EnergyDiagram: Energy charge stability diagram object

        Raises:
        NotImplementedError: If the solver is "rf", as energy diagrams
                             are not available for RF solvers.
        """

        if self.solver == "rf":
            raise NotImplementedError(
                "Energy diagrams are not available for RF solvers."
            )
        return EnergyDiagram(self)

    def get_occupancy_diagram(self) -> OccupancyDiagram:
        """
        Get the occupancy charge stability diagram object

        Returns:
            OccupancyDiagram: Occupancy charge stability diagram object

        Raises:
        NotImplementedError: If the solver is "rf", as energy diagrams
                             are not available for RF solvers.
        """

        if self.solver == "rf":
            raise NotImplementedError(
                "Occupancy diagrams are not available for RF solvers."
            )
        return OccupancyDiagram(self)

    def get_rf_diagram(self, temperature: float, sensor_gate: str) -> RFDiagram:
        """
        Get the RF charge stability diagram object

        Raises:
        NotImplementedError: If the solver is not "rf", as RF diagrams
                             are available only for RF solvers.

        Returns:
            RFDiagram: RF charge stability diagram object
        """
        if self.solver != "rf":
            raise NotImplementedError("RF diagrams are only available for RF solvers.")
        return RFDiagram(self, temperature, sensor_gate)

    def add_noise(self):
        """
        Not implemented for base charge stability diagram
        """
        raise NotImplementedError(
            "Noise cannot be added to the base charge stability diagram."
        )

    def remove_noise(self):
        """
        Not implemented for base charge stability diagram
        """
        raise NotImplementedError(
            "Noise cannot be added to the base charge stability diagram."
        )
