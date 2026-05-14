import copy
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest
from matplotlib import colors as mcolors
from matplotlib.axes import Axes as MatplotlibAxes
from matplotlib.figure import Figure as MatplotlibFigure

from rfsquad.core import Device
from rfsquad.csd import EnergyDiagram, OccupancyDiagram, RFDiagram
from rfsquad.noise import AlphaNoise, GaussianNoise, NoiseModel
from rfsquad.solvers import CISolver
from rfsquad.utils.utils_csd import reshape_and_mask_data


## FIXTURES
@pytest.fixture
def dqd_device():
    c_dd = jnp.array([[100.0, -10.0], [-10.0, 100.0]])
    c_dg = jnp.array([[10.0, 2.0], [2.0, 10.0]])
    dev = Device(c_dd, c_dg, gate_names=["gate1", "gate2"])
    dev.add_static_voltages(gate={"gate1": 0.1, "gate2": 0.2})
    yield dev


@pytest.fixture
def dqd_device_barrier_end():
    c_dd = jnp.array([[100.0, -10.0], [-10.0, 100.0]])
    c_dg = jnp.array(
        [[3, 0.1, 1.2], [1.2, 0.1, 3]]
    )  # Barrier gate is the middle column

    dev = Device(c_dd, c_dg, gate_names=["P1", "B", "P2"])
    dev.add_static_voltages(gate={"P1": 0.1, "P2": 0.2})
    yield dev


@pytest.fixture
def dqd_device_barrier_start():
    c_dd = jnp.array([[100.0, -10.0], [-10.0, 100.0]])
    c_dg = jnp.array([[0.1, 3, 1.2], [0.1, 1.2, 3]])
    dev = Device(c_dd, c_dg, gate_names=["B", "P1", "P2"])
    dev.add_static_voltages(gate={"P1": 0.1, "P2": 0.2})
    yield dev


@pytest.fixture
def dqd_sweep():
    yield dict(gate2=jnp.linspace(0, 0.2, 10), gate1=jnp.linspace(0, 0.1, 10))


@pytest.fixture
def dqd_csd(dqd_device, dqd_sweep):
    solver = CISolver(dqd_device)
    csd = solver.run(dqd_sweep, update_solver={"max_charge_carriers": 1})
    yield csd


@pytest.fixture
def dqd_csd_cim_rfs(dqd_device, dqd_sweep):
    solver = CISolver(dqd_device, reduce_fock_space=True)
    csd = solver.run(dqd_sweep, update_solver={"max_charge_carriers": 1})
    yield csd


@pytest.fixture
def tqd_device():
    c_dd = jnp.array(
        [[100.0, -10.0, -1.0], [-10.0, 100.0, -10.0], [-1.0, -10.0, 100.0]]
    )
    c_dg = jnp.array([[10.0, 2.0, 1.0], [2.0, 10.0, 2.0], [1.0, 2.0, 10.0]])
    dev = Device(c_dd, c_dg)
    yield dev


@pytest.fixture
def tqd_csd(tqd_device):
    sweep = dict(
        gate_1=jnp.linspace(0, 0.1, 10),
        gate_3=jnp.linspace(0, 0.15, 10),
    )
    solver = CISolver(tqd_device)
    csd = solver.run(sweep, update_solver={"max_charge_carriers": 1})
    yield csd


@pytest.fixture
def tqd_csd_3s(tqd_device):
    sweep = dict(
        gate_1=jnp.linspace(0, 0.1, 5),
        gate_2=jnp.linspace(0, 0.2, 10),
        gate_3=jnp.linspace(0, 0.3, 15),
    )
    solver = CISolver(tqd_device)
    csd = solver.run(sweep, update_solver={"max_charge_carriers": 1})
    yield csd


## BASE CHARGE STABILITY DIAGRAM TESTS
class TestCSD:
    def test_csd_init(self, dqd_csd):
        assert hasattr(dqd_csd, "solver")
        assert hasattr(dqd_csd, "raw_data")
        assert hasattr(dqd_csd, "virtual_gates")
        assert hasattr(dqd_csd, "cmap_label")
        assert dqd_csd.cmap_label == "Energy (eV)"

    def test_plot(self, dqd_csd):
        with pytest.raises(NotImplementedError):
            dqd_csd.plot()

    def test_get_occupancy_diagram(self, dqd_csd):
        od = dqd_csd.get_occupancy_diagram()
        assert isinstance(od, OccupancyDiagram)

    def test_get_energy_diagram(self, dqd_csd):
        ed = dqd_csd.get_energy_diagram()
        assert isinstance(ed, EnergyDiagram)

    def test_add_noise(self, dqd_csd):
        with pytest.raises(NotImplementedError):
            dqd_csd.add_noise()

    def test_remove_noise(self, dqd_csd):
        with pytest.raises(NotImplementedError):
            dqd_csd.remove_noise()


class TestOccupancyDiargam:
    def test_init(self, dqd_csd):
        od = OccupancyDiagram(dqd_csd)
        assert hasattr(od, "solver")
        assert hasattr(od, "raw_data")
        assert hasattr(od, "virtual_gates")
        assert hasattr(od, "cmap_label")
        assert od.cmap_label == "Occupancy"
        assert hasattr(od, "data")

    def test_repr(self, dqd_csd):
        csd = OccupancyDiagram(dqd_csd)
        assert isinstance(csd.__repr__(), str)
        representation = csd.__repr__()
        assert "OccupancyDiagram" in representation
        assert "Sweeping:" in representation
        assert "gate1: 0.000 to 0.200 "

    def test_init_from_csd(self, dqd_csd):
        od = dqd_csd.get_occupancy_diagram()
        assert hasattr(od, "solver")
        assert hasattr(od, "raw_data")
        assert hasattr(od, "virtual_gates")
        assert hasattr(od, "cmap_label")
        assert od.cmap_label == "Occupancy"
        assert hasattr(od, "data")

    def test_plot_1D(self, dqd_csd):
        od = OccupancyDiagram(dqd_csd)
        figure = od.plot(slice={"gate1": 0.1})
        assert isinstance(figure, MatplotlibFigure)
        assert figure.get_axes()[0].get_xlabel() == "Gate voltage at gate2 (V)"
        assert figure.get_axes()[0].get_ylabel() == "Occupancy"

    def test_plot_2D(self, dqd_csd):
        od = OccupancyDiagram(dqd_csd)
        figure = od.plot()
        assert isinstance(figure, MatplotlibFigure)
        assert figure.get_axes()[0].get_xlabel() == "Gate voltage at gate1 (V)"
        assert figure.get_axes()[0].get_ylabel() == "Gate voltage at gate2 (V)"
        # check that colormap has been discretised?

    def test_plot_3D(self, tqd_csd_3s):
        od = OccupancyDiagram(tqd_csd_3s)

        with pytest.raises(ValueError, match="Data has more than 2 dimensions"):
            od.plot()

    def test_plot_0D(self, tqd_csd_3s):
        od = OccupancyDiagram(tqd_csd_3s)

        with pytest.raises(ValueError, match="Data has no dimensions"):
            od.plot(slice={"gate_1": 0.1, "gate_2": 0, "gate_3": 0})

    def test_colours_helper(self, dqd_csd):
        od = OccupancyDiagram(dqd_csd)
        cmap = od._colour_helper(plot_gates=["gate1", "gate2"], cmap="viridis")
        assert isinstance(cmap, mcolors.ListedColormap)
        assert mcolors.rgb2hex(cmap(0)) == "#440154"

        cmap = od._colour_helper(plot_gates=["gate1"], cmap="#ff4342")
        assert cmap == "#ff4342"

    def test_add_gaussian_noise(self, dqd_csd):
        od = OccupancyDiagram(dqd_csd)
        data_1 = copy.deepcopy(od.data)
        gn = GaussianNoise(0.1)
        od.add_noise(gn)
        data_2 = copy.deepcopy(od.data)
        assert not np.array_equal(data_1, data_2)

    def test_add_noise_model(self, dqd_csd):
        od = OccupancyDiagram(dqd_csd)
        data_1 = copy.deepcopy(od.data)
        nm = NoiseModel()
        nm.add(GaussianNoise(0.1))
        od.add_noise(nm)
        data_2 = copy.deepcopy(od.data)
        assert not np.array_equal(data_1, data_2)

    def test_alpha_noise(self, dqd_csd):
        od = OccupancyDiagram(dqd_csd)
        data_1 = copy.deepcopy(od.data)
        nm = NoiseModel()
        nm.add(AlphaNoise(0.1, 1000))
        od.add_noise(nm)
        data_2 = copy.deepcopy(od.data)
        assert not np.array_equal(data_1, data_2)

    def test_remove_noise(self, dqd_csd):
        od = OccupancyDiagram(dqd_csd)
        data_1 = copy.deepcopy(od.data)
        gn = GaussianNoise(0.1)
        od.add_noise(gn)
        data_2 = copy.deepcopy(od.data)
        od.remove_noise()
        assert np.array_equal(data_1, od.data)
        assert not np.array_equal(data_2, od.data)
        assert not hasattr(od, "noise_model")


class TestCSDSameness:
    def test_same_energy(self, dqd_device_barrier_end, dqd_device_barrier_start):
        sweep = dict(P1=jnp.linspace(0, 0.1, 10), P2=jnp.linspace(0, 0.2, 10))
        solver = CISolver(dqd_device_barrier_end)
        csd_end = solver.run(sweep)

        solver = CISolver(dqd_device_barrier_start)
        csd_start = solver.run(sweep)

        ed_end = csd_end.get_energy_diagram()
        ed_start = csd_start.get_energy_diagram()

        np.testing.assert_array_almost_equal(
            ed_end.raw_data, ed_start.raw_data, decimal=8
        )

    def test_same_occupancy(self, dqd_device_barrier_end, dqd_device_barrier_start):
        sweep = dict(P1=jnp.linspace(0, 0.1, 10), P2=jnp.linspace(0, 0.2, 10))
        solver = CISolver(dqd_device_barrier_end)
        csd_end = solver.run(sweep, update_solver={"max_charge_carriers": 1})

        solver = CISolver(dqd_device_barrier_start)
        csd_start = solver.run(sweep, update_solver={"max_charge_carriers": 1})

        od_end = csd_end.get_occupancy_diagram()
        od_start = csd_start.get_occupancy_diagram()

        np.testing.assert_array_equal(od_end.data, od_start.data)

    def test_csd_dict_gate_names(self, dqd_device_barrier_end):
        sweep = dict(P1=jnp.linspace(0, 0.1, 10), P2=jnp.linspace(0, 0.2, 10))
        solver = CISolver(dqd_device_barrier_end)
        _ = solver.run(sweep)
        # Before jit
        assert dqd_device_barrier_end.gate_names == ["P1", "B", "P2"]
        # After jit
        assert solver.device.gate_names == ["P1", "B", "P2"]


class TestEnergyDiargam:
    def test_init(self, dqd_csd):
        ed = EnergyDiagram(dqd_csd)
        assert hasattr(ed, "solver")
        assert hasattr(ed, "raw_data")
        assert hasattr(ed, "virtual_gates")
        assert hasattr(ed, "cmap_label")
        assert ed.cmap_label == "Energy (eV)"
        assert hasattr(ed, "data")

    def test_init_from_csd(self, dqd_csd):
        ed = dqd_csd.get_energy_diagram()
        assert hasattr(ed, "solver")
        assert hasattr(ed, "raw_data")
        assert hasattr(ed, "virtual_gates")
        assert hasattr(ed, "cmap_label")
        assert ed.cmap_label == "Energy (eV)"
        assert hasattr(ed, "data")

    def test_repr(self, dqd_csd):
        csd = EnergyDiagram(dqd_csd)
        assert isinstance(csd.__repr__(), str)
        representation = csd.__repr__()
        assert "EnergyDiagram" in representation
        assert "Sweeping:" in representation
        assert "gate1: 0.000 to 0.200 "

    def test_plot_1D(self, dqd_csd):
        ed = EnergyDiagram(dqd_csd)
        figure = ed.plot(slice={"gate1": 0.1})
        assert isinstance(figure, MatplotlibFigure)
        assert figure.get_axes()[0].get_xlabel() == "Gate voltage at gate2 (V)"
        assert figure.get_axes()[0].get_ylabel() == "Energy (eV)"

    def test_plot_2D(self, dqd_csd):
        ed = EnergyDiagram(dqd_csd)
        figure = ed.plot()
        assert isinstance(figure, MatplotlibFigure)
        assert figure.get_axes()[0].get_xlabel() == "Gate voltage at gate1 (V)"
        assert figure.get_axes()[0].get_ylabel() == "Gate voltage at gate2 (V)"

    def test_plot_3D(self, tqd_csd_3s):
        ed = EnergyDiagram(tqd_csd_3s)

        with pytest.raises(ValueError, match="Data has more than 2 dimensions"):
            ed.plot()

    def test_plot_0D(self, tqd_csd_3s):
        ed = EnergyDiagram(tqd_csd_3s)

        with pytest.raises(ValueError, match="Data has no dimensions"):
            ed.plot(slice={"gate_1": 0.1, "gate_2": 0, "gate_3": 0})

    def test_add_gaussian_noise(self, dqd_csd):
        od = EnergyDiagram(dqd_csd)
        data_1 = copy.deepcopy(od.data)
        gn = GaussianNoise(0.1)
        od.add_noise(gn)
        data_2 = copy.deepcopy(od.data)
        assert not np.array_equal(data_1, data_2)

    def test_add_noise_model(self, dqd_csd):
        od = EnergyDiagram(dqd_csd)
        data_1 = copy.deepcopy(od.data)
        nm = NoiseModel()
        nm.add(GaussianNoise(0.1))
        od.add_noise(nm)
        data_2 = copy.deepcopy(od.data)
        assert not np.array_equal(data_1, data_2)

    def test_alpha_noise(self, dqd_csd):
        od = EnergyDiagram(dqd_csd)
        data_1 = copy.deepcopy(od.data)
        nm = NoiseModel()
        nm.add(AlphaNoise(0.1, 1000))
        od.add_noise(nm)
        data_2 = copy.deepcopy(od.data)
        assert not np.array_equal(data_1, data_2)

    def test_remove_noise(self, dqd_csd):
        ed = EnergyDiagram(dqd_csd)
        data_1 = copy.deepcopy(ed.data)
        gn = GaussianNoise(0.1)
        ed.add_noise(gn)
        data_2 = copy.deepcopy(ed.data)
        ed.remove_noise()
        assert np.array_equal(data_1, ed.data)
        assert not np.array_equal(data_2, ed.data)
        assert not hasattr(ed, "noise_model")

    def test_process_data_cim_reduced(self, dqd_csd):
        """
        Test _process_data for the solver with reduced Fock space.
        """
        dqd_csd.solver = "cim"
        occu = OccupancyDiagram(dqd_csd)
        occu._process_data()

        # Ensure that raw_data was reshaped and masked
        assert occu.raw_data.shape == (10, 10, 4)

        assert (occu.raw_data != jnp.inf).all()

    def test_process_data_cim(self, dqd_csd_cim_rfs):
        """
        Ensure _process_data masks all states when all Fock states are invalid.
        """
        occu = OccupancyDiagram(dqd_csd_cim_rfs)

        occu._process_data()

        assert jnp.any(occu.raw_data == jnp.inf), (
            "Certain should be masked to infinity."
        )

    def test_process_data_all_negative_fock_states(self):
        """
        Ensure _process_data masks all states when all Fock states are negative.
        """
        # Create a DQD_CSD object with all negative Fock states
        fock_states = -jnp.ones((9, 2, 25))
        data = jnp.ones((5, 5, 9))
        masked_data = reshape_and_mask_data(data, fock_states)

        # Assert all raw_data is masked (set to infinity)
        assert jnp.all(masked_data == jnp.inf), (
            "All states should be masked to infinity for negative Fock states."
        )

    def test_process_data_all_positive_fock_states(self):
        """
        Ensure _process_data masks all states when all Fock states are negative.
        """
        # Create a DQD_CSD object with all negative Fock states
        fock_states = jnp.ones((9, 2, 25))
        data = jnp.ones((5, 5, 9))
        masked_data = reshape_and_mask_data(data, fock_states)

        # Assert all raw_data is masked (set to infinity)
        assert jnp.all(masked_data != jnp.inf), (
            "All states should be not masked to infinity for positive Fock states."
        )

    def test_x_y_axis_labels(self, dqd_csd):
        ed = EnergyDiagram(dqd_csd)
        fig1 = ed.plot(x_axis="gate1", y_axis="gate2")
        fig2 = ed.plot(x_axis="gate2", y_axis="gate1")
        assert fig1.get_axes()[0].get_xlabel() == "Gate voltage at gate1 (V)"
        assert fig1.get_axes()[0].get_ylabel() == "Gate voltage at gate2 (V)"
        assert fig2.get_axes()[0].get_xlabel() == "Gate voltage at gate2 (V)"
        assert fig2.get_axes()[0].get_ylabel() == "Gate voltage at gate1 (V)"


## CHARGE STABILITY DIAGRAM PLOTTING TESTS
class TestAbstractedMethods:
    def test_check_plot_gates_2D(self, tqd_csd):
        gate_tuple = tqd_csd._check_plot_gates_2D(plot_gates=["gate_1", "gate_3"])
        assert "gate_1" in gate_tuple[0]
        assert "gate_3" in gate_tuple[1]
        assert "Virtual" not in gate_tuple[0]
        assert "Virtual" not in gate_tuple[1]
        # 0->0.1 for x and 0->0.15 for y
        np.testing.assert_array_equal(gate_tuple[2], jnp.linspace(0, 0.1, 10))
        np.testing.assert_array_equal(gate_tuple[3], jnp.linspace(0, 0.15, 10))

        with pytest.raises(
            ValueError, match="Only two gates can be plotted at any time"
        ):
            _ = tqd_csd._check_plot_gates_2D(plot_gates=["gate_1", "gate_3", "gate_2"])

        with pytest.raises(KeyError, match="Not all gates selected for"):
            _ = tqd_csd._check_plot_gates_2D(plot_gates=["waa", "waa2"])

        _ = tqd_csd._check_plot_gates_2D(plot_gates=None)
        assert "gate_1" in gate_tuple[0]
        assert "gate_3" in gate_tuple[1]

    def test_check_plot_gates_2D_virtual(self, tqd_csd):
        tqd_csd.virtual_gates = True

        gate_tuple = tqd_csd._check_plot_gates_2D(plot_gates=["gate_1", "gate_3"])
        assert "gate_1" in gate_tuple[0]
        assert "gate_3" in gate_tuple[1]
        assert "Virtual" in gate_tuple[0]
        assert "Virtual" in gate_tuple[1]

    def test_plot(self, tqd_csd):
        fig, axs = tqd_csd._plot(
            jnp.min(tqd_csd.raw_data, axis=-1),
            "viridis",
            figsize=(4, 3),
            plot_gates=["gate_1", "gate_3"],
        )
        assert isinstance(fig, MatplotlibFigure)
        assert isinstance(axs, MatplotlibAxes)
        assert axs.get_xlabel() == "Gate voltage at gate_1 (V)"
        assert axs.get_ylabel() == "Gate voltage at gate_3 (V)"

    def test_check_colours_1D(self, tqd_csd):
        cmap = tqd_csd._check_colours_1D(cmap="#ff4342")
        assert cmap == "#ff4342"

        cmap = tqd_csd._check_colours_1D(cmap="viridis")
        assert cmap == "#440154"

    def test_get_slice(self, tqd_csd_3s):
        # shapes of gates are 5, 10, 15

        ener = tqd_csd_3s.get_energy_diagram()

        data = ener.get_slice({"gate_1": 0})

        assert data.shape == (10, 15)
        assert data.ndim == 2

        data2 = ener.get_slice({"gate_1": 0, "gate_2": 0.1})

        assert data2.shape == (15,)
        assert data2.ndim == 1

        data3 = ener.get_slice({"gate_1": 0.00001})
        assert data3.shape == (10, 15)
        np.testing.assert_array_equal(data, data3)

        data4 = ener.get_slice({"gate_3": 0.2})
        assert data4.shape == (5, 10)
        assert data4.ndim == 2

        data5 = ener.get_slice({"gate_1": 0.1})
        assert data3.shape == (10, 15)
        # definitely not the same :)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(data, data5)

    def test_get_slice_plot(self, tqd_csd_3s):
        ener = tqd_csd_3s.get_energy_diagram()

        data, pg = ener._get_slice_plot({"gate_1": 0})
        assert data.shape == (10, 15)
        assert data.ndim == 2
        assert pg == ["gate_2", "gate_3"]

        data2, pg2 = ener._get_slice_plot({"gate_1": 0, "gate_2": 0.1})
        assert data2.shape == (15,)
        assert data2.ndim == 1
        assert pg2 == ["gate_3"]

    def test_get_slice_plot_with_axes_energy(self, tqd_csd_3s):
        """
        Test specifying x_axis and y_axis to check transposition behavior.
        """
        ener = tqd_csd_3s.get_energy_diagram()

        data, pg = ener._get_slice_plot({"gate_1": 0}, x_axis="gate_2", y_axis="gate_3")
        assert data.shape == (
            10,
            15,
        )
        assert data.ndim == 2
        assert pg == ["gate_2", "gate_3"]

        data, pg = ener._get_slice_plot({"gate_2": 0}, x_axis="gate_3", y_axis="gate_1")
        assert data.shape == (
            15,
            5,
        )
        assert data.ndim == 2
        assert pg == ["gate_3", "gate_1"]

    def test_get_slice_plot_with_axes_occupancy(self, tqd_csd_3s):
        """
        Test specifying x_axis and y_axis to check transposition behavior.
        """
        ener = tqd_csd_3s.get_occupancy_diagram()

        data, pg = ener._get_slice_plot({"gate_1": 0}, x_axis="gate_2", y_axis="gate_3")
        assert data.shape == (
            10,
            15,
        )
        assert data.ndim == 2
        assert pg == ["gate_2", "gate_3"]

        data, pg = ener._get_slice_plot({"gate_2": 0}, x_axis="gate_3", y_axis="gate_1")
        assert data.shape == (
            15,
            5,
        )
        assert data.ndim == 2
        assert pg == ["gate_3", "gate_1"]


# My first attempt at pytest parametrize - I hope this is how it's done
@pytest.mark.parametrize(
    "slice, x_axis, y_axis, expected_exception, message",
    [
        # missing axis
        (
            {"gate_1": 0},
            "gate_2",
            None,
            ValueError,
            "Both x_axis and y_axis must be provided or neither.",
        ),
        (
            {"gate_1": 0},
            None,
            "gate_2",
            ValueError,
            "Both x_axis and y_axis must be provided or neither.",
        ),
        # same axis
        (
            {"gate_1": 0},
            "gate_2",
            "gate_2",
            ValueError,
            "x_axis and y_axis must be different.",
        ),
        # invalid axis
        (
            {"gate_1": 0, "gate_2": 0.1},
            "gate_1",
            "gate_3",
            ValueError,
            "Cannot supply x_axis and y_axis when less than 2 gates are swept.",
        ),
        (
            {"gate_1": 0, "gate_2": 0.1},
            "gate_3",
            "gate_1",
            ValueError,
            "Cannot supply x_axis and y_axis when less than 2 gates are swept.",
        ),
        # 1d slice
        (
            {"gate_1": 0, "gate_2": 0.1},
            "gate_2",
            "gate_2",
            ValueError,
            "x_axis and y_axis must be different.",
        ),
        (
            {"gate_1": 0},
            "gate_2",
            "gate_2",
            ValueError,
            "x_axis and y_axis must be different.",
        ),
        # correct!
        ({"gate_1": 0}, "gate_2", "gate_3", None, None),
    ],
)
def test_get_slice_plot_validation(
    tqd_csd_3s, slice, x_axis, y_axis, expected_exception, message
):
    ener = tqd_csd_3s.get_energy_diagram()

    if expected_exception:
        with pytest.raises(expected_exception, match=message):
            ener._get_slice_plot(slice, x_axis, y_axis)
    else:
        data, plot_gates = ener._get_slice_plot(slice, x_axis, y_axis)
        assert data is not None
        assert plot_gates is not None


# IO TESTS
class TestIO:
    def test_to_file_wrong_format(self, tqd_csd):
        with pytest.raises(ValueError):
            filename = pathlib.Path("test.toml")
            energy = EnergyDiagram(tqd_csd)
            energy.to_file(filename)
            assert filename.exists()

    def test_to_file(self, tqd_csd):
        filename = pathlib.Path("test.hdf5")
        energy = EnergyDiagram(tqd_csd)
        energy.to_file(filename)
        assert filename.exists()
        filename.unlink()

        occupancy = OccupancyDiagram(tqd_csd)
        occupancy.to_file(filename)
        assert filename.exists()
        filename.unlink()


@pytest.fixture
def rf_csd(dqd_csd):
    dqd_csd.solver = "rf"
    rf = RFDiagram(dqd_csd, temperature=0.3, sensor_gate="gate1")
    return rf


class TestRFDiagram:
    def test_init(self, rf_csd):
        assert hasattr(rf_csd, "solver")
        assert hasattr(rf_csd, "temperature")
        assert hasattr(rf_csd, "sensor_gate")
        assert hasattr(rf_csd, "raw_data")
        assert hasattr(rf_csd, "virtual_gates")
        assert hasattr(rf_csd, "cmap_label")
        assert rf_csd.cmap_label == r"$\Delta C$ (1/V)"
        assert hasattr(rf_csd, "data")

    def test_init_from_csd(self, rf_csd):
        assert hasattr(rf_csd, "solver")
        assert hasattr(rf_csd, "temperature")
        assert hasattr(rf_csd, "sensor_gate")
        assert hasattr(rf_csd, "raw_data")
        assert hasattr(rf_csd, "virtual_gates")
        assert hasattr(rf_csd, "cmap_label")
        assert rf_csd.cmap_label == r"$\Delta C$ (1/V)"
        assert hasattr(rf_csd, "data")

    def test_repr(self, rf_csd):
        representation = rf_csd.__repr__()
        assert "RFDiagram" in representation
        assert "Sweeping:" in representation
        assert "gate2: 0.000 to 0.200 "

    def test_add_gaussian_noise(self, rf_csd):
        data_1 = copy.deepcopy(rf_csd.data)
        gn = GaussianNoise(0.1)
        rf_csd.add_noise(gn)
        data_2 = copy.deepcopy(rf_csd.data)
        assert not np.array_equal(data_1, data_2)

    def test_add_noise_model(self, rf_csd, dqd_csd):
        data_1 = copy.deepcopy(rf_csd.data)
        nm = NoiseModel()
        nm.add(GaussianNoise(0.1))
        rf_csd.add_noise(nm)

        rf = RFDiagram(dqd_csd, temperature=0.3, sensor_gate="gate1")
        data_1 = copy.deepcopy(rf.data)
        nm = NoiseModel()
        nm.add(GaussianNoise(0.1))
        rf.add_noise(nm)
        data_2 = copy.deepcopy(rf.data)
        assert not np.array_equal(data_1, data_2)

    def test_alpha_noise(self, rf_csd):
        # because it's not a real RFD, we have to fake the data to be in the right shape
        rf_csd.data = jnp.zeros((100, 100))
        data_1 = copy.deepcopy(rf_csd.data)
        nm = NoiseModel()
        nm.add(AlphaNoise(0.1, 1000))
        rf_csd.add_noise(nm)
        data_2 = copy.deepcopy(rf_csd.data)
        assert not np.array_equal(data_1, data_2)

    def test_remove_noise(self, rf_csd):
        data_1 = copy.deepcopy(rf_csd.data)
        gn = GaussianNoise(0.1)
        rf_csd.add_noise(gn)
        data_2 = copy.deepcopy(rf_csd.data)
        rf_csd.remove_noise()
        np.testing.assert_array_almost_equal(data_1, rf_csd.data, decimal=8)
        assert not np.array_equal(data_2, rf_csd.data)
        assert not hasattr(rf_csd, "noise_model")
