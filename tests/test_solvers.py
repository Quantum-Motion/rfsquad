import pathlib
from typing import Any

import jax.numpy as jnp
import numpy as np
import pytest

from rfsquad.core import Device
from rfsquad.csd import ChargeStabilityDiagram, RFDiagram
from rfsquad.solvers import CISolver, RFSolver
from rfsquad.utils.utils_ci import (
    _create_fock_full,
    _create_pairwise,
)


@pytest.fixture
def c_dd():
    yield jnp.array([[100.0, -10.0], [-10.0, 100.0]])


@pytest.fixture
def c_dg():
    yield jnp.array([[10.0, 2.0], [2.0, 10.0]])


@pytest.fixture
def dev(c_dd, c_dg):
    yield Device(C_dd=c_dd, C_dg=c_dg)


@pytest.fixture
def dev_with_vgm(c_dd, c_dg):
    yield Device(
        c_dd,
        c_dg,
        virtual_gate_matrix=jnp.array([[1.0, 0.0], [0.0, 1.0]]),
    )


@pytest.fixture
def sweep():
    yield {
        "gate_1": np.linspace(0, 0.2, 20),
        "gate_2": np.linspace(0, 0.3, 20),
    }


@pytest.fixture
def dev_23():
    C_dd = jnp.array([[20.0, -4.0], [-4.0, 20.0]])
    C_dg = jnp.array([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]])
    dev = Device(C_dd=C_dd, C_dg=C_dg)
    return dev


@pytest.fixture
def valid_device_tqd():
    C_dd = jnp.array([[20.0, -4.0, -2.0], [-4.0, 20.0, -4.0], [-2.0, -4.0, 20.0]])
    C_dg = jnp.array([[2.0, 0.5, 0.1], [0.1, 2.5, 0.5], [0.1, 0.5, 2.0]])
    dev = Device(C_dd=C_dd, C_dg=C_dg)
    return dev


class TestCISolverInit:
    """
    this just tests the class instantiates correctly and returns the right type of values - it doesn't actually test the correctness of the solver
    """

    def test_init(self, dev: Device):
        solver = CISolver(dev)
        assert hasattr(solver, "device")

    def test_repr(self, dev: Device):
        solver = CISolver(dev)
        assert isinstance(solver.__repr__(), str)
        representation = solver.__repr__()
        assert "Device instantiated with:" in representation
        assert "CISolver using:" in representation
        assert "Solver parameters:" in representation

    @pytest.mark.parametrize("fname", ["test.toml", "test.json", "test.h5"])
    def test_to_from_file(self, dev: Device, fname):
        filename = pathlib.Path(fname)
        solver = CISolver(dev)
        solver.to_file(filename)
        assert filename.exists()

        solver2 = CISolver.from_file(filename)
        np.testing.assert_allclose(solver.device.C_dd, solver2.device.C_dd, rtol=1e-5)
        assert solver.max_charge_carriers == solver2.max_charge_carriers
        filename.unlink()

    def test_run(self, dev: Device, sweep: dict[str, Any]):
        solver = CISolver(dev)
        csd = solver.run(sweep, update_solver={"max_charge_carriers": 1})
        assert isinstance(csd, ChargeStabilityDiagram)

    def test_validate_solver(self, dev: Device):
        solver = CISolver(dev)
        # no errors raised
        solver._validate_solver()

        solver.max_charge_carriers = -1
        with pytest.raises(ValueError, match="must be greater than 0"):
            solver._validate_solver()

        solver.max_charge_carriers = 1
        solver.max_charge_carriers = None  # type: ignore
        with pytest.raises(ValueError, match="If reduce_fock_space is False"):
            solver._validate_solver()

    def test_update_solver(self, dev: Device):
        solver = CISolver(dev)
        solver._update_solver({"max_charge_carriers": 4})
        assert solver.max_charge_carriers == 4

        with pytest.raises(KeyError, match="Invalid solver parameter: extra_param"):
            solver._update_solver({"extra_param": 4})

    def test_virtual_gates_matrix_supplied(self, dev: Device, sweep: dict[str, Any]):
        solver = CISolver(dev)
        csd1 = solver.run(sweep, update_solver={"max_charge_carriers": 1})

        dev.virtual_gate_matrix = dev.lever_arm_matrix
        solver = CISolver(dev)
        csd2 = solver.run(
            sweep, update_solver={"max_charge_carriers": 1, "use_virtual_gates": True}
        )

        with pytest.raises(AssertionError):
            np.testing.assert_array_almost_equal(csd1.raw_data, csd2.raw_data)

    def test_run_more_gates(self, dev_23: Device, sweep: dict[str, Any]):
        solver = CISolver(dev_23)
        csd = solver.run(sweep, update_solver={"max_charge_carriers": 1})
        assert isinstance(csd, ChargeStabilityDiagram)


class TestCISolverCorrectness:
    """
    This class tests the correctness of the solver, i.e. is the maths consistently correct
    """

    pass


@pytest.fixture
def differences():
    sweep = {"gate_1": jnp.linspace(0, 0.2, 20), "gate_2": jnp.linspace(0, 0.3, 20)}
    threshold = {"gate_1": 0, "gate_2": 0}
    gate = {"gate_1": 0, "gate_2": 0}
    pairwise, _ = _create_pairwise(sweep, threshold, gate)

    fock = _create_fock_full(1, 2, pairwise)
    differences = fock[:, None, :, :] - fock[None, :, :, :]
    return differences


class TestRFSolverInit:
    def test_init(self, dev: Device):
        solver = RFSolver(dev)
        assert hasattr(solver, "device")

    def test_repr(self, dev: Device):
        solver = RFSolver(dev)
        assert isinstance(solver.__repr__(), str)
        representation = solver.__repr__()
        assert "Device instantiated with:" in representation
        assert "RFSolver using:" in representation
        assert "Solver parameters:" in representation

    @pytest.mark.parametrize("fname", ["test.toml", "test.json", "test.h5"])
    def test_to_from_file(self, dev: Device, fname):
        filename = pathlib.Path(fname)
        solver = RFSolver(dev)
        solver.to_file(filename)
        assert filename.exists()

        solver2 = RFSolver.from_file(filename)
        np.testing.assert_allclose(solver.device.C_dd, solver2.device.C_dd, rtol=1e-5)
        assert solver.max_charge_carriers == solver2.max_charge_carriers
        filename.unlink()

    def test_run(self, dev: Device, sweep: dict[str, Any]):
        solver = RFSolver(dev)
        csd = solver.run(
            sweep,
            update_solver={"max_charge_carriers": 1},
            sensor_gate="gate_1",
            temperature=1,
        )
        assert isinstance(csd, RFDiagram)

    def test_run_rfs(self, dev: Device, sweep: dict[str, Any]):
        solver = RFSolver(dev)
        csd = solver.run(
            sweep,
            update_solver={"reduce_fock_space": True},
            sensor_gate="gate_1",
            temperature=1,
        )
        assert isinstance(csd, RFDiagram)

    def test_validate_solver(self, dev: Device):
        solver = RFSolver(dev)
        # no errors raised
        solver._validate_solver()

        solver.max_charge_carriers = -1
        with pytest.raises(ValueError, match="must be greater than 0"):
            solver._validate_solver()

        solver.max_charge_carriers = 1
        solver.max_charge_carriers = None  # type: ignore
        with pytest.raises(ValueError, match="If reduce_fock_space is False"):
            solver._validate_solver()

    def test_update_solver(self, dev: Device):
        solver = RFSolver(dev)
        solver._update_solver({"max_charge_carriers": 4})
        assert solver.max_charge_carriers == 4

        with pytest.raises(KeyError, match="Invalid solver parameter: extra_param"):
            solver._update_solver({"extra_param": 4})

    def test_run_more_gates(self, dev_23: Device, sweep: dict[str, Any]):
        solver = RFSolver(dev_23)
        csd = solver.run(
            sweep,
            update_solver={"max_charge_carriers": 1},
            sensor_gate="gate_1",
            temperature=1,
        )
        assert isinstance(csd, RFDiagram)
