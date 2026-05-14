import jax.numpy as jnp
import numpy as np
import pytest

from rfsquad.core import Device
from rfsquad.solvers import CISolver
from rfsquad.utils.utils_ci import (
    _create_fock_full,
    _create_pairwise,
    _create_single_fock_reduced,
    _create_virtual,
    _create_virtual_gate_matrix,
    _create_virtual_helper,
)


@pytest.fixture
def device_instance():
    dev = Device(
        C_dd=[[100.0, -10.0], [-10.0, 100.0]],
        C_dg=[[10.0, 2.0], [2.0, 10.0]],
    )
    return dev


@pytest.fixture
def tqd_device_instance():
    dev = Device(
        C_dd=[[100.0, -10.0, -1.0], [-10.0, 100.0, -1.0], [-1.0, -1.0, 100.0]],
        C_dg=[[10.0, 2.0, 0.1], [2.0, 10.0, 0.1], [0.1, 0.1, 10.0]],
    )
    return dev


@pytest.fixture
def two_dots_three_gate_device():
    dev = Device(
        C_dd=jnp.array([[200, -20], [-20, 200]]),
        C_dg=jnp.array([[2.4, 1, 0.1], [0.5, 3.85, 0.1]]),
    )
    return dev


@pytest.fixture
def solver_instance(device_instance):
    return CISolver(device_instance)


class TestSolverFunctions:
    ## FOCK STATES

    def test_create_fock_full(self):
        max_occu = 1
        num_dots = 2

        sweep = {"gate_1": jnp.linspace(0, 0.2, 20), "gate_3": jnp.linspace(0, 0.3, 20)}
        threshold = {"gate_1": 0, "gate_2": 0, "gate_3": 0}
        gate = {"gate_1": 0, "gate_2": 0.5, "gate_3": 0}
        pairwise, _ = _create_pairwise(sweep, threshold, gate)

        fock = _create_fock_full(max_occu, num_dots, pairwise)
        exact_focks = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

        # Extra dimension here due to gate voltage configuration.
        exact_focks_broadcast = np.repeat(
            exact_focks[:, :, jnp.newaxis], len(pairwise), axis=2
        )
        np.testing.assert_array_equal(fock, exact_focks_broadcast)
        assert fock.shape == ((max_occu + 1) ** num_dots, num_dots, len(pairwise))

    ## PAIRWISE

    def test_create_1D_pairwise(self):
        sweep = {"gate_1": jnp.linspace(0, 0.2, 20)}
        threshold = {"gate_1": 0, "gate_2": 0}
        gate = {"gate_1": 0, "gate_2": 0.5}

        pairwise, axes_for_csd = _create_pairwise(sweep, threshold, gate)
        assert pairwise.shape == (20, 2)
        assert len(axes_for_csd) == 1
        assert "gate_2" not in axes_for_csd

        np.testing.assert_array_almost_equal(
            pairwise[:, 0], np.linspace(0, 0.2, 20, dtype=np.float32)
        )

        # all 0.5 for the second gate
        np.testing.assert_array_equal(
            pairwise[:, 1], np.array([0.5] * 20, dtype=np.float32)
        )

    def test_create_pairwise(self):
        sweep = {"gate_1": jnp.linspace(0, 0.2, 20), "gate_2": jnp.linspace(0, 0.3, 20)}
        threshold = {"gate_1": 0, "gate_2": 0}
        gate = {"gate_1": 0, "gate_2": 0}
        pairwise, axes_for_csd = _create_pairwise(sweep, threshold, gate)
        assert pairwise.shape == (20 * 20, 2)
        assert len(axes_for_csd) == 2

        np.testing.assert_array_equal(axes_for_csd["gate_1"], sweep["gate_1"])

        np.testing.assert_array_equal(pairwise[0], jnp.array([0.0, 0.0]))
        np.testing.assert_array_equal(pairwise[-1], jnp.array([0.2, 0.3]))

    def test_create_pairwise_more_gates(self):
        sweep = {"gate_1": jnp.linspace(0, 0.2, 20), "gate_3": jnp.linspace(0, 0.3, 20)}
        threshold = {"gate_1": 0, "gate_2": 0, "gate_3": 0}
        gate = {"gate_1": 0, "gate_2": 0.5, "gate_3": 0}
        pairwise, axes_for_csd = _create_pairwise(sweep, threshold, gate)
        # still only 400 points in sweep bc only 2 gates swept
        assert pairwise.shape == (20 * 20, 3)
        assert len(axes_for_csd) == 2
        assert "gate_2" not in axes_for_csd

        np.testing.assert_array_equal(axes_for_csd["gate_1"], sweep["gate_1"])

        # assert gate_2 is 0.5
        assert np.all(pairwise[:, 1] == 0.5)

    def test_create_pairwise_3_gates_swept(self):
        sweep = {
            "gate_1": jnp.linspace(0, 0.2, 20),
            "gate_2": jnp.linspace(0, 0.3, 20),
            "gate_3": jnp.linspace(0, 0.3, 20),
        }
        threshold = {"gate_1": 0, "gate_2": 0, "gate_3": 0}
        gate = {"gate_1": 0, "gate_2": 0.5, "gate_3": 0}
        pairwise, axes_for_csd = _create_pairwise(sweep, threshold, gate)
        assert pairwise.shape == (20 * 20 * 20, 3)
        assert len(axes_for_csd) == 3
        assert "gate_2" in axes_for_csd

    def test_create_pairwise_with_threshold(self):
        sweep = {"gate_1": jnp.linspace(0, 0.2, 20), "gate_2": jnp.linspace(0, 0.3, 20)}
        threshold1 = {"gate_1": 0, "gate_2": 0}
        threshold2 = {"gate_1": 0, "gate_2": 1}
        gate = {"gate_1": 0, "gate_2": 0}

        pairwise1, axes_for_csd1 = _create_pairwise(sweep, threshold1, gate)
        pairwise2, axes_for_csd2 = _create_pairwise(sweep, threshold2, gate)

        np.testing.assert_array_equal(pairwise1[:, 0], pairwise2[:, 0])
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(pairwise1[:, 1], pairwise2[:, 1])

        # pairwise changes
        assert pairwise1[0, 1] == 0
        assert pairwise2[0, 1] == -1

        # axes_for_csd does not
        assert axes_for_csd1["gate_2"][0] == 0
        assert axes_for_csd2["gate_2"][0] == 0

    ## VIRTUAL GATES

    def test_create_virtual_matrix(self, device_instance):
        virtual_gate_matrix = _create_virtual_gate_matrix(
            device_instance.C_dd_inv, device_instance.C_dg
        )

        # I am checking the shape
        expected_shape = (2, 2)
        assert virtual_gate_matrix.shape == expected_shape, (
            f"Expected shape {expected_shape}, but got {virtual_gate_matrix.shape}"
        )

        # I am checking the values
        np.testing.assert_almost_equal(
            virtual_gate_matrix,
            np.array([[1.0, -0.2941176], [-0.2941177, 1.0]], dtype=jnp.float32),
        )

    def test_create_virtual_matrix_three(self, tqd_device_instance):
        virtual_gate_matrix = _create_virtual_gate_matrix(
            tqd_device_instance.C_dd_inv, tqd_device_instance.C_dg
        )

        # I am checking the shape
        expected_shape = (3, 3)
        assert virtual_gate_matrix.shape == expected_shape, (
            f"Expected shape {expected_shape}, but got {virtual_gate_matrix.shape}"
        )

        # I am checking the values
        np.testing.assert_almost_equal(
            virtual_gate_matrix,
            np.array(
                [
                    [1.0, -0.29394, -0.0166639],
                    [-0.29394, 1.0, -0.0166638],
                    [-0.0164712, -0.016471, 1.0],
                ],
                dtype=jnp.float32,
            ),
            decimal=5,
        )

    def test_create_virtual_matrix_more_gates(self, two_dots_three_gate_device):
        virtual_gate_matrix = _create_virtual_gate_matrix(
            two_dots_three_gate_device.C_dd_inv, two_dots_three_gate_device.C_dg
        )

        # I am checking the shape
        expected_shape = (3, 3)
        assert virtual_gate_matrix.shape == expected_shape, (
            f"Expected shape {expected_shape}, but got {virtual_gate_matrix.shape}"
        )

        # I am checking the values
        np.testing.assert_almost_equal(
            virtual_gate_matrix,
            np.array(
                [
                    [1.0, -0.56545436, 0],
                    [-0.18813574, 1.0, 0],
                    [0.02851878, 0.00330041, 0.0],
                ],
                dtype=jnp.float32,
            ),
        )

    def test_create_virtual_helper(self):
        sweep = {"gate_1": jnp.linspace(0, 0.2, 20), "gate_2": jnp.linspace(0, 0.3, 20)}
        threshold = {"gate_1": 0, "gate_2": 0}
        gate = {"gate_1": 0, "gate_2": 0}
        pairwise, _ = _create_pairwise(sweep, threshold, gate)

        # because i'm using identity matrix, the gate_voltages should be the same as pairwise
        virtual_gate_matrix = jnp.array([[1, 0], [0, 1]])
        gate_voltages = _create_virtual_helper(pairwise, virtual_gate_matrix)
        np.testing.assert_array_equal(gate_voltages.T, pairwise)

        # if we use something else, the arrays should be different and raise the error
        virtual_gate_matrix = jnp.array([[1, 0], [0, 2]])
        gate_voltages2 = _create_virtual_helper(pairwise, virtual_gate_matrix)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(gate_voltages2.T, pairwise)

    def test_create_virtual(self):
        sweep = {"gate_1": jnp.linspace(0, 0.2, 20), "gate_2": jnp.linspace(0, 0.3, 20)}
        threshold = {"gate_1": 0, "gate_2": 0}
        gate = {"gate_1": 0, "gate_2": 0}
        pairwise, axes_for_csd = _create_pairwise(sweep, threshold, gate)

        # identity matrix
        virtual_gate_matrix_id = jnp.array([[1, 0], [0, 1]])
        virtual_id, axes_for_csd_virutal_id = _create_virtual(
            sweep, threshold, gate, virtual_gate_matrix_id
        )

        # actual matrix
        virtual_gate_matrix = jnp.array([[1, 0], [0, 2]])
        virtual, axes_for_csd_virtual = _create_virtual(
            sweep, threshold, gate, virtual_gate_matrix
        )

        # test that pairwise is calculated correctly
        np.testing.assert_array_equal(virtual_id, pairwise)
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(virtual, pairwise)

        # test that axes_for_csd is calculated correctly
        np.testing.assert_array_equal(
            axes_for_csd_virutal_id["gate_1"], axes_for_csd["gate_1"]
        )
        np.testing.assert_array_equal(
            axes_for_csd_virutal_id["gate_2"], axes_for_csd["gate_2"]
        )

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(
                axes_for_csd_virtual["gate_2"], axes_for_csd["gate_2"]
            )

    def test_fock_state_output_shape_and_values(self):
        C_dd_inv = jnp.array([[1.0, 0.0], [0.0, 1.0]])
        C_dg = jnp.array([[0.5, 0.1], [0.2, 0.3]])
        voltage_config = jnp.array([0.2, 0.3])
        k = 1
        fock_states = _create_single_fock_reduced(
            C_dd_inv, C_dg, voltage_config, max_fock_depth=k
        )

        expected_shape = ((2 * k + 1) ** 2, voltage_config.shape[0])
        assert fock_states.shape == expected_shape, (
            f"Expected shape {expected_shape}, got {fock_states.shape}"
        )

        # Example expected values for validation
        expected_values = jnp.array(
            [
                [-1, -1],
                [-1, 0],
                [-1, 1],
                [0, -1],
                [0, 0],
                [0, 1],
                [1, -1],
                [1, 0],
                [1, 1],
            ]
        )

        assert jnp.allclose(fock_states, expected_values), (
            f"Expected values {expected_values}, got {fock_states}"
        )

        for state in fock_states:
            assert jnp.all(jnp.isfinite(state)), (
                f"State {state} contains non-finite values (NaN or Inf)"
            )


class TestValidateSolver:
    def test_max_charge_carriers_positive(self, solver_instance):
        # Test with a non-positive max_charge_carriers
        solver_instance.max_charge_carriers = 0
        with pytest.raises(
            ValueError, match="Max charge carriers must be greater than 0"
        ):
            solver_instance._validate_solver()

        solver_instance.max_charge_carriers = -1
        with pytest.raises(
            ValueError, match="Max charge carriers must be greater than 0"
        ):
            solver_instance._validate_solver()

    def test_max_charge_carriers_specified_when_not_reduced(self, solver_instance):
        # Test that max_charge_carriers must be specified if reduce_fock_space is False
        solver_instance.max_charge_carriers = None
        solver_instance.reduce_fock_space = False
        with pytest.raises(
            ValueError,
            match="If reduce_fock_space is False, max_charge_carriers must be specified",
        ):
            solver_instance._validate_solver()

    def test_valid_configuration(self, solver_instance):
        # Test with a valid configuration to ensure no error is raised
        solver_instance.max_charge_carriers = 2
        solver_instance.reduce_fock_space = False

        try:
            solver_instance._validate_solver()
        except ValueError as e:
            pytest.fail(f"Unexpected ValueError raised: {e}")
