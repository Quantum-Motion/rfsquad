from contextlib import nullcontext

import jax.numpy as jnp
import numpy as np
import pytest

import rfsquad.utils.validation as dtv
from rfsquad.core import Device


@pytest.fixture
def device():
    c_dd = jnp.array([[100.0, -10.0], [-10.0, 100.0]])
    c_dg = jnp.array([[10.0, 2.0], [2.0, 10.0]])
    dev = Device(c_dd, c_dg, gate_names=["gate1", "gate2"])
    dev.add_static_voltages(gate={"gate1": 0.1, "gate2": 0.2})
    return dev


def test_validate_capacitance_matrix_shapes_valid():
    """Test that no error is raised for valid input."""
    C_dd = np.zeros((3, 3))
    C_dg = np.zeros((3, 4))
    num_dots = 3
    num_gates = 4
    try:
        dtv.validate_capacitance_matrix_shapes(C_dd, C_dg, num_dots, num_gates)
    except ValueError:
        pytest.fail("Unexpected ValueError for valid inputs.")


def test_validate_capacitance_matrix_shapes_invalid_dd_shape():
    """Test that an error is raised for an invalid C_dd shape."""
    C_dd = np.zeros((2, 3))  # Invalid shape
    C_dg = np.zeros((3, 4))
    num_dots = 3
    num_gates = 4
    with pytest.raises(
        ValueError,
        match="Dot-dot capacitance matrix must be of a num_dots x num_dots shape.",
    ):
        dtv.validate_capacitance_matrix_shapes(C_dd, C_dg, num_dots, num_gates)


def test_validate_capacitance_matrix_shapes_invalid_dg_rows():
    """Test that an error is raised for invalid C_dg row count."""
    C_dd = np.zeros((3, 3))
    C_dg = np.zeros((2, 4))  # Invalid row count
    num_dots = 3
    num_gates = 4
    with pytest.raises(
        ValueError, match=f"Dot-gate capacitance matrix must have {num_dots} rows."
    ):
        dtv.validate_capacitance_matrix_shapes(C_dd, C_dg, num_dots, num_gates)


def test_validate_capacitance_matrix_shapes_invalid_gate_count():
    """Test that an error is raised for insufficient number of gates."""
    C_dd = np.zeros((3, 3))
    C_dg = np.zeros((3, 2))  # Insufficient gates
    num_dots = 3
    num_gates = 2
    with pytest.raises(
        ValueError,
        match=f"Dot-gate capacitance matrix must have {num_dots} or more columns .*",
    ):
        dtv.validate_capacitance_matrix_shapes(C_dd, C_dg, num_dots, num_gates)


def test_validate_capacitance_matrix_elements_valid():
    """Test that no error is raised for a valid C_dd matrix."""
    C_dd = jnp.array(
        [
            [2, -1, -1],
            [-1, 2, -1],
            [-1, -1, 2],
        ]
    )
    try:
        dtv.validate_capacitance_matrix_elements(C_dd)
    except ValueError:
        pytest.fail("Unexpected ValueError for valid C_dd matrix.")


def test_validate_capacitance_matrix_elements_invalid_diagonal():
    """Test that an error is raised for a matrix with non-positive diagonal elements."""
    C_dd = jnp.array(
        [
            [0, -1, -1],  # Diagonal element is not positive
            [-1, 2, -1],
            [-1, -1, 2],
        ]
    )
    with pytest.raises(
        ValueError,
        match="Dot-dot capacitance matrix diagonal elements must all be positive.",
    ):
        dtv.validate_capacitance_matrix_elements(C_dd)


def test_validate_capacitance_matrix_elements_invalid_off_diagonal():
    """Test that an error is raised for a matrix with non-negative off-diagonal elements."""
    C_dd = jnp.array(
        [
            [2, 1, -1],  # Off-diagonal element is not negative
            [1, 2, -1],
            [-1, -1, 2],
        ]
    )
    with pytest.raises(
        ValueError,
        match="Dot-dot capacitance matrix off-diagonal elements must all be negative.",
    ):
        dtv.validate_capacitance_matrix_elements(C_dd)


def test_validate_capacitance_matrix_elements_not_symmetric():
    """Test that an error is raised for a non-symmetric matrix."""
    C_dd = jnp.array(
        [
            [2, -1, -1],
            [-2, 2, -1],  # Asymmetry between (1, 0) and (0, 1)
            [-1, -1, 2],
        ]
    )
    with pytest.raises(
        ValueError, match="Dot-dot capacitance matrix must be symmetric."
    ):
        dtv.validate_capacitance_matrix_elements(C_dd)


@pytest.mark.parametrize(
    "gate_names, num_gates, exp, msg",
    [
        (["gate_1", "gate_2", "gate_3"], 3, None, None),
        (
            ["gate_1", "gate_2", "gate_3"],
            2,
            ValueError,
            "Number of named gates .* must match .* defined gates .*",
        ),
        (["gate_1", "gate_1"], 2, ValueError, "Gate names must be unique."),
    ],
)
def test_validate_gate_definitions_valid(gate_names, num_gates, exp, msg):
    with nullcontext() if exp is None else pytest.raises(exp, match=msg):
        dtv.validate_gate_definitions(gate_names, num_gates)


@pytest.mark.parametrize(
    "gate_names, gate_voltages, threshold_voltages, exp, msg",
    [
        (["g", "g2"], {"g": 0, "g2": 0}, {"g": 0}, None, None),
        (["g1", "g2"], {"g1": 0, "g2": 0}, {"g1": 0, "g2": 0}, None, None),
        (
            ["g1", "g2"],
            {"g1": 0.0},
            {"waa": 0.0},
            KeyError,
            "Gate voltage keys and threshold voltage keys must be a subset of gate names.",
        ),
        (
            ["g1", "g2"],
            {"waa": 0.0},
            {"g1": 0.0},
            KeyError,
            "Gate voltage keys and threshold voltage keys must be a subset of gate names.",
        ),
    ],
)
def test_validate_threshold_gate_definitions(
    gate_names, gate_voltages, threshold_voltages, exp, msg
):
    with nullcontext() if exp is None else pytest.raises(exp, match=msg):
        dtv.validate_threshold_gate_definitions(
            gate_names, gate_voltages, threshold_voltages
        )


@pytest.mark.parametrize(
    "gate_names, plunger, barrier, exp, msg",
    [
        (["g1", "g2"], ["g1"], ["g1"], ValueError, "must be mutually exclusive."),
        (["g1", "g2", "g3"], ["g1", "g2"], ["g3"], None, None),
        (
            ["gate1", "gate2", "gate3"],
            ["gate1", "gate2"],
            ["waa"],
            ValueError,
            "Each gate must be defined as either plunger or barrier.",
        ),
    ],
)
def test_validate_barrier_plunger_gates(gate_names, plunger, barrier, exp, msg):
    """Test that an error is raised when barrier gates and plunger gates are not mutually exclusive."""
    if exp is None:
        dtv.validate_barrier_plunger_gates(gate_names, plunger, barrier)
    else:
        with pytest.raises(exp, match=msg):
            dtv.validate_barrier_plunger_gates(gate_names, plunger, barrier)


def test_validate_virtual_gate_matrix_valid():
    """Test that no error is raised for a valid virtual gate matrix."""
    virtual_gate_matrix = jnp.zeros((3, 3))
    num_gates = 3

    try:
        dtv.validate_virtual_gate_matrix(virtual_gate_matrix, num_gates)
    except (ValueError, TypeError):
        pytest.fail("Unexpected error for a valid virtual gate matrix.")


def test_validate_virtual_gate_matrix_wrong_type():
    """Test that an error is raised if the matrix is not a jnp.ndarray."""
    virtual_gate_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]  # Not a jnp.ndarray
    num_gates = 3

    with pytest.raises(
        TypeError, match="Virtual gate matrix must be supplied as a jnp.ndarray."
    ):
        dtv.validate_virtual_gate_matrix(virtual_gate_matrix, num_gates)


def test_validate_virtual_gate_matrix_wrong_shape():
    """Test that an error is raised if the matrix has the wrong shape."""
    virtual_gate_matrix = jnp.zeros((3, 4))  # Incorrect shape
    num_gates = 3

    with pytest.raises(
        ValueError, match="Virtual gate matrix is not the correct shape .*"
    ):
        dtv.validate_virtual_gate_matrix(virtual_gate_matrix, num_gates)
