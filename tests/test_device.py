import copy
import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

from rfsquad.core import Device, _make_2d_array


@pytest.mark.parametrize(
    "inp, out",
    [
        (1, jnp.array([[1]])),
        (0.2, jnp.array([[0.2]])),
        ([1, 2], jnp.array([[1, 2]])),
        ([[1, 2], [3, 4]], jnp.array([[1, 2], [3, 4]])),
        (jnp.array([1, 2]), jnp.array([[1, 2]])),
        (jnp.array([[1, 2], [3, 4]]), jnp.array([[1, 2], [3, 4]])),
    ],
)
def test_make_2d_array(inp, out):
    assert jnp.all(_make_2d_array(inp) == out)


@pytest.mark.parametrize("inp", ["waa", ["a", "b", "c"]])
def test_make_2d_array_incorrect(inp):
    with pytest.raises(ValueError, match="Invalid input"):
        _make_2d_array(inp)


@pytest.fixture
def c_dd():
    return jnp.array([[100.0, -10.0], [-10.0, 100.0]])


@pytest.fixture
def c_dg():
    return jnp.array([[10.0, 2.0], [2.0, 10.0]])


@pytest.fixture
def two_dot_three_gate_matrices():
    C_dd = jnp.array([[20.0, -4.0], [-4.0, 20.0]])
    C_dg = jnp.array([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]])
    return C_dd, C_dg


@pytest.fixture
def three_dot_five_gate_matrices():
    C_dd = jnp.array([[20.0, -4.0, -4.0], [-4.0, 20.0, -4.0], [-4.0, -4.0, 20.0]])

    C_dg = jnp.array(
        [
            [2.0, 0.5, 0.1, 0.05, 0.01],
            [0.1, 2.0, 0.1, 0.01, 0.05],
            [0.1, 0.5, 2.0, 0.00, 0.00],
        ]
    )
    return C_dd, C_dg


@pytest.fixture
def valid_device():
    C_dd = jnp.array([[20.0, -4.0], [-4.0, 20.0]])
    C_dg = jnp.array([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]])
    return Device(C_dd=C_dd, C_dg=C_dg)


@pytest.fixture
def valid_device_vgm():
    C_dd = jnp.array([[20.0, -4.0], [-4.0, 20.0]])
    C_dg = jnp.array([[2.0, 0.5, 0.1], [0.1, 0.5, 2.0]])
    return Device(
        C_dd=C_dd,
        C_dg=C_dg,
        virtual_gate_matrix=jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]),
    )


@pytest.fixture
def valid_device_tqd():
    C_dd = jnp.array([[20.0, -4.0, -2.0], [-4.0, 20.0, -4.0], [-2.0, -4.0, 20.0]])
    C_dg = jnp.array([[2.0, 0.5, 0.1], [0.1, 2.5, 0.5], [0.1, 0.5, 2.0]])
    return Device(C_dd=C_dd, C_dg=C_dg)


@pytest.fixture
def gate_voltages():
    return dict(gate1=0.1, gate2=0.2)


class TestDevice:
    def test_init(self, c_dd, c_dg):
        device = Device(c_dd, c_dg)
        assert device.C_dd.shape == (2, 2)
        assert hasattr(device, "C_dd")
        assert hasattr(device, "C_dd_inv")
        assert device.C_dd_inv is not None
        assert device.C_dd_inv.shape == (2, 2)
        assert "gate_1" in device.gate_names
        assert "gate_2" in device.gate_names
        assert "dot_1" in device.dot_names
        assert "dot_2" in device.dot_names

    def test_init_single_dot(self):
        device = Device(20, 5)
        assert device.C_dd.shape == (1, 1)
        assert isinstance(device.C_dd, jnp.ndarray)
        assert device.num_dots == 1
        assert device.num_gates == 1

    def test_two_dot_three_gate(self, two_dot_three_gate_matrices):
        dev = Device(*two_dot_three_gate_matrices)
        assert len(dev.gate_names) == 3
        assert len(dev.dot_names) == 2

    def test_three_dot_five_gate(self, three_dot_five_gate_matrices):
        dev = Device(*three_dot_five_gate_matrices)
        assert len(dev.gate_names) == 5
        assert len(dev.dot_names) == 3

    def test_gate_names(self, three_dot_five_gate_matrices):
        # gate_names - changing to make non alphabetical here due to Jax jit being silly
        dev = Device(
            *three_dot_five_gate_matrices,
            gate_names=["gate_f", "gate_e", "gate_c", "gateb", "gatea"],
        )
        assert dev.gate_names == ["gate_f", "gate_e", "gate_c", "gateb", "gatea"]

    def test_assign_gates(self, three_dot_five_gate_matrices):
        # plunger gates
        dev = Device(
            *three_dot_five_gate_matrices,
            gate_names=["gate_c", "gate_b", "gate_a", "gate_z", "gate_x"],
        )
        dev.assign_gates(
            plunger=["gate_c", "gate_b", "gate_a"], barrier=["gate_z", "gate_x"]
        )

        assert dev.gate_names == ["gate_c", "gate_b", "gate_a", "gate_z", "gate_x"]
        assert dev.plunger_gates == ["gate_c", "gate_b", "gate_a"]
        assert dev.barrier_gates == ["gate_z", "gate_x"]

    @pytest.mark.parametrize(
        "gv, tv, exp, msg",
        [
            # all defined
            (
                {"gate_1": 0.1, "gate_2": 0.2, "gate_3": 0.3},
                {"gate_1": 0.1, "gate_2": 0.2, "gate_3": 0.3},
                None,
                None,
            ),
            # fewer defined still ok
            (
                {"gate_1": 0.1, "gate_2": 0.2},
                {"gate_1": 0.1, "gate_2": 0.2},
                None,
                None,
            ),
            # only one defined
            (None, {"gate_1": 0.1, "gate_2": 0.2, "gate_3": 0.3}, None, None),
            ({"gate_1": 0.1, "gate_2": 0.2, "gate_3": 0.3}, None, None, None),
            # incorrect names
            ({"waa": 2}, None, KeyError, "must be a subset of gate names"),
            (None, {"waa": 2}, KeyError, "must be a subset of gate names"),
        ],
    )
    def test_add_static_voltages(self, gv, tv, exp, msg, two_dot_three_gate_matrices):
        dev = Device(*two_dot_three_gate_matrices)
        if exp is None:
            dev.add_static_voltages(gate=gv, threshold=tv)
            if gv is not None:
                assert all(
                    key in dev.gate_voltages and gv[key] == dev.gate_voltages[key]
                    for key in gv
                )
            if tv is not None:
                assert all(
                    key in dev.threshold_voltages
                    and tv[key] == dev.threshold_voltages[key]
                    for key in tv
                )

        else:
            with pytest.raises(exp, match=msg):
                dev.add_static_voltages(gate=gv, threshold=tv)


class TestDeviceIO:
    def test_from_file(self, capfd):
        device = Device.from_file("examples/device_config.toml")
        assert device.C_dd.shape == (3, 3)
        assert device.C_dg.shape == (3, 3)
        assert "gate_1" in device.gate_names
        assert "gate_2" in device.gate_names

        # file not existing and incorrect keys are tested in helpers
        _ = Device.from_file("examples/device_config.toml", verbose=True)
        captured = capfd.readouterr()
        assert "Device instantiated with:" in captured.out
        assert "Dots: dot1, dot_2, dot_3" in captured.out

    @pytest.mark.parametrize("name", [("waa.toml"), ("waa.h5"), ("waa.json")])
    def test_to_file(self, name):
        fname = pathlib.Path(name)

        dev1 = Device.from_file("examples/device_config.toml")
        dev1.to_file(fname)

        assert fname.is_file()

        dev2 = Device.from_file(fname)
        # allclose for h5, doesnt like exact matches
        np.testing.assert_allclose(dev1.C_dd, dev2.C_dd, rtol=1e-5)

        fname.unlink()


class TestDeviceRepr:
    def test_repr(self, c_dd, c_dg):
        device = Device(c_dd, c_dg, gate_names=["gate1", "gate2"])
        assert isinstance(device.__repr__(), str)
        representation = device.__repr__()
        assert "Device instantiated with:" in representation
        assert "Dots: dot_1, dot_2" in representation
        assert "Gates: gate1, gate2" in representation
        assert "Plunger gates: gate1, gate2" in representation
        assert "Barrier gates: None" in representation
        assert "625" in representation
        assert "-62.5" in representation
        assert "12.5" in representation

    def test_repr_gates(self, two_dot_three_gate_matrices):
        device = Device(*two_dot_three_gate_matrices)
        assert isinstance(device.__repr__(), str)
        representation = device.__repr__()
        assert "Device instantiated with:" in representation
        assert "Dots: dot_1, dot_2" in representation
        assert "Gates: gate_1, gate_2, gate_3" in representation
        assert "Plunger gates: gate_1, gate_2" in representation
        assert "Barrier gates: gate_3" in representation

    def test_repr_virtual_gates(self, c_dd, c_dg):
        virtual_matrix = jnp.array([[0.101, -0.025], [-0.025, 0.101]])
        device = Device(c_dd, c_dg, virtual_gate_matrix=virtual_matrix)
        assert isinstance(device.__repr__(), str)
        representation = device.__repr__()
        assert "Virtual gate matrix:" in representation
        assert "0.101" in representation
        assert "0.025" in representation


class TestDeviceValidation:
    # Array shapes -----------------------------------------------------------
    @pytest.mark.parametrize(
        "cdd, cdg, exp, msg",
        [
            (
                [[100.0, -10.0], [-10.0, 100.0], [1, 2]],
                [[10.0, 2.0], [2.0, 10.0]],
                ValueError,
                "Dot-dot capacitance matrix must be of a",
            ),
            (
                [[100.0, -10.0], [-10.0, 100.0]],
                [[10.0, 2.0], [2.0, 10.0], [1, 2]],
                ValueError,
                "Dot-gate capacitance matrix must have 2 rows",
            ),
            (
                [[100.0, -10.0], [-10.0, 100.0]],
                [[10.0], [10.0]],
                ValueError,
                "Dot-gate capacitance matrix must have 2 or more columns",
            ),
        ],
    )
    def test_array_shapes(self, cdd, cdg, exp, msg):
        with pytest.raises(exp, match=msg):
            Device(cdd, cdg)

    # Array values -----------------------------------------------------------
    @pytest.mark.parametrize(
        "cdd, exp, msg",
        [
            (
                jnp.array([[100.0, 10.0], [-10.0, 100.0]]),
                ValueError,
                "Dot-dot capacitance matrix off-diagonal elements must all be negative",
            ),
            (
                jnp.array([[100.0, -10.0], [-10.0, -100.0]]),
                ValueError,
                "Dot-dot capacitance matrix diagonal elements must all be positive",
            ),
            (
                jnp.array([[100, -5], [-10, 100]]),
                ValueError,
                "Dot-dot capacitance matrix must be symmetric",
            ),
        ],
    )
    def test_array_values(self, c_dg, cdd, exp, msg):
        with pytest.raises(exp, match=msg):
            Device(cdd, c_dg)

    # Gate definitions -------------------------------------------------------------
    def test_gate_names(self, c_dd, c_dg):
        with pytest.raises(ValueError, match="Gate names must be unique"):
            Device(c_dd, c_dg, gate_names=["gate1", "gate1"])

    def test_gate_numbers(self, c_dd, c_dg):
        with pytest.raises(
            ValueError,
            match=r"Number of named gates \(3",
        ):
            Device(c_dd, c_dg, gate_names=["gate1", "gate2", "gate3"])

    def test_barrier_plunger_gates(self, c_dd, c_dg):
        with pytest.raises(
            ValueError,
            match="Plunger gates and barrier gates must be mutually exclusive.",
        ):
            device = Device(c_dd, c_dg)
            device.assign_gates(plunger=["gate_1"], barrier=["gate_1"])

        with pytest.raises(
            ValueError,
            match="Each gate must be defined as either plunger or barrier.",
        ):
            device = Device(c_dd, c_dg)
            device.assign_gates(plunger=["gate_1"], barrier=["gate_3"])

    # Virtual gates -----------------------------------------------------
    def test_virtual_gates_matrix(self, c_dd, c_dg):
        with pytest.raises(TypeError, match="Virtual gate matrix must be supplied as"):
            device = Device(c_dd, c_dg, virtual_gate_matrix="waa")

        with pytest.raises(
            ValueError, match="Virtual gate matrix is not the correct shape"
        ):
            device = Device(
                c_dd,
                c_dg,
                virtual_gate_matrix=jnp.array(
                    [[100.0, -10.0], [-10.0, 100.0], [-1, -2]]
                ),
            )

        # add a test that tests when it works
        device = Device(
            c_dd,
            c_dg,
            virtual_gate_matrix=jnp.array([[100.0, -10.0], [-10.0, 100.0]]),
        )
        assert device.virtual_gate_matrix is not None


# Testing the equality of two devices
def test_device_equality(valid_device, valid_device_tqd):
    # Test equality of two identical devices
    assert valid_device_tqd == valid_device_tqd

    # Test inequality of two different devices
    assert valid_device_tqd != valid_device


def test_device_equality_with_different_attributes_C_matrices(valid_device):
    # Modify an attribute and test inequality
    valid_copy = copy.deepcopy(valid_device)
    valid_device2 = valid_device.C_dd.at[0, 0].set(0.5)
    assert valid_device2 != valid_copy


def test_device_equality_with_different_types(valid_device):
    # Test inequality with a different type
    assert valid_device != "not a device"
