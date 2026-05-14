from __future__ import annotations

import pathlib

import pytest

from rfsquad.core import Device
from rfsquad.solvers import CISolver
from rfsquad.utils.globals import (
    DEVICE_ALLOWED,
    DEVICE_REQUIRED,
)
from rfsquad.utils.io import HDF5, Json, Toml


@pytest.fixture
def toml():
    return Toml.load("examples/device_config.toml")


@pytest.fixture
def json_device():
    return Json.load("examples/device_config.json")


@pytest.fixture
def json_csd():
    return Json.load("examples/csd.json")


@pytest.fixture
def toml_solver():
    return Toml.load("examples/solver_config.toml")


class TestToml:
    def test_check_filename(self):
        with pytest.raises(
            ValueError,
            match="Incorrect file extension. File must have one of the following extensions: .toml",
        ):
            toml = Toml("examples/non_existent")
            toml._check_filename([".toml"])

    def test_check_format_device(self, toml: Toml):
        # we dont acc care about the values, just the keys
        # values are checked at device instantiation time

        # no exceptions raised
        toml._check_data_format(
            kind="Device",
            required_keys=DEVICE_REQUIRED,
            allowed_keys=DEVICE_ALLOWED,
        )

        with pytest.raises(KeyError, match="File is formatted incorrectly"):
            toml.data = {"Device": {}}
            toml._check_data_format(
                kind="device",
                required_keys=DEVICE_REQUIRED,
                allowed_keys=DEVICE_ALLOWED,
            )

        with pytest.raises(
            KeyError,
            match="Some required keys are missing from the loaded file. Missing keys: C_dd, C_dg",
        ):
            toml.data = {"Device": {}}
            toml._check_data_format(
                kind="Device",
                required_keys=DEVICE_REQUIRED,
                allowed_keys=DEVICE_ALLOWED,
            )

        with pytest.raises(
            KeyError,
            match="Some extra keys are present in the loaded file. Extra keys: extra.",
        ):
            toml.data = {
                "Device": {
                    "C_dd": "array",
                    "C_dg": "array",
                    "extra": "key",
                },
            }
            toml._check_data_format(
                kind="Device",
                required_keys=DEVICE_REQUIRED,
                allowed_keys=DEVICE_ALLOWED,
            )

    def test_load(self, toml: Toml):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            Toml.load("examples/non_existent.toml")

        with pytest.raises(
            ValueError,
            match="Incorrect file extension. File must have one of the following extensions: .toml.",
        ):
            Toml.load("examples/non_existent")

        assert isinstance(toml, Toml)
        assert toml.filename == pathlib.Path("examples/device_config.toml")
        assert toml.data is not None
        assert isinstance(toml.data, dict)

    def test_dump_error(self):
        class Dummy:
            a = 2

        with pytest.raises(
            ValueError, match="The object is not recognised. Allowed objects:"
        ):
            Toml.dump("dummy.toml", [Dummy()])

    def test_add_device_to_params(self):
        toml = Toml("waa.toml")
        device = Device.from_file("examples/device_config.toml")
        params2 = toml._add_device_to_params(device.__dict__)

        assert "C_dd" in params2
        assert "C_dg" in params2


class TestHDF5:
    def test_check_filename(self):
        with pytest.raises(
            ValueError,
            match="Incorrect file extension. File must have one of the following extensions: .h5",
        ):
            h5 = HDF5("examples/non_existent")
            h5._check_filename([".h5", ".hdf5"])

    def test_load(self):
        with pytest.raises(FileNotFoundError, match="does not exist"):
            HDF5.load("examples/non_existent.h5")

        with pytest.raises(
            ValueError,
            match="Incorrect file extension. File must have one of the following extensions: .h5",
        ):
            HDF5.load("examples/non_existent")

    def test_dump(self):
        fn_dev = pathlib.Path("test_device.h5")
        fn_solver = pathlib.Path("test_solver.h5")
        # DEVICE
        device = Device(
            C_dd=[[20, -0.4], [-0.4, 20]],
            C_dg=[[2, 0.4], [0.4, 2]],
        )
        device.add_static_voltages(gate={"gate_1": 0.1, "gate_2": 0.2})
        HDF5.dump(fn_dev, [device])

        # SOLVER
        solver = CISolver(device)
        HDF5.dump(fn_solver, [solver])

        assert fn_dev.exists()
        assert fn_solver.exists()

        fn_dev.unlink()
        fn_solver.unlink()

    def test_dump_error(self):
        class Dummy:
            pass

        with pytest.raises(ValueError, match="The object is not recognised"):
            HDF5.dump("dummy.h5", [Dummy()])


class TestJson:
    def test_check_filename(self):
        with pytest.raises(
            ValueError,
            match="Incorrect file extension. File must have one of the following extensions: .json",
        ):
            json = Json("examples/non_existent")
            json._check_filename([".json"])

    def test_check_format(self, json_device):
        # no exceptions raised
        assert (
            json_device._check_data_format(
                kind="Device",
                required_keys=DEVICE_REQUIRED,
                allowed_keys=DEVICE_ALLOWED,
            )
            is None
        )

    @pytest.mark.parametrize(
        "kind, data, exp, msg",
        [
            ("Device", {"device": {}}, KeyError, "File is formatted incorrectly"),
            (
                "Device",
                {"Device": {}},
                KeyError,
                "Some required keys are missing from the loaded file. Missing keys: C_dd, C_dg",
            ),
            (
                "Device",
                {"Device": {"C_dd": "array", "C_dg": "array", "extra": "key"}},
                KeyError,
                "Some extra keys are present in the loaded file. Extra keys: extra.",
            ),
        ],
    )
    def test_check_format_device_error(self, json_device, kind, data, exp, msg):
        with pytest.raises(exp, match=msg):
            json_device.data = data
            json_device._check_data_format(
                kind=kind,
                required_keys=DEVICE_REQUIRED,
                allowed_keys=DEVICE_ALLOWED,
            )

    @pytest.mark.parametrize(
        "filename, exp, msg",
        [
            ("examples/non_existent.json", FileNotFoundError, "does not exist"),
            ("examples/non_existent", ValueError, "the following extensions: .json."),
        ],
    )
    def test_load_errors(self, filename, exp, msg):
        with pytest.raises(exp, match=msg):
            Json.load(filename)

    @pytest.mark.parametrize(
        "json, filename",
        [
            ("json_device", "examples/device_config.json"),
            ("json_csd", "examples/csd.json"),
        ],
    )
    def test_load(self, json, filename, request):
        file = request.getfixturevalue(json)
        assert isinstance(file, Json)
        assert file.filename == pathlib.Path(filename)
        assert file.data is not None
        assert isinstance(file.data, dict)

    def test_dump_error(self):
        class Dummy:
            a = 2

        with pytest.raises(
            ValueError, match="The object is not recognised. Allowed objects:"
        ):
            Json.dump("dummy.json", [Dummy()])

    def test_add_device_to_params(self):
        file = Json("waa.json")
        device = Device.from_file("examples/device_config.json")
        params2 = file._add_device_to_params(device.__dict__)

        assert "C_dd" in params2
        assert "C_dg" in params2
