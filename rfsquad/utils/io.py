from __future__ import annotations

import json
import os
import pathlib
from abc import ABC, abstractmethod
from codecs import decode, encode
from dataclasses import dataclass
from typing import TYPE_CHECKING, Union

import h5py  # type: ignore
import tomli_w
import tomllib

from rfsquad.utils.globals import (
    CSD_ALLOWED,
    DEVICE_ALLOWED,
    DEVICE_REQUIRED,
    NOISE_ALLOWED,
    SOLVER_ALLOWED,
)

if TYPE_CHECKING:
    from rfsquad.core import Device
    from rfsquad.csd import ChargeStabilityDiagram
    from rfsquad.noise import AlphaNoise, GaussianNoise, INoiseModel, NoiseModel
    from rfsquad.solvers import ISolver

    DTObject = Union[Device, ChargeStabilityDiagram, ISolver, INoiseModel]
    DTInputObjcts = Union[
        Device, ISolver, NoiseModel, INoiseModel, AlphaNoise, GaussianNoise
    ]

import jax.numpy as jnp

PathLike = Union[str, os.PathLike, pathlib.Path]


# This module contains classes that can be used to load and dump Device and Simulation objects to and from different file formats.
# Currently supports only toml, json and h5. Other formats could be added at some point also.


def _from_file_helper(filename: PathLike) -> FileFormat:
    suffix = pathlib.Path(filename).suffix
    match suffix:
        case ".toml":
            file_obj = Toml.load(filename)

        case ".hdf5" | ".h5":
            file_obj = HDF5.load(filename)  # type: ignore # how dramatic

        case ".json":
            file_obj = Json.load(filename)  # type: ignore # how dramatic

        case _:
            raise ValueError(
                "File extension not supported. Please use either .toml, .json or .h5/.hdf5."
            )

    return file_obj


class FileFormat(ABC):
    """
    Abstract class for IO handling.
    """

    filename: PathLike
    data: dict = None  # type: ignore

    def __post_init__(self):
        # change to pathlib.Path object for easier manipulation
        self.filename = pathlib.Path(self.filename)

    @abstractmethod
    def load(self, filename):
        pass

    @abstractmethod
    def dump(self, filename, object) -> None:
        pass

    def _check_data_format(
        self, kind: str, required_keys: list[str], allowed_keys: list[str]
    ) -> None:
        """Checks the format of the loaded toml file

        Args:
            kind (str): Type of the object being loaded.
            required_keys (list[str]): Required parameters of the object.
            allowed_keys (list[str]): Allowed parameters.

        Raises:
            KeyError: 'kind' key is missing from the dictionary
            KeyError: Required keys are missing from the file
            KeyError: Additional keys not in allowed keys list are present in the file.
        """
        if kind not in self.data:
            raise KeyError(
                f"File is formatted incorrectly -- {kind} key is missing from the loaded file."
            )

        missing_keys = [k for k in required_keys if k not in self.data[kind]]
        if len(missing_keys) > 0:
            raise KeyError(
                "Some required keys are missing from the loaded file. Missing keys: "
                + ", ".join(missing_keys)
                + "."
            )

        extra_keys = [k for k in self.data[kind].keys() if k not in allowed_keys]
        if len(extra_keys) > 0:
            raise KeyError(
                "Some extra keys are present in the loaded file. Extra keys: "
                + ", ".join(extra_keys)
                + "."
            )

    def _check_filename(self, suffixes: list[str]) -> None:
        """
        Checks if the filename has the correct extension.

        Raises:
            ValueError: if filename does not have the correct extension
        """
        path = pathlib.Path(self.filename)
        filename_suffix = path.suffix

        if filename_suffix not in suffixes:
            raise ValueError(
                f"Incorrect file extension. File must have one of the following extensions: {', '.join(suffixes)}."
            )


@dataclass
class Toml(FileFormat):
    """
    Class for reading and writing toml files of Devices and Solvers.
    """

    filename: PathLike
    data: dict = None  # type: ignore

    @classmethod
    def load(cls, filename: PathLike) -> Toml:
        """
        Checks if file exists and is a .toml file, then loads it in.

        Args:
            filename (Path-like): Filename

        Raises:
            ValueError: if file is not a .toml file
            FileNotFoundError: if file does not exist

        Returns:
            cls (Toml): Toml object with the loaded data
        """

        inst = cls(filename)
        inst._check_filename([".toml"])

        if not inst.filename.exists():  # type: ignore # changed in post_init
            raise FileNotFoundError(f"File {str(filename)} does not exist.")

        with open(filename, "rb") as f:
            toml_data = tomllib.load(f)

        inst.data = toml_data

        return inst

    @classmethod
    def dump(cls, filename: PathLike, objects: list[DTInputObjcts]) -> None:
        """
        Creates a toml file describing the Device, Solver or NoiseModels.

        Dot-dot and dot-gate capacitance matrices are converted back to aF.

        Args:
            filename (PathLike): Filename
            objects (list[object]): List of objects to be saved (Solver, Device, NoiseModel)

        Raises:
            ValueError: if filename does not have .toml extension
        """
        inst = cls(filename)
        inst._check_filename([".toml"])

        data_dict: dict = {"Device": {}, "Solver": {}, "NoiseModel": {}}

        for obj in objects:
            # this would almost be easier with if statements
            # but i just learned match case exists
            # and i am enchanted by it
            match obj.__class__.__name__:
                case "Device":
                    data_dict["Device"] = inst._add_device_to_params(obj.__dict__)

                case "CISolver" | "RFSolver":
                    data_dict["Device"] = inst._add_device_to_params(
                        obj.device.__dict__
                    )  # type: ignore

                    data_dict["Solver"] = inst._add_solver_to_params(obj.__dict__)
                    data_dict["Solver"]["name"] = obj.__class__.__name__

                case "NoiseModel":
                    order = [i.__class__.__name__ for i in obj.noise_models]  # type: ignore
                    data_dict["NoiseModel"]["order"] = order
                    for noise in obj.noise_models:  # type: ignore
                        data_dict["NoiseModel"][noise.__class__.__name__] = (
                            inst._add_noise_to_params(noise.__dict__)
                        )

                case "AlphaNoise" | "GaussianNoise":
                    data_dict["NoiseModel"][obj.__class__.__name__] = (
                        inst._add_noise_to_params(obj.__dict__)
                    )

                case _:
                    raise ValueError(
                        "The object is not recognised. Allowed objects: Device, CISolver, RFSolver, NoiseModel, AlphaNoise, GaussianNoise."
                    )

        # this should get rid of the empty ones?
        final_dict: dict = {k: v for k, v in data_dict.items() if v}
        # get the params into a toml file
        with open(filename, "wb") as f:
            tomli_w.dump(final_dict, f)

    def _add_device_to_params(self, device: dict) -> dict:
        """
        Helper for dump()

        Args:
            device (dict): dict representation of the device

        Returns:
            dict: params
        """
        params: dict = {}
        for k in DEVICE_ALLOWED:
            v = device[k]

            if v is None:
                # toml doesn't allow none values
                continue

            if isinstance(v, jnp.ndarray):
                # assuming its the c_dd and c_dg that got amended, needs to be back to aF
                if k in DEVICE_REQUIRED:
                    v = v * 1.6e-19 / 1e-18

                params[k] = v.tolist()

            else:
                params[k] = v
        return params

    def _add_solver_to_params(self, solver: dict) -> dict:
        """
        Helper for dump()

        Args:
            env (dict): dict representation of the solver

        Returns:
            dict: params
        """
        params: dict = {}
        for k in SOLVER_ALLOWED:
            v = solver[k]

            if v is None:
                # toml doesn't allow none values
                # there shouldn't be any but you never know
                continue

            else:
                params[k] = v

        return params

    def _add_noise_to_params(self, noise: dict) -> dict:
        params: dict = {}
        for k in NOISE_ALLOWED:
            try:
                v = noise[k]

                if v is None:
                    continue

                else:
                    params[k] = v

            except KeyError:
                pass

        return params


@dataclass
class HDF5(FileFormat):
    """
    Class to load and save any DT objects to HDF5 files.

    The allowed keys in the file structure are as follows:
        - "CSD": ChargeStabilityDiagram or the derivations
        - "Device": Device object
        - "Solver": Solver object (contains sweep parameters and solver name, device is saved to Device)
    """

    filename: PathLike
    data: dict = None  # type: ignore

    def __post_init__(self):
        # change to pathlib.Path object for easier manipulation
        self.filename = pathlib.Path(self.filename)

    @classmethod
    def load(cls, filename: PathLike) -> HDF5:
        """
        Checks if file exists and is a .h5 file, then loads it in.

        Args:
            filename (Path-like): Filename

        Raises:
            ValueError: if file is not a .h5 file
            FileNotFoundError: if file does not exist

        Returns:
            cls (HDF5): Toml object with the loaded data
        """
        inst = cls(filename)
        inst._check_filename([".h5", ".hdf5"])

        if not inst.filename.exists():  # type: ignore # changed in post_init
            raise FileNotFoundError(f"File {str(filename)} does not exist.")

        inst.data = {"Device": {}, "Solver": {}, "CSD": {}}

        # chose to use context manager rather than opening a file in case of exceptions
        # dunno how legit that is
        with h5py.File(filename, "r") as f:
            for key in list(f.keys()):
                match key:
                    case "Device":
                        inst.data["Device"] = inst._load_device(f[key])

                    case "Solver":
                        # get the name
                        inst.data["Solver"] = inst._load_solver(f[key])

                    case "CSD":
                        inst.data["CSD"] = inst._load_csd(f[key])

                    case "NoiseModel":
                        inst.data["NoiseModel"] = inst._load_noise(f[key])

                    case _:
                        raise KeyError(
                            "The file contains unrecognised keys. Allowed keys: Device, Environment, Solver, CSD"
                        )

        return inst

    @classmethod
    def dump(cls, filename: PathLike, objects: list[DTObject]) -> None:
        inst = cls(filename)
        inst._check_filename([".h5", ".hdf5"])

        # objects can be a list of csd, device, env or solver objects - solver gets unpacked into device and env objects but it can be just CSD on its own

        data_dict: dict = {"CSD": {}, "Device": {}, "Solver": {}, "NoiseModel": {}}

        for object in objects:
            match object.__class__.__name__:
                case "Device":
                    data_dict["Device"] = inst._add_device_to_params(object.__dict__)

                case "ChargeStabilityDiagram" | "EnergyDiagram" | "OccupancyDiagram":
                    data_dict["CSD"] = inst._add_csd_to_params(object.__dict__)
                    data_dict["CSD"]["name"] = encode(object.__class__.__name__)

                case "CISolver" | "RFSolver":
                    dev = object.device  # type: ignore

                    data_dict["Device"] = inst._add_device_to_params(dev.__dict__)
                    data_dict["Solver"] = inst._add_solver_to_params(object.__dict__)  # type: ignore

                    data_dict["Solver"]["name"] = object.__class__.__name__

                case "NoiseModel":
                    nm_order = [
                        encode(i.__class__.__name__) for i in object.noise_models
                    ]  # type: ignore
                    # saving order explicitly because h5 seemingly reorders the keys alphabetically
                    # only relevant for composite noise
                    data_dict["NoiseModel"]["order"] = nm_order
                    for noise in object.noise_models:  # type: ignore
                        data_dict["NoiseModel"][noise.__class__.__name__] = (
                            inst._add_noise_to_params(noise.__dict__)
                        )

                case "AlphaNoise" | "GaussianNoise":
                    data_dict["NoiseModel"][object.__class__.__name__] = (
                        inst._add_noise_to_params(object.__dict__)
                    )

                case _:
                    raise ValueError(
                        "The object is not recognised. Allowed objects: Device, ChargeStabilityDiagram (and children), ISolver (and children), INoiseModel (and children)."
                    )

        # some of the data_dict should be populated now, only save those values
        with h5py.File(filename, "w") as f:
            for k, v in data_dict.items():
                # k are solver, device, environment, csd, if v is empty
                #  dict it does nothing
                if v:
                    # this makes group of the key
                    g = f.create_group(k)
                    # adds all non-dict values
                    for k2, v2 in v.items():
                        if not isinstance(v2, dict):
                            g.create_dataset(k2, data=v2)
                        # adds dictionary values (e.g. gate_voltage, threshold_voltage)
                        else:
                            h = g.create_group(k2)
                            for k3, v3 in v2.items():
                                try:
                                    h.create_dataset(k3, data=v3)

                                except TypeError:
                                    l = h.create_group(k3)  # noqa: E741
                                    # gunna assume this is for
                                    for k4, v4 in v3.items():
                                        l.create_dataset(k4, data=v4)

    def _add_device_to_params(
        self,
        device: dict,
    ) -> dict:
        """
        Helper for dump() - encodes relevant strings

        Args:
            device (dict): dictionary representation of the device

        Returns:
            dict: device params
        """
        d: dict = {}
        values_to_encode = ["gate_names", "dot_names", "plunger_gates", "barrier_gates"]

        for k in DEVICE_ALLOWED:
            v = device[k]

            if v is None:
                # h5 doesn't like None values
                continue

            elif k in values_to_encode:
                d[k] = [encode(i) for i in v]

            elif isinstance(v, jnp.ndarray):
                # assuming its the c_dd and c_dg that got amended, needs to be back to aF
                if k in DEVICE_REQUIRED:
                    v = v * 1.6e-19 / 1e-18
                d[k] = v.tolist()

            else:
                d[k] = v

        return d

    def _add_solver_to_params(self, solver: dict) -> dict:
        """
        Helper for dump()

        Args:
            solver (dict): dictionary representation of solver

        Returns:
            dict: solver params
        """

        d = {}
        for k in SOLVER_ALLOWED:
            v = solver[k]

            if v is None:
                # h5 doesn't like None values
                continue

            d[k] = v

        return d

    def _add_csd_to_params(self, csd: dict) -> dict:
        """
        Helper for dump() - encodes relevant strings. Does not include the noise model in the csd h5 representation even if it is present in the CSD being saved.

        Args:
            csd (dict): dictionary representation of the CSD

        Returns:
            dict: csd params
        """

        d: dict = {}

        for k in CSD_ALLOWED:
            # because we're just adding an existing object that's
            # formatted correctly, RFD will have temp but the others won't
            # so it makes sense to try and except
            match k:
                case "temperature":
                    try:
                        v = csd[k]
                        d[k] = v
                    except KeyError:
                        continue

                case "cmap_label" | "solver" | "sensor_gate":
                    try:
                        v = csd[k]
                        d[k] = encode(v)
                    except KeyError:
                        continue

                case "data":
                    try:
                        v = csd[k]
                        d[k] = v
                    except KeyError:
                        continue
                case _:
                    v = csd[k]
                    d[k] = v

        return d

    def _add_noise_to_params(self, noise: dict) -> dict:
        """
        Helper for dump() - encodes relevant strings

        Args:
            noise (dict): representation of the individual noise model (i.e. not NoiseModel)

        Returns:
            dict: noise params
        """
        d: dict = {}
        for k in NOISE_ALLOWED:
            try:
                v = noise[k]

                if v is None:
                    continue

                if k == "sweep_gate_order":
                    d[k] = [encode(i) for i in v]

                else:
                    d[k] = v

            except KeyError:
                continue

        return d

    def _load_device(self, device: dict) -> dict:
        """
        Helper for load() for devices - for decoding lists of strings
        """

        d = {}
        # had to hard code these in because it just does not behave correctly otherwise
        values_to_decode = ["gate_names", "dot_names", "plunger_gates", "barrier_gates"]
        values_as_dicts = ["gate_voltages", "threshold_voltages"]

        for k, v in device.items():
            if k in values_to_decode:
                d[k] = [decode(i) for i in v]

            elif k in values_as_dicts:
                d[k] = {}  # type: ignore
                for k2, v2 in v.items():
                    d[k][k2] = v2[()]

            else:
                d[k] = v[()]

        return d

    def _load_solver(self, solver: dict) -> dict:
        """
        Helper for load() for solver - for converting into a normal dict
        """

        d: dict = {}
        for k in solver.keys():
            if k == "name":
                d[k] = decode(solver[k][()])
            else:
                d[k] = solver[k][()]

        return d

    def _load_csd(self, csd: dict) -> dict:
        """
        Helper for load() for csd - for decoding strings
        """

        d = {}
        # had to hard code these in because it just does not behave correctly otherwise
        values_to_decode = ["name", "cmap_label", "solver", "sensor_gate"]
        values_as_dicts = ["sweep"]

        for k, v in csd.items():
            if k in values_to_decode:
                d[k] = decode(csd[k][()])

            elif k in values_as_dicts:
                d[k] = {}  # type: ignore
                for k2, v2 in v.items():
                    d[k][k2] = v2[()]  # type: ignore
            else:
                d[k] = v[()]

        return d

    def _load_noise(self, noise: dict) -> dict:
        d: dict = {}
        values_to_decode = ["sweep_gate_order"]

        # this is the noise model level, need one more for the individual noise models
        for k, v in noise.items():
            d[k] = {}
            if k == "order":
                d[k] = [decode(i) for i in v[()]]
            else:
                for kk, vv in v.items():
                    if kk in values_to_decode:
                        d[k][kk] = [decode(i) for i in vv]

                    else:
                        d[k][kk] = vv[()]

        return d


@dataclass
class Json(FileFormat):
    """
    Class for reading and writing json files of Devices and Solvers.
    """

    filename: PathLike
    data: dict = None  # type: ignore

    @classmethod
    def load(cls, filename: PathLike) -> Json:
        """
        Checks if file exists and is a .json file, then loads it in.

        Args:
            filename (Path-like): Filename

        Raises:
            ValueError: if file is not a .json file
            FileNotFoundError: if file does not exist

        Returns:
            cls (Json): Json object with the loaded data
        """

        inst = cls(filename)
        inst._check_filename([".json"])

        if not inst.filename.exists():  # type: ignore
            raise FileNotFoundError(f"File {str(filename)} does not exist.")

        with open(filename) as f:
            inst.data = json.load(f)

        return inst

    @classmethod
    def dump(cls, filename: PathLike, objects: list[DTInputObjcts]) -> None:
        """
        Creates a json file describing the Device or a dictionary describing te Solver.

        Dot-dot and dot-gate capacitance matrices are converted back to aF.

        Args:
            filename (PathLike): Filename
            objects (list[object]): List of objects to be saved (Solver, Device)

        Raises:
            ValueError: if filename does not have .json extension
        """
        inst = cls(filename)
        inst._check_filename([".json"])

        data_dict: dict = {"Device": {}, "Solver": {}, "NoiseModel": {}}

        for obj in objects:
            match obj.__class__.__name__:
                case "Device":
                    data_dict["Device"] = inst._add_device_to_params(obj.__dict__)

                case "CISolver" | "RFSolver":
                    data_dict["Device"] = inst._add_device_to_params(
                        obj.device.__dict__  # type: ignore[union-attr]
                    )

                    data_dict["Solver"] = inst._add_solver_to_params(obj.__dict__)
                    data_dict["Solver"]["name"] = obj.__class__.__name__

                case "NoiseModel":
                    order = [i.__class__.__name__ for i in obj.noise_models]  # type: ignore
                    data_dict["NoiseModel"]["order"] = order
                    for noise in obj.noise_models:  # type: ignore
                        data_dict["NoiseModel"][noise.__class__.__name__] = (
                            inst._add_noise_to_params(noise.__dict__)
                        )

                case "AlphaNoise" | "GaussianNoise":
                    data_dict["NoiseModel"][obj.__class__.__name__] = (
                        inst._add_noise_to_params(obj.__dict__)
                    )

                case "ChargeStabilityDiagram" | "EnergyDiagram" | "OccupancyDiagram":
                    data_dict["CSD"] = inst._add_csd_to_params(obj.__dict__)

                case _:
                    raise ValueError(
                        "The object is not recognised. Allowed objects: Device, CISolver, NoiseModel, AlphaNoise, GaussianNoise."
                    )

        # this should get rid of the empty ones?
        final_dict: dict = {k: v for k, v in data_dict.items() if v}

        with open(filename, "w") as f:
            json.dump(final_dict, f, indent=4)

    def _add_device_to_params(self, device: dict) -> dict:
        """
        Helper for dump()

        Args:
            device (dict): dict representation of the device

        Returns:
            dict: params
        """
        params: dict = {}
        for k in DEVICE_ALLOWED:
            v = device[k]

            if isinstance(v, jnp.ndarray):
                # assuming its the c_dd and c_dg that got amended, needs to be back to aF
                if k in DEVICE_REQUIRED:
                    v = v * 1.6e-19 / 1e-18

                params[k] = v.tolist()

            else:
                params[k] = v
        return params

    def _add_solver_to_params(self, solver: dict) -> dict:
        """
        Helper for dump()

        Args:
            env (dict): dict representation of the solver

        Returns:
            dict: params
        """
        params: dict = {}
        for k in SOLVER_ALLOWED:
            params[k] = solver[k]

        return params

    def _add_noise_to_params(self, noise: dict) -> dict:
        params: dict = {}
        for k in NOISE_ALLOWED:
            try:
                params[k] = noise[k]

            except KeyError:
                pass

        return params

    def _add_csd_to_params(self, csd: dict) -> dict:
        """
        Helper for dump()

        Args:
            csd (dict): dict representation of the CSD

        Returns:
            dict: csd params
        """

        d: dict = {}

        for k in CSD_ALLOWED:
            try:
                v = csd[k]
                if isinstance(v, jnp.ndarray):
                    d[k] = v.tolist()

                elif k == "sweep":
                    d[k] = {}
                    for k2, v2 in v.items():
                        d[k][k2] = v2.tolist()

                else:
                    d[k] = v

            except KeyError:
                continue

        return d
