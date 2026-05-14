from __future__ import annotations

import pathlib
from abc import ABC, abstractmethod

import jax.numpy as jnp  # type: ignore

from rfsquad.utils.io import HDF5, Json, PathLike, Toml, _from_file_helper

# -------------------------
# NOISE MODELS
# -------------------------
# We have an interface for the noise models that all actual implementations should inherit from
# At the moment we have one concrete implementation of the interface, GaussianNoise but Telegraph noise and other noise models can be added in the future that would require time-domain knowledge to be implemented correctly
# To combine noise models we have a NoiseModel class that takes a list of noise models and applies them in sequence
# The noise models are applied in CSD in the .add_noise(noisemodel=noisemodel) method in the ICSD class
# -------------------------

# use:
# noise = NoiseModel()
# noise.add(GaussianNoise(amplitude=0.1))
# noise.add(AlphaNoise(amplitude=0.1, time=1000, alpha=1)) # pink noise
# csd = solver.run(...)
# csd.add_noise() <- this will apply the noise models in sequence


class INoiseModel(ABC):
    def __repr__(self):
        representation = f"{self.__class__.__name__} with parameters:"
        for key, value in self.__dict__.items():
            representation += f"\n    {key}: {value}"

        return representation

    @abstractmethod
    def apply(self, state: jnp.ndarray, sweep: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Apply XX noise to the state.

        Args:
            state (jnp.ndarray): State to apply noise to
            sweep (dict[str, jnp.ndarray]): Sweep parameters

        Returns:
            jnp.ndarray: Noisy state
        """
        pass

    @classmethod
    def from_file(cls, filename: PathLike):
        """
        Load the noise model from a file.

        Toml file requires a [NoiseModel] section with required noise model name (which matches one of the class names in this module) and amplitude. The file should have additional parameters for the specific noise model.

        HDF5 file requires a group with the noise model name and the required parameters for the specific noise model. The file should have additional parameters for the specific noise model The string inputs are automatically decoded from binary.


        Usage:
            noise = NoiseModel.from_file("noise.toml")

        Args:
            filename (PathLike): Path to file containing the noise model

        Returns:
            NoiseModel: noise model object
        """

        file_obj = _from_file_helper(filename)

        match cls.__name__:
            case "GaussianNoise" | "AlphaNoise":
                # check name exists in the file
                if cls.__name__ not in file_obj.data["NoiseModel"]:
                    raise ValueError(f"Missing {cls.__name__} data in file")

                return cls(**file_obj.data["NoiseModel"][cls.__name__])

            # good grief this is gammy
            case "NoiseModel":
                noise = NoiseModel()
                # so that its in the correct order
                for nm in file_obj.data["NoiseModel"]["order"]:
                    for k, v in file_obj.data["NoiseModel"].items():
                        if nm == k == "GaussianNoise":
                            # I need to do this here due to circular imports
                            from rfsquad.noise.gaussian_noise import GaussianNoise

                            noise.add(GaussianNoise(**v))
                        elif nm == k == "AlphaNoise":
                            # I need to do this here due to circular imports
                            from rfsquad.noise.alpha_noise import AlphaNoise

                            noise.add(AlphaNoise(**v))
                        else:
                            pass

                return noise

            case _:
                raise ValueError(f"Unknown noise model {cls.__name__}")

    def to_file(self, filename: PathLike) -> None:
        """
        Save the noise model to a file.

        Args:
            filename (PathLike): Path to file to save the noise model

        """
        match pathlib.Path(filename).suffix:
            case ".toml":
                Toml.dump(filename, [self])
            case ".h5" | ".hdf5":
                HDF5.dump(filename, [self])
            case ".json":
                Json.dump(filename, [self])
            case _:
                raise ValueError(
                    "File extension not supported. Please use either .toml, .json or .h5/.hdf5."
                )


class NoiseModel(INoiseModel):
    def __init__(self):
        self.noise_models = []

    def __repr__(self):
        if len(self.noise_models) > 0:
            representation = f"{self.__class__.__name__} composed of:\n"
            for noise_model in self.noise_models:
                representation += f"\n{noise_model}\n"

        else:
            representation = f"Empty {self.__class__.__name__}"

        return representation

    def add(self, noise_model: INoiseModel):
        """
        Add a noise model to the composite NoiseModel. Use any other NoiseModels implemented here (e.g. GaussianNoise, AlphaNoise). The noise will be applied to the data in the order they were added here.

        Args:
            noise_model (INoiseModel): Noise model to add.
        """
        self.noise_models.append(noise_model)

    def apply(self, state: jnp.ndarray, sweep: dict[str, jnp.ndarray]) -> jnp.ndarray:
        """
        Apply the noise models to the state

        Args:
            state (jnp.ndarray): State to apply noise to
            sweep (dict[str, jnp.ndarray]): Sweep parameters

        Returns:
            jnp.ndarray: Noisy state
        """
        for noise_model in self.noise_models:
            state = noise_model.apply(state, sweep)
        return state
