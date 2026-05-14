import pathlib

import jax.numpy as jnp
import numpy as np
import pytest

from rfsquad.noise import AlphaNoise, GaussianNoise, NoiseModel


@pytest.fixture
def sweep():
    return dict(
        gate_1=jnp.linspace(0, 0.1, 100),
        gate_2=jnp.linspace(0, 0.15, 100),
    )


class TestGaussianNoise:
    def test_init(self):
        amplitude = 0.1
        mean = 0.0
        std = 1.0
        noise = GaussianNoise(amplitude, mean, std)
        assert noise.amplitude == amplitude
        assert noise.mean == mean
        assert noise.std == std

    def test_repr(self):
        noise = GaussianNoise(0.01)
        assert isinstance(noise.__repr__(), str)
        representation = noise.__repr__()
        assert "GaussianNoise with parameters:" in representation
        assert "    amplitude: 0.01" in representation
        assert "    mean" in representation
        assert "    random_seed: 42" in representation

    def test_apply(self, sweep):
        amplitude = 0.1
        mean = 0.0
        std = 1.0
        gn = GaussianNoise(amplitude, mean, std)
        state = jnp.array([0.0, 0.0, 0.0])
        noise = gn.apply(state, sweep)
        assert noise.shape == state.shape
        assert noise.dtype == state.dtype
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(noise, state)

    @pytest.mark.parametrize("name", ["waa.toml", "waa.json", "waa.hdf5"])
    def test_to_from_file(self, name):
        amplitude = 0.1
        mean = 0.0
        std = 1.0
        filename = pathlib.Path(name)

        gn = GaussianNoise(amplitude, mean, std)
        gn.to_file(filename)

        assert filename.exists()

        gn2 = GaussianNoise.from_file(filename)

        assert gn.amplitude == gn2.amplitude
        assert gn.mean == gn2.mean
        assert gn.std == gn2.std

        filename.unlink()


class TestAlphaNoise:
    def test_init(self):
        amplitude = 0.1
        time = 1000
        an = AlphaNoise(amplitude, time)
        assert an.amplitude == amplitude
        assert an.time == time

    def test_repr(self):
        an = AlphaNoise(0.1, 1000)
        assert isinstance(an.__repr__(), str)
        representation = an.__repr__()
        assert "AlphaNoise with parameters:" in representation
        assert "    amplitude: 0.1" in representation
        assert "    time: 1000" in representation
        assert "    random_seed: 42" in representation

    def test_apply(self, sweep):
        an = AlphaNoise(amplitude=0.1, time=1000)
        state = jnp.array([[0.0, 0.0], [0.0, 0.0]])
        noise = an.apply(state, sweep)
        assert noise.shape == state.shape
        assert noise.dtype == state.dtype
        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(noise, state)

    def test_order(self, sweep):
        an = AlphaNoise(amplitude=0.1, time=1000)
        assert an._get_order(sweep) == (0, 1)

        an2 = AlphaNoise(
            amplitude=0.1, time=1000, sweep_gate_order=["gate_2", "gate_1"]
        )
        assert an2._get_order(sweep) == (1, 0)

    def test_fmin_valueerror(self, sweep):
        an = AlphaNoise(amplitude=0.1, time=1000, fmin=1)
        with pytest.raises(ValueError, match="fmin must be chosen"):
            an.apply(jnp.array([0.0, 0.0]), sweep)

    @pytest.mark.parametrize("name", ["waa.toml", "waa.json", "waa.hdf5"])
    def test_to_from_file_toml(self, name):
        amplitude = 0.1
        time = 1000
        filename = pathlib.Path(name)

        an = AlphaNoise(amplitude, time)
        an.to_file(filename)
        assert filename.exists()

        an2 = AlphaNoise.from_file(filename)
        assert an.amplitude == an2.amplitude
        assert an.alpha == an2.alpha
        assert an.time == an2.time

        filename.unlink()


class TestNoiseModel:
    def test_init(self):
        noise = NoiseModel()
        assert noise is not None
        assert isinstance(noise, NoiseModel)

    def test_repr(self):
        nm = NoiseModel()
        assert isinstance(nm.__repr__(), str)
        representation = nm.__repr__()
        assert "Empty NoiseModel" in representation

        nm.add(GaussianNoise(0.1))
        representation2 = nm.__repr__()
        assert "NoiseModel composed of" in representation2
        assert "GaussianNoise with parameters:" in representation2
        assert "    amplitude: 0.1" in representation2

    def test_add(self):
        gn = GaussianNoise(0.1)
        nm = NoiseModel()
        nm.add(gn)

        assert len(nm.noise_models) == 1
        assert nm.noise_models[0] == gn

    def test_apply(self, sweep):
        gn = GaussianNoise(0.1)
        nm = NoiseModel()
        nm.add(gn)

        state = jnp.array([0.0, 0.0, 0.0])
        noise = nm.apply(state, sweep)
        assert noise.shape == state.shape
        assert noise.dtype == state.dtype

        with pytest.raises(AssertionError):
            np.testing.assert_array_equal(noise, state)

    @pytest.mark.parametrize("name", ["waa.toml", "waa.json", "waa.h5"])
    def test_to_from_file_toml(self, name):
        filename = pathlib.Path(name)

        an = AlphaNoise(1, 1000)
        gn = GaussianNoise(0.1)
        nm = NoiseModel()
        nm.add(gn)
        nm.add(an)

        nm.to_file(filename)
        assert filename.exists()

        nm2 = NoiseModel.from_file(filename)
        assert len(nm.noise_models) == len(nm2.noise_models)
        # order is retained
        assert (
            nm.noise_models[0].__class__.__name__
            == nm2.noise_models[0].__class__.__name__
        )
        assert nm.noise_models[0].amplitude == nm2.noise_models[0].amplitude

        filename.unlink()
