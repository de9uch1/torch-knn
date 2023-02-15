from contextlib import nullcontext as does_not_raise

import pytest
import torch
from torch_knn.storage.flat import FlatStorage

D = 8


class TestFlatStorage:
    @pytest.mark.parametrize(
        "shape,expectation",
        [
            ((1,), pytest.raises(ValueError)),
            ((3, D), does_not_raise()),
            ((3, 16), pytest.raises(ValueError)),
            ((4, 2, 3), pytest.raises(ValueError)),
        ],
    )
    def test_check_shape(self, shape, expectation):
        cfg = FlatStorage.Config(D)
        x = torch.rand(shape)
        storage = FlatStorage(cfg)
        with expectation:
            storage.check_shape(x)

    def test_encode(self):
        cfg = FlatStorage.Config(D)
        x = torch.rand(3, D)
        storage = FlatStorage(cfg)
        torch.testing.assert_close(storage.encode(x), x)

    def test_decode(self):
        cfg = FlatStorage.Config(D)
        x = torch.rand(3, D)
        storage = FlatStorage(cfg)
        torch.testing.assert_close(storage.decode(x), x)

    def test_train(self):
        cfg = FlatStorage.Config(D)
        x = torch.rand(3, D)
        storage = FlatStorage(cfg)
        assert storage.train(x) is storage

        cfg = FlatStorage.Config(D)
        x = torch.rand(3, D)
        storage = FlatStorage(cfg)
        torch.testing.assert_close(storage.train(x).storage, storage.storage)

    def test_is_trained(self):
        cfg = FlatStorage.Config(D)
        storage = FlatStorage(cfg)
        assert storage.is_trained
