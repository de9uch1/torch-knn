from contextlib import nullcontext as does_not_raise

import pytest
import torch
from torch_knn.storage.flat import FlatStorage

D = 8


class TestFlatStorage:
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
        torch.testing.assert_close(storage.train(x).data, storage.data)

    def test_is_trained(self):
        cfg = FlatStorage.Config(D)
        storage = FlatStorage(cfg)
        assert storage.is_trained
