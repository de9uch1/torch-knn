import torch

from torch_knn import utils
from torch_knn.storage.base import Storage

N = 3
D = 8


class StorageMock(Storage):
    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def train(self, x):
        return self

    @property
    def is_trained(self):
        return True


class TestStorage:
    def test___init__(self):
        x = torch.rand(N, D)
        cfg = StorageMock.Config(x.size(-1))
        storage = StorageMock(cfg)
        assert storage.cfg == cfg
        assert storage.metric == cfg.metric

    def test_N(self):
        x = torch.rand(N, D)
        cfg = StorageMock.Config(x.size(-1))
        storage = StorageMock(cfg)
        storage.add(x)
        assert storage.N == x.size(0)

    def test_D(self):
        cfg = StorageMock.Config(D)
        storage = StorageMock(cfg)
        assert storage.D == D

    def test_shape(self):
        cfg = StorageMock.Config(int())
        storage = StorageMock(cfg)
        assert utils.is_equal_shape(storage.shape, [0])

        x = torch.rand(N, D)
        cfg = StorageMock.Config(D)
        storage = StorageMock(cfg)
        storage.add(x)
        assert utils.is_equal_shape(storage.shape, x)

    def test_add(self):
        x = torch.rand(N, D)
        cfg = StorageMock.Config(D)
        storage = StorageMock(cfg)
        storage.add(x)
        torch.testing.assert_close(storage.data, x)
