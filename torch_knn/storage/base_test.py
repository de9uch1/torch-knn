import torch
import torch.nn as nn

from torch_knn import utils

from .base import Storage

N = 3
D = 8


class StorageMock(Storage):
    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def fit(self, x):
        return self


class ModuleMock(nn.Module):
    def __init__(self):
        super().__init__()
        self.storage = StorageMock(StorageMock.Config(D))


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

    def test_load_state_dict(self):
        x = torch.rand(N, D)
        cfg = StorageMock.Config(D)
        storage = StorageMock(cfg)
        storage.add(x)
        state_dict = storage.state_dict()
        new_storage = StorageMock(cfg)
        new_storage.load_state_dict(state_dict)
        torch.testing.assert_close(new_storage.data, storage.data)

    def test_load_state_dict_as_children(self):
        x = torch.rand(N, D)
        m = ModuleMock()
        m.storage.add(x)
        state_dict = m.state_dict()
        new_m = ModuleMock()
        new_m.load_state_dict(state_dict)
        torch.testing.assert_close(new_m.storage.data, m.storage.data)
