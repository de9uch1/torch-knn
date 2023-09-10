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

    def test_fit(self):
        cfg = FlatStorage.Config(D)
        x = torch.rand(3, D)
        storage = FlatStorage(cfg)
        assert storage.fit(x) is storage

        cfg = FlatStorage.Config(D)
        x = torch.rand(3, D)
        storage = FlatStorage(cfg)
        torch.testing.assert_close(storage.fit(x).data, storage.data)
