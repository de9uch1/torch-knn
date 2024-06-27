import torch

from .flat import StorageFlat

D = 8


class TestStorageFlat:
    def test_encode(self):
        cfg = StorageFlat.Config(D)
        x = torch.rand(3, D)
        storage = StorageFlat(cfg)
        torch.testing.assert_close(storage.encode(x), x)

    def test_decode(self):
        cfg = StorageFlat.Config(D)
        x = torch.rand(3, D)
        storage = StorageFlat(cfg)
        torch.testing.assert_close(storage.decode(x), x)

    def test_fit(self):
        cfg = StorageFlat.Config(D)
        x = torch.rand(3, D)
        storage = StorageFlat(cfg)
        assert storage.fit(x) is storage

        cfg = StorageFlat.Config(D)
        x = torch.rand(3, D)
        storage = StorageFlat(cfg)
        torch.testing.assert_close(storage.fit(x).data, storage.data)
