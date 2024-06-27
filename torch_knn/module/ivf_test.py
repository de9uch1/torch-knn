import torch

from torch_knn.module.ivf import InvertedFile
from torch_knn.storage.flat import StorageFlat

N = 1024
D = 4
NLISTS = 8


class TestInvertedFile:
    def test___init__(self):
        storage = StorageFlat(StorageFlat.Config(D))
        ivf = InvertedFile(storage, NLISTS)
        assert ivf.metric == storage.metric
        assert ivf.nlists == NLISTS
        assert len(ivf.invlists) == NLISTS

    def test_add(self):
        storage = StorageFlat(StorageFlat.Config(D))
        ivf = InvertedFile(storage, NLISTS)
        torch.manual_seed(0)
        x = torch.rand(N, D)
        xb = x[:NLISTS]
        ivf.centroids = xb
        assigns = ivf.add(xb)
        assert torch.equal(assigns, torch.arange(NLISTS))
        assert all([len(plist) == 1 for plist in ivf.invlists])

    def test_load_state_dict(self):
        storage = StorageFlat(StorageFlat.Config(D))
        ivf = InvertedFile(storage, NLISTS)
        x = torch.rand(N, D)
        xb = x[:NLISTS]
        ivf.centroids = xb
        _ = ivf.add(x)
        state_dict = ivf.state_dict()
        new_ivf = InvertedFile(storage, NLISTS)
        new_ivf.load_state_dict(state_dict)
        torch.testing.assert_close(new_ivf.invlists, ivf.invlists)
