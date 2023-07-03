import torch

from torch_knn.module.ivf import InvertedFile
from torch_knn.storage.flat import FlatStorage

N = 1024
D = 4
NLISTS = 8


class TestInvertedFile:
    def test___init__(self):
        storage = FlatStorage(FlatStorage.Config(D))
        ivf = InvertedFile(storage, NLISTS)
        assert ivf.metric == storage.metric
        assert ivf.nlists == NLISTS
        assert len(ivf.invlists) == NLISTS

    def test_add(self):
        storage = FlatStorage(FlatStorage.Config(D))
        ivf = InvertedFile(storage, NLISTS)
        torch.manual_seed(0)
        x = torch.rand(N, D)
        xb = x[:NLISTS]
        ivf.centroids = xb
        assigns = ivf.add(xb)
        assert torch.equal(assigns, torch.arange(NLISTS))
        assert all([len(plist) == 1 for plist in ivf.invlists])
