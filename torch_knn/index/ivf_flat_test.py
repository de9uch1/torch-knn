from typing import Tuple

import pytest
import torch
from torch_knn import utils
from torch_knn.index.ivf_flat import IVFFlatIndex
from torch_knn.module.ivf import InvertedFile

D = 8
NLISTS = 4
N = 100


class TestIVFFlatIndex:
    """Inverted file index class.

    Args:
        cfg (IVFIndex.Config): Configuration for this class.
    """

    def test___init__(self):
        index = IVFFlatIndex(IVFFlatIndex.Config(D, nlists=NLISTS))
        assert isinstance(index.ivf, InvertedFile)

    @property
    def test_is_trained(self):
        index = IVFFlatIndex(IVFFlatIndex.Config(D, nlists=NLISTS))
        x = torch.rand(N, D)
        assert not index.is_trained
        index.ivf.train(x)
        assert index.is_trained

    def test_train(self):
        index = IVFFlatIndex(IVFFlatIndex.Config(D, nlists=NLISTS))
        x = torch.rand(N, D)
        index.train(x)
        assert index.ivf.is_trained and index.is_trained
        assert index.ivf.centroids is not None and utils.is_equal_shape(
            index.ivf.centroids, [NLISTS, D]
        )

    @property
    def test_nprobe(self):
        index = IVFFlatIndex(IVFFlatIndex.Config(D, nlists=NLISTS))
        assert index.nprobe == 1
        index.nprobe = 16
        assert index.nprobe == 16
        index.nprobe = 0
        assert index.nprobe == 1
        index.nprobe = -1
        assert index.nprobe == 1

    def test_add(self):
        index = IVFFlatIndex(IVFFlatIndex.Config(D, nlists=NLISTS))
        x = torch.rand(N, D)
        index.train(x)
        assert index.N == 0
        index.add(x)
        assert index.N == N
        assert torch.cat(index.ivf.invlists).size(0) == N
        assert torch.equal(index.data, x)
        index.add(x)
        assert index.N == 2 * N

    @pytest.mark.parametrize("k", [1, 4])
    def test_search(self, k: int):
        torch.manual_seed(0)
        index = IVFFlatIndex(IVFFlatIndex.Config(D, nlists=NLISTS))
        x = torch.rand(N, D)
        index.train(x)
        index.add(x)
        dists, idxs = index.search(x, k=k)
        assert utils.is_equal_shape(dists, idxs)
        assert utils.is_equal_shape(dists, [N, k])
        torch.testing.assert_close(dists[:, 0], torch.zeros(dists[:, 0].shape))
        torch.testing.assert_close(idxs[:, 0], torch.arange(N))

