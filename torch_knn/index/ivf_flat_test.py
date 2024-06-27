import pytest
import torch

from torch_knn import metrics, utils
from torch_knn.index.ivf_flat import IndexIVFFlat
from torch_knn.module.ivf import InvertedFile

D = 8
NLISTS = 4
N = 100


class TestIndexIVFFlat:
    def test___init__(self):
        index = IndexIVFFlat(IndexIVFFlat.Config(D, nlists=NLISTS))
        assert isinstance(index.ivf, InvertedFile)

    def test_centroids(self):
        index = IndexIVFFlat(IndexIVFFlat.Config(D, nlists=NLISTS))
        x = torch.rand(N, D)
        index.fit(x)
        assert utils.is_equal_shape(index.centroids, [NLISTS, D])

    def test_fit(self):
        index = IndexIVFFlat(IndexIVFFlat.Config(D, nlists=NLISTS))
        x = torch.rand(N, D)
        index.fit(x)
        assert index.ivf.centroids is not None and utils.is_equal_shape(
            index.ivf.centroids, [NLISTS, D]
        )

    def test_add(self):
        index = IndexIVFFlat(IndexIVFFlat.Config(D, nlists=NLISTS))
        x = torch.rand(N, D)
        index.fit(x)
        assert index.N == 0
        index.add(x)
        assert index.N == N
        assert torch.cat(index.ivf.invlists).size(0) == N
        assert torch.equal(index.data, x)
        index.add(x)
        assert index.N == 2 * N

    @pytest.mark.parametrize("nprobe", [1, 3])
    @pytest.mark.parametrize("k", [1, 4])
    def test_search(self, k: int, nprobe: int):
        metric = metrics.MetricL2()
        torch.manual_seed(0)
        index = IndexIVFFlat(IndexIVFFlat.Config(D, metric=metric, nlists=NLISTS))
        x = torch.rand(N, D)
        index.fit(x)
        index.add(x)
        dists, idxs = index.search(x, k=k, nprobe=nprobe)
        assert utils.is_equal_shape(dists, idxs)
        assert utils.is_equal_shape(dists, [N, k])
        torch.testing.assert_close(dists[:, 0], torch.zeros(dists[:, 0].shape))

        if isinstance(metric, metrics.MetricL2):
            torch.testing.assert_close(idxs[:, 0], torch.arange(N))
