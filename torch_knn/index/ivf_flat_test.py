import pytest
import torch

from torch_knn import metrics, utils
from torch_knn.index.ivf_flat import IVFFlatIndex
from torch_knn.module.ivf import InvertedFile

D = 8
NLISTS = 4
N = 100


class TestIVFFlatIndex:
    def test___init__(self):
        index = IVFFlatIndex(IVFFlatIndex.Config(D, nlists=NLISTS))
        assert isinstance(index.ivf, InvertedFile)

    def test_centroids(self):
        index = IVFFlatIndex(IVFFlatIndex.Config(D, nlists=NLISTS))
        with pytest.raises(RuntimeError):
            index.centroids
        x = torch.rand(N, D)
        index.train(x)
        assert utils.is_equal_shape(index.centroids, [NLISTS, D])

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

    @pytest.mark.parametrize("metric", [metrics.L2Metric(), metrics.CosineMetric()])
    @pytest.mark.parametrize("nprobe", [1, 3])
    @pytest.mark.parametrize("k", [1, 4])
    def test_search(self, metric: metrics.Metric, k: int, nprobe: int):
        torch.manual_seed(0)
        index = IVFFlatIndex(IVFFlatIndex.Config(D, metric=metric, nlists=NLISTS))
        x = torch.rand(N, D)
        index.train(x)
        index.add(x)
        dists, idxs = index.search(x, k=k, nprobe=nprobe)
        assert utils.is_equal_shape(dists, idxs)
        assert utils.is_equal_shape(dists, [N, k])
        if isinstance(metric, metrics.CosineMetric):
            torch.testing.assert_close(dists[:, 0], torch.ones(dists[:, 0].shape))
        else:
            torch.testing.assert_close(dists[:, 0], torch.zeros(dists[:, 0].shape))

        if isinstance(metric, metrics.L2Metric):
            torch.testing.assert_close(idxs[:, 0], torch.arange(N))
