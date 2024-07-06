import pytest
import torch

from torch_knn import utils
from torch_knn.metrics import MetricIP, MetricL2
from torch_knn.module.kmeans import Kmeans, ParallelKmeans

N = 100
D = 8
C = 4
M = 2


class TestKmeans:
    @pytest.mark.parametrize("metric", [MetricL2(), MetricIP()])
    @pytest.mark.parametrize("init", list(Kmeans.Init))
    def test___init__(self, metric, init):
        kmeans = Kmeans(C, D, metric=metric, init=init)
        assert kmeans.ncentroids == C
        assert kmeans.dim == D
        assert kmeans.metric == metric
        assert kmeans.init == init

    def test_centroids(self):
        kmeans = Kmeans(C, D)
        x = torch.rand(N, D)
        kmeans.fit(x)
        assert utils.is_equal_shape(kmeans.centroids, [C, D])

        kmeans = Kmeans(C, D)
        x = torch.rand(N, D)
        with pytest.raises(ValueError):
            kmeans.centroids = x
        x = torch.rand(C, D)
        kmeans.centroids = x
        assert torch.equal(kmeans.centroids, x)

    def test_centroids_shape(self):
        kmeans = Kmeans(C, D)
        assert kmeans.centroids_shape == (C, D)

    def test_assign(self):
        x = torch.rand(N, D)
        centroids = torch.rand(C, D)
        kmeans = Kmeans(C, D)
        kmeans.centroids = centroids
        assigns = kmeans.assign(x)
        expected = ((x[:, None] - centroids[None, :]) ** 2).sum(dim=-1).argmin(dim=1)
        assert torch.equal(assigns, expected)

    def test_update(self):
        x = torch.rand(N, D)
        centroids = torch.rand(C, D)
        assigns = torch.randint(0, C, (N,))
        kmeans = Kmeans(C, D)
        kmeans.centroids = centroids
        new_centroids = kmeans.update(x, assigns)
        assert torch.allclose(new_centroids, kmeans.centroids)

        expected = torch.zeros(C, D)
        for c in range(C):
            expected[c] = x[assigns == c].mean(dim=0)
        assert torch.allclose(new_centroids, expected)

    @pytest.mark.parametrize("init", list(Kmeans.Init))
    def test_fit(self, init):
        torch.manual_seed(0)
        kmeans = Kmeans(C, D, init=init)
        x = torch.rand(N, D)
        centroids = kmeans.fit(x)
        assert utils.is_equal_shape(centroids, [C, D])


class TestParallelKmeans:
    @pytest.mark.parametrize("metric", [MetricL2(), MetricIP()])
    @pytest.mark.parametrize("init", list(ParallelKmeans.Init))
    def test___init__(self, metric, init):
        kmeans = ParallelKmeans(C, D, M, metric=metric, init=init)
        assert kmeans.ncentroids == C
        assert kmeans.dim == D
        assert kmeans.nspaces == M
        assert kmeans.metric == metric
        assert kmeans.init == init

    def test_centroids(self):
        kmeans = ParallelKmeans(C, D, M)
        x = torch.rand(M, N, D)
        kmeans.fit(x)
        assert utils.is_equal_shape(kmeans.centroids, [M, C, D])

        kmeans = ParallelKmeans(C, D, M)
        with pytest.raises(ValueError):
            kmeans.centroids = torch.rand(N, M, D)

        x = torch.rand(M, C, D)
        kmeans.centroids = x
        assert torch.equal(kmeans.centroids, x)

    def test_centroids_shape(self):
        kmeans = ParallelKmeans(C, D, M)
        assert kmeans.centroids_shape == (M, C, D)

    def test_assign(self):
        x = torch.rand(M, N, D)
        centroids = torch.rand(M, C, D)
        kmeans = ParallelKmeans(C, D, M)
        kmeans.centroids = centroids
        assigns = kmeans.assign(x)
        expected = torch.cdist(x, centroids).argmin(dim=-1)
        assert torch.equal(assigns, expected)

    def test_update(self):
        x = torch.rand(M, N, D)
        centroids = torch.rand(M, C, D)
        assigns = torch.randint(0, C, (M, N))
        kmeans = ParallelKmeans(C, D, M)
        kmeans.centroids = centroids
        new_centroids = kmeans.update(x, assigns)
        assert torch.allclose(new_centroids, kmeans.centroids)

        expected = torch.zeros(M, C, D)
        for m in range(M):
            for c in range(C):
                expected[m, c] = x[m][assigns[m] == c].mean(dim=0)
        assert torch.allclose(new_centroids, expected)

    @pytest.mark.parametrize("init", list(ParallelKmeans.Init))
    def test_fit(self, init):
        torch.manual_seed(0)
        x = torch.rand(M, N, D)
        kmeans = ParallelKmeans(C, D, M, init=init)
        centroids = kmeans.fit(x)
        assert utils.is_equal_shape(centroids, [M, C, D])
