import pytest
import torch

from torch_knn import utils
from torch_knn.metrics import CosineMetric, IPMetric, L2Metric
from torch_knn.module.kmeans import Kmeans, ParallelKmeans

N = 100
D = 8
C = 4
M = 2


class TestKmeans:
    @pytest.mark.parametrize("metric", [L2Metric(), IPMetric(), CosineMetric()])
    @pytest.mark.parametrize("init", list(Kmeans.Init))
    def test___init__(self, metric, init):
        kmeans = Kmeans(C, D, metric=metric, init=init)
        assert kmeans.ncentroids == C
        assert kmeans.dim == D
        assert kmeans.metric == metric
        assert kmeans.init == init

    @pytest.mark.parametrize("init", list(Kmeans.Init))
    def test_init_centroids(self, init):
        kmeans = Kmeans(C, D, init=init)
        if init not in {Kmeans.Init.RANDOM}:
            with pytest.raises(NotImplementedError):
                kmeans.init_centroids(init)
        else:
            assert utils.is_equal_shape(kmeans.init_centroids(init), [C, D])

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

    def test_is_trained(self):
        torch.manual_seed(0)
        kmeans = Kmeans(C, D)
        x = torch.rand(N, D)
        assert not kmeans.is_trained
        kmeans.train(x)
        assert kmeans.is_trained

    @pytest.mark.parametrize("init", list(Kmeans.Init))
    def test_train(self, init):
        torch.manual_seed(0)
        kmeans = Kmeans(C, D, init=init)
        x = torch.rand(N, D)
        if init not in {Kmeans.Init.RANDOM}:
            with pytest.raises(NotImplementedError):
                kmeans.train(x)
        else:
            centroids = kmeans.train(x)
            assert utils.is_equal_shape(centroids, [C, D])


class TestParallelKmeans:
    @pytest.mark.parametrize("metric", [L2Metric(), IPMetric(), CosineMetric()])
    @pytest.mark.parametrize("init", list(ParallelKmeans.Init))
    def test___init__(self, metric, init):
        kmeans = ParallelKmeans(C, D, M, metric=metric, init=init)
        assert kmeans.ncentroids == C
        assert kmeans.dim == D
        assert kmeans.nspaces == M
        assert kmeans.metric == metric
        assert kmeans.init == init

    @pytest.mark.parametrize("init", list(ParallelKmeans.Init))
    def test_init_centroids(self, init):
        kmeans = ParallelKmeans(C, D, M, init=init)
        if init not in {ParallelKmeans.Init.RANDOM}:
            with pytest.raises(NotImplementedError):
                kmeans.init_centroids(init)
        else:
            assert utils.is_equal_shape(kmeans.init_centroids(init), [M, C, D])

    def test_assign(self):
        x = torch.rand(M, N, D)
        centroids = torch.rand(C, D)
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
    def test_train(self, init):
        torch.manual_seed(0)
        x = torch.rand(N, M, D)
        kmeans = ParallelKmeans(C, D, M, init=init)
        if init not in {ParallelKmeans.Init.RANDOM}:
            with pytest.raises(NotImplementedError):
                kmeans.train(x)
        else:
            centroids = kmeans.train(x)
            assert utils.is_equal_shape(centroids, [M, C, D])
