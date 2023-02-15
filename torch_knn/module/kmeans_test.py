import pytest
import torch
from torch_knn import utils
from torch_knn.constants import CentroidsInit, Metric
from torch_knn.module.kmeans import Kmeans, ParallelKmeans

N = 100
D = 8
C = 4
M = 2


class TestKmeans:
    @pytest.mark.parametrize("metric", list(Metric))
    @pytest.mark.parametrize("init", list(CentroidsInit))
    def test___init__(self, metric, init):
        if init not in {CentroidsInit.RANDOM}:
            with pytest.raises(NotImplementedError):
                Kmeans(C, D, metric=metric, init=init)
        else:
            kmeans = Kmeans(C, D, metric=metric, init=init)
            assert kmeans.ncentroids == C
            assert kmeans.dim == D
            assert kmeans.metric == metric

    @pytest.mark.parametrize("init", list(CentroidsInit))
    def test_init_centroids(self, init):
        if init not in {CentroidsInit.RANDOM}:
            with pytest.raises(NotImplementedError):
                Kmeans(C, D, init=init)

    def test_assign(self):
        x = torch.rand(N, D)
        kmeans = Kmeans(C, D)
        centroids = kmeans.centroids
        assigns = kmeans.assign(x)
        expected = ((x[:, None] - centroids[None, :]) ** 2).sum(dim=-1).argmin(dim=1)
        assert torch.equal(assigns, expected)

    def test_update(self):
        x = torch.rand(N, D)
        assigns = torch.randint(0, C, (N,))
        kmeans = Kmeans(C, D)
        new_centroids = kmeans.update(x, assigns)
        assert torch.allclose(new_centroids, kmeans.centroids)

        expected = torch.zeros(C, D)
        for c in range(C):
            expected[c] = x[assigns == c].mean(dim=0)
        assert torch.allclose(new_centroids, expected)

    def test_train(self):
        torch.manual_seed(0)
        x = torch.rand(N, D)
        kmeans = Kmeans(C, D)
        centroids = kmeans.train(x)
        assert utils.is_equal_shape(centroids, [C, D])


class TestParallelKmeans:
    @pytest.mark.parametrize("metric", list(Metric))
    @pytest.mark.parametrize("init", list(CentroidsInit))
    def test___init__(self, metric, init):
        if init not in {CentroidsInit.RANDOM}:
            with pytest.raises(NotImplementedError):
                ParallelKmeans(C, D, M, metric=metric, init=init)
        else:
            kmeans = ParallelKmeans(C, D, M, metric=metric, init=init)
            assert kmeans.ncentroids == C
            assert kmeans.dim == D
            assert kmeans.nspaces == M
            assert kmeans.metric == metric

    @pytest.mark.parametrize("init", list(CentroidsInit))
    def test_init_centroids(self, init):
        if init not in {CentroidsInit.RANDOM}:
            with pytest.raises(NotImplementedError):
                ParallelKmeans(C, D, M, init=init)

    def test_assign(self):
        x = torch.rand(M, N, D)
        kmeans = ParallelKmeans(C, D, M)
        centroids = kmeans.centroids
        assigns = kmeans.assign(x)
        expected = torch.cdist(x, centroids).argmin(dim=-1)
        assert torch.equal(assigns, expected)

    def test_update(self):
        x = torch.rand(M, N, D)
        assigns = torch.randint(0, C, (M, N))
        kmeans = ParallelKmeans(C, D, M)
        new_centroids = kmeans.update(x, assigns)
        assert torch.allclose(new_centroids, kmeans.centroids)

        expected = torch.zeros(M, C, D)
        for m in range(M):
            for c in range(C):
                expected[m, c] = x[m][assigns[m] == c].mean(dim=0)
        assert torch.allclose(new_centroids, expected)

    def test_train(self):
        torch.manual_seed(0)
        x = torch.rand(N, M, D)
        kmeans = ParallelKmeans(C, D, M)
        centroids = kmeans.train(x)
        assert utils.is_equal_shape(centroids, [M, C, D])
