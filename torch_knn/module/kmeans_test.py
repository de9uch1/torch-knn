import pytest
import torch
from torch_knn import utils
from torch_knn.constants import CentroidsInit, Metric
from torch_knn.module.kmeans import Kmeans

N = 100
D = 8
C = 4


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
        assert torch.equal(new_centroids, expected)

    def test_train(self):
        torch.manual_seed(0)
        x = torch.rand(N, D)
        kmeans = Kmeans(C, D)
        centroids = kmeans.train(x)
        assert utils.is_equal_shape(centroids, [C, D])
