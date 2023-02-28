import pytest
import torch

from torch_knn import metrics, utils
from torch_knn.index.linear_pq import LinearPQIndex
from torch_knn.module.kmeans import ParallelKmeans

D = 8
M = 4
ksub = 16
N = ksub * 4
Nq = 3


class TestLinearPQIndex:
    @pytest.mark.parametrize("metric", [metrics.L2Metric(), metrics.CosineMetric()])
    def test_add(self, metric: metrics.Metric):
        x = torch.rand(N, D)
        index = LinearPQIndex(LinearPQIndex.Config(D, metric=metric, M=M, ksub=ksub))
        index.train(x)
        index.add(x)
        torch.testing.assert_close(index.data, index.encode(index.transform(x)))

    @pytest.mark.parametrize("metric", [metrics.L2Metric(), metrics.CosineMetric()])
    def test_train(self, metric: metrics.Metric):
        x = torch.rand(N, D)
        index = LinearPQIndex(LinearPQIndex.Config(D, metric=metric, M=M, ksub=ksub))
        torch.manual_seed(0)
        index.train(x)

        dsub = D // M
        kmeans = ParallelKmeans(ksub, dsub, M)
        torch.manual_seed(0)
        codebook = kmeans.train(index.transform(x).view(N, M, dsub))
        torch.testing.assert_close(index.codebook, codebook)

    @pytest.mark.parametrize("metric", [metrics.L2Metric(), metrics.CosineMetric()])
    @pytest.mark.parametrize("k", [1, 2, 8])
    def test_search(self, metric: metrics.Metric, k: int):
        torch.manual_seed(0)
        x = torch.rand(N, D)
        index = LinearPQIndex(LinearPQIndex.Config(D, metric=metric, M=M, ksub=ksub))
        index.train(x)
        index.add(x)
        xq = x[:Nq]
        dists, idxs = index.search(xq, k=k)
        # Shape
        assert utils.is_equal_shape(dists, idxs)
        assert utils.is_equal_shape(dists, [Nq, k])
        # Self search
        if isinstance(metric, metrics.CosineMetric):
            assert torch.greater_equal(dists[:, 0].mean(), 0.9)
        else:
            assert torch.less(dists[:, 0].mean(), 0.1)

        if isinstance(metric, metrics.L2Metric):
            assert torch.equal(idxs[:, 0], torch.arange(Nq))
