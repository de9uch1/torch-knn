import pytest
import torch

from torch_knn import metrics, utils
from torch_knn.index.linear_pq import LinearPQIndex

D = 8
M = 4
ksub = 16
N = ksub * 4
Nq = 3


class TestLinearPQIndex:
    @pytest.mark.parametrize("k", [1, 2, 8])
    def test_search(self, k: int):
        metric = metrics.L2Metric()

        torch.manual_seed(0)
        x = torch.rand(N, D)
        index = LinearPQIndex(LinearPQIndex.Config(D, metric=metric, M=M, ksub=ksub))
        index.fit(x)
        index.add(x)
        xq = x[:Nq]
        dists, idxs = index.search(xq, k=k)
        # Shape
        assert utils.is_equal_shape(dists, idxs)
        assert utils.is_equal_shape(dists, [Nq, k])
        # Self search
        if isinstance(metric, metrics.L2Metric):
            assert torch.equal(idxs[:, 0], torch.arange(Nq))
