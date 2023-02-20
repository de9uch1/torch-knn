import torch
import pytest

from torch_knn import utils
from torch_knn.index.linear_pq import LinearPQIndex

D = 8
M = 4
ksub = 256
N = ksub * 8
Nq = 3


class TestLinearPQIndex:
    @pytest.mark.parametrize("k", [1, 2, 8])
    def test_search(self, k):
        torch.manual_seed(0)
        x = torch.rand(N, D)
        index = LinearPQIndex(LinearPQIndex.Config(D, M=M, ksub=16))
        index.train(x)
        index.add(x)
        xq = x[:Nq]
        dists, idxs = index.search(xq, k=k)
        # Shape
        assert utils.is_equal_shape(dists, idxs)
        assert utils.is_equal_shape(dists, [Nq, k])
        # Self search
        assert torch.less(dists[:, 0].mean(), 0.1)
        assert torch.equal(idxs[:, 0], torch.arange(Nq))
        
