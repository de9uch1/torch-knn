import pytest
import torch
from torch_knn import utils
from torch_knn.index.linear_flat import LinearFlatIndex

N = 10
D = 8


class TestLinearFlatIndex:
    @pytest.mark.parametrize("k", [1, 2, 8])
    def test_search(self, k):
        x = torch.rand(N, D)
        index = LinearFlatIndex(LinearFlatIndex.Config(D))
        index.add(x)
        dists, idxs = index.search(x, k=k)
        # Shape
        assert utils.is_equal_shape(dists, idxs)
        assert utils.is_equal_shape(dists, [N, k])
        # Exact search
        assert torch.allclose(dists[:, 0], torch.zeros(dists.size(0)))
        assert torch.equal(idxs[:, 0], torch.arange(N))
