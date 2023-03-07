import pytest
import torch

from torch_knn import metrics, utils
from torch_knn.index.linear_flat import LinearFlatIndex

N = 10
D = 8


class TestLinearFlatIndex:
    @pytest.mark.parametrize("metric", [metrics.L2Metric(), metrics.IPMetric()])
    @pytest.mark.parametrize("k", [1, 2, 8])
    def test_search(self, metric: metrics.Metric, k: int):
        torch.manual_seed(0)
        x = torch.rand(N, D)
        index = LinearFlatIndex(LinearFlatIndex.Config(D, metric=metric))
        index.add(x)
        dists, idxs = index.search(x, k=k)
        # Shape
        assert utils.is_equal_shape(dists, idxs)
        assert utils.is_equal_shape(dists, [N, k])

        if isinstance(metric, metrics.IPMetric):
            distance_matrix = x @ x.T
            expected_dists, expected_idxs = distance_matrix.topk(k=k, dim=-1)
            assert torch.allclose(dists, expected_dists)
            assert torch.allclose(idxs, expected_idxs)

        # Exact search
        if isinstance(metric, metrics.L2Metric):
            assert torch.equal(idxs[:, 0], torch.arange(N))
