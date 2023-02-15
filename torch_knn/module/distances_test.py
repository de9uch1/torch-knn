import pytest
import torch
from torch_knn.constants import Metric
from torch_knn.module.distances import compute_distance


@pytest.mark.parametrize("a", [torch.rand(3, 8), torch.rand(5, 3, 8)])
@pytest.mark.parametrize("b", [torch.rand(4, 8), torch.rand(5, 4, 8)])
@pytest.mark.parametrize("metric", [Metric.L2, Metric.IP, Metric.COS])
def test_compute_distance(a, b, metric):
    if metric == Metric.L2:
        distance = compute_distance(a, b, metric)
        expected = (a.unsqueeze(-2) - b.unsqueeze(-3)).sum(dim=-1) ** 0.5
    elif metric == Metric.IP:
        distance = compute_distance(a, b, metric)
        expected = torch.matmul(a, (b.transpose(-1, -2)))
        torch.testing.assert_close(distance, expected)
    else:
        with pytest.raises(NotImplementedError):
            compute_distance(a, b, metric)
