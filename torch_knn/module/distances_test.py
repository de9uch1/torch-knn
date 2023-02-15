import pytest
import torch
from torch_knn.constants import Metric
from torch_knn.module.distances import compute_distance


@pytest.mark.parametrize("metric", [Metric.L2, Metric.IP, Metric.COS])
def test_compute_distance(metric):
    a = torch.rand(3, 8)
    b = torch.rand(4, 8)

    if metric == Metric.L2:
        distance = compute_distance(a, b, metric)
        expected = ((a[:, None] - b[None, :]) ** 2).sum(dim=-1) ** 0.5
    elif metric == Metric.IP:
        distance = compute_distance(a, b, metric)
        expected = a.mm(b.transpose(0, 1))
        torch.testing.assert_close(distance, expected)
    else:
        with pytest.raises(NotImplementedError):
            compute_distance(a, b, metric)
