import pytest
import torch
from torch import Tensor
from torch_knn.module.metrics import IPMetric, L2Metric, Metric

B = 3
N = 2
M = 8


class MetricMock(Metric):
    @staticmethod
    def compute_distance(a: Tensor, b: Tensor) -> Tensor:
        return torch.cdist(a, b) ** 2


class TestMetric:
    @pytest.mark.parametrize("distances", [torch.rand(N, M), torch.rand(B, N, M)])
    @pytest.mark.parametrize("k", [1, 2, 8])
    def test_topk(self, distances: Tensor, k: int):
        k_distances, k_indices = Metric.topk(distances, k=k)
        expected_indices = torch.argsort(distances, dim=-1)[..., : min(k, M)]
        expected_distances = distances.take_along_dim(expected_indices, -1)
        torch.testing.assert_close(k_distances, expected_distances)
        torch.testing.assert_close(k_indices, expected_indices)

    @pytest.mark.parametrize("a", [torch.rand(3, 8), torch.rand(5, 3, 8)])
    @pytest.mark.parametrize("b", [torch.rand(4, 8), torch.rand(5, 4, 8)])
    def test_assign(self, a, b):
        assignments = MetricMock.assign(a, b)
        expected = ((a.unsqueeze(-2) - b.unsqueeze(-3)) ** 2).sum(dim=-1).argmin(dim=-1)
        torch.testing.assert_close(assignments, expected)


class TestL2Metric:
    @pytest.mark.parametrize("a", [torch.rand(3, 8), torch.rand(5, 3, 8)])
    @pytest.mark.parametrize("b", [torch.rand(4, 8), torch.rand(5, 4, 8)])
    def test_compute_distance(self, a: Tensor, b: Tensor):
        expected = ((a.unsqueeze(-2) - b.unsqueeze(-3)) ** 2).sum(dim=-1)
        torch.testing.assert_close(L2Metric.compute_distance(a, b), expected)


class TestIPMetric:
    @pytest.mark.parametrize("a", [torch.rand(3, 8), torch.rand(5, 3, 8)])
    @pytest.mark.parametrize("b", [torch.rand(4, 8), torch.rand(5, 4, 8)])
    def test_compute_distance(self, a: Tensor, b: Tensor):
        expected = torch.matmul(a, (b.transpose(-1, -2)))
        torch.testing.assert_close(IPMetric.compute_distance(a, b), expected)

    @pytest.mark.parametrize("distances", [torch.rand(N, M), torch.rand(B, N, M)])
    @pytest.mark.parametrize("k", [1, 2, 8])
    def test_topk(self, distances: Tensor, k: int):
        k_distances, k_indices = IPMetric.topk(distances, k=k)
        expected_indices = torch.argsort(distances, dim=-1, descending=True)[
            ..., : min(k, M)
        ]
        expected_distances = distances.take_along_dim(expected_indices, -1)
        torch.testing.assert_close(k_distances, expected_distances)
        torch.testing.assert_close(k_indices, expected_indices)

    @pytest.mark.parametrize("a", [torch.rand(3, 8), torch.rand(5, 3, 8)])
    @pytest.mark.parametrize("b", [torch.rand(4, 8), torch.rand(5, 4, 8)])
    def test_assign(self, a, b):
        assignments = IPMetric.assign(a, b)
        expected = torch.matmul(a, (b.transpose(-1, -2))).argmax(dim=-1)
        torch.testing.assert_close(assignments, expected)
