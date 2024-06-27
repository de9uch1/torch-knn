import abc

import torch
from torch import Tensor


class Metric(abc.ABC):
    """Base class for metric classes."""

    @staticmethod
    @abc.abstractmethod
    def compute_distance(a: Tensor, b: Tensor) -> Tensor:
        """Computes distance between two vectors.

        Args:
            a (torch.Tensor): Input vectors of shape `(n, dim)` or `(b, n, dim)`.
            b (torch.Tensor): Input vectors of shape `(m, dim)` or `(b, m, dim)`.

        Returns:
            torch.Tensor: Distance tensor of shape `(n, m)` or `(b, n, m)`.
        """

    @staticmethod
    def topk(distances: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Gets k-nearest-neighbors under the metric.

        Args:
            distances (torch.Tensor): Distance table of shape `(n, m)`.
            k (int): Top-k width.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
              - torch.Tensor: Distances between querys and keys of shape `(Nq, k)`.
              - torch.Tensor: Indices of the k-nearest-neighbors of shape `(Nq, k)`.
        """
        return torch.topk(distances, k=k, dim=-1, largest=False)

    @classmethod
    def assign(cls, querys: Tensor, keys: Tensor) -> Tensor:
        """Assigns the nearest neighbor IDs.

        Args:
            querys (torch.Tensor): Input vectors of shape `(n, dim)` or `(b, n, dim)`.
            keys (torch.Tensor): Input vectors of shape `(m, dim)` or `(b, m, dim)`.

        Returns:
            torch.Tensor: Indices of the nearest-neighbors of shape `(n,)`.
        """
        return cls.compute_distance(querys, keys).argmin(dim=-1)

    @property
    def farthest_value(self) -> float:
        """Returns the farthest value under this metric."""
        return float("inf")

    def mask(self, distances: Tensor, padding_mask: Tensor) -> Tensor:
        """Masks the distance tensor with the padding mask.

        Args:
            distances (torch.Tensor): Distance tensor of shape `(..., n, m)`.
            padding_mask (torch.Tensor): Padding boolean mask of shape `(..., n, m)`.

        Returns:
            torch.Tensor: Masked distance tensor.
        """
        return distances.masked_fill_(padding_mask, self.farthest_value)


class MetricL2(Metric):
    """L2 metric for squared Euclidean distance computation."""

    @staticmethod
    def compute_distance(a: Tensor, b: Tensor) -> Tensor:
        """Computes distance between two vectors.

        This method uses `torch.cdist()`, which switches the following two modes
        according to the sizes of the input tensors for the computation performance.

        - SIMD-based: Compute :math:`||a - b||`
        - BLAS-based: Compute :math:`||a||, ||b||`, and :math:`-2ab^T`

        Args:
            a (torch.Tensor): Input vectors of shape `(n, dim)` or `(b, n, dim)`.
            b (torch.Tensor): Input vectors of shape `(m, dim)` or `(b, m, dim)`.

        Returns:
            torch.Tensor: Distance tensor of shape `(n, m)` or `(b, n, m)`.
        """
        return torch.cdist(a, b, p=2) ** 2


class MetricIP(Metric):
    """IP metric for inner product distance computation."""

    @staticmethod
    def compute_distance(a: Tensor, b: Tensor) -> Tensor:
        """Computes distance between two vectors.

        Args:
            a (torch.Tensor): Input vectors of shape `(n, dim)` or `(b, n, dim)`.
            b (torch.Tensor): Input vectors of shape `(m, dim)` or `(b, m, dim)`.

        Returns:
            torch.Tensor: Distance tensor of shape `(n, m)` or `(b, n, m)`.
        """
        return torch.einsum("...nd,...md->...nm", a, b)

    @staticmethod
    def topk(distances: Tensor, k: int) -> tuple[Tensor, Tensor]:
        """Gets k-nearest-neighbors under the metric.

        Args:
            distances (torch.Tensor): Distance table of shape `(n, m)`.
            k (int): Top-k width.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
              - torch.Tensor: Distances between querys and keys of shape `(Nq, k)`.
              - torch.Tensor: Indices of the k-nearest-neighbors of shape `(Nq, k)`.
        """
        return torch.topk(distances, k=k, dim=-1, largest=True)

    @classmethod
    def assign(cls, querys: Tensor, keys: Tensor) -> Tensor:
        """Assigns the nearest neighbor IDs.

        Args:
            querys (torch.Tensor): Input vectors of shape `(n, dim)` or `(b, n, dim)`.
            keys (torch.Tensor): Input vectors of shape `(m, dim)` or `(b, m, dim)`.

        Returns:
            torch.Tensor: Indices of the nearest-neighbors of shape `(n,)`.
        """
        return cls.compute_distance(querys, keys).argmax(dim=-1)

    @property
    def farthest_value(self) -> float:
        """Returns the farthest value under this metric."""
        return float("-inf")
