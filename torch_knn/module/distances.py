import torch
from torch_knn.constants import Metric


def compute_distance(a: torch.Tensor, b: torch.Tensor, metric: Metric) -> torch.Tensor:
    """Computes distance between two vectors.

    Args:
        a (torch.Tensor): Input vectors of shape `(n, dim)`.
        b (torch.Tensor): Input vectors of shape `(m, dim)`.

    Returns:
        torch.Tensor: Distance tensor of shape `(n, m)`.
    """
    if metric == Metric.L2:
        return torch.cdist(a, b)
    elif metric == Metric.IP:
        return torch.einsum("nd,md->nm", a, b)
    else:
        raise NotImplementedError
