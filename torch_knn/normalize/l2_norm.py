import torch.nn.functional as F
from torch import Tensor

from .base import Normalize


class L2NormNormalize(Normalize):
    """L2 norm normalization class."""

    def encode(self, x) -> Tensor:
        """Transforms the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d)`.

        Returns:
            Tensor: Normalized vectors of shape `(n, d)`.
        """
        return F.normalize(x, dim=-1)
