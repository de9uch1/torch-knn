from dataclasses import dataclass

import torch.nn.functional as F
from torch import Tensor

from torch_knn.transform.base import Transform


class L2NormalizationTransform(Transform):
    """L2 normalization transform class."""

    @dataclass
    class Config(Transform.Config):
        d_in: int = -1
        d_out: int = -1

    cfg: "L2NormalizationTransform.Config"

    @property
    def is_trained(self) -> bool:
        """Returns whether this class is trained or not."""
        return True

    def train(self, x) -> "L2NormalizationTransform":
        """Trains vector transformation for this class.

        Args:
            x (Tensor): Training vectors of shape `(n, d_in)`.

        Returns:
            L2NormalizationTransform: Trained this class.
        """
        return self

    def encode(self, x) -> Tensor:
        """Transforms the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d_in)`.

        Returns:
            Tensor: Transformed vectors of shape `(n, d_out)`.
        """
        return F.normalize(x, dim=-1)

    def decode(self, x) -> Tensor:
        """Returns the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d_out)`.

        Returns:
            Tensor: Input vectors of shape `(n, d_in)`.
        """
        return x
