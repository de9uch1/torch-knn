from dataclasses import dataclass

from torch import Tensor

from torch_knn.transform.base import Transform


class L2NormalizationTransform(Transform):
    """L2 normalization transform class."""

    @dataclass
    class Config(Transform.Config):
        """Base class for transform config.

        Args:
            d_in (int): Dimension size of input vectors.
            d_out (int): Dimension size of output vectors.
            dtype (torch.dtype): The input vector dtype. (default: torch.float32)
        """

        def __post_init__(self):
            """Validates the dimension size."""
            if self.d_in != self.d_out:
                raise ValueError("d_in must be == d_out.")

    cfg: Config

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
        return x / x.norm(dim=-1, keepdim=True)

    def decode(self, x) -> Tensor:
        """Returns the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d_out)`.

        Returns:
            Tensor: Input vectors of shape `(n, d_in)`.
        """
        return x
