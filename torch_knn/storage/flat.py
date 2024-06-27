import torch

from .base import Storage


class StorageFlat(Storage):
    """Flat storage class.

    StorageFlat stores the `(N, D)` vectors.

    Args:
        cfg (StorageFlat.Config): Configuration for this class.
    """

    cfg: "StorageFlat.Config"

    @property
    def data(self) -> torch.Tensor:
        """Storage object of shape `(N, D)`."""
        return self._data

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the given vectors.

        StorageFlat class returns the identity mapping of `x`.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            torch.Tensor: Encoded vectors of shape `(N, D)`.
        """
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes the given vectors.

        StorageFlat class returns the identity mapping of `x`.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            torch.Tensor: Decoded vectors of shape `(N, D)`.
        """
        return x

    def fit(self, x: torch.Tensor) -> "StorageFlat":
        """Trains the index with the given vectors.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            StorageFlat: The storage object.
        """
        return self
