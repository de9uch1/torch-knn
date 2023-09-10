import torch

from torch_knn.storage.base import Storage


class FlatStorage(Storage):
    """Flat storage class.

    FlatStorage stores the `(N, D)` vectors.

    Args:
        cfg (FlatStorage.Config): Configuration for this class.
    """

    cfg: "FlatStorage.Config"

    @property
    def data(self) -> torch.Tensor:
        """Storage object of shape `(N, D)`."""
        return self._data

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the given vectors.

        FlatStorage class returns the identity mapping of `x`.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            torch.Tensor: Encoded vectors of shape `(N, D)`.
        """
        return x

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes the given vectors.

        FlatStorage class returns the identity mapping of `x`.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            torch.Tensor: Decoded vectors of shape `(N, D)`.
        """
        return x

    def train(self, x: torch.Tensor) -> "FlatStorage":
        """Trains the index with the given vectors.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            FlatStorage: The storage object.
        """
        return self
