import torch
from torch_knn.storage.base import Storage


class FlatStorage(Storage):
    """Flat storage class.

    FlatStorage stores the `(N, D)` vectors.

    Args:
        cfg (FlatStorage.Config): Configuration for this class.
        storage (torch.Tensor): The storage object.
    """

    cfg: "FlatStorage.Config"

    def check_shape(self, storage: torch.Tensor) -> torch.Tensor:
        """Checks whether the storage tensor shape is valid or not.

        Args:
            storage (torch.Tensor): The storage tensor.

        Returns:
            torch.Tensor: The input storage tensor.

        Raises:
            ValueError: When given the wrong shape storage.
        """
        if storage.dim() != 2 or storage.size(-1) != self.D:
            raise ValueError(f"The storage must be `N x D` dimensions.")
        return storage

    @property
    def storage(self) -> torch.Tensor:
        """Storage object of shape `(N, D)`."""
        return self._storage

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
            FlatStorage: The trained storage object.
        """
        return self

    @property
    def is_trained(self) -> bool:
        """Returns whether the storage is trained or not."""
        return True
