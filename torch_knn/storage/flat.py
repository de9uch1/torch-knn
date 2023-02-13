import torch
from torch_knn.storage.base import Storage


class FlatStorage(Storage):
    """Flat storage class.

    FlatStorage stores the `(N x D)` vectors.

    Args:
        storage (torch.Tensor): The storage object.
        dtype (torch.dtype): The storage dtype. (default: torch.float32)
    """

    @classmethod
    def check_shape(cls, storage: torch.Tensor) -> torch.Tensor:
        """Checks whether the storage tensor shape is valid or not.

        Args:
            storage (torch.Tensor): The storage tensor.

        Returns:
            torch.Tensor: The input storage tensor.

        Raises:
            ValueError: When given the wrong shape storage.
        """
        if storage.dim() != 2:
            raise ValueError(f"The storage must be `N x D` dimensions.")
        return storage

    @property
    def storage(self) -> torch.Tensor:
        """Storage object of shape `(N, D)`."""
        return self._storage
