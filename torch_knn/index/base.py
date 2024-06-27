import abc

import torch

from torch_knn.storage.base import Storage


class Index(Storage, metaclass=abc.ABCMeta):
    """Base class for index classes."""

    @abc.abstractmethod
    def search(
        self, query: torch.Tensor, k: int = 1, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Searches the k-nearest-neighbor vectors.

        Args:
            query (torch.Tensor): Query vectors of shape `(Nq, D)`.
            k (int): Number of nearest neighbors to be returned.
            **kwargs (Dict[str, Any]): Keyword arguments for the search method.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
              - torch.Tensor: Distances between querys and keys of shape `(Nq, k)`.
              - torch.Tensor: Indices of the k-nearest-neighbors of shape `(Nq, k)`.
        """
