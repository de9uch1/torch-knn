from typing import Tuple

import torch

from torch_knn.storage.flat import FlatStorage


class LinearFlatIndex(FlatStorage):
    """Flat linear scan index.

    Args:
        cfg (LinearFlatIndex.Config): Configuration for this class.
    """

    def add(self, x: torch.Tensor) -> None:
        """Adds the given vectors to the index.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.
        """
        x = self.transform(x)
        super().add(x)

    def search(
        self, query: torch.Tensor, k: int = 1, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Searches the k-nearest-neighbor vectors.

        Args:
            query (torch.Tensor): Query vectors of shape `(Nq, D)`.
            k (int): Number of nearest neighbors to be returned.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
              - torch.Tensor: Distances between querys and keys of shape `(Nq, k)`.
              - torch.Tensor: Indices of the k-nearest-neighbors of shape `(Nq, k)`.
        """
        query = self.transform(query)
        distances = self.metric.compute_distance(query, self.data)
        return self.metric.topk(distances, k=k)
