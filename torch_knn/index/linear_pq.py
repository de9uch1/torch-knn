from typing import Tuple

import torch

from torch_knn.storage.pq import PQStorage


class LinearPQIndex(PQStorage):
    """PQ linear scan index.

    Args:
        cfg (LinearPQIndex.Config): Configuration for this class.
    """

    def add(self, x: torch.Tensor) -> None:
        """Adds the given vectors to the index.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.
        """
        x = self.transform(x)
        super().add(x)

    def train(self, x: torch.Tensor) -> "PQStorage":
        """Trains the index with the given vectors.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            LinearPQIndex: The trained index object.
        """
        return super().train(self.transform(x))

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
        adtable = self.compute_adtable(query)
        distances = adtable.lookup(self.data)
        return self.metric.topk(distances, k=k)
