import torch

from torch_knn.storage.pq import PQStorage

from .base import Index


class LinearPQIndex(PQStorage, Index):
    """PQ linear scan index.

    Args:
        cfg (LinearPQIndex.Config): Configuration for this class.
    """

    def search(
        self, query: torch.Tensor, k: int = 1, **kwargs
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Searches the k-nearest-neighbor vectors.

        Args:
            query (torch.Tensor): Query vectors of shape `(Nq, D)`.
            k (int): Number of nearest neighbors to be returned.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
              - torch.Tensor: Distances between querys and keys of shape `(Nq, k)`.
              - torch.Tensor: Indices of the k-nearest-neighbors of shape `(Nq, k)`.
        """
        adtable = self.compute_adtable(query)
        distances = adtable.lookup(self.data)
        return self.metric.topk(distances, k=k)
