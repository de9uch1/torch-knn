from typing import Tuple

import torch
from torch_knn.storage.pq import PQStorage


class LinearPQIndex(PQStorage):
    """PQ linear scan index.

    Args:
        cfg (LinearPQIndex.Config): Configuration for this class.
    """

    def search(
        self,
        query: torch.Tensor,
        k: int = 1,
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
        Nq, D = query.size()
        query = query.view(Nq, self.M, self.dsub).transpose(0, 1).contiguous()
        # adtable: Nq x M x ksub -> Nq x N x M x ksub
        # data: N x M -> Nq x N x M x 1
        # Note the Tensor.expand() does not allocate new memory.
        adtable = (
            self.metric.compute_distance(query, self.codebook)
            .transpose(0, 1)
            .contiguous()[:, None]
            .expand(Nq, self.N, self.M, self.ksub)
        )
        distances = (
            adtable.gather(
                dim=-1,
                index=self.data.long()[None, :, :, None].expand(Nq, self.N, self.M, 1),
            )
            .squeeze(-1)
            .sum(dim=-1)
        )
        return self.metric.topk(distances, k=k)
