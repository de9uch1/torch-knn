from typing import List, Tuple

import torch
from torch import Tensor

from torch_knn.module.kmeans import Kmeans
from torch_knn.storage.base import Storage


class InvertedFile(Kmeans):
    """Inverted file class.

    Args:
        storage (Storage): Storage object.
        nlists (int): The number of centroids.

    Attributes:
        centroids (torch.Tensor): Centroids tensor of shape `(nlists, D)`.
        invlists (List[Tensor]): Inverted file that stores each cluster member.
    """

    def __init__(self, storage: Storage, nlists: int) -> None:
        cfg = storage.cfg
        super().__init__(nlists, cfg.D, cfg.metric)
        self.metric = cfg.metric
        self.nlists = nlists
        self.N = storage.N
        self.invlists: List[Tensor] = [Tensor().long() for _ in range(nlists)]

    def add(self, x: Tensor) -> Tensor:
        """Adds the given vectors to the inverted file.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            torch.Tensor: Assigned cluster IDs of shape `(N,)`.
        """
        assignments = self.assign(x)
        for cluster_idx in range(self.nlists):
            data_idx = assignments.eq(cluster_idx).nonzero()[:, 0] + self.N
            self.invlists[cluster_idx] = torch.cat(
                [self.invlists[cluster_idx], data_idx.cpu()]
            )
        self.N += len(assignments)
        return assignments

    def get_extra_state(self) -> Tuple[List[Tensor], int]:
        return (self.invlists, self.N)

    def set_extra_state(self, extra_state: Tuple[List[Tensor], int]) -> None:
        self.invlists = extra_state[0]
        self.N = extra_state[1]
