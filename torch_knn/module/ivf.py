from typing import List

import torch
from torch_knn.module.kmeans import Kmeans
from torch_knn.storage.base import Storage


class InvertedFile(Kmeans):
    """Inverted file class.

    Args:
        storage (Storage): Storage object.
        nlists (int): The number of centroids.

    Attributes:
        centroids (torch.Tensor): Centroids tensor of shape `(nlists, D)`.
        invlists (List[List[int]]): Inverted file that stores each cluster member.
    """

    def __init__(self, storage: Storage, nlists: int) -> None:
        cfg = storage.cfg
        super().__init__(nlists, cfg.D, cfg.metric)
        self.metric = cfg.metric
        self.nlists = nlists
        self.storage = storage
        self.invlists: List[torch.Tensor] = [
            torch.Tensor().long() for _ in range(nlists)
        ]

    def add(self, x: torch.Tensor) -> torch.Tensor:
        """Adds the given vectors to the inverted file.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            torch.Tensor: Assigned cluster IDs of shape `(N,)`.
        """
        assignments = self.assign(x)
        for i, assign in enumerate(assignments, start=self.storage.N):
            self.invlists[assign] = torch.cat(
                [self.invlists[assign], torch.LongTensor([i])]
            )
        return assignments
