from typing import Any, Dict, List, Tuple

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
        for i in range(nlists):
            self.register_buffer(f"_cluster_{i}", Tensor().long())
        self._register_load_state_dict_pre_hook(self._load_state_dict_hook)

    def _load_state_dict_hook(
        self, state_dict: Dict[str, Any], prefix: str, *args, **kwargs
    ):
        for i in range(self.nlists):
            self.set_invlists(i, state_dict[f"{prefix}_cluster_{i}"])

    def get_invlists(self, idx: int) -> Tensor:
        return getattr(self, f"_cluster_{idx}")

    def set_invlists(self, idx: int, value: Tensor) -> None:
        setattr(self, f"_cluster_{idx}", value)

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
            self.set_invlists(
                cluster_idx, torch.cat([self.get_invlists(cluster_idx), data_idx])
            )
        self.N += len(assignments)
        return assignments

    def get_extra_state(self) -> int:
        return self.N

    def set_extra_state(self, extra_state: int) -> None:
        self.N = extra_state
