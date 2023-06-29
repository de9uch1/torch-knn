from dataclasses import dataclass
from typing import Tuple

import torch

from torch_knn import utils
from torch_knn.index.base import Index
from torch_knn.module.ivf import InvertedFile
from torch_knn.storage.flat import FlatStorage


class IVFFlatIndex(FlatStorage, Index):
    """Inverted file index class.

    Args:
        cfg (IVFFlatIndex.Config): Configuration for this class.
    """

    def __init__(self, cfg: "IVFFlatIndex.Config"):
        super().__init__(cfg)
        self.ivf = InvertedFile(self, cfg.nlists)

    @dataclass
    class Config(FlatStorage.Config):
        """IVFFlatIndex configuration.

        - D (int): Dimension size of input vectors.
        - metric (Metric): Metric for dinstance computation.
        - nlists (int): Number of clusters.
        """

        nlists: int = 1

    cfg: "IVFFlatIndex.Config"

    @property
    def centroids(self) -> torch.Tensor:
        """Returns centroid tensor of shape `(nlists, D)`"""
        return self.ivf.centroids

    @property
    def is_trained(self) -> bool:
        """Returns whether the index is trained or not."""
        return super().is_trained and self.ivf.is_trained

    def train(self, x: torch.Tensor) -> "IVFFlatIndex":
        """Trains the index with the given vectors.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            IVFFlatIndex: The trained index object.
        """
        self.ivf.train(x)
        return self

    def add(self, x: torch.Tensor) -> None:
        """Adds the given vectors to the storage.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.
        """
        self.ivf.add(x)
        super().add(x)

    def search(
        self, query: torch.Tensor, k: int = 1, nprobe: int = 1, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Searches the k-nearest-neighbor vectors.

        Args:
            query (torch.Tensor): Query vectors of shape `(Nq, D)`.
            k (int): Number of nearest neighbors to be returned.
            nprobe (int): Number of probing clusters.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
              - torch.Tensor: Distances between querys and keys of shape `(Nq, k)`.
              - torch.Tensor: Indices of the k-nearest-neighbors of shape `(Nq, k)`.
        """
        nprobe = min(max(nprobe, 1), self.cfg.nlists)
        coarse_distances = self.metric.compute_distance(query, self.centroids)
        _, centroid_indices = self.metric.topk(coarse_distances, k=nprobe)
        keys = [
            torch.cat([self.ivf.invlists[i] for i in cents])
            for cents in centroid_indices.cpu()
        ]
        key_indices = utils.pad(keys, -1)
        Nq, Nk = key_indices.size()
        # query: Nq x 1 x D
        # key: Nq x Nk x D
        # distances: Nq x 1 x Nk -> Nq x Nk
        distances = self.metric.compute_distance(
            query[:, None], self.data[key_indices]
        ).squeeze(1)
        distances = self.metric.mask(distances, key_indices.eq(-1))
        k_cand_distances, k_cand_probed_indices = self.metric.topk(
            distances, k=min(k, Nk)
        )
        k_cand_indices = key_indices.gather(-1, k_cand_probed_indices)

        k_distances = k_cand_distances.new_full(
            (Nq, k), fill_value=self.metric.farthest_value
        )
        k_indices = k_cand_indices.new_full((Nq, k), fill_value=-1)
        k_distances[:, : min(k, Nk)] = k_cand_distances
        k_indices[:, : min(k, Nk)] = k_cand_indices
        return k_distances, k_indices
