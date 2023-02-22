from dataclasses import dataclass
from typing import Tuple

import torch

from torch_knn import utils
from torch_knn.module.ivf import InvertedFile
from torch_knn.storage.flat import FlatStorage


class IVFFlatIndex(FlatStorage):
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

        Args:
            D (int): Dimension size of input vectors.
            dtype (torch.dtype): The input vector dtype. (default: torch.float32)
            metric (Metric): Metric for dinstance computation.
            nlists (int): Number of clusters.
        """

        nlists: int = 1

    cfg: "IVFFlatIndex.Config"

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
        x = self.transform(x)
        self.ivf.train(x)
        return self

    def nprobe(self, n: int) -> None:
        self._nprobe = max(n, 1)

    def add(self, x: torch.Tensor) -> None:
        """Adds the given vectors to the storage.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.
        """
        x = self.transform(x)
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
        query = self.transform(query)
        nprobe = min(max(nprobe, 1), self.cfg.nlists)
        coarse_distances = self.metric.compute_distance(query, self.ivf.centroids)
        _, centroid_indices = self.metric.topk(coarse_distances, k=nprobe)
        keys = [
            torch.cat([self.ivf.invlists[i] for i in cents])
            for cents in centroid_indices.cpu()
        ]
        key_indices = utils.pad(keys, -1)
        # query: Nq x 1 x D
        # key: Nq x Nk x D
        # distances: Nq x 1 x Nk -> Nq x Nk
        distances = self.metric.compute_distance(
            query[:, None], self.data[key_indices]
        ).squeeze(1)
        distances = self.metric.mask(distances, key_indices.eq(-1))
        k_distances, k_probed_indices = self.metric.topk(distances, k=k)
        k_indices = key_indices.gather(-1, k_probed_indices)
        return k_distances, k_indices
