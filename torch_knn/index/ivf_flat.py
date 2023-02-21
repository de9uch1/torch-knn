from dataclasses import dataclass
from typing import Tuple

import torch
from torch_knn import utils
from torch_knn.module.ivf import InvertedFile
from torch_knn.storage.flat import FlatStorage


class IVFFlatIndex(FlatStorage):
    """Inverted file index class.

    Args:
        cfg (IVFIndex.Config): Configuration for this class.
    """

    def __init__(self, cfg: "IVFFlatIndex.Config"):
        super().__init__(cfg)
        self.ivf = InvertedFile(self, cfg.nlists)
        self._nprobe = 1

    @dataclass
    class Config(FlatStorage.Config):
        """IVFIndex configuration.

        Args:
            D (int): Dimension size of input vectors.
            dtype (torch.dtype): The input vector dtype. (default: torch.float32)
            metric (Metric): Metric for dinstance computation.
            nlists (int): Number of clusters.
        """

        nlists: int = 1

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

    @property
    def nprobe(self) -> int:
        return self._nprobe

    @nprobe.setter
    def nprobe(self, n: int) -> None:
        self._nprobe = max(n, 1)

    def add(self, x: torch.Tensor) -> None:
        """Adds the given vectors to the storage.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.
        """
        self.ivf.add(x)
        self._data = torch.cat([self.data, self.encode(x.to(self.dtype))])

    def search(
        self, query: torch.Tensor, k: int = 1
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
        coarse_distances = self.metric.compute_distance(query, self.ivf.centroids)
        _, centroid_indices = self.metric.topk(coarse_distances, k=self.nprobe)
        keys = [
            torch.cat([self.ivf.invlists[i] for i in cents])
            for cents in centroid_indices.cpu()
        ]
        key_indices = utils.pad(keys, -1)
        distances = self.metric.compute_distance(
            query[:, None], self.data[key_indices]
        ).squeeze(1)
        distances = self.metric.mask(distances, key_indices.eq(-1))
        k_distances, k_probed_indices = self.metric.topk(distances, k=k)
        k_indices = key_indices.gather(-1, k_probed_indices)
        return k_distances, k_indices
