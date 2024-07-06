from enum import Enum
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from torch_knn import utils
from torch_knn.metrics import Metric, MetricL2


class Kmeans(nn.Module):
    """k-means clustering class.

    Args:
        ncentroids (int): The number of centroids.
        dim (int): The dimension size of centroids.
        metric (Metric): Distance metric function.
        init (Kmeans.Init): Initialization method of the centroids.
    """

    class Init(Enum):
        RANDOM_PICK = "random_pick"
        KMEANSPP = "kmeans++"

    def __init__(
        self,
        ncentroids: int,
        dim: int,
        metric: Metric = MetricL2(),
        init: Init = Init.RANDOM_PICK,
    ) -> None:
        super().__init__()
        self.ncentroids = ncentroids
        self.dim = dim
        self.metric = metric
        self.init = init
        self.register_buffer("_centroids", torch.zeros(self.centroids_shape))

    @property
    def centroids_shape(self) -> tuple[int, ...]:
        """Returns the shape of centroids.

        Returns:
            tuple[int, ...]: Shape of centroids.
        """
        return (self.ncentroids, self.dim)

    @property
    def centroids(self) -> Tensor:
        """Returns centroids tensor of shape `(ncentroids, dim)`."""
        return self._centroids

    @centroids.setter
    def centroids(self, centroids: Tensor) -> None:
        """Sets the given tensor as the centroids.

        Args:
            centroids (Tensor): Centroids tensor of shape `(ncentroids, dim)`.
        """
        if not utils.is_equal_shape(centroids, self.centroids_shape):
            raise ValueError(
                "Centroids tensor must be the shape of `(ncentroids, dim)`."
            )
        self._centroids = centroids

    def assign(self, x: Tensor) -> Tensor:
        """Assigns the nearest neighbor centroid ID.

        Args:
            x (torch.Tensor): Assigned vectors of shape `(n, dim)`.

        Returns:
            torch.Tensor: Assigned IDs of shape `(n,)`.
        """
        return self.metric.assign(x, self.centroids)

    def update(self, x: Tensor, assigns: Tensor) -> Tensor:
        """Updates the centroids.

        Args:
            x (torch.Tensor): Sample vectors of shape `(n, dim)`.
            assigns (torch.Tensor): Assigned centroids of the given input vectors of shape `(n,)`.

        Returns:
            torch.Tensor: New centroid vectors of shape `(ncentroids, dim)`.
        """
        new_centroids = self.centroids
        for k in range(self.ncentroids):
            if (assigns == k).any():
                new_centroids[k] = x[assigns == k].mean(dim=0)
        return new_centroids

    def init_kmeanspp(self, x: Tensor) -> Tensor:
        """Initializes the centroids via k-means++.

        Args:
            x (Tensor): Input vectors of shape `(n, dim)`.

        Returns:
            Tensor: Centroid vectors obtained using k-means++.
        """
        chosen_idxs: set[int] = set()
        initial_idx = torch.randint(x.size(0), size=(1,)).long().item()
        chosen_idxs.add(initial_idx)
        centroids = x[None, initial_idx]
        for _ in range(self.ncentroids - 1):
            # Nc x N
            sqdists = torch.cdist(centroids, x, p=2) ** 2
            neighbor_sqdists = sqdists.min(dim=0).values.float().clamp(min=1e-5)
            weights = neighbor_sqdists / neighbor_sqdists.sum()
            new_centroid_idx = torch.multinomial(weights, 1).item()
            while new_centroid_idx in chosen_idxs:
                new_centroid_idx = torch.multinomial(weights, 1).item()
            chosen_idxs.add(new_centroid_idx)
            new_centroid = x[None, new_centroid_idx]
            centroids = torch.cat([centroids, new_centroid])
        return centroids

    def fit(
        self, x: Tensor, niter: int = 50, initial_centroids: Optional[Tensor] = None
    ) -> Tensor:
        """Trains k-means.

        Args:
            x (torch.Tensor): Input vectors of shape `(n, dim)`.
            niter (int): Number of training iteration.

        Returns:
            Tensor: Centroids tensor of shape `(ncentroids, dim)`.
        """
        if initial_centroids is not None:
            self.centroids = initial_centroids.to(x)
        elif self.init == self.Init.RANDOM_PICK:
            self.centroids = x[torch.randperm(x.size(0))[: self.ncentroids]]
        elif self.init == self.Init.KMEANSPP:
            self.centroids = self.init_kmeanspp(x)
        else:
            self.centroids = self.centroids.to(x)
        assigns = x.new_full((x.size(0),), fill_value=-1)
        for i in range(niter):
            new_assigns = self.assign(x)
            if torch.equal(new_assigns, assigns):
                break
            assigns = new_assigns
            self.centroids = self.update(x, assigns)
        return self.centroids


class ParallelKmeans(Kmeans):
    """Parallel k-means clustering class.

    This class trains multiple k-means in parallel.
    ParallelKmeans is useful for training a PQ codebook.

    Args:
        ncentroids (int): The number of centroids.
        dim (int): The dimension size of centroids.
        nspaces (int): The number of subspaces.
        metric (Type[Metric]): Distance metric function.
        init (Init): Initialization method of the centroids.
    """

    def __init__(
        self,
        ncentroids: int,
        dim: int,
        nspaces: int,
        metric: Metric = MetricL2(),
        init: Kmeans.Init = Kmeans.Init.RANDOM_PICK,
    ) -> None:
        self.nspaces = nspaces
        super().__init__(ncentroids, dim, metric=metric, init=init)

    @property
    def centroids_shape(self) -> tuple[int, ...]:
        """Returns the shape of centroids.

        Returns:
            tuple[int, ...]: Shape of centroids.
        """
        return (self.nspaces, self.ncentroids, self.dim)

    @property
    def centroids(self) -> Tensor:
        """Returns centroids tensor of shape `(nspaces, ncentroids, dim)`."""
        return self._centroids

    @centroids.setter
    def centroids(self, centroids: Tensor) -> None:
        """Sets the given tensor as the centroids.

        Args:
            centroids (Tensor): Centroids tensor of shape `(nspaces, ncentroids, dim)`.
        """
        if not utils.is_equal_shape(centroids, self.centroids_shape):
            raise ValueError(
                "Centroids tensor must be the shape of `(nspaces, ncentroids, dim)`."
            )
        self._centroids = centroids

    def assign(self, x: Tensor) -> Tensor:
        """Assigns the nearest neighbor centroid ID.

        Args:
            x (torch.Tensor): Assigned vectors of shape `(nspaces, n, dim)`.

        Returns:
            torch.Tensor: Assigned IDs of shape `(nspaces, n)`.
        """
        return self.metric.assign(x, self.centroids)

    def update(self, x: Tensor, assigns: Tensor) -> Tensor:
        """Updates the centroids.

        Args:
            x (torch.Tensor): Sample vectors of shape `(nspaces, n, dim)`.
            assigns (torch.Tensor): Assigned centroids of the given input vectors of shape `(nspaces, n)`.

        Returns:
            torch.Tensor: New centroid vectors of shape `(nspaces, ncentroids, dim)`.
        """
        new_centroids = self.centroids
        dtype = x.dtype
        x = x.float()
        for k in range(self.ncentroids):
            # nspaces x n
            is_assigned = assigns.eq(k)
            update_mask = is_assigned.any(dim=-1)
            mean_mask = is_assigned[update_mask].unsqueeze(-1).float()
            new_centroids[update_mask, k] = (
                (x[update_mask] * mean_mask / (mean_mask.sum(dim=1, keepdim=True)))
                .sum(1)
                .to(dtype)
            )
        return new_centroids

    def init_kmeanspp(self, x: Tensor) -> Tensor:
        """Initializes the centroids via k-means++.

        Args:
            x (Tensor): Input vectors of shape `(nspaces, n, dim)`.

        Returns:
            Tensor: Centroid vectors obtained using k-means++.
        """
        centroids = []
        for m in range(self.nspaces):
            centroids.append(super().init_kmeanspp(x[m]))
        return torch.stack(centroids)

    def fit(
        self, x: Tensor, niter: int = 50, initial_centroids: Optional[Tensor] = None
    ) -> Tensor:
        """Trains k-means.

        Args:
            x (torch.Tensor): Input vectors of shape `(nspaces, n, dim)`.
            niter (int): Number of training iteration.

        Returns:
            Tensor: Centroids tensor of shape `(nspaces, ncentroids, dim)`.
        """
        n = x.size(1)
        if initial_centroids is not None:
            self.centroids = initial_centroids.to(x)
        elif self.init == self.Init.RANDOM_PICK:
            self.centroids = x[:, torch.randperm(n)[: self.ncentroids]]
        elif self.init == self.Init.KMEANSPP:
            self.centroids = self.init_kmeanspp(x)
        else:
            self.centroids = self.centroids.to(x)

        assigns = x.new_full((self.nspaces, n), fill_value=-1)
        for i in range(niter):
            new_assigns = self.assign(x)
            if torch.equal(new_assigns, assigns):
                break
            assigns = new_assigns
            self.centroids = self.update(x, assigns)
        return self.centroids
