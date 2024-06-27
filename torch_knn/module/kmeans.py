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
        init (CentroidsInit): Initialization method of the centroids.
    """

    class Init(Enum):
        RANDOM_PICK = "random_pick"

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

    def fit(
        self, x: Tensor, niter: int = 50, initial_centroids: Optional[Tensor] = None
    ) -> Tensor:
        """Trains k-means.

        Args:
            x (torch.Tensor): Input vectors of shape `(n, nspaces, dim)`.
            niter (int): Number of training iteration.

        Returns:
            Tensor: Centroids tensor of shape `(nspaces, ncentroids, dim)`.
        """
        if initial_centroids is not None:
            self.centroids = initial_centroids.to(x)
        elif self.init == self.Init.RANDOM_PICK:
            self.centroids = (
                x[torch.randperm(x.size(0))[: self.ncentroids]]
                .transpose(0, 1)
                .contiguous()
            )
        else:
            self.centroids = self.centroids.to(x)
        x = x.transpose(0, 1).contiguous()
        assigns = x.new_full((self.nspaces, x.size(1)), fill_value=-1)
        for i in range(niter):
            new_assigns = self.assign(x)
            if torch.equal(new_assigns, assigns):
                break
            assigns = new_assigns
            self.centroids = self.update(x, assigns)
        return self.centroids
