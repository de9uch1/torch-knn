from enum import Enum

import torch
import torch.nn as nn
from torch import Tensor

from torch_knn import utils
from torch_knn.metrics import L2Metric, Metric


class Kmeans(nn.Module):
    """k-means clustering class.

    Args:
        ncentroids (int): The number of centroids.
        dim (int): The dimension size of centroids.
        metric (Metric): Distance metric function.
        init (CentroidsInit): Initialization method of the centroids.
    """

    class Init(Enum):
        RANDOM = "random"

    def __init__(
        self,
        ncentroids: int,
        dim: int,
        metric: Metric = L2Metric(),
        init: Init = Init.RANDOM,
    ) -> None:
        super().__init__()
        self.ncentroids = ncentroids
        self.dim = dim
        self.metric = metric
        self.init = init
        self.register_buffer("_centroids", None)

    @property
    def centroids(self) -> Tensor:
        """Returns centroids tensor of shape `(ncentroids, dim)`."""
        if self._centroids is None:
            raise RuntimeError("Centroids are not trained.")
        return self._centroids

    @centroids.setter
    def centroids(self, centroids: Tensor) -> None:
        """Sets the given tensor as the centroids.

        Args:
            centroids (Tensor): Centroids tensor of shape `(ncentroids, dim)`.
        """
        if centroids.dim() != 2 or not utils.is_equal_shape(
            centroids, [self.ncentroids, self.dim]
        ):
            raise ValueError(
                "Centroids tensor must be the shape of `(ncentroids, dim)`."
            )
        self._centroids = centroids

    def init_centroids(self, init: Init) -> Tensor:
        """Initializes cluster centorids.

        Args:
            init (Init): Initialization method.

        Returns:
            Tensor: A centroids tensor.

        Raises:
            NotImplementedError: When the given method is not implemented.
        """
        if init == self.Init.RANDOM:
            return torch.rand(self.ncentroids, self.dim)
        else:
            raise NotImplementedError

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

    @property
    def is_trained(self) -> bool:
        """Returns whether the centroids are trained or not."""
        return self._centroids is not None

    def train(self, x: Tensor, niter: int = 10) -> Tensor:
        """Trains k-means.

        Args:
            x (torch.Tensor): Input vectors of shape `(n, dim)`.
            niter (int): Number of training iteration.

        Returns:
            Tensor: Centroids tensor of shape `(ncentroids, dim)`.
        """
        self.centroids = self.init_centroids(self.init)

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
        metric: Metric = L2Metric(),
        init: Kmeans.Init = Kmeans.Init.RANDOM,
    ) -> None:
        super().__init__(ncentroids, dim, metric=metric, init=init)
        self.nspaces = nspaces

    @property
    def centroids(self) -> Tensor:
        """Returns centroids tensor of shape `(nspaces, ncentroids, dim)`."""
        if self._centroids is None:
            raise RuntimeError("Centroids are not trained.")
        return self._centroids

    @centroids.setter
    def centroids(self, centroids: Tensor) -> None:
        """Sets the given tensor as the centroids.

        Args:
            centroids (Tensor): Centroids tensor of shape `(nspaces, ncentroids, dim)`.
        """
        if centroids.dim() != 3 or not utils.is_equal_shape(
            centroids, [self.nspaces, self.ncentroids, self.dim]
        ):
            raise ValueError(
                "Centroids tensor must be the shape of `(nspaces, ncentroids, dim)`."
            )
        self._centroids = centroids

    def init_centroids(self, init: Kmeans.Init) -> Tensor:
        """Initializes cluster centorids.

        Args:
            init (Init): Initialization method.

        Returns:
            Tensor: A centroids tensor.

        Raises:
            NotImplementedError: When the given method is not implemented.
        """
        if init == self.Init.RANDOM:
            return torch.rand(self.nspaces, self.ncentroids, self.dim)
        else:
            raise NotImplementedError

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
                x[update_mask] * mean_mask / (mean_mask.sum(dim=1, keepdim=True))
            ).sum(1)
        return new_centroids.to(dtype)

    def train(self, x: Tensor, niter: int = 10) -> Tensor:
        """Trains k-means.

        Args:
            x (torch.Tensor): Input vectors of shape `(n, nspaces, dim)`.
            niter (int): Number of training iteration.

        Returns:
            Tensor: Centroids tensor of shape `(nspaces, ncentroids, dim)`.
        """
        self.centroids = self.init_centroids(self.init)

        x = x.transpose(0, 1).contiguous()
        assigns = x.new_full((self.nspaces, x.size(1)), fill_value=-1)
        for i in range(niter):
            new_assigns = self.assign(x)
            if torch.equal(new_assigns, assigns):
                break
            assigns = new_assigns
            self.centroids = self.update(x, assigns)
        return self.centroids
