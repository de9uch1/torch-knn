import torch
from torch import Tensor
from torch_knn.constants import CentroidsInit
from torch_knn.metrics import L2Metric, Metric


class Kmeans:
    """k-means clustering class.

    Args:
        ncentroids (int): The number of centroids.
        dim (int): The dimension size of centroids.
        metric (Metric): Distance metric function.
        init (CentroidsInit): Initialization method of the centroids.
    """

    def __init__(
        self,
        ncentroids: int,
        dim: int,
        metric: Metric = L2Metric(),
        init: CentroidsInit = CentroidsInit.RANDOM,
    ) -> None:
        self.ncentroids = ncentroids
        self.dim = dim
        self.metric = metric
        self.init = init

        self.centroids = self.init_centroids(init)

    def init_centroids(self, init: CentroidsInit) -> Tensor:
        """Initializes cluster centorids.

        Args:
            init (CentroidsInit): Initialization method.

        Returns:
            Tensor: A centroids tensor.

        Raises:
            NotImplementedError: When the given method is not implemented.
        """
        if init == CentroidsInit.RANDOM:
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
            new_centroids[k] = x[assigns == k].mean(dim=0)
        return new_centroids

    def train(self, x: Tensor, niter: int = 10) -> Tensor:
        """Trains k-means.

        Args:
            x (torch.Tensor): Input vectors of shape `(n, dim)`.
            niter (int): Number of training iteration.

        Returns:
            Tensor: Centroids tensor of shape `(ncentroids, dim)`.
        """
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
        init (CentroidsInit): Initialization method of the centroids.
    """

    def __init__(
        self,
        ncentroids: int,
        dim: int,
        nspaces: int,
        metric: Metric = L2Metric(),
        init: CentroidsInit = CentroidsInit.RANDOM,
    ) -> None:
        self.nspaces = nspaces
        super().__init__(ncentroids, dim, metric=metric, init=init)

    def init_centroids(self, init: CentroidsInit) -> Tensor:
        """Initializes cluster centorids.

        Args:
            init (CentroidsInit): Initialization method.

        Returns:
            Tensor: A centroids tensor.

        Raises:
            NotImplementedError: When the given method is not implemented.
        """
        if init == CentroidsInit.RANDOM:
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
            k_mask = assigns.eq(k).unsqueeze(-1).float()
            new_centroids[:, k] = (
                x * k_mask / (k_mask.sum(dim=1, keepdim=True) + 1e-9)
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
        x = x.transpose(0, 1).contiguous()
        assigns = x.new_full((self.nspaces, x.size(1)), fill_value=-1)
        for i in range(niter):
            new_assigns = self.assign(x)
            if torch.equal(new_assigns, assigns):
                break
            assigns = new_assigns
            self.centroids = self.update(x, assigns)
        return self.centroids
