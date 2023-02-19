import abc
from dataclasses import asdict, dataclass, fields
from typing import Tuple

import torch
from torch_knn.metrics import L2Metric, Metric
from torch_knn.storage.base import Storage


class Index(abc.ABC):
    """Base class for kNN index.

    Args:
        cfg (Index.Config): Configuration for this class.

    Attributes:
        storage_type (Type[Storage]): Storage class for this index.
    """

    storage_type = Storage

    def __init__(self, cfg: "Index.Config") -> None:
        self.cfg = cfg
        self.storage = self.new_storage(cfg)
        self.metric = cfg.metric

    @dataclass
    class Config(storage_type.Config):
        """Base class for index config.

        Args:
            D (int): Dimension size of input vectors.
            dtype (torch.dtype): The input vector dtype. (default: torch.float32)
            metric (Metric): Metric for dinstance computation.
        """

        metric: Metric = L2Metric()

    @classmethod
    def new_storage(cls, cfg: "Index.Config") -> Storage:
        index_cfg_dict = asdict(cfg)
        storage_kwargs = {}
        for field in fields(cls.storage_type.Config):
            name = field.name
            if name in index_cfg_dict:
                storage_kwargs[name] = index_cfg_dict[name]
        return cls.storage_type(cls.storage_type.Config(**storage_kwargs))

    @property
    def D(self) -> int:
        """Dimension size of the vectors."""
        return self.cfg.D

    @property
    @abc.abstractmethod
    def is_trained(self) -> bool:
        """Returns whether the index is trained or not."""

    @abc.abstractmethod
    def train(self, x: torch.Tensor) -> "Index":
        """Trains the index with the given vectors.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            Index: The trained index object.
        """

    def add(self, x: torch.Tensor) -> None:
        """Adds the given vectors to the index.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.
        """
        self.storage.add(x)

    @abc.abstractmethod
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
