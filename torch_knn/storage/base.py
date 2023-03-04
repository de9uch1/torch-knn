import abc
from dataclasses import dataclass
from typing import List, Set

import torch

from torch_knn.metrics import CosineMetric, L2Metric, Metric
from torch_knn.transform.base import Transform
from torch_knn.transform.l2_normalization import L2NormalizationTransform


class Storage(abc.ABC):
    """Base class for storage.

    Args:
        cfg (Storage.Config): Configuration for this class.
    """

    support_dtypes: Set[torch.dtype] = {torch.float32, torch.float16}

    def __init__(self, cfg: "Storage.Config"):
        self.cfg = cfg
        self.dtype = cfg.dtype
        self.metric = cfg.metric
        self._data = torch.Tensor()
        self.pre_transforms: List[Transform] = []
        if isinstance(self.metric, CosineMetric):
            self.pre_transforms.append(
                L2NormalizationTransform(L2NormalizationTransform.Config(cfg.D, cfg.D)),
            )

    @dataclass
    class Config:
        """Base class for storage config.

        Args:
            D (int): Dimension size of input vectors.
            dtype (torch.dtype): The input vector dtype. (default: torch.float32)
            metric (Metric): Metric for dinstance computation.
        """

        D: int
        dtype: torch.dtype = torch.float32
        metric: Metric = L2Metric()

        def __post_init__(self):
            Storage.check_supported_dtype(self.dtype)

    @property
    def N(self) -> int:
        """The number of vectors that are added to the storage."""
        return self.data.size(0)

    @property
    def D(self) -> int:
        """Dimension size of the vectors."""
        return self.cfg.D

    @classmethod
    def check_supported_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        """Checks whether the specified dtype is supported or not.

        Args:
            dtype (torch.dtype): The specified dtype.

        Returns:
            torch.dtype: The specified dtype.

        Raises:
            ValueError: When given the unsupported dtype.
        """
        if dtype not in cls.support_dtypes:
            raise ValueError(f"The dtype `{dtype}` is not supported for this storage.")
        return dtype

    @property
    def data(self) -> torch.Tensor:
        """Data object."""
        return self._data

    @property
    def shape(self) -> torch.Size:
        """Returns the shape of the storage tensor."""
        return self.data.shape

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        """Pre-transforms the given vectors.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, Din)`.

        Returns:
            torch.Tensor: Transformed vectors of shape `(N, D)`.
        """
        for t in self.pre_transforms:
            x = t.encode(x)
        return x

    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the given vectors.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            torch.Tensor: Encoded vectors.
        """

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """Decodes the given vectors or codes.

        Args:
            x (torch.Tensor): The input vectors or codes.

        Returns:
            torch.Tensor: Decoded vectors.
        """

    @abc.abstractmethod
    def train(self, x: torch.Tensor) -> "Storage":
        """Trains the index with the given vectors.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            Storage: The trained storage object.
        """

    @property
    @abc.abstractmethod
    def is_trained(self) -> bool:
        """Returns whether the storage is trained or not."""

    def add(self, x: torch.Tensor) -> None:
        """Adds the given vectors to the storage.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.
        """
        self._data = torch.cat([self.data, self.encode(x.to(self.dtype))])
