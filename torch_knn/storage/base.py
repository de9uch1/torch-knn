import abc
from dataclasses import dataclass
from typing import Set

import torch


class Storage(abc.ABC):
    """Base class for storage.

    Args:
        cfg (Storage.Config): Configuration for this class.
    """

    support_dtypes: Set[torch.dtype] = {torch.float32, torch.float16}

    def __init__(self, cfg: "Storage.Config"):
        self.cfg = cfg
        self.dtype = cfg.dtype
        self._storage = torch.Tensor()

    @dataclass
    class Config:
        """Base class for storage config.

        Args:
            D (int): Dimension size of input vectors.
            dtype (torch.dtype): The input vector dtype. (default: torch.float32)
        """

        D: int
        dtype: torch.dtype = torch.float32

        def __post_init__(self):
            Storage.check_supported_dtype(self.dtype)

    @property
    def N(self) -> int:
        """The number of vectors that are added to the storage."""
        return self.storage.size(0)

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
    def storage(self) -> torch.Tensor:
        """Storage object."""
        return self._storage

    @property
    def shape(self) -> torch.Size:
        """Returns the shape of the storage tensor."""
        return self.storage.shape

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
        self._storage = torch.cat([self.storage, self.encode(x.to(self.dtype))])
