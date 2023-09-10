import abc
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn

from torch_knn.metrics import L2Metric, Metric


class Storage(nn.Module, metaclass=abc.ABCMeta):
    """Base class for storage.

    Args:
        cfg (Storage.Config): Configuration for this class.
    """

    def __init__(self, cfg: "Storage.Config"):
        super().__init__()
        self.cfg = cfg
        self.metric = cfg.metric
        self.register_buffer("_data", torch.Tensor())

    @dataclass
    class Config:
        """Base class for storage config.

        - D (int): Dimension size of input vectors.
        - metric (Metric): Metric for dinstance computation.
        """

        D: int
        metric: Metric = L2Metric()

    @property
    def N(self) -> int:
        """The number of vectors that are added to the storage."""
        return self.data.size(0)

    @property
    def D(self) -> int:
        """Dimension size of the vectors."""
        return self.cfg.D

    @property
    def data(self) -> torch.Tensor:
        """Data object."""
        return self._data

    @property
    def shape(self) -> torch.Size:
        """Returns the shape of the storage tensor."""
        return self.data.shape

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
            Storage: The storage object.
        """

    def add(self, x: torch.Tensor) -> None:
        """Adds the given vectors to the storage.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.
        """
        self._data = torch.cat([self.data, self.encode(x)])

    def get_extra_state(self):
        """Gets extra state."""
        return {"cfg": asdict(self.cfg)}

    def set_extra_state(self, extra_state):
        """Gets extra state."""
        self.cfg = self.Config(**extra_state["cfg"])

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True):
        self._data = state_dict["_data"]
        return super().load_state_dict(state_dict, strict)

    @classmethod
    def load(cls, state_dict: Mapping[str, Any], strict: bool = True):
        self = cls(cls.Config(**state_dict["_extra_state"]["cfg"]))
        self.load_state_dict(state_dict, strict)
        return self
