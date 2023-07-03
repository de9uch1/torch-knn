import abc
from dataclasses import dataclass
from typing import Optional

from torch import Tensor


class Normalize(abc.ABC):
    def __init__(self, cfg: Optional["Normalize.Config"] = None) -> None:
        self.cfg = cfg

    @dataclass
    class Config:
        """Base class for normalize config."""

    @abc.abstractmethod
    def encode(self, x) -> Tensor:
        """Normalizes the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d)`.

        Returns:
            Tensor: Normalized vectors of shape `(n, d)`.
        """
