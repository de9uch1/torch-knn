import abc
from dataclasses import dataclass

import torch.nn as nn
from torch import Tensor


class Transform(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, cfg: "Transform.Config") -> None:
        super().__init__()
        self.cfg = cfg
        self.d_in = cfg.d_in
        self.d_out = cfg.d_out

    @dataclass
    class Config:
        """Base class for transform config.

        - d_in (int): Dimension size of input vectors.
        - d_out (int): Dimension size of output vectors.
        """

        d_in: int
        d_out: int

    @property
    @abc.abstractmethod
    def is_trained(self) -> bool:
        """Returns whether this class is trained or not."""

    @abc.abstractmethod
    def train(self, x) -> "Transform":
        """Trains vector transformation for this class.

        Args:
            x (Tensor): Training vectors of shape `(n, d_in)`.

        Returns:
            Transform: Trained this class.
        """

    @abc.abstractmethod
    def encode(self, x) -> Tensor:
        """Transforms the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d_in)`.

        Returns:
            Tensor: Transformed vectors of shape `(n, d_out)`.
        """

    @abc.abstractmethod
    def decode(self, x) -> Tensor:
        """Inverse transforms the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d_out)`.

        Returns:
            Tensor: Inverse transformed vectors of shape `(n, d_in)`.
        """
