from dataclasses import dataclass

import torch
import torch.linalg as LA
from torch import Tensor

from .base import Transform


class PCATransform(Transform):
    def __init__(self, cfg: "PCATransform.Config") -> None:
        super().__init__(cfg)
        self.register_buffer("_weight", torch.Tensor(cfg.d_in, cfg.d_out))
        self.register_buffer("_mean", torch.Tensor(cfg.d_in))

    @dataclass
    class Config(Transform.Config):
        """PCA transform config.

        - d_in (int): Dimension size of input vectors.
        - d_out (int): Dimension size of output vectors.
        """

        def __post_init__(self):
            """Validates the dimension size."""
            if self.d_in < self.d_out:
                raise ValueError("d_in must be >= d_out.")

    cfg: Config

    @property
    def weight(self) -> Tensor:
        """Weight matrix of shape `(d_in, d_out)`."""
        return self._weight

    @weight.setter
    def weight(self, x: Tensor) -> None:
        """Sets the weight matrix of shape `(d_in, d_out)`."""
        self._weight = x

    @property
    def mean(self) -> Tensor:
        """Mean weights of shape `(d_in,)`."""
        return self._mean

    @mean.setter
    def mean(self, x: Tensor) -> None:
        """Sets the mean weights of shape `(d_in,)`."""
        self._mean = x

    def fit(self, x) -> "PCATransform":
        """Trains vector transformation for this class.

        Args:
            x (Tensor): Training vectors of shape `(n, d_in)`.

        Returns:
            Transform: This class.
        """
        # Centring
        mean = x.mean(dim=0)
        x = x - mean[None, :]
        # Compute eigenvectors
        cov = torch.cov(x.T)
        eigenvalues, eigenvectors = LA.eigh(cov)
        eigenvalues, eigenvectors = eigenvalues.to(x), eigenvectors.to(x)
        # Select top-k components
        indices = torch.argsort(eigenvalues, descending=True)
        full_weight = eigenvectors[:, indices]
        weight = full_weight[:, : self.cfg.d_out].contiguous()

        self.mean = mean
        self.weight = weight
        return self

    def encode(self, x) -> Tensor:
        """Transforms the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d_in)`.

        Returns:
            Tensor: Transformed vectors of shape `(n, d_out)`.
        """
        return (x - self.mean[None, :]) @ self.weight

    def decode(self, x) -> Tensor:
        """Inverse transforms the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d_out)`.

        Returns:
            Tensor: Inverse transformed vectors of shape `(n, d_in)`.
        """
        return (x @ self.weight.T) + self.mean[None, :]
