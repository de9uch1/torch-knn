from dataclasses import dataclass

import torch
import torch.linalg as LA
from torch import Tensor

from torch_knn.storage.pq import StoragePQ

from .base import Transform


class OPQTransform(Transform):
    def __init__(self, cfg: "OPQTransform.Config") -> None:
        super().__init__(cfg)
        self.register_buffer("_weight", torch.eye(cfg.d_in))
        self.pq = StoragePQ(
            StoragePQ.Config(
                cfg.d_out,
                M=cfg.M,
                ksub=cfg.ksub,
                code_dtype=cfg.code_dtype,
                train_niter=cfg.train_pq_niter,
            )
        )

    @dataclass
    class Config(Transform.Config):
        """OPQ transform config.

        - d_in (int): Dimension size of input vectors.
        - d_out (int): Dimension size of output vectors.
        - M (int): The number of sub-vectors.
        - ksub (int): Codebook size of a sub-space. (default: 256)
        - code_dtype (torch.dtype): DType for stored codes. (default: torch.uint8)
        - train_niter (int): Number of training iteration.
        """

        M: int = 1
        ksub: int = 256
        code_dtype: torch.dtype = torch.uint8
        train_niter: int = 10
        train_pq_niter: int = 5

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

    def fit(self, x) -> "OPQTransform":
        """Trains vector transformation for this class.

        Args:
            x (Tensor): Training vectors of shape `(n, d_in)`.

        Returns:
            Transform: This class.
        """
        cfg = self.cfg
        Q, _ = LA.qr(self.weight.float())
        self.weight = Q[: cfg.d_out].to(self.weight)

        for i in range(cfg.train_niter):
            x_proj = self.encode(x)
            # TODO(deguchi): Set manual seed
            torch.manual_seed(0)
            if i == 0:
                self.pq.fit(x_proj)
            else:
                self.pq.fit(x_proj, self.pq.codebook)
            recons = self.pq.decode(self.pq.encode(x_proj))
            U, s, Vt = LA.svd((recons.T @ x).float(), full_matrices=False)
            self.weight = (U @ Vt).to(self.weight)

        return self

    def encode(self, x) -> Tensor:
        """Transforms the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d_in)`.

        Returns:
            Tensor: Transformed vectors of shape `(n, d_out)`.
        """
        return x @ self.weight.T

    def decode(self, x) -> Tensor:
        """Inverse transforms the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d_out)`.

        Returns:
            Tensor: Inverse transformed vectors of shape `(n, d_in)`.
        """
        return x @ self.weight
