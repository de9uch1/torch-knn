from dataclasses import dataclass
from typing import Optional

import torch
import torch.linalg as LA
from torch import Tensor

from torch_knn.storage.pq import PQStorage
from torch_knn.transform.base import Transform


class OPQTransform(Transform):
    def __init__(self, cfg: "OPQTransform.Config") -> None:
        super().__init__(cfg)
        self._weight: Optional[Tensor] = None

    @dataclass
    class Config(Transform.Config):
        """Base class for transform config.

        Args:
            d_in (int): Dimension size of input vectors.
            d_out (int): Dimension size of output vectors.
            M (int): The number of sub-vectors.
            ksub (int): Codebook size of a sub-space. (default: 256)
            code_dtype (torch.dtype): DType for stored codes. (default: torch.uint8)
            train_niter (int): Number of training iteration.
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
        """Weight matrix of shape `(d_out, d_in)`."""
        if self._weight is None:
            raise RuntimeError("Transform matrix has not been trained.")
        return self._weight

    @weight.setter
    def weight(self, x: Tensor) -> None:
        """Sets the weight matrix of shape `(d_out, d_in)`."""
        self._weight = x

    @property
    def is_trained(self) -> bool:
        """Returns whether this class is trained or not."""
        return self._weight is not None

    def train(self, x) -> "OPQTransform":
        """Trains vector transformation for this class.

        Args:
            x (Tensor): Training vectors of shape `(n, d_in)`.

        Returns:
            Transform: Trained this class.
        """
        cfg = self.cfg
        initial_weight = torch.rand(cfg.d_in, cfg.d_in)
        Q, _ = LA.qr(initial_weight)
        self.weight = Q[: cfg.d_out]

        for i in range(cfg.train_niter):
            x_proj = self.encode(x)
            pq = PQStorage(
                PQStorage.Config(
                    cfg.d_out,
                    M=cfg.M,
                    ksub=cfg.ksub,
                    code_dtype=cfg.code_dtype,
                    train_niter=cfg.train_pq_niter,
                ),
            )

            # TODO(deguchi): Set manual seed
            torch.manual_seed(0)
            pq.train(x_proj)
            recons = pq.decode(pq.encode(x_proj))
            U, s, Vt = LA.svd(recons.T @ x, full_matrices=False)
            self.weight = U @ Vt

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
