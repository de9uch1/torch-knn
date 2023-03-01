from __future__ import annotations

import pytest
import torch
from torch import Tensor

from torch_knn.transform.base import Transform


class TransformMock(Transform):
    @property
    def is_trained(self) -> bool:
        """Returns whether this class is trained or not."""
        return True

    def train(self, x) -> Transform:
        """Trains vector transformation for this class.

        Args:
            x (Tensor): Training vectors of shape `(n, d_in)`.

        Returns:
            Transform: Trained this class.
        """
        return self

    def encode(self, x) -> Tensor:
        """Transforms the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d_in)`.

        Returns:
            Tensor: Transformed vectors of shape `(n, d_out)`.
        """
        return x

    def decode(self, x) -> Tensor:
        """Inverse transforms the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d_out)`.

        Returns:
            Tensor: Inverse transformed vectors of shape `(n, d_in)`.
        """
        return x


class TestTransform:
    @pytest.mark.parametrize("d_in", [8, 16])
    @pytest.mark.parametrize("d_out", [8, 16])
    @pytest.mark.parametrize(
        "dtype",
        [torch.float32, torch.float16, torch.float, torch.half],
    )
    def test___init__(self, d_in: int, d_out: int, dtype: torch.dtype):
        cfg = Transform.Config(d_in, d_out, dtype)
        transform = TransformMock(cfg)
        assert transform.cfg.d_in == d_in
        assert transform.cfg.d_out == d_out
        assert transform.cfg.dtype == dtype
