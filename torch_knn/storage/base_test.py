from contextlib import nullcontext as does_not_raise

import pytest
import torch
from torch_knn.storage.base import Storage


class StorageMock(Storage):
    @classmethod
    def check_shape(cls, storage: torch.Tensor) -> torch.Tensor:
        """Checks whether the storage tensor shape is valid or not.

        Args:
            storage (torch.Tensor): The storage tensor.

        Returns:
            torch.Tensor: The input storage tensor.

        Raises:
            ValueError: When given the wrong shape storage.
        """
        return storage


class TestStorage:
    @pytest.mark.parametrize(
        "dtype,expectation",
        [
            (torch.float32, does_not_raise()),
            (torch.float16, does_not_raise()),
            (torch.float, does_not_raise()),
            (torch.half, does_not_raise()),
            (torch.long, pytest.raises(ValueError)),
            (torch.uint8, pytest.raises(ValueError)),
        ],
    )
    def test___init__(self, dtype, expectation):
        x = torch.rand(3, 8)
        with expectation:
            storage = StorageMock(x, dtype)
        if expectation is does_not_raise():
            assert torch.equal(storage.storage, x.to(dtype))

    @pytest.mark.parametrize(
        "dtype,expectation",
        [
            (torch.float32, does_not_raise()),
            (torch.float16, does_not_raise()),
            (torch.float, does_not_raise()),
            (torch.half, does_not_raise()),
            (torch.long, pytest.raises(ValueError)),
            (torch.uint8, pytest.raises(ValueError)),
        ],
    )
    def test_check_dtype(self, dtype, expectation):
        with expectation:
            Storage.check_dtype(dtype)
