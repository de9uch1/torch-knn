from contextlib import nullcontext as does_not_raise

import pytest
import torch

from torch_knn import metrics, utils
from torch_knn.storage.base import Storage

N = 3
D = 8


class StorageMock(Storage):
    def encode(self, x):
        return x

    def decode(self, x):
        return x

    def train(self, x):
        return self

    @property
    def is_trained(self):
        return True


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
        x = torch.rand(N, D)
        with expectation:
            cfg = StorageMock.Config(x.size(-1), dtype=dtype)
            StorageMock(cfg)

    def test_N(self):
        x = torch.rand(N, D)
        cfg = StorageMock.Config(x.size(-1))
        storage = StorageMock(cfg)
        storage.add(x)
        assert storage.N == x.size(0)

    def test_D(self):
        cfg = StorageMock.Config(D)
        storage = StorageMock(cfg)
        assert storage.D == D

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
    def test_check_supported_dtype(self, dtype, expectation):
        with expectation:
            Storage.check_supported_dtype(dtype)

    def test_shape(self):
        cfg = StorageMock.Config(int())
        storage = StorageMock(cfg)
        assert utils.is_equal_shape(storage.shape, [0])

        x = torch.rand(N, D)
        cfg = StorageMock.Config(D)
        storage = StorageMock(cfg)
        storage.add(x)
        assert utils.is_equal_shape(storage.shape, x)

    @pytest.mark.parametrize("metric", [metrics.L2Metric(), metrics.CosineMetric()])
    def test_transform(self, metric: metrics.Metric):
        x = torch.rand(N, D)
        cfg = StorageMock.Config(D, metric=metric)
        storage = StorageMock(cfg)
        if isinstance(metric, metrics.CosineMetric):
            torch.testing.assert_close(
                storage.transform(x),
                x / (x**2).sum(-1, keepdim=True) ** 0.5,
            )
        else:
            torch.testing.assert_close(storage.transform(x), x)

    def test_add(self):
        x = torch.rand(N, D)
        cfg = StorageMock.Config(D)
        storage = StorageMock(cfg)
        storage.add(x)
        torch.testing.assert_close(storage.data, x)
