import pytest
import torch

from torch_knn.transform.l2_normalization import L2NormalizationTransform

D_IN = 8
D_OUT = 8
N = 10


class TestL2NormalizationTransform:
    class TestConfig:
        with pytest.raises(ValueError):
            L2NormalizationTransform.Config(8, 16)

        with pytest.raises(ValueError):
            L2NormalizationTransform.Config(16, 8)

    def test_is_trained(self):
        cfg = L2NormalizationTransform.Config(D_IN, D_OUT)
        l2norm = L2NormalizationTransform(cfg)
        assert l2norm.is_trained

    def test_train(self):
        cfg = L2NormalizationTransform.Config(D_IN, D_OUT)
        l2norm = L2NormalizationTransform(cfg)
        x = torch.rand(N, D_IN)
        assert l2norm == l2norm.train(x)

    def test_encode(self):
        cfg = L2NormalizationTransform.Config(D_IN, D_OUT)
        l2norm = L2NormalizationTransform(cfg)
        x = torch.rand(N, D_IN)
        torch.testing.assert_close(
            l2norm.encode(x),
            x / (x**2).sum(-1, keepdim=True) ** 0.5,
        )

    def test_decode(self):
        cfg = L2NormalizationTransform.Config(D_IN, D_OUT)
        l2norm = L2NormalizationTransform(cfg)
        x = torch.rand(N, D_IN)
        torch.testing.assert_close(l2norm.encode(x), l2norm.decode(l2norm.encode(x)))
