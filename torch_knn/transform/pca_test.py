import pytest
import torch

from torch_knn import utils
from torch_knn.transform.pca import PCATransform

D_IN = 8
D_OUT = 8
N = 10


class TestPCATransform:
    def test___init__(self):
        cfg = PCATransform.Config(D_IN, D_OUT)
        pca = PCATransform(cfg)
        assert pca.cfg.d_in == D_IN
        assert pca.cfg.d_out == D_OUT

    class TestConfig:
        with pytest.raises(ValueError):
            PCATransform.Config(8, 16)

    def test_weight(self):
        cfg = PCATransform.Config(D_IN, D_OUT)
        pca = PCATransform(cfg)
        x = torch.rand(D_OUT, D_IN)
        pca.weight = x
        assert torch.equal(pca.weight, x)

    def test_mean(self):
        cfg = PCATransform.Config(D_IN, D_OUT)
        pca = PCATransform(cfg)
        x = torch.rand(D_IN)
        pca.mean = x
        assert torch.equal(pca.mean, x)

    @pytest.mark.parametrize(["d_in", "d_out"], [(8, 8), (16, 16), (16, 8)])
    def test_fit(self, d_in: int, d_out: int):
        cfg = PCATransform.Config(d_in, d_out)
        pca = PCATransform(cfg)
        torch.manual_seed(0)
        x = torch.rand(N, d_in)
        pca.fit(x)

        assert utils.is_equal_shape(pca.weight, [d_in, d_out])

        # Checks the orthogonal matrices
        if d_in == d_out:
            torch.testing.assert_close(pca.weight @ pca.weight.T, torch.eye(d_in))

    def test_encode(self):
        cfg = PCATransform.Config(D_IN, D_OUT)
        pca = PCATransform(cfg)
        torch.manual_seed(0)
        x = torch.rand(N, D_IN)
        pca.fit(x)
        assert utils.is_equal_shape(pca.encode(x), [N, D_OUT])

    def test_decode(self):
        cfg = PCATransform.Config(D_IN, D_OUT)
        pca = PCATransform(cfg)
        torch.manual_seed(0)
        x = torch.rand(N, D_IN)
        pca.fit(x)
        assert utils.is_equal_shape(pca.decode(torch.rand(N, D_OUT)), [N, D_IN])
        torch.testing.assert_close(pca.decode(pca.encode(x)), x)
