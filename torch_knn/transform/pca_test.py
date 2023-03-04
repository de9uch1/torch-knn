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
        assert pca._weight is None
        assert pca._mean is None

    class TestConfig:
        with pytest.raises(ValueError):
            PCATransform.Config(8, 16)

    def test_weight(self):
        cfg = PCATransform.Config(D_IN, D_OUT)
        pca = PCATransform(cfg)
        with pytest.raises(RuntimeError):
            pca.weight

        x = torch.rand(D_OUT, D_IN)
        pca.weight = x
        assert torch.equal(pca.weight, x)

    def test_mean(self):
        cfg = PCATransform.Config(D_IN, D_OUT)
        pca = PCATransform(cfg)
        with pytest.raises(RuntimeError):
            pca.mean

        x = torch.rand(D_IN)
        pca.mean = x
        assert torch.equal(pca.mean, x)

    def test_is_trained(self):
        cfg = PCATransform.Config(D_IN, D_OUT)
        pca = PCATransform(cfg)
        assert not pca.is_trained

        torch.manual_seed(0)
        x = torch.rand(N, D_IN)
        pca.train(x)
        assert pca.is_trained

    @pytest.mark.parametrize(["d_in", "d_out"], [(8, 8), (16, 16), (16, 8)])
    def test_train(self, d_in: int, d_out: int):
        cfg = PCATransform.Config(d_in, d_out)
        pca = PCATransform(cfg)
        torch.manual_seed(0)
        x = torch.rand(N, d_in)
        pca.train(x)

        assert utils.is_equal_shape(pca.weight, [d_in, d_out])

        # Checks the orthogonal matrices
        if d_in == d_out:
            torch.testing.assert_close(pca.weight @ pca.weight.T, torch.eye(d_in))

    def test_encode(self):
        cfg = PCATransform.Config(D_IN, D_OUT)
        pca = PCATransform(cfg)
        torch.manual_seed(0)
        x = torch.rand(N, D_IN)
        pca.train(x)
        assert utils.is_equal_shape(pca.encode(x), [N, D_OUT])

    def test_decode(self):
        cfg = PCATransform.Config(D_IN, D_OUT)
        pca = PCATransform(cfg)
        torch.manual_seed(0)
        x = torch.rand(N, D_IN)
        pca.train(x)
        assert utils.is_equal_shape(pca.decode(torch.rand(N, D_OUT)), [N, D_IN])
        torch.testing.assert_close(pca.decode(pca.encode(x)), x)
