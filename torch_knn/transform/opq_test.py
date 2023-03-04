import pytest
import torch

from torch_knn import utils
from torch_knn.transform.opq import OPQTransform

D_IN = 16
D_OUT = 16
M = 8
KSUB = 16
N = KSUB * 4


class TestOPQTransform:
    def test___init__(self):
        cfg = OPQTransform.Config(D_IN, D_OUT)
        opq = OPQTransform(cfg)
        assert opq.cfg.d_in == D_IN
        assert opq.cfg.d_out == D_OUT
        assert opq._weight is None

    class TestConfig:
        with pytest.raises(ValueError):
            OPQTransform.Config(8, 16)

    def test_weight(self):
        cfg = OPQTransform.Config(D_IN, D_OUT, M=M, ksub=KSUB)
        opq = OPQTransform(cfg)
        with pytest.raises(RuntimeError):
            opq.weight

        x = torch.rand(D_OUT, D_IN)
        opq.weight = x
        assert torch.equal(opq.weight, x)

    def test_is_trained(self):
        cfg = OPQTransform.Config(D_IN, D_OUT, M=M, ksub=KSUB, train_niter=10)
        opq = OPQTransform(cfg)
        assert not opq.is_trained

        torch.manual_seed(0)
        x = torch.rand(N, D_IN)
        opq.train(x)
        assert opq.is_trained

    @pytest.mark.parametrize(["d_in", "d_out"], [(8, 8), (16, 16), (16, 8)])
    def test_train(self, d_in: int, d_out: int):
        cfg = OPQTransform.Config(d_in, d_out, M=M, ksub=KSUB, train_niter=50)
        opq = OPQTransform(cfg)
        torch.manual_seed(0)
        x = torch.rand(N, d_in)
        opq.train(x)

        # Checks the orthogonal matrices
        if d_in >= d_out:
            torch.testing.assert_close(opq.weight @ opq.weight.T, torch.eye(d_out))
        if d_in <= d_out:
            torch.testing.assert_close(opq.weight.T @ opq.weight, torch.eye(d_in))

    def test_encode(self):
        cfg = OPQTransform.Config(D_IN, D_OUT, M=M, ksub=KSUB)
        opq = OPQTransform(cfg)
        torch.manual_seed(0)
        x = torch.rand(N, D_IN)
        opq.train(x)
        assert utils.is_equal_shape(opq.encode(x), [N, D_OUT])

    def test_decode(self):
        cfg = OPQTransform.Config(D_IN, D_OUT, M=M, ksub=KSUB)
        opq = OPQTransform(cfg)
        torch.manual_seed(0)
        x = torch.rand(N, D_IN)
        opq.train(x)
        assert utils.is_equal_shape(opq.decode(torch.rand(N, D_OUT)), [N, D_IN])
        torch.testing.assert_close(opq.decode(opq.encode(x)), x)
