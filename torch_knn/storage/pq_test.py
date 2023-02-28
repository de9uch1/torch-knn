from contextlib import nullcontext as does_not_raise

import pytest
import torch

from torch_knn import utils
from torch_knn.module.kmeans import ParallelKmeans
from torch_knn.storage.pq import PQStorage

D = 8
M = 4
dsub = 2
ksub = 16
N = ksub * 5


class TestPQStorage:
    class TestConfig:
        def test___post_init__(self):
            with pytest.raises(ValueError):
                PQStorage.Config(D=8, M=3)

            cfg = PQStorage.Config(D=D, M=M)
            assert isinstance(cfg, PQStorage.Config)
            assert cfg.D == D
            assert cfg.M == M

    def test_M(self):
        storage = PQStorage(PQStorage.Config(D, M=M, ksub=ksub))
        assert storage.M == M

    def test_dsub(self):
        storage = PQStorage(PQStorage.Config(D, M=M, ksub=ksub))
        assert storage.dsub == D // M

    def test_ksub(self):
        storage = PQStorage(PQStorage.Config(D, M=M, ksub=ksub))
        assert storage.ksub == ksub

    def test_encode(self):
        cfg = PQStorage.Config(D, M=M, ksub=ksub)
        x = torch.rand(N, D)
        storage = PQStorage(cfg)
        with pytest.raises(RuntimeError):
            storage.encode(x)
        storage.train(x)
        codes = storage.encode(x)
        assert utils.is_equal_shape(codes, [N, M])
        torch.testing.assert_close(
            codes,
            torch.cdist(
                x.view(N, M, dsub).transpose(0, 1).contiguous(), storage.codebook
            )
            .argmin(-1)
            .transpose(0, 1)
            .contiguous()
            .to(torch.uint8),
        )

    def test_decode(self):
        cfg = PQStorage.Config(D, M=M, ksub=ksub)
        x = torch.rand(N, D)
        storage = PQStorage(cfg)
        with pytest.raises(RuntimeError):
            storage.decode(x)
        storage.train(x)
        codes = storage.encode(x)

        with pytest.raises(RuntimeError):
            storage.decode(codes[None, :])
        with pytest.raises(RuntimeError):
            storage.decode(codes.transpose(0, 1))

        recons = storage.decode(codes)
        assert utils.is_equal_shape(x, recons)
        assert torch.less((x - recons).norm() ** 2 / x.norm() ** 2, 0.1)

    def test_is_trained(self):
        cfg = PQStorage.Config(D, M=M, ksub=ksub)
        x = torch.rand(N, D)
        storage = PQStorage(cfg)
        assert not storage.is_trained
        storage = storage.train(x)
        assert storage.is_trained

    @pytest.mark.parametrize(
        "x,exception",
        [
            (torch.rand(N, D), does_not_raise()),
            (torch.rand(2, N, D), pytest.raises(RuntimeError)),
            (torch.rand(3), pytest.raises(RuntimeError)),
            (torch.rand(N * M, dsub), pytest.raises(RuntimeError)),
        ],
    )
    def test_train(self, x, exception):
        cfg = PQStorage.Config(D, M=M, ksub=ksub)
        storage = PQStorage(cfg)
        torch.manual_seed(0)
        with exception:
            storage = storage.train(x)
        if exception == does_not_raise():
            torch.manual_seed(0)
            kmeans = ParallelKmeans(ksub, dsub, M)
            codebook = kmeans.train(x.view(N, M, dsub))
            torch.testing.assert_close(storage.codebook, codebook)

    class TestADTable:
        def test_Nq(self):
            x = torch.rand(N, M, ksub)
            adtable = PQStorage.ADTable(x)
            assert adtable.Nq == N

        def test_M(self):
            x = torch.rand(N, M, ksub)
            adtable = PQStorage.ADTable(x)
            assert adtable.M == M

        def test_ksub(self):
            x = torch.rand(N, M, ksub)
            adtable = PQStorage.ADTable(x)
            assert adtable.ksub == ksub

        def test_lookup(self):
            x = torch.rand(N, M, ksub)
            adtable = PQStorage.ADTable(x)
            codes = torch.randint(ksub, size=(N, M), dtype=torch.uint8)
            adists = adtable.lookup(codes)
            assert utils.is_equal_shape(adists, [N, N])

    def test_compute_adtable(self):
        cfg = PQStorage.Config(D, M=M, ksub=ksub)
        x = torch.rand(N, D)
        storage = PQStorage(cfg)
        with pytest.raises(RuntimeError):
            storage.compute_adtable(x)
        storage.train(x)
        codes = storage.encode(x)
        assert utils.is_equal_shape(codes, [N, M])
        adtable = storage.compute_adtable(x)
        assert isinstance(adtable, PQStorage.ADTable)
        assert utils.is_equal_shape(adtable, [N, M, ksub])

    def test_adc(self):
        """Tests ADC consistency."""
        torch.manual_seed(0)
        x = torch.rand(N, D)
        cfg = PQStorage.Config(D, M=M, ksub=ksub)
        storage = PQStorage(cfg)
        storage.train(x)
        adtable = storage.compute_adtable(x)
        adists = adtable.lookup(storage.encode(x))
        assert utils.is_equal_shape(adists, [N, N])
        recons_x = storage.decode(storage.encode(x))
        torch.testing.assert_close(adists, storage.metric.compute_distance(x, recons_x))
