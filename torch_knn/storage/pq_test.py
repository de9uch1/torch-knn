from contextlib import nullcontext as does_not_raise

import pytest
import torch

from torch_knn import utils
from torch_knn.module.kmeans import ParallelKmeans

from .pq import StoragePQ

D = 8
M = 4
dsub = 2
ksub = 16
N = ksub * 5


class TestStoragePQ:
    class TestConfig:
        def test___post_init__(self):
            with pytest.raises(ValueError):
                StoragePQ.Config(D=8, M=3)

            cfg = StoragePQ.Config(D=D, M=M)
            assert isinstance(cfg, StoragePQ.Config)
            assert cfg.D == D
            assert cfg.M == M

    def test_M(self):
        storage = StoragePQ(StoragePQ.Config(D, M=M, ksub=ksub))
        assert storage.M == M

    def test_dsub(self):
        storage = StoragePQ(StoragePQ.Config(D, M=M, ksub=ksub))
        assert storage.dsub == D // M

    def test_ksub(self):
        storage = StoragePQ(StoragePQ.Config(D, M=M, ksub=ksub))
        assert storage.ksub == ksub

    def test_codebook(self):
        storage = StoragePQ(StoragePQ.Config(D, M=M, ksub=ksub))
        x = torch.rand(N, D)
        storage.fit(x)
        assert utils.is_equal_shape(storage.codebook, [M, ksub, dsub])

        storage = StoragePQ(StoragePQ.Config(D, M=M, ksub=ksub))
        x = torch.rand(M, N, D)
        with pytest.raises(ValueError):
            storage.codebook = x

        x = torch.rand(M, ksub, dsub)
        storage.codebook = x
        assert torch.equal(storage.codebook, x)

    def test_encode(self):
        cfg = StoragePQ.Config(D, M=M, ksub=ksub)
        x = torch.rand(N, D)
        storage = StoragePQ(cfg)
        storage.fit(x)
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
        cfg = StoragePQ.Config(D, M=M, ksub=ksub)
        x = torch.rand(N, D)
        storage = StoragePQ(cfg)
        with pytest.raises(RuntimeError):
            storage.decode(x)
        storage.fit(x)
        codes = storage.encode(x)

        with pytest.raises(RuntimeError):
            storage.decode(codes[None, :])
        with pytest.raises(RuntimeError):
            storage.decode(codes.transpose(0, 1))

        recons = storage.decode(codes)
        assert utils.is_equal_shape(x, recons)
        assert torch.less((x - recons).norm() ** 2 / x.norm() ** 2, 0.1)

    @pytest.mark.parametrize(
        "x,exception",
        [
            (torch.rand(N, D), does_not_raise()),
            (torch.rand(2, N, D), pytest.raises(RuntimeError)),
            (torch.rand(3), pytest.raises(RuntimeError)),
            (torch.rand(N * M, dsub), pytest.raises(RuntimeError)),
        ],
    )
    def test_fit(self, x, exception):
        cfg = StoragePQ.Config(D, M=M, ksub=ksub)
        storage = StoragePQ(cfg)
        torch.manual_seed(0)
        with exception:
            storage = storage.fit(x)
        if exception == does_not_raise():
            torch.manual_seed(0)
            kmeans = ParallelKmeans(ksub, dsub, M)
            codebook = kmeans.fit(x.view(N, M, dsub))
            torch.testing.assert_close(storage.codebook, codebook)

    class TestADTable:
        def test_Nq(self):
            x = torch.rand(N, M, ksub)
            adtable = StoragePQ.ADTable(x)
            assert adtable.Nq == N

        def test_M(self):
            x = torch.rand(N, M, ksub)
            adtable = StoragePQ.ADTable(x)
            assert adtable.M == M

        def test_ksub(self):
            x = torch.rand(N, M, ksub)
            adtable = StoragePQ.ADTable(x)
            assert adtable.ksub == ksub

        def test_lookup(self):
            x = torch.rand(N, M, ksub)
            adtable = StoragePQ.ADTable(x)

            wrong_shape_codes = torch.randint(ksub, size=(N,), dtype=torch.uint8)
            with pytest.raises(ValueError):
                adtable.lookup(wrong_shape_codes)
            wrong_shape_codes = torch.randint(
                ksub, size=(N + 1, N, M), dtype=torch.uint8
            )
            with pytest.raises(ValueError):
                adtable.lookup(wrong_shape_codes)

            codes = torch.randint(ksub, size=(N, M), dtype=torch.uint8)
            adists = adtable.lookup(codes)
            assert utils.is_equal_shape(adists, [N, N])

    def test_compute_adtable(self):
        cfg = StoragePQ.Config(D, M=M, ksub=ksub)
        x = torch.rand(N, D)
        storage = StoragePQ(cfg)
        storage.fit(x)
        codes = storage.encode(x)
        assert utils.is_equal_shape(codes, [N, M])
        adtable = storage.compute_adtable(x)
        assert isinstance(adtable, StoragePQ.ADTable)
        assert utils.is_equal_shape(adtable, [N, M, ksub])

    def test_adc(self):
        """Tests ADC consistency."""
        torch.manual_seed(0)
        x = torch.rand(N, D)
        cfg = StoragePQ.Config(D, M=M, ksub=ksub)
        storage = StoragePQ(cfg)
        storage.fit(x)
        adtable = storage.compute_adtable(x)
        adists = adtable.lookup(storage.encode(x))
        assert utils.is_equal_shape(adists, [N, N])
        recons_x = storage.decode(storage.encode(x))
        torch.testing.assert_close(adists, storage.metric.compute_distance(x, recons_x))
