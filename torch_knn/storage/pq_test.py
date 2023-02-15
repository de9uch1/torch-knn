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
    @pytest.mark.parametrize(
        "shape,expectation",
        [
            ((N, D), does_not_raise()),
        ],
    )
    def test_check_shape(self, shape, expectation):
        cfg = PQStorage.Config(D, M=M, ksub=ksub)
        x = torch.rand(shape)
        storage = PQStorage(cfg)
        storage.train(x)
        codes = storage.encode(x)
        with expectation:
            storage.check_shape(codes)
        storage.add(x)
        with expectation:
            storage.check_shape(storage.storage)

    def test_encode(self):
        cfg = PQStorage.Config(D, M=M, ksub=ksub)
        x = torch.rand(N, D)
        storage = PQStorage(cfg)
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
        storage.train(x)
        codes = storage.encode(x)
        recons = storage.decode(codes)
        assert utils.is_equal_shape(x, recons)
        assert torch.less((x - recons).norm() ** 2 / x.norm() ** 2, 0.1)

    def test_train(self):
        cfg = PQStorage.Config(D, M=M, ksub=ksub)
        x = torch.rand(N, D)
        storage = PQStorage(cfg)
        torch.manual_seed(0)
        storage = storage.train(x)
        torch.manual_seed(0)
        kmeans = ParallelKmeans(ksub, dsub, M)
        codebook = kmeans.train(x.view(N, M, dsub))
        torch.testing.assert_close(storage.codebook, codebook)

    def test_is_trained(self):
        cfg = PQStorage.Config(D, M=M, ksub=ksub)
        x = torch.rand(N, D)
        storage = PQStorage(cfg)
        assert not storage.is_trained
        storage = storage.train(x)
        assert storage.is_trained
