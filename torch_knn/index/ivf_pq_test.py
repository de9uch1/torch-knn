import pytest
import torch

from torch_knn import metrics, utils
from torch_knn.index.ivf_pq import IVFPQIndex
from torch_knn.module.ivf import InvertedFile

D = 8
NLISTS = 5
M = 4
ksub = 16
N = ksub * 4
Nq = 8


class TestIVFPQIndex:
    def test___init__(self):
        index = IVFPQIndex(IVFPQIndex.Config(D, M=M, ksub=ksub, nlists=NLISTS))
        assert isinstance(index.ivf, InvertedFile)

    def test_centroids(self):
        index = IVFPQIndex(IVFPQIndex.Config(D, M=M, ksub=ksub, nlists=NLISTS))
        with pytest.raises(RuntimeError):
            index.centroids
        x = torch.rand(N, D)
        index.train(x)
        assert utils.is_equal_shape(index.centroids, [NLISTS, D])

    def test_is_trained(self):
        index = IVFPQIndex(IVFPQIndex.Config(D, M=M, ksub=ksub, nlists=NLISTS))
        x = torch.rand(N, D)
        assert not index.is_trained
        index.train(x)
        assert index.is_trained

    @pytest.mark.parametrize("residual", [True, False])
    def test_train(self, residual: bool):
        index = IVFPQIndex(
            IVFPQIndex.Config(D, M=M, ksub=ksub, nlists=NLISTS, residual=residual)
        )
        x = torch.rand(N, D)
        index.train(x)
        assert index.ivf.is_trained and index.is_trained
        assert index.ivf.centroids is not None and utils.is_equal_shape(
            index.ivf.centroids, [NLISTS, D]
        )

    @pytest.mark.parametrize("residual", [True, False])
    def test_add(self, residual: bool):
        index = IVFPQIndex(
            IVFPQIndex.Config(D, M=M, ksub=ksub, nlists=NLISTS, residual=residual)
        )
        x = torch.rand(N, D)
        index.train(x)
        assert index.ivf.centroids is not None
        assert index.N == 0
        index.add(x)
        assert index.N == N
        assert torch.cat(index.ivf.invlists).size(0) == N

        if residual:
            x = x - index.ivf.centroids[index.ivf.assign(x)]
        codes = index.encode(x)
        assert torch.equal(index.data, codes)
        index.add(x)
        assert index.N == 2 * N

    @pytest.mark.parametrize(
        "metric,eps",
        [
            (metrics.L2Metric(), 0.05),
            (metrics.CosineMetric(), 0.98),
        ],
    )
    @pytest.mark.parametrize("residual", [True, False])
    @pytest.mark.parametrize("precompute", [True, False])
    @pytest.mark.parametrize("nprobe", [1, 2, 4])
    @pytest.mark.parametrize("k", [1, 4])
    def test_search(
        self,
        metric: metrics.Metric,
        eps: float,
        residual: bool,
        nprobe: int,
        precompute: bool,
        k: int,
    ):
        torch.manual_seed(0)
        index = IVFPQIndex(
            IVFPQIndex.Config(
                D,
                M=M,
                ksub=ksub,
                metric=metric,
                nlists=NLISTS,
                residual=residual,
                precompute=precompute,
            )
        )
        x = torch.rand(N, D)
        xq = x[:Nq]
        index.train(x)
        index.add(x)
        dists, idxs = index.search(xq, k=k, nprobe=nprobe)
        # Shape
        assert utils.is_equal_shape(dists, idxs)
        assert utils.is_equal_shape(dists, [Nq, k])
        # Self search
        if isinstance(metric, metrics.CosineMetric):
            assert torch.greater_equal(dists[:, 0].mean(), eps)
        else:
            assert torch.less_equal(dists[:, 0].mean() / D, eps)

        if isinstance(metric, metrics.L2Metric):
            assert torch.equal(idxs[:, 0], torch.arange(Nq))
