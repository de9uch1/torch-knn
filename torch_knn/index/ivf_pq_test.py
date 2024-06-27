import pytest
import torch
from torch import Tensor

from torch_knn import metrics, utils
from torch_knn.module.ivf import InvertedFile

from .ivf_pq import IndexIVFPQ
from .linear_pq import IndexLinearPQ

D = 8
NLISTS = 5
M = 4
ksub = 16
N = ksub * 4
Nq = 3
K = 4


class MockMetric(metrics.Metric):
    """L2 metric for squared Euclidean distance computation."""

    @staticmethod
    def compute_distance(a: Tensor, b: Tensor) -> Tensor:
        """Computes distance between two vectors.

        Args:
            a (torch.Tensor): Input vectors of shape `(n, dim)` or `(b, n, dim)`.
            b (torch.Tensor): Input vectors of shape `(m, dim)` or `(b, m, dim)`.

        Returns:
            torch.Tensor: Distance tensor of shape `(n, m)` or `(b, n, m)`.
        """
        return torch.cdist(a, b, p=2) ** 2


class TestIndexIVFPQ:
    def test___init__(self):
        index = IndexIVFPQ(IndexIVFPQ.Config(D, M=M, ksub=ksub, nlists=NLISTS))
        assert isinstance(index.ivf, InvertedFile)

    def test_centroids(self):
        index = IndexIVFPQ(IndexIVFPQ.Config(D, M=M, ksub=ksub, nlists=NLISTS))
        x = torch.rand(N, D)
        index.fit(x)
        assert utils.is_equal_shape(index.centroids, [NLISTS, D])

    @pytest.mark.parametrize("residual", [True, False])
    def test_compute_residual(self, residual: bool):
        index = IndexIVFPQ(
            IndexIVFPQ.Config(D, M=M, ksub=ksub, nlists=NLISTS, residual=residual)
        )
        x = torch.rand(N, D)
        index.ivf.fit(x)
        if not residual:
            assert torch.equal(index.compute_residual(x), x)
        else:
            assigns = index.ivf.assign(x)
            assert torch.equal(index.compute_residual(x), x - index.centroids[assigns])
            assert torch.allclose(
                index.centroids[assigns] + index.compute_residual(x), x
            )

    @pytest.mark.parametrize("residual", [True, False])
    @pytest.mark.parametrize("metric", [metrics.MetricL2(), metrics.MetricIP()])
    @pytest.mark.parametrize("precompute", [True, False])
    def test_fit(self, residual: bool, metric: metrics.Metric, precompute: bool):
        index = IndexIVFPQ(
            IndexIVFPQ.Config(
                D,
                metric=metric,
                M=M,
                ksub=ksub,
                nlists=NLISTS,
                residual=residual,
                precompute=precompute,
            )
        )
        x = torch.rand(N, D)
        index.fit(x)
        assert index.ivf.centroids is not None and utils.is_equal_shape(
            index.ivf.centroids, [NLISTS, D]
        )
        if residual and isinstance(metric, metrics.MetricL2) and precompute:
            assert index.precompute_table is not None
            assert utils.is_equal_shape(index.precompute_table, [NLISTS, M, ksub])
        else:
            assert index.precompute_table is None

    @pytest.mark.parametrize("residual", [True, False])
    @pytest.mark.parametrize("metric", [metrics.MetricL2(), metrics.MetricIP()])
    def test_build_precompute_table(self, residual: bool, metric: metrics.Metric):
        cfg = IndexIVFPQ.Config(
            D, metric=metric, M=M, ksub=ksub, nlists=NLISTS, residual=residual
        )
        index = IndexIVFPQ(cfg)
        x = torch.rand(N, D)
        index.fit(x)
        assert index.precompute_table is None
        table = index.build_precompute_table()
        if not residual or not isinstance(metric, metrics.MetricL2):
            assert index.precompute_table is None
            assert table is None
        else:
            assert index.precompute_table is not None
            assert table is not None
            assert torch.equal(index.precompute_table, table)
            assert utils.is_equal_shape(index.precompute_table, [NLISTS, M, ksub])
            dsub = D // M
            # centroids: nlists x M x dsub -> M x nlists x dsub
            # codebook: M x ksub x dsub
            centroids = (
                index.centroids.view(NLISTS, M, dsub).transpose(0, 1).contiguous()
            )
            codebook = index.codebook

            # cr_table: M x nlists x ksub -> nlists x M x ksub
            cr_table = (
                torch.bmm(centroids, codebook.transpose(-1, -2))
                .transpose(0, 1)
                .contiguous()
            )
            # r_sqnorms: 1 x M x ksub
            r_sqnorms = (codebook**2).sum(dim=-1).unsqueeze(0)
            term2 = 2 * cr_table + r_sqnorms
            assert torch.allclose(index.precompute_table, term2)

    @pytest.mark.parametrize("residual", [True, False])
    def test_add(self, residual: bool):
        index = IndexIVFPQ(
            IndexIVFPQ.Config(D, M=M, ksub=ksub, nlists=NLISTS, residual=residual)
        )
        x = torch.rand(N, D)
        index.fit(x)
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

    @pytest.mark.parametrize("metric", [metrics.MetricL2(), metrics.MetricIP()])
    def test_search_preassigned_noresidual(self, metric: metrics.Metric):
        ivfpq_index = IndexIVFPQ(
            IndexIVFPQ.Config(
                D, metric=metric, M=M, ksub=ksub, nlists=NLISTS, residual=False
            )
        )
        x = torch.rand(N, D)
        ivfpq_index.ivf.fit(x)
        torch.manual_seed(0)
        super(IndexIVFPQ, ivfpq_index).fit(x)
        ivfpq_index.add(x)

        linearpq_index = IndexLinearPQ(
            IndexLinearPQ.Config(D=D, metric=metric, M=M, ksub=ksub)
        )
        torch.manual_seed(0)
        linearpq_index.fit(x)
        linearpq_index.add(x)

        xq = torch.rand(Nq, D)
        centroid_indices = torch.arange(NLISTS).repeat((Nq, 1))
        dists, idxs = ivfpq_index.search_preassigned_noresidual(
            xq, K, NLISTS, centroid_indices
        )
        expected_dists, expected_idxs = linearpq_index.search(xq, k=K)
        assert torch.equal(idxs, expected_idxs)
        assert torch.allclose(dists, expected_dists)

    @pytest.mark.parametrize(
        "metric",
        [metrics.MetricL2(), metrics.MetricIP(), MockMetric()],
    )
    @pytest.mark.parametrize("precompute", [True, False])
    def test_search_preassigned_residual(
        self, metric: metrics.Metric, precompute: bool
    ):
        ivfpq_index = IndexIVFPQ(
            IndexIVFPQ.Config(
                D,
                metric=metric,
                M=M,
                ksub=ksub,
                nlists=NLISTS,
                residual=True,
                precompute=precompute,
            )
        )
        torch.manual_seed(0)
        x = torch.rand(N, D)
        ivfpq_index.fit(x)
        ivfpq_index.add(x)

        xq = x[:Nq]
        centroid_distances = ivfpq_index.metric.compute_distance(
            xq, ivfpq_index.centroids
        )
        centroid_indices = torch.arange(NLISTS).repeat((Nq, 1))

        if isinstance(metric, MockMetric):
            with pytest.raises(NotImplementedError):
                ivfpq_index.search_preassigned_residual(
                    xq, K, NLISTS, centroid_distances, centroid_indices
                )
        else:
            dists, idxs = ivfpq_index.search_preassigned_residual(
                xq, K, NLISTS, centroid_distances, centroid_indices
            )
            recons_data = x.new_zeros((N, D))
            for i, r_i in enumerate(ivfpq_index.ivf.invlists):
                recons_data[r_i] = ivfpq_index.centroids[i] + ivfpq_index.decode(
                    ivfpq_index.data[r_i]
                )
            recons_dists = ivfpq_index.metric.compute_distance(xq, recons_data)
            expected_dists, expected_idxs = ivfpq_index.metric.topk(recons_dists, k=K)
            assert torch.equal(idxs, expected_idxs)
            torch.testing.assert_close(dists, expected_dists)

    @pytest.mark.parametrize("precompute", [True, False])
    def test_compute_residual_adtable_L2(self, precompute: bool):
        metric = metrics.MetricL2()
        index = IndexIVFPQ(
            IndexIVFPQ.Config(
                D,
                metric=metric,
                M=M,
                ksub=ksub,
                nlists=NLISTS,
                residual=True,
                precompute=precompute,
            )
        )
        torch.manual_seed(0)
        x = torch.rand(N, D)
        index.fit(x)
        index.add(x)

        recons_data = x.new_zeros((N, D))
        for i, r_i in enumerate(index.ivf.invlists):
            recons_data[r_i] = index.centroids[i] + index.decode(index.data[r_i])

        xq = x[:Nq]
        centroid_distances = index.metric.compute_distance(xq, index.centroids)
        centroid_indices = torch.arange(NLISTS).repeat((Nq, 1))
        dists = x.new_zeros((Nq, N))
        adtable = index.compute_residual_adtable_L2(
            xq,
            NLISTS,
            centroid_distances.transpose(0, 1).contiguous(),
            centroid_indices.transpose(0, 1).contiguous(),
        ).view(NLISTS, Nq, M, ksub)
        for i in range(NLISTS):
            codes = index.data[index.ivf.invlists[i]]
            dists[:, index.ivf.invlists[i]] = IndexIVFPQ.ADTable(adtable[i]).lookup(
                codes
            )

        recons_dists = index.metric.compute_distance(xq, recons_data)
        torch.testing.assert_close(dists, recons_dists)

    @pytest.mark.parametrize("metric", [metrics.MetricIP()])
    @pytest.mark.parametrize("precompute", [True, False])
    def test_compute_residual_adtable_IP(
        self, metric: metrics.Metric, precompute: bool
    ):
        index = IndexIVFPQ(
            IndexIVFPQ.Config(
                D,
                metric=metric,
                M=M,
                ksub=ksub,
                nlists=NLISTS,
                residual=True,
                precompute=precompute,
            )
        )
        torch.manual_seed(0)
        x = torch.rand(N, D)
        index.fit(x)
        index.add(x)

        recons_data = x.new_zeros((N, D))
        for i, r_i in enumerate(index.ivf.invlists):
            recons_data[r_i] = index.centroids[i] + index.decode(index.data[r_i])

        xq = x[:Nq]
        centroid_distances = index.metric.compute_distance(xq, index.centroids)
        dists = x.new_zeros((Nq, N))
        adtable = index.compute_residual_adtable_IP(
            xq, NLISTS, centroid_distances.transpose(0, 1).contiguous()
        ).view(NLISTS, Nq, M, ksub)
        for i in range(NLISTS):
            codes = index.data[index.ivf.invlists[i]]
            dists[:, index.ivf.invlists[i]] = IndexIVFPQ.ADTable(adtable[i]).lookup(
                codes
            )

        recons_dists = index.metric.compute_distance(xq, recons_data)
        torch.testing.assert_close(dists, recons_dists)

    @pytest.mark.parametrize(
        "metric,eps", [(metrics.MetricL2(), 0.1), (metrics.MetricIP(), 0.25)]
    )
    @pytest.mark.parametrize("residual", [True, False])
    @pytest.mark.parametrize("precompute", [True, False])
    @pytest.mark.parametrize("nprobe", [1, 2, 4])
    @pytest.mark.parametrize("k", [1, K])
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
        index = IndexIVFPQ(
            IndexIVFPQ.Config(
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
        index.fit(x)
        index.add(x)
        dists, idxs = index.search(xq, k=k, nprobe=nprobe)
        # Shape
        assert utils.is_equal_shape(dists, idxs)
        assert utils.is_equal_shape(dists, [Nq, k])
        # Self search
        distance_matrix = metric.compute_distance(xq, x)
        expected_dists, expected_idxs = metric.topk(distance_matrix, k=k)
        assert torch.less_equal((dists - expected_dists).square().mean().sqrt(), eps)

        if isinstance(metric, metrics.MetricL2):
            assert torch.less_equal(dists[:, 0].mean() / D, eps)
            assert torch.equal(idxs[:, 0], torch.arange(Nq))
