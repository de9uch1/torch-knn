import pytest
import torch
from torch_knn.index.base import Index
from torch_knn.metrics import CosineMetric, IPMetric, L2Metric, Metric
from torch_knn.storage.flat import FlatStorage

D = 8
DTYPE = torch.float32


class IndexMock(Index):
    storage_type = FlatStorage

    def is_trained(self) -> bool:
        return True

    def train(self, x):
        pass

    def search(self, query, k):
        pass


class DummyMetric(Metric):
    @staticmethod
    def compute_distance(a, b):
        pass


class TestIndex:
    @pytest.mark.parametrize("metric", [L2Metric(), IPMetric(), CosineMetric()])
    def test___init__(self, metric: Metric):
        index_cfg = Index.Config(D, DTYPE, metric)
        index = IndexMock(index_cfg)
        assert index.cfg == index_cfg
        assert index.cfg != Index.Config(D + 1, DTYPE, metric)
        assert index.cfg != Index.Config(D, DTYPE, DummyMetric())
        assert index.metric == metric

    def new_storage(self):
        cfg = Index.Config(D, DTYPE, L2Metric())
        storage = IndexMock.new_storage(cfg)
        assert cfg.D == storage.D
        assert cfg.dtype == storage.dtype

    def test_D(self):
        index_cfg = Index.Config(D, DTYPE, L2Metric())
        index = IndexMock(index_cfg)
        assert index.D == D

    def test_add(self):
        index_cfg = Index.Config(D, DTYPE, L2Metric())
        storage_cfg = FlatStorage.Config(D, DTYPE)
        storage = FlatStorage(storage_cfg)
        index = IndexMock(index_cfg)
        index.storage = storage

        n = 3
        niter = 4
        r = torch.rand(n * niter, D)
        for i in range(niter):
            index.add(r[n * i : n * (i + 1)])
            assert storage.N == n * (i + 1)
        assert torch.equal(storage.data, r)
