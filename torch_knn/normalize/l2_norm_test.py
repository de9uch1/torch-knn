import torch

from torch_knn.normalize.l2_norm import L2NormNormalize

D_IN = 8
N = 10


class TestL2NormalizationTransform:
    def test_encode(self):
        cfg = L2NormNormalize.Config()
        l2norm = L2NormNormalize(cfg)
        x = torch.rand(N, D_IN)
        torch.testing.assert_close(
            l2norm.encode(x),
            x / (x**2).sum(-1, keepdim=True) ** 0.5,
        )
