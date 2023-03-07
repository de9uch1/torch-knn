import torch
from torch import Tensor

from torch_knn.index import LinearFlatIndex
from torch_knn.index.linear_pq import LinearPQIndex
from torch_knn.pipeline import Pipeline
from torch_knn.transform.base import Transform

N = 3
D = 8


class AddOneTransform(Transform):
    """Add-one transform class for tests."""

    cfg: Transform.Config

    @property
    def is_trained(self) -> bool:
        """Returns whether this class is trained or not."""
        return True

    def train(self, x) -> "AddOneTransform":
        """Trains vector transformation for this class.

        Args:
            x (Tensor): Training vectors of shape `(n, d_in)`.

        Returns:
            AddOneTransform: Trained this class.
        """
        return self

    def encode(self, x) -> Tensor:
        """Transforms the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d_in)`.

        Returns:
            Tensor: Transformed vectors of shape `(n, d_out)`.
        """
        return x + 1

    def decode(self, x) -> Tensor:
        """Returns the input vectors.

        Args:
            x (Tensor): Input vectors of shape `(n, d_out)`.

        Returns:
            Tensor: Input vectors of shape `(n, d_in)`.
        """
        return x - 1


class LearnedTransform(AddOneTransform):
    def __init__(self, cfg: "Transform.Config") -> None:
        super().__init__(cfg)
        self._is_trained = False

    @property
    def is_trained(self) -> bool:
        """Returns whether this class is trained or not."""
        return self._is_trained

    def train(self, x) -> "LearnedTransform":
        """Trains vector transformation for this class.

        Args:
            x (Tensor): Training vectors of shape `(n, d_in)`.

        Returns:
            AddOneTransform: Trained this class.
        """
        self._is_trained = True
        return self


class TestPipeline:
    def test___init__(self):
        index = LinearFlatIndex(LinearFlatIndex.Config(D))

        pipeline = Pipeline(index)
        assert pipeline.index is index
        assert pipeline.pre_transforms == []

        transform = AddOneTransform(AddOneTransform.Config(D, D))
        pipeline = Pipeline(index, [transform])
        assert pipeline.index is index
        assert pipeline.pre_transforms == [transform]

    def test_N(self):
        x = torch.rand(N, D)
        index = LinearFlatIndex(LinearFlatIndex.Config(D))
        transform = AddOneTransform(AddOneTransform.Config(D, D))
        index.add(x)

        pipeline = Pipeline(index, [transform])
        assert pipeline.N == index.N

    def test_D(self):
        index = LinearFlatIndex(LinearFlatIndex.Config(D))

        pipeline = Pipeline(index)
        assert pipeline.D == index.D

        pipeline = Pipeline(index, [AddOneTransform(AddOneTransform.Config(D, D))])
        assert pipeline.D == D
        pipeline = Pipeline(
            index,
            [
                AddOneTransform(AddOneTransform.Config(D * 2, D)),
                AddOneTransform(AddOneTransform.Config(D, D)),
            ],
        )
        assert pipeline.D == D * 2

    def test_chain(self):
        index = LinearFlatIndex(LinearFlatIndex.Config(D))
        pipeline = Pipeline(index)
        assert list(pipeline.chain()) == [index]

        transform = AddOneTransform(AddOneTransform.Config(D, D))
        pipeline = Pipeline(index, [transform])
        assert list(pipeline.chain()) == [transform, index]

    def test_transform(self):
        x = torch.rand(N, D)
        index = LinearFlatIndex(LinearFlatIndex.Config(D))

        pipeline = Pipeline(index)
        torch.equal(pipeline.transform(x), x)

        transform = AddOneTransform(AddOneTransform.Config(D, D))
        pipeline = Pipeline(index, [transform])
        torch.equal(pipeline.transform(x), x + 1)

        pipeline = Pipeline(index, [transform for _ in range(10)])
        torch.equal(pipeline.transform(x), x + 10)

    def test_encode(self):
        x = torch.rand(N, D)
        index = LinearFlatIndex(LinearFlatIndex.Config(D))
        pipeline = Pipeline(index)
        assert torch.equal(pipeline.encode(x), x)

        transform = AddOneTransform(AddOneTransform.Config(D, D))
        pipeline = Pipeline(index, [transform])
        assert torch.equal(pipeline.encode(x), x + 1)

    def test_decode(self):
        x = torch.rand(N, D)
        index = LinearFlatIndex(LinearFlatIndex.Config(D))
        transform = AddOneTransform(AddOneTransform.Config(D, D))
        pipeline = Pipeline(index, [transform])
        assert torch.allclose(pipeline.decode(pipeline.encode(x)), x)

    def test_is_trained(self):
        x = torch.rand(N, D)
        index = LinearPQIndex(LinearPQIndex.Config(D, M=D // 2, ksub=4))
        pipeline = Pipeline(index)
        assert not pipeline.is_trained
        index.train(x)
        assert pipeline.is_trained

        index = LinearPQIndex(LinearPQIndex.Config(D, M=D // 2, ksub=4))
        transform = LearnedTransform(LearnedTransform.Config(D, D))
        pipeline = Pipeline(index, [transform])
        assert not pipeline.is_trained
        index.train(x)
        assert not pipeline.is_trained
        transform.train(x)
        assert pipeline.is_trained

    def test_train(self):
        x = torch.rand(N, D)
        index = LinearPQIndex(LinearPQIndex.Config(D, M=D // 2, ksub=4))
        torch.manual_seed(0)
        index.train(x)
        expected_codebook = index.codebook

        pipeline = Pipeline(index)
        torch.manual_seed(0)
        pipeline.train(x)
        assert torch.equal(pipeline.index.codebook, expected_codebook)

        transform = AddOneTransform(AddOneTransform.Config(D, D))
        pipeline = Pipeline(index, [transform])
        torch.manual_seed(0)
        pipeline.train(x - 1)
        assert torch.equal(pipeline.index.codebook, expected_codebook)

        index = LinearPQIndex(LinearPQIndex.Config(D, M=D // 2, ksub=4))
        transform = LearnedTransform(LearnedTransform.Config(D, D))
        pipeline = Pipeline(index, [transform])
        assert not pipeline.is_trained
        pipeline.train(x)
        assert pipeline.is_trained

    def test_add(self):
        x = torch.rand(N, D)
        index = LinearFlatIndex(LinearFlatIndex.Config(D))
        pipeline = Pipeline(index)
        pipeline.add(x)
        assert index.N == N

        index = LinearFlatIndex(LinearFlatIndex.Config(D))
        transform = AddOneTransform(AddOneTransform.Config(D, D))
        pipeline = Pipeline(index, [transform])
        pipeline.add(x)
        assert index.N == N
        assert torch.equal(index.data, x + 1)

    def test_search(self):
        x = torch.rand(N, D)
        index = LinearFlatIndex(LinearFlatIndex.Config(D))
        transform = AddOneTransform(AddOneTransform.Config(D, D))
        pipeline = Pipeline(index, [transform])
        pipeline.add(x)
        dists, idxs = pipeline.search(x, k=N)
        torch.testing.assert_close(dists[:, 0], torch.zeros(N))
        torch.testing.assert_close(idxs[:, 0], torch.arange(N))
