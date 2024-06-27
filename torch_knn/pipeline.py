from typing import Generator, Optional

import torch.nn as nn
from torch import Tensor

from torch_knn.index.base import Index
from torch_knn.transform.base import Transform


class Pipeline(nn.Module):
    """Pipeline class.

    Args:
        index (Index): An index.
        pre_transforms (list[Transform], optional): Pre-transformations.
    """

    def __init__(self, index: Index, pre_transforms: Optional[list[Transform]] = None):
        super().__init__()
        self.index = index
        self.pre_transforms = nn.ModuleList(
            pre_transforms if pre_transforms is not None else []
        )

    @property
    def N(self) -> int:
        """The number of vectors that are added to the index."""
        return self.index.N

    @property
    def D(self) -> int:
        """Dimension size of the input vectors."""
        if len(self.pre_transforms) > 0:
            return self.pre_transforms[0].d_in
        return self.index.D

    def chain(self) -> Generator[Transform | Index, None, None]:
        """Yields pipeline modules.

        Yields:
            Transform | Index: The transformations or the index.
        """
        yield from self.pre_transforms
        yield self.index

    def transform(self, x: Tensor) -> Tensor:
        """Transforms the input vectors.

        Args:
            x (Tensor): The input vectors of shape `(n, d_in)`
              where `d_in` is the input dimension size of the first transformation.

        Returns:
            Tensor: The transformed vectors of shape `(n, d_out)`
              where `d_out` is the output dimension size of the last transformation.
        """
        for t in self.pre_transforms:
            x = t.encode(x)
        return x

    def encode(self, x: Tensor) -> Tensor:
        """Encodes the given vectors.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, d_in)`
              where `d_in` is the input dimension size of the first transformation..

        Returns:
            torch.Tensor: Encoded vectors of shape `(N, d_out)`
              where `d_out` is the output dimension size of the index.
        """
        x = self.transform(x)
        return self.index.encode(x)

    def decode(self, x: Tensor) -> Tensor:
        """Decodes the given vectors or codes.

        Args:
            x (torch.Tensor): The input vectors or codes.

        Returns:
            torch.Tensor: Decoded vectors.
        """
        x = self.index.decode(x)
        for t in self.pre_transforms:
            x = t.decode(x)
        return x

    def fit(self, x: Tensor) -> "Pipeline":
        """Trains the pipeline with the given vectors.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            Pipeline: The pipeline object.
        """
        for t in self.pre_transforms:
            t.fit(x)
            x = t.encode(x)
        self.index.fit(x)
        return self

    def add(self, x: Tensor) -> None:
        """Adds the given vectors to the pipeline.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.
        """
        x = self.transform(x)
        self.index.add(x)

    def search(self, query: Tensor, k: int = 1, **kwargs) -> tuple[Tensor, Tensor]:
        """Searches the k-nearest-neighbor vectors.

        Args:
            query (torch.Tensor): Query vectors of shape `(Nq, D)`.
            k (int): Number of nearest neighbors to be returned.
            **kwargs (Dict[str, Any]): Keyword arguments for the search method.

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
              - torch.Tensor: Distances between querys and keys of shape `(Nq, k)`.
              - torch.Tensor: Indices of the k-nearest-neighbors of shape `(Nq, k)`.
        """
        query = self.transform(query)
        return self.index.search(query, k=k, **kwargs)
