from dataclasses import dataclass
from typing import Optional

import torch
from torch_knn.module.kmeans import ParallelKmeans
from torch_knn.storage.base import Storage


class PQStorage(Storage):
    """Product Quantized storage class.

    Args:
        storage (torch.Tensor): The storage object.

    Attributes:
        - storage (torch.Tensor): The PQ code storage of shape `(N, M)`.
        - codebook (torch.Tensor): The PQ codebook of shape `(M, ksub, dsub)`.
    """

    def __init__(
        self,
        cfg: "PQStorage.Config",
        storage: torch.Tensor = torch.Tensor(),
        codebook: Optional[torch.Tensor] = None,
    ):
        super().__init__(cfg, storage=storage)
        self._codebook = codebook

    @dataclass
    class Config(Storage.Config):
        """Base class for storage config.

        Args:
            D (int): Dimension size of input vectors.
            dtype (torch.dtype): The input vector dtype. (default: torch.float32)
            M (int): The number of sub-vectors.
            ksub (int): Codebook size of a sub-space. (default: 256)
            code_dtype (torch.dtype): DType for stored codes. (default: torch.uint8)
        """

        M: int = 1
        ksub: int = 256
        code_dtype: torch.dtype = torch.uint8

        def __post_init__(self):
            if self.D % self.M > 0:
                raise ValueError(f"D={self.D} must be divisible by M={self.M}.")

    cfg: "PQStorage.Config"

    @property
    def M(self) -> int:
        """Number of sub-vectors."""
        return self.cfg.M

    @property
    def dsub(self) -> int:
        """Dimension size of each sub-vector."""
        return self.D // self.M

    @property
    def ksub(self) -> int:
        """Codebook size of a sub-space."""
        return self.cfg.ksub

    @property
    def codebook(self) -> Optional[torch.Tensor]:
        """PQ codebook of shape `(M, ksub, dsub)`."""
        return self._codebook

    def check_shape(self, storage: torch.Tensor) -> torch.Tensor:
        """Checks whether the storage tensor shape is valid or not.

        Args:
            storage (torch.Tensor): The storage tensor.

        Returns:
            torch.Tensor: The input storage tensor.

        Raises:
            ValueError: When given the wrong shape storage.
        """
        if storage.dim() != 2 or storage.size(-1) != self.M:
            raise ValueError(f"The storage must be `N x M` dimensions.")
        return storage

    @property
    def storage(self) -> torch.Tensor:
        """Storage object of shape `(N, M)`."""
        return self._storage

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the given vectors.

        PQStorage class encodes `x` by looking up the codebook.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            torch.Tensor: Encoded vectors of shape `(N, M)`.
        """
        if not self.is_trained:
            raise RuntimeError("The storage must be trained.")

        if x.dim() != 2 or x.size(-1) != self.D:
            raise RuntimeError("The input vectors must have `(N, D)` dimensions.")

        N, D = x.size()
        # x: N x D -> M x N x dsub
        x = x.view(N, self.M, self.dsub).transpose(0, 1).contiguous()

        # distance: M x N x ksub
        distance = torch.cdist(x, self.codebook)
        return (
            distance.argmin(dim=-1).to(self.cfg.code_dtype).transpose(0, 1).contiguous()
        )

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Decodes the given vectors.

        Args:
            codes (torch.Tensor): The input codes of shape `(N, M)`.

        Returns:
            torch.Tensor: Decoded vectors of shape `(N, D)`.
        """
        if codes.dim() != 2 or codes.size(-1) != self.M:
            raise RuntimeError("The input codes must have `(N, M)` dimensions.")

        N = codes.size(0)
        codes = codes.long()

        # x[n, m, d] = codebook[m][codes[n][m]][d]
        x = self.codebook[torch.arange(self.M), codes]
        # (N, M, 1, dsub) -> (N, D)
        return x.view(N, self.D).to(self.dtype)

    @property
    def is_trained(self) -> bool:
        """Returns whether the storage is trained or not."""
        return self.codebook is not None

    def train(self, x: torch.Tensor) -> "PQStorage":
        """Trains the index with the given vectors.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            PQStorage: The trained storage object.
        """
        kmeans = ParallelKmeans(self.ksub, self.dsub, self.M)
        self._codebook = kmeans.train(x.view(x.size(0), self.M, self.dsub))
        return self
