from dataclasses import dataclass
from typing import Optional

import torch

from torch_knn import utils
from torch_knn.metrics import MetricL2
from torch_knn.module.kmeans import ParallelKmeans

from .base import Storage


class StoragePQ(Storage):
    """Product Quantized storage class.

    Args:
        cfg (StoragePQ.Config): Configuration for this class.

    Attributes:
        - data (torch.Tensor): The PQ code storage of shape `(N, M)`.
        - codebook (torch.Tensor): The PQ codebook of shape `(M, ksub, dsub)`.
    """

    def __init__(self, cfg: "StoragePQ.Config"):
        super().__init__(cfg)
        self._data = self.data.to(cfg.code_dtype)
        self.register_buffer("_codebook", torch.rand(self.M, self.ksub, self.dsub))

    @dataclass
    class Config(Storage.Config):
        """Base class for storage config.

        - D (int): Dimension size of input vectors.
        - metric (Metric): Metric for dinstance computation.
        - M (int): The number of sub-vectors.
        - ksub (int): Codebook size of a sub-space. (default: 256)
        - code_dtype (torch.dtype): DType for stored codes. (default: torch.uint8)
        - train_niter (int): Number of training iteration.
        """

        M: int = 1
        ksub: int = 256
        code_dtype: torch.dtype = torch.uint8
        train_niter: int = 50

        def __post_init__(self):
            if self.D % self.M > 0:
                raise ValueError(f"D={self.D} must be divisible by M={self.M}.")

    cfg: "StoragePQ.Config"

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
    def codebook(self) -> torch.Tensor:
        """PQ codebook of shape `(M, ksub, dsub)`."""
        return self._codebook

    @codebook.setter
    def codebook(self, codebook: torch.Tensor) -> None:
        """Sets PQ codebook of shape `(M, ksub, dsub)`."""
        if not utils.is_equal_shape(codebook, [self.M, self.ksub, self.dsub]):
            raise ValueError("The codebook must be the shape of `(M, ksub, dsub)`.")
        self._codebook = codebook

    @property
    def data(self) -> torch.Tensor:
        """Storage object of shape `(N, M)`."""
        return self._data

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encodes the given vectors.

        StoragePQ class encodes `x` by looking up the codebook.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            torch.Tensor: Encoded vectors of shape `(N, M)`.
        """
        N, D = x.size()
        # x: N x D -> M x N x dsub
        x = x.view(N, self.M, self.dsub).transpose(0, 1).contiguous()

        # assignments: M x N x ksub
        assignments = MetricL2.assign(x, self.codebook)
        return assignments.to(self.cfg.code_dtype).transpose(0, 1).contiguous()

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
        # x: N x M x dsub -> N x D
        return x.view(N, self.D)

    def fit(
        self, x: torch.Tensor, codebook: Optional[torch.Tensor] = None
    ) -> "StoragePQ":
        """Trains the index with the given vectors.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            StoragePQ: The storage object.
        """
        kmeans = ParallelKmeans(self.ksub, self.dsub, self.M)
        self._codebook = kmeans.fit(
            x.view(x.size(0), self.M, self.dsub).transpose(0, 1).contiguous(),
            niter=self.cfg.train_niter,
            initial_centroids=codebook,
        )
        return self

    class ADTable(torch.Tensor):
        """Asymmetric distance table class.

        This class is inherited from :class:`torch.Tensor`.

        The shape is `(Nq, M, ksub)`.
        """

        @property
        def Nq(self) -> int:
            """Number of query vectors."""
            return self.size(0)

        @property
        def M(self) -> int:
            """Number of sub-vectors."""
            return self.size(1)

        @property
        def ksub(self) -> int:
            """Codebook size of a sub-space."""
            return self.size(2)

        def lookup(self, codes: torch.Tensor) -> torch.Tensor:
            """Looks up the distance table by PQ codes.

            Args:
                codes (torch.Tensor): PQ codes of shape `(Nk, M)` or `(Nq, Nk, M)`.
                  If the shape is `(Nk, M)`, PQ codes are shared between Nq queries.

            Returns:
                torch.Tensor: Distances of shape `(Nq, Nk)`.
            """
            if codes.dim() == 2:
                codes = codes.expand(self.Nq, *codes.size())
            elif codes.dim() != 3 or codes.size(0) != self.Nq:
                raise ValueError(
                    "The argument `codes` must have [Nq, Nk, M] or [Nk, M] dimensions."
                )
            Nk = codes.size(1)

            # adtable: Nq x M x ksub -> Nq x Nk x M x ksub
            # Note that Tensor.expand() does not allocate new memory.
            adtable = self[:, None].expand(self.Nq, Nk, self.M, self.ksub)

            ## This is the parallel version of PQ linear scan.
            ## Because of memory efficient, for-loop scanning over PQ subspaces is
            ## employed in this implementation.

            ## codes: Nq x Nk x M -> Nq x Nk x M x 1
            ## gather() -> Nq x Nk x M x 1
            ## squeeze() -> Nq x Nk x M
            ## sum() -> Nq x Nk
            # distances = torch.Tensor(
            #     adtable.gather(dim=-1, index=codes.long().unsqueeze(-1))
            #     .squeeze(-1)
            #     .sum(dim=-1)
            # )

            ## This is the for-loop version of PQ linear scan that is employed.
            distances = adtable.new_zeros(self.Nq, Nk)
            for m in range(self.M):
                distances += (
                    adtable[:, :, m]
                    .gather(dim=-1, index=codes[:, :, m].long().unsqueeze(-1))
                    .squeeze(-1)
                )
            return distances

    def compute_adtable(self, query: torch.Tensor) -> "ADTable":
        """Computes an ADC table.

        Args:
            query (torch.Tensor): Query vectors of shape `(Nq, D)`.

        Returns:
            ADTable: Asymmetric distance table of shape `(Nq, M, ksub)`.
        """
        Nq, D = query.size()
        # query: Nq x M x dsub -> M x Nq x dsub
        query = query.view(Nq, self.M, self.dsub).transpose(0, 1).contiguous()
        # codebook: M x ksub x dsub
        # adtable: M x Nq x ksub -> Nq x M x ksub
        return self.ADTable(
            self.metric.compute_distance(query, self.codebook)
            .transpose(0, 1)
            .contiguous()
        )
