from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from torch_knn import metrics, utils
from torch_knn.index.linear_pq import LinearPQIndex
from torch_knn.module.ivf import InvertedFile
from torch_knn.storage.pq import PQStorage


class IVFPQIndex(LinearPQIndex):
    """Inverted file index class.

    Args:
        cfg (IVFPQIndex.Config): Configuration for this class.
    """

    def __init__(self, cfg: "IVFPQIndex.Config"):
        super().__init__(cfg)
        self.ivf = InvertedFile(self, cfg.nlists)
        self.precompute_table = None

    @dataclass
    class Config(PQStorage.Config):
        """IVFPQIndex configuration.

        Args:
            D (int): Dimension size of input vectors.
            M (int): The number of sub-vectors.
            ksub (int): Codebook size of a sub-space. (default: 256)
            code_dtype (torch.dtype): DType for stored codes. (default: torch.uint8)
            train_niter (int): Number of training iteration.
            metric (Metric): Metric for dinstance computation.
            nlists (int): Number of clusters.
            residual (bool): Trains PQ by the residual vectors. (default: True)
            precompute (bool): Precompute distance table for faster L2 search.
        """

        nlists: int = 1
        residual: bool = True
        precompute: bool = False

    cfg: "IVFPQIndex.Config"

    @property
    def centroids(self) -> torch.Tensor:
        """Returns centroid tensor of shape `(nlists, D)`"""
        return self.ivf.centroids

    @property
    def is_trained(self) -> bool:
        """Returns whether the index is trained or not."""
        return super().is_trained and self.ivf.is_trained

    def compute_residual(self, x: torch.Tensor) -> torch.Tensor:
        """Computes residual vectors from the assigned centorids.

        If cfg.residual is False, this method returns just inputs.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            torch.Tensor: Residual vectors of shape `(N, D)`.
        """
        if not self.cfg.residual:
            return x
        assigns = self.ivf.assign(x)
        return x - self.centroids[assigns]

    def train(self, x: torch.Tensor) -> "IVFPQIndex":
        """Trains the index with the given vectors.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.

        Returns:
            IVFFlatIndex: The trained index object.
        """
        self.ivf.train(x)
        pq_training_vectors = self.compute_residual(x)
        super().train(pq_training_vectors)
        if self.cfg.precompute:
            self.build_precompute_table()
        return self

    def build_precompute_table(self) -> Optional[torch.Tensor]:
        """Builds precompute table for faster L2 search.

        Returns:
            torch.Tensor, optional: Precompute table of shape `(nlists, M, ksub)`.
        """
        if not self.cfg.residual or not isinstance(self.metric, metrics.L2Metric):
            return None

        # cr_table: nlists x M x ksub
        cr_table = torch.einsum(
            "nmd,mkd->nmk",
            self.centroids.view(self.cfg.nlists, self.M, self.dsub),
            self.codebook,
        )
        # r_sqnorms: M x ksub
        r_sqnorms = self.codebook.norm(dim=-1) ** 2
        # term2: nlists x M x ksub
        term2 = 2 * cr_table + r_sqnorms[None, :]

        self.precompute_table = term2
        return term2

    def add(self, x: torch.Tensor) -> None:
        """Adds the given vectors to the storage.

        Args:
            x (torch.Tensor): The input vectors of shape `(N, D)`.
        """
        self.ivf.add(x)
        x = self.compute_residual(x)
        super().add(x)

    def search_preassigned_noresidual(
        self,
        query: torch.Tensor,
        k: int,
        nprobe: int,
        centroid_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Searches the k-nearest-neighbor vectors with pre-assigned coarse centroids.

        This method is used when `residual=False`.

        Args:
            query (torch.Tensor): Query vectors of shape `(Nq, D)`.
            k (int): Number of nearest neighbors to be returned.
            nprobe (int): Number of probing clusters.
            centroid_indices (torch.Tensor): Indices of the nprobe-nearest centroids
              of shape `(Nq, nprobe)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
              - torch.Tensor: Distances between querys and keys of shape `(Nq, k)`.
              - torch.Tensor: Indices of the k-nearest-neighbors of shape `(Nq, k)`.
        """
        keys = [
            torch.cat([self.ivf.invlists[c] for c in cents])
            for cents in centroid_indices.cpu()
        ]
        key_indices = utils.pad(keys, -1)
        Nq, Nk = key_indices.size()
        adtable = self.compute_adtable(query)
        distances = adtable.lookup(self.data[key_indices])
        distances = self.metric.mask(distances, key_indices.eq(-1))
        k_cand_distances, k_cand_probed_indices = self.metric.topk(
            distances, k=min(k, Nk)
        )
        k_cand_indices = key_indices.gather(-1, k_cand_probed_indices)

        k_distances = k_cand_distances.new_full(
            (Nq, k), fill_value=self.metric.farthest_value
        )
        k_indices = k_cand_indices.new_full((Nq, k), fill_value=-1)
        k_distances[:, : min(k, Nk)] = k_cand_distances
        k_indices[:, : min(k, Nk)] = k_cand_indices
        return k_distances, k_indices

    def compute_residual_adtable_L2(
        self,
        query: torch.Tensor,
        nprobe: int,
        centroid_distances: torch.Tensor,
        centroid_indices: torch.Tensor,
    ) -> PQStorage.ADTable:
        """Computes euclidean distance between a query and keys using residual vectors.

        The distance is calculated as follows:

        .. math::
          d_{L2}(q, k) &= ||q - (c + r)||^2 \\
                       &= ||q - c||^2 + (2 * (c | r) + ||r||^2) - 2 * (q | r),

        where c is the centroid vectors and r is the residual vectors.
        - The 1st term is computed by the coarse quantizer.
        - The 2nd term can be pre-computed before search time.
        - The 3rd term can be computed in only search time.

        Args:
            query (torch.Tensor): The query vectors of shape `(Nq, D)`.
            nprobe (int): Number of probing clusters.
            centroid_distances (torch.Tensor): Distances between an input and
              `nprobe`-centroids of shape `(Nq, nprobe)`.
            centroid_indices (torch.Tensor): The `nprobe`-nearest centroid IDs of
              shape `(Nq, nprobe)`.

        Returns:
            ADTable: Look up table of shape `(Nq * nprobe, M, ksub)`.
        """
        Nq, _ = centroid_distances.size()
        # term1: Nq x nprobe
        term1 = centroid_distances / self.M

        if self.precompute_table is not None:
            term2 = self.precompute_table[centroid_indices]
        else:
            # cr_table: (Nq * nprobe) x M x ksub
            cr_table = torch.einsum(
                "nmd,mkd->nmk",
                self.centroids[centroid_indices].view(Nq * nprobe, self.M, self.dsub),
                self.codebook,
            )
            # r_sqnorms: M x ksub
            r_sqnorms = self.codebook.norm(dim=-1) ** 2
            # term2: (Nq * nprobe) x M x ksub -> Nq x nprobe x M x ksub
            term2 = (2 * cr_table + r_sqnorms[None, :]).view(
                Nq, nprobe, self.M, self.ksub
            )
        # term3: Nq x M x ksub
        term3 = -2 * torch.einsum(
            "nmd,mkd->nmk", query.view(Nq, self.M, self.dsub), self.codebook
        )

        # (Nq * nprobe) x M x ksub
        return self.ADTable(
            (term1[:, :, None, None] + term2 + term3[:, None]).view(
                Nq * nprobe, self.M, self.ksub
            )
        )

    def compute_residual_adtable_IP(
        self,
        query: torch.Tensor,
        nprobe: int,
        centroid_distances: torch.Tensor,
    ) -> PQStorage.ADTable:
        """Computes inner product between a query and keys using residual vectors.

        The distance is calculated as follows:

        .. math::
          d_{IP}(q, k) &= (q | (c + r)) \\
                       &= (q | c) + (q | r)

        where c is the centroid vectors and r is the residual vectors.
        - The 1st term is computed by the coarse quantizer.
        - The 2nd term can be computed in only search time.

        Args:
            query (torch.Tensor): The query vectors of shape `(Nq, D)`.
            nprobe (int): Number of probing clusters.
            centroid_distances (torch.Tensor): Distances between an input and
              `nprobe`-centroids of shape `(Nq, nprobe)`.

        Returns:
            ADTable: Look up table of shape `(Nq * nprobe, M, ksub)`.
        """
        Nq, _ = centroid_distances.size()
        # term1: Nq x nprobe
        term1 = centroid_distances / self.M
        # term2: Nq x M x ksub
        term2 = torch.einsum(
            "nmd,mkd->nmk", query.view(Nq, self.M, self.dsub), self.codebook
        )

        # (Nq * nprobe) x M x ksub
        return self.ADTable(
            (term1[:, :, None, None] + term2[:, None]).view(
                Nq * nprobe, self.M, self.ksub
            )
        )

    def search_preassigned_residual(
        self,
        query: torch.Tensor,
        k: int,
        nprobe: int,
        centroid_distances: torch.Tensor,
        centroid_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Searches the k-nearest-neighbor vectors with pre-assigned coarse centroids.

        This method is used when `residual=True`.

        Args:
            query (torch.Tensor): Query vectors of shape `(Nq, D)`.
            k (int): Number of nearest neighbors to be returned.
            nprobe (int): Number of probing clusters.
            centroid_distances (torch.Tensor): Distance between a query and the
              nprobe-nearest centroids of shape `(Nq, nprobe)`.
            centroid_distances (torch.Tensor): Indices of the nprobe-nearest centroids
              of shape `(Nq, nprobe)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
              - torch.Tensor: Distances between querys and keys of shape `(Nq, k)`.
              - torch.Tensor: Indices of the k-nearest-neighbors of shape `(Nq, k)`.
        """
        Nq = query.size(0)
        cand_distances = centroid_distances.new_empty(Nq, nprobe, k).fill_(
            self.metric.farthest_value
        )
        cand_indices = centroid_indices.new_empty(Nq, nprobe, k).fill_(-1)

        if isinstance(self.metric, metrics.L2Metric):
            adtable = self.compute_residual_adtable_L2(
                query, nprobe, centroid_distances, centroid_indices
            ).view(Nq, nprobe, self.M, self.ksub)
        elif isinstance(self.metric, metrics.IPMetric):
            adtable = self.compute_residual_adtable_IP(
                query, nprobe, centroid_distances
            ).view(Nq, nprobe, self.M, self.ksub)
        else:
            raise NotImplementedError

        centroid_indices = centroid_indices.cpu()
        for i in range(nprobe):
            key_indices = utils.pad(
                [self.ivf.invlists[c] for c in centroid_indices[:, i]], -1
            )
            # data[key_indices]: Nq x Nk x M
            distances = self.ADTable(adtable[:, i]).lookup(self.data[key_indices])
            distances = self.metric.mask(distances, key_indices.eq(-1))
            cand_len = min(k, distances.size(1))
            k_cand_distances, k_cand_probed_indices = self.metric.topk(
                distances, k=cand_len
            )
            cand_distances[:, i, :cand_len] = k_cand_distances
            cand_indices[:, i, :cand_len] = key_indices.gather(
                -1, k_cand_probed_indices
            )

        cand_distances = cand_distances.view(Nq, nprobe * k)
        cand_indices = cand_indices.view(Nq, nprobe * k)
        k_distances, k_cand_indices = self.metric.topk(cand_distances, k=k)
        k_indices = cand_indices.gather(-1, k_cand_indices)
        return k_distances, k_indices

    def search(
        self, query: torch.Tensor, k: int = 1, nprobe: int = 1, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Searches the k-nearest-neighbor vectors.

        Args:
            query (torch.Tensor): Query vectors of shape `(Nq, D)`.
            k (int): Number of nearest neighbors to be returned.
            nprobe (int): Number of probing clusters.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
              - torch.Tensor: Distances between querys and keys of shape `(Nq, k)`.
              - torch.Tensor: Indices of the k-nearest-neighbors of shape `(Nq, k)`.
        """
        nprobe = min(max(nprobe, 1), self.cfg.nlists)
        # 1st stage search
        coarse_distances = self.metric.compute_distance(query, self.centroids)
        centroid_distances, centroid_indices = self.metric.topk(
            coarse_distances, k=nprobe
        )

        # 2nd stage search
        if self.cfg.residual:
            return self.search_preassigned_residual(
                query, k, nprobe, centroid_distances, centroid_indices
            )
        return self.search_preassigned_noresidual(query, k, nprobe, centroid_indices)
