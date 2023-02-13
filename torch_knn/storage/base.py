import abc
from typing import Set

import torch


class Storage(abc.ABC):
    """ "Base class for storage.

    Args:
        storage (torch.Tensor): The storage object.
        dtype (torch.dtype): The storage dtype. (default: torch.float32)
    """

    support_dtypes: Set[torch.dtype] = {torch.float32, torch.float16}

    def __init__(
        self,
        storage: torch.Tensor = torch.Tensor(),
        dtype: torch.dtype = torch.float32,
    ):
        self.dtype: torch.dtype = self.check_dtype(dtype)
        self._storage = storage.to(self.dtype)

    @classmethod
    @abc.abstractmethod
    def check_shape(cls, storage: torch.Tensor) -> torch.Tensor:
        """Checks whether the storage tensor shape is valid or not.

        Args:
            storage (torch.Tensor): The storage tensor.

        Returns:
            torch.Tensor: The input storage tensor.

        Raises:
            ValueError: When given the wrong shape storage.
        """

    @classmethod
    def check_dtype(cls, dtype: torch.dtype) -> torch.dtype:
        """Checks whether the specified dtype is supported or not.

        Args:
            dtype (torch.dtype): The specified dtype.

        Returns:
            torch.dtype: The specified dtype.

        Raises:
            ValueError: When given the unsupported dtype.
        """
        if dtype not in cls.support_dtypes:
            raise ValueError(f"The dtype `{dtype}` is not supported for this storage.")
        return dtype

    @property
    def storage(self) -> torch.Tensor:
        """Storage object."""
        return self._storage
