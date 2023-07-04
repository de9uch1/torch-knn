from typing import List, Tuple, Union

import torch


def is_equal_shape(
    a: Union[torch.Tensor, torch.Size],
    b: Union[torch.Tensor, torch.Size, List[int], Tuple[int]],
) -> bool:
    """Returns whether a and b have the same shape.

    Args:
        a (Union[torch.Tensor, torch.Size]): An input tensor.
        b (Union[torch.Tensor, torch.Size, List[int], Tuple[int]):
          An input tensor compared the shape with a.

    Returns:
        bool: Whether a and b have the same shape.

    Raises:
        NotImplementedError: When an unsupported type object is given.
    """
    if isinstance(a, torch.Tensor):
        a_shape = a.shape
    elif isinstance(a, torch.Size):
        a_shape = a
    else:
        raise NotImplementedError(f"Type of `a` (`{type(a)}`) is not supported.")

    if isinstance(b, torch.Tensor):
        b_shape = b.shape
    elif isinstance(b, torch.Size):
        b_shape = b
    elif isinstance(b, (list, tuple)):
        b_shape = torch.Size(b)
    else:
        raise NotImplementedError(f"Type of `b` (`{type(b)}`) is not supported.")
    return a_shape == b_shape


def pad(tensors: List[torch.Tensor], padding_idx: int) -> torch.Tensor:
    """Pads multiple sequences into a single tensor.

    Args:
        tensors (List[Tensor]): A list of 1-D tensors.
        padding_idx (int): Padding index.

    Returns:
        torch.Tensor: Padded tensor.
    """
    max_len = max(len(t) for t in tensors)
    new_tensor = torch.full(
        (len(tensors), max_len),
        fill_value=padding_idx,
        dtype=tensors[0].dtype,
        device=tensors[0].device,
    )
    for i, t in enumerate(tensors):
        new_tensor[i, : len(t)] = t
    return new_tensor
