import pytest
import torch

from torch_knn import utils


@pytest.mark.parametrize("a", [torch.rand(3, 8), torch.rand(1), torch.rand(4, 9, 5)])
@pytest.mark.parametrize("b", [torch.rand(3, 8), torch.rand(1), torch.rand(4, 9, 5)])
@pytest.mark.parametrize("a_type", [torch.Tensor, torch.Size, list, tuple, set])
@pytest.mark.parametrize("b_type", [torch.Tensor, torch.Size, list, tuple, set])
def test_is_equal_shape(a, b, a_type, b_type):
    def get_shape(x, x_type):
        if x_type is torch.Tensor:
            return x
        elif x_type is torch.Size:
            return x.shape
        elif x_type in {list, tuple, set}:
            return x_type(x.shape)

    a_shape = get_shape(a, a_type)
    b_shape = get_shape(b, b_type)
    expected = a.shape == b.shape
    if a_type not in {torch.Tensor, torch.Size} or b_type not in {
        torch.Tensor,
        torch.Size,
        list,
        tuple,
    }:
        with pytest.raises(NotImplementedError):
            utils.is_equal_shape(a_shape, b_shape)
    else:
        assert utils.is_equal_shape(a_shape, b_shape) == expected


def test_pad():
    N = 4
    L = 5
    padding_idx = -1
    tensors = [torch.arange(i + L) for i in range(N)]
    expected = torch.zeros(N, L + N - 1, dtype=torch.long).fill_(padding_idx)
    for i, t in enumerate(tensors):
        expected[i, : len(t)] = t
    assert torch.equal(utils.pad(tensors, padding_idx), expected)
