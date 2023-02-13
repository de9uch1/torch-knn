from contextlib import nullcontext as does_not_raise

import pytest
import torch
from torch_knn.storage.flat import FlatStorage


class TestStorage:
    @pytest.mark.parametrize(
        "shape,expectation",
        [
            ((3, 8), does_not_raise()),
            ((1, 1), does_not_raise()),
            ((2,), pytest.raises(ValueError)),
            ((3, 2, 2), pytest.raises(ValueError)),
        ],
    )
    def test_check_shape(self, shape, expectation):
        x = torch.rand(shape)
        with expectation:
            storage = FlatStorage.check_shape(x)
