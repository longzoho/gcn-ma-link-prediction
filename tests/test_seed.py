import numpy as np
import torch

from src.utils.seed import set_seed


def test_set_seed_reproduces_numpy():
    set_seed(42)
    a = np.random.randn(3)
    set_seed(42)
    b = np.random.randn(3)
    np.testing.assert_array_equal(a, b)


def test_set_seed_reproduces_torch():
    set_seed(42)
    a = torch.randn(3)
    set_seed(42)
    b = torch.randn(3)
    torch.testing.assert_close(a, b)
