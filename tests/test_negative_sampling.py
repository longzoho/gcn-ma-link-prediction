import torch

from src.training.negative_sampling import sample_negative_edges


def test_returns_correct_count():
    pos = torch.tensor([[0, 1, 2], [1, 2, 3]])  # 3 edges
    neg = sample_negative_edges(pos, num_nodes=10, num_samples=3, seed=42)
    assert neg.shape == (2, 3)


def test_no_self_loops():
    pos = torch.tensor([[0, 1], [1, 2]])
    neg = sample_negative_edges(pos, num_nodes=10, num_samples=100, seed=42)
    assert (neg[0] != neg[1]).all()


def test_does_not_contain_positive_edges():
    pos = torch.tensor([[0, 1, 2], [1, 2, 3]])
    pos_set = {(int(u), int(v)) for u, v in pos.t().tolist()}
    neg = sample_negative_edges(pos, num_nodes=20, num_samples=100, seed=42)
    neg_set = {(int(u), int(v)) for u, v in neg.t().tolist()}
    assert neg_set.isdisjoint(pos_set)


def test_deterministic_with_seed():
    pos = torch.tensor([[0, 1], [2, 3]])
    a = sample_negative_edges(pos, num_nodes=50, num_samples=20, seed=42)
    b = sample_negative_edges(pos, num_nodes=50, num_samples=20, seed=42)
    assert torch.equal(a, b)


def test_different_seeds_differ():
    pos = torch.tensor([[0, 1], [2, 3]])
    a = sample_negative_edges(pos, num_nodes=50, num_samples=20, seed=42)
    b = sample_negative_edges(pos, num_nodes=50, num_samples=20, seed=999)
    assert not torch.equal(a, b)
