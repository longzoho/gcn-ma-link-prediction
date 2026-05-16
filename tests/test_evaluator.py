import torch
from torch_geometric.data import Data

from src.data.base import DynamicGraph
from src.eval.evaluator import evaluate_dynamic
from src.models.base import DynamicLinkPredictor


class _ConstantModel(DynamicLinkPredictor):
    """Returns deterministic embeddings independent of input.

    With identity embeddings and dot-product decoder, AUC ≈ 0.5 on random
    positives/negatives.
    """

    def __init__(self, num_nodes: int, dim: int = 4):
        super().__init__()
        self.Z = torch.nn.Parameter(torch.randn(num_nodes, dim), requires_grad=False)

    def forward(self, snapshots, time_step):
        return self.Z


def _dummy_graph(N: int, T: int) -> DynamicGraph:
    snaps = []
    for _ in range(T):
        d = Data(edge_index=torch.randint(0, N, (2, N)), num_nodes=N)
        d.x = torch.randn(N, 3)
        d.S_hat = torch.eye(N)
        snaps.append(d)
    return DynamicGraph(
        snapshots=snaps,
        node_features=torch.eye(N),
        num_nodes=N,
        num_time_steps=T,
        dataset_name="dummy",
    )


def test_evaluator_returns_auc_and_ap():
    N, T = 20, 5
    graph = _dummy_graph(N, T)
    model = _ConstantModel(N, dim=4)
    test_pairs = {
        t + 1: {
            "pos": torch.tensor([[0, 1, 2], [3, 4, 5]]),
            "neg": torch.tensor([[6, 7, 8], [9, 10, 11]]),
        }
        for t in range(T - 2, T - 1)  # one test step
    }
    metrics = evaluate_dynamic(
        model, graph, time_steps=[T - 2], test_pairs=test_pairs
    )
    assert "auc" in metrics and "ap" in metrics
    assert 0.0 <= metrics["auc"] <= 1.0
    assert 0.0 <= metrics["ap"] <= 1.0


def test_evaluator_pools_across_time_steps():
    """A single AUC computed over concatenated scores from all evaluated steps."""
    N, T = 20, 5
    graph = _dummy_graph(N, T)
    model = _ConstantModel(N, dim=4)
    test_pairs = {
        t + 1: {
            "pos": torch.tensor([[0, 1], [2, 3]]),
            "neg": torch.tensor([[4, 5], [6, 7]]),
        }
        for t in range(T - 3, T - 1)  # two test steps
    }
    metrics = evaluate_dynamic(
        model, graph, time_steps=list(range(T - 3, T - 1)), test_pairs=test_pairs
    )
    assert "auc" in metrics
