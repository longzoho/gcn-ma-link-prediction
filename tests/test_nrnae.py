import networkx as nx
import torch

from src.data.preprocess import aggregation_strength, clustering_coefficient, enhanced_adjacency, pairwise_aggregation


def _triangle_plus_tail():
    """Nodes 0,1,2 form a triangle; node 3 attached to 2; node 4 attached to 3.

    Degrees: 0:2, 1:2, 2:3, 3:2, 4:1
    Triangles passing through node: 0:1, 1:1, 2:1, 3:0, 4:0
    CC(i) = 2*R(i) / (K(i)*(K(i)-1))
        CC(0) = 2*1 / (2*1) = 1.0
        CC(1) = 2*1 / (2*1) = 1.0
        CC(2) = 2*1 / (3*2) = 0.333...
        CC(3) = 0   (degree 2 but no triangle)
        CC(4) = 0   (degree 1 → undefined; convention 0)
    AS(i) = degree(i) * CC(i)
        AS(0) = 2.0, AS(1) = 2.0, AS(2) = 1.0, AS(3) = 0.0, AS(4) = 0.0
    """
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
    G = nx.Graph()
    G.add_edges_from(edges)
    return G


def test_clustering_coefficient_matches_paper_formula():
    G = _triangle_plus_tail()
    cc = clustering_coefficient(G, num_nodes=5)
    assert cc.shape == (5,)
    torch.testing.assert_close(cc[0], torch.tensor(1.0))
    torch.testing.assert_close(cc[1], torch.tensor(1.0))
    torch.testing.assert_close(cc[2], torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(cc[3], torch.tensor(0.0))
    torch.testing.assert_close(cc[4], torch.tensor(0.0))


def test_aggregation_strength():
    G = _triangle_plus_tail()
    cc = clustering_coefficient(G, num_nodes=5)
    as_ = aggregation_strength(G, cc, num_nodes=5)
    assert as_.shape == (5,)
    torch.testing.assert_close(as_[0], torch.tensor(2.0))
    torch.testing.assert_close(as_[1], torch.tensor(2.0))
    torch.testing.assert_close(as_[2], torch.tensor(1.0))
    torch.testing.assert_close(as_[3], torch.tensor(0.0))
    torch.testing.assert_close(as_[4], torch.tensor(0.0))


def test_handles_isolated_node():
    G = nx.Graph()
    G.add_node(0)
    G.add_node(1)
    cc = clustering_coefficient(G, num_nodes=2)
    assert cc.shape == (2,)
    torch.testing.assert_close(cc, torch.zeros(2))


def test_pairwise_aggregation_shape_and_values():
    """For the triangle-plus-tail graph:
    N(0)={1,2}, N(1)={0,2}, N(2)={0,1,3}, N(3)={2,4}, N(4)={3}
    |N(0) ∩ N(1)| = |{2}| = 1
    |N(0) ∩ N(2)| = |{1}| = 1
    |N(1) ∩ N(2)| = |{0}| = 1
    |N(2) ∩ N(3)| = |{}| = 0
    S(i,j) = |N(i)∩N(j)| * AS(i)
        S(0,1) = 1 * 2 = 2
        S(0,2) = 1 * 2 = 2
        S(1,0) = 1 * 2 = 2
        S(1,2) = 1 * 2 = 2
        S(2,0) = 1 * 1 = 1
        S(2,1) = 1 * 1 = 1
    """
    G = _triangle_plus_tail()
    cc = clustering_coefficient(G, num_nodes=5)
    as_ = aggregation_strength(G, cc, num_nodes=5)
    S = pairwise_aggregation(G, as_, num_nodes=5)
    assert S.shape == (5, 5)
    assert S[0, 1].item() == 2.0
    assert S[1, 0].item() == 2.0
    assert S[2, 0].item() == 1.0
    assert S[0, 0].item() == 0.0  # no self
    assert S[2, 3].item() == 0.0  # no common neighbor


def test_enhanced_adjacency_includes_identity():
    """Ŝ = A + β·S + I."""
    G = _triangle_plus_tail()
    cc = clustering_coefficient(G, num_nodes=5)
    as_ = aggregation_strength(G, cc, num_nodes=5)
    S = pairwise_aggregation(G, as_, num_nodes=5)
    A = torch.zeros(5, 5)
    for u, v in G.edges():
        A[u, v] = 1.0
        A[v, u] = 1.0
    S_hat = enhanced_adjacency(A, S, beta=0.8)
    expected_diag = torch.ones(5)  # identity contribution
    torch.testing.assert_close(torch.diag(S_hat), expected_diag)
    # Off-diagonal at (0,1): A=1, S=2, beta=0.8 → 1 + 0.8*2 = 2.6
    torch.testing.assert_close(S_hat[0, 1], torch.tensor(2.6))


from src.data.preprocess import compute_snapshot_features


def test_compute_snapshot_features_returns_3dim_node_features_and_S_hat():
    """compute_snapshot_features(edges, num_nodes, beta) →
       node_features [N, 3] = [degree, CC, AS], S_hat [N, N] sparse-friendly dense.
    """
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4)]
    features, S_hat = compute_snapshot_features(edges, num_nodes=5, beta=0.8)
    assert features.shape == (5, 3)
    # node 2: degree=3, CC=1/3, AS=1
    torch.testing.assert_close(features[2, 0], torch.tensor(3.0))
    torch.testing.assert_close(features[2, 1], torch.tensor(1.0 / 3.0))
    torch.testing.assert_close(features[2, 2], torch.tensor(1.0))
    assert S_hat.shape == (5, 5)


def test_compute_snapshot_features_empty_edges():
    """Snapshot with no edges → all zero features, S_hat = I."""
    features, S_hat = compute_snapshot_features([], num_nodes=5, beta=0.8)
    torch.testing.assert_close(features, torch.zeros(5, 3))
    torch.testing.assert_close(S_hat, torch.eye(5))


def test_pairwise_aggregation_vectorized_matches_python_loop():
    """Vectorized rewrite must match the original closed-form values.

    Graph: (0,1),(0,2),(1,2),(2,3),(3,4)
    N(0)={1,2}, N(1)={0,2}, N(2)={0,1,3}, N(3)={2,4}, N(4)={3}
    Common-neighbor counts (non-zero pairs):
      |N(0)∩N(1)|=1  |N(0)∩N(2)|=1  |N(0)∩N(3)|=1  (node 2 shared with 3)
      |N(1)∩N(2)|=1  |N(1)∩N(3)|=1                 (node 2 shared with 3)
      |N(2)∩N(4)|=1                                 (node 3 shared with 4)
    S(i,j) = |N(i)∩N(j)| * AS(i); AS = [2,2,1,0,0]
      S[0,1]=2, S[0,2]=2, S[0,3]=2
      S[1,0]=2, S[1,2]=2, S[1,3]=2
      S[2,0]=1, S[2,1]=1, S[2,4]=1
    """
    G = _triangle_plus_tail()
    cc = clustering_coefficient(G, num_nodes=5)
    as_ = aggregation_strength(G, cc, num_nodes=5)
    S = pairwise_aggregation(G, as_, num_nodes=5)
    expected = torch.zeros(5, 5)
    expected[0, 1] = 2.0; expected[0, 2] = 2.0; expected[0, 3] = 2.0
    expected[1, 0] = 2.0; expected[1, 2] = 2.0; expected[1, 3] = 2.0
    expected[2, 0] = 1.0; expected[2, 1] = 1.0; expected[2, 4] = 1.0
    torch.testing.assert_close(S, expected)


def test_pairwise_aggregation_larger_random_graph():
    """Run on a 50-node Erdős–Rényi graph; verify properties hold."""
    import networkx as nx
    G = nx.erdos_renyi_graph(50, p=0.1, seed=42)
    cc = clustering_coefficient(G, num_nodes=50)
    as_ = aggregation_strength(G, cc, num_nodes=50)
    S = pairwise_aggregation(G, as_, num_nodes=50)
    assert S.shape == (50, 50)
    # Diagonal must be zero
    assert torch.diag(S).abs().sum().item() == 0.0
    # S[i, j] should be 0 when AS[i] == 0
    for i in range(50):
        if as_[i].item() == 0.0:
            assert S[i].abs().sum().item() == 0.0
