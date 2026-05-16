"""NRNAE preprocessing per paper §3.

For each snapshot:
    CC(i) = 2*R(i) / (K(i)*(K(i)-1))     # clustering coefficient
    AS(i) = degree(i) * CC(i)             # aggregation strength
    S(i,j) = |N(i) ∩ N(j)| * AS(i)        # pairwise aggregation
    Ŝ = A + β·S + I                       # enhanced adjacency
"""
import networkx as nx
import torch


def clustering_coefficient(G: nx.Graph, num_nodes: int) -> torch.Tensor:
    """Return per-node CC as a [num_nodes] tensor.

    Convention: CC = 0 for isolated nodes or nodes with degree < 2.
    """
    cc_dict = nx.clustering(G)
    cc = torch.zeros(num_nodes)
    for i, val in cc_dict.items():
        cc[i] = val
    return cc


def aggregation_strength(G: nx.Graph, cc: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return AS(i) = degree(i) * CC(i) as a [num_nodes] tensor."""
    deg = torch.zeros(num_nodes)
    for i, d in G.degree():
        deg[i] = d
    return deg * cc


def pairwise_aggregation(G: nx.Graph, as_: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute S(i,j) = |N(i) ∩ N(j)| * AS(i) as a dense [N, N] tensor.

    Vectorized: builds adjacency A, computes common-neighbor count via A @ A,
    then broadcasts AS along rows. Diagonal forced to zero.

    Edge case: an empty graph (no edges) returns the zero matrix because A is
    all-zero and the matmul yields zero.
    """
    A = torch.zeros(num_nodes, num_nodes)
    for u, v in G.edges():
        A[u, v] = 1.0
        A[v, u] = 1.0
    common = A @ A                       # common[i,j] = |N(i) ∩ N(j)|
    common.fill_diagonal_(0.0)           # no self contribution
    S = common * as_.unsqueeze(1)        # broadcast AS along rows
    return S


def enhanced_adjacency(A: torch.Tensor, S: torch.Tensor, beta: float) -> torch.Tensor:
    """Compute Ŝ = A + β·S + I per paper Eq. 5."""
    n = A.shape[0]
    return A + beta * S + torch.eye(n, dtype=A.dtype, device=A.device)


def compute_snapshot_features(
    edges: list[tuple[int, int]],
    num_nodes: int,
    beta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute per-snapshot node features [N, 3] = [degree, CC, AS] and Ŝ [N, N].

    Args:
        edges: undirected edge list. Duplicates and self-loops are ignored.
        num_nodes: total nodes in the network (constant across snapshots).
        beta: NRNAE mixing factor.

    Returns:
        features: torch.Tensor [N, 3]
        S_hat: torch.Tensor [N, N], dense
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from((u, v) for u, v in edges if u != v)

    cc = clustering_coefficient(G, num_nodes)
    as_ = aggregation_strength(G, cc, num_nodes)

    deg = torch.zeros(num_nodes)
    for i, d in G.degree():
        deg[i] = d

    features = torch.stack([deg, cc, as_], dim=1)

    A = torch.zeros(num_nodes, num_nodes)
    for u, v in G.edges():
        A[u, v] = 1.0
        A[v, u] = 1.0
    S = pairwise_aggregation(G, as_, num_nodes)
    S_hat = enhanced_adjacency(A, S, beta)
    return features, S_hat
