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

    Diagonal is zero (no self-loop contribution).
    """
    S = torch.zeros(num_nodes, num_nodes)
    neighbors: dict[int, set[int]] = {n: set(G.neighbors(n)) for n in G.nodes()}
    for i in G.nodes():
        for j in G.nodes():
            if i == j:
                continue
            common = neighbors.get(i, set()) & neighbors.get(j, set())
            if not common:
                continue
            S[i, j] = len(common) * as_[i].item()
    return S


def enhanced_adjacency(A: torch.Tensor, S: torch.Tensor, beta: float) -> torch.Tensor:
    """Compute Ŝ = A + β·S + I per paper Eq. 5."""
    n = A.shape[0]
    return A + beta * S + torch.eye(n, dtype=A.dtype, device=A.device)
