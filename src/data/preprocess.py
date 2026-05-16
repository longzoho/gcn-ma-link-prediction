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
