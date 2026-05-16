"""Adapter for IBM/EvolveGCN EvolveGCN-O baseline.

Wraps the upstream EGCN class (in `third_party/EvolveGCN/egcn_o.py`) as a
DynamicLinkPredictor. Replaces the one-hot identity feature convention
(spec §6.6) with a learnable nn.Embedding to keep RAM tractable on
large datasets.
"""
import sys
from pathlib import Path

import torch
from torch import nn

# Upstream submodule must be on sys.path for `from egcn_o import EGCN`
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UPSTREAM = _REPO_ROOT / "third_party" / "EvolveGCN"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from egcn_o import EGCN as UpstreamEGCN_O  # noqa: E402
import utils as upstream_utils  # noqa: E402 — provides Namespace

from src.models.base import DynamicLinkPredictor
from src.models.gcn_ma.link_decoder import LinkDecoderMLP


class EvolveGCN_O(DynamicLinkPredictor):
    """EvolveGCN-O wrapped as a DynamicLinkPredictor.

    Differences from upstream:
        - Uses our learnable nn.Embedding (not one-hot identity).
        - MLP decoder shared with GCN_MA for fair comparison.
        - Adapter consumes our DynamicGraph format (translates internally).

    Note: `num_layers` is informational — upstream EGCN always uses 2 layers
    (layer_1_feats + layer_2_feats). The argument is accepted for config
    uniformity with GCN_MA's API but must be 2.
    """

    def __init__(
        self,
        num_nodes: int,
        feat_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers != 2:
            raise ValueError(
                f"EvolveGCN-O upstream is fixed at 2 layers; got num_layers={num_layers}"
            )
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # Learnable node embedding (replaces one-hot identity, spec §6.6 deviation)
        self.node_emb = nn.Embedding(num_nodes, feat_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # Upstream EGCN-O construction
        upstream_args = upstream_utils.Namespace(
            {
                "feats_per_node": feat_dim,
                "layer_1_feats": hidden_dim,
                "layer_2_feats": hidden_dim,
            }
        )
        self.core = UpstreamEGCN_O(
            args=upstream_args,
            activation=nn.RReLU(),
            device=torch.device("cpu"),
        )

        # Shared MLP decoder pattern with GCN_MA
        self.decoder = LinkDecoderMLP(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

    def forward(self, snapshots, time_step):
        """Run upstream EGCN-O through snapshots [0..time_step].

        Implemented in Task 4. This stub keeps the class instantiable.
        """
        raise NotImplementedError("EvolveGCN_O.forward — implemented in Task 4")

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
