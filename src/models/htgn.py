"""Adapter for marlin-codes/HTGN baseline.

Wraps the upstream HTGN encoder as a DynamicLinkPredictor. Projects
Poincaré ball output to Euclidean tangent space via log_map_origin
before feeding the shared MLP decoder. Uses learnable nn.Embedding for
input features (spec §6.6 deviation, same as EvolveGCN-O).

Upstream class lives at third_party/HTGN/script/models/HTGN.py and is
imported via sys.path manipulation. config.py runs argparse at import
time — we shield it by resetting sys.argv during the import.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

# Upstream submodule path — upstream uses 'script.' prefix for all internal imports,
# so we add 'third_party/HTGN' (not 'third_party/HTGN/script') to sys.path.
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_UPSTREAM = _REPO_ROOT / "third_party" / "HTGN"
if str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

# Shim: upstream config.py runs argparse.parse_args() at module-load time.
# It grabs whatever argv pytest/torch passes and may SystemExit. Reset argv
# around the import to neutralize it.
_saved_argv = sys.argv
sys.argv = sys.argv[:1]
try:
    from script.models.HTGN import HTGN as UpstreamHTGN  # noqa: E402
finally:
    sys.argv = _saved_argv

from src.models.base import DynamicLinkPredictor  # noqa: E402
from src.models.gcn_ma.link_decoder import LinkDecoderMLP  # noqa: E402
from src.models.hyperbolic_ops import log_map_origin  # noqa: E402


def _build_upstream_args(
    num_nodes: int,
    feat_dim: int,
    hidden_dim: int,
    curvature: float,
    trainable_curvature: bool,
    dropout: float,
) -> SimpleNamespace:
    """Construct the args namespace upstream HTGN expects.

    Field set derived from Task 2 inspection. Defaults for fields we don't
    care about are chosen to keep the model behavior sensible.
    """
    return SimpleNamespace(
        manifold="PoincareBall",
        curvature=float(curvature),
        fixed_curvature=int(not trainable_curvature),  # paper convention: 1 means fixed
        num_nodes=num_nodes,
        nfeat=feat_dim,
        nhid=hidden_dim,
        nout=hidden_dim,  # output dim = hidden dim
        device=torch.device("cpu"),  # adapter handles device transfer
        use_hta=1,  # hyperbolic temporal attention; paper default
        aggregation="deg",  # degree-based aggregation; paper default
        dropout=dropout,
        heads=1,
        nb_window=1,  # window size for windowed hidden states; minimal
        use_gru=True,
        model="HTGN",
    )


class HTGN(DynamicLinkPredictor):
    """HTGN encoder wrapped as a DynamicLinkPredictor.

    Differences from upstream HTGN paper:
        - Adam optimizer (not RAdam) — Hybrid policy.
        - Fixed curvature at 1.0 (paper allows learnable).
        - Shared MLP decoder (not paper's Fermi-Dirac).
        - Symmetric adjacency (Plan 3a carry-forward).
        - Learnable nn.Embedding input features (not one-hot identity).

    Implementation notes:
        - Upstream forward is per-snapshot; we loop in our forward().
        - Upstream requires init_hiddens() before first forward; we call
          it lazily.
        - update_hiddens_all_with(z) is called after each snapshot to
          advance the window.
    """

    def __init__(
        self,
        num_nodes: int,
        feat_dim: int = 64,
        hidden_dim: int = 64,
        num_layers: int = 2,
        curvature: float = 1.0,
        trainable_curvature: bool = False,
        dropout: float = 0.1,
    ):
        super().__init__()
        if num_layers != 2:
            raise ValueError(
                f"HTGN upstream uses fixed 2-layer architecture; got num_layers={num_layers}"
            )
        self.num_nodes = num_nodes
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim

        # Learnable node embedding (replaces one-hot identity, spec §6.6 deviation)
        self.node_emb = nn.Embedding(num_nodes, feat_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # Build upstream HTGN with constructed args namespace
        upstream_args = _build_upstream_args(
            num_nodes=num_nodes,
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            curvature=curvature,
            trainable_curvature=trainable_curvature,
            dropout=dropout,
        )
        self.curvature = float(curvature)
        self.core = UpstreamHTGN(upstream_args)

        # Shared decoder
        self.decoder = LinkDecoderMLP(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

    def forward(self, snapshots, time_step):
        """Run upstream HTGN through snapshots [0..time_step].

        Returns Z^{time_step} ∈ R^{num_nodes × hidden_dim} (Euclidean
        approximation via log_map_origin).
        """
        device = self.node_emb.weight.device
        N = self.num_nodes
        node_ids = torch.arange(N, device=device)
        node_emb = self.node_emb(node_ids)  # [N, feat_dim]

        # Initialize hidden state windows for upstream encoder
        self.core.init_hiddens()

        # Process snapshots [0..time_step] sequentially
        z_hyp = None
        for tau in range(time_step + 1):
            snap = snapshots[tau]
            ei = snap.edge_index.to(device)
            if ei.numel() == 0:
                ei = torch.stack([node_ids, node_ids], dim=0)
            # Symmetrize (Plan 3a EvolveGCN-O fix — bipartite signal preservation)
            ei_sym = torch.cat([ei, ei.flip(0)], dim=1)

            # Upstream forward is per-snapshot: (edge_index, x) → hyperbolic [N, nout]
            z_hyp = self.core(ei_sym, x=node_emb)

            # Advance the windowed hidden state for next snapshot
            if tau < time_step:
                self.core.update_hiddens_all_with(z_hyp)

        assert z_hyp is not None, "no snapshots processed"

        # Project Poincaré ball → Euclidean tangent space at origin
        return log_map_origin(z_hyp, c=self.curvature)

    def predict_link(self, Z, edges):
        return self.decoder(Z, edges)
