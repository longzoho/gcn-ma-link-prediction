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
    device: torch.device,
) -> SimpleNamespace:
    """Construct the args namespace upstream HTGN expects.

    Field set derived from Task 2 inspection. Defaults for fields we don't
    care about are chosen to keep the model behavior sensible.

    ``device`` must be the final target device because upstream HTGN stores
    plain tensors (not nn.Parameters/buffers) using args.device at __init__
    time; these won't migrate when .to(device) is called later.
    """
    return SimpleNamespace(
        manifold="PoincareBall",
        curvature=float(curvature),
        fixed_curvature=int(not trainable_curvature),  # paper convention: 1 means fixed
        num_nodes=num_nodes,
        nfeat=feat_dim,
        nhid=hidden_dim,
        nout=hidden_dim,  # output dim = hidden dim
        device=device,
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

    Device placement note:
        Upstream HTGN has two device-pinning issues that prevent normal
        .to(device) from working:

        1. ``hidden_initial`` is a plain tensor (not Parameter/buffer), so
           .to(device) on the module doesn't move it.
        2. The curvature ``c`` is an nn.Parameter that moves with .to(device),
           but child layers (HypLinear etc.) store *slices* of ``c`` at
           construction time.  After .to(device), those slices are stale
           references to the old CPU storage.

        Solution: we override ``to()`` / ``cuda()`` / ``cpu()`` to reconstruct
        ``self.core`` directly on the target device whenever the device changes.
        The core is registered as a proper submodule (self.core), so after the
        override the optimizer call ``model.parameters()`` sees fresh parameters
        on the right device.

    Implementation notes:
        - Upstream forward is per-snapshot; we loop in our forward().
        - Upstream requires init_hiddens() before first forward; we call it at
          the start of every forward() to reset hidden state per epoch.
        - update_hiddens_all_with(z) is called after each snapshot to advance
          the window.
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
        self.curvature = float(curvature)
        self._trainable_curvature = trainable_curvature
        self._dropout = dropout

        # Learnable node embedding (replaces one-hot identity, spec §6.6 deviation)
        self.node_emb = nn.Embedding(num_nodes, feat_dim)
        nn.init.xavier_uniform_(self.node_emb.weight)

        # Build upstream core on CPU initially so it is registered as a submodule
        # and visible to model.parameters() before any device migration.
        self.core = self._build_core(torch.device("cpu"))

        # Shared decoder (device-agnostic; follows .to(device))
        self.decoder = LinkDecoderMLP(
            embed_dim=hidden_dim, hidden_dim=hidden_dim, dropout=dropout
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_core(self, device: torch.device) -> UpstreamHTGN:
        """Construct UpstreamHTGN fully on *device*.

        Upstream HTGN has two categories of device-pinned tensors that are
        NOT migrated by .to(device):

        1. ``hidden_initial``: plain attribute tensor, placed on ``args.device``
           at __init__ time.  We pass the target device in args so this lands
           on the right device from the start.

        2. Curvature slices (``layer1.linear.c``, ``layer1.agg.c``, etc.): these
           are views into ``self.c`` (nn.Parameter) at construction time, stored
           as plain tensor attributes in child modules.  After .to(device), the
           Parameter storage moves but the old slices remain as stale CPU
           references.  We fix this by explicitly re-assigning every such
           attribute after calling .to(device) on the constructed core.
        """
        upstream_args = _build_upstream_args(
            num_nodes=self.num_nodes,
            feat_dim=self.feat_dim,
            hidden_dim=self.hidden_dim,
            curvature=self.curvature,
            trainable_curvature=self._trainable_curvature,
            dropout=self._dropout,
            device=device,
        )
        core = UpstreamHTGN(upstream_args).to(device)

        # Patch stale plain-tensor curvature references in every child layer.
        # After .to(device), core.c is on `device` but child module attributes
        # such as layer1.linear.c, layer1.agg.c, etc. still hold CPU slices.
        # We walk all submodules and move any plain tensor attribute to `device`.
        params_and_buffers = (
            {n for n, _ in core.named_parameters()}
            | {n for n, _ in core.named_buffers()}
        )
        for mod_name, mod in core.named_modules():
            for attr_name, val in list(mod.__dict__.items()):
                full_name = f"{mod_name}.{attr_name}" if mod_name else attr_name
                if (
                    isinstance(val, torch.Tensor)
                    and full_name not in params_and_buffers
                    and val.device != device
                ):
                    setattr(mod, attr_name, val.to(device))

        return core

    # ------------------------------------------------------------------
    # Device migration overrides
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device_like) -> torch.device | None:
        """Normalize a device specifier to a concrete torch.device, or None."""
        if device_like is None:
            return None
        if isinstance(device_like, torch.dtype):
            return None  # dtype-only call
        d = torch.device(device_like)
        # torch.device('cuda') has index None; resolve to current default.
        if d.type == "cuda" and d.index is None:
            d = torch.device("cuda", torch.cuda.current_device())
        return d

    def to(self, *args, **kwargs):  # type: ignore[override]
        """Rebuild self.core on the new device before delegating to super."""
        # Resolve target device from positional args (mirrors nn.Module.to logic).
        device_like = None
        for a in args:
            if isinstance(a, (str, torch.device)):
                device_like = a
                break
        if device_like is None:
            device_like = kwargs.get("device", None)

        target = self._resolve_device(device_like)
        if target is not None:
            current = self._resolve_device(next(self.node_emb.parameters()).device)
            if target != current:
                self.core = self._build_core(target)
        return super().to(*args, **kwargs)

    def cuda(self, device=None):  # type: ignore[override]
        idx = device if device is not None else torch.cuda.current_device()
        target = torch.device("cuda", idx)
        current = self._resolve_device(next(self.node_emb.parameters()).device)
        if target != current:
            self.core = self._build_core(target)
        return super().cuda(device)

    def cpu(self):  # type: ignore[override]
        target = torch.device("cpu")
        current = self._resolve_device(next(self.node_emb.parameters()).device)
        if target != current:
            self.core = self._build_core(target)
        return super().cpu()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, snapshots, time_step):
        """Run upstream HTGN through snapshots [0..time_step].

        Returns Z^{time_step} ∈ R^{num_nodes × hidden_dim} (Euclidean
        approximation via log_map_origin).
        """
        device = self.node_emb.weight.device
        N = self.num_nodes
        node_ids = torch.arange(N, device=device)
        node_emb = self.node_emb(node_ids)  # [N, feat_dim]

        # Reset per-sequence hidden state windows at the start of each forward.
        # hidden_initial was created on *device* (same as core), so this is safe.
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
