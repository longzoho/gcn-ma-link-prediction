# GCN_MA Link Prediction (Reproduction)

Reproduction of *Dynamic network link prediction with node representation learning from graph convolutional networks* (Mei & Zhao, Scientific Reports 2024).

## Setup

Requires Python 3.11 and CUDA 12.1 capable GPU.

```bash
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.4.0+cu121.html
```

## Quickstart

```bash
python scripts/download_datasets.py --dataset collegemsg
python scripts/train.py --config configs/experiments/gcn_ma_collegemsg.yaml
```

## Layout

See `docs/superpowers/specs/2026-05-16-gcn-ma-link-prediction-design.md` for design.

## Slides

Thesis defense deck (Vietnamese, ~20 min) organized around temporal graph evolution.

- Outline (Gamma input):  `docs/slides/thesis_defense_outline.md` (22 main + 6 appendix)
- Spec:                    `docs/superpowers/specs/2026-05-18-thesis-slides-design.md`
- Plan:                    `docs/superpowers/plans/2026-05-18-thesis-slides.md`
- Generated deck URL:      `docs/slides/gamma_deck_url.txt` (10-slide highlights — Gamma plan caps numCards at 10)
- Topology plots:          `results/report/plots/topology_map_2d*.png`, `dataset_snapshots_grid.png`, `edge_growth_density.png`

Render all plots (existing + topology):

```bash
python scripts/make_plots.py --plots all
```
