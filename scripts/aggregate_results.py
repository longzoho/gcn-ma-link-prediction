"""Aggregate model results across seeds and emit Markdown tables.

Usage:
  Single-model summary:
    python scripts/aggregate_results.py --model gcn_ma
  Cross-model comparison:
    python scripts/aggregate_results.py --models gcn_ma evolvegcn_o
"""
import argparse
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PAPER_TABLE2_GCN_MA = {
    "collegemsg":   {"auc": 0.9149, "ap": 0.8926},
    "bitcoinotc":   {"auc": 0.9120, "ap": 0.8943},
    "eut":          {"auc": 0.9222, "ap": 0.9082},
    "mooc_actions": {"auc": 0.9880, "ap": 0.9863},
    "lastfm":       {"auc": 0.8757, "ap": 0.8704},
    "wikipedia":    {"auc": 0.8742, "ap": 0.8575},
}

DATASETS = ["collegemsg", "bitcoinotc", "eut", "mooc_actions", "lastfm", "wikipedia"]


def _mean_std(values):
    if not values:
        return None, None
    m = statistics.mean(values)
    s = statistics.stdev(values) if len(values) > 1 else 0.0
    return m, s


def _load_records(metrics_path: Path):
    by_model_dataset = {}
    with metrics_path.open() as f:
        for line in f:
            r = json.loads(line)
            key = (r["model"], r["dataset"])
            by_model_dataset.setdefault(key, []).append(r)
    return by_model_dataset


def _single_model_table(by_md, model: str) -> str:
    lines = [f"# {model.upper()} Reproduction — Per-Dataset Summary\n"]
    lines.append("| Dataset | n seeds | AUC (mean ± std) | AP (mean ± std) | Paper AUC | Paper AP | Δ AUC | Δ AP |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds in DATASETS:
        recs = by_md.get((model, ds), [])
        paper = PAPER_TABLE2_GCN_MA.get(ds, {"auc": float("nan"), "ap": float("nan")})
        if not recs:
            lines.append(f"| {ds} | 0 | — | — | {paper['auc']:.4f} | {paper['ap']:.4f} | — | — |")
            continue
        auc_m, auc_s = _mean_std([r["auc"] for r in recs])
        ap_m, ap_s = _mean_std([r["ap"] for r in recs])
        lines.append(
            f"| {ds} | {len(recs)} | {auc_m:.4f} ± {auc_s:.4f} | {ap_m:.4f} ± {ap_s:.4f} "
            f"| {paper['auc']:.4f} | {paper['ap']:.4f} | {auc_m - paper['auc']:+.4f} | {ap_m - paper['ap']:+.4f} |"
        )
    return "\n".join(lines) + "\n"


def _cross_model_table(by_md, models: list[str]) -> str:
    lines = [f"# Cross-Model Comparison — {', '.join(models)}\n"]
    header_cols = ["Dataset"]
    for m in models:
        header_cols.append(f"{m} AUC")
        header_cols.append(f"{m} AP")
    header_cols.append("Paper GCN_MA AUC")
    header_cols.append("Paper GCN_MA AP")
    lines.append("| " + " | ".join(header_cols) + " |")
    lines.append("|" + "|".join(["---"] * len(header_cols)) + "|")
    for ds in DATASETS:
        row = [ds]
        for m in models:
            recs = by_md.get((m, ds), [])
            auc_m, auc_s = _mean_std([r["auc"] for r in recs])
            ap_m, ap_s = _mean_std([r["ap"] for r in recs])
            if auc_m is None:
                row.extend(["—", "—"])
            else:
                row.append(f"{auc_m:.4f} ± {auc_s:.4f}")
                row.append(f"{ap_m:.4f} ± {ap_s:.4f}")
        paper = PAPER_TABLE2_GCN_MA.get(ds, {"auc": float("nan"), "ap": float("nan")})
        row.append(f"{paper['auc']:.4f}")
        row.append(f"{paper['ap']:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/metrics.jsonl", type=Path)
    parser.add_argument("--output", default="results/report/results_summary.md", type=Path)
    parser.add_argument("--model", default=None, help="single-model mode")
    parser.add_argument("--models", nargs="+", default=None, help="cross-model mode (≥2 names)")
    args = parser.parse_args()

    by_md = _load_records(REPO_ROOT / args.metrics)

    if args.models and len(args.models) >= 2:
        out = _cross_model_table(by_md, args.models)
    elif args.model:
        out = _single_model_table(by_md, args.model)
    else:
        parser.error("Specify --model NAME or --models NAME1 NAME2 [...]")

    print(out)
    (REPO_ROOT / args.output).parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / args.output).write_text(out)


if __name__ == "__main__":
    main()
