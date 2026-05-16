"""Aggregate GCN_MA results across seeds and print a Markdown table.

Reads results/metrics.jsonl (one record per run) and emits:
  - results/report/gcn_ma_summary.md (Markdown table)
  - stdout (same table)
"""
import argparse
import json
import statistics
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

PAPER_TABLE2 = {
    "collegemsg":   {"auc": 0.9149, "ap": 0.8926},
    "bitcoinotc":   {"auc": 0.9120, "ap": 0.8943},
    "eut":          {"auc": 0.9222, "ap": 0.9082},
    "mooc_actions": {"auc": 0.9880, "ap": 0.9863},
    "lastfm":       {"auc": 0.8757, "ap": 0.8704},
    "wikipedia":    {"auc": 0.8742, "ap": 0.8575},
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default="results/metrics.jsonl", type=Path)
    parser.add_argument("--output", default="results/report/gcn_ma_summary.md", type=Path)
    parser.add_argument("--model", default="gcn_ma")
    args = parser.parse_args()

    by_dataset: dict[str, list[dict]] = {}
    with (REPO_ROOT / args.metrics).open() as f:
        for line in f:
            r = json.loads(line)
            if r.get("model") != args.model:
                continue
            by_dataset.setdefault(r["dataset"], []).append(r)

    lines = []
    lines.append("# GCN_MA Reproduction — Per-Dataset Summary\n")
    lines.append("| Dataset | n seeds | AUC (mean ± std) | AP (mean ± std) | Paper AUC | Paper AP | Δ AUC | Δ AP |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for ds in ["collegemsg", "bitcoinotc", "eut", "mooc_actions", "lastfm", "wikipedia"]:
        recs = by_dataset.get(ds, [])
        if not recs:
            lines.append(f"| {ds} | 0 | — | — | {PAPER_TABLE2[ds]['auc']:.4f} | {PAPER_TABLE2[ds]['ap']:.4f} | — | — |")
            continue
        aucs = [r["auc"] for r in recs]
        aps = [r["ap"] for r in recs]
        auc_mean, ap_mean = statistics.mean(aucs), statistics.mean(aps)
        auc_std = statistics.stdev(aucs) if len(aucs) > 1 else 0.0
        ap_std = statistics.stdev(aps) if len(aps) > 1 else 0.0
        paper_auc = PAPER_TABLE2[ds]["auc"]
        paper_ap = PAPER_TABLE2[ds]["ap"]
        lines.append(
            f"| {ds} | {len(recs)} | {auc_mean:.4f} ± {auc_std:.4f} | {ap_mean:.4f} ± {ap_std:.4f} "
            f"| {paper_auc:.4f} | {paper_ap:.4f} | {auc_mean - paper_auc:+.4f} | {ap_mean - paper_ap:+.4f} |"
        )

    out = "\n".join(lines) + "\n"
    print(out)
    (REPO_ROOT / args.output).parent.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / args.output).write_text(out)


if __name__ == "__main__":
    main()
