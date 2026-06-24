#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Aggregate eval/results/*.json into a survival-rate comparison.

Prints a table, writes summary.json + summary.md, and (if matplotlib is available)
a bar chart eval/results/survival_comparison.png. Stdlib-only otherwise.

  python eval/scripts/compare.py
"""

from __future__ import annotations

import argparse
import glob
import json
import os

_THIS = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start):
    d = start
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "baselines")):
            return d
        d = os.path.dirname(d)
    return start


REPO = _find_repo_root(_THIS)
RESULTS = os.path.join(REPO, "eval", "results")
SCHEME_ORDER = ["ward", "ragwm", "sentinel"]


def _load_results(results_dir):
    out = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        if os.path.basename(path) in ("summary.json",):
            continue
        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception as e:
            print(f"[warn] could not read {path}: {e}")
            continue
        if isinstance(data, dict) and "scheme" in data and "survival_rate" in data:
            out[data["scheme"]] = data
    return out


def _fmt_table(results):
    cols = ["scheme", "dataset", "victim_model", "n_injected",
            "n_detected_victim", "n_detected_surrogate", "survival_rate"]
    rows = [cols]
    ordered = [s for s in SCHEME_ORDER if s in results] + \
              [s for s in results if s not in SCHEME_ORDER]
    for s in ordered:
        r = results[s]
        rows.append([
            r.get("scheme", "?"),
            r.get("dataset", "?"),
            str(r.get("victim_model", "?")),
            str(r.get("n_injected", "?")),
            str(r.get("n_detected_victim", "?")),
            str(r.get("n_detected_surrogate", "?")),
            f"{r.get('survival_rate', 0):.3f}",
        ])
    widths = [max(len(row[i]) for row in rows) for i in range(len(cols))]
    lines = []
    for ri, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if ri == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(cols))))
    return "\n".join(lines), ordered


def _maybe_plot(results, ordered, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("[info] matplotlib not available; skipping chart.")
        return None
    schemes = ordered
    rates = [results[s]["survival_rate"] for s in schemes]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(schemes, rates, color=["#4C72B0", "#DD8452", "#55A868"][:len(schemes)])
    ax.set_ylabel("Watermark survival rate")
    ax.set_ylim(0, 1)
    ax.set_title("Signals surviving extraction (worst-case probe harvest)")
    for b, r in zip(bars, rates):
        ax.text(b.get_x() + b.get_width() / 2, r + 0.02, f"{r:.2f}", ha="center")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"[info] wrote chart -> {out_png}")
    return out_png


def main():
    ap = argparse.ArgumentParser(description="Compare extraction-attack survival across schemes")
    ap.add_argument("--results-dir", default=RESULTS)
    args = ap.parse_args()

    results = _load_results(args.results_dir)
    if not results:
        print(f"No scheme results found in {args.results_dir}. Run the extractors first.")
        return

    table, ordered = _fmt_table(results)
    print("\nExtraction-attack watermark survival\n")
    print(table)
    print()

    # summary.json
    summary = {s: {k: results[s].get(k) for k in
                   ("dataset", "victim_model", "n_injected",
                    "n_detected_victim", "n_detected_surrogate", "survival_rate")}
               for s in ordered}
    with open(os.path.join(args.results_dir, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    # summary.md
    md = ["# Extraction-attack watermark survival\n",
          "| scheme | dataset | victim model | N | detected (victim) | survived (surrogate) | survival rate |",
          "|---|---|---|---|---|---|---|"]
    for s in ordered:
        r = results[s]
        md.append(f"| {r['scheme']} | {r.get('dataset')} | {r.get('victim_model')} | "
                  f"{r.get('n_injected')} | {r.get('n_detected_victim')} | "
                  f"{r.get('n_detected_surrogate')} | {r.get('survival_rate'):.3f} |")
    with open(os.path.join(args.results_dir, "summary.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    _maybe_plot(results, ordered, os.path.join(args.results_dir, "survival_comparison.png"))
    print(f"[info] wrote summary.json + summary.md -> {args.results_dir}")


if __name__ == "__main__":
    main()
