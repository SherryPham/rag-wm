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
RESULTS = os.path.join(REPO, "evaluation", "results")
SCHEME_ORDER = ["ward", "ragwm", "sentinel"]

def _load_results(results_dir):
    out = {}
    for path in glob.glob(os.path.join(results_dir, "*.json")):
        if os.path.basename(path) == "summary.json":
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
            str(r.get("scheme", "?")), str(r.get("dataset", "?")),
            str(r.get("victim_model", "?")), str(r.get("n_injected", "?")),
            str(r.get("n_detected_victim", "?")), str(r.get("n_detected_surrogate", "?")),
            f"{r.get('survival_rate', 0):.3f}",
        ])
    widths = [max(len(row[i]) for row in rows) for i in range(len(cols))]
    lines = []
    for ri, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[i]) for i, cell in enumerate(row)))
        if ri == 0:
            lines.append("  ".join("-" * widths[i] for i in range(len(cols))))
    return "\n".join(lines), ordered

def main():
    ap = argparse.ArgumentParser(description="Compare extraction-attack survival across schemes")
    ap.add_argument("--results-dir", default=RESULTS)
    args = ap.parse_args()

    results = _load_results(args.results_dir)
    if not results:
        print(f"No scheme results found in {args.results_dir}. Run the extractors first.")
        return

    table, _ = _fmt_table(results)
    print("\nExtraction-attack watermark survival\n")
    print(table)
    print()

if __name__ == "__main__":
    main()
