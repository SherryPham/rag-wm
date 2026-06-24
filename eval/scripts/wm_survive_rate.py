#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Orchestrate the extraction-attack comparison across all three baselines.

Runs each scheme's extractor IN ITS OWN VENV via subprocess (the three repos have
incompatible deps), collecting standardized JSON into eval/results/, then runs compare.py.

Run with ANY python (stdlib only); it dispatches to the per-scheme venvs itself:
  python eval/scripts/run_all.py --n 5
  python eval/scripts/run_all.py --n 50 --schemes sentinel ragwm ward
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start):
    d = start
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "baselines")):
            return d
        d = os.path.dirname(d)
    return start


REPO = _find_repo_root(_THIS)
ADV = os.path.join(REPO, "adversaries", "naive-extraction")
RESULTS = os.path.join(REPO, "eval", "results")


def venv_python(scheme):
    """Interpreter to run a scheme's extractor.

    Uses the single unified env at the repo root (.venv); falls back to a per-scheme
    venv (baselines/<scheme>/.venv) if the root one isn't present yet.
    """
    for base in (REPO, os.path.join(REPO, "baselines", scheme)):
        win = os.path.join(base, ".venv", "Scripts", "python.exe")
        posix = os.path.join(base, ".venv", "bin", "python")
        if os.path.exists(win):
            return win
        if os.path.exists(posix):
            return posix
    # default to the root layout (clear error if missing)
    return os.path.join(REPO, ".venv", "Scripts", "python.exe")


def _newest(pattern):
    files = glob.glob(pattern, recursive=True)
    return max(files, key=os.path.getmtime) if files else None


def _run(cmd, cwd):
    print(f"\n>>> ({os.path.basename(cwd)}) {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=cwd).returncode


def main():
    ap = argparse.ArgumentParser(description="Run extraction-attack comparison across schemes")
    ap.add_argument("--n", type=int, default=50, help="Fixed number of signals per scheme (smoke 5)")
    ap.add_argument("--schemes", nargs="+", default=["sentinel", "ragwm", "ward"],
                    choices=["sentinel", "ragwm", "ward"])
    ap.add_argument("--ragwm-wmunit", default=None,
                    help="Path to wmuint_inject.json (else newest under baselines/ragwm/output)")
    ap.add_argument("--sentinel-injection", default=None,
                    help="Path to injection_result.json (else auto-found by --n + dataset)")
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    status = {}

    for s in args.schemes:
        py = venv_python(s)
        script = os.path.join(ADV, f"{s}_extract.py")
        out = os.path.join(RESULTS, f"{s}.json")
        if not os.path.exists(py):
            print(f"[skip] {s}: venv python not found at {py} (create it first)")
            status[s] = "no-venv"
            continue
        if not os.path.exists(script):
            print(f"[skip] {s}: extractor not found at {script}")
            status[s] = "no-script"
            continue

        if s == "sentinel":
            cwd = os.path.join(REPO, "baselines", "sentinel")
            cmd = [py, script, "--n", str(args.n), "--out", out]
            if args.sentinel_injection:
                cmd += ["--injection-result-path", args.sentinel_injection]
        elif s == "ragwm":
            cwd = os.path.join(REPO, "baselines", "ragwm")
            wm = args.ragwm_wmunit or _newest(os.path.join(cwd, "output", "**", "wmuint_inject.json"))
            if not wm:
                print(f"[skip] ragwm: no wmuint_inject.json found under {cwd}\\output; run prep "
                      f"(rag/vectorstore.py -> entity_generate/* -> main.py --doc --inject)")
                status[s] = "no-prep"
                continue
            cmd = [py, script, "--n", str(args.n), "--out", out, "--wmunit-path", wm]
        else:  # ward
            cwd = os.path.join(REPO, "baselines", "ward")
            cmd = [py, script, "--n", str(args.n), "--out", out]

        status[s] = _run(cmd, cwd)

    print("\n=== extractor exit status ===")
    for s, v in status.items():
        print(f"  {s}: {v}")

    if not args.no_compare:
        compare = os.path.join(_THIS, "compare.py")
        print("\n=== comparison ===")
        subprocess.run([sys.executable, compare])


if __name__ == "__main__":
    main()
