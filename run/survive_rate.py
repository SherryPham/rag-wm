from __future__ import annotations

import argparse
import os
import subprocess
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))

def _repo_root(start):
    d = start
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "baselines")):
            return d
        d = os.path.dirname(d)
    return start

REPO = _repo_root(_THIS)

def _python():

    for p in (os.path.join(REPO, ".venv", "Scripts", "python.exe"),
              os.path.join(REPO, ".venv", "bin", "python")):
        if os.path.exists(p):
            return p
    return sys.executable

def main():
    ap = argparse.ArgumentParser(description="Run the extraction attack + comparison")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--schemes", nargs="+", default=["ward", "ragwm", "sentinel"],
                    choices=["ward", "ragwm", "sentinel"])
    ap.add_argument("--background", type=int, default=200)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--no-compare", action="store_true")
    args = ap.parse_args()

    py = _python()
    extract = os.path.join(REPO, "adversaries", "naive-extraction", "extract.py")
    cmd = [py, extract, "--n", str(args.n), "--background", str(args.background),
           "--top_k", str(args.top_k), "--schemes", *args.schemes]
    print(f">>> {' '.join(cmd)}", flush=True)
    rc = subprocess.run(cmd, cwd=REPO).returncode

    if not args.no_compare:
        print("\n=== comparison ===")
        subprocess.run([py, os.path.join(_THIS, "compare.py")], cwd=REPO)
    return rc

if __name__ == "__main__":
    raise SystemExit(main())
