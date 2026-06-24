from __future__ import annotations

import argparse
import importlib
import os
import sys
import traceback

_THIS = os.path.dirname(os.path.abspath(__file__))

def _repo_root(start):
    d = start
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "baselines")):
            return d
        d = os.path.dirname(d)
    return start

_REPO = _repo_root(_THIS)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from run.pipeline import RagPipeline
from run.datasets import load_corpus, sample_background
from run.schema import SchemeResult, SignalResult

SCHEME_CLASSES = {
    "ward": ("baselines.ward.watermark", "WardWatermark"),
    "ragwm": ("baselines.ragwm.watermark", "RagwmWatermark"),
    "sentinel": ("baselines.sentinel.watermark", "SentinelWatermark"),
}

def _load_scheme(key):
    mod, cls = SCHEME_CLASSES[key]
    return getattr(importlib.import_module(mod), cls)

def run_scheme(key, background, n, top_k, model, dataset, results_dir):
    scheme = _load_scheme(key)()
    corpus, signals = scheme.build_watermarked_corpus(background, n)
    N = len(signals)
    if N == 0:
        print(f"[{key}] no signals were built; skipping")
        return None

    pipe = RagPipeline(top_k=top_k, model=model)

    pipe.build(corpus)
    victim_answers, victim_flags = [], []
    for i, s in enumerate(signals):
        ans, _ = pipe.answer(s.probe)
        victim_answers.append(ans)
        hit = bool(scheme.detect(s, ans))
        victim_flags.append(hit)
        print(f"[{key}] victim {i + 1}/{N} {'HIT' if hit else 'miss'}")
    n_victim = sum(victim_flags)

    surrogate_corpus = {f"h_{i}": (victim_answers[i] or " ") for i in range(N)}
    pipe.build(surrogate_corpus)
    per_signal = []
    for i, s in enumerate(signals):
        ans, _ = pipe.answer(s.probe)
        hit = bool(scheme.detect(s, ans))
        survived = bool(victim_flags[i] and hit)
        detail = s.meta if isinstance(s.meta, dict) else {}
        per_signal.append(SignalResult(id=i, detected_victim=victim_flags[i],
                                       detected_surrogate=survived, detail=detail))
        print(f"[{key}] surrogate {i + 1}/{N} {'HIT' if hit else 'miss'}")
    n_surrogate = sum(1 for x in per_signal if x.detected_surrogate)

    result = SchemeResult(
        scheme=key, dataset=dataset, victim_model=model, adversary_model=model,
        n_injected=N, n_detected_victim=n_victim, n_detected_surrogate=n_surrogate,
        params={"top_k": top_k, "attack": "naive-extraction",
                "shared_pipeline": "contriever+chromadb+ollama"},
        per_signal=per_signal,
    )
    result.write(os.path.join(results_dir, f"{key}.json"))
    return result

def main():
    ap = argparse.ArgumentParser(description="Watermark success rate across schemes under a shared adversary")
    ap.add_argument("--n", type=int, default=50, help="signals per scheme (smoke: 5)")
    ap.add_argument("--schemes", nargs="+", default=["ward", "ragwm", "sentinel"],
                    choices=["ward", "ragwm", "sentinel"])
    ap.add_argument("--background", type=int, default=200, help="shared clean background docs")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--dataset", default="nfcorpus")
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen2.5:3b"))
    ap.add_argument("--results-dir", default=os.path.join(_REPO, "results"))
    args = ap.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)
    base = load_corpus(args.dataset, "test")
    background = sample_background(base, args.background, seed=1)

    failures = []
    for key in args.schemes:
        print(f"\n===== {key} =====")
        try:
            if run_scheme(key, background, args.n, args.top_k, args.model,
                          args.dataset, args.results_dir) is None:
                failures.append(key)
        except Exception as e:
            traceback.print_exc()
            print(f"[{key}] FAILED: {e}")
            failures.append(key)
    if failures:
        print(f"\n[extract] schemes that failed or produced no signals: {failures}")
        sys.exit(1)

if __name__ == "__main__":
    main()
