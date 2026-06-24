

from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def _find_repo_root(start):
    """Ascend until we find the dir that contains 'baselines' (robust to nesting)."""
    d = start
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "baselines")):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.dirname(start))


_REPO_ROOT = _find_repo_root(_THIS_DIR)
sys.path.insert(0, _THIS_DIR)  # for schema
import schema  


def _detected(check_response: str) -> bool:
    """Mirror RAG-WM main.py: a unit is detected when Checker says yes."""
    norm = (check_response or "").strip().rstrip(".").lower()
    return norm == "yes" or norm.startswith("yes")


def _load_wmunits(path, n):
    """Accept either wmunit.json ([[E1,E2,R],...]) or wmuint_{doc,inject,verify}.json
    (where each entry's first element is the wmunit)."""
    import json
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    units = []
    for entry in data:
        if (isinstance(entry, list) and len(entry) == 3
                and all(isinstance(x, str) for x in entry)):
            units.append(entry)                # raw wmunit.json row
        elif isinstance(entry, list) and entry and isinstance(entry[0], list):
            units.append(entry[0])             # doc/inject/verify row -> wmunit is [0]
        else:
            raise ValueError(f"Unrecognized wmunit row shape: {entry!r}")
        if len(units) >= n:
            break
    return units


def main():
    ap = argparse.ArgumentParser(description="RAG-WM extraction-attack extractor")
    ap.add_argument("--n", type=int, default=50, help="Number of watermark-unit signals (smoke 5)")
    ap.add_argument("--wmunit-path", dest="wmunit_path", required=True,
                    help="Path to wmuint_inject.json (preferred) or wmunit.json")
    ap.add_argument("--out", default="eval/results/ragwm.json")
    ap.add_argument("--eval_dataset", default="nfcorpus")
    ap.add_argument("--eval_model_code", default="contriever",
                    choices=["contriever", "contriever-msmarco", "ance"])
    ap.add_argument("--score_function", default="cosine", choices=["cosine", "l2", "ip"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--ragwm-root", dest="ragwm_root",
                    default=os.path.join(_REPO_ROOT, "baselines", "ragwm"),
                    help="RAG-WM repo root (added to sys.path)")
    ap.add_argument("--model-config", dest="model_config", default="model_configs/qwen_config.json",
                    help="LLM config json (relative to ragwm root or absolute)")
    args = ap.parse_args()

    # make `from src...` / `from rag...` importable
    sys.path.insert(0, os.path.abspath(args.ragwm_root))

    import torch
    from src.utils import load_models, load_beir_datasets, load_json  # noqa: E402
    from src.models import create_model  # noqa: E402
    from src.watermark_role import Visiter, Checker  # noqa: E402
    from rag.vectorstore import VectorStore, ChromadbPath, check_collection  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_cfg_path = args.model_config
    if not os.path.isabs(model_cfg_path):
        model_cfg_path = os.path.join(os.path.abspath(args.ragwm_root), model_cfg_path)
    cfg = load_json(model_cfg_path)
    model_name = cfg.get("model_info", {}).get("name", "qwen")

    wmunits = _load_wmunits(args.wmunit_path, args.n)
    N = len(wmunits)
    print(f"[ragwm] using N={N} watermark units; model={model_name}")

    # --- embedding model + victim collection (existing, with injected WT) ---
    emb_model, _c, tokenizer, get_emb = load_models(args.eval_model_code)
    corpus, _q, _qr = load_beir_datasets(args.eval_dataset, args.split)
    collection_name = f"{args.eval_dataset}_{args.eval_model_code}_{args.score_function}"
    exists, _len = check_collection(collection_name)
    if not exists:
        raise ValueError(
            f"Collection '{collection_name}' missing. Build it + inject watermarks first "
            f"(rag/vectorstore.py, then main.py --doc --inject)."
        )
    victim_vs = VectorStore(emb_model, tokenizer, get_emb, corpus, device,
                            collection_name, use_local=True)

    # --- LLM clients (victim RAG answerer + verification judge = same qwen2.5) ---
    llm = create_model(model_cfg_path)      # judge / paraphrase role
    rllm = create_model(model_cfg_path)     # RAG answer role
    checker = Checker(llm)

    # =========================== VICTIM ===========================
    victim_visiter = Visiter(llm, rllm, victim_vs)
    victim_flags, harvested = [], []
    for i, wmunit in enumerate(wmunits):
        victim_visiter.wm_unit = wmunit
        answer, _db = victim_visiter.ask_wm()
        harvested.append(answer)
        checker.wm_unit = wmunit
        checker.rag_document = answer
        det = _detected(checker.check_wm())
        victim_flags.append(det)
        print(f"[ragwm] victim unit {i+1}/{N} {'DETECTED' if det else 'miss'}")
    n_detected_victim = sum(victim_flags)

    # ========================= SURROGATE ==========================
    surrogate_vs = VectorStore(emb_model, tokenizer, get_emb, {}, device,
                               "surrogate_ragwm_extract", use_local=False)
    for ans in harvested:
        surrogate_vs.inject_direct(ans if ans else " ")

    surrogate_visiter = Visiter(llm, rllm, surrogate_vs)
    per_signal = []
    for i, wmunit in enumerate(wmunits):
        surrogate_visiter.wm_unit = wmunit
        answer, _db = surrogate_visiter.ask_wm()
        checker.wm_unit = wmunit
        checker.rag_document = answer
        det_s = _detected(checker.check_wm())
        survived = bool(victim_flags[i] and det_s)
        per_signal.append(schema.SignalResult(
            id=i, detected_victim=victim_flags[i], detected_surrogate=survived,
            detail={"wmunit": wmunit},
        ))
        print(f"[ragwm] surrogate unit {i+1}/{N} {'DETECTED' if det_s else 'miss'}")
    n_detected_surrogate = sum(1 for s in per_signal if s.detected_surrogate)

    result = schema.SchemeResult(
        scheme="ragwm", dataset=args.eval_dataset,
        victim_model=model_name, adversary_model=model_name,
        n_injected=N, n_detected_victim=n_detected_victim,
        n_detected_surrogate=n_detected_surrogate,
        params={"eval_model_code": args.eval_model_code,
                "score_function": args.score_function,
                "attack": "extraction-worstcase", "chroma_path": ChromadbPath,
                "wmunit_path": os.path.abspath(args.wmunit_path)},
        per_signal=per_signal,
    )
    result.write(args.out)


if __name__ == "__main__":
    main()
