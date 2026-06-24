

from __future__ import annotations

import argparse
import asyncio
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
import schema


def _find_repo_root(start):
    d = start
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "baselines")):
            return d
        d = os.path.dirname(d)
    return start


# Make the sentinelrag package importable even if it was not `pip install -e .`'d
# into the venv (src layout fallback).
_SENT_SRC = os.path.join(_find_repo_root(_THIS_DIR), "baselines", "sentinel", "src")
if os.path.isdir(_SENT_SRC):
    sys.path.insert(0, _SENT_SRC)

import torch

from sentinelrag.utils import (  
    load_json,
    load_models,
    load_beir_datasets,
    load_dataset_for_watermark_generation,
    create_llm_client_from_preset,
    find_latest_injection_result,
)
from sentinelrag.rag import ( 
    VectorStore,
    check_collection_exists,
    check_and_clean_existing_watermarks,
)
from sentinelrag.rag.vectorstore import ChromadbPath  
from sentinelrag.core.detector import (  
    RAGWatermarkDetector,
    inject_watermarks,
    cleanup_watermarks,
    test_questions_on_database_async,
)
from sentinelrag.utils.paths import default_models_dir  


class _PrintLogger:
    """Minimal logger so we can reuse detector helpers without the full Log setup."""

    def info(self, msg):
        print(f"[INFO] {msg}")

    def warning(self, msg):
        print(f"[WARN] {msg}")

    def error(self, msg):
        print(f"[ERROR] {msg}")


def _count_detected(per_ko, alpha):
    """A KO 'signal' is detected when its binomial p_value <= alpha."""
    n = 0
    for r in per_ko:
        p = r.get("p_value")
        if p is not None and p <= alpha:
            n += 1
    return n


def _build_surrogate_vectorstore(model, tokenizer, get_emb, device, harvested_answers):
    """Fresh ChromaDB collection whose entire corpus = the harvested answers."""
    vs = VectorStore(
        model, tokenizer, get_emb,
        dataset={},                       # not used: we inject_direct instead of populate_vectors
        device=device,
        collection_name="surrogate_sentinel_extract",
        use_local=False,                  # delete+recreate for a clean surrogate each run
    )
    for ans in harvested_answers:
        vs.inject_direct(ans if ans else " ")
    return vs


async def _run(args):
    logger = _PrintLogger()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    alpha, p0 = args.alpha, args.p0
    models_dir = args.models_dir or str(default_models_dir())

    # --- locate injection_result.json ---
    inj_path = args.injection_result_path
    if inj_path is None:
        inj_path = find_latest_injection_result(args.basepath, args.eval_dataset, args.n)
    print(f"[sentinel] injection result: {inj_path}")
    injection_result = load_json(inj_path)

    selected_kos = injection_result["selected_kos"]
    watermark_texts = injection_result["watermark_texts"]
    all_questions = injection_result.get("all_questions", [])
    all_correct = injection_result.get("all_correct_answers", []) or [""] * len(all_questions)
    if not all_questions:
        raise ValueError("injection_result.json has no 'all_questions'; re-run inject-watermark.")

    # --- cap to N signals (questions_per_ko is assumed 1 for the comparison) ---
    qpk = args.questions_per_ko
    N = min(args.n, len(selected_kos))
    sampled_kos = selected_kos[:N]
    sampled_texts = watermark_texts[:N]
    sampled_questions = all_questions[: N * qpk]
    sampled_correct = all_correct[: N * qpk]
    print(f"[sentinel] using N={N} KOs, {len(sampled_questions)} probe questions")

    # --- embedding model + victim corpus (existing nfcorpus collection) ---
    collection_name = f"{args.eval_dataset}_{args.eval_model_code}_{args.score_function}"
    if not check_collection_exists(collection_name):
        raise ValueError(
            f"Collection '{collection_name}' missing. Run: "
            f"sentinelrag-build-chroma --eval_dataset {args.eval_dataset} "
            f"--eval_model_code {args.eval_model_code}"
        )
    model, _c, tokenizer, get_emb = load_models(args.eval_model_code)
    # vector store is keyed on the BEIR corpus (as in detect_watermark.setup_vectorstore);
    # full_dataset feeds the detector's verification.
    corpus, _q, _qr = load_beir_datasets(args.eval_dataset, args.split)
    _sampled_ds, full_dataset, _nd = load_dataset_for_watermark_generation(
        args.eval_dataset, args.split, 0)
    victim_vs = VectorStore(model, tokenizer, get_emb, corpus, device,
                            collection_name, use_local=True)
    check_and_clean_existing_watermarks(victim_vs, len(corpus), logger)

    # --- LLM clients (for victim RAG answers and verification judge) ---
    rllm_client, rllm_preset = create_llm_client_from_preset(args.rllm, models_dir=models_dir)
    dllm_client, dllm_preset = create_llm_client_from_preset(args.dllm, models_dir=models_dir)
    print(f"[sentinel] rllm={rllm_preset.model} dllm={dllm_preset.model}")

    # =========================== VICTIM ===========================
    watermark_ids = inject_watermarks(victim_vs, sampled_texts, logger)
    detector_v = RAGWatermarkDetector(
        full_dataset, victim_vs, dllm_client, rllm_client,
        top_k=args.top_k, questions_per_ko=qpk, dllm_client=dllm_client,
    )
    per_ko_v = await test_questions_on_database_async(
        detector_v, sampled_kos, sampled_questions, qpk, p0, logger,
        batch_size=args.batch_size, db_label="Victim",
        watermark_texts=sampled_texts, watermark_ids=watermark_ids,
        all_watermark_ids=watermark_ids, correct_answers=sampled_correct,
        verification_mode=args.verification_mode,
    )
    n_detected_victim = _count_detected(per_ko_v, alpha)

    # harvest victim answers (worst-case == same probe questions)
    qa_map = {}
    for info in detector_v.retrieved_documents:
        q = info.get("question")
        if q is not None:
            qa_map[q] = info.get("answer", "")
    harvested = [qa_map.get(q, "") for q in sampled_questions]

    cleanup_watermarks(victim_vs, watermark_ids, True, logger)

    # ========================= SURROGATE ==========================
    surrogate_vs = _build_surrogate_vectorstore(model, tokenizer, get_emb, device, harvested)
    detector_s = RAGWatermarkDetector(
        full_dataset, surrogate_vs, dllm_client, rllm_client,
        top_k=args.top_k, questions_per_ko=qpk, dllm_client=dllm_client,
    )
    per_ko_s = await test_questions_on_database_async(
        detector_s, sampled_kos, sampled_questions, qpk, p0, logger,
        batch_size=args.batch_size, db_label="Surrogate",
        watermark_texts=sampled_texts, watermark_ids=[],
        all_watermark_ids=[], correct_answers=sampled_correct,
        verification_mode=args.verification_mode,
    )
    n_detected_surrogate = _count_detected(per_ko_s, alpha)

    # a signal only "survives" if it was detected on the victim too
    per_signal = []
    for i in range(N):
        dv = (per_ko_v[i].get("p_value") is not None and per_ko_v[i]["p_value"] <= alpha)
        ds = (per_ko_s[i].get("p_value") is not None and per_ko_s[i]["p_value"] <= alpha)
        per_signal.append(schema.SignalResult(
            id=i, detected_victim=bool(dv), detected_surrogate=bool(dv and ds),
            detail={"p_victim": per_ko_v[i].get("p_value"),
                    "p_surrogate": per_ko_s[i].get("p_value")},
        ))
    n_detected_surrogate = sum(1 for s in per_signal if s.detected_surrogate)

    result = schema.SchemeResult(
        scheme="sentinel", dataset=args.eval_dataset,
        victim_model=rllm_preset.model, adversary_model=rllm_preset.model,
        n_injected=N, n_detected_victim=n_detected_victim,
        n_detected_surrogate=n_detected_surrogate,
        params={"alpha": alpha, "p0": p0, "top_k": args.top_k,
                "questions_per_ko": qpk, "eval_model_code": args.eval_model_code,
                "verification_mode": args.verification_mode,
                "attack": "extraction-worstcase", "chroma_path": ChromadbPath},
        per_signal=per_signal,
    )
    result.write(args.out)


def main():
    ap = argparse.ArgumentParser(description="SentinelRAG extraction-attack extractor")
    ap.add_argument("--n", type=int, default=50, help="Number of KO signals (default 50; smoke 5)")
    ap.add_argument("--injection-result-path", dest="injection_result_path", default=None)
    ap.add_argument("--out", default="eval/results/sentinel.json")
    ap.add_argument("--eval_dataset", default="nfcorpus")
    ap.add_argument("--eval_model_code", default="contriever",
                    choices=["contriever", "contriever-msmarco", "ance"])
    ap.add_argument("--score_function", default="cosine", choices=["cosine", "l2", "ip"])
    ap.add_argument("--split", default="test")
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--questions_per_ko", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.01)
    ap.add_argument("--p0", type=float, default=0.02)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--verification_mode", default="correct_answer_based",
                    choices=["ko_based", "correct_answer_based"])
    ap.add_argument("--rllm", default="qwen2.5", help="model preset for victim RAG answers")
    ap.add_argument("--dllm", default="qwen2.5", help="model preset for the verification judge")
    ap.add_argument("--basepath", default="output", help="root used to auto-find injection_result.json")
    ap.add_argument("--models_dir", default=None, help="override SentinelRAG models/ dir")
    args = ap.parse_args()
    asyncio.run(_run(args))


if __name__ == "__main__":
    main()
