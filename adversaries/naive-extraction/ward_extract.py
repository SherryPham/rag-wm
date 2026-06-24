

from __future__ import annotations

import argparse
import json
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


def _load_farad(data_dir, n, article_model):
    """Return [(id, doc_text)] for the first n FARAD docs."""
    corpus = []
    i = 0
    while len(corpus) < n:
        path = os.path.join(data_dir, f"{i:04}.json")
        if not os.path.exists(path):
            break
        with open(path, "r", encoding="utf-8") as fh:
            item = json.load(fh)
        text = ""
        arts = item.get("articles", {})
        if article_model in arts and arts[article_model].get("article"):
            text = arts[article_model]["article"]
        else:
            text = item.get("original_doc", "")
        if text:
            corpus.append((f"wm_{i}", text))
        i += 1
    return corpus


def main():
    ap = argparse.ArgumentParser(description="WARD extraction-attack extractor")
    ap.add_argument("--n", type=int, default=50, help="Number of watermarked-doc signals (smoke 5)")
    ap.add_argument("--out", default="eval/results/ward.json")
    ap.add_argument("--ward-root", dest="ward_root",
                    default=os.path.join(_REPO_ROOT, "baselines", "ward"))
    ap.add_argument("--data-dir", dest="data_dir", default=None,
                    help="FARAD dir (default <ward-root>/farad)")
    ap.add_argument("--wm-model", dest="wm_model", default="Qwen/Qwen2.5-3B-Instruct",
                    help="HF model used to EMBED the KGW watermark (needs logit access)")
    ap.add_argument("--rag-model", dest="rag_model", default="qwen2.5-ollama",
                    help="APIModel name for the victim/surrogate RAG answerer (Ollama)")
    ap.add_argument("--article-model", dest="article_model", default="gpt4o",
                    help="Which FARAD article variant to watermark")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--n_shots", type=int, default=3)
    ap.add_argument("--seeding-scheme", dest="seeding_scheme",
                    default="ff-position_prf-2-False-1548585")
    ap.add_argument("--gamma", type=float, default=0.25)
    ap.add_argument("--delta", type=float, default=3.5)
    ap.add_argument("--z-threshold", dest="z_threshold", type=float, default=4.0)
    args = ap.parse_args()

    ward_root = os.path.abspath(args.ward_root)
    sys.path.insert(0, ward_root)
    os.chdir(ward_root)  # register_corpus writes cache/ relative to cwd
    data_dir = args.data_dir or os.path.join(ward_root, "farad")

   
    os.environ.setdefault("OPENAI_API_KEY", "sk-noop")

    import torch
    from src.config.ragwm_config import (  # noqa: E402
        MetaConfig, ModelConfig, AttackerConfig, RagConfig,
        WatermarkConfig, WatermarkGenerationConfig, WatermarkDetectionConfig,
        WatermarkScheme, AttackerAlgo,
    )
    from src.models import APIModel, HfModel  # noqa: E402
    from src.rag_system import RagSystem  # noqa: E402
    from src.attackers.watermark_attacker import WatermarkAttacker  # noqa: E402

    device = "cuda" if torch.cuda.is_available() else "cpu"

    meta = MetaConfig(device=device, seed=args.seed, out_root_dir="out/",
                      result_dir="results/extract")
    wm_cfg = WatermarkConfig(
        scheme=WatermarkScheme.KGW,
        generation=WatermarkGenerationConfig(
            seeding_scheme=args.seeding_scheme, gamma=args.gamma, delta=args.delta),
        detection=WatermarkDetectionConfig(
            normalizers=[], ignore_repeated_ngrams=True, z_threshold=args.z_threshold),
    )
    wm_model_cfg = ModelConfig(
        skip=False, name=args.wm_model, use_fp16=(device == "cuda"), use_flashattn2=False,
        prompt_max_len=8192, response_max_len=2048, n_beams=1,
        use_sampling=True, sampling_temp=0.7,
    )
    attacker_cfg = AttackerConfig(
        algos=[AttackerAlgo.WATERMARK], model=wm_model_cfg, watermark=wm_cfg,
        give_explicit_ids=True, queries_per_doc=1, dropout_percent=0,
    )
    rag_cfg = RagConfig(
        model=ModelConfig(skip=True, name=args.rag_model),
        embedder_name="sentence-transformers/all-MiniLM-L6-v2",
        real_retrieval=False, n_shots=args.n_shots, defended=True, memfree_n=-1,
    )

    # --- watermark-embedding model + attacker ---
    aux_model = HfModel(meta, wm_model_cfg)            
    rag_model = APIModel(args.rag_model)               
    attacker = WatermarkAttacker(meta, attacker_cfg)
    attacker.set_model(aux_model)                      # sets attacker.watermark + attacker.gpt
    attacker.gpt = APIModel(args.rag_model)            

    corpus = _load_farad(data_dir, args.n, args.article_model)
    if not corpus:
        raise ValueError(f"No FARAD docs found under {data_dir}")
    N = len(corpus)
    print(f"[ward] embedding KGW into {N} docs; wm_model={args.wm_model} rag={rag_model.full_model_name}")
    attacker.register_corpus(corpus)                   # embed watermark + generate 1 probe/doc

    ids = list(attacker.data.keys())[:N]
    questions = [attacker.data[idd]["question"] for idd in ids]
    watermarked_corpus = [(idd, attacker.data[idd]["docwm"]) for idd in ids]

    def detected(answer):
        res = attacker.watermark.detect([answer])
        r = res[0] if res else None
        if r is None:
            return False, None
        return (r["z_score"] > args.z_threshold), r["z_score"]

    # =========================== VICTIM ===========================
    victim_rag = RagSystem(meta, rag_cfg, watermarked_corpus, aux_model=rag_model)
    victim_answers = victim_rag.generate(questions, explicit_ids=ids)
    victim_flags, victim_z = [], []
    for ans in victim_answers:
        d, z = detected(ans)
        victim_flags.append(d)
        victim_z.append(z)
    n_detected_victim = sum(victim_flags)
    print(f"[ward] victim detected {n_detected_victim}/{N}")

    # ========================= SURROGATE ==========================
    surrogate_ids = [f"surr_{i}" for i in range(N)]
    surrogate_corpus = [(surrogate_ids[i], victim_answers[i] or " ") for i in range(N)]
    surrogate_rag = RagSystem(meta, rag_cfg, surrogate_corpus, aux_model=rag_model)
    surrogate_answers = surrogate_rag.generate(questions, explicit_ids=surrogate_ids)

    per_signal = []
    for i in range(N):
        ds, zs = detected(surrogate_answers[i])
        survived = bool(victim_flags[i] and ds)
        per_signal.append(schema.SignalResult(
            id=i, detected_victim=victim_flags[i], detected_surrogate=survived,
            detail={"z_victim": victim_z[i], "z_surrogate": zs, "doc_id": ids[i]},
        ))
    n_detected_surrogate = sum(1 for s in per_signal if s.detected_surrogate)
    print(f"[ward] surrogate survived {n_detected_surrogate}/{n_detected_victim}")

    result = schema.SchemeResult(
        scheme="ward", dataset="farad",
        victim_model=rag_model.full_model_name, adversary_model=rag_model.full_model_name,
        n_injected=N, n_detected_victim=n_detected_victim,
        n_detected_surrogate=n_detected_surrogate,
        params={"watermark_embed_model": args.wm_model, "seeding_scheme": args.seeding_scheme,
                "gamma": args.gamma, "delta": args.delta, "z_threshold": args.z_threshold,
                "n_shots": args.n_shots, "article_model": args.article_model,
                "signal_def": "per-probe KGW z>z_threshold", "real_retrieval": False,
                "attack": "extraction-worstcase"},
        per_signal=per_signal,
    )
    # write relative to original cwd if --out is relative (we chdir'd into ward_root)
    out = args.out if os.path.isabs(args.out) else os.path.join(_REPO_ROOT, args.out)
    result.write(out)


if __name__ == "__main__":
    main()
