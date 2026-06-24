# Watermark Survival Under Extraction Attack

Comparing how many **RAG-corpus watermark signals survive an extraction attack** across three
watermarking schemes — **WARD**, **RAG-WM**, and **SentinelRAG** — under one controlled, fully
local (free, no paid APIs) setup on Windows.

**The experiment.** For each scheme we inject a fixed number `N` of watermark signals, build the
victim RAG (local `qwen2.5`), then run a **worst-case extraction attack**: the adversary harvests
the victim's answers to the owner's probe questions, rebuilds a *surrogate* corpus + RAG from those
answers, and the owner re-runs the scheme's native detector on the surrogate. The headline metric is

```
survival_rate = (signals still detected on the surrogate) / (signals detected on the victim)
```

reported per scheme and compared side by side. The same victim/adversary/judge model (`qwen2.5`) is
used everywhere so the only variable is the watermarking scheme.

---

## Repository layout

```
baselines/
  ward/        WARD  — KGW token-level LLM watermark, RAG dataset inference (ICLR'25)
  ragwm/       RAG-WM — entity-graph "watermark units" injected into the corpus
  sentinel/    SentinelRAG — synthetic "sentinel" knowledge objects about fictitious entities
adversaries/
  naive-extraction/   
    schema.py            
    sentinel_extract.py  } 
    ragwm_extract.py     }
    ward_extract.py      }
  rag-crawler/                                  
```

Because the three repos have **incompatible dependencies** (RAG-WM `chromadb==0.5.20` /
`transformers<5` / `numpy 1.26`; SentinelRAG `numpy<2` + recent chromadb; WARD cu128 torch / py3.12),
each scheme runs in **its own virtualenv** and the only cross-scheme code is `eval/`, which exchanges
results through `adversaries/naive-extraction/schema.py`.

---

## The three baselines

- **WARD** — *Provable RAG Dataset Inference via LLM Watermarks* (ICLR 2025). A KGW **token-level**
  watermark is embedded by rephrasing corpus documents; detection is an aggregate z-score over the
  RAG's generated text. Here a "signal" = one watermarked doc with its probe question, detected when
  the answer's KGW `z > 4.0`. Dataset: FARAD (`baselines/ward/farad/`).
- **RAG-WM** — entity-graph **watermark units** `(entity1, entity2, relationship)` expanded into
  passages injected into the corpus; verified by asking about the relationship and LLM-judging the
  answer. A "signal" = one watermark unit (`Checker.check_wm == yes`). Dataset: BEIR nfcorpus.
- **SentinelRAG** — *Synthetic Sentinel Knowledge for RAG Database Copyright Protection*
  (arXiv:2606.05787). Isolated **sentinel KOs** about fictitious entities, invisible to normal
  queries but triggerable by owner probes; per-KO binomial detection. A "signal" = one KO
  (`p_value <= alpha`). Dataset: BEIR nfcorpus.

---

## Prerequisites

- **Ollama** running locally with qwen2.5 pulled (victim / adversary / judge for all three):
  ```powershell
  ollama pull qwen2.5:3b
  ```
- **WARD only:** a local Hugging Face `Qwen/Qwen2.5-3B-Instruct` (its KGW watermark *embedding*
  needs token-level logit access, which the Ollama API can't provide; the RAG answering still goes
  through Ollama via the added `qwen2.5-ollama` provider).
- `uv` (Python 3.12) for the single unified environment.

---

## Setup — one unified virtualenv at the repo root

All three schemes **and** the eval harness share **one** environment defined by the root
[`pyproject.toml`](pyproject.toml) (Python 3.12, cu128 torch, reconciled pins). From the repo root:

```powershell
uv venv
uv sync
```

This creates a single `.venv/` at the root, installs the reconciled union of all three baselines'
dependencies, and installs SentinelRAG **editable** so its `sentinelrag-*` console scripts work.
You no longer need per-scheme `baselines/*/.venv` folders.

> Reconciled pins (where the originals conflicted): `transformers` 4.4x (RAG-WM needs `<5`, WARD `>=4.36`),
> `chromadb==0.5.20` (RAG-WM hard pin), `numpy<2`. If `uv sync` reports a conflict, that's the place to look.

Model configs are already wired for local qwen2.5: SentinelRAG `baselines/sentinel/models/qwen2.5.json`,
RAG-WM `baselines/ragwm/model_configs/qwen_config.json`, WARD via the `qwen2.5-ollama` provider
(`OLLAMA_BASE_URL` / `OLLAMA_MODEL` env vars override the defaults).

---

## Prep — create the watermark signals (run once per scheme)

The extractors **consume** these artifacts; they do not generate them. Activate the unified env first
(`.\.venv\Scripts\Activate.ps1`), then run each scheme's prep from its own folder so outputs land where
the extractors look.

**SentinelRAG** (run from `baselines\sentinel`; console scripts come from the editable install):
```powershell
cd baselines\sentinel
sentinelrag-build-chroma     --eval_dataset nfcorpus --eval_model_code contriever --score_function cosine
sentinelrag-generate-ko-pool --eval_dataset nfcorpus --target_ko_count 50 --num_examples 10 --ko-generation-llm qwen2.5 --abstract-llm qwen2.5
sentinelrag-inject-watermark --ko_pool_path output\ko_pools\<preset>\<run>\ko_pool.json --secret_key mykey --eval_dataset nfcorpus --eval_model_code contriever --num_select_kos 50 --llm qwen2.5
cd ..\..
# -> baselines\sentinel\output\watermark_injections\nfcorpus\k50\<run>\injection_result.json
```

**RAG-WM** (from `baselines/ragwm`, unified env active). Pass an absolute Windows `--basepath`; the
entity/hash scripts expect `...\baselines\ragwm\output\wm_prepare` and `src/main.py` expects
`...\baselines\ragwm\output`:
```powershell
python rag\vectorstore.py                          --eval_dataset nfcorpus --eval_model_code contriever --score_function cosine
python entity_generate\generate_entity_llm_check.py --eval_dataset nfcorpus --dataset_prob 1
python entity_generate\generate_hash_entity.py      --eval_dataset nfcorpus -t scratch --entity_num 100 --edge_prob 0.05
python src\main.py --eval_dataset nfcorpus --eval_model_code contriever --score_function cosine --model_name_llm qwen --model_name_rllm qwen --doc 1 --inject 0 --verify 0 --stat 0 --mutual_times 10
python src\main.py --eval_dataset nfcorpus --eval_model_code contriever --score_function cosine --model_name_llm qwen --model_name_rllm qwen --doc 0 --inject 1 --verify 0 --stat 0 --mutual_times 10
# -> output\...\wmuint_inject.json
```

**WARD** — no separate prep: `ward_extract.py` embeds the KGW watermark into FARAD docs on the fly
(FARAD already ships in `baselines/ward/farad/`).

---

## Run the extraction-attack comparison

```powershell
# smoke test (small N), all three + comparison (uses the root .venv):
.\.venv\Scripts\python.exe eval\scripts\wm_survive_rate.py --n 5

# full run
.\.venv\Scripts\python.exe eval\scripts\wm_survive_rate.py --n 50

# run a single scheme directly:
.\.venv\Scripts\python.exe adversaries\naive-extraction\sentinel_extract.py --n 50 --out eval\results\sentinel.json
```

The orchestrator auto-finds RAG-WM's `wmuint_inject.json` and SentinelRAG's `injection_result.json`
(override with `--ragwm-wmunit` / `--sentinel-injection`), then runs `compare.py`, which writes
`eval/results/summary.md`, `summary.json`, and `survival_comparison.png`.

---

## Notes & caveats

- **WARD survival is a 2-hop token-level test** (victim generation → surrogate corpus → surrogate
  generation). Token-level watermarks degrade fast across regeneration, so low WARD survival is an
  expected, legitimate result. Per-answer z-scores can be noisy on short answers; the JSON also keeps
  per-signal z-scores.
- **SentinelRAG sentinels are probe-only** by design — under worst-case probe harvesting they can
  still fail to reconstruct a retrievable surrogate. That contrast is the point of the comparison.
- Determinism: fix seeds (`meta.seed`, SentinelRAG `--secret_key`, RAG-WM hash seed) so the victim
  and surrogate use the same `N` signals across reruns.

---

## Citations

WARD:
```bibtex
@inproceedings{jovanovic2025ward,
    author    = {Jovanović, Nikola and Staab, Robin and Baader, Maximilian and Vechev, Martin},
    title     = {Ward: Provable RAG Dataset Inference via LLM Watermarks},
    booktitle = {{ICLR}},
    year      = {2025}
}
```

SentinelRAG:
```bibtex
@misc{kwok2026sentinelrag,
    title         = {SentinelRAG: Synthetic Sentinel Knowledge for RAG Database Copyright Protection},
    author        = {Tsun On Kwok and Xi Yang and Ki Sen Hung and Chang Liu and Yangqiu Song},
    year          = {2026},
    eprint        = {2606.05787},
    archivePrefix = {arXiv},
    primaryClass  = {cs.CR},
    url           = {https://arxiv.org/abs/2606.05787}
}
```


