# Watermark Survival Under Extraction Attack

Comparing how many **RAG-corpus watermark signals survive an extraction attack** across three
watermarking schemes — **WARD**, **RAG-WM**, and **SentinelRAG**.


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


