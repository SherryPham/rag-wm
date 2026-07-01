# Watermark Survival Under Extraction Attack

Comparing RAG corpus watermark signals survive rate against an extraction attack across three
watermarking schemes — **WARD**, **RAG-WM**, and **SentinelRAG**.


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



## Running the evaluation

**Setup — Windows **
```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull qwen2.5:3b            # LLM: victim / adversary / judge
```

**Setup — macOS / Linux **
```bash
python3 -m venv .venv            
source .venv/bin/activate
pip install -r requirements.txt
ollama pull qwen2.5:3b            
```

**Running watermark survival rate evaluation** 
```bash
python evaluation/survive_rate.py --n 50                     
```
`--n` = watermark signals per scheme. Results are written to
`evaluation/results/{ward,ragwm,sentinel}.json` 



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

RAG-WM:
```bibtex
@inproceedings{lv2025ragwm,
    author    = {Lv, Peizhuo and Sun, Mengjie and Wang, Hao and Wang, Xiaofeng and Zhang, Shengzhi and Chen, Yuxuan and Chen, Kai and Sun, Limin},
    title     = {RAG-WM: An Efficient Black-Box Watermarking Approach for Retrieval-Augmented Generation of Large Language Models},
    booktitle = {Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security (CCS '25)},
    year      = {2025},
    doi       = {10.1145/3719027.3744813},
    url       = {https://dl.acm.org/doi/epdf/10.1145/3719027.3744813}
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


