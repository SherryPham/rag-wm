from __future__ import annotations

import os
import random

_THIS = os.path.dirname(os.path.abspath(__file__))

def _find_repo_root(start):
    d = start
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "baselines")):
            return d
        d = os.path.dirname(d)
    return start

REPO = _find_repo_root(_THIS)
DATASETS_DIR = os.environ.get("WM_DATASETS_DIR", os.path.join(REPO, "datasets"))
_BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip"

def load_corpus(name: str = "nfcorpus", split: str = "test",
                datasets_dir: str | None = None) -> dict[str, str]:

    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    base = datasets_dir or DATASETS_DIR
    os.makedirs(base, exist_ok=True)
    data_path = os.path.join(base, name)
    if not os.path.isdir(data_path):
        data_path = util.download_and_unzip(_BEIR_URL.format(name), base)
    corpus, _queries, _qrels = GenericDataLoader(data_path).load(split=split)
    return {k: (v.get("text") or "") for k, v in corpus.items()}

def sample_background(corpus: dict[str, str], n: int, seed: int = 1,
                      exclude: set | None = None) -> dict[str, str]:

    exclude = exclude or set()
    ids = sorted(k for k in corpus if k not in exclude)
    rng = random.Random(seed)
    rng.shuffle(ids)
    return {i: corpus[i] for i in ids[:n]}
