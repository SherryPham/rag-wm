from __future__ import annotations

import json
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

def _download_beir(name: str, base: str) -> str:

    import urllib.request
    import zipfile

    os.makedirs(base, exist_ok=True)
    url = _BEIR_URL.format(name)
    zip_path = os.path.join(base, f"{name}.zip")
    print(f"[datasets] downloading {url}")
    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(base)
    return os.path.join(base, name)

def load_corpus(name: str = "nfcorpus", split: str = "test",
                datasets_dir: str | None = None) -> dict[str, str]:

    # `split` only selects the qrels/queries subset in BEIR; the corpus is the
    # same file regardless, and this harness uses only the corpus text.
    base = datasets_dir or DATASETS_DIR
    data_path = os.path.join(base, name)
    if not os.path.isdir(data_path):
        data_path = _download_beir(name, base)

    corpus: dict[str, str] = {}
    with open(os.path.join(data_path, "corpus.jsonl"), "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            doc = json.loads(line)
            corpus[doc["_id"]] = doc.get("text") or ""
    return corpus

def sample_background(corpus: dict[str, str], n: int, seed: int = 1,
                      exclude: set | None = None) -> dict[str, str]:

    exclude = exclude or set()
    ids = sorted(k for k in corpus if k not in exclude)
    rng = random.Random(seed)
    rng.shuffle(ids)
    return {i: corpus[i] for i in ids[:n]}
