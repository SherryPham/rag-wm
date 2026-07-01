from __future__ import annotations

import json
import re

from evaluation.llm import OllamaLLM
from evaluation.scheme import Signal, WatermarkScheme

WATERMARK_GENERATE = """
You are a knowledge graph expert and a linguist. Your task is to generate {K} distinct natural language sentences that describe a given relationship (R1) between two entities (E1 and E2) in a knowledge graph. Each sentence should convey the relationship between (E1) and (E2) using a unique syntactic structure to ensure variation, while maintaining clarity and readability.

Input:
E1: {E1}
E2: {E2}
R1: {R1}

Output:
Return exactly {K} distinct sentences, formatted as a JSON list, like this:
[
    "Sentence 1.",
    "Sentence 2.",
    "Sentence 3.",
    "Sentence 4.",
    ...,
    "Sentence K."
]
"""

WATERMARK_ASK_RAG = "What is the relationship between {E1} and {E2} ?"

WATERMARK_CHECK = """
You are an expert in knowledge graphs and linguistics. Your task is to evaluate the text: (rag_doc). Identify whether it suggests a relationship (R1) exists between the entities (E1) and (E2).

Input:
- rag_doc: ({rag_doc})
- R1: ({R1})
- E1: ({E1})
- E2: ({E2})

Output:
Reply strictly with "yes" if the relationship is implied, or "no" if it is not. No additional information is required.
"""

_JSON_LIST = re.compile(r"\[.*\]", re.DOTALL)

def _extract_json_list(text: str):
    m = _JSON_LIST.search(text or "")
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None

class RagwmWatermark(WatermarkScheme):

    name = "ragwm"
    needs_local_hf_model = False

    def __init__(self, k: int = 5, domain: str = "biomedical and nutrition research",
                 llm: OllamaLLM | None = None):
        self.k = k
        self.domain = domain
        self.llm = llm or OllamaLLM()

    def _generate_triplets(self, n: int):

        prompt = (
            f"Generate {n} distinct knowledge-graph triplets in the {self.domain} domain. "
            f"Each triplet is two specific named entities and a relationship type between them. "
            f"Prefer specific, less-common entity pairs so the relationship is unlikely to already "
            f"be widely stated. Return ONLY a JSON list of objects like "
            f'[{{"e1": "...", "e2": "...", "r": "..."}}].'
        )
        triplets = []
        for _ in range(4):
            data = _extract_json_list(self.llm.ask(prompt, temperature=0.8)) or []
            for d in data:
                if isinstance(d, dict) and d.get("e1") and d.get("e2") and d.get("r"):
                    triplets.append((str(d["e1"]), str(d["e2"]), str(d["r"])))
                if len(triplets) >= n:
                    return triplets[:n]
        return triplets[:n]

    def _expand_unit(self, e1: str, e2: str, r1: str) -> str:

        prompt = WATERMARK_GENERATE.format(K=self.k, E1=e1, E2=e2, R1=r1)
        for _ in range(4):
            sents = _extract_json_list(self.llm.ask(prompt, temperature=0.7))
            if isinstance(sents, list) and sents:
                return " ".join(str(s) for s in sents)
        return f"{e1} {r1} {e2}."

    def build_watermarked_corpus(self, background, n):
        triplets = self._generate_triplets(n)
        corpus = dict(background)
        signals = []
        for i, (e1, e2, r1) in enumerate(triplets):
            corpus[f"wm_{i}"] = self._expand_unit(e1, e2, r1)
            signals.append(Signal(
                id=i, probe=WATERMARK_ASK_RAG.format(E1=e1, E2=e2),
                meta={"e1": e1, "e2": e2, "r1": r1},
            ))
            print(f"[ragwm] embedded {i + 1}/{len(triplets)}")
        return corpus, signals

    def detect(self, signal: Signal, answer: str) -> bool:
        if not answer:
            return False
        m = signal.meta
        prompt = WATERMARK_CHECK.format(rag_doc=answer, R1=m["r1"], E1=m["e1"], E2=m["e2"])
        reply = (self.llm.ask(prompt, temperature=0.0) or "").strip().rstrip(".").lower()
        return reply.startswith("yes")
