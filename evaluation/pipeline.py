from __future__ import annotations

import os

import numpy as np
import openai
import torch
from transformers import AutoModel, AutoTokenizer

DEFAULT_SYS_PROMPT = (
    "You are a helpful assistant. Answer the question using the provided context. "
    "Be specific and ground your answer in the context."
)

class Contriever:

    def __init__(self, name: str = "facebook/contriever", device: str | None = None,
                 max_length: int = 256):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        self.max_length = max_length

    @staticmethod
    def _mean_pool(last_hidden, mask):
        m = mask.unsqueeze(-1).float()
        return (last_hidden * m).sum(1) / m.sum(1).clamp(min=1e-9)

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [t if t else " " for t in texts[i:i + batch_size]]
            enc = self.tokenizer(batch, padding=True, truncation=True,
                                 max_length=self.max_length, return_tensors="pt").to(self.device)
            hidden = self.model(**enc).last_hidden_state
            emb = self._mean_pool(hidden, enc["attention_mask"])
            out.extend(emb.cpu().tolist())
        return out

class RagPipeline:

    def __init__(self, top_k: int = 5, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, embedder: Contriever | None = None,
                 sys_prompt: str = DEFAULT_SYS_PROMPT, temperature: float = 0.0):
        self.top_k = top_k
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
        base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
        self.client = openai.OpenAI(
            base_url=base_url, api_key=api_key or os.environ.get("OLLAMA_API_KEY", "ollama"))
        self.embedder = embedder or Contriever()
        self.sys_prompt = sys_prompt
        self.temperature = temperature
        self._docs: list[str] = []
        self._emb: np.ndarray | None = None

    @staticmethod
    def _l2_normalize(mat: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(mat, axis=-1, keepdims=True)
        return mat / np.clip(norms, 1e-9, None)

    def build(self, corpus: dict[str, str]) -> "RagPipeline":

        ids = list(corpus.keys())
        texts = [corpus[i] if corpus[i] else " " for i in ids]
        embs = np.asarray(self.embedder.encode(texts), dtype=np.float32)
        self._docs = texts
        self._emb = self._l2_normalize(embs) if len(texts) else None
        return self

    def retrieve(self, question: str, k: int | None = None) -> list[str]:
        k = k or self.top_k
        if self._emb is None:
            raise RuntimeError("RagPipeline.build() must be called before retrieve().")
        n = min(k, len(self._docs))
        if n <= 0:
            return []
        qe = np.asarray(self.embedder.encode([question])[0], dtype=np.float32)
        qe = self._l2_normalize(qe)
        sims = self._emb @ qe
        top = np.argsort(-sims)[:n]
        return [self._docs[i] for i in top]

    def answer(self, question: str, k: int | None = None) -> tuple[str, list[str]]:
        docs = self.retrieve(question, k=k)
        context = "\n\n".join(docs)
        messages = [
            {"role": "system", "content": self.sys_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"},
        ]
        try:
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=self.temperature)
            answer = resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[rag] generation error: {e}")
            answer = ""
        return answer, docs
