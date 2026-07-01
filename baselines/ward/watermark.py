from __future__ import annotations

import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

_THIS = os.path.dirname(os.path.abspath(__file__))

def _repo_root(start):
    d = start
    while os.path.dirname(d) != d:
        if os.path.isdir(os.path.join(d, "baselines")):
            return d
        d = os.path.dirname(d)
    return start

_REPO = _repo_root(_THIS)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from baselines.ward.kgw import WatermarkDetector, WatermarkLogitsProcessor
from evaluation.scheme import Signal, WatermarkScheme
from evaluation.llm import OllamaLLM

PARAPHRASER_PROMPT = (
    "You are an expert rewriter. Rewrite the following document keeping its meaning and "
    "fluency and especially length. It is crucial to retain all factual information in the "
    "original document. DO NOT MAKE THE TEXT SHORTER. Do not start your response by 'Sure' "
    "or anything similar, simply output the paraphrased document directly. Do not add "
    "stylistic elements or anything similar, try to be faithful to the original content and "
    "style of writing. Do not be too formal. Keep all the factual information."
)
QUESTION_GEN_SYSTEM = (
    "Given a document, generate a question that can only be answered by reading the document. "
    "The answer should be a longer detailed response, so avoid factual and simple yes/no "
    "questions and steer more towards questions that ask for opinions or explanations of "
    "events or topics described in the documents. Do not provide the answer, provide just the "
    "question."
)

class WardWatermark(WatermarkScheme):

    name = "ward"
    needs_local_hf_model = True

    def __init__(self, hf_model: str = "Qwen/Qwen2.5-3B-Instruct",
                 gamma: float = 0.25, delta: float = 3.5,
                 seeding_scheme: str = "ff-position_prf-2-False-1548585",
                 z_threshold: float = 4.0, response_max_len: int = 512,
                 seed: int = 1, llm: OllamaLLM | None = None):
        self.gamma, self.delta, self.seeding_scheme = gamma, delta, seeding_scheme
        self.z_threshold, self.response_max_len, self.seed = z_threshold, response_max_len, seed
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.qgen = llm or OllamaLLM()

        self.tokenizer = AutoTokenizer.from_pretrained(hf_model)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            hf_model, torch_dtype=(torch.float16 if self.device == "cuda" else torch.float32),
        ).to(self.device).eval()
        self.vocab = list(self.tokenizer.get_vocab().values())

        self.detector = WatermarkDetector(
            vocab=self.vocab, gamma=self.gamma, seeding_scheme=self.seeding_scheme,
            device=self.device, tokenizer=self.tokenizer, normalizers=[],
            z_threshold=self.z_threshold, ignore_repeated_ngrams=True,
        )

    def _spawn_processor(self):
        return WatermarkLogitsProcessor(
            vocab=self.vocab, gamma=self.gamma, delta=self.delta,
            seeding_scheme=self.seeding_scheme, device=self.device, tokenizer=self.tokenizer,
        )

    @torch.no_grad()
    def _watermarked_rephrase(self, doc: str, processor) -> str:
        prompt = f"{PARAPHRASER_PROMPT}\nDOCUMENT:\n{doc}"
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        enc = self.tokenizer(text, return_tensors="pt").to(self.device)
        torch.manual_seed(self.seed)
        out = self.model.generate(
            **enc, max_new_tokens=self.response_max_len, num_beams=1,
            do_sample=True, temperature=0.7, pad_token_id=self.tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([processor]),
        )
        gen = out[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(gen, skip_special_tokens=True).strip()

    def build_watermarked_corpus(self, background, n):

        ids = sorted(background)[:n]
        processor = self._spawn_processor()
        corpus = dict(background)
        signals = []
        for idx, did in enumerate(ids):
            doc = background[did]
            docwm = self._watermarked_rephrase(doc, processor)
            question = self.qgen.ask(f"DOCUMENT:\n{doc}", system=QUESTION_GEN_SYSTEM).strip()
            if not docwm or not question:
                continue
            corpus[did] = docwm
            signals.append(Signal(id=idx, probe=question, meta={"source_doc_id": did}))
            print(f"[ward] embedded {idx + 1}/{len(ids)}")
        return corpus, signals

    def detect(self, signal: Signal, answer: str) -> bool:
        if not answer:
            return False
        try:
            result = self.detector.detect(answer)
        except Exception:
            return False
        return bool(result.get("z_score", 0.0) > self.z_threshold)
