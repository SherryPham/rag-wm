from __future__ import annotations

import json
import random

from run.llm import OllamaLLM
from run.scheme import Signal, WatermarkScheme

ABSTRACT_TO_KO = """
Please extract core, key entities and relationships from the following long text and organize them into a logically clear JSON object. The basic unit is (entity, relationship, entity), which is a triplet.
You can use the key names you think are most appropriate to describe this data.

Text content:
---
{text_document}
---
"""

GENERATE_FAKE_KOS = """
You are a data architect who excels at mimicking the structure and style of existing data to create new fictional data.

Below are some examples of real data abstracted into JSON Knowledge Objects (KO):
---
{examples_str}
---

Your task is to analyze these examples and create {num_to_generate} brand new, fictional knowledge objects.

**Analysis Guidelines:**
1. **Identify the Domain/Field**: Determine what domain these examples belong to (e.g., medical research, technology, finance, science, social science, etc.)
2. **Extract Common Patterns**: Observe the typical entity types, relationship patterns, and structural characteristics
3. **Note the Terminology Level**: Identify the level of technical/domain-specific terminology used

**Generation Requirements:**
- **Stay Within Domain**: Generate fictional KOs that belong to the SAME domain/field as the examples
- **Match Complexity**: Use similar levels of technical terminology and conceptual complexity
- **Maintain Structure**: Follow similar structural patterns (types of relationships, entity hierarchies)
- **Be Plausible**: Create fictional content that sounds realistic and could plausibly exist in the same domain
- **Full Fictional Details**: While staying in the same domain, specific entities, names, and numbers must be completely fictional

Please put all generated objects in a JSON array named "fake_kos".
"""

EXPAND_KO_TO_TEXT = """
  You are a professional writer who can perfectly mimic writing styles by learning from examples.

  **Your task is:**
  First, carefully study the multiple [Writing Examples] provided below, understanding their common tone, structure, and information density.
  Then, based on the given [Core Facts] (a JSON object), write a completely new paragraph that is fully consistent with the example style.

  {examples_str}

  **[Core Facts] (Your writing must strictly revolve around the following facts and not deviate):**
  ---
  {fake_ko_str}
  ---

  Please begin your writing. Output your written paragraph directly, without including any other explanations or titles.
"""

GENERATE_VERIFICATION_QUESTIONS = """
You are a Q&A test designer. Your task is to generate {num_questions} simple fact-based verification questions directly from the given [Watermark Text].

The questions closely match the wording and facts in the text.

**Main Goal**
- Each question should ask about a **single, explicit fact** stated in the watermark text.
- The answer must be found by **directly reading one sentence or phrase** from the text (no inference).

**Question Rules**
1. **Keep questions simple and literal**
   - Ask about one fact only (one relation, number, one name, one method, one claim).
   - Avoid rephrasing too creatively-stay close to the original wording.

2. **Use clear retrieval keywords**
   - Must include 2 exact keywords from the watermark text.
   - Prefer exact names, numbers, datasets, methods, or technical terms.
   - Do NOT add extra background or hypothetical context.

3. **Prefer surface-level facts**
   - Good targets:
     - Numbers (counts, years, sizes)
     - Names (models, methods, datasets, organizations)
     - Explicit statements ("X is used for Y")
     - Relations ("What is the relationship between A and B?")
   - Avoid:
     - "Why" or "How does this compare" questions
     - Implicit assumptions or reasoning
     - Yes / No questions

4. **Natural but straightforward language**
   - Questions should look like something a user would ask to confirm a fact.
   - Keep each question to **one short sentence**.
   - Do NOT use the term "watermark" or "text" in the questions.

**[Watermark Text]:**
---
{watermark_text}
---

**Steps to Follow**
1. Identify clear, explicit facts in the text.
2. Select simple keywords directly from those facts.
3. Write one short question per fact.
4. Ensure each question can be answered by directly quoting the watermark text.
5. Avoid the answer being directly in the question.

**Output Format**
Return a JSON object with a single field "questions", which is a list of objects. Each object has two fields: "question" (the question string) and "correct_answer" (the answer that can be directly found in the watermark text).

**Example**
{{
  "questions": [
    {{"question": "Which dataset was utilized for the training phase?", "correct_answer": "MSMARCO"}},
    {{"question": "How many parameters does the model have?", "correct_answer": "7 billion"}},
    {{"question": "What method is used for retrieval?", "correct_answer": "dense retrieval"}}
  ]
}}
"""

JUDGE_ANSWER_CORRECTNESS = """
You are a strict answer evaluator. Determine if the [RAG Answer] correctly answers the [Question] by comparing against the [Ground Truth].

**[Question]**: {question}
**[RAG Answer]**: {rag_answer}
**[Ground Truth]**: {ground_truth}

**Evaluation Criteria:**
1. The RAG answer must convey the same core information as the ground truth
2. Minor wording differences or additional context are acceptable
3. "Cannot answer", "I don't know", or irrelevant responses are incorrect
4. Partial matches that include the key answer are acceptable

Respond only with "yes" or "no".

- "yes": The RAG answer correctly captures the ground truth
- "no": The RAG answer is incorrect, irrelevant, or indicates inability to answer
"""

_REFUSALS = ("cannot answer", "i cannot answer", "cannot be answered")

class SentinelWatermark(WatermarkScheme):

    name = "sentinel"
    needs_local_hf_model = False

    def __init__(self, num_style_examples: int = 5, writing_examples: int = 3,
                 questions_per_ko: int = 1, llm: OllamaLLM | None = None, seed: int = 1):
        self.num_style_examples = num_style_examples
        self.writing_examples = writing_examples
        self.questions_per_ko = questions_per_ko
        self.llm = llm or OllamaLLM()
        self.seed = seed

    def _abstract_real_kos(self, docs):
        kos = []
        for d in docs:
            ko = self.llm.ask_json(ABSTRACT_TO_KO.format(text_document=d[:4000]))
            kos.append(ko if ko is not None else {"text": d[:500]})
        return kos

    def _fabricate_kos(self, examples_str, n):
        kos = []
        while len(kos) < n:
            batch = min(10, n - len(kos))
            res = self.llm.ask_json(
                GENERATE_FAKE_KOS.format(examples_str=examples_str, num_to_generate=batch),
                default={},
            )
            new = (res or {}).get("fake_kos") if isinstance(res, dict) else None
            if not new:
                break
            kos.extend(new)
        return kos[:n]

    def _expand(self, ko, writing_examples_str):
        ko_str = json.dumps(ko, indent=2, ensure_ascii=False)
        return self.llm.ask(EXPAND_KO_TO_TEXT.format(
            examples_str=writing_examples_str, fake_ko_str=ko_str)).strip()

    def _make_questions(self, passage):
        res = self.llm.ask_json(GENERATE_VERIFICATION_QUESTIONS.format(
            num_questions=self.questions_per_ko, watermark_text=passage), default={})
        qs = (res or {}).get("questions") if isinstance(res, dict) else None
        return qs or []

    def build_watermarked_corpus(self, background, n):
        rng = random.Random(self.seed)
        docs = list(background.values())
        rng.shuffle(docs)

        style_docs = docs[: self.num_style_examples]
        real_kos = self._abstract_real_kos(style_docs)
        examples_str = json.dumps(real_kos, indent=2, ensure_ascii=False)

        writing_docs = docs[: self.writing_examples]
        writing_examples_str = "".join(
            f"--- Writing Example {i + 1} ---\n{d}\n\n" for i, d in enumerate(writing_docs))

        kos = self._fabricate_kos(examples_str, n)
        corpus = dict(background)
        signals = []
        sid = 0
        for ko in kos:
            passage = self._expand(ko, writing_examples_str)
            if not passage:
                continue
            questions = self._make_questions(passage)
            if not questions:
                continue
            q = questions[0]
            question = (q.get("question") if isinstance(q, dict) else str(q)) or ""
            correct = (q.get("correct_answer") if isinstance(q, dict) else "") or ""
            if not question:
                continue
            corpus[f"wm_{sid}"] = passage
            signals.append(Signal(id=sid, probe=question,
                                  meta={"correct_answer": correct, "ko": ko}))
            sid += 1
            print(f"[sentinel] embedded {sid}/{n}")
        return corpus, signals

    def detect(self, signal: Signal, answer: str) -> bool:
        if not answer:
            return False
        low = answer.lower()
        if any(p in low for p in _REFUSALS):
            return False
        correct = signal.meta.get("correct_answer", "")
        reply = self.llm.ask(JUDGE_ANSWER_CORRECTNESS.format(
            question=signal.probe, rag_answer=answer, ground_truth=correct), temperature=0.0)
        return (reply or "").strip().lower().startswith("yes")
