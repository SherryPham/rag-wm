from __future__ import annotations

import json
import os
import re

import openai

_JSON_BLOCK = re.compile(r"\[.*\]|\{.*\}", re.DOTALL)

class OllamaLLM:

    def __init__(self, model: str | None = None, base_url: str | None = None,
                 api_key: str | None = None, temperature: float = 0.0):
        self.model = model or os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")
        self.client = openai.OpenAI(
            base_url=base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=api_key or os.environ.get("OLLAMA_API_KEY", "ollama"),
        )
        self.temperature = temperature

    def ask(self, prompt: str, system: str = "You are a helpful assistant.",
            temperature: float | None = None) -> str:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": prompt}],
                temperature=self.temperature if temperature is None else temperature,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            print(f"[llm] error: {e}")
            return ""

    def ask_json(self, prompt: str, system: str = "You are a helpful assistant. Reply with JSON only.",
                 default=None):

        raw = self.ask(prompt, system=system)
        for candidate in (raw, (_JSON_BLOCK.search(raw).group(0) if _JSON_BLOCK.search(raw) else "")):
            try:
                return json.loads(candidate)
            except Exception:
                continue
        return default
