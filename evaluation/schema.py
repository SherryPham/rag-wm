from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any

SCHEMES = ("ward", "ragwm", "sentinel")
SCHEMA_VERSION = 2

@dataclass
class SignalResult:
    id: int
    detected_victim: bool
    detected_surrogate: bool
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

@dataclass
class SchemeResult:

    scheme: str
    dataset: str
    victim_model: str
    adversary_model: str
    n_injected: int
    n_detected_victim: int
    n_detected_surrogate: int
    params: dict[str, Any] = field(default_factory=dict)
    per_signal: list[SignalResult] = field(default_factory=list)
    schema_version: int = SCHEMA_VERSION

    @property
    def success_rate_clean(self) -> float:

        return self.n_detected_victim / max(1, self.n_injected)

    @property
    def survival_rate(self) -> float:

        return self.n_detected_surrogate / max(1, self.n_detected_victim)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "dataset": self.dataset,
            "victim_model": self.victim_model,
            "adversary_model": self.adversary_model,
            "n_injected": self.n_injected,
            "n_detected_victim": self.n_detected_victim,
            "n_detected_surrogate": self.n_detected_surrogate,
            "success_rate_clean": self.success_rate_clean,
            "survival_rate": self.survival_rate,
            "params": self.params,
            "per_signal": [s.to_dict() for s in self.per_signal],
            "schema_version": self.schema_version,
        }

    def validate(self) -> None:
        if self.scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {self.scheme!r}")
        if self.n_detected_victim > self.n_injected:
            raise ValueError("n_detected_victim > n_injected")
        if self.n_detected_surrogate > self.n_detected_victim:
            raise ValueError("n_detected_surrogate > n_detected_victim")

    def write(self, path: str) -> str:
        self.validate()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        print(f"[schema] {self.scheme}: victim={self.n_detected_victim}/{self.n_injected} "
              f"surrogate={self.n_detected_surrogate} survival={self.survival_rate:.3f} -> {path}")
        return path
