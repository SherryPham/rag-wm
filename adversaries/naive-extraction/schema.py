

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any

SCHEMES = ("ward", "ragwm", "sentinel")

SCHEMA_VERSION = 1


@dataclass
class SignalResult:
    """Per-signal outcome (one watermark unit / KO / watermarked-doc probe)."""

    id: int
    detected_victim: bool
    detected_surrogate: bool
    # Free-form, scheme-specific detail (p_value, z_score, relationship triplet, ...).
    detail: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class SchemeResult:
    """Standardized output for one watermarking scheme under the extraction attack."""

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
    def survival_rate(self) -> float:
        """Fraction of victim-detectable signals that still survive on the surrogate."""
        denom = max(1, self.n_detected_victim)
        return self.n_detected_surrogate / denom

    def to_dict(self) -> dict[str, Any]:
        return {
            "scheme": self.scheme,
            "dataset": self.dataset,
            "victim_model": self.victim_model,
            "adversary_model": self.adversary_model,
            "n_injected": self.n_injected,
            "n_detected_victim": self.n_detected_victim,
            "n_detected_surrogate": self.n_detected_surrogate,
            "survival_rate": self.survival_rate,
            "params": self.params,
            "per_signal": [s.to_dict() for s in self.per_signal],
            "schema_version": self.schema_version,
        }

    def validate(self) -> None:
        """Fail fast on an obviously malformed result before it reaches compare.py."""
        if self.scheme not in SCHEMES:
            raise ValueError(f"scheme must be one of {SCHEMES}, got {self.scheme!r}")
        for name in ("n_injected", "n_detected_victim", "n_detected_surrogate"):
            val = getattr(self, name)
            if not isinstance(val, int) or val < 0:
                raise ValueError(f"{name} must be a non-negative int, got {val!r}")
        if self.n_detected_victim > self.n_injected:
            raise ValueError(
                f"n_detected_victim ({self.n_detected_victim}) > n_injected ({self.n_injected})"
            )
        if self.n_detected_surrogate > self.n_detected_victim:
            # A signal can only survive if it was present on the victim to begin with.
            raise ValueError(
                f"n_detected_surrogate ({self.n_detected_surrogate}) > "
                f"n_detected_victim ({self.n_detected_victim})"
            )
        if not (0.0 <= self.survival_rate <= 1.0):
            raise ValueError(f"survival_rate out of [0,1]: {self.survival_rate}")

    def write(self, path: str) -> str:
        """Validate and atomically write this result to `path`. Returns the path."""
        self.validate()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        tmp = f"{path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
        print(f"[schema] wrote {self.scheme} result -> {path} "
              f"(victim={self.n_detected_victim}/{self.n_injected}, "
              f"surrogate={self.n_detected_surrogate}, survival={self.survival_rate:.3f})")
        return path


def load_result(path: str) -> dict[str, Any]:
    """Read a result JSON written by an extractor (used by compare.py)."""
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)
