from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

@dataclass
class Signal:

    id: int
    probe: str
    meta: dict = field(default_factory=dict)

class WatermarkScheme(ABC):

    name: str = "scheme"

    needs_local_hf_model: bool = False

    @abstractmethod
    def build_watermarked_corpus(
        self, base_corpus: dict[str, str], n: int
    ) -> tuple[dict[str, str], list[Signal]]:

        raise NotImplementedError

    @abstractmethod
    def detect(self, signal: Signal, answer: str) -> bool:

        raise NotImplementedError
