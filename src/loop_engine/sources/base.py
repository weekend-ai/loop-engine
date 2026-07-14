from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from loop_engine.models import CanonicalEvent


class EventSource(Protocol):
    source_id: str

    def iter_events(self) -> Iterable[CanonicalEvent]: ...
