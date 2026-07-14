from __future__ import annotations

from typing import Protocol

from loop_engine.models import CanonicalEvent, TaskRun, TaskSemanticAnalysis


class SemanticAnalyzer(Protocol):
    def analyze(self, task: TaskRun, events: list[CanonicalEvent]) -> TaskSemanticAnalysis: ...
