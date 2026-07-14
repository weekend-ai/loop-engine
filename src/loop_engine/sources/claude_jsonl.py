from __future__ import annotations

import glob
import hashlib
import json
from collections.abc import Iterable
from pathlib import Path

from loop_engine.models import CanonicalEvent, RawRecordEnvelope
from loop_engine.sources.claude_normalization import (
    ClaudeRecordNormalizer,
    RuleBasedClaudeRecordNormalizer,
    finalize_candidates,
)


class ClaudeCodeJsonlSource:
    def __init__(
        self,
        source_id: str,
        path_pattern: str,
        normalizer: ClaudeRecordNormalizer | None = None,
        *,
        max_record_bytes: int = 10 * 1024 * 1024,
        max_total_bytes: int = 100 * 1024 * 1024,
    ) -> None:
        if max_record_bytes < 1 or max_total_bytes < max_record_bytes:
            raise ValueError(
                "Claude JSONL byte limits require max_total_bytes >= "
                "max_record_bytes >= 1"
            )
        self.source_id = source_id
        self.path_pattern = path_pattern
        self.normalizer = normalizer or RuleBasedClaudeRecordNormalizer()
        self.max_record_bytes = max_record_bytes
        self.max_total_bytes = max_total_bytes

    def iter_envelopes(self) -> Iterable[RawRecordEnvelope]:
        total_bytes = 0
        for filename in sorted(glob.glob(self.path_pattern, recursive=True)):
            path = Path(filename)
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    line_bytes = len(line.encode("utf-8"))
                    if line_bytes > self.max_record_bytes:
                        raise ValueError(
                            f"Claude JSONL record size limit exceeded: {path}:{line_number}"
                        )
                    total_bytes += line_bytes
                    if total_bytes > self.max_total_bytes:
                        raise ValueError("Claude JSONL total byte limit exceeded")
                    raw_ref = f"file://{path.resolve()}#line={line_number}"
                    try:
                        raw = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Invalid Claude JSONL at {raw_ref}") from error
                    record_id = hashlib.sha256(raw_ref.encode()).hexdigest()[:24]
                    yield RawRecordEnvelope(
                        source_id=self.source_id,
                        record_id=record_id,
                        raw_ref=raw_ref,
                        line_number=line_number,
                        raw=raw,
                    )

    def iter_events(self) -> Iterable[CanonicalEvent]:
        envelopes = list(self.iter_envelopes())
        candidates = self.normalizer.normalize(envelopes)
        return finalize_candidates(envelopes, candidates)
