from __future__ import annotations

import glob
import hashlib
import json
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from loop_engine.identifiers import namespaced_id
from loop_engine.models import CanonicalEvent


def _text_hash(text: str | None) -> str | None:
    return hashlib.sha256(text.encode()).hexdigest() if text else None


def _message_content(record: dict[str, Any]) -> tuple[str | None, str | None]:
    message = record.get("message") or {}
    content = message.get("content")
    if isinstance(content, list):
        text_blocks: list[str] = []
        for block in content:
            if isinstance(block, str):
                text_blocks.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                text_blocks.append(block["text"])
        content = "\n".join(text_blocks) or None
    elif content is not None and not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    return message.get("role"), content


def _timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        if not value:
            raise ValueError("Claude Code record is missing timestamp")
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


class ClaudeCodeJsonlSource:
    def __init__(self, source_id: str, path_pattern: str) -> None:
        self.source_id = source_id
        self.path_pattern = path_pattern

    def iter_events(self) -> Iterable[CanonicalEvent]:
        for filename in sorted(glob.glob(self.path_pattern, recursive=True)):
            path = Path(filename)
            with path.open(encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    event = self._to_event(record, path, line_number)
                    if event is not None:
                        yield event

    def _to_event(
        self, record: dict[str, Any], path: Path, line_number: int
    ) -> CanonicalEvent | None:
        message = record.get("message")
        tool = record.get("toolUseResult") or {}
        if not isinstance(message, dict) and not tool:
            return None
        if not isinstance(message, dict):
            message = {}
        role, content = _message_content(record)
        tool_name = tool.get("toolName")
        tool_result = tool.get("content")
        if tool_result is not None and not isinstance(tool_result, str):
            tool_result = json.dumps(tool_result, ensure_ascii=False)
        if content is None and tool_result is not None:
            content = tool_result
        raw_ref = f"file://{path.resolve()}#line={line_number}"
        fallback_id = hashlib.sha256(raw_ref.encode()).hexdigest()[:24]
        raw_event_id = str(record.get("uuid") or f"fallback-{fallback_id}")
        event_id = namespaced_id(self.source_id, raw_event_id)
        raw_session_id = record.get("sessionId")
        session_hint = (
            namespaced_id(self.source_id, raw_session_id)
            if raw_session_id is not None
            else None
        )
        raw_parent_id = record.get("parentUuid")
        parent_hint = (
            namespaced_id(self.source_id, raw_parent_id)
            if raw_parent_id is not None
            else None
        )
        is_error = tool.get("isError")
        usage = message.get("usage") or {}
        model = message.get("model") or record.get("model")
        if isinstance(model, dict):
            model = model.get("id")
        api_error = record.get("isApiErrorMessage") is True or bool(record.get("error"))
        status = (
            "error"
            if is_error is True or api_error
            else ("success" if is_error is False else None)
        )
        return CanonicalEvent(
            event_id=event_id,
            source_id=self.source_id,
            timestamp=_timestamp(record.get("timestamp")),
            event_type=str(record.get("type") or role or "unknown"),
            session_hint=session_hint,
            parent_hint=parent_hint,
            actor_id=record.get("userId"),
            role=role or ("tool" if tool else None),
            content=content,
            content_hash=_text_hash(content),
            model=model,
            tool_name=tool_name,
            tool_arguments=tool.get("input"),
            tool_result=tool_result,
            status=status,
            input_tokens=usage.get("input_tokens"),
            output_tokens=usage.get("output_tokens"),
            asset_markers=list(record.get("assetMarkers") or []),
            raw_ref=raw_ref,
        )
