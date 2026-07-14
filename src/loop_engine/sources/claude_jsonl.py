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


def _text_from_content(content: Any) -> str | None:
    if content is None:
        return None
    if isinstance(content, str):
        return content or None
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and isinstance(block.get("text"), str):
                parts.append(block["text"])
        return "\n".join(parts) or None
    return json.dumps(content, ensure_ascii=False)


def _iter_blocks(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, list):
        return [block for block in content if isinstance(block, dict)]
    return []


def _tool_result_status(block: dict[str, Any], legacy: dict[str, Any]) -> str | None:
    for source in (block, legacy):
        if not source:
            continue
        for key in ("is_error", "isError"):
            if key in source:
                return "error" if bool(source[key]) else "success"
    return None


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
                    yield from self._to_events(record, path, line_number)

    def _base_ids(
        self, record: dict[str, Any], path: Path, line_number: int, suffix: str | None
    ) -> tuple[str, str, str | None, str | None]:
        raw_ref = f"file://{path.resolve()}#line={line_number}"
        if suffix:
            raw_ref = f"{raw_ref}&block={suffix}"
        fallback_id = hashlib.sha256(raw_ref.encode()).hexdigest()[:24]
        base_uuid = str(record.get("uuid") or f"fallback-{fallback_id}")
        raw_event_id = f"{base_uuid}:{suffix}" if suffix else base_uuid
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
        return raw_ref, event_id, session_hint, parent_hint

    def _to_events(
        self, record: dict[str, Any], path: Path, line_number: int
    ) -> Iterable[CanonicalEvent]:
        message = record.get("message")
        legacy_tool = record.get("toolUseResult") or {}
        if not isinstance(message, dict) and not legacy_tool:
            return
        if not isinstance(message, dict):
            message = {}

        timestamp = _timestamp(record.get("timestamp"))
        role = message.get("role")
        content = message.get("content")
        model = message.get("model") or record.get("model")
        if isinstance(model, dict):
            model = model.get("id")
        actor_id = record.get("userId")
        asset_markers = list(record.get("assetMarkers") or [])
        message_id = message.get("id")
        usage = message.get("usage") or {}
        input_tokens = usage.get("input_tokens")
        output_tokens = usage.get("output_tokens")
        record_type = str(record.get("type") or role or "unknown")
        api_error = record.get("isApiErrorMessage") is True or bool(record.get("error"))

        blocks = _iter_blocks(content)
        tool_blocks = [
            block for block in blocks if block.get("type") in {"tool_use", "tool_result"}
        ]

        if not tool_blocks and not legacy_tool:
            text = _text_from_content(content) if content is not None else None
            raw_ref, event_id, session_hint, parent_hint = self._base_ids(
                record, path, line_number, None
            )
            status = "error" if api_error else None
            yield CanonicalEvent(
                event_id=event_id,
                source_id=self.source_id,
                timestamp=timestamp,
                event_type=record_type,
                session_hint=session_hint,
                parent_hint=parent_hint,
                actor_id=actor_id,
                role=role,
                content=text,
                content_hash=_text_hash(text),
                model=model,
                status=status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                message_id=message_id,
                asset_markers=asset_markers,
                raw_ref=raw_ref,
            )
            return

        if not tool_blocks and legacy_tool:
            raw_ref, event_id, session_hint, parent_hint = self._base_ids(
                record, path, line_number, None
            )
            tool_name = legacy_tool.get("toolName")
            tool_result_text = _text_from_content(legacy_tool.get("content"))
            status = _tool_result_status({}, legacy_tool)
            yield CanonicalEvent(
                event_id=event_id,
                source_id=self.source_id,
                timestamp=timestamp,
                event_type="tool_result",
                session_hint=session_hint,
                parent_hint=parent_hint,
                actor_id=actor_id,
                role=role or "tool",
                content=tool_result_text,
                content_hash=_text_hash(tool_result_text),
                model=model,
                tool_name=str(tool_name) if tool_name is not None else None,
                tool_arguments=legacy_tool.get("input"),
                tool_result=tool_result_text,
                status=status,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                message_id=message_id,
                asset_markers=asset_markers,
                raw_ref=raw_ref,
            )
            return

        for index, block in enumerate(tool_blocks):
            block_type = block.get("type")
            suffix = f"{block_type}-{block.get('id') or index}"
            raw_ref, event_id, session_hint, parent_hint = self._base_ids(
                record, path, line_number, suffix
            )
            if block_type == "tool_use":
                tool_name = block.get("name")
                tool_arguments = block.get("input")
                block_content = _text_from_content(tool_arguments)
                yield CanonicalEvent(
                    event_id=event_id,
                    source_id=self.source_id,
                    timestamp=timestamp,
                    event_type="tool_use",
                    session_hint=session_hint,
                    parent_hint=parent_hint,
                    actor_id=actor_id,
                    role=role or "assistant",
                    content=block_content,
                    content_hash=_text_hash(block_content),
                    model=model,
                    tool_name=str(tool_name) if tool_name is not None else None,
                    tool_arguments=tool_arguments,
                    status="error" if api_error else None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    message_id=message_id,
                    asset_markers=asset_markers,
                    raw_ref=raw_ref,
                )
                continue
            # tool_result block
            legacy_for_status = legacy_tool if index == 0 else {}
            tool_name = legacy_tool.get("toolName") if index == 0 else None
            tool_result_content = block.get("content")
            if tool_result_content is None:
                tool_result_content = legacy_tool.get("content")
            tool_result_text = _text_from_content(tool_result_content)
            status = _tool_result_status(block, legacy_for_status)
            yield CanonicalEvent(
                event_id=event_id,
                source_id=self.source_id,
                timestamp=timestamp,
                event_type="tool_result",
                session_hint=session_hint,
                parent_hint=parent_hint,
                actor_id=actor_id,
                role=role or "tool",
                content=tool_result_text,
                content_hash=_text_hash(tool_result_text),
                model=model,
                tool_name=str(tool_name) if tool_name is not None else None,
                tool_arguments=legacy_tool.get("input") if index == 0 else None,
                tool_result=tool_result_text,
                status=status,
                input_tokens=input_tokens if index == 0 else None,
                output_tokens=output_tokens if index == 0 else None,
                message_id=message_id,
                asset_markers=asset_markers,
                raw_ref=raw_ref,
            )
