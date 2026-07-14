from __future__ import annotations

import glob
import hashlib
import json
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from loop_engine.identifiers import compound_id, namespaced_id
from loop_engine.models import CanonicalEvent


def _hash(text: str | None) -> str | None:
    return hashlib.sha256(text.encode()).hexdigest() if text else None


def _content(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False)


def _messages(record: dict[str, Any]) -> list[tuple[str, str | None]]:
    raw_messages = record.get("messages") or (record.get("request") or {}).get("messages") or []
    return [
        (str(message.get("role") or "unknown"), _content(message.get("content")))
        for message in raw_messages
    ]


def _response(record: dict[str, Any]) -> tuple[str, str | None]:
    response = record.get("response") or {}
    text = _content(
        response.get("content")
        or ((response.get("choices") or [{}])[0].get("message") or {}).get("content")
    )
    return str(response.get("role") or "assistant"), text


def _timestamp(value: Any) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        if not value:
            raise ValueError("LiteLLM record is missing timestamp")
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _request_id(record: dict[str, Any], raw_ref: str) -> str:
    explicit = record.get("request_id") or record.get("id")
    if explicit is not None:
        return str(explicit)
    return hashlib.sha256(raw_ref.encode()).hexdigest()[:24]


def _first_not_none(*values: Any) -> Any:
    return next((value for value in values if value is not None), None)


def _records_from_text(text: str, suffix: str) -> list[dict[str, Any]]:
    if suffix.endswith(".jsonl"):
        raw_records = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        payload = json.loads(text)
        raw_records = payload if isinstance(payload, list) else [payload]
    if not all(isinstance(record, dict) for record in raw_records):
        raise ValueError("Each LiteLLM record must be a JSON object")
    return [record for record in raw_records if isinstance(record, dict)]


@dataclass
class _SessionState:
    session_id: str
    actor_model: tuple[str | None, str | None]
    transcript: list[tuple[str, str | None]]


class LiteLLMSessionizer:
    """Infer request lineage from explicit IDs or exact message-history prefixes."""

    def __init__(self) -> None:
        self._states: list[_SessionState] = []

    def assign(self, record: dict[str, Any], raw_ref: str) -> tuple[str, int]:
        request_id = _request_id(record, raw_ref)
        metadata = record.get("metadata") or {}
        explicit = metadata.get("session_id") or record.get("session_id")
        actor_model = (record.get("virtual_key") or record.get("user"), record.get("model"))
        messages = _messages(record)

        state: _SessionState | None = None
        transcript_prefix: list[tuple[str, str | None]] = []
        if explicit:
            session_id = str(explicit)
            same_session = [item for item in self._states if item.session_id == session_id]
            prefix_candidates = [
                item
                for item in same_session
                if len(messages) >= len(item.transcript)
                and messages[: len(item.transcript)] == item.transcript
            ]
            state = max(prefix_candidates, key=lambda item: len(item.transcript), default=None)
            if state is None and same_session:
                transcript_prefix = max(
                    same_session, key=lambda item: len(item.transcript)
                ).transcript
        else:
            candidates = [
                item
                for item in self._states
                if item.actor_model == actor_model
                and len(messages) >= len(item.transcript)
                and messages[: len(item.transcript)] == item.transcript
            ]
            state = max(candidates, key=lambda item: len(item.transcript), default=None)
            session_id = state.session_id if state else request_id

        offset = len(state.transcript) if state else 0
        response_role, response_text = _response(record)
        transcript = transcript_prefix + list(messages)
        if response_text is not None:
            transcript.append((response_role, response_text))
        self._states.append(_SessionState(session_id, actor_model, transcript))
        return session_id, offset


def events_from_litellm_record(
    record: dict[str, Any],
    source_id: str,
    raw_ref: str,
    *,
    session_hint: str | None = None,
    message_offset: int = 0,
) -> Iterator[CanonicalEvent]:
    request_id = _request_id(record, raw_ref)
    resolved_session = (
        session_hint or (record.get("metadata") or {}).get("session_id") or request_id
    )
    namespaced_session = namespaced_id(source_id, resolved_session)
    model = record.get("model")
    actor_id = record.get("virtual_key") or record.get("user")
    timestamp_raw = record.get("start_time") or record.get("timestamp")
    timestamp = _timestamp(timestamp_raw)
    markers = list((record.get("metadata") or {}).get("asset_markers") or [])
    messages = _messages(record)
    for index, (role, text) in enumerate(messages[message_offset:], start=message_offset):
        yield CanonicalEvent(
            event_id=namespaced_id(
                source_id, compound_id(request_id, "request", index)
            ),
            source_id=source_id,
            timestamp=timestamp,
            event_type="message",
            session_hint=namespaced_session,
            actor_id=actor_id,
            role=role,
            content=text,
            content_hash=_hash(text),
            model=model,
            asset_markers=markers,
            raw_ref=raw_ref,
        )
    response = record.get("response") or {}
    response_role, response_text = _response(record)
    raw_status = str(record.get("status") or "success").lower()
    status = "error" if raw_status in {"error", "failure", "failed"} else raw_status
    if response_text is not None or status == "error":
        usage = record.get("usage") or response.get("usage") or {}
        start = timestamp
        end_raw = record.get("end_time") or timestamp
        end = _timestamp(end_raw)
        response_content = response_text or _content(record.get("error")) or raw_status
        yield CanonicalEvent(
            event_id=namespaced_id(source_id, compound_id(request_id, "response")),
            source_id=source_id,
            timestamp=end,
            event_type="message" if response_text is not None else "request_error",
            session_hint=namespaced_session,
            actor_id=actor_id,
            role=response_role,
            content=response_content,
            content_hash=_hash(response_content),
            model=model,
            status=status,
            input_tokens=_first_not_none(
                usage.get("prompt_tokens"), usage.get("input_tokens")
            ),
            output_tokens=_first_not_none(
                usage.get("completion_tokens"), usage.get("output_tokens")
            ),
            cost_usd=_first_not_none(record.get("cost"), record.get("cost_usd")),
            latency_ms=max(0, int((end - start).total_seconds() * 1000)),
            asset_markers=markers,
            raw_ref=raw_ref,
        )


def _events_from_records(
    records: list[tuple[dict[str, Any], str]], source_id: str
) -> Iterator[CanonicalEvent]:
    sessionizer = LiteLLMSessionizer()
    sorted_records = sorted(
        records,
        key=lambda item: _timestamp(
            item[0].get("start_time") or item[0].get("timestamp")
        ),
    )
    for record, raw_ref in sorted_records:
        session_id, offset = sessionizer.assign(record, raw_ref)
        yield from events_from_litellm_record(
            record,
            source_id,
            raw_ref,
            session_hint=session_id,
            message_offset=offset,
        )


class LiteLLMLocalJsonSource:
    def __init__(self, source_id: str, path_pattern: str) -> None:
        self.source_id = source_id
        self.path_pattern = path_pattern

    def iter_events(self) -> Iterable[CanonicalEvent]:
        records: list[tuple[dict[str, Any], str]] = []
        for filename in sorted(glob.glob(self.path_pattern, recursive=True)):
            path = Path(filename)
            text = path.read_text(encoding="utf-8")
            for index, record in enumerate(_records_from_text(text, path.name)):
                records.append((record, f"file://{path.resolve()}#record={index}"))
        return _events_from_records(records, self.source_id)


class LiteLLMS3JsonSource:
    def __init__(
        self,
        source_id: str,
        uri: str,
        aws_profile: str | None = None,
        max_object_bytes: int = 10 * 1024 * 1024,
        max_total_bytes: int = 100 * 1024 * 1024,
        *,
        client: Any | None = None,
    ) -> None:
        parsed = urlparse(uri)
        if parsed.scheme != "s3" or not parsed.netloc:
            raise ValueError(f"Invalid S3 URI: {uri}")
        self.source_id = source_id
        self.bucket = parsed.netloc
        prefix = parsed.path.lstrip("/")
        self.prefix = prefix.rstrip("/") + "/" if prefix else ""
        if max_object_bytes < 1 or max_total_bytes < max_object_bytes:
            raise ValueError(
                "S3 byte limits require max_total_bytes >= max_object_bytes >= 1"
            )
        self.max_object_bytes = max_object_bytes
        self.max_total_bytes = max_total_bytes
        if client is None:
            import boto3

            client = boto3.Session(profile_name=aws_profile).client("s3")
        self.client = client

    def iter_events(self) -> Iterable[CanonicalEvent]:
        records: list[tuple[dict[str, Any], str]] = []
        total_bytes = 0
        paginator = self.client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=self.prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if self.prefix and not key.startswith(self.prefix):
                    continue
                if not key.endswith((".json", ".jsonl")):
                    continue
                listed_size = item.get("Size")
                if listed_size is not None and int(listed_size) > self.max_object_bytes:
                    raise ValueError(f"S3 object size limit exceeded: s3://{self.bucket}/{key}")
                body_stream = self.client.get_object(Bucket=self.bucket, Key=key)["Body"]
                remaining_total = self.max_total_bytes - total_bytes
                read_limit = min(self.max_object_bytes, remaining_total) + 1
                body = body_stream.read(read_limit)
                if len(body) > self.max_object_bytes:
                    raise ValueError(f"S3 object size limit exceeded: s3://{self.bucket}/{key}")
                if len(body) > remaining_total:
                    raise ValueError("S3 total byte limit exceeded")
                total_bytes += len(body)
                text = body.decode("utf-8")
                for index, record in enumerate(_records_from_text(text, key)):
                    records.append((record, f"s3://{self.bucket}/{key}#record={index}"))
        return _events_from_records(records, self.source_id)
