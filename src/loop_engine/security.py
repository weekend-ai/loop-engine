from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_SECRET_NAME = (
    r"api[_-]?key|access[_-]?token|refresh[_-]?token|session[_-]?token|id[_-]?token|"
    r"client[_-]?secret|private[_-]?key|token|authorization|password|passwd|secret|cookie"
)
_SECRET_ASSIGNMENT = re.compile(
    rf"(?i)(?<![\w-])([\"']?)({_SECRET_NAME})\1"
    r"(\s*[:=]\s*)([\"']?)([^\"'\s,;}]+)\4"
)
_BEARER_TOKEN = re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+")
_PROVIDER_TOKEN = re.compile(r"\b(?:sk-ant-|sk-)[A-Za-z0-9_-]{12,}\b")
_AWS_ACCESS_KEY = re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b")
_JWT_TOKEN = re.compile(r"\beyJ[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\.[A-Za-z0-9_-]{6,}\b")
_DB_CONNECTION = re.compile(
    r"\b(?:mongodb(?:\+srv)?|postgres(?:ql)?|mysql|redis|amqp|amqps|mssql|jdbc:[^:]+)"
    r"://[^\s\"'<>]+"
)
_SECRET_KEY = re.compile(
    r"(?i)^(?:api[_-]?key|access[_-]?token|refresh[_-]?token|session[_-]?token|"
    r"id[_-]?token|client[_-]?secret|private[_-]?key|token|authorization|password|"
    r"passwd|secret|cookie)$"
)

_TRUNCATION_MARKER = "...[TRUNCATED]"
_MAX_REDACT_DEPTH = 64


def redact_text(value: str | None, max_chars: int) -> str | None:
    if value is None:
        return None
    if max_chars < len(_TRUNCATION_MARKER):
        raise ValueError(
            f"max_chars={max_chars} smaller than truncation marker length "
            f"{len(_TRUNCATION_MARKER)}"
        )
    redacted = _SECRET_ASSIGNMENT.sub(r"\1\2\1\3\4[REDACTED]\4", value)
    redacted = _BEARER_TOKEN.sub("Bearer [REDACTED]", redacted)
    redacted = _PROVIDER_TOKEN.sub("[REDACTED]", redacted)
    redacted = _AWS_ACCESS_KEY.sub("[REDACTED]", redacted)
    redacted = _JWT_TOKEN.sub("[REDACTED]", redacted)
    redacted = _DB_CONNECTION.sub("[REDACTED]", redacted)
    if len(redacted) > max_chars:
        head = redacted[: max_chars - len(_TRUNCATION_MARKER)]
        return head + _TRUNCATION_MARKER
    return redacted


def redact_value(value: Any, max_chars: int, *, _depth: int = 0) -> Any:
    if _depth > _MAX_REDACT_DEPTH:
        raise ValueError(
            f"redaction recursion depth exceeded ({_MAX_REDACT_DEPTH}); refusing to serialize"
        )
    if isinstance(value, str):
        return redact_text(value, max_chars)
    if isinstance(value, list):
        return [redact_value(item, max_chars, _depth=_depth + 1) for item in value]
    if isinstance(value, dict):
        return {
            str(key): "[REDACTED]"
            if _SECRET_KEY.fullmatch(str(key)) and item is not None
            else redact_value(item, max_chars, _depth=_depth + 1)
            for key, item in value.items()
        }
    return value


def secure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    path.chmod(0o700)


def secure_file(path: Path) -> None:
    path.chmod(0o600)


def secure_write_text(path: Path, content: str) -> None:
    if not path.parent.exists():
        secure_directory(path.parent)
    path.write_text(content)
    secure_file(path)
