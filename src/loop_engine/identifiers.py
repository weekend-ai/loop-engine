from __future__ import annotations


def compound_id(*parts: object) -> str:
    """Encode identifier components without delimiter ambiguity."""
    return "|".join(f"{len(str(part))}:{part}" for part in parts)


def namespaced_id(source_id: str, raw_id: object) -> str:
    """Return an unambiguous source-scoped identifier."""
    return compound_id(source_id, raw_id)
