from pathlib import Path

import pytest
from pydantic import ValidationError

from loop_engine.config import load_config


@pytest.mark.parametrize(
    "fragment",
    [
        "version: 999\nsources: []\n",
        "version: 1\nsources: []\nunknown_root: true\n",
        (
            "version: 1\nsources:\n"
            "  - id: bad\n"
            "    type: claude_code_jsonl\n"
            "    path: ./logs/*.jsonl\n"
            "    path_typo: ./ignored\n"
        ),
    ],
)
def test_config_rejects_unsupported_version_and_unknown_fields(
    tmp_path: Path, fragment: str
) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text(fragment)

    with pytest.raises(ValidationError):
        load_config(path)


def test_claude_provider_requires_explicit_external_egress_opt_in(
    tmp_path: Path,
) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text(
        "version: 1\n"
        "sources:\n"
        "  - id: local\n"
        "    type: claude_code_jsonl\n"
        "    path: ./logs/*.jsonl\n"
        "analysis:\n"
        "  provider: claude_sdk\n"
    )

    with pytest.raises(ValidationError, match="external_data_egress_allowed"):
        load_config(path)

    path.write_text(path.read_text() + "  external_data_egress_allowed: true\n")
    assert load_config(path).analysis.external_data_egress_allowed is True


@pytest.mark.parametrize(
    "fragment",
    [
        (
            "version: 1\nsources:\n"
            "  - id: a:b\n"
            "    type: claude_code_jsonl\n"
            "    path: ./one.jsonl\n"
        ),
        (
            "version: 1\nsources:\n"
            "  - id: duplicate\n"
            "    type: claude_code_jsonl\n"
            "    path: ./one.jsonl\n"
            "  - id: duplicate\n"
            "    type: claude_code_jsonl\n"
            "    path: ./two.jsonl\n"
        ),
        (
            "version: 1\nsources: []\nmetrics:\n"
            "  group_by: [task_typo]\n"
        ),
    ],
)
def test_config_rejects_ambiguous_source_ids_and_unknown_group_dimensions(
    tmp_path: Path, fragment: str
) -> None:
    path = tmp_path / "loop.yaml"
    path.write_text(fragment)

    with pytest.raises(ValidationError):
        load_config(path)
