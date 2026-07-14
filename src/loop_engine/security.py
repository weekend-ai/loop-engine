from __future__ import annotations

from pathlib import Path


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
