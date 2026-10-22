"""Shared JSONL readers and appenders."""

from __future__ import annotations

import json
import os
import tempfile
import warnings
from pathlib import Path
from typing import Any


def delete_jsonl_lines(path: Path, line_numbers: set[int]) -> None:
    """Atomically delete selected 1-based lines, preserving the remaining bytes and permissions."""
    with path.open(encoding="utf-8", newline="") as handle:
        lines = handle.readlines()
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", dir=path.parent, delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        try:
            handle.writelines(line for number, line in enumerate(lines, 1) if number not in line_numbers)
            handle.close()
            temporary_path.chmod(path.stat().st_mode)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)


def read_jsonl_objects(
    path: str | Path,
    *,
    missing_ok: bool = False,
    skip_invalid: bool = False,
) -> list[dict[str, Any]]:
    """Read a JSONL file whose nonblank lines must all be JSON objects."""
    data_path = Path(path)
    if not data_path.exists():
        if missing_ok:
            return []
        raise FileNotFoundError(f"JSONL file does not exist: {data_path}")

    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data_path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            if skip_invalid:
                warnings.warn(
                    f"Skipping invalid JSONL at {data_path}:{line_number}: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            raise ValueError(f"Invalid JSONL at {data_path}:{line_number}") from exc
        if not isinstance(row, dict):
            if skip_invalid:
                warnings.warn(
                    f"Skipping non-object JSONL row at {data_path}:{line_number}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                continue
            raise ValueError(f"Expected JSON object at {data_path}:{line_number}")
        rows.append(row)
    return rows


def append_jsonl_object(
    path: str | Path,
    row: dict[str, Any],
    *,
    sort_keys: bool = True,
    fsync: bool = False,
    append: bool = True,
) -> None:
    """Write one JSON object to a JSONL file."""
    data_path = Path(path)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with data_path.open(mode, encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=sort_keys, ensure_ascii=False))
        handle.write("\n")
        if fsync:
            handle.flush()
            os.fsync(handle.fileno())
