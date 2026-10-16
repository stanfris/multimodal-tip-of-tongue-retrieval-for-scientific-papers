from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from queries.count_query_judgements import main


@pytest.mark.parametrize("delete_failed", [False, True])
def test_failed_rows_are_deleted_only_with_flag(tmp_path, monkeypatch, capsys, delete_failed):
    path = tmp_path / "query_judgements.jsonl"
    kept = [
        json.dumps({"query_id": "good", "judgement": {"relevance_score": 1}}) + "\r\n",
        json.dumps({"query_id": "recovered", "judgement": {
            "parse_error": True, "raw_output": '{"relevance_score": 2, "reasoning": "unfinished',
        }}) + "\r\n",
    ]
    failed = json.dumps({"query_id": "failed", "judgement": {"parse_error": True}}) + "\r\n"
    original = kept[0] + "\r\n" + failed + kept[1]
    path.write_bytes(original.encode())
    path.chmod(0o640)
    other = tmp_path / "other" / "query_judgements.jsonl"
    other.parent.mkdir()
    other.write_text(failed)
    untouched = tmp_path / "queries.jsonl"
    untouched.write_text(failed)
    monkeypatch.setattr(sys, "argv", ["count", str(tmp_path)] + (["--delete-failed"] if delete_failed else []))

    assert main() == 0

    output = capsys.readouterr().out
    assert f"{path}:3 | query_id=failed" in output
    assert "Missing/null\t2" in output
    expected = kept[0] + "\r\n" + kept[1] if delete_failed else original
    assert path.read_bytes() == expected.encode()
    assert path.stat().st_mode & 0o777 == 0o640
    assert other.read_bytes() == (b"" if delete_failed else failed.encode())
    assert untouched.read_bytes() == failed.encode()
    assert ("Deleted" in output) == delete_failed
    assert sorted(item.name for item in tmp_path.iterdir()) == ["other", "queries.jsonl", "query_judgements.jsonl"]


def test_invalid_input_prevents_deletion(tmp_path: Path, monkeypatch):
    first = tmp_path / "a" / "query_judgements.jsonl"
    first.parent.mkdir()
    original = '{"judgement": {}}\n'
    first.write_text(original)
    second = tmp_path / "z" / "query_judgements.jsonl"
    second.parent.mkdir()
    second.write_text("invalid JSON\n")
    monkeypatch.setattr(sys, "argv", ["count", str(tmp_path), "--delete-failed"])

    with pytest.raises(ValueError, match="Invalid JSONL"):
        main()

    assert first.read_text() == original
