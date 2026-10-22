from __future__ import annotations

import json
import sys

import pytest

from queries.filter_query_dataset import main


@pytest.mark.parametrize("dry_run", [False, True])
def test_filters_only_matching_queries(tmp_path, monkeypatch, capsys, dry_run):
    collection = tmp_path / "train" / "visual_and_text"
    collection.mkdir(parents=True)
    judgements = {
        "zero": {"relevance_score": 0},
        "one": {"relevance_score": 1},
        "two": {"relevance_score": 2},
        "missing": {},
        "null": {"relevance_score": None},
        "unreadable": {"relevance_score": "invalid"},
        "boolean": {"relevance_score": True},
        "nonfinite": {"relevance_score": float("nan")},
        "recovered": {"parse_error": True, "raw_output": '{"relevance_score": 2, "reasoning": "unfinished'},
        "recovered_zero": {"parse_error": True, "raw_output": '{"relevance_score": 0,'},
        "failed": {"parse_error": True, "raw_output": "unreadable"},
        "absent_query": {"relevance_score": 0},
    }
    judgement_path = collection / "query_judgements.jsonl"
    judgement_bytes = "".join(
        json.dumps({"query_id": query_id, "judgement": judgement}) + "\r\n"
        for query_id, judgement in judgements.items()
    ).encode()
    judgement_path.write_bytes(judgement_bytes)
    query_lines = {
        query_id: json.dumps({"query_id": query_id, "query": "図 — query"}, ensure_ascii=False) + "\r\n"
        for query_id in [*judgements, "unjudged"] if query_id != "absent_query"
    }
    original = "\r\n" + "".join(query_lines.values())
    query_path = collection / "queries.jsonl"
    query_path.write_bytes(original.encode())
    query_path.chmod(0o640)
    metadata_path = collection / "metadata.json"
    metadata_path.write_text('{"count": 12}\n')
    # Identical query IDs in a different collection must be matched locally.
    other = tmp_path / "test"
    other.mkdir()
    other_queries = other / "queries.jsonl"
    other_original = query_lines["zero"]
    other_queries.write_bytes(other_original.encode())
    other_judgements = other / "query_judgements.jsonl"
    other_judgements.write_text('{"query_id": "zero", "judgement": {"relevance_score": 2}}\n')
    other_judgement_bytes = other_judgements.read_bytes()
    monkeypatch.setattr(sys, "argv", ["filter", str(tmp_path)] + (["--dry-run"] if dry_run else []))

    assert main() == 0

    output = capsys.readouterr().out
    expected = "\r\n" + "".join(query_lines[key] for key in ["one", "two", "recovered", "unjudged"])
    assert query_path.read_bytes() == (original if dry_run else expected).encode()
    assert query_path.stat().st_mode & 0o777 == 0o640
    assert judgement_path.read_bytes() == judgement_bytes
    assert metadata_path.read_text() == '{"count": 12}\n'
    assert other_queries.read_bytes() == other_original.encode()
    assert other_judgements.read_bytes() == other_judgement_bytes
    assert f"{'Would remove' if dry_run else 'Removed'}: 8" in output
    assert sorted(path.name for path in collection.iterdir()) == [
        "metadata.json", "queries.jsonl", "query_judgements.jsonl",
    ]
    if not dry_run:
        assert main() == 0
        assert "Removed: 0" in capsys.readouterr().out
        assert query_path.read_bytes() == expected.encode()


@pytest.mark.parametrize("invalid_file", ["queries.jsonl", "query_judgements.jsonl"])
def test_invalid_input_prevents_any_query_changes(tmp_path, monkeypatch, invalid_file):
    for name in ("a", "z"):
        collection = tmp_path / name
        collection.mkdir()
        (collection / "query_judgements.jsonl").write_text(
            '{"query_id": "zero", "judgement": {"relevance_score": 0}}\n'
        )
        (collection / "queries.jsonl").write_text('{"query_id": "zero"}\n')
    (tmp_path / "z" / invalid_file).write_text("invalid JSON\n")
    original = {path: path.read_bytes() for path in tmp_path.rglob("*.jsonl")}
    monkeypatch.setattr(sys, "argv", ["filter", str(tmp_path)])

    with pytest.raises(ValueError, match="Invalid JSONL"):
        main()

    assert {path: path.read_bytes() for path in original} == original


def test_single_judgement_file_and_empty_result(tmp_path, monkeypatch):
    judgement_path = tmp_path / "query_judgements.jsonl"
    judgement_path.write_text('{"query_id": "zero", "judgement": {"relevance_score": 0}}\n')
    query_path = tmp_path / "queries.jsonl"
    query_path.write_text('{"query_id": "zero"}\n')
    monkeypatch.setattr(sys, "argv", ["filter", str(judgement_path)])

    assert main() == 0
    assert query_path.read_bytes() == b""
    assert judgement_path.read_text() == '{"query_id": "zero", "judgement": {"relevance_score": 0}}\n'
