"""Count saved query judgements by score without loading an inference model."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

from common.jsonl import read_jsonl_objects


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", type=Path, default=Path("data/query_collections"),
        help="Judgement JSONL file or directory to search recursively (default: data/query_collections).",
    )
    parser.add_argument(
        "--field", default="relevance_score",
        help="Judgement field to count (default: relevance_score).",
    )
    args = parser.parse_args()
    if not args.path.exists():
        raise FileNotFoundError(args.path)
    paths = [args.path] if args.path.is_file() else sorted(args.path.rglob("query_judgements.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No query_judgements.jsonl files found under {args.path}")

    counts: Counter[str] = Counter()
    failures = []
    raw_field = re.compile(
        rf'(?<!\\)"{re.escape(args.field)}"\s*:\s*'
        r'(true|false|null|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)(?=\s*[,}\]]|\s*$)'
    )
    for path in paths:
        rows = read_jsonl_objects(path)
        with path.open(encoding="utf-8") as handle:
            line_numbers = [number for number, line in enumerate(handle, 1) if line.strip()]
        assert len(line_numbers) == len(rows)
        for line_number, row in zip(line_numbers, rows):
            judgement = row["judgement"]
            value = judgement.get(args.field)
            if value is None:
                match = raw_field.search(judgement.get("raw_output") or "")
                if match is not None:
                    value = json.loads(match.group(1))
            if value is None:
                reason = judgement.get("error") or (
                    "parse_error; no readable value in raw_output" if judgement.get("parse_error")
                    else f"{args.field} missing/null; no readable value in raw_output"
                )
                failures.append((path.resolve(), line_number, row.get("query_id"), reason))
            else:
                counts[json.dumps(value, ensure_ascii=False)] += 1

    total = sum(counts.values()) + len(failures)
    print(f"Files: {len(paths)} | Judgements: {total} | Field: {args.field}")
    print("Value\tCount")
    for value, count in sorted(counts.items()):
        print(f"{value}\t{count}")
    print(f"Missing/null\t{len(failures)}")
    if failures:
        print("\nJudgements without a readable value:")
        for path, line_number, query_id, reason in failures:
            print(f"{path}:{line_number} | query_id={query_id} | {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
