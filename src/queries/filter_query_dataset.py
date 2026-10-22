"""Remove queries with zero or unreadable relevance scores, leaving judgements unchanged."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from common.jsonl import delete_jsonl_lines, read_jsonl_objects
from queries.count_query_judgements import read_judgement_value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path", nargs="?", type=Path, default=Path("data/query_collections"),
        help="Judgement JSONL file or directory to search recursively (default: data/query_collections).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report removals without changing files.")
    args = parser.parse_args()
    if not args.path.exists():
        raise FileNotFoundError(args.path)
    paths = [args.path] if args.path.is_file() else sorted(args.path.rglob("query_judgements.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No query_judgements.jsonl files found under {args.path}")

    rewrites = []
    total = 0
    for judgement_path in paths:
        rejected_ids = set()
        for row in read_jsonl_objects(judgement_path):
            score = read_judgement_value(row["judgement"])
            if type(score) not in (int, float) or not math.isfinite(score) or score == 0:
                rejected_ids.add(row["query_id"])

        query_path = judgement_path.with_name("queries.jsonl")
        rows = read_jsonl_objects(query_path)
        with query_path.open(encoding="utf-8", newline="") as handle:
            line_numbers = [number for number, line in enumerate(handle, 1) if line.strip()]
        assert len(line_numbers) == len(rows)
        rejected_lines = {
            number for number, row in zip(line_numbers, rows)
            if row["query_id"] in rejected_ids
        }
        total += len(rows)
        rewrites.append((query_path, rejected_lines))

    for query_path, rejected_lines in rewrites:
        if rejected_lines and not args.dry_run:
            delete_jsonl_lines(query_path, rejected_lines)
        action = "Would remove" if args.dry_run else "Removed"
        print(f"{action} {len(rejected_lines)} queries from {query_path.resolve()}")
    removed = sum(len(lines) for _, lines in rewrites)
    print(f"Files: {len(paths)} | Queries: {total} | {'Would remove' if args.dry_run else 'Removed'}: {removed}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
