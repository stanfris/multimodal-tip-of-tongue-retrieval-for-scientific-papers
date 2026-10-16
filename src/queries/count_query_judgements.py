"""Count saved query judgements by score without loading an inference model."""

from __future__ import annotations

import argparse
import json
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
    missing = 0
    for path in paths:
        for row in read_jsonl_objects(path):
            value = row["judgement"].get(args.field)
            if value is None:
                missing += 1
            else:
                counts[json.dumps(value, ensure_ascii=False)] += 1

    total = sum(counts.values()) + missing
    print(f"Files: {len(paths)} | Judgements: {total} | Field: {args.field}")
    print("Value\tCount")
    for value, count in sorted(counts.items()):
        print(f"{value}\t{count}")
    print(f"Missing/null\t{missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
