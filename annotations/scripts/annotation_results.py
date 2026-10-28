"""Print the annotation-results LaTeX table using the two annotator CSVs.

Frequencies pool all nonempty ratings. One-vs-rest Cohen's kappa uses only
items rated by both annotators; an undefined kappa is displayed as ``--``.
Run with: python annotations/scripts/annotation_results.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path


METRICS = {"relevance": "Relevance", "well_formedness": "Well-formedness"}
KEY_COLUMNS = ("task_type", "query_id", "dataset", "document_id")


def load_annotations(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        rows = list(csv.DictReader(file))
    annotations = {}
    for row in rows:
        key = tuple(row[column] for column in KEY_COLUMNS)
        if key in annotations:
            raise ValueError(f"Duplicate item in {path}: {key}")
        scores = {}
        for metric in METRICS:
            value = row[metric].strip()
            if value not in {"", "0", "1", "2"}:
                raise ValueError(f"Invalid {metric} score in {path}: {value!r}")
            scores[metric] = int(value) if value else None
        annotations[key] = scores
    return annotations


def calculate_results(first, second, metric):
    counts = Counter(
        item[metric]
        for annotations in (first, second)
        for item in annotations.values()
        if item[metric] is not None
    )
    total = sum(counts.values())
    if total == 0:
        raise ValueError(f"No completed ratings for {metric}")
    pairs = [
        (first[key][metric], second[key][metric])
        for key in first.keys() & second.keys()
        if first[key][metric] is not None and second[key][metric] is not None
    ]
    n = len(pairs)
    results = []
    for score in range(3):
        # Binary decisions: this score versus every other score.
        positives_a = sum(a == score for a, b in pairs)
        positives_b = sum(b == score for a, b in pairs)
        agreements = sum((a == score) == (b == score) for a, b in pairs)
        expected = positives_a * positives_b + (n - positives_a) * (n - positives_b)
        denominator = n * n - expected
        kappa = (n * agreements - expected) / denominator if denominator else None
        results.append((100 * counts[score] / total, kappa))
    print(f"{metric}: {total} ratings; {n} paired items; counts={dict(sorted(counts.items()))}",
          file=sys.stderr)
    return results


def main():
    directory = Path(__file__).resolve().parents[1] / "data"
    first = load_annotations(directory / "original_query_annotator_1.csv")
    second = load_annotations(directory / "original_query_annotator_2.csv")
    results = {metric: calculate_results(first, second, metric) for metric in METRICS}

    print(r"""\begin{table}[t]
\centering
\footnotesize
""")
    for index, (metric, title) in enumerate(METRICS.items()):
        if index:
            print(r"\hfill")
        print(r"\begin{minipage}[t]{0.48\textwidth}")
        print(r"\centering")
        print(r"\textbf{" + title + r"}\\[4pt]")
        print(r"""\begin{tabularx}{\linewidth}{@{}Xrr@{}}
\toprule
\textbf{Score} & \textbf{Freq. (\%)} & \textbf{$\kappa$} \\
\midrule""")
        for score, (frequency, kappa) in enumerate(results[metric]):
            agreement = "--" if kappa is None else f"{kappa:.3f}"
            print(f"{score} & {frequency:.1f} & {agreement} " + r"\\")
        print(r"""\bottomrule
\end{tabularx}
\end{minipage}""")
    print(r"""
\caption{Distribution of annotation scores for relevance and
well-formedness. Frequency denotes the percentage of completed annotations
assigned each score, and Cohen's $\kappa$ measures inter-annotator
agreement for each score (one-vs-rest) on items rated by both annotators.
A dash denotes undefined $\kappa$.}
\label{tab:annotation-results}
    \vspace{-0.5cm}
\end{table}""")


if __name__ == "__main__":
    main()
