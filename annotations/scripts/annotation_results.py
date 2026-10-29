"""Print the annotation-results LaTeX table using the two annotator CSVs.

Frequencies pool all nonempty ratings. Linear weighted Cohen's kappa uses only
items rated by both annotators; an undefined kappa is displayed as ``--``.
Run with: python annotations/scripts/annotation_results.py
"""

import csv
import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score


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


def weighted_kappa(pairs):
    """Return linear weighted kappa, or None when it is undefined."""
    if not pairs or len({score for pair in pairs for score in pair}) == 1:
        return None
    first, second = zip(*pairs)
    return cohen_kappa_score(first, second, labels=[0, 1, 2], weights="linear")


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
    frequencies = [100 * counts[score] / total for score in range(3)]
    print(f"{metric}: {total} ratings; {n} paired items; counts={dict(sorted(counts.items()))}",
          file=sys.stderr)
    return frequencies, weighted_kappa(pairs), n


def main():
    directory = Path(__file__).resolve().parents[1] / "data"
    first = load_annotations(directory / "original_query_annotator_1.csv")
    second = load_annotations(directory / "original_query_annotator_2.csv")
    results = {metric: calculate_results(first, second, metric) for metric in METRICS}

    print(r"""\begin{table}[t]
\centering
\footnotesize
\caption{Distribution of human annotation scores for relevance and
well-formedness. Frequencies indicate the percentage of completed
annotations assigned each score. Linear-weighted Cohen's $\kappa$
measures inter-annotator agreement on paired items.""")
    print(f"Relevance uses {results['relevance'][2]} paired items and well-formedness "
          f"uses {results['well_formedness'][2]} paired items." + "}")
    print(r"""\label{tab:annotation-results}
""")
    for index, (metric, title) in enumerate(METRICS.items()):
        if index:
            print(r"\hfill")
        print(r"\begin{minipage}[t]{0.48\textwidth}")
        print(r"\centering")
        frequencies, kappa, n = results[metric]
        kappa_text = "--" if kappa is None else f"{kappa:.3f}"
        print(r"\textbf{" + title + r" ($\kappa = " + kappa_text + r"$)}\\[4pt]")
        print(r"""\begin{tabularx}{\linewidth}{@{}Xr@{}}
\toprule
\textbf{Score} & \textbf{Freq. (\%)} \\
\midrule""")
        for score, frequency in enumerate(frequencies):
            print(f"{score} & {frequency:.1f} " + r"\\")
        print(r"""\bottomrule
\end{tabularx}
\end{minipage}""")
    print(r"""
\vspace{-0.5cm}
\end{table}""")


if __name__ == "__main__":
    main()
