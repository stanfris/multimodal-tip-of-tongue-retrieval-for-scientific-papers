"""Print the LLM relevance table: uv run python annotations/scripts/llm_annotation_results.py.

Subset frequencies count each query once. Linear weighted kappa and unweighted
Gwet's AC1 compare each rater pair using the fixed score categories 0, 1, and 2.
Full frequencies count all saved judge records under data/query_collections.
"""

import sys
from collections import Counter
from pathlib import Path
from statistics import mean

from annotation_results import load_annotations, weighted_kappa
from common.jsonl import read_jsonl_objects
from queries.count_query_judgements import read_judgement_value


def gwet_ac1(pairs):
    """Unweighted AC1 for two raters on the full three-category score scale.

    Gwet's expected agreement is sum(p_k * (1 - p_k)) / (q - 1),
    where p_k pools both raters' category proportions and q = 3.
    See https://www.agreestat.com/documents/agreestatguide.pdf.
    """
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(first == second for first, second in pairs) / n
    counts = Counter(score for pair in pairs for score in pair)
    proportions = [counts[score] / (2 * n) for score in range(3)]
    expected = sum(p * (1 - p) for p in proportions) / 2
    return (observed - expected) / (1 - expected)


def main():
    directory = Path(__file__).resolve().parents[1] / "data"
    root = directory.parents[1] / "data" / "query_collections"
    paths = sorted(root.rglob("query_judgements.jsonl"))
    if not paths:
        raise FileNotFoundError(f"No judge files under {root}")
    judges = {}
    for path in paths:
        dataset = path.relative_to(root).parts[0]
        for row in read_jsonl_objects(path):
            key = (dataset, row["query_id"], row["paper_id"])
            if key in judges:
                raise ValueError(f"Duplicate judge record: {key}")
            score = read_judgement_value(row["judgement"])
            if type(score) is not int or score not in (0, 1, 2):
                raise ValueError(f"Invalid LLM relevance score for {key}: {score!r}")
            judges[key] = score

    subset = {}
    comparisons = []
    for annotator in (1, 2):
        annotations = load_annotations(directory / f"original_query_annotator_{annotator}.csv")
        pairs = []
        completed = matched = 0
        for (task, query_id, dataset, document_id), scores in annotations.items():
            assert task == "original_query"
            human_score = scores["relevance"]
            if human_score is None:
                continue
            completed += 1
            key = (dataset, query_id, document_id)
            if key in judges:
                matched += 1
                subset[key] = judges[key]
                pairs.append((judges[key], human_score))
        print(f"Annotator {annotator}: {matched}/{completed} completed ratings matched", file=sys.stderr)
        comparisons.append((f"LLM vs. annotator {annotator}", weighted_kappa(pairs), gwet_ac1(pairs), len(pairs)))
    if not subset:
        raise ValueError("No overlap between LLM and completed human relevance ratings")

    kappas = [comparison[1] for comparison in comparisons]
    ac1s = [comparison[2] for comparison in comparisons]
    mean_kappa = None if None in kappas else mean(kappas)
    mean_ac1 = None if None in ac1s else mean(ac1s)
    subset_counts = Counter(subset.values())
    full_counts = Counter(judges.values())
    print(f"Subset: {len(subset)} unique queries; {dict(subset_counts)}",
          file=sys.stderr)
    for comparison, kappa, ac1, n in comparisons:
        print(f"{comparison}: {n} paired items; linear weighted kappa={kappa}; AC1={ac1}", file=sys.stderr)
    print(f"Full: {len(judges)} saved judge records; {dict(sorted(full_counts.items()))}", file=sys.stderr)

    print(r"""\begin{table}[t]
\centering
\footnotesize
\caption{Distribution of LLM-assigned relevance scores for
the annotated test subset (left) and all saved LLM judgements (right).
Frequencies count each query once. Linear-weighted Cohen's $\bar\kappa$ is
the arithmetic mean of the two LLM--annotator comparisons, with equal weight
per annotator.""")
    ac1_text = "--" if mean_ac1 is None else f"{mean_ac1:.3f}"
    print("Mean unweighted Gwet's AC1 over the fixed categories 0, 1, and 2 is " + ac1_text + ".}")
    print(r"""\label{tab:llm-annotation-results}

\begin{minipage}[t]{0.48\textwidth}
\centering""")
    kappa_text = "--" if mean_kappa is None else f"{mean_kappa:.3f}"
    print(r"\textbf{Annotated Test Set ($\bar\kappa = " + kappa_text + r"$)}\\[4pt]")
    print(r"""\begin{tabularx}{\linewidth}{@{}Xr@{}}
\toprule
\textbf{Score} & \textbf{Freq. (\%)} \\
\midrule""")
    for score in range(3):
        count = subset_counts[score]
        print(f"{score} & {100 * count / len(subset):.1f} " + r"\\")
    print(r"""\bottomrule
\end{tabularx}
\end{minipage}
\hfill
\begin{minipage}[t]{0.48\textwidth}
\centering
\textbf{Full Dataset}\\[4pt]
\begin{tabularx}{\linewidth}{@{}Xrr@{}}
\toprule
\textbf{Score} & \textbf{Freq.} & \textbf{Freq. (\%)} \\
\midrule""")
    for score in range(3):
        count = full_counts[score]
        print(f"{score} & {count} & {100 * count / len(judges):.2f} " + r"\\")
    print(r"""\bottomrule
\end{tabularx}
\end{minipage}
\vspace{-0.5cm}
\end{table}""")


if __name__ == "__main__":
    main()
