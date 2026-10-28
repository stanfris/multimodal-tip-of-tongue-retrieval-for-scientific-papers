"""Print the LLM relevance table: uv run python annotations/scripts/llm_annotation_results.py.

Subset frequencies count each query once. Linear weighted kappa compares
the LLM with each annotator separately and includes human-human agreement.
Full frequencies count all saved judge records under data/query_collections.
"""

import sys
from collections import Counter
from pathlib import Path

from annotation_results import calculate_results, load_annotations, weighted_kappa
from common.jsonl import read_jsonl_objects
from queries.count_query_judgements import read_judgement_value


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
    human_annotations = []
    comparisons = []
    for annotator in (1, 2):
        annotations = load_annotations(directory / f"original_query_annotator_{annotator}.csv")
        human_annotations.append(annotations)
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
        comparisons.append((f"LLM vs. annotator {annotator}", weighted_kappa(pairs), len(pairs)))
    if not subset:
        raise ValueError("No overlap between LLM and completed human relevance ratings")

    _, human_kappa, human_n = calculate_results(*human_annotations, "relevance")
    comparisons.append(("Annotator 1 vs. annotator 2", human_kappa, human_n))
    subset_counts = Counter(subset.values())
    full_counts = Counter(judges.values())
    print(f"Subset: {len(subset)} unique queries; {dict(subset_counts)}",
          file=sys.stderr)
    for comparison, kappa, n in comparisons:
        print(f"{comparison}: {n} paired items; linear weighted kappa={kappa}", file=sys.stderr)
    print(f"Full: {len(judges)} saved judge records; {dict(sorted(full_counts.items()))}", file=sys.stderr)

    print(r"""\begin{table}[t]
\centering
\footnotesize

\begin{minipage}[t]{0.48\textwidth}
\centering
\textbf{Annotated Test Set}\\[4pt]
\begin{tabularx}{\linewidth}{@{}Xrr@{}}
\toprule
\textbf{Score} & \textbf{Freq.} & \textbf{Freq. (\%)} \\
\midrule""")
    for score in range(3):
        count = subset_counts[score]
        print(f"{score} & {count} & {100 * count / len(subset):.1f} " + r"\\")
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

\medskip
\begin{tabularx}{\linewidth}{@{}Xrr@{}}
\toprule
\textbf{Comparison} & \textbf{Paired items} & \textbf{Linear weighted $\kappa$} \\
\midrule""")
    for comparison, kappa, n in comparisons:
        agreement = "--" if kappa is None else f"{kappa:.3f}"
        print(f"{comparison} & {n} & {agreement} " + r"\\")
    print(r"""\bottomrule
\end{tabularx}

\caption{Distribution of LLM-assigned relevance scores for
the manually annotated test subset (left) and the full set of saved
LLM judgements (right). Frequencies count each query once.
Linear weighted Cohen's $\kappa$ measures overall relevance agreement
across scores 0, 1, and 2 for each pair of raters separately.
Each comparison uses all items with completed ratings from both raters;
the human--human reference does not require an LLM judgement.
A dash denotes undefined $\kappa$.}
\label{tab:llm-annotation-results}
\vspace{-0.5cm}
\end{table}""")


if __name__ == "__main__":
    main()
