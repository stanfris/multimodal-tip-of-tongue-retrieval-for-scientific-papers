"""Print the LLM relevance table: uv run python annotations/scripts/llm_annotation_results.py.

Subset frequencies count each query once. Agreement pools all completed
human ratings, pairing the LLM score with each annotator independently.
Full frequencies count all saved judge records under data/query_collections.
"""

import sys
from collections import Counter
from pathlib import Path

from sklearn.metrics import cohen_kappa_score

from annotation_results import load_annotations
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
    pairs = []
    for annotator in (1, 2):
        annotations = load_annotations(directory / f"original_query_annotator_{annotator}.csv")
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
    if not pairs:
        raise ValueError("No overlap between LLM and completed human relevance ratings")

    subset_counts = Counter(subset.values())
    full_counts = Counter(judges.values())
    kappas = []
    for score in range(3):
        llm = [judge == score for judge, human in pairs]
        humans = [human == score for judge, human in pairs]
        # Kappa is undefined when both sides always choose the same binary value.
        constant = len(set(llm + humans)) == 1
        kappas.append(None if constant else cohen_kappa_score(llm, humans))
    print(f"Subset: {len(subset)} unique queries; {len(pairs)} LLM-human pairs; {dict(subset_counts)}",
          file=sys.stderr)
    print(f"Full: {len(judges)} saved judge records; {dict(sorted(full_counts.items()))}", file=sys.stderr)

    print(r"""\begin{table}[t]
\centering
\footnotesize

\begin{minipage}[t]{0.48\textwidth}
\centering
\textbf{Annotated Test Set}\\[4pt]
\begin{tabularx}{\linewidth}{@{}Xrrr@{}}
\toprule
\textbf{Score} & \textbf{Freq.} & \textbf{Freq. (\%)} & \textbf{$\kappa$} \\
\midrule""")
    for score, kappa in enumerate(kappas):
        count = subset_counts[score]
        agreement = "--" if kappa is None else f"{kappa:.3f}"
        print(f"{score} & {count} & {100 * count / len(subset):.1f} & {agreement} " + r"\\")
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

\caption{Distribution of LLM-assigned relevance scores for
the manually annotated test subset (left) and the full set of saved
LLM judgements (right). Frequencies count each query once.
Cohen's $\kappa$ measures one-vs-rest agreement between the LLM judge
and human relevance annotations, pooling all available LLM--human pairs.
Queries rated by both annotators contribute two agreement pairs.
A dash denotes undefined $\kappa$.}
\label{tab:llm-annotation-results}
\vspace{-0.5cm}
\end{table}""")


if __name__ == "__main__":
    main()
