"""Compare code frequencies. Run: uv run python annotations/scripts/plot_code_frequencies.py"""

import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def normalize_code(code):
    code = code.strip().replace("/ ", "/")
    # The online codebook defines Structural as visual layout/appearance.
    return "Visual Appearance" if code == "Structural" else code


def main():
    directory = Path(__file__).resolve().parents[1] / "data"
    online = read_rows(directory / "online_annotated_queries.csv")
    codes = [normalize_code(column) for column in online[0] if column not in {"Source/Domain", "Link"}]
    graph_codes = ["Present", "Graph Type", "Visual Appearance", "Semantic Details"]
    context_codes = ["Academic Activity", "Physical Relation"]
    codes = [code for code in codes if code not in graph_codes + context_codes] + graph_codes + context_codes
    online_counts = Counter(
        normalize_code(column)
        for row in online
        for column, value in row.items()
        if column not in {"Source/Domain", "Link"} and value.strip().lower() == "x"
    )
    series = [("Online Queries", len(online), online_counts)]
    alternative_counts = Counter()
    alternative_total = 0
    for annotator in (1, 2):
        rows = read_rows(directory / f"codebook_alternative_query_annotator_{annotator}.csv")
        completed = [row for row in rows if row["status"] == "done"]
        counts = Counter(
            code
            for row in completed
            for code in {normalize_code(value) for value in row["criteria"].split(";") if value.strip()}
        )
        unknown = counts.keys() - set(codes)
        if unknown:
            raise ValueError(f"Unknown codes for annotator {annotator}: {unknown}")
        alternative_counts.update(counts)
        alternative_total += len(completed)
    series.append(("Generated Queries", alternative_total, alternative_counts))

    plt.style.use("default")
    plt.rcParams.update({"font.size": 15})
    fig, ax = plt.subplots(figsize=(14, 4.2))
    boundaries = (len(codes) - len(graph_codes) - len(context_codes), len(codes) - len(context_codes))
    group_gap = 0.35
    positions = [index + group_gap * sum(index >= boundary for boundary in boundaries)
                 for index in range(len(codes))]
    width = 0.38
    for index, (label, total, counts) in enumerate(series):
        if not total:
            raise ValueError(f"No completed annotations for {label}")
        frequencies = [100 * counts[code] / total for code in codes]
        ax.bar(
            [position + (index - 0.5) * width for position in positions],
            frequencies,
            width=width,
            label=label,
            color=f"C{index}",
            hatch="//" if index else None,
            edgecolor="black",
            linewidth=0.4,
        )
        print(f"{label}: {total} completed annotations")
        for code, frequency in zip(codes, frequencies, strict=True):
            print(f"  {code}: {counts[code]}/{total} ({frequency:.1f}%)")

    for boundary in boundaries:
        separator = (positions[boundary - 1] + positions[boundary]) / 2
        ax.axvline(separator, color="0.3", linestyle="--", linewidth=1.2)
    ax.set_xticks(positions, codes, rotation=30, ha="right", fontsize=15)
    ax.set_ylim(0, 105)
    ax.set_xlim(-0.7, positions[-1] + 0.7)
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=100))
    ax.set_ylabel("Frequency of code (%)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.15, 0.99), fontsize=15)
    fig.tight_layout()
    results = directory.parent / "results"
    results.mkdir(exist_ok=True)
    for extension in ("png", "pdf"):
        output = results / f"code_frequencies.{extension}"
        fig.savefig(output, dpi=200)
        print(f"Saved {output}")
    plt.close(fig)


if __name__ == "__main__":
    main()
