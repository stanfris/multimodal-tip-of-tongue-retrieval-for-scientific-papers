from __future__ import annotations

import json
from pathlib import Path

import pytest

from inference import GenerationResult
from queries import judge_queries as stage
from preprocessing import preprocessed


@pytest.mark.parametrize("pdf_index", [True, False])
def test_judgement_reads_indexed_split_and_caps_prompt_and_images(tmp_path: Path, monkeypatch, capsys, pdf_index):
    dataset = tmp_path / "preprocessed"
    paper_dir = dataset / "papers" / "ACL" / "ACL_train"
    images_dir = paper_dir / "images"
    images_dir.mkdir(parents=True)
    figures = []
    for index in range(18):
        image_path = images_dir / f"figure-{index:02}.png"
        image_path.touch()  # The fake backend checks paths without decoding images.
        figures.append({"figure_id": f"figure-{index:02}", "image_relpath": str(image_path.relative_to(paper_dir))})
    (paper_dir / "paper.json").write_text(json.dumps({"paper_id": "ACL_train", "figures": figures}))
    (paper_dir / "markdown.md").write_text("Full paper text")
    split_index = tmp_path / "split.json"
    split_index.write_text(json.dumps({"train": ["ACL/train.pdf" if pdf_index else "ACL_train"], "test": []}))
    if pdf_index:
        monkeypatch.setattr(preprocessed, "iter_preprocessed_paper_dirs",
                            lambda *a, **kw: pytest.fail("Judgement must not recursively scan a PDF-indexed corpus"))
    query_dir = tmp_path / "queries" / "visual_and_text"
    query_dir.mkdir(parents=True)
    (query_dir / "queries.jsonl").write_text(json.dumps({"query_id": "q", "query": "find this",
                                                       "relevant_ids": ["ACL_train"]}) + "\n")
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Judge {query}\n{paper_text}\n{figures}")
    calls = []

    class Backend:
        def generate(self, requests):
            calls.extend(requests)
            return [GenerationResult(request.request_id, '{"grounded": true}') for request in requests]

    monkeypatch.setattr(stage, "load_query_judge", lambda config: Backend())
    args = stage.build_parser().parse_args(["--dataset", str(dataset), "--split-index", str(split_index),
                                           "--input-dir", str(query_dir.parent), "--prompt", str(prompt)])
    assert stage.main(args) == 0
    assert len(calls) == 1
    assert len(calls[0].images) == 15
    assert "figure-14" in calls[0].prompt
    assert "figure-15" not in calls[0].prompt
    assert "Full paper text" in calls[0].prompt
    log = capsys.readouterr().out
    assert "Judgement papers ready: 1 papers" in log
    assert "Judgement model initialization took" in log
    assert "starting backend image processing and inference" in log


def test_managed_judgement_uses_generation_image_limit(tmp_path, monkeypatch):
    config = stage.QueryGenerationConfig(dataset=tmp_path, visual_interpretations=None,
        textual_interpretations=None, clues_dir=tmp_path, output_dir=tmp_path, max_images=3)
    paper = {"paper_id": "ACL_paper", "source_paper_dataset": "ACL", "split": "train",
             "figures": [{"figure_id": str(i)} for i in range(20)]}
    monkeypatch.setattr(stage, "load_managed_judgement_args", lambda args: (args, config))
    monkeypatch.setattr(stage, "read_preprocessed_papers", lambda *a, **kw: [paper])
    query_dir = tmp_path / "collections" / "ACL" / config.collection_id / "visual_and_text"
    query_dir.mkdir(parents=True)
    (query_dir / "queries.jsonl").write_text(json.dumps({"query_id": "q", "relevant_ids": ["ACL_paper"]}))
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Judge")
    from dataclasses import replace
    config = replace(config, judgement_prompt=prompt)
    seen = []
    monkeypatch.setattr(stage, "format_judgement_prompt",
                        lambda template, mode, paper, selected, query: seen.append(paper) or "Judge")

    class Backend:
        def generate(self, requests):
            return [GenerationResult(request.request_id, '{"grounded": true}') for request in requests]

    monkeypatch.setattr(stage, "load_query_judge", lambda config: Backend())
    args = stage.build_parser().parse_args(["--settings", str(tmp_path / "settings.yaml"),
        "--dataset", str(tmp_path), "--input-dir", str(tmp_path / "collections")])
    assert stage.main(args) == 0
    assert len(seen[0]["figures"]) == 3


@pytest.mark.parametrize("save_failure", [False, True])
def test_judgement_consumes_prefetched_batches_and_closes_stream(tmp_path, monkeypatch, save_failure):
    paper = {"paper_id": "paper", "figures": []}
    monkeypatch.setattr(stage, "read_preprocessed_papers", lambda *a, **kw: [paper])
    monkeypatch.setattr(stage, "filter_papers_by_split", lambda papers, **kw: papers)
    monkeypatch.setattr(stage, "format_judgement_prompt", lambda *args: "Judge")
    query_dir = tmp_path / "visual_and_text"
    query_dir.mkdir()
    rows = [{"query_id": name, "query": name, "relevant_ids": ["paper"]}
            for name in ["done", "bad", "good", "last"]]
    (query_dir / "queries.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    output = query_dir / "query_judgements.jsonl"
    output.write_text('{"query_id": "done"}\n')
    prompt = tmp_path / "prompt.txt"
    prompt.write_text("Judge")
    seen = []
    closed = []

    class Backend:
        def generate(self, requests):
            pytest.fail("Standalone judgement should use the prefetch API")

        def generate_batches(self, batches):
            try:
                for requests in batches:
                    seen.extend(request.request_id for request in requests)
                    yield [GenerationResult(request.request_id, '{"overall_grounded": true}',
                                            error=ValueError("bad input") if request.request_id == "bad" else None)
                           for request in reversed(requests)]
            finally:
                closed.append(True)

    monkeypatch.setattr(stage, "load_query_judge", lambda config: Backend())
    if save_failure:
        def fail_save(*args):
            raise OSError("disk failure")
        monkeypatch.setattr(stage, "append_judgement_row", fail_save)
    args = stage.build_parser().parse_args(["--input-dir", str(tmp_path), "--dataset", str(tmp_path),
                                           "--prompt", str(prompt), "--batch-size", "2"])
    if save_failure:
        with pytest.raises(OSError, match="disk failure"):
            stage.main(args)
        assert seen == ["bad", "good"]
    else:
        assert stage.main(args) == 0
        assert seen == ["bad", "good", "last"]
        saved = [json.loads(line) for line in output.read_text().splitlines()]
        assert [row["query_id"] for row in saved] == ["done", "good", "last"]
    assert closed == [True]
