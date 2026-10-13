from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / "scripts/query_generation/interactive_judgements.sh"


@pytest.mark.parametrize("mode", ["session", "probe", "run"])
def test_interactive_dispatch(tmp_path: Path, mode: str) -> None:
    capture = tmp_path / "capture.json"
    executable = tmp_path / "qsub"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, sys\n"
        "from pathlib import Path\n"
        "Path(os.environ['CAPTURE']).write_text(json.dumps({"
        "'args': sys.argv[1:], 'runner': os.getenv('VLLM_USE_V2_MODEL_RUNNER'),"
        "'mp': os.getenv('VLLM_ENABLE_V1_MULTIPROCESSING')}))\n"
    )
    executable.chmod(0o755)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}",
           "VTT_PYTHON": str(executable), "CAPTURE": str(capture),
           "VTT_PBS_PROJECT": "test_project", "VTT_PBS_QUEUE": "test_queue"}
    for key in ("VLLM_USE_V2_MODEL_RUNNER", "VLLM_ENABLE_V1_MULTIPROCESSING"):
        env.pop(key, None)
    subprocess.run([str(WRAPPER), mode, "extra=value"], env=env, check=True)
    actual = json.loads(capture.read_text())
    args = actual["args"]
    assert args[-1] == "extra=value"
    if mode == "session":
        assert "-I" in args
        assert args[args.index("-P") + 1] == "test_project"
        assert args[args.index("-q") + 1] == "test_queue"
        assert "select=1:ncpus=16:mem=160gb:ngpus=1" in args
    elif mode == "probe":
        assert args[1] == str(ROOT / "scripts/query_generation/probe_judge_startup.py")
        assert actual["runner"] == actual["mp"] == "0"
    else:
        assert "-c" in args
        assert 'runpy.run_module("hydra_entry", run_name="__main__")' in args[args.index("-c") + 1]
        assert "launcher.kind=local" in args
        assert "visual_query.judgement.provider=vllm" in args
        assert "stage.limit=2" in args
        assert "++launcher.vllm.enforce_eager=true" in args
        assert "visual_query.judgement.batch_size=16" in args
        assert "launcher.vllm.max_num_seqs=16" in args
        assert "++launcher.environment.VLLM_WORKER_MULTIPROC_METHOD=spawn" in args
        assert actual["runner"] == "0"
        assert actual["mp"] == "1"
