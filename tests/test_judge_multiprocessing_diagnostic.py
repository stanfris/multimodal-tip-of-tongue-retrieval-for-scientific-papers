from __future__ import annotations

import faulthandler
import importlib.metadata
import os
from pathlib import Path
import runpy
import sys
import types

import pytest


@pytest.mark.parametrize("mode", ["fork", "spawn", "off"])
def test_matched_startup_diagnostic_selects_process_mode_before_backend_import(monkeypatch, mode):
    monkeypatch.delenv("VTT_JUDGE_MP_TRACE", raising=False)
    # main sets these through os.environ; ensure monkeypatch restores them.
    for key in ("VTT_JUDGE_MP_TRACE", "VLLM_ENABLE_V1_MULTIPROCESSING",
                "VLLM_WORKER_MULTIPROC_METHOD"):
        monkeypatch.setenv(key, "")
    monkeypatch.setattr(faulthandler, "enable", lambda: None)
    monkeypatch.setattr(faulthandler, "dump_traceback_later", lambda *a, **kw: None)
    monkeypatch.setattr(faulthandler, "cancel_dump_traceback_later", lambda: None)
    if hasattr(os, "register_at_fork"):
        monkeypatch.setattr(os, "register_at_fork", lambda **kw: None)
    monkeypatch.setattr(importlib.metadata, "version", lambda package: "test-version")
    calls = []

    class Backend:
        def __init__(self, model, runtime, image):
            calls.append((os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"],
                          os.environ["VLLM_WORKER_MULTIPROC_METHOD"], runtime))

    fake = types.ModuleType("inference.vllm")
    fake.VLLMBackend = Backend
    monkeypatch.setitem(sys.modules, "inference.vllm", fake)
    path = Path(__file__).resolve().parents[1] / "scripts/query_generation/diagnose_judge_multiprocessing.py"
    monkeypatch.setattr(sys, "argv", [str(path), "--mode", mode])
    runpy.run_path(str(path), run_name="__main__")
    assert len(calls) == 1
    mp, method, runtime = calls[0]
    assert mp == ("0" if mode == "off" else "1")
    assert method == ("fork" if mode == "off" else mode)
    assert runtime["limit_mm_per_prompt"] == {"image": 16}
    assert runtime["max_num_batched_tokens"] == 16384
    assert "max_model_len" not in runtime
