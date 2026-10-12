"""Isolate Gemma/vLLM startup from dataset I/O, with periodic stack dumps."""
from __future__ import annotations

import argparse
import faulthandler
from importlib.metadata import PackageNotFoundError, version
import os
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="google/gemma-3-27b-it")
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--stack-interval", type=int, default=120)
    args = parser.parse_args()
    if args.stack_interval < 1 or args.max_model_len < 1:
        parser.error("stack interval and model length must be positive")

    # These must be set before importing vLLM. Explicit values allow A/B runs.
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    faulthandler.enable()
    faulthandler.dump_traceback_later(args.stack_interval, repeat=True)
    try:
        print(f"Python: {sys.executable}", flush=True)
        for package in ("torch", "vllm", "transformers", "flashinfer-python"):
            try:
                print(f"{package}: {version(package)}", flush=True)
            except PackageNotFoundError:
                print(f"{package}: not installed", flush=True)
        for key in ("VLLM_USE_V2_MODEL_RUNNER", "VLLM_ENABLE_V1_MULTIPROCESSING",
                    "HF_HOME", "XDG_CACHE_HOME"):
            print(f"{key}={os.environ.get(key, '(unset)')}", flush=True)

        print("Checking CUDA and allocating a tiny tensor...", flush=True)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Run inside the allocated GPU session.")
        print(f"GPU: {torch.cuda.get_device_name(0)}; torch CUDA: {torch.version.cuda}", flush=True)
        tensor = torch.ones(1, device="cuda")
        torch.cuda.synchronize()
        del tensor

        print("Importing vLLM...", flush=True)
        started = time.monotonic()
        from vllm import LLM, SamplingParams
        print(f"vLLM import completed in {time.monotonic() - started:.1f}s", flush=True)
        print("Constructing the engine (one sequence, one image allowance, eager)...", flush=True)
        started = time.monotonic()
        llm = LLM(
            model=args.model, dtype="bfloat16", tensor_parallel_size=1,
            enforce_eager=True, max_model_len=args.max_model_len,
            max_num_seqs=1, max_num_batched_tokens=2048,
            gpu_memory_utilization=0.85, limit_mm_per_prompt={"image": 1},
        )
        print(f"Engine ready in {time.monotonic() - started:.1f}s", flush=True)
        result = llm.generate("Say hello.", SamplingParams(max_tokens=16, temperature=0))
        print(f"Smoke response: {result[0].outputs[0].text!r}", flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
