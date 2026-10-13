"""Compare fork, spawn and in-process Gemma startup without initializing CUDA first.

Run each mode in a fresh process on the same GPU allocation. Unlike the small
probe, this reproduces the pipeline's engine budgets and import ordering.
"""
from __future__ import annotations

import argparse
import faulthandler
import os
import time


def arm_trace() -> None:
    interval = int(os.environ.get("VTT_DEBUG_STACK_INTERVAL", "120"))
    if interval < 1:
        raise ValueError("VTT_DEBUG_STACK_INTERVAL must be positive")
    faulthandler.enable()
    faulthandler.dump_traceback_later(interval, repeat=True)
    print(f"Startup stack traces armed: pid={os.getpid()}, interval={interval}s", flush=True)


# Spawn imports this script as __mp_main__; fork uses the registered callback.
if os.environ.get("VTT_JUDGE_MP_TRACE") == "1":
    arm_trace()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("fork", "spawn", "off"))
    args = parser.parse_args()
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0" if args.mode == "off" else "1"
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "fork" if args.mode == "off" else args.mode
    os.environ["VTT_JUDGE_MP_TRACE"] = "1"
    for key, value in {
        "VLLM_USE_V2_MODEL_RUNNER": "0", "VLLM_USE_DEEP_GEMM": "0",
        "VLLM_USE_FLASHINFER_SAMPLER": "0", "VLLM_LOGGING_LEVEL": "DEBUG",
        "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
        "OPENBLAS_NUM_THREADS": "4", "NUMEXPR_NUM_THREADS": "4",
    }.items():
        os.environ.setdefault(key, value)
    arm_trace()
    if hasattr(os, "register_at_fork"):
        os.register_at_fork(after_in_child=arm_trace)
    try:
        # Do not import torch, check CUDA, or allocate tensors before LLM starts:
        # vLLM automatically changes fork to spawn after CUDA initialization.
        from importlib.metadata import version
        print(f"vLLM={version('vllm')}, mode={args.mode}", flush=True)
        for key in ("VLLM_USE_V2_MODEL_RUNNER", "VLLM_WORKER_MULTIPROC_METHOD",
                    "VLLM_ENABLE_V1_MULTIPROCESSING", "OMP_NUM_THREADS", "HF_HOME"):
            print(f"{key}={os.environ.get(key, '(unset)')}", flush=True)
        from inference.vllm import VLLMBackend
        started = time.monotonic()
        backend = VLLMBackend("google/gemma-3-27b-it", {
            "dtype": "bfloat16", "tensor_parallel_size": 1,
            "gpu_memory_utilization": 0.92, "max_num_seqs": 1,
            "max_num_batched_tokens": 16384, "kv_cache_dtype": "auto",
            "enable_prefix_caching": True, "enforce_eager": True,
            "mm_processor_cache_gb": 1, "limit_mm_per_prompt": {"image": 16},
        }, {})
        print(f"Matched pipeline engine ready in {time.monotonic() - started:.1f}s", flush=True)
        # Hold a reference through the end of initialization.
        del backend
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
