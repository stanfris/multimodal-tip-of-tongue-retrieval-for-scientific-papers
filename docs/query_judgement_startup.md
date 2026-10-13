# Query judgement startup investigation

## Current judge configuration

The judge now defaults to `google/gemma-4-12B-it`, with application batch size 16
and H200 vLLM concurrency 16. Standalone judgement enables engine multiprocessing
with the `spawn` start method. `submit_judgements_h200.sh` submits all pending
queries for up to 24 hours; `interactive_judgements.sh run` retains its two-query
diagnostic limit (append `stage.limit=null` to run all pending queries). The small
probe remains in-process. Older Gemma 3 logs and configurations below document
the investigation, not the current defaults. The separate multiprocessing
diagnostic deliberately still targets Gemma 3 for reproducing that issue.

Gemma 4 is recognized as multimodal by both inference backends and retains the
16-image engine allowance. The prior 15-image selection cap remains. Saved
judgements are skipped by query ID irrespective of model; newly written rows
include the Gemma 4 model ID. No existing output is deleted by this switch.

The Gemma 4 12B model uses `Gemma4UnifiedForConditionalGeneration`, registered in
[vLLM 0.26's model registry](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/model_executor/models/registry.py).
Configuration and adapter tests are run locally; actual GPU startup, batch
memory requirements and throughput remain unverified.

The later `AmbiguousGlobalPerLayerAttributeError` traceback establishes a
separate configuration incompatibility before worker creation: older vLLM reads
`hf_text_config.head_dim` globally, but Transformers 5.15+ makes that read an
error for Gemma 4's heterogeneous layers. The project now pins Transformers
5.14.0, which includes Gemma 4 Unified while preserving the older configuration
API. The lockfile also selects its compatible Tokenizers dependency. This fixes
the configuration read without setting
`allow_global_per_layer_attribute_access=True` or modifying layer dimensions.
The upstream multi-file fix is
[vllm#49797](https://github.com/vllm-project/vllm/pull/49797).

For an existing cluster environment:

```bash
uv pip install --python .venv/bin/python 'transformers==5.14.0'
```

Then rerun the ordinary judgement wrapper. The configuration API regression is
verified with the pinned packages in a temporary environment; full GPU inference
is still unverified locally. Batch size 16 and spawned multiprocessing remain
configured; neither causes this particular exception.

## Stall at `Using V2 Model Runner`

The supplied cluster excerpt stops after distributed initialization and runner
selection, before any weight-loading message. This narrows the investigation to
worker memory setup, runner imports/construction, or the transition to the model
loader. It does not establish a particular deadlock. The configuration dump's
CUDA graph sizes are planned settings, not evidence that graph capture has begun.
In [vLLM 0.21's worker source](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/v1/worker/gpu_worker.py),
runner selection precedes runner construction and `load_model`; profiling and
graph capture follow weight loading. The installed cluster version must be
checked before assuming its implementation is identical.

Try the interactive diagnostic wrapper from the repository root on the login
node:

```bash
scripts/query_generation/interactive_judgements.sh session
```

It requests one GPU using the repository's PBS H200 resource defaults. When the
shell opens, change to the repository using the command printed by the script.
Then run:

```bash
scripts/query_generation/interactive_judgements.sh probe 2>&1 | tee /tmp/judge-startup.log
scripts/query_generation/interactive_judgements.sh run
```

The probe reports package versions, checks a small CUDA allocation, times vLLM
import, loads Gemma, and generates 16 tokens without reading the dataset. It
defaults to the earlier runner (`VLLM_USE_V2_MODEL_RUNNER=0`), eager execution,
one sequence, a 32K context, and a one-image allowance. It keeps the engine in
process (`VLLM_ENABLE_V1_MULTIPROCESSING=0`) so Python stack traces every 120
seconds can show where startup is waiting. These switches are documented in
[vLLM's environment definitions](https://github.com/vllm-project/vllm/blob/v0.21.0/vllm/envs.py).
Stack traces diagnose waits; they do not terminate the process. Ctrl-C stops it.
The one-image probe tests basic engine startup, not the full multimodal workload.

Compare the V2 runner with all other probe settings held fixed:

```bash
VLLM_USE_V2_MODEL_RUNNER=1 scripts/query_generation/interactive_judgements.sh probe
```

If only V2 stalls, the runner path is implicated. If both stall, the stack
identifies whether to investigate CUDA/driver calls, optional kernel imports/JIT,
or shared-filesystem/cache access. If the probe succeeds but the pipeline stalls,
compare the larger image allowance, token budget, context, and multiprocessing.
To test multiprocessing alone:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=1 scripts/query_generation/interactive_judgements.sh probe
```

In that mode, periodic traces only show the parent, not the separate engine.

`run` executes locally on the allocated node with the H200 profile; it does not
submit another PBS job. Like the probe, it now defaults to an in-process engine
and prints periodic stack traces (every 120 seconds, configurable through
`VTT_DEBUG_STACK_INTERVAL`). It judges two pending queries with batch size one and
eager execution, retaining the pipeline's image cap and full context. Override
Hydra settings at the end, for example `run stage.limit=null` for all pending
judgements, or `run model.provider=transformers visual_query.judgement.provider=transformers`
to compare backends. Both provider overrides are necessary because the shared
visual-query defaults overwrite the stage's judge-provider selection. These runs
write normal judgement outputs and use normal resume behavior. Set `HF_HOME`,
`VTT_DATA_ROOT`, or `VTT_CACHE_ROOT` if the cluster uses different paths. The wrapper
preserves the original saved token location via `HF_TOKEN_PATH` before changing
the model cache; set it explicitly if your saved token uses a custom path.

Shell syntax, CLI help, local Hydra dry runs, and wrapper dispatch are checked
locally. GPU startup remains unverified because this workspace has no cluster GPU.

## Earlier dataset-path investigation

The code contains unnecessary filesystem work and a mismatch in image selection,
but the dominant delay cannot be measured from the available logs: the checked-in
query generation and judgement log files are empty. No GPU inference was launched
as part of this investigation.

## Comparison of the current paths

| Work | Generation | Standalone judgement |
| --- | --- | --- |
| Default stage model | `microsoft/phi-4` | `google/gemma-3-27b-it` |
| Default provider, including H200 | Transformers | Transformers |
| Inference input | Selected textual and visual cue descriptions, sent as text | Selected cues, query, full paper Markdown, figure listing, actual images |
| Image selection before this fix | First `selection.max_images` figures, default 15 | All extracted figures |
| Startup ordering | Load model, then load papers and clue sidecars | Load papers, plan pending queries, then load model |
| Dataset reads before this fix | Full paper traversal twice: main paper list and component loading | Full paper traversal once, then split filtering |
| H200 application batch size | 128 | 64 |
| Maximum generated tokens | 300 | 900 |
| H200 vLLM engine settings when explicitly selected | 128 sequences; 16,384 batched tokens | 64 sequences; 16,384 batched tokens; up to 16 images per prompt |

Generation normally disables inline judgement. When enabled, it uses the same
judge loader and prompt formatter as standalone judgement, but its paper records
already have the image cap applied. The standalone stage does not load generation
weights or regenerate clue descriptions. Image decoding and model image processing
happen inside the backend at inference time, after model initialization.

## Confirmed problems and fixes

1. **Unnecessary full-tree traversal.** Standalone judgement did not pass its
   split index to `read_preprocessed_papers`, even though that reader can directly
   locate PDF-indexed paper directories. It now passes the index and split. Legacy
   paper-ID indexes retain their recursive fallback. This reduces metadata I/O
   on large or network-mounted datasets. Generation still traverses the full tree
   twice; traversal alone does not establish why judgement is slower.
2. **Unbounded image input.** Standalone judgement now applies the same
   `visual_query.selection.max_images` setting as generation, default 15. Direct
   CLI callers can use `--max-images`. The cap applies to both the figure listing
   and attached images. Previously a paper could exceed vLLM's 16-image limit,
   or produce much larger image processing and inference workloads.
3. **Silent batch retries.** Both backends caught failed batches and recursively
   retried smaller batches without reporting the exception. Over-limit images,
   memory exhaustion, or long-context errors could cause repeated attempts before
   anything was saved. Retries now report the batch size and exception.
4. **Insufficient phase visibility.** Judgement now reports dataset loading,
   planning, model initialization, request preparation, and backend elapsed time.
   Request preparation reports total prompt characters and attached images.
   vLLM separately reports engine and chat processor initialization. Transformers
   reports the actual device placement, exposing CPU/disk offload.

Existing resume behavior remains: the judge model is not loaded if all queries
are already completed. Results are saved after a backend batch returns; there is
no streaming of individual results while that batch is still running.

## Likely expensive phases that need a cluster log to confirm

- **Before model ready, vLLM:** weight loading, multimodal memory profiling,
  compilation, and CUDA graph warmup. This is a larger multimodal model with an
  image allowance absent from generation. No explicit `max_model_len` or
  `enforce_eager` is configured. vLLM documents that eager execution skips
  compilation and graph capture, reducing startup at the expense of decode
  throughput: [vLLM optimization documentation](https://docs.vllm.ai/en/stable/configuration/optimization/#faster-startup).
- **Before model ready, Transformers:** download/cache access and weight loading.
  The model's parameter count implies roughly 54 GB of BF16 weights before
  runtime allocations. `device_map=auto` can offload on smaller GPUs. The H200
  launcher does not automatically switch the model to vLLM.
- **After batch preparation:** full-paper tokenization, image decoding/encoding,
  long-prompt prefill, and generation for the entire batch. A 64-paper judgement
  batch can attach up to 960 images after the fix; this remains much heavier than
  a generation batch of short text cues. Transformers padding also couples the
  batch to the longest input.

## Small diagnostic run

For a run intended to use H200 and vLLM, this command limits work to two pending
queries, submits one query at a time, reduces engine concurrency, and disables
compile/graph startup. Its Hydra overrides were checked locally without running
inference:

```bash
scripts/query_generation/judge_train_queries.sh \
  launcher=pbs_h200 model.provider=vllm \
  stage.limit=2 visual_query.judgement.batch_size=1 \
  launcher.vllm.max_num_seqs=8 +launcher.vllm.enforce_eager=true
```

This is a diagnostic configuration, not a measured throughput optimum. Compare
the reported model initialization time against backend batch time before adopting
it for a full collection. Smaller application batches shorten time to the first
saved result; they do not by themselves reduce engine warmup. Keep multimodal
memory profiling enabled until actual memory requirements have been measured.

## Verification

Regression tests cover PDF-indexed loading without recursive traversal, legacy
index compatibility, the default 15-image cap, matching figure listings and
attachments, full paper text preservation, and managed image-limit overrides.
The existing backend, batching, resume, and H200 configuration tests also pass.
GPU startup time and resulting cost savings remain unmeasured.

## Follow-up cluster log: vLLM 0.26.0, earlier runner

The follow-up log loads 11,000 papers in 147.2 seconds, schedules two judgements,
and initializes an eager engine. It stops after sampler initialization, still
before `Starting to load model`. There is no V2 runner selection message. In
[vLLM 0.26's earlier runner](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/worker/gpu_model_runner.py),
sampler construction and subsequent runner setup precede weight loading. This
rules out CUDA graph capture as the current waiting phase and weakens the
earlier V2-specific hypothesis. It does not prove which operation is blocked.

If the isolated probe succeeds, compare its settings with this run:

| Setting | Probe | Follow-up run |
| --- | --- | --- |
| Engine multiprocessing | Off | On (`EngineCore pid=...`) |
| Maximum context | 32,768 | 131,072 |
| Batched token budget | 2,048 | 16,384 |
| Image allowance | 1 | 16 |
| GPU memory fraction | 0.85 | 0.92 |

First retry with only multiprocessing changed, on the same GPU allocation:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 scripts/query_generation/interactive_judgements.sh run
```

The updated wrapper does this by default and adds periodic traces. If it still
stalls, compare smaller runtime budgets while preserving the 15 attached images:

```bash
VLLM_ENABLE_V1_MULTIPROCESSING=0 scripts/query_generation/interactive_judgements.sh run \
  ++launcher.vllm.max_model_len=32768 \
  launcher.vllm.max_num_batched_tokens=2048 \
  launcher.vllm.gpu_memory_utilization=0.85
```

The 32K context is a diagnostic limit: oversized full-paper/image prompts may
fail at inference time. This does not truncate paper text or reduce attached
images. Investigate the printed stack before adopting any throughput settings.

## Multiprocessing root-cause audit

The successful in-process run establishes a workaround, not the exact blocking
operation. A more specific candidate is fork safety during multimodal CPU
initialization. The repository uses the same `VLLMBackend` constructor for query
generation and judgement. Dataset loading contains no explicit Torch/CUDA
initialization. Loading papers before loading the judge is therefore not, by
itself, evidence of a CUDA-before-fork error.

In vLLM 0.26, `LLMEngine` creates its renderer before starting EngineCore. For
multimodal models the renderer constructs processors inside
`set_default_torch_num_threads`. The engine's earlier runner later constructs
`MultiModalBudget`, again changing Torch thread settings, after the sampler log
but before the weight-loading log. A fork can inherit native runtime/lock state;
spawn starts a new interpreter. This is consistent with the logs, but a stack
trace or controlled start-method comparison is required to confirm it.

Sources: [engine initialization](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/v1/engine/llm_engine.py),
[renderer](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/renderers/base.py),
[multimodal budget](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/multimodal/encoder_budget.py),
[process context selection](https://github.com/vllm-project/vllm/blob/v0.26.0/vllm/utils/system_utils.py).

Phi-4 and Qwen text generation skip multimodal processor initialization. Qwen-VL
uses a different processor/model path, so it is not proof that Gemma has the same
startup behavior. MinerU uses a separate environment. Previous successful logs
are needed to compare the actual provider, package versions and process method.

The original probe also initializes CUDA before importing vLLM. If multiprocessing
is enabled for that probe, vLLM automatically forces spawn; its successful startup
would not establish that fork works. The new matched diagnostic avoids this:

```bash
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -u scripts/query_generation/diagnose_judge_multiprocessing.py --mode spawn
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" \
  .venv/bin/python -u scripts/query_generation/diagnose_judge_multiprocessing.py --mode fork
```

Run sequentially on the same otherwise idle GPU, with the same `HF_HOME` used by
the pipeline. The helper reproduces engine budgets and arms traces in parent and
children. A startup failure remains visible as an exception; a hang emits traces
every two minutes. Stop a hung run before starting another. If spawn works and
fork hangs in Torch thread/processor initialization, keep multiprocessing enabled
and set `VLLM_WORKER_MULTIPROC_METHOD=spawn` instead of disabling EngineCore
multiprocessing entirely. GPU confirmation remains outstanding.
