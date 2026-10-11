# Query judgement startup investigation

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
