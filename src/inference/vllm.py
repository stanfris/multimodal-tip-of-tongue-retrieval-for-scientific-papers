"""Offline vLLM backend. A request window enters one engine call."""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

from inference.base import GenerationRequest, GenerationResult, ordered_results
from inference.formatting import chat_prompt, processor_image_kwargs


@dataclass
class _PreparedRequest:
    request: GenerationRequest
    prompt: dict[str, Any] | None = None
    params: Any = None
    images: tuple[Any, ...] = ()
    error: Exception | None = None

    def close(self) -> None:
        for image in self.images:
            image.close()


class VLLMBackend:
    def __init__(self, model: str, runtime: dict[str, Any], image: dict[str, Any]) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RuntimeError("vLLM backend requested but vllm is not installed. Install it in a supported GPU environment.") from exc
        from transformers import AutoProcessor, AutoTokenizer

        self.sampling_params = SamplingParams
        kwargs = {key: value for key, value in runtime.items() if value is not None}
        # Adapter-only option: never pass it to vLLM's engine constructor.
        self.image_load_workers = int(kwargs.pop("image_load_workers", 4))
        if self.image_load_workers < 1:
            raise ValueError("image_load_workers must be at least 1")
        if any(name in model.lower() for name in ("vl", "gemma-3", "gemma-4")) and "limit_mm_per_prompt" not in kwargs:
            kwargs["limit_mm_per_prompt"] = {"image": 16}
        mm_kwargs = processor_image_kwargs(model, image)
        if mm_kwargs:
            kwargs["mm_processor_kwargs"] = {**kwargs.get("mm_processor_kwargs", {}), **mm_kwargs}
        print(f"Initializing vLLM engine for {model}; loading weights, profiling and warmup.", flush=True)
        started = time.monotonic()
        self.llm = LLM(model=model, **kwargs)
        print(f"vLLM engine ready in {time.monotonic() - started:.1f}s; loading chat processor.", flush=True)
        started = time.monotonic()
        try:
            self.processor = AutoProcessor.from_pretrained(model)
            if not hasattr(self.processor, "apply_chat_template"):
                self.processor = AutoTokenizer.from_pretrained(model)
        except (OSError, ValueError):
            self.processor = AutoTokenizer.from_pretrained(model)
        print(f"vLLM chat processor ready in {time.monotonic() - started:.1f}s", flush=True)

    def generate(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        if not requests:
            return []
        try:
            return self._generate_batch(requests)
        except Exception as exc:
            if len(requests) == 1:
                return [GenerationResult(requests[0].request_id, error=exc)]
            print(f"vLLM batch of {len(requests)} requests failed; retrying as smaller batches: {exc}",
                  flush=True)
            middle = len(requests) // 2
            return self.generate(requests[:middle]) + self.generate(requests[middle:])

    def _generate_batch(self, requests: list[GenerationRequest]) -> list[GenerationResult]:
        with ThreadPoolExecutor(max_workers=self.image_load_workers) as workers:
            prepared = self._prepare_batch(requests, workers)
        try:
            return self._generate_prepared(prepared)
        finally:
            for item in prepared:
                item.close()

    def generate_batches(self, batches: Iterable[list[GenerationRequest]]) -> Iterator[list[GenerationResult]]:
        """Prepare at most one future window while the engine runs the current one.

        The caller must close this iterator when stopping early. Engine calls stay
        on the consuming thread; only chat formatting and image decoding run ahead.
        """
        iterator = iter(batches)
        pending = None
        with (
            ThreadPoolExecutor(max_workers=self.image_load_workers) as workers,
            ThreadPoolExecutor(max_workers=1) as preparation,
        ):
            def prepare_next() -> list[_PreparedRequest] | None:
                requests = next(iterator, None)
                return None if requests is None else self._prepare_batch(requests, workers)

            try:
                pending = preparation.submit(prepare_next)
                while True:
                    prepared = pending.result()
                    pending = None
                    if prepared is None:
                        break
                    try:
                        pending = preparation.submit(prepare_next)
                        results = self._generate_prepared(prepared)
                    finally:
                        for item in prepared:
                            item.close()
                    yield results
            finally:
                if pending is not None and not pending.cancel():
                    # Release prefetched images even if saving results failed or
                    # the consumer closed the iterator before its next window.
                    prefetched = pending.result()
                    for item in prefetched or []:
                        item.close()

    def _prepare_batch(self, requests: list[GenerationRequest], workers: ThreadPoolExecutor) -> list[_PreparedRequest]:
        # Keep tokenizer/chat processor use serial; parallelize only image I/O.
        prepared = list(workers.map(self._load_request_images, requests))
        for item in prepared:
            if item.error is not None:
                continue
            try:
                item.prompt = {"prompt": chat_prompt(self.processor, item.request.prompt,
                               len(item.request.images), thinking=item.request.thinking)}
                if item.images:
                    item.prompt["multi_modal_data"] = {"image": item.images[0] if len(item.images) == 1 else list(item.images)}
                item.params = self.sampling_params(max_tokens=item.request.max_tokens,
                                                  temperature=item.request.temperature)
            except Exception as exc:
                item.error = exc
        return prepared

    @staticmethod
    def _load_request_images(request: GenerationRequest) -> _PreparedRequest:
        from PIL import Image

        images = []
        try:
            for path in request.images:
                with Image.open(Path(path)) as source:
                    images.append(source.convert("RGB"))
            return _PreparedRequest(request, images=tuple(images))
        except Exception as exc:
            for image in images:
                image.close()
            return _PreparedRequest(request, error=exc)

    def _generate_prepared(self, prepared: list[_PreparedRequest]) -> list[GenerationResult]:
        valid = [item for item in prepared if item.error is None]
        results = [GenerationResult(item.request.request_id, error=item.error)
                   for item in prepared if item.error is not None]
        if valid:
            results.extend(self._infer_prepared(valid))
        return ordered_results([item.request for item in prepared], results)

    def _infer_prepared(self, prepared: list[_PreparedRequest]) -> list[GenerationResult]:
        try:
            outputs = self.llm.generate([item.prompt for item in prepared],
                                       sampling_params=[item.params for item in prepared], use_tqdm=False)
            return [GenerationResult(request_id=item.request.request_id,
                                     text=output.outputs[0].text.strip(),
                                     generated_tokens=len(output.outputs[0].token_ids))
                    for item, output in zip(prepared, outputs, strict=True)]
        except Exception as exc:
            if len(prepared) == 1:
                return [GenerationResult(prepared[0].request.request_id, error=exc)]
            print(f"vLLM batch of {len(prepared)} requests failed; retrying as smaller batches: {exc}", flush=True)
            middle = len(prepared) // 2
            return self._infer_prepared(prepared[:middle]) + self._infer_prepared(prepared[middle:])
