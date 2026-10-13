from __future__ import annotations

from threading import Barrier, Event, get_ident
from types import SimpleNamespace

import pytest

from inference import GenerationRequest
from inference.vllm import VLLMBackend


def backend_with_images(monkeypatch, generate, workers=2):
    from PIL import Image

    opened = []

    class RGB:
        closed = False

        def close(self):
            self.closed = True

    class Source:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def convert(self, mode):
            image = RGB()
            opened.append(image)
            return image

    monkeypatch.setattr(Image, 'open', lambda path: Source())
    backend = VLLMBackend.__new__(VLLMBackend)
    backend.image_load_workers = workers
    backend.processor = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: 'chat')
    backend.sampling_params = lambda **kwargs: kwargs
    backend.llm = SimpleNamespace(generate=generate)
    return backend, opened


def outputs(prompts):
    return [SimpleNamespace(outputs=[SimpleNamespace(text=str(index), token_ids=[1])])
            for index, _ in enumerate(prompts)]


def test_prefetch_overlaps_inference_is_bounded_and_closes_early(monkeypatch):
    next_ready = Event()
    requested = []
    engine_thread = get_ident()

    def generate(prompts, **kwargs):
        assert get_ident() == engine_thread
        assert next_ready.wait(3), 'Next window must prepare during inference'
        assert requested == ['first', 'second']
        assert all(not image.closed for image in opened)
        return outputs(prompts)

    backend, opened = backend_with_images(monkeypatch, generate)
    load = backend._load_request_images

    def load_images(request):
        result = load(request)
        if request.request_id == 'second':
            next_ready.set()
        return result

    monkeypatch.setattr(backend, '_load_request_images', load_images)

    def batches():
        for name in ['first', 'second', 'third']:
            requested.append(name)
            yield [GenerationRequest(name, name, images=('image.png',))]

    stream = backend.generate_batches(batches())
    assert next(stream)[0].request_id == 'first'
    assert opened[0].closed
    stream.close()
    assert requested == ['first', 'second']
    assert len(opened) == 2 and all(image.closed for image in opened)


def test_image_loads_use_multiple_workers(monkeypatch):
    from PIL import Image

    backend, opened = backend_with_images(monkeypatch, lambda prompts, **kwargs: outputs(prompts))
    open_image = Image.open
    barrier = Barrier(2)

    def synchronized_open(path):
        barrier.wait(timeout=3)
        return open_image(path)

    monkeypatch.setattr(Image, 'open', synchronized_open)
    results = backend.generate([GenerationRequest('a', 'a', images=('a.png',)),
                                GenerationRequest('b', 'b', images=('b.png',))])
    assert [result.request_id for result in results] == ['a', 'b']
    assert all(result.error is None for result in results)
    assert all(image.closed for image in opened)


def test_bad_image_closes_partial_request_and_preserves_good_results(monkeypatch):
    from PIL import Image

    calls = []

    def generate(prompts, **kwargs):
        calls.append(len(prompts))
        return outputs(prompts)

    backend, opened = backend_with_images(monkeypatch, generate)
    open_image = Image.open

    def sometimes_bad(path):
        if str(path) == 'bad.png':
            raise OSError('corrupt image')
        return open_image(path)

    monkeypatch.setattr(Image, 'open', sometimes_bad)
    results = list(backend.generate_batches([
        [GenerationRequest('bad', 'bad', images=('first.png', 'bad.png')),
         GenerationRequest('good', 'good', images=('good.png',))],
        [GenerationRequest('next', 'next')],
    ]))
    assert [result.request_id for batch in results for result in batch] == ['bad', 'good', 'next']
    assert isinstance(results[0][0].error, OSError)
    assert results[0][1].error is None and results[1][0].error is None
    assert calls == [1, 1]
    assert len(opened) == 2 and all(image.closed for image in opened)


def test_engine_retries_reuse_decoded_images(monkeypatch):
    calls = []

    def generate(prompts, **kwargs):
        calls.append(len(prompts))
        if len(prompts) > 1:
            raise RuntimeError('batch too large')
        assert not prompts[0]['multi_modal_data']['image'].closed
        return outputs(prompts)

    backend, opened = backend_with_images(monkeypatch, generate)
    results = list(backend.generate_batches([[GenerationRequest('a', 'a', images=('a.png',)),
                                              GenerationRequest('b', 'b', images=('b.png',))]]))[0]
    assert [result.request_id for result in results] == ['a', 'b']
    assert all(result.error is None for result in results)
    assert calls == [2, 1, 1]
    assert len(opened) == 2 and all(image.closed for image in opened)


def test_engine_failure_closes_current_and_prefetched_images(monkeypatch):
    next_ready = Event()

    def generate(prompts, **kwargs):
        assert next_ready.wait(3)
        raise KeyboardInterrupt()

    backend, opened = backend_with_images(monkeypatch, generate)
    load = backend._load_request_images

    def load_images(request):
        result = load(request)
        if request.request_id == 'b':
            next_ready.set()
        return result

    monkeypatch.setattr(backend, '_load_request_images', load_images)
    stream = backend.generate_batches([[GenerationRequest('a', 'a', images=('a.png',))],
                                       [GenerationRequest('b', 'b', images=('b.png',))]])
    with pytest.raises(KeyboardInterrupt):
        next(stream)
    assert len(opened) == 2 and all(image.closed for image in opened)
