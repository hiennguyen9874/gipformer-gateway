"""Real QuanEngine behind the worker thread and scheduler: load, warm-up, concurrent lanes give the M0 baseline text.

Needs a CUDA GPU and staged assets (no network). Pick a free GPU: CUDA_VISIBLE_DEVICES=<id> uv run pytest tests/gpu
"""

import asyncio
import json
import os
import time
from pathlib import Path

import pytest
import soundfile as sf
import torch

from app.engine import EngineState, EngineWorker, QuanEngine
from app.observability import Metrics
from app.scheduler import JobKind, Scheduler
from tests.fake_engine import config

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("GIPFORMER_MODELS_DIR", ROOT / "models"))
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (MODELS / "embeddinggemma-2").is_dir(), reason="assets not staged"),
]


def test_real_engine_serves_concurrent_lanes_with_baseline_text():
    baseline = json.loads((ROOT / "docs/baseline/offline.json").read_text())["results"]
    clips = {name: sf.read(ROOT / "tests/fixtures" / name, dtype="float32")[0] for name in ("vi_a.wav", "vi_b.wav")}

    async def main():
        cfg = config(models_dir=MODELS)
        metrics = Metrics()
        worker = EngineWorker(QuanEngine(MODELS, "cuda", cfg.max_chunk_seconds), cfg.infer_hang_seconds, metrics)
        scheduler = Scheduler(worker, cfg, metrics)
        worker.start()
        scheduler.start()
        deadline = time.monotonic() + 300
        while worker.state == EngineState.LOADING:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.1)
        assert worker.state == EngineState.READY

        async def lane_run(kind, name):
            async with scheduler.open_lane(kind) as lane:
                return [(await (await lane.submit(clips[name]))).text for _ in range(3)]

        try:
            return await asyncio.gather(lane_run(JobKind.FILE, "vi_a.wav"), lane_run(JobKind.LIVE, "vi_b.wav"),
                                        lane_run(JobKind.FILE, "vi_b.wav"))
        finally:
            assert await scheduler.close(30)
            assert await asyncio.to_thread(worker.stop, 30)

    a, b_live, b_file = asyncio.run(main())
    assert a == [baseline["vi_a.wav:full"]["text"]] * 3
    assert b_live == b_file == [baseline["vi_b.wav:full"]["text"]] * 3


def test_chunk_result_path_is_the_greedy_decode():
    """The engine's best path equals the package's `greedy_decode` and its text is the baseline text."""
    from quan_transcribe.model import greedy_decode

    baseline = json.loads((ROOT / "docs/baseline/offline.json").read_text())["results"]
    engine = QuanEngine(MODELS, "cuda", warmup_seconds=1.0)
    engine.load()
    pieces = engine.pieces()
    assert len(pieces) == engine._asr.model.cfg.vocab_size and pieces[:2] == ("", "")
    for name in ("vi_a.wav", "vi_b.wav", "silence_2s.wav", "short_200ms.wav"):
        wav = sf.read(ROOT / "tests/fixtures" / name, dtype="float32")[0]
        result = engine.transcribe(wav)
        assert result.text == baseline[f"{name}:full"]["text"], name
        logits, lens = engine._asr._forward([wav], False, None)
        assert list(result.units) == greedy_decode(logits, lens)[0], name
        assert result.frames == int(lens[0]) == -(-(len(wav) - 160) // 640), name
        assert all(f <= l for f, l in zip(result.first, result.last))
        assert all(l < f for l, f in zip(result.last, result.first[1:]))
        assert not result.last or result.last[-1] < result.frames
