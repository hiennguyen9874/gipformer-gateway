"""M2 pipeline on the real engine: FFmpeg decode -> quiet splitter -> scheduler lane -> QUAN, chunk texts joined
with one space (empty chunks add nothing).

- WAV at 16 kHz float decodes bit-exactly, so the transcript must equal the M0 baseline.
- Lossy/resampled containers and the long fixture (splitter differs from the reference `split_quiet`) are checked
  against the FLEURS reference with a WER bound, not text parity.
- The smallest chunk the policy forwards (MIN_AUDIO_SAMPLES + 1) runs through the real frontend without error.

Needs a CUDA GPU, staged assets and ffmpeg: CUDA_VISIBLE_DEVICES=<id> uv run pytest tests/gpu
"""

import asyncio
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import pytest
import torch

from app.audio import AudioIngress
from app.chunking import MIN_AUDIO_SAMPLES, split
from app.engine import EngineState, EngineWorker, QuanEngine
from app.observability import Metrics
from app.scheduler import JobKind, Scheduler
from scripts.smoke_inference import wer
from tests.fake_engine import config

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("GIPFORMER_MODELS_DIR", ROOT / "models"))
FIXTURES = ROOT / "tests/fixtures"
LONG = ROOT / "fixtures-large/vi_long.wav"
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (MODELS / "embeddinggemma-2").is_dir(), reason="assets not staged"),
]


def encode(src: Path, dst: Path, *args: str) -> Path:
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(src), *args, str(dst)],
                   check=True)
    return dst


def test_decoded_and_chunked_audio_transcribes_like_the_baseline(tmp_path):
    baseline = json.loads((ROOT / "docs/baseline/offline.json").read_text())["results"]
    refs = {k: v["reference"] for k, v in json.loads((FIXTURES / "manifest.json").read_text())["files"].items()}
    inputs = {
        "vi_a.wav": FIXTURES / "vi_a.wav",
        "vi_a.mp3": encode(FIXTURES / "vi_a.wav", tmp_path / "a.mp3", "-ar", "44100", "-c:a", "libmp3lame"),
        "vi_b.webm": encode(FIXTURES / "vi_b.wav", tmp_path / "b.webm", "-ac", "2", "-c:a", "libopus"),
        "silence_2s.wav": FIXTURES / "silence_2s.wav",
    }
    if LONG.exists():
        inputs["vi_long.m4a"] = encode(LONG, tmp_path / "long.m4a", "-c:a", "aac")

    async def main():
        cfg = config(models_dir=MODELS)
        metrics = Metrics()
        worker = EngineWorker(QuanEngine(MODELS, "cuda", cfg.max_chunk_seconds), cfg.infer_hang_seconds, metrics)
        scheduler = Scheduler(worker, cfg, metrics)
        ingress = AudioIngress(cfg, metrics)
        worker.start()
        scheduler.start()
        deadline = time.monotonic() + 300
        while worker.state == EngineState.LOADING:
            assert time.monotonic() < deadline
            await asyncio.sleep(0.1)
        assert worker.state == EngineState.READY

        async def transcribe(path: Path) -> tuple[str, int]:
            wav = await ingress.decode(path, time.monotonic() + 60)
            spans = split(wav, cfg.max_chunk_samples)
            texts = []
            async with scheduler.open_lane(JobKind.FILE) as lane:
                for s in spans:
                    texts.append((await (await lane.submit(wav[s.start:s.end]))).text)
            return " ".join(t for t in texts if t), len(spans)

        try:
            out = {name: await transcribe(path) for name, path in inputs.items()}
            async with scheduler.open_lane(JobKind.FILE) as lane:
                tiny = np.zeros(MIN_AUDIO_SAMPLES + 1, np.float32)
                out["min_chunk"] = ((await (await lane.submit(tiny))).text, 1)
            return out
        finally:
            assert await scheduler.close(30)
            assert await asyncio.to_thread(worker.stop, 30)

    out = asyncio.run(main())
    assert out["vi_a.wav"] == (baseline["vi_a.wav:full"]["text"], 1)
    assert out["silence_2s.wav"] == ("", 1)
    assert out["min_chunk"] == ("", 1)
    assert wer(refs["vi_a.wav"], out["vi_a.mp3"][0]) <= 0.15
    assert wer(refs["vi_b.wav"], out["vi_b.webm"][0]) <= 0.15
    if LONG.exists():
        text, chunks = out["vi_long.m4a"]
        assert chunks == 4  # 106.6 s at <= 30 s per chunk
        assert wer(refs["vi_long.wav"], text) <= 0.10  # M0 baseline: 5.2% with the reference splitter
