"""R3a GPU gate: the realtime endpoint on the real engine over a real Uvicorn server (websockets-sansio).

FLEURS fixtures (16 kHz) are resampled to 24 kHz PCM16 (soxr HQ, quantized) and sent in 20 ms packets, as a live
client would. (a) The transcript equals REST on the exact committed 16 kHz waveform (same split/run_chunks/Joiner);
(b) WER vs the reference is at most REST-on-the-original + 2 points (soxr + PCM16 vs no resampling, so no text parity);
(c) two concurrent sessions with different audio do not mix; (d) a turn > 30 s yields several deltas in one item.

Needs a CUDA GPU, staged assets and ffmpeg: CUDA_VISIBLE_DEVICES=<id> uv run pytest tests/gpu
"""

import json
import os
import threading
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import soxr
import torch

from app.engine import QuanEngine
from app.realtime import PCM_RATE, TurnBuffer
from scripts.smoke_inference import wer
from tests.live_server import serve, SR, wav_file, WS

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("GIPFORMER_MODELS_DIR", ROOT / "models"))
FIXTURES = ROOT / "tests/fixtures"
MODEL = "quan-transcribe-1.0"
PACKET = PCM_RATE // 50 * 2  # 20 ms of PCM16
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (MODELS / "embeddinggemma-2").is_dir(), reason="assets not staged"),
]


@pytest.fixture(scope="module")
def srv():
    engine = QuanEngine(MODELS, "cuda", warmup_seconds=30.0)
    with serve(engine, ready_timeout=300, http_timeout=300, models_dir=MODELS) as s:
        yield s


def load(name: str) -> np.ndarray:
    return sf.read(FIXTURES / name, dtype="float32")[0]


def to_pcm24(wav16: np.ndarray) -> bytes:
    up = soxr.resample(wav16, SR, PCM_RATE, quality="HQ")
    return np.round(np.clip(up, -1, 32767 / 32768) * 32768).astype("<i2").tobytes()


def committed(pcm: bytes) -> np.ndarray:
    """The 16 kHz waveform the server transcribes for this turn."""
    buf = TurnBuffer(len(pcm))
    buf.append(pcm)
    return buf.commit()[0]


def rest(srv, path: Path) -> str:
    r = srv.client.post("/v1/audio/transcriptions", files={"file": (path.name, path.read_bytes())},
                        data={"model": MODEL})
    assert r.status_code == 200, r.text
    return r.json()["text"]


def realtime(srv, turns: list[bytes]) -> list[tuple[list[str], dict]]:
    ws = WS(srv, timeout=120)
    try:
        out = []
        for pcm in turns:
            ws.append(pcm, packet=PACKET)
            ws.send("input_audio_buffer.commit")
            _, deltas, done = ws.turn()
            assert done["type"] == "conversation.item.input_audio_transcription.completed", done
            assert done["transcript"] == "".join(deltas)
            assert done["usage"] == {"type": "duration", "seconds": len(pcm) / 2 / PCM_RATE}
            out.append((deltas, done))
        return out
    finally:
        ws.close()


def test_turns_match_rest_on_the_committed_waveform_and_wer(srv, tmp_path):
    refs = json.loads((FIXTURES / "manifest.json").read_text())["files"]
    names = ("vi_a.wav", "vi_b.wav", "silence_2s.wav", "short_200ms.wav")
    pcms = [to_pcm24(load(n)) for n in names]
    results = realtime(srv, pcms)  # one socket, several commits
    for name, pcm, (_, done) in zip(names, pcms, results):
        text = done["transcript"]
        assert text == rest(srv, wav_file(tmp_path / f"c_{name}", committed(pcm))), name  # (a)
        ref = refs[name]["reference"]
        if ref:
            assert wer(ref, text) <= wer(ref, rest(srv, FIXTURES / name)) + 0.02, (name, text)  # (b)
        elif name == "silence_2s.wav":
            assert text == ""


def test_concurrent_sessions_do_not_mix(srv):
    pcms = {n: to_pcm24(load(n)) for n in ("vi_a.wav", "vi_b.wav")}
    expected = {n: realtime(srv, [p])[0][1]["transcript"] for n, p in pcms.items()}
    results: dict[tuple[str, int], list[str]] = {}

    def run(name: str, i: int):
        results[(name, i)] = [d["transcript"] for _, d in realtime(srv, [pcms[name]] * 2)]

    threads = [threading.Thread(target=run, args=(n, i)) for i in range(2) for n in pcms]
    for t in threads:
        t.start()
    for t in threads:
        t.join(300)
    assert results == {(n, i): [expected[n]] * 2 for i in range(2) for n in pcms}  # (c)


def test_long_turn_is_one_item_with_several_deltas(srv, tmp_path):
    a, b = load("vi_a.wav"), load("vi_b.wav")
    gap = np.zeros(SR, np.float32)
    pcm = to_pcm24(np.concatenate([a, gap, b, gap, a, gap, b]))  # 37.7 s > GIPFORMER_MAX_CHUNK_SECONDS (30)
    [(deltas, done)] = realtime(srv, [pcm])
    assert len(deltas) == 2 and all(deltas)  # (d)
    assert done["transcript"] == rest(srv, wav_file(tmp_path / "long.wav", committed(pcm)))
