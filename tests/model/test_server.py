"""Seam 2: the real app on CPU over a real Uvicorn server: json, text and SSE transcription of the committed fixtures.

Transcripts through the HTTP API equal the reference script's output; every format carries the same text; concurrent
requests do not mix; silence -> ""; clips at and just above the short-audio floors never fail. Needs the staged model
and ffmpeg.
"""

import json
import threading
from pathlib import Path

import pytest
import soundfile as sf

from app.chunking import MIN_AUDIO_SAMPLES
from app.config import SAMPLE_RATE
from app.engine import GipformerEngine, MIN_FORWARD_SAMPLES
from tests.fake_engine import config
from tests.live_server import serve
from tests.model.conftest import FIXTURES, MODELS, reference_text

URL = "/v1/audio/transcriptions"
MODEL = "gipformer1.5-68M-rnnt"


@pytest.fixture(scope="module")
def srv():
    engine = GipformerEngine(MODELS, "fp32", "modified_beam_search", 4, 4, warmup_seconds=config().max_chunk_seconds)
    with serve(engine, ready_timeout=300, http_timeout=300, models_dir=MODELS) as s:
        yield s


def post(srv, path: Path, **fields):
    r = srv.client.post(URL, files={"file": (path.name, path.read_bytes())}, data={"model": MODEL, **fields})
    assert r.status_code == 200, r.text
    return r


def stream(srv, path: Path) -> tuple[list[str], str]:
    events = [json.loads(b.removeprefix("data: ")) for b in post(srv, path, stream="true").text.split("\n\n")
              if b.strip()]
    assert events[-1]["type"] == "transcript.text.done", events[-1]
    return [e["delta"] for e in events[:-1]], events[-1]["text"]


def test_json_text_and_sse_equal_the_reference(srv):
    assert srv.client.get("/ready").status_code == 200
    for name in ("vi_a.wav", "vi_b.wav"):
        path = FIXTURES / name
        expected = reference_text(path, "fp32", "modified_beam_search")
        body = post(srv, path).json()
        assert body["text"] == expected and body["usage"]["seconds"] > 0
        assert post(srv, path, response_format="text").text == expected
        deltas, done = stream(srv, path)
        assert "".join(deltas) == done == expected


def test_silence_and_short_clip_match_the_reference(srv):
    # Digital silence is not forced to "": the model's own output is returned, as in the reference script.
    silence = FIXTURES / "silence_2s.wav"
    expected = reference_text(silence, "fp32", "modified_beam_search")
    assert post(srv, silence).json()["text"] == expected
    assert stream(srv, silence)[1] == expected
    assert isinstance(post(srv, FIXTURES / "short_200ms.wav").json()["text"], str)


@pytest.mark.parametrize("samples", [MIN_FORWARD_SAMPLES, MIN_AUDIO_SAMPLES, MIN_AUDIO_SAMPLES + 1, 2 * MIN_AUDIO_SAMPLES])
def test_clips_at_and_just_above_the_floors_succeed_in_every_format(srv, tmp_path, samples):
    # Speech cut mid-utterance: at or below the short-audio floor -> "" without the model, just above it -> the
    # model's own (possibly empty) output; never an error, timed formats included.
    x = sf.read(FIXTURES / "vi_a.wav", dtype="float32")[0][2 * SAMPLE_RATE:][:samples]
    path = tmp_path / "cut.wav"
    sf.write(path, x, SAMPLE_RATE, subtype="FLOAT")
    text = post(srv, path).json()["text"]
    assert samples > MIN_AUDIO_SAMPLES or text == ""
    assert post(srv, path, response_format="text").text == text
    assert stream(srv, path)[1] == text
    assert post(srv, path, response_format="verbose_json").json()["text"] == text
    for fmt in ("srt", "vtt"):
        post(srv, path, response_format=fmt)


def test_concurrent_requests_do_not_mix(srv):
    expected = {n: post(srv, FIXTURES / n).json()["text"] for n in ("vi_a.wav", "vi_b.wav")}
    results: dict[tuple[str, int], str] = {}

    def run(name: str, i: int):
        results[(name, i)] = post(srv, FIXTURES / name).json()["text"]

    threads = [threading.Thread(target=run, args=(n, i)) for i in range(3) for n in expected]
    for t in threads:
        t.start()
    for t in threads:
        t.join(120)
    assert results == {(n, i): expected[n] for i in range(3) for n in expected}
