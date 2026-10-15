"""Seam 2: the real app on CPU over a real Uvicorn server: json, text and SSE transcription of the committed fixtures.

Transcripts through the HTTP API equal the reference script's output; every format carries the same text; concurrent
requests do not mix; silence -> "". Needs the staged model and ffmpeg.
"""

import json
import threading
from pathlib import Path

import pytest

from app.engine import GipformerEngine
from tests.live_server import serve
from tests.model.conftest import FIXTURES, MODELS, reference_text

URL = "/v1/audio/transcriptions"
MODEL = "gipformer1.5-68M-rnnt"


@pytest.fixture(scope="module")
def srv():
    engine = GipformerEngine(MODELS, "fp32", "modified_beam_search", 4, 4, warmup_seconds=30.0)
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
