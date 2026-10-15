"""Quiet-point splitter: exact coverage, bounds, short-clip and tail policy, cut placement, determinism."""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from app.chunking import MIN_AUDIO_SAMPLES, Span, split
from app.engine import MIN_FORWARD_SAMPLES

SR = 16000
MAX = 30 * SR
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
LONG = Path(__file__).resolve().parents[2] / "fixtures-large" / "vi_long.wav"


def check(spans: list[Span], n: int, max_samples: int) -> None:
    assert spans[0].start == 0 and spans[-1].end == n
    for a, b in zip(spans, spans[1:]):
        assert a.end == b.start  # contiguous: no gap, no overlap, nothing dropped
    for s in spans:
        assert MIN_AUDIO_SAMPLES < s.end - s.start <= max_samples
        assert s.end - s.start >= MIN_FORWARD_SAMPLES


def noise(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).normal(0, 0.1, n).astype(np.float32)


@pytest.mark.parametrize("n", [0, 1, MIN_FORWARD_SAMPLES - 1, MIN_FORWARD_SAMPLES, MIN_AUDIO_SAMPLES])
def test_too_short_audio_yields_no_chunk(n):
    assert split(noise(n), MAX) == []


def test_short_audio_floor_never_forwards_what_the_engine_rejects():
    # Measured floor (docs/baseline/limits_probe.json): the recognizer fails below MIN_FORWARD_SAMPLES.
    assert MIN_FORWARD_SAMPLES == 1360 and MIN_AUDIO_SAMPLES == 1600
    assert split(noise(MIN_AUDIO_SAMPLES + 1), MAX) == [Span(0, MIN_AUDIO_SAMPLES + 1)]


@pytest.mark.parametrize("n", [MIN_AUDIO_SAMPLES + 1, 8 * SR, MAX])
def test_audio_up_to_the_limit_is_one_chunk(n):
    assert split(noise(n), MAX) == [Span(0, n)]


@pytest.mark.parametrize("extra", [1, MIN_AUDIO_SAMPLES, MIN_AUDIO_SAMPLES + 1, 3 * SR, MAX - 1, MAX, 5 * MAX + 7])
def test_long_audio_is_covered_exactly_and_tail_is_never_too_short(extra):
    n = MAX + extra
    spans = split(noise(n), MAX)
    check(spans, n, MAX)
    assert len(spans) >= 2


def test_cut_lands_in_the_quiet_gap():
    # 40 s of noise with a 300 ms gap at 27.0-27.3 s: the cut must fall inside the gap.
    x = noise(40 * SR)
    x[27 * SR:int(27.3 * SR)] = 0
    spans = split(x, MAX)
    check(spans, len(x), MAX)
    assert 27 * SR <= spans[0].end <= int(27.3 * SR)


def test_short_limit_searches_the_full_five_seconds():
    x = noise(8 * SR)
    x[2 * SR:int(2.3 * SR)] = 0
    spans = split(x, 6 * SR)
    check(spans, len(x), 6 * SR)
    assert 2 * SR <= spans[0].end <= int(2.3 * SR)


def test_gap_outside_the_search_window_is_ignored():
    x = noise(40 * SR)
    x[10 * SR:11 * SR] = 0  # quiet, but more than 5 s before the 30 s limit
    spans = split(x, MAX)
    assert spans[0].end >= 25 * SR


def test_pure_silence_gives_full_length_chunks():
    x = np.zeros(100 * SR, np.float32)
    spans = split(x, MAX)
    check(spans, len(x), MAX)
    assert [s.end - s.start for s in spans[:-1]] == [MAX - 400] * (len(spans) - 1)  # latest quiet window wins


def test_deterministic():
    x = noise(95 * SR, seed=3)
    assert split(x, MAX) == split(x.copy(), MAX)


def test_small_limits_keep_the_floor():
    x = noise(10 * SR)
    for max_samples in (SR, SR + 1, 2 * SR):
        check(split(x, max_samples), len(x), max_samples)
    with pytest.raises(ValueError):
        split(x, 4000)


@pytest.mark.parametrize("name", ["vi_a.wav", "vi_b.wav", "silence_2s.wav", "short_200ms.wav"])
def test_fixture_clips_below_the_limit_are_one_chunk(name):
    x = sf.read(FIXTURES / name, dtype="float32")[0]
    assert split(x, MAX) == [Span(0, len(x))]


@pytest.mark.skipif(not LONG.exists(), reason="large fixtures not fetched (scripts/fetch_fixtures.py)")
def test_long_speech_fixture_cuts_in_low_energy_regions():
    x = sf.read(LONG, dtype="float32")[0]
    spans = split(x, MAX)
    check(spans, len(x), MAX)
    rms = np.sqrt(np.mean(x ** 2))
    for s in spans[:-1]:
        window = x[s.end - 400:s.end + 400]
        assert np.sqrt(np.mean(window ** 2)) < 0.25 * rms  # cut placed in a pause, not mid-word
