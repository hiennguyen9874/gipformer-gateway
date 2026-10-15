"""Split decoded audio into inference chunks at quiet points, keeping absolute sample offsets.

Policy (docs/api-contract.vi.md §4):
- Audio of at most `MIN_AUDIO_SAMPLES` (100 ms at 16 kHz), empty included, yields no chunk: the transcript is ""
  and the model is not called. The recognizer fails below 1,360 samples (`app.engine.MIN_FORWARD_SAMPLES`), and cuts
  of speech of 85-100 ms almost never yield text (docs/baseline.vi.md §5), so the floor keeps a margin above it.
- Otherwise the chunks cover the audio exactly and in order: chunk i ends where chunk i+1 starts, the first starts
  at 0 and the last ends at `len(wav)`. Nothing is dropped, silence included (no amplitude threshold).
- Each chunk is at most `max_samples` long and longer than `MIN_AUDIO_SAMPLES`; a short tail is avoided by
  constraining the previous cut, never by dropping it (the reference `split_quiet` silently drops tails <= 100 ms).
- A cut is placed in the middle of the lowest-energy 50 ms window within the last `SEARCH_SECONDS` before the
  maximum length (as the reference splitter), ties going to the latest window so silence yields full-length chunks.

Deterministic: same audio and limit, same spans.
"""

from typing import NamedTuple

import numpy as np

from app.config import SAMPLE_RATE

MIN_AUDIO_SAMPLES = SAMPLE_RATE // 10  # <= 100 ms: empty transcript, no inference
QUIET_WINDOW = SAMPLE_RATE // 20  # 50 ms energy window
SEARCH_SECONDS = 5.0


class Span(NamedTuple):
    start: int  # absolute sample offsets at 16 kHz, end exclusive
    end: int


def split(wav: np.ndarray, max_samples: int) -> list[Span]:
    n = len(wav)
    if max_samples < 2 * (MIN_AUDIO_SAMPLES + QUIET_WINDOW):
        raise ValueError("max_samples too small to split with the short-chunk floor")
    if n <= MIN_AUDIO_SAMPLES:
        return []
    min_chunk = MIN_AUDIO_SAMPLES + 1
    search = int(SEARCH_SECONDS * SAMPLE_RATE)
    spans, start = [], 0
    while n - start > max_samples:
        hi = min(
            start + max_samples, n - min_chunk
        )  # keep the remainder long enough to be transcribed
        lo = max(start + max_samples - search, start + min_chunk)
        cut = _quietest(wav, lo, hi)
        spans.append(Span(start, cut))
        start = cut
    spans.append(Span(start, n))
    return spans


def _quietest(wav: np.ndarray, lo: int, hi: int) -> int:
    """Centre of the lowest-energy window inside [lo, hi]; the latest one on ties."""
    if hi - lo < QUIET_WINDOW:
        return hi
    x = wav[lo:hi].astype(np.float64)
    csum = np.concatenate(([0.0], np.cumsum(x * x)))
    energy = csum[QUIET_WINDOW:] - csum[:-QUIET_WINDOW]  # energy of window starting at each offset
    # Cumulative sums leave rounding noise; compare on a grid relative to the loudest window so ties are stable.
    scale = energy.max()
    if scale > 0:
        energy = np.round(energy / scale, 9)
    best = len(energy) - 1 - int(np.argmin(energy[::-1]))
    return lo + best + QUIET_WINDOW // 2
