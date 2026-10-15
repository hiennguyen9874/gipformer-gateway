"""M4 P0 spike: measure what CTC-path timestamps can be trusted for (docs/baseline/alignment_probe.json).

    CUDA_VISIBLE_DEVICES=<free gpu> uv run scripts/probe_alignment.py --out docs/baseline/alignment_probe.json

Runs the real `QuanEngine` (greedy, full context) on the committed fixtures (and fixtures-large/vi_long.wav when
present) and reports:
  (a) encoder output length vs the frame-geometry formula `#{k : 640 k + 160 < L}` over many lengths;
  (b) shift equivariance: prepend `s` zero samples, the word starts/ends should move by `s` (within one frame);
  (c) spike latency: word times vs the edges of known digital-zero gaps (vi_a | 1 s | vi_b | 1 s | vi_a | 1 s | vi_b,
      > 30 s so it also crosses a chunk boundary) and vs an energy-based speech onset/offset of each clip;
  (d) the piece/text invariant of app/alignment.py on every chunk; (e) how often <unk> is emitted;
  plus the inter-word gap distribution (to pick the cue pause) and the cues of the composite.
Times use app.alignment with its current EMISSION_SHIFT. Reports numbers and word counts only, no transcripts.
"""

import argparse
import difflib
import json
import os
import statistics
import sys
from itertools import pairwise
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402

from app.alignment import align, AlignmentError, EMISSION_SHIFT, segments, Word  # noqa: E402
from app.chunking import split  # noqa: E402
from app.config import SAMPLE_RATE  # noqa: E402

SR = SAMPLE_RATE
FRAME = 640
LENGTHS = [
    161,
    319,
    320,
    480,
    481,
    799,
    800,
    801,
    1119,
    1120,
    1121,
    1600,
    1601,
    3200,
    8000,
    12345,
    16000,
    16001,
    47999,
    160000,
    480000,
]
SHIFTS = [160, 320, 640, 1000, 1280, 3200, 4000, 8000]
GAP_S = 1.0
# Speech onset/offset: first/last 10 ms window within this of the clip's loudest one (FLEURS noise floor: ~-30 dB).
ENERGY_DB = -20.0
LONG_GAP = (
    4800  # vi_long.wav: utterances joined by 0.3 s of digital zeros (scripts/fetch_fixtures.py)
)


def ms(samples: float) -> float:
    return round(samples / SR * 1000, 1)


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    v = sorted(values)
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]  # noqa: E731
    return {
        "n": len(v),
        "min": round(v[0], 1),
        "p10": round(q(0.1), 1),
        "median": round(statistics.median(v), 1),
        "p90": round(q(0.9), 1),
        "max": round(v[-1], 1),
        "mean": round(statistics.fmean(v), 1),
    }


def energy_edges(x: np.ndarray) -> tuple[int, int]:
    win = SR // 100
    n = len(x) // win
    rms = np.sqrt((x[: n * win].astype(np.float64).reshape(n, win) ** 2).mean(1) + 1e-20)
    db = 20 * np.log10(rms / rms.max())
    loud = np.flatnonzero(db > ENERGY_DB)
    return int(loud[0] * win), int((loud[-1] + 1) * win)


def zero_runs(x: np.ndarray, min_len: int) -> list[tuple[int, int]]:
    z = np.concatenate(([0], (x == 0).astype(np.int8), [0]))
    d = np.diff(z)
    return [
        (int(a), int(b))
        for a, b in zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=True)
        if b - a >= min_len
    ]


def edge_deltas(x: np.ndarray, ws: list[Word], regions: list[tuple[int, int]]) -> list[dict]:
    """Per utterance region: first word start - speech onset, last word end - speech offset (ms)."""
    rows = []
    for lo, hi in regions:
        inside = [w for w in ws if lo <= w.start < hi]
        if not inside:
            continue
        on, off = energy_edges(x[lo:hi])
        rows.append(
            {
                "region_ms": [ms(lo), ms(hi)],
                "first_start_minus_onset_ms": ms(inside[0].start - lo - on),
                "last_end_minus_offset_ms": ms(inside[-1].end - lo - off),
            }
        )
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--models-dir", type=Path, default=Path(os.environ.get("GIPFORMER_MODELS_DIR", ROOT / "models"))
    )
    ap.add_argument("--max-chunk-seconds", type=float, default=30.0)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    from app.engine import QuanEngine

    engine = QuanEngine(args.models_dir, "cuda", warmup_seconds=args.max_chunk_seconds)
    engine.load()
    engine.warmup()
    pieces = engine.pieces()
    max_chunk = int(args.max_chunk_seconds * SR)
    report: dict = {"emission_shift": EMISSION_SHIFT, "max_chunk_seconds": args.max_chunk_seconds}
    failures: list[str] = []
    units_total = unk_total = 0
    invariant = {"chunks": 0, "errors": 0}

    def transcribe(x: np.ndarray) -> list[tuple]:
        nonlocal units_total, unk_total
        chunks = [
            (s, engine.transcribe(np.ascontiguousarray(x[s.start : s.end])))
            for s in split(x, max_chunk)
        ]
        for _, r in chunks:
            units_total += len(r.units)
            unk_total += r.units.count(1)
        return chunks

    def timed_words(x: np.ndarray) -> list[Word]:
        chunks = transcribe(x)
        invariant["chunks"] += len(chunks)
        try:
            return align(chunks, pieces)
        except AlignmentError as e:
            invariant["errors"] += 1
            failures.append(f"alignment error: {e}")
            return []

    clips = {
        n: sf.read(ROOT / "tests/fixtures" / n, dtype="float32")[0]
        for n in ("vi_a.wav", "vi_b.wav", "silence_2s.wav", "short_200ms.wav")
    }
    long_path = ROOT / "fixtures-large/vi_long.wav"
    if long_path.exists():
        clips["vi_long.wav"] = sf.read(long_path, dtype="float32")[0]

    # (a) frame geometry
    speech = clips["vi_a.wav"]
    geometry = []
    for n in LENGTHS:
        x = np.resize(speech, n).astype(np.float32)
        frames = engine.transcribe(x).frames
        formula = -(-(n - 160) // FRAME)
        geometry.append({"samples": n, "frames": frames, "formula": formula})
        if frames != formula:
            failures.append(f"frames({n}) = {frames} != formula {formula}")
    report["a_frame_geometry"] = {
        "all_equal": all(g["frames"] == g["formula"] for g in geometry),
        "lengths": geometry,
    }

    # (d)+(e) on every fixture, inter-word gaps
    base_words = {}
    gaps = []
    for name, x in clips.items():
        ws = timed_words(x)
        base_words[name] = ws
        gaps += [ms(b.start - a.end) for a, b in pairwise(ws)]
    report["inter_word_gap_ms"] = stats(gaps)
    report["inter_word_gap_ms"]["share_ge_500"] = round(
        sum(g >= 500 for g in gaps) / max(len(gaps), 1), 4
    )
    report["word_duration_ms"] = stats(
        [ms(w.end - w.start) for ws in base_words.values() for w in ws]
    )

    # (b) shift equivariance
    shift_rows = []
    for name in ("vi_a.wav", "vi_b.wav"):
        x, ref = clips[name], base_words[name]
        for s in SHIFTS:
            ws = timed_words(np.concatenate([np.zeros(s, np.float32), x]))
            sm = difflib.SequenceMatcher(
                a=[w.text for w in ref], b=[w.text for w in ws], autojunk=False
            )
            pairs = [
                (ref[i + k], ws[j + k])
                for i, j, size in sm.get_matching_blocks()
                for k in range(size)
            ]
            ds = [ms(b.start - a.start - s) for a, b in pairs]
            de = [ms(b.end - a.end - s) for a, b in pairs]
            shift_rows.append(
                {
                    "clip": name,
                    "shift_samples": s,
                    "words": len(ref),
                    "matched": len(pairs),
                    "start_within_1_frame": round(
                        sum(abs(d) <= 40 for d in ds) / max(len(ds), 1), 4
                    ),
                    "end_within_1_frame": round(sum(abs(d) <= 40 for d in de) / max(len(de), 1), 4),
                    "start_delta_ms": stats(ds),
                    "end_delta_ms": stats(de),
                }
            )
    starts = [r["start_within_1_frame"] for r in shift_rows]
    report["b_shift"] = {
        "min_start_within_1_frame": min(starts),
        "min_end_within_1_frame": min(r["end_within_1_frame"] for r in shift_rows),
        "rows": shift_rows,
    }

    # (c) known zero gaps and energy edges
    gap = np.zeros(int(GAP_S * SR), np.float32)
    parts = [
        clips["vi_a.wav"],
        gap,
        clips["vi_b.wav"],
        gap,
        clips["vi_a.wav"],
        gap,
        clips["vi_b.wav"],
    ]
    composite = np.concatenate(parts)
    offsets = np.cumsum([0] + [len(p) for p in parts])
    zero_gaps = [(int(offsets[i]), int(offsets[i + 1])) for i in (1, 3, 5)]
    spans = split(composite, max_chunk)
    ws = timed_words(composite)
    cues = segments(ws)
    gap_rows = []
    for g0, g1 in zero_gaps:
        before = [w for w in ws if w.start < g0]
        after = [w for w in ws if w.start >= g0]
        gap_rows.append(
            {
                "gap_ms": [ms(g0), ms(g1)],
                "last_word_end_minus_gap_start_ms": ms(before[-1].end - g0) if before else None,
                "first_word_start_minus_gap_end_ms": ms(after[0].start - g1) if after else None,
                "words_inside_gap": sum(1 for w in ws if w.start < g1 and w.end > g0),
                "cues_straddling_gap": sum(1 for c in cues if c.start < g0 and c.end > g1),
            }
        )
    clip_edges = edge_deltas(
        composite, ws, [(int(offsets[i]), int(offsets[i + 1])) for i in (0, 2, 4, 6)]
    )
    if "vi_long.wav" in clips:
        x = clips["vi_long.wav"]
        bounds = [0] + [v for run in zero_runs(x, LONG_GAP) for v in run] + [len(x)]
        clip_edges += edge_deltas(
            x, base_words["vi_long.wav"], list(zip(bounds[::2], bounds[1::2], strict=True))
        )
    report["c_known_gaps"] = {
        "audio_ms": ms(len(composite)),
        "chunk_spans_ms": [[ms(s.start), ms(s.end)] for s in spans],
        "gaps": gap_rows,
        "energy_threshold_db": ENERGY_DB,
        "utterance_edges": clip_edges,
        "first_start_minus_onset_ms": stats([r["first_start_minus_onset_ms"] for r in clip_edges]),
        "last_end_minus_offset_ms": stats([r["last_end_minus_offset_ms"] for r in clip_edges]),
        "cues": len(cues),
        "cue_duration_ms": stats([ms(c.end - c.start) for c in cues]),
        "cue_chars": stats([len(c.text) for c in cues]),
    }
    # (d), (e) totals (also covers the shift and composite runs)
    report["d_invariant"] = invariant
    report["e_unk"] = {
        "units": units_total,
        "unk": unk_total,
        "share": round(unk_total / max(units_total, 1), 6),
    }
    report["failures"] = failures

    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print("wrote", args.out)
    print(text)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
