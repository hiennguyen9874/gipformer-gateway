"""The only code that calls QUAN, and the single thread allowed to run it.

`QuanCTC._acts` is mutable hook state, so every load/warm-up/inference on the model instance runs on one dedicated
worker thread (`EngineWorker`). Endpoints and sessions never touch the model; they go through `app.scheduler`, which
hands the worker at most one job at a time.

Cancellation is soft: a call that has started always runs to completion (a CUDA kernel cannot be interrupted safely);
the caller discards the result. A call running longer than `hang_seconds` marks the worker hung: `/health` then fails
so the supervisor restarts the process.
"""

import asyncio
import concurrent.futures
import enum
import logging
import queue
import threading
import time
from pathlib import Path
from typing import NamedTuple, Protocol

import numpy as np

from app.config import SAMPLE_RATE
from app.errors import inference_failed, model_not_ready
from app.observability import log, Metrics

# The log-mel frontend crashes below 161 samples (docs/baseline.vi.md §7). Audio policy for short clips (return ""
# without inference) belongs to the callers; this is the hard safety floor.
MIN_FORWARD_SAMPLES = 161
BLANK = 0  # CTC unit ids: 0 = blank, 1 = <unk>, 2.. = Gemma tokens


class ChunkResult(NamedTuple):
    """Greedy CTC result of one chunk: the text and the best path it was decoded from (no tensors).

    `units` is the collapsed path without blanks (<unk> kept, so it matches the path); unit i was emitted on encoder
    frames `first[i]..last[i]` (inclusive, relative to the chunk). `frames` is the number of valid encoder frames.
    """

    text: str
    units: tuple[int, ...]
    first: tuple[int, ...]
    last: tuple[int, ...]
    frames: int


def ctc_path(best: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Collapse an argmax frame sequence like `quan_transcribe.model.greedy_decode`, keeping each unit's frames."""
    best = np.asarray(best)
    if best.size == 0:
        return (), (), ()
    starts = np.flatnonzero(np.concatenate(([True], best[1:] != best[:-1])))
    ends = np.concatenate((starts[1:], [len(best)])) - 1
    keep = best[starts] != BLANK
    return (
        tuple(best[starts][keep].tolist()),
        tuple(starts[keep].tolist()),
        tuple(ends[keep].tolist()),
    )


class Engine(Protocol):
    def load(self) -> None: ...
    def warmup(self) -> None: ...
    def transcribe(self, wav: np.ndarray) -> ChunkResult: ...
    def pieces(self) -> tuple[str, ...]:
        """Unit id -> token piece ("" for blank and <unk>); called once after `load`."""
        ...


class QuanEngine:
    """Greedy, full-context decoding of one 16 kHz mono chunk (the M0 baseline decoder settings, no LM)."""

    def __init__(self, models_dir: Path, device: str, warmup_seconds: float):
        self._models_dir, self._device, self._warmup_seconds = models_dir, device, warmup_seconds
        self._asr = None

    def load(self) -> None:
        from app.quan_loader import load_quan

        self._asr = load_quan(self._models_dir, self._device)

    def warmup(self) -> None:
        """Also used to recover after a failed call. Running the longest allowed chunk reserves its peak memory."""
        import torch

        if self._asr.device.type == "cuda":
            torch.cuda.empty_cache()
        self.transcribe(np.zeros(int(self._warmup_seconds * SAMPLE_RATE), dtype=np.float32))

    def transcribe(self, wav: np.ndarray) -> ChunkResult:
        # The reference's greedy batch decode step (`_run` with decoder=None) without `split_quiet` (which silently
        # drops pieces <= 100 ms; chunking is the caller's job), keeping the best path for timestamps. The text is
        # decoded from the same units as `_run`, so it equals the baseline.
        logits, lens = self._asr._forward([wav], False, None)
        frames = int(lens[0])
        units, first, last = ctc_path(logits[0, :frames].argmax(-1).cpu().numpy())
        return ChunkResult(self._asr.decode_units(list(units)), units, first, last, frames)

    def pieces(self) -> tuple[str, ...]:
        return ("", "") + tuple(self._asr.tokenizer.convert_ids_to_tokens(self._asr.gemma_ids))


def check_waveform(wav: np.ndarray) -> None:
    if not isinstance(wav, np.ndarray) or wav.dtype != np.float32 or wav.ndim != 1:
        raise ValueError("waveform must be a 1-D float32 numpy array (16 kHz mono)")
    if len(wav) < MIN_FORWARD_SAMPLES:
        raise ValueError(
            f"waveform shorter than {MIN_FORWARD_SAMPLES} samples must not reach the model"
        )
    if not np.isfinite(wav).all():
        raise ValueError("waveform contains non-finite samples")


class EngineState(enum.StrEnum):
    LOADING = "loading"
    READY = "ready"
    RECOVERING = "recovering"
    FAILED = "failed"
    STOPPED = "stopped"


class EngineWorker:
    def __init__(self, engine: Engine, hang_seconds: float, metrics: Metrics):
        self._engine, self._hang_seconds, self._metrics = engine, hang_seconds, metrics
        self._queue: queue.SimpleQueue = queue.SimpleQueue()
        self._thread = threading.Thread(target=self._main, name="gipformer-inference", daemon=True)
        self._state = EngineState.LOADING
        self._call_started: float | None = None
        self.pieces: tuple[str, ...] = ()  # set before the worker becomes ready
        metrics.gauge_fn(
            "ready",
            "1 when the model is loaded, warmed up and the worker is not hung.",
            lambda: float(self.ready),
        )
        metrics.gauge_fn(
            "worker_busy",
            "1 while an inference call is running.",
            lambda: float(self._call_started is not None),
        )
        metrics.gauge_fn(
            "worker_hung",
            "1 when the running call exceeded GIPFORMER_INFER_HANG_SECONDS.",
            lambda: float(self.hung),
        )

    @property
    def state(self) -> EngineState:
        return self._state

    @property
    def hung(self) -> bool:
        started = self._call_started
        return started is not None and time.monotonic() - started > self._hang_seconds

    @property
    def ready(self) -> bool:
        return self._state == EngineState.READY and not self.hung

    @property
    def healthy(self) -> bool:
        """Liveness: false when the process needs a restart (load/recovery failed or a call hangs)."""
        return self._state not in (EngineState.FAILED, EngineState.STOPPED) and not self.hung

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float) -> bool:
        """Ask the thread to exit after its current call; False if it is still running after `timeout`."""
        self._queue.put(None)
        if self._thread.is_alive():
            self._thread.join(timeout)
        return not self._thread.is_alive()

    async def transcribe(self, wav: np.ndarray) -> ChunkResult:
        check_waveform(wav)
        fut: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((wav, fut))
        try:
            return await asyncio.wrap_future(fut)
        finally:
            wav = fut = None  # exception tracebacks must not keep completed audio alive

    def _set_state(self, state: EngineState) -> None:
        if state != self._state:
            self._state = state
            log(logging.INFO, "engine.state", state=state.value)

    def _main(self) -> None:
        t0 = time.monotonic()
        try:
            self._engine.load()
            self.pieces = tuple(self._engine.pieces())
            self._warmup()
        except Exception:
            log(logging.ERROR, "engine.load_failed", exc_info=True)
            self._metrics.inc(
                "worker_failures_total", "Load, inference or recovery failures.", stage="load"
            )
            self._set_state(EngineState.FAILED)
        else:
            log(logging.INFO, "engine.loaded", seconds=round(time.monotonic() - t0, 3))
            self._set_state(EngineState.READY)

        while (item := self._queue.get()) is not None:
            wav, fut = item
            if not fut.set_running_or_notify_cancel():
                wav = item = fut = None
                continue
            if self._state != EngineState.READY:
                fut.set_exception(model_not_ready())
                wav = item = fut = None
                continue
            self._call_started = started = time.monotonic()
            try:
                result = self._engine.transcribe(wav)
            except Exception as e:
                self._call_started = None
                log(logging.ERROR, "engine.inference_failed", exc_info=True, samples=len(wav))
                self._metrics.inc(
                    "worker_failures_total", "Load, inference or recovery failures.", stage="infer"
                )
                err = inference_failed()
                err.__cause__ = e.with_traceback(
                    None
                )  # already logged; the traceback can retain audio
                fut.set_exception(err)
                self._recover()
            else:
                self._call_started = None
                self._metrics.observe(
                    "infer_seconds", "Duration of one inference call.", time.monotonic() - started
                )
                fut.set_result(result)
            finally:
                wav = item = fut = err = (
                    None  # idle worker must not pin completed audio or failure tracebacks
                )
        self._set_state(EngineState.STOPPED)

    def _warmup(self) -> None:
        self._call_started = time.monotonic()
        try:
            self._engine.warmup()
        finally:
            self._call_started = None

    def _recover(self) -> None:
        """Re-run warm-up on this thread; queued calls wait. If it fails the model stays unavailable."""
        self._set_state(EngineState.RECOVERING)
        try:
            self._warmup()
        except Exception:
            log(logging.ERROR, "engine.recovery_failed", exc_info=True)
            self._metrics.inc(
                "worker_failures_total", "Load, inference or recovery failures.", stage="recover"
            )
            self._set_state(EngineState.FAILED)
        else:
            self._set_state(EngineState.READY)
