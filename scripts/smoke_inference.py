"""Inference baseline: full-context and streaming decoding on the fixtures, with timings and memory.

    CUDA_VISIBLE_DEVICES=<gpu> uv run --extra cu126 scripts/smoke_inference.py --out /tmp/online.json
    CUDA_VISIBLE_DEVICES=<gpu> HF_HOME=$(mktemp -d) uv run --extra cu126 --offline scripts/smoke_inference.py --no-network \
        --out docs/baseline/offline.json

Loads only staged assets (app/quan_loader.py) after verifying them against assets/manifest.json. Decoding is greedy
(no LM) with the reference defaults below. --no-network sets HF_HUB_OFFLINE and installs an audit hook that fails
any socket connection or DNS lookup made by this process, so a hidden Hub access is an error, not a silent download.
Outputs a JSON report; exits non-zero when a gate check fails.
"""

import argparse
import json
import os
import platform
import resource
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.scoring import wer  # noqa: E402

DECODER = {
    "decoder": "greedy",
    "use_lm": False,
    "chunk_seconds": 30.0,
    "batch_seconds": 600.0,
    "left_chunks": None,
}
# Lengths (samples @16 kHz) probed through the model without the quiet splitter, to find the frontend minimum.
SHORT_PROBE = [1, 160, 161, 320, 480, 640, 1280, 1600, 1601, 3200, 8000]


def block_network() -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    def hook(event, args):
        if event in ("socket.connect", "socket.getaddrinfo", "socket.sendto"):
            raise RuntimeError(
                f"network access blocked: {event} {args[1:] if event == 'socket.connect' else args}"
            )

    sys.addaudithook(hook)


def rss_mb() -> dict:
    status = Path("/proc/self/status").read_text()
    cur = next(int(line.split()[1]) for line in status.splitlines() if line.startswith("VmRSS:"))
    return {
        "rss_mb": round(cur / 1024),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }


def gpu_mem(torch) -> dict:
    mb = 1 << 20
    return {
        "allocated_mb": round(torch.cuda.memory_allocated() / mb),
        "reserved_mb": round(torch.cuda.memory_reserved() / mb),
        "peak_allocated_mb": round(torch.cuda.max_memory_allocated() / mb),
        "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / mb),
    }


def environment(torch) -> dict:
    import transformers

    env = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "cpu_count": os.cpu_count(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        env |= {"gpu": p.name, "gpu_total_mb": p.total_memory >> 20}
        try:
            env["driver"] = (
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                        "-i",
                        "0",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                .stdout.strip()
                .splitlines()[0]
            )
        except Exception as e:  # informational only
            env["driver"] = f"unknown ({type(e).__name__})"
    return env


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--models-dir", type=Path, default=Path(os.environ.get("GIPFORMER_MODELS_DIR", ROOT / "models"))
    )
    ap.add_argument("--device", default="cuda")
    ap.add_argument(
        "--repeats", type=int, default=5, help="warm timing repeats per fixture and mode"
    )
    ap.add_argument(
        "--no-network",
        action="store_true",
        help="fail on any network access (offline reload check)",
    )
    ap.add_argument("--out", type=Path, help="write the JSON report here")
    args = ap.parse_args()
    if args.no_network:
        block_network()

    import numpy as np
    import soundfile as sf
    import torch

    from app.quan_loader import load_quan
    from scripts.fetch_fixtures import LARGE_DIR
    from scripts.fetch_fixtures import MANIFEST as FIXTURES_MANIFEST
    from scripts.fetch_fixtures import SMALL_DIR
    from scripts.fetch_fixtures import verify as verify_fixtures
    from scripts.stage_models import check as check_assets
    from scripts.stage_models import MANIFEST as ASSET_MANIFEST

    failures = []
    assets = json.loads(ASSET_MANIFEST.read_text())
    for name, entry in assets["models"].items():
        failures += check_assets(name, entry, args.models_dir / entry["dir"], update=False)
    fixtures = json.loads(FIXTURES_MANIFEST.read_text())
    failures += verify_fixtures(fixtures, include_large=True)
    if failures:
        print("\n".join(failures), file=sys.stderr)
        return 1

    report = {
        "environment": environment(torch),
        "network_blocked": args.no_network,
        "hf_home": os.environ.get("HF_HOME"),
        "revisions": {
            "code": assets["code"],
            "models": {k: v["revision"] for k, v in assets["models"].items()},
        },
        "decoder": DECODER,
    }
    on_cuda = torch.device(args.device).type == "cuda"
    sync = torch.cuda.synchronize if on_cuda else (lambda: None)

    t0 = time.perf_counter()
    asr = load_quan(args.models_dir, args.device)
    sync()
    load = {"cold_load_s": round(time.perf_counter() - t0, 2)} | rss_mb()
    if on_cuda:
        load["gpu"] = gpu_mem(torch)
        torch.cuda.reset_peak_memory_stats()
    report["load"] = load
    print("load", load, flush=True)

    def transcribe(wav, streaming):
        return asr.transcribe(
            wav,
            streaming=streaming,
            left_chunks=DECODER["left_chunks"],
            chunk_seconds=DECODER["chunk_seconds"],
            batch_seconds=DECODER["batch_seconds"],
            use_lm=False,
        )

    wavs = {}
    for name, e in fixtures["files"].items():
        x, sr = sf.read((SMALL_DIR if e["set"] == "small" else LARGE_DIR) / name, dtype="float32")
        assert sr == 16000 and x.ndim == 1
        wavs[name] = x

    t0 = time.perf_counter()
    transcribe(wavs["vi_a.wav"], False)
    sync()
    report["first_inference_s"] = round(time.perf_counter() - t0, 3)

    results = {}
    for name, x in wavs.items():
        ref = fixtures["files"][name]["reference"]
        for streaming in (False, True):
            texts, times = [], []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                texts.append(transcribe(x, streaming))
                sync()
                times.append(time.perf_counter() - t0)
            dur = len(x) / 16000
            med = statistics.median(times)
            r = {
                "text": texts[0],
                "deterministic": len(set(texts)) == 1,
                "audio_s": round(dur, 3),
                "median_s": round(med, 4),
                "min_s": round(min(times), 4),
                "rtf": round(med / dur, 5),
            }
            if ref:
                r["wer"] = round(wer(ref, texts[0]), 4)
            results[f"{name}:{'streaming' if streaming else 'full'}"] = r
            print(name, "streaming" if streaming else "full", r, flush=True)
            if not r["deterministic"]:
                failures.append(f"{name}: non-deterministic output across repeats")
            if ref == "" and texts[0]:
                failures.append(f"{name}: expected empty transcript, got {texts[0]!r}")
            if ref and r["wer"] > 0.5:
                failures.append(f"{name}: WER {r['wer']} looks broken")
    report["results"] = results
    if on_cuda:
        report["inference_gpu"] = gpu_mem(torch)
    report["inference_cpu"] = rss_mb()

    # Batch of two different clips must give the same text as each clip alone (no cross-talk through padding).
    a, b = wavs["vi_a.wav"], wavs["vi_b.wav"]
    batched = asr.transcribe([a, b], use_lm=False)
    report["batch_isolation"] = {
        "equal_to_single": batched
        == [results["vi_a.wav:full"]["text"], results["vi_b.wav:full"]["text"]],
        "texts": batched,
    }
    if not report["batch_isolation"]["equal_to_single"]:
        failures.append("batched [vi_a, vi_b] differs from single-clip results")

    # Frontend minimum: call the model directly (transcribe() drops pieces <= 1600 samples via split_quiet).
    speech = wavs["vi_a.wav"][32000:]
    probe = {}
    for n in SHORT_PROBE:
        try:
            probe[n] = {"text": asr._run([speech[:n]], False, None, None, 32)[0], "error": None}
        except Exception as e:
            probe[n] = {"text": None, "error": f"{type(e).__name__}: {str(e)[:200]}"}
        probe[n]["transcribe"] = transcribe(speech[:n].copy(), False)
    report["short_probe"] = probe
    report["empty_input_transcribe"] = repr(transcribe(np.zeros(0, np.float32), False))
    print("short_probe", json.dumps(probe, ensure_ascii=False), flush=True)

    report["failures"] = failures
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
        print("wrote", args.out)
    for f in failures:
        print("FAIL", f, file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
