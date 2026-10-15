# Roadmap

Current release: `v0.1.0-rc.1` (pre-release). Details of what ships: [implementation-plan.vi.md](implementation-plan.vi.md).

## Shipped

| Release | Scope |
|---|---|
| R1 | `/v1/models`, `POST /v1/audio/transcriptions` (`json`, `text`, SSE), auth, OpenAI errors, limits, Docker (one GPU) |
| R2a | `verbose_json` (no `segments`), `srt`, `vtt` with timestamps from the greedy CTC path |
| R3a | Realtime transcription over WebSocket `/v1/realtime`, explicit commit (`turn_detection: null`) |

## Not implemented yet

Ordered by priority. A feature is enabled only after its gate passes with recorded evidence.

### 1. Production capacity and limits (R4)

Current defaults (`QUAN_QUEUE_MAX_LANES=32`, `QUAN_MAX_SESSIONS=16`, …) are finite safety budgets, **not measured
capacity**. Tooling exists (`scripts/benchmark.py`); the measurement does not.

- [ ] Benchmark on an **exclusive** GPU: REST short/long, SSE, WebSocket commits and mixed, ramping clients;
  ≥ 3 runs per level, ≥ 100 successes per group for p95. Fix workload, p95 SLO and error budget before measuring.
- [ ] Soak ≥ 1 h at the chosen level behind real TLS; RSS/VRAM headroom, no occupancy growth.
- [ ] Failure drills on the real deployment: SIGTERM drain, process/container kill and restart, CUDA OOM recovery,
  hung-worker watchdog restart.
- [ ] Set limits from the measurements, record a decision table (hardware, image digest, config, max tested load,
  selected limit) and a rollback digest.
- [ ] cu126 GPU container gate on a compatible GPU (A100/H100); so far only cu128 passed on GPU.

```bash
uv run --extra cu128 scripts/benchmark.py --engine gpu --hardware '<gpu>-exclusive-cu128' \
  --base-url http://127.0.0.1:18000 --clients 1,2,4,8,16,32 --requests-per-client 100 \
  --workload mixed --phase warm --server-pid <host pid> --out docs/baseline/capacity-mixed-warm.json
```

A benchmark report (exit 0) is not a capacity claim; client counts are closed-loop, not microphone sessions.

**Diagnostic run (not gate evidence).** One run per level at `8cffbe7` on a shared host (A100-PCIE-40GB, cu126,
local server, no proxy), `--workload short|long`, 30 requests/client, 0 errors/overloads. WS, SSE and mixed workloads, ≥ 3 runs
and soak were not done.

| Clients | Short 8.88 s: audio s/s · p50 · p95 | Long 106.62 s: audio s/s · p50 · p95 |
|---:|---|---|
| 1 | 71 · 0.12 s · 0.13 s | 279 · 0.39 s · 0.40 s |
| 2 | 128 · 0.13 s · 0.16 s | 412 · 0.50 s · 0.62 s |
| 4 | 137 · 0.25 s · 0.28 s | 421 · 1.00 s · 1.14 s |
| 8 | 132 · 0.52 s · 0.62 s | 426 · 1.99 s · 2.08 s |
| 16 | 136 · 1.03 s · 1.16 s | 413 · 4.10 s · 4.36 s |

Throughput saturates at ~2 clients; beyond that latency grows linearly with queue wait. At saturation the GPU was
~38 % utilized while the server process used ~470 % CPU: the single model thread is launch-bound (~50 ms per call
regardless of chunk length, see [baseline.vi.md](baseline.vi.md) §4), so §4 optimizations raise this ceiling.

### 2. Word timestamps (R2b)

`timestamp_granularities[]=word` returns 400 until this gate passes (`WORD_TIMESTAMPS = False` in
`app/transcription.py`). Alignment code and the evaluator (`scripts/eval_alignment.py`) exist.

- [ ] Hand-annotated word boundaries in `tests/fixtures/alignment/` (Praat/Audacity, not another aligner):
  ≥ 200 words, ≥ 10 utterances, `dev`/`test` splits.
- [ ] Gate on `test`: ≥ 90 % of matched words within 100 ms, median error ≤ 50 ms (thresholds fixed in advance).
- [ ] Flip the switch in the commit that carries the report; add `words` to `verbose_json`.

### 3. Server VAD and partial transcripts (R3b)

- [ ] Acoustic VAD (pin model, check license and CPU cost) with `threshold`, `prefix_padding_ms`,
  `silence_duration_ms`; emit `input_audio_buffer.speech_started/stopped` and auto-commit. Do not reuse the CTC
  blank endpoint as VAD. `semantic_vad` stays rejected.
- [ ] Optional: partial text before commit via a stable-prefix decoder (append-only deltas, defined final
  correction policy), only with full integration tests.

### 4. Scale and performance

- [ ] Multiple GPUs: one container per GPU behind a load balancer with sticky WebSocket routing.
- [ ] REST micro-batching of chunks from different requests. Measured basis: one call with 8 × 8.9 s chunks takes
  60.5 ms vs ~50 ms for one chunk (≈ 6× throughput headroom on A100); requires parity and per-request deadlines.
- [ ] Cut per-call launch overhead (~2,440 kernel launches, ~14 ms GPU time out of ~50 ms): CUDA graphs or
  `torch.compile` with fixed chunk shapes; gate on identical text vs `docs/baseline/offline.json`.
- [ ] Optional KenLM decoding for final transcripts (opt-in, measured first; timestamps need forced alignment).

### 5. Smaller items

- [ ] `conversation.item.created` in Realtime if a real client needs it (currently `conversation.item.added`).
- [ ] G.711 `audio/pcmu` / `audio/pcma` Realtime input formats.
- [ ] Browser access: ephemeral client secrets or a token service (today: server-to-server Bearer only).
- [ ] Optional punctuation / inverse text normalization as a separate, versioned post-processor.
- [ ] Python 3.11 support (only 3.12 tested).

## Out of scope

Not provided by this model; would need other engines behind the gateway:

- Languages other than Vietnamese, language auto-detection, `/v1/audio/translations`.
- `/v1/audio/speech` (TTS), chat/responses, Realtime conversation, WebRTC/SIP.
- Diarization (`diarized_json`, `known_speaker_*`), `prompt`/keywords biasing, `include[]=logprobs`.
- `verbose_json.segments`: the SDK schema requires Whisper-only fields (`avg_logprob`, `no_speech_prob`, `tokens`)
  that QUAN cannot fill with real data.
- TLS, CORS and per-client rate limiting inside the gateway: provided by the infrastructure in front of it.
