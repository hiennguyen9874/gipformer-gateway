# Change rules by area

Read the section for each area you touch before editing it. File roles: [code-map.md](code-map.md).

## Inference path (`engine.py`, `scheduler.py`)

- `GipformerEngine` is the only code that touches sherpa-onnx: one `OfflineRecognizer`, files from the staged
  local directory only (never the Hub), one chunk per call. Settings: `GIPFORMER_PRECISION`,
  `GIPFORMER_DECODING_METHOD`, `GIPFORMER_BEAM_SIZE` (-> `max_active_paths`), `GIPFORMER_NUM_THREADS`.
- `EngineWorker` is the single thread that calls the recognizer. Work reaches it as jobs (one per chunk) on
  `Scheduler` lanes (one per request or realtime session).
- Engine text must equal `references/gipformer/infer_onnx.py` for the same audio, precision and decoding
  (`tests/model/test_engine.py`). Do not post-process the text.
- A lane holds at most `GIPFORMER_QUEUE_MAX_JOBS_PER_LANE` (2) queued chunks; long files are fed incrementally. LIVE
  lanes get bounded priority (`live_priority_ratio`).
- Cancel drops queued chunks and discards the running chunk's result. A running decode finishes; the worker and
  budgets are released only when the real call returns.

## Audio (`audio.py`, `chunking.py`)

- Decode client audio only with `AudioIngress.decode`; chunk only with `chunking.split`.
- FFmpeg gets fixed argv, the `file` protocol and the demuxer/decoder allowlists; client filenames, paths and URLs
  stay out of its arguments, out of disk paths and out of logs.
- The engine refuses fewer than 161 samples (`MIN_FORWARD_SAMPLES`, a QUAN-era floor pending the probe). Audio <= 100 ms (1,600 samples) yields `""` without a model
  call (docs/baseline.vi.md §5).
- Every resource is budgeted: body limit before the multipart parser, spool bytes, decoded audio in RAM (`hold`,
  realtime pending audio via `reserve`). Exceeding one returns 413/429/503 in the OpenAI envelope.

## Timestamps (`alignment.py`, `formats.py`)

- Times come only from the token emission times of the same inference that produced the text, plus the chunk offset.
  On mismatch raise `AlignmentError` -> 500 `alignment_failed`.
- `WORD_TIMESTAMPS = False` in `app/transcription.py` until the R2b gate passes (docs/roadmap.md §2); flip it in the
  commit that carries the gate report.

## Realtime (`realtime.py`)

- Run Uvicorn with `--ws websockets-sansio --ws-max-size 2097152`; `--ws auto` silently disables WebSockets.
- Auth, model and capacity checks happen before `accept`.
- One LIVE lane per session; committed items run FIFO through the same `split` / `run_chunks` / `Joiner` as REST.
- The JS realtime SDK connects only over `wss:`; tests use a self-signed cert via `serve(tls_dir=...)`.

## API contract (`transcription.py`, `realtime.py`, `errors.py`)

- `tests/contract/examples/` plus the SDK tests are the source of truth; docs/api-contract.vi.md describes them.
- To change the wire format, update in this order: the examples, `tests/contract/test_python_sdk.py`,
  `tests/contract/js/contract.test.ts` (and `server.test.ts` when server behaviour changes), then the contract doc
  and capability matrix.
- Omit optional fields instead of sending `null`; the JS types reject `null` for several of them.
- Example files are byte-exact (SRT/VTT/SSE end with a blank line, `transcription.txt` has no trailing newline);
  pre-commit excludes them, so keep editors from normalising them too.

## Config (`config.py`)

- Every `GIPFORMER_*` variable is a server setting, and unknown ones abort startup. Leftover `QUAN_*` variables abort
  too. `GIPFORMER_TEST_*` (JS server test) is therefore kept out of the server's environment.
- Each limit is finite and validated at startup. A new setting also gets a row in the README config table.

## Pins, dependencies, image

- Change a model pin only in `assets/manifest.json`, then run `scripts/stage_models.py --update-manifest`. The
  `sherpa-onnx` version in `pyproject.toml` and `uv.lock` must equal the manifest's `code` pin
  (`tests/unit/test_manifests.py`).
- The image contains only `app/` and the venv; models are mounted read-only, keys come from a Docker secret.
- No torch or CUDA dependency: the image is CPU only. After changing the base image, sherpa-onnx or FFmpeg pins:
  rerun unit + contract tests and the full container gate with no skips (docs/container-gate.vi.md).
- `docs/baseline/*.json` reports hold metadata only. CPU-diagnostic container reports stay uncommitted.
