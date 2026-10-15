# Trạng thái triển khai gipformer1.5-68M-rnnt

Kiến trúc: [deployment-design.vi.md](deployment-design.vi.md). Việc còn lại: [roadmap.md](roadmap.md).
Pre-release: tag `v0.1.0-rc.1`.

## 1. Release

| Release | Nội dung | Trạng thái |
|---|---|---|
| R1 — REST/SSE | models, transcription `json`/`text`, SSE, auth, lỗi OpenAI, giới hạn, Docker CPU | **Xong** trên server local (`docs/baseline/server.json`); container gate: **chưa chạy**, xem [container-gate.vi.md](container-gate.vi.md) |
| R2a — Rich output | `verbose_json` (không `segments`), `srt`, `vtt` với timestamps từ thời điểm phát token | **Xong** (gate: `docs/baseline/r2a_gate.json`) |
| R2b — Word timestamps | `timestamp_granularities[]=word` | Chưa: chờ bộ annotate tay |
| R3a — Realtime explicit commit | WS `/v1/realtime`, PCM16 24 kHz, append/commit/clear | **Xong** (gate: `docs/baseline/r3a_gate.json`) |
| R3b — Server VAD / partial | acoustic VAD, partial trước commit | Chưa làm |
| R4 — Capacity | giới hạn production đo trên CPU riêng; batching, nhiều recognizer, scale ngang | Tooling có, chưa đo |

## 2. Milestone đã hoàn tất

| Milestone | Kết quả | Kiểm chứng |
|---|---|---|
| M0 — Contract + baseline | Pins model/sherpa-onnx/SDK, engine offline, fixtures, số đo | [baseline.vi.md](baseline.vi.md), `tests/model/test_engine.py`, `tests/unit/test_manifests.py`, `tests/contract/` |
| M1 — Service + scheduler | Config typed, lifespan/health/ready, error envelope, single worker, queue công bằng có giới hạn, logs/metrics | `tests/scheduler/`, `tests/unit/test_app.py`, `tests/unit/test_config.py`, `tests/model/test_engine.py` |
| M2 — Audio ingress | Giới hạn body trước parser, spool có budget, FFmpeg sandbox, chunking tại điểm yên lặng, policy clip ngắn | `tests/audio/`, `tests/unit/test_chunking.py`, `docs/baseline/limits_probe.json` |
| M3 — REST/SSE + image | Endpoint, joiner, disconnect/send timeout, Dockerfile/compose CPU | `tests/unit/test_transcription_*`, `tests/contract/test_server_sdk.py`, `tests/model/test_server.py`, [container-gate.vi.md](container-gate.vi.md) |
| M4 (R2a) — Rich output | Alignment token emission time → cue, serializers; probe + cue gate | `tests/unit/test_alignment.py`, `tests/model/test_server.py`, `docs/baseline/alignment_probe.json`, `docs/baseline/r2a_gate.json` |
| M5 — Realtime | Handshake trước accept, session.update, turn buffer + soxr stream, item FIFO, giới hạn session | `tests/unit/test_realtime_*.py`, `tests/model/test_realtime.py`, `docs/baseline/r3a_gate.json` |
| M6 — Chuyển sang Gipformer | Thay model CTC trên GPU bằng gipformer1.5-68M-rnnt trên CPU (sherpa-onnx), đổi tên `GIPFORMER_*`, chạy lại mọi gate | Issue #1 (#2–#12), [ADR 0001](adr/0001-transducer-on-cpu-replaces-quan-ctc-on-gpu.md), §3 |
| M7 (một phần) — Hardening | `scripts/benchmark.py`, OOM/hang/recovery giả lập, soak CPU có giới hạn, redaction log | `tests/unit/test_benchmark.py`, `tests/unit/test_capacity_hardening.py` |

## 3. Gate và bằng chứng (Gipformer, CPU)

Report trước migration đã xoá; mọi bằng chứng dưới đây đo trên gipformer1.5-68M-rnnt (sherpa-onnx 1.13.8, revision
`dd9227dc`), chỉ metadata.

| Gate | Trạng thái | Bằng chứng |
|---|---|---|
| Parity engine vs `references/gipformer/infer_onnx.py` | Đạt (fp32/int8 × beam/greedy) | `tests/model/test_engine.py` |
| Clip ngắn + độ dài chunk | Đạt: sàn 1,360 mẫu, `GIPFORMER_MAX_CHUNK_SECONDS=20` | `docs/baseline/limits_probe.json`, [baseline §5](baseline.vi.md) |
| Baseline offline | Đạt: RTF/WER fp32/int8 × beam/greedy, số thread; mặc định `GIPFORMER_NUM_THREADS=4`, `GIPFORMER_QUEUE_MAX_BUFFERED_SECONDS=600` | `docs/baseline/offline.json`, [baseline §4](baseline.vi.md) |
| Server smoke (local) | Đạt: mọi format + Realtime, khớp hash baseline; `vi_long` WER 4.0% | `docs/baseline/server.json` |
| R2a cue gate | Đạt | `docs/baseline/alignment_probe.json`, `docs/baseline/r2a_gate.json`, [baseline §6](baseline.vi.md) |
| R3a realtime gate | Đạt | `docs/baseline/r3a_gate.json`, [baseline §7](baseline.vi.md) |
| Container gate (image CPU) | **Chưa chạy** (môi trường phát triển không có Docker); emulation trên host chỉ là chẩn đoán | `docs/baseline/server-container.json` (chưa có), [container-gate.vi.md](container-gate.vi.md) |
| CPU suite | Đạt: unit + scheduler + audio + contract (Python SDK `openai==3.31.0`, JS SDK `openai@7.33.0`, server HTTP/WSS thật), 445 passed | `uv run pytest tests/unit tests/scheduler tests/audio tests/contract` |

Migration (#1) chỉ được coi là xong khi container gate pass và `docs/baseline/server-container.json` được commit.

## 4. Lệch so với thiết kế ban đầu

- `verbose_json` không có `segments`: schema SDK bắt buộc các field kiểu Whisper (`avg_logprob`, `no_speech_prob`,
  `tokens`, …) mà model không có dữ liệu thật. Cue chỉ có ở `srt`/`vtt` (spec #1 ghi `segments` trong `verbose_json`).
- Realtime phát `conversation.item.added` (GA) thay vì `conversation.item.created` (beta).
- Realtime chỉ infer khi commit, qua cùng splitter/joiner với REST; không dùng session streaming của reference.
- Spec #1 dự kiến 5 commit theo giai đoạn; thực tế giao theo từng issue con (#2–#12), mỗi commit xanh trên CPU tier.
- Sàn clip ngắn giữ 100 ms (1,600 mẫu) dù sàn cứng đo được là 1,360 mẫu: 85–100 ms gần như không cho text
  ([baseline §5](baseline.vi.md)). `GIPFORMER_MAX_CHUNK_SECONDS` 30 → 20 s,
  `GIPFORMER_QUEUE_MAX_BUFFERED_SECONDS` 900 → 600 s theo số đo CPU.
- Giới hạn cue 7 s / 80 ký tự là quy ước phụ đề, không suy ra từ probe; `EMISSION_SHIFT = 0` vì probe không thấy độ
  trễ cố định ([baseline §6](baseline.vi.md)).
- Container gate chưa chạy được trong môi trường phát triển (không có Docker); xem §3.

## 5. Definition of done

Mỗi tính năng phát hành có code + test + tài liệu, cleanup/cancel được kiểm thử, không log dữ liệu nhạy cảm, pins
cố định, capability matrix cập nhật và gate có bằng chứng. Tính năng chưa qua gate thì không bật.
