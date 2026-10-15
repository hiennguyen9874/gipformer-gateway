# Baseline inference và pins — QUAN-Transcribe-1.0

Số đo nền của model (ngoài gateway), các pin tái lập và giới hạn frontend. Đây là smoke/baseline trên vài fixture,
**không phải benchmark năng lực phục vụ** và không phải đánh giá WER.

## 1. Pins

| Thành phần | Pin |
|---|---|
| Code `quan-transcribe` | `github.com/qnguyen3/QUAN-Transcribe@88ddcd2130a3c355d081bc002439f29e5e849060` (`pyproject.toml`, `uv.lock`) |
| Weights `qnguyen3/QUAN-Transcribe-1.0` | `a08726d9fffeb5f16fc1ab05eecc8ad0b86af5a0` (không stage `best.pt`) |
| Base `google/embeddinggemma-2` | `914f7f89142e33e77833254d9c9b90c3cef7303b` (encoder, feature extractor, tokenizer) |
| Fixtures `google/fleurs` | `70bb2e84b976b7e960aa89f1c648e09c59f894dd` (vi_vn dev, CC BY 4.0) |
| Python / transformers | 3.12.14 / 5.19.0 |
| torch | `cu126`: 2.14.1+cu126 · `cu128`: 2.11.0+cu128 |
| OpenAI SDK (contract) | Python `openai==3.31.0`, JS `openai@7.33.0` |

`assets/manifest.json` là nguồn pin duy nhất cho model (repo, commit, size, SHA256 từng file);
`tests/unit/test_manifests.py` kiểm tra commit QUAN trong `pyproject.toml` khớp manifest. Đổi pin: sửa manifest rồi
chạy `scripts/stage_models.py --update-manifest`.

**License:** weights QUAN và base theo [Gemma Terms of Use](https://ai.google.dev/gemma/terms); repo không phân phối
weights. Lúc staging, cả hai repo đều không bị gate (không cần `HF_TOKEN`); `stage_models.py` vẫn đọc `HF_TOKEN` nếu có.

## 2. Loader offline

`app/quan_loader.load_quan(models_dir, device)` lặp lại các bước của `QuanTranscribe.from_pretrained` (dùng lại
`QuanCTC`, `load_encoder`, `LogMel` của reference, không sửa reference) nhưng trỏ base model/tokenizer vào thư mục đã
stage. Lý do: `from_pretrained` resolve base theo Hub `main`, không pin được. Parity
(`tests/gpu/test_loader_parity.py`, cần GPU + mạng): state dict, token pieces, logits full/streaming `torch.equal`.

## 3. Fixtures

`scripts/fetch_fixtures.py` tạo lại tất định từ FLEURS và kiểm tra SHA256 (`tests/fixtures/manifest.json`).

| File | Nội dung |
|---|---|
| `tests/fixtures/vi_a.wav`, `vi_b.wav` | 2 clip FLEURS khác nội dung (8.88 s, 8.46 s) + transcript tham chiếu |
| `tests/fixtures/silence_2s.wav` | 2 s số 0 |
| `tests/fixtures/short_200ms.wav` | 200 ms cắt giữa câu của `vi_a` |
| `fixtures-large/vi_long.wav` (gitignored) | 10 câu nối nhau, 106.62 s (đi qua splitter) |

FLEURS vi_vn không có speaker id: `vi_a`/`vi_b` chắc chắn khác nội dung, không chắc khác người nói.

## 4. Số đo (A100-PCIE-40GB, cu126, offline)

Greedy, không LM, chunk 30 s, autocast bf16. Median 5 lần sau warm-up. Report: [`baseline/offline.json`](baseline/offline.json)
(text ở đây là chuẩn so sánh cho smoke và GPU tests).

| Chỉ số | Giá trị |
|---|---|
| Cold load (file trong page cache → model trên GPU) | 4.9 s |
| Inference đầu tiên (8.88 s audio) | 0.77 s |
| Peak CPU RSS / RSS sau load | 2,537 MB / 1,131 MB |
| GPU sau load: allocated / reserved | 1,361 / 1,384 MB |
| GPU peak (≤ 106 s audio/lần): allocated / reserved | 2,130 / 2,424 MB |

| Fixture | Audio | Full: median | Full: WER | Streaming mask: median | Streaming mask: WER |
|---|---:|---:|---:|---:|---:|
| `vi_a.wav` | 8.88 s | 55 ms | 6.3% | 53 ms | 6.3% |
| `vi_b.wav` | 8.46 s | 56 ms | 4.0% | 54 ms | 16.0% |
| `vi_long.wav` | 106.62 s | 124 ms | 5.2% | 106 ms | 7.0% |
| `silence_2s.wav` | 2.00 s | 53 ms → `""` | — | 51 ms → `""` | — |
| `short_200ms.wav` | 0.20 s | 53 ms → `""` | — | 50 ms → `""` | — |

WER thô so với transcript FLEURS, chỉ để phát hiện model hỏng. Output tất định qua các lần lặp; batch hai clip cho
text giống chạy riêng; online = offline. Có sàn ~50 ms mỗi lời gọi, chi phối với utterance ngắn. Gateway chỉ dùng
chế độ full-context.

**Nguồn gốc sàn ~50 ms** (tái kiểm tại `8cffbe7`, cùng máy; số liệu ở trên vẫn khớp). Mỗi lời gọi bị giới hạn bởi
kernel launch, không bởi compute GPU:

| Đo `_forward` (A100, cu126, bf16, median 10 lần) | Giá trị |
|---|---:|
| 1 s / 8.9 s / 30 s audio, batch 1 | 51.7 / 50.4 / 52.5 ms |
| Batch 8 × 8.9 s, một lời gọi | 60.5 ms |
| Kernel launch mỗi lời gọi | ~2,440 |
| Thời gian kernel trên GPU (torch profiler, 8.9 s) | ~14 ms |

Trên gateway, REST còn cộng thêm decode FFmpeg (~69 ms/request trung bình, `quan_decode_seconds`), nên REST
8.88 s ≈ 125 ms trong khi Realtime commit → completed ≈ 57 ms. Hệ quả cho tải: [roadmap.md](roadmap.md) §1, §4.

## 5. Giới hạn frontend và clip ngắn

| Số mẫu 16 kHz | Gọi model trực tiếp |
|---:|---|
| 1 … 160 | **RuntimeError** (cần ≥ 161 mẫu cho một frame log-mel) |
| 161 … 3,200 | `""` |
| 8,000 (0.5 s) | `"có thể"` |

Policy của gateway: không bao giờ forward waveform < 161 mẫu; audio ≤ 1,600 mẫu (100 ms) → `""` không gọi model.
Silence dài vẫn qua model và cho `""` (không dùng ngưỡng biên độ). Splitter giữ cả phần đuôi ngắn (khác
`split_quiet` của reference, vốn bỏ đuôi ≤ 100 ms).

## 6. Tái tạo

```bash
uv sync --extra cu126
uv run --extra cu126 scripts/stage_models.py
uv run --extra cu126 scripts/fetch_fixtures.py
CUDA_VISIBLE_DEVICES=<gpu> HF_HOME=$(mktemp -d) uv run --extra cu126 --offline scripts/smoke_inference.py \
    --no-network --out docs/baseline/offline.json
```

`--no-network` đặt `HF_HUB_OFFLINE=1` và chặn mọi `socket.connect`/`getaddrinfo` trong process.

## 7. Reports trong `docs/baseline/`

| File | Nội dung |
|---|---|
| `offline.json` | Baseline inference ở trên (chuẩn text cho smoke/GPU tests) |
| `server.json` | Smoke gateway local trên GPU: mọi format + Realtime |
| `server-container-cu128.json` | Smoke trong GPU container gate (chỉ metadata) |
| `alignment_probe.json` | Hình học frame CTC và độ trễ spike, căn cứ cho timestamps `srt`/`vtt` |

Chưa đo: LM/KenLM, Python 3.11, cold load khi page cache lạnh, tải đồng thời (xem [roadmap.md](roadmap.md)).
