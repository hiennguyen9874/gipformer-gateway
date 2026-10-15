# QUAN-Transcribe gateway

Server nhận dạng tiếng nói tiếng Việt cho [QUAN-Transcribe-1.0](https://huggingface.co/qnguyen3/QUAN-Transcribe-1.0),
triển khai bằng Docker, với một **tập con** API OpenAI:

- `GET /v1/models`, `GET /v1/models/{id}`
- `POST /v1/audio/transcriptions`: `json`, `text`, SSE (`stream=true`), `verbose_json`, `srt`, `vtt`
- `GET /v1/realtime`: Realtime transcription qua WebSocket, commit tường minh (`turn_detection: null`)

Không phải tương thích OpenAI đầy đủ. Payload, mã lỗi và capability matrix: [docs/api-contract.vi.md](docs/api-contract.vi.md).

Giới hạn cần biết:

- Output là text gốc của model: chữ thường, không dấu câu, số viết bằng chữ. Chỉ `language=vi`; không có logprobs.
- `srt`/`vtt` có thời gian từ đường CTC greedy (start của cue có thể trễ ~1 âm tiết). `verbose_json` không có
  `segments`; word timestamps chưa bật.
- Realtime: transcript chỉ có sau `input_audio_buffer.commit`; không có partial, không có VAD.
- Alias `whisper-1` (tắt mặc định, `QUAN_WHISPER_ALIAS=true`) chỉ để client cũ chạy được: vẫn là model QUAN.

Tài liệu: [kiến trúc](docs/deployment-design.vi.md) · [API contract](docs/api-contract.vi.md) ·
[baseline và pins](docs/baseline.vi.md) · [kiểm định image](docs/container-gate.vi.md) ·
[trạng thái](docs/implementation-plan.vi.md) · [roadmap](docs/roadmap.md)

## 1. Yêu cầu

- Linux, Docker Engine + Compose v2, NVIDIA Container Toolkit, một GPU NVIDIA (~2.5 GB VRAM cho mỗi instance).
- [uv](https://docs.astral.sh/uv/) trên host, chỉ để tải model (staging) và chạy smoke test.
- Chấp nhận [Gemma Terms of Use](https://ai.google.dev/gemma/terms) (weights QUAN và base `google/embeddinggemma-2`).

Chọn **một** CUDA profile theo GPU:

| Profile | PyTorch | GPU | Driver Linux tối thiểu |
|---|---|---|---|
| `cu126` (mặc định) | 2.14.1+cu126 | tới `sm_90` (A100, H100, …) | 560.28.03 |
| `cu128` | 2.11.0+cu128 | thêm Blackwell `sm_120` (RTX 5090, …) | 570.26 |

Model và code pins giống nhau ở cả hai profile. Không cần cài CUDA toolkit trên host.

## 2. Triển khai

```bash
# 1. Tải model đã pin vào ./models và kiểm tra SHA256 (một lần, cần mạng)
uv run --extra cu126 scripts/stage_models.py          # HF_TOKEN=<token> nếu repo yêu cầu
uv run --extra cu126 scripts/stage_models.py --verify # kiểm tra lại, không cần mạng

# 2. API key: một key mỗi dòng, ≥ 16 ký tự, đọc được bởi UID 10001 trong container
mkdir -p secrets
python3 -c "import secrets; print(secrets.token_urlsafe(32))" > secrets/quan_api_keys
sudo chown 10001 secrets/quan_api_keys && sudo chmod 0400 secrets/quan_api_keys

# 3. Build + chạy trên một GPU (nvidia-smi để chọn GPU trống)
QUAN_GPU=0 docker compose up -d --build                         # cu126
QUAN_BUILD_CUDA=cu128 QUAN_GPU=0 docker compose up -d --build   # Blackwell

# 4. Chờ sẵn sàng (load + warm-up)
curl -s 127.0.0.1:18000/ready                                   # {"status":"ready"}

# 5. Thử nhanh với file mẫu có sẵn trong repo (tests/fixtures/vi_a.wav, 8.9 s, FLEURS, CC BY 4.0)
curl -s 127.0.0.1:18000/v1/audio/transcriptions \
    -H "Authorization: Bearer $(sudo head -1 secrets/quan_api_keys)" \
    -F model=quan-transcribe-1.0 -F file=@tests/fixtures/vi_a.wav
# {"text":"các sản phẩm thủ công có thể được coi là độ cổ mặc dù tuổi đời của chúng ch trẻ hơn rất nhiều ..."}

# 6. Smoke test (tùy chọn): mọi format + Realtime, so với baseline
OPENAI_API_KEY=$(sudo cat secrets/quan_api_keys | head -1) \
    uv run --extra cu126 scripts/smoke_server.py --base-url http://127.0.0.1:18000
```

`uv run` luôn kèm đúng một `--extra cu126` hoặc `--extra cu128` (không chạy `uv` trần, không `--all-extras`).
`QUAN_BUILD_CUDA` và `QUAN_GPU` chỉ dùng cho compose.

Image (`quan-transcribe:r1-<profile>`): Ubuntu 24.04 (pin digest), Python 3.12.14, FFmpeg 6.1.1, user `10001`, chỉ
chứa `app/` và venv. Không có model, `HF_TOKEN` hay API key trong image; server không truy cập Hub.
`compose.yaml` chạy:

- model mount `./models:/models:ro`, key qua Docker secret `/run/secrets/quan_api_keys`;
- rootfs read-only, tmpfs `/tmp` 512 MiB (spool upload), `HF_HUB_OFFLINE=1`;
- port chỉ bind `127.0.0.1:18000`; `restart: unless-stopped`; `stop_grace_period: 70s`.

**Một container = một model = một GPU**, luôn một Uvicorn worker. Nhiều GPU: chạy nhiều container (mỗi container
một `QUAN_GPU`, một port host) sau load balancer; WebSocket cần sticky routing.

**TLS:** gateway không làm TLS, rate limit theo client hay CORS. Khi expose ra ngoài localhost, đặt sau hạ tầng
HTTPS của bạn với: SSE không buffer, hỗ trợ WebSocket upgrade, read timeout ≥ `QUAN_REQUEST_TIMEOUT_SECONDS`
(600 s), body ≥ `QUAN_MAX_UPLOAD_BYTES`. JS SDK Realtime chỉ kết nối `wss:`.

## 3. Gọi API

```bash
KEY=$(sudo cat secrets/quan_api_keys | head -1)
curl -s 127.0.0.1:18000/v1/models -H "Authorization: Bearer $KEY"
curl -s 127.0.0.1:18000/v1/audio/transcriptions -H "Authorization: Bearer $KEY" \
    -F model=quan-transcribe-1.0 -F file=@tests/fixtures/vi_a.wav
curl -sN 127.0.0.1:18000/v1/audio/transcriptions -H "Authorization: Bearer $KEY" \
    -F model=quan-transcribe-1.0 -F stream=true -F file=@tests/fixtures/vi_a.wav
curl -s 127.0.0.1:18000/v1/audio/transcriptions -H "Authorization: Bearer $KEY" \
    -F model=quan-transcribe-1.0 -F response_format=srt -F file=@tests/fixtures/vi_a.wav   # text, vtt, verbose_json
```

OpenAI Python SDK (đã kiểm thử `openai==3.31.0`; JS `openai@7.33.0`):

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:18000/v1", api_key="<QUAN key>")
with open("audio.mp3", "rb") as f:
    print(client.audio.transcriptions.create(file=f, model="quan-transcribe-1.0").text)
with open("audio.mp3", "rb") as f:
    for event in client.audio.transcriptions.create(file=f, model="quan-transcribe-1.0", stream=True):
        print(event.type, getattr(event, "delta", None) or getattr(event, "text", ""))
```

Realtime (`openai[realtime]`): PCM16 little-endian mono 24 kHz, base64; nhiều commit trên một socket, mỗi commit là
một item với các `delta` (một mỗi chunk ≤ 30 s) rồi `completed`.

```python
import base64

with client.realtime.connect(model="quan-transcribe-1.0") as conn:
    conn.recv()                                            # session.created
    for packet in pcm16_24k_packets:                       # bytes, ví dụ 20–200 ms mỗi packet
        conn.input_audio_buffer.append(audio=base64.b64encode(packet).decode())
    conn.input_audio_buffer.commit()
    for event in conn:
        if event.type == "conversation.item.input_audio_transcription.completed":
            print(event.transcript)
            break
```

Audio: WAV, FLAC, MP3, MP4/M4A (AAC/ALAC), OGG (Vorbis/Opus/FLAC), WEBM/Matroska, nhận diện theo nội dung. Audio dài
được cắt thành chunk ≤ 30 s tại điểm yên lặng; SSE gửi một delta mỗi chunk. Audio ≤ 100 ms → `""`. `srt`/`vtt`/
`verbose_json` không stream (`stream=true` → 400). Lỗi dùng envelope OpenAI; danh sách mã lỗi: contract §7.

## 4. Cấu hình

Thêm biến `QUAN_*` vào `environment:` của service (ví dụ trong `compose.override.yaml`). Biến lạ, giá trị ≤ 0 hoặc
không nhất quán → container không start. Đầy đủ: `app/config.py`.

| Biến | Mặc định | Ý nghĩa |
|---|---|---|
| `QUAN_API_KEYS_FILE` / `QUAN_API_KEYS` | (bắt buộc) | Key Bearer (compose dùng file secret) |
| `QUAN_WHISPER_ALIAS` | `false` | Chấp nhận model id `whisper-1` |
| `QUAN_LOG_LEVEL` | `INFO` | Mức log |
| `QUAN_MAX_UPLOAD_BYTES` | 25 MiB | Cả body multipart → 413 |
| `QUAN_MAX_AUDIO_SECONDS` | 1800 | Thời lượng sau decode → 400 `audio_too_long` |
| `QUAN_MAX_DECODED_SECONDS` | 3600 | Audio đã decode giữ trong RAM của mọi request (≈ 64 KB/s) → 429 |
| `QUAN_MAX_SPOOL_BYTES` | 200 MiB | File tạm của upload đang nhận (nằm trong tmpfs 512 MiB) → 429 |
| `QUAN_MAX_CHUNK_SECONDS` | 30 | Độ dài chunk (giữ 30 để giống baseline) |
| `QUAN_QUEUE_MAX_LANES` | 32 | Request/session đồng thời → 429 |
| `QUAN_QUEUE_TIMEOUT_SECONDS` | 30 | Chờ chỗ trong queue tối đa → 429 |
| `QUAN_REQUEST_TIMEOUT_SECONDS` | 600 | Tổng upload + decode + queue + infer + gửi → 503 `request_timeout` |
| `QUAN_SEND_TIMEOUT_SECONDS` | 10 | Một lần gửi SSE/WebSocket; client không đọc → ngắt |
| `QUAN_MAX_SESSIONS` | 16 | Realtime session đồng thời (≤ lanes) → handshake 429 |
| `QUAN_MAX_SESSION_SECONDS` | 3600 | Tuổi tối đa một session → `session_expired` |
| `QUAN_SESSION_IDLE_TIMEOUT_SECONDS` | 120 | Session không hoạt động → `idle_timeout` |
| `QUAN_MAX_TURN_SECONDS` | 120 | Audio chưa commit của một session → `input_audio_buffer_full` |
| `QUAN_MAX_PENDING_ITEMS` | 4 | Item đã commit chưa xong → `too_many_pending_items` |
| `QUAN_INFER_HANG_SECONDS` | 120 | Một call inference lâu hơn → worker "hung", `/health` 503 |
| `QUAN_SHUTDOWN_DRAIN_SECONDS` | 30 | Chờ job đang chạy khi tắt |

Các mặc định là giới hạn an toàn hữu hạn, **chưa phải capacity đã đo** (xem [roadmap](docs/roadmap.md) R4). Đo
trên phần cứng của bạn trước khi tăng.

## 5. Vận hành

- **Readiness/liveness:** `/ready` 200 khi model đã load + warm-up và không đang drain/recover (dùng cho load
  balancer). `/health` 503 khi load/recovery lỗi hoặc một call inference treo quá `QUAN_INFER_HANG_SECONDS`.
  Docker không tự restart container chỉ vì `unhealthy`: dùng watchdog/orchestrator (autoheal, systemd, Kubernetes
  `livenessProbe`). Lỗi inference đơn lẻ trả 500 `inference_failed`, worker tự warm-up lại.
- **Dừng/cập nhật:** `docker compose stop` (SIGTERM) ngừng nhận kết nối mới, request đang chạy được làm xong (≤ 30 s),
  drain queue (≤ 30 s); log `server.stopped` có `drained`. WebSocket **không** được drain: đóng với code 1012, item
  chưa `completed` bị mất, client kết nối lại và gửi lại audio. Rolling update: bỏ instance khỏi load balancer trước.
- **Rollback:** giữ image digest + CUDA profile của bản tốt, chạy lại digest đó với cùng `./models`; chờ `/ready`
  và chạy smoke trước khi mở traffic.
- **Xoay vòng key** (key chỉ đọc lúc khởi động): thêm key mới vào file, restart, chuyển client, bỏ key cũ, restart.
- **Log/metrics:** log JSON ra stderr (`docker compose logs`), chỉ metadata (`request_id`, thời lượng, outcome);
  không ghi audio, transcript, key, tên file. `/metrics` (Prometheus, cần API key; không expose public):
  `quan_transcriptions_total`, `quan_queue_wait_seconds`, `quan_infer_seconds`, `quan_ready`, `quan_worker_hung`,
  `quan_realtime_sessions`, `quan_realtime_commit_to_completed_seconds`, …
- `HF_TOKEN` chỉ dùng lúc staging, không đưa vào container.

## 6. Kiểm thử

```bash
uv run --extra cu126 pytest                                         # CPU: unit, scheduler, audio, contract
QUAN_GPU=0 uv run --extra cu128 pytest tests/container --container-cuda=cu128   # gate image trên GPU
```

Chi tiết gate image: [docs/container-gate.vi.md](docs/container-gate.vi.md). Lệnh phát triển: [docs/agents/commands.md](docs/agents/commands.md).
