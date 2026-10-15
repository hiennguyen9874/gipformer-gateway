# Kiểm định Docker image

Gate phải đạt trước khi phát hành một image: `tests/container/test_image.py`. Cách triển khai: [README](../README.md).

## 1. Trạng thái

| Image | Runtime | Kết quả | Bằng chứng |
|---|---|---|---|
| `gipformer-gateway` (CPU) | sherpa-onnx (pin trong `uv.lock`), không torch/CUDA | **Chưa đạt: chưa chạy** (môi trường phát triển không có Docker) | `docs/baseline/server-container.json` (chưa có) |

Report của image QUAN (`server-container-cu128.json`) đã xoá cùng QUAN; không dùng làm bằng chứng cho image này.

Kiểm tra thay thế đã chạy (chỉ chẩn đoán, **không thay gate**, không commit report): venv dựng như builder
(`uv sync --frozen --no-dev`, Python 3.12.14, ~166 MB, không có `torch`/`transformers`) + chỉ `app/`, model copy
read-only, key qua file, `HF_HUB_OFFLINE=1`, chạy trên host không có network namespace riêng. Các script của gate
(`smoke_server.py`, auth/lỗi/overload/isolation, SIGTERM khi đang stream → `done` + `drained: true`, restart → ready)
đều đạt, key không xuất hiện trong log. Gate thật vẫn cần chạy trên host có Docker.

## 2. Chạy gate

Cần Docker, model đã stage (`./models`, `scripts/stage_models.py`) và fixtures (`scripts/fetch_fixtures.py`, gồm
`fixtures-large/vi_long.wav`). Không cần GPU.

```bash
uv run pytest tests/container -v      # build gipformer-gateway:test rồi chạy cả gate; phải pass hết, không skip
```

Test chỉ skip khi không có `docker`; thiếu model hoặc `vi_long.wav` là fail. Smoke ghi report vào
`docs/baseline/server-container.json` (chỉ metadata: thời gian, số cue, hash transcript, image id, commit; không có
transcript, audio hay key). Commit report cùng bản cập nhật bảng §1 khi gate pass trên cây sạch
(`source_dirty: false`).

## 3. Nội dung gate

| Test | Kiểm tra |
|---|---|
| Image | Build; chạy UID 10001; `/srv` chỉ có `app/` và `.venv`; không có assets, references, models, tests, `.git`, fixtures; Python 3.12.14, FFmpeg 6.1.1; `import sherpa_onnx` được, không có `torch`/`transformers` |
| Metadata | `docker inspect`/`history` không chứa `HF_TOKEN`, `GIPFORMER_API_KEYS*` hay giá trị `HF_TOKEN` |
| Offline + smoke | Container `--network none`, rootfs read-only, model mount read-only, key qua file secret (như compose) → `/ready`; `scripts/smoke_server.py` trên 5 fixture: json/text/SSE/verbose_json/SRT/VTT và Realtime, hash transcript khớp `docs/baseline/offline.json` |
| Auth/lỗi/overload/isolation | 401, model 404, audio lỗi 400, 429 khi hết lane; hai request đồng thời không trộn transcript; key không có trong log |
| Shutdown + restart | SIGTERM khi đang stream 6 × `vi_long.wav` → SSE `done`, exit 0, `drained: true`; restart → ready |

`compose.yaml` không có cấu hình GPU; Dockerfile/compose không nhắc CUDA/torch (`tests/unit/test_manifests.py`).

Khi đổi base image, sherpa-onnx, FFmpeg pins, Dockerfile hoặc compose: chạy lại unit/contract và toàn bộ gate, không
skip.
