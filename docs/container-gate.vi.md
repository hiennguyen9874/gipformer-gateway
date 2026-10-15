# Kiểm định Docker image

Gate phải đạt trước khi phát hành một image: `tests/container/test_image.py`. Cách triển khai: [README](../README.md).

## 1. Trạng thái

| Profile | PyTorch | Kết quả | Bằng chứng |
|---|---|---|---|
| `cu128` | 2.11.0+cu128 | **Đạt trên GPU** (RTX 5090, driver 575.64.03): 6 passed, không skip | [`baseline/server-container-cu128.json`](baseline/server-container-cu128.json) |
| `cu126` | 2.14.1+cu126 | Chưa chạy GPU gate trong container (host chỉ có RTX 5090, wheel cu126 không có `sm_120`); 6 passed ở chế độ CPU diagnostic | — |

cu126 cần chạy lại GPU gate trên GPU tương thích (ví dụ A100) trước khi phát hành image cu126. Chế độ CPU không
thay thế GPU gate; report CPU (`server-container-cpu*.json`) không được commit.

## 2. Chạy gate

Cần Docker, NVIDIA Container Toolkit, một GPU trống, model đã stage (`./models`) và fixtures
(`scripts/fetch_fixtures.py`, gồm `fixtures-large/vi_long.wav`).

```bash
GIPFORMER_GPU=<gpu> uv run --extra cu128 --offline pytest tests/container --container-cuda=cu128 -v
GIPFORMER_GPU=<gpu> uv run --extra cu126 --offline pytest tests/container -v                    # GPU sm_≤90
uv run --extra cu126 --offline pytest tests/container --container-device=cpu -v           # chẩn đoán, không phải gate
```

Report GPU ghi vào `docs/baseline/server-container[-cu128].json` (chỉ metadata, không transcript).

## 3. Nội dung gate

| Test | Kiểm tra |
|---|---|
| Image | Build với profile đích; chạy UID 10001; `/srv` chỉ có `app/` và `.venv`; không có references, models, tests, `.git`, secrets; Python 3.12.14, FFmpeg 6.1.1, đúng phiên bản torch/CUDA |
| Offline + smoke | Container `--network none`, rootfs/models read-only, key qua file secret → `/ready`; `scripts/smoke_server.py` trên 5 fixture: json/text/SSE/verbose_json/SRT/VTT và Realtime khớp baseline |
| Auth/lỗi/overload/isolation | 401, model 404, audio lỗi 400, 429 khi hết lane; hai request đồng thời không trộn transcript |
| Shutdown + restart | SIGTERM khi đang stream 6 × `vi_long.wav` → SSE `done`, exit 0, `drained: true`; restart → ready |

Khi đổi torch/CUDA/base image/FFmpeg pins: chạy lại unit/contract và toàn bộ gate với profile đích, không skip.
