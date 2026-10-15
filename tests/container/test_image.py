"""R1 container gate (docs/container-gate.vi.md): build, non-root, no secrets or extra files in the image, offline
start to /ready with the models mounted read-only, smoke against the baseline, graceful SIGTERM, restart.

Needs Docker; the runtime checks also need the NVIDIA Container Toolkit, a free GPU and staged models:
    GIPFORMER_GPU=<free gpu id> uv run --extra cu128 pytest tests/container --container-cuda=cu128
The gateway runs with `--network none`; requests come from a sidecar container sharing its network namespace, with
this repository mounted read-only (scripts/smoke_server.py + fixtures). The smoke report is written to
docs/baseline/server-container[-cu128].json (metadata only).

CPU diagnostics (not the GPU release gate):
    uv run --extra cu126 pytest tests/container --container-device=cpu
These use a smaller drain workload and write server-container-cpu.json, never the GPU gate report.
"""

import json
import os
import secrets
import selectors
import shutil
import subprocess
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("GIPFORMER_MODELS_DIR", ROOT / "models")).resolve()
IMAGE = "gipformer-gateway:test"
KEY = "ctr-" + secrets.token_hex(16)
pytestmark = [pytest.mark.container,
              pytest.mark.skipif(shutil.which("docker") is None, reason="needs docker")]


def docker(*args: str, check: bool = True, timeout: float = 120, **kw) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], check=check, capture_output=True, text=True, timeout=timeout, **kw)


@pytest.fixture(scope="module")
def image(container_cuda) -> str:
    tag = f"{IMAGE}-{container_cuda}"
    docker("build", "--build-arg", f"TORCH_CUDA={container_cuda}", "-t", tag, str(ROOT), timeout=3600)
    return tag


def run_in_image(image: str, script: str) -> str:
    return docker("run", "--rm", "--network", "none", "--entrypoint", "sh", image, "-c", script).stdout


def test_runs_as_non_root(image):
    assert run_in_image(image, "id -u").strip() == "10001"


def test_image_contains_only_the_app(image, container_cuda):
    listing = run_in_image(image, "ls -A /srv").split()
    assert sorted(listing) == [".venv", "app"]
    for path in ("/srv/assets", "/srv/references", "/srv/models", "/srv/tests", "/srv/.git", "/srv/fixtures-large", "/models"):
        assert run_in_image(image, f"test -e {path} && echo present || echo absent").strip() == "absent", path
    versions = run_in_image(image, "python --version; ffmpeg -version | head -1")
    assert "Python 3.12.14" in versions and "ffmpeg version 6.1.1" in versions
    torch_version = run_in_image(image, 'python -c "import torch; print(torch.__version__, torch.version.cuda)"').strip()
    assert torch_version == {"cu126": "2.14.1+cu126 12.6", "cu128": "2.11.0+cu128 12.8"}[container_cuda]


def test_no_secrets_in_image_metadata(image):
    meta = docker("inspect", image).stdout + docker("history", "--no-trunc", image).stdout
    config_env = json.loads(docker("inspect", image).stdout)[0]["Config"]["Env"]
    assert not any(e.startswith(("HF_TOKEN=", "GIPFORMER_API_KEYS=", "GIPFORMER_API_KEYS_FILE=")) for e in config_env)
    assert "HF_TOKEN" not in meta and "GIPFORMER_API_KEYS" not in meta
    hf_token = os.environ.get("HF_TOKEN")
    assert not hf_token or hf_token not in meta


def gpu() -> str:
    gpu_id = os.environ.get("GIPFORMER_GPU")
    if gpu_id is None:
        pytest.skip("set GIPFORMER_GPU to a free GPU id for the runtime checks")
    assert (MODELS / "embeddinggemma-2").is_dir(), "models not staged; run scripts/stage_models.py"
    return gpu_id


def wait_ready(cid: str, timeout: float = 600) -> None:
    probe = "import urllib.request; urllib.request.urlopen('http://127.0.0.1:18000/ready', timeout=4)"
    deadline = time.monotonic() + timeout
    while docker("exec", cid, "python", "-c", probe, check=False).returncode != 0:
        state = json.loads(docker("inspect", cid).stdout)[0]["State"]
        logs = docker("logs", cid, check=False).stderr
        assert state["Running"], logs[-4000:]
        assert '"event": "engine.load_failed"' not in logs, logs[-6000:]
        assert time.monotonic() < deadline, "container did not become ready"
        time.sleep(2)


def sidecar(cid: str, image: str, *argv: str, timeout: float = 900, detach: bool = False):
    cmd = ["docker", "run", "--rm", "--network", f"container:{cid}", "-v", f"{ROOT}:/w:ro",
           "-e", f"OPENAI_API_KEY={KEY}", "--entrypoint", "python", image, *argv]
    if detach:
        return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# Streams 6 x vi_long.wav (~640 s, ~22 chunks) and prints the event types.
STREAM_LONG = """
import io, json, os, urllib.request, uuid, wave
src = wave.open('/w/fixtures-large/vi_long.wav'); frames = src.readframes(src.getnframes())
buf = io.BytesIO(); out = wave.open(buf, 'wb'); out.setparams(src.getparams()); out.writeframes(frames * 6); out.close()
b = uuid.uuid4().hex
body = (f'--{b}\\r\\nContent-Disposition: form-data; name="model"\\r\\n\\r\\nquan-transcribe-1.0\\r\\n'
        f'--{b}\\r\\nContent-Disposition: form-data; name="stream"\\r\\n\\r\\ntrue\\r\\n'
        f'--{b}\\r\\nContent-Disposition: form-data; name="file"; filename="a.wav"\\r\\n\\r\\n').encode()
body += buf.getvalue() + f'\\r\\n--{b}--\\r\\n'.encode()
req = urllib.request.Request('http://127.0.0.1:18000/v1/audio/transcriptions', data=body, method='POST', headers={
    'Authorization': 'Bearer ' + os.environ['OPENAI_API_KEY'], 'Content-Type': f'multipart/form-data; boundary={b}'})
with urllib.request.urlopen(req, timeout=600) as r:
    for line in r:
        if line.strip():
            e = json.loads(line.decode().removeprefix('data: '))
            print(e.get('type', 'error'), flush=True)
"""


@contextmanager
def running_container(image, *env, device="cuda"):
    gpu_id = gpu() if device == "cuda" else None
    # Exercise the same file-secret path as compose, without putting the key in container metadata.
    with tempfile.TemporaryDirectory(prefix="gipformer-gate-") as tmp:
        key_file = Path(tmp) / "api_keys"
        key_file.write_text(KEY + "\n")
        key_file.chmod(0o444)  # bind mount must be readable by uid 10001
        args = ["run", "-d", "--network", "none", "--read-only",
                "--tmpfs", "/tmp:size=512m,mode=1777", "--mount", f"type=bind,src={MODELS},dst=/models,readonly",
                "--mount", f"type=bind,src={key_file},dst=/run/secrets/gipformer_api_keys,readonly",
                "-e", "GIPFORMER_API_KEYS_FILE=/run/secrets/gipformer_api_keys"]
        if gpu_id is not None:
            args += ["--gpus", f"device={gpu_id}"]
        else:
            args += ["-e", "GIPFORMER_DEVICE=cpu", "-e", "OMP_NUM_THREADS=2", "-e", "MKL_NUM_THREADS=2"]
        for value in env:
            args += ["-e", value]
        cid = docker(*args, image).stdout.strip()
        try:
            wait_ready(cid)
            assert KEY not in docker("inspect", cid).stdout
            yield cid
        finally:
            docker("rm", "-f", cid, check=False)


def test_offline_ready_and_smoke(image, container_device, container_cuda):
    with running_container(image, device=container_device) as cid:
        r = sidecar(cid, image, "/w/scripts/smoke_server.py", "--base-url", "http://127.0.0.1:18000")
        assert r.returncode == 0, r.stderr[-4000:]  # stdout contains transcripts: never include it in test logs
        report = json.loads(r.stdout)
        report["image"] = json.loads(docker("inspect", image).stdout)[0]["Id"]
        report["commit"] = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                                          text=True, check=True).stdout.strip()
        report["source_dirty"] = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip())
        report["cuda_profile"] = container_cuda
        report["device"] = container_device
        report["gpu_release_gate"] = container_device == "cuda"
        # Persist gate evidence, not transcripts (the smoke verifies them in memory).
        for group in ("results", "realtime"):
            for result in report.get(group, {}).values():
                for field in ("text", "text_body", "sse_joined", "sse_done", "baseline"):
                    result.pop(field, None)
        suffix = "" if container_cuda == "cu126" else f"-{container_cuda}"
        cpu = "" if container_device == "cuda" else "-cpu"
        name = f"server-container{cpu}{suffix}.json"
        (ROOT / "docs/baseline" / name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


# Two idle realtime sessions deterministically occupy both lanes: no timing-dependent GPU saturation.
ERRORS_AND_ISOLATION = """
import concurrent.futures, json, os, sys, urllib.error
sys.path.insert(0, '/w')
from scripts.smoke_server import Gateway
from websockets.sync.client import connect
gw = Gateway('http://127.0.0.1:18000', os.environ['OPENAI_API_KEY'], 120)
def error(call, status, code):
    try:
        call()
    except urllib.error.HTTPError as e:
        assert e.code == status, e.code
        assert json.load(e)['error']['code'] == code
        assert e.headers['x-request-id'].startswith('req_')
    else:
        raise AssertionError('expected HTTP error')
error(lambda: gw.get_json('/v1/models', auth=False), 401, 'invalid_api_key')
error(lambda: gw.get_json('/v1/models/not-a-model'), 404, 'model_not_found')
audio = open('/w/tests/fixtures/vi_a.wav', 'rb').read()
error(lambda: gw.transcribe(audio, model='not-a-model'), 404, 'model_not_found')
error(lambda: gw.transcribe(b'not audio'), 400, 'invalid_audio')
headers = {'Authorization': 'Bearer ' + gw.key}
url = 'ws://127.0.0.1:18000/v1/realtime?model=quan-transcribe-1.0'
with connect(url, additional_headers=headers) as a, connect(url, additional_headers=headers) as b:
    assert json.loads(a.recv(30))['type'] == 'session.created'
    assert json.loads(b.recv(30))['type'] == 'session.created'
    error(lambda: gw.transcribe(audio), 429, 'server_busy')
# Closing a socket releases its lane asynchronously; wait for the active-lane gauge, not a fixed sleep.
import time
for _ in range(100):
    metrics = gw.request('GET', '/metrics').read().decode()
    if 'quan_open_lanes 0' in metrics:
        break
    time.sleep(.1)
else:
    raise AssertionError('lanes not released')
baseline = json.load(open('/w/docs/baseline/offline.json'))['results']
def transcribe(name):
    data = open('/w/tests/fixtures/' + name, 'rb').read()
    with gw.transcribe(data) as r:
        assert json.load(r)['text'] == baseline[name + ':full']['text']
with concurrent.futures.ThreadPoolExecutor(2) as pool:
    list(pool.map(transcribe, ['vi_a.wav', 'vi_b.wav']))
print('auth/errors/overload/isolation passed')
"""


def test_errors_overload_and_isolation(image, container_device):
    with running_container(image, "GIPFORMER_QUEUE_MAX_LANES=2", "GIPFORMER_MAX_SESSIONS=2", device=container_device) as cid:
        r = sidecar(cid, image, "-c", ERRORS_AND_ISOLATION)
        assert r.returncode == 0, r.stdout[-4000:] + r.stderr[-4000:]
        assert KEY not in docker("logs", cid).stderr


def test_stream_drain_and_restart(image, container_device):
    if not (ROOT / "fixtures-large/vi_long.wav").is_file():
        if os.environ.get("GIPFORMER_GPU") or container_device == "cpu":
            pytest.fail("drain/restart gate needs fixtures-large/vi_long.wav; run scripts/fetch_fixtures.py")
        pytest.skip("drain/restart gate needs fixtures-large/vi_long.wav; run scripts/fetch_fixtures.py")
    with running_container(image, device=container_device) as cid:
        # Keep the GPU release workload unchanged; CPU diagnostics must fit the same 30s Uvicorn drain budget.
        script = STREAM_LONG if container_device == "cuda" else STREAM_LONG.replace("frames * 6", "frames * 2")
        stream = sidecar(cid, image, "-c", script, detach=True)
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(stream.stdout, selectors.EVENT_READ)
                assert selector.select(timeout=120), "stream produced no first event within 120s"
                assert stream.stdout.readline().strip() == "transcript.text.delta"  # stream is running
            started = time.monotonic()
            docker("stop", "-t", "70", cid, timeout=120)
            out, err = stream.communicate(timeout=120)
            assert stream.returncode == 0, err[-4000:]
            assert out.strip().splitlines()[-1] == "transcript.text.done"  # finished while draining
            assert time.monotonic() - started < 70
            state = json.loads(docker("inspect", cid).stdout)[0]["State"]
            assert state["ExitCode"] == 0
            logs = docker("logs", cid).stderr
            stopped = [json.loads(line) for line in logs.splitlines() if '"server.stopped"' in line]
            assert stopped and stopped[-1]["drained"] is True
            assert KEY not in logs

            docker("start", cid)
            wait_ready(cid)
        finally:
            if stream.poll() is None:
                stream.kill()
            stream.communicate(timeout=120)
