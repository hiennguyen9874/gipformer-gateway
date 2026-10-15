"""CPU lane: pins are exact commits and the committed fixtures match their manifest."""

import json
import re
import tomllib
from pathlib import Path

from scripts.fetch_fixtures import MANIFEST as FIXTURES_MANIFEST
from scripts.fetch_fixtures import verify

ROOT = Path(__file__).resolve().parents[2]
COMMIT = re.compile(r"^[0-9a-f]{40}$")


def test_asset_manifest_pins_commits_and_hashes():
    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    assert list(manifest["models"]) == ["gipformer1.5-68M-rnnt"]
    for entry in manifest["models"].values():
        assert COMMIT.match(entry["revision"])
        for meta in entry["files"].values():
            assert re.fullmatch(r"[0-9a-f]{64}", meta["sha256"]) and meta["size"] > 0


def test_manifest_lists_only_the_files_the_gateway_needs():
    entry = json.loads((ROOT / "assets/manifest.json").read_text())["models"]["gipformer1.5-68M-rnnt"]
    onnx = {f"{part}{suffix}.onnx" for part in ("encoder", "decoder", "joiner") for suffix in ("", ".int8")}
    assert set(entry["files"]) == onnx | {"tokens.txt", "config.json", "README.md"}


def test_pyproject_and_lockfile_pin_sherpa_onnx_to_manifest_version():
    deps = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["dependencies"]
    version = json.loads((ROOT / "assets/manifest.json").read_text())["code"]["sherpa-onnx"]["version"]
    locked = {p["name"]: p["version"] for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
    for name in ("sherpa-onnx", "sherpa-onnx-core"):
        assert f"{name}=={version}" in deps
        assert locked[name] == version


def test_cuda_profiles_are_exclusive_and_locked():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())
    expected = {"cu126": "2.14.1", "cu128": "2.11.0"}
    assert project["tool"]["uv"]["conflicts"] == [[{"extra": "cu126"}, {"extra": "cu128"}]]
    indexes = {i["name"]: i for i in project["tool"]["uv"]["index"]}
    sources = {s["extra"]: s["index"] for s in project["tool"]["uv"]["sources"]["torch"]}
    locked = tomllib.loads((ROOT / "uv.lock").read_text())["package"]
    for cuda, version in expected.items():
        assert project["project"]["optional-dependencies"][cuda] == [f"torch=={version}"]
        index = indexes[sources[cuda]]
        assert index["explicit"] is True
        assert index["url"] == f"https://download.pytorch.org/whl/{cuda}"
        wheels = [p for p in locked if p["name"] == "torch" and p["version"] == f"{version}+{cuda}"]
        assert len(wheels) == 1
        assert wheels[0]["source"]["registry"] == index["url"]
        assert any("cp312-cp312" in w["url"] and "x86_64" in w["url"] for w in wheels[0]["wheels"])


def test_docker_cuda_profile_selection():
    dockerfile = (ROOT / "Dockerfile").read_text()
    assert "ARG TORCH_CUDA=cu126" in dockerfile
    assert 'case "$TORCH_CUDA" in cu126|cu128)' in dockerfile
    assert 'uv sync --frozen --no-dev --extra "$TORCH_CUDA"' in dockerfile
    compose = (ROOT / "compose.yaml").read_text()
    assert 'TORCH_CUDA: "${GIPFORMER_BUILD_CUDA:-cu126}"' in compose


def test_small_fixtures_match_manifest():
    assert verify(json.loads(FIXTURES_MANIFEST.read_text()), include_large=False) == []
