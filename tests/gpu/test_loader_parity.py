"""app/quan_loader.load_quan must produce exactly the model of the reference QuanTranscribe.from_pretrained.

Needs a CUDA GPU, staged assets and Hub access: the reference loader resolves google/embeddinggemma-2 at `main`,
so the test also checks that `main` is still the pinned revision (otherwise the comparison is not meaningful).
"""

import json
import os
from pathlib import Path

import pytest
import soundfile as sf
import torch

ROOT = Path(__file__).resolve().parents[2]
MODELS = Path(os.environ.get("GIPFORMER_MODELS_DIR", ROOT / "models"))
pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(not (MODELS / "embeddinggemma-2").is_dir(), reason="assets not staged"),
]


def test_local_loader_matches_reference_loader():
    from huggingface_hub import HfApi
    from quan_transcribe import QuanTranscribe

    from app.quan_loader import load_quan

    manifest = json.loads((ROOT / "assets/manifest.json").read_text())
    base = manifest["models"]["embeddinggemma-2"]
    try:
        main_sha = HfApi().model_info(base["repo_id"], revision="main").sha
    except Exception as e:  # no network: cannot run the reference loader
        pytest.skip(f"Hub not reachable: {type(e).__name__}")
    assert main_sha == base["revision"], "Hub main moved; reference loader would load a different base model"

    local = load_quan(MODELS, "cuda")
    ref = QuanTranscribe.from_pretrained(str(MODELS / "quan-transcribe-1.0"), device="cuda", lm=False)

    sd_local, sd_ref = local.model.state_dict(), ref.model.state_dict()
    assert sd_local.keys() == sd_ref.keys()
    for k in sd_local:
        assert torch.equal(sd_local[k], sd_ref[k]), k
    assert local.gemma_ids == ref.gemma_ids
    assert local.tokenizer.convert_ids_to_tokens(local.gemma_ids) == ref.tokenizer.convert_ids_to_tokens(ref.gemma_ids)

    for name in ("vi_a.wav", "vi_b.wav"):
        wav, _ = sf.read(ROOT / "tests/fixtures" / name, dtype="float32")
        for streaming in (False, True):
            la, na = local._forward([wav], streaming, None)
            lb, nb = ref._forward([wav], streaming, None)
            assert torch.equal(na, nb) and torch.equal(la, lb), (name, streaming)
            assert local.transcribe(wav, streaming=streaming) == ref.transcribe(wav, streaming=streaming)
