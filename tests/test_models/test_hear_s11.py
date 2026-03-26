from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader

import mteb
from mteb.models.model_implementations import hear_s11_models


class _DummyHFModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(pooler_output_size=384)

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def requires_grad_(self, _: bool):
        return self

    def forward(self, input_values, return_dict=True):
        del return_dict
        pooled = input_values.float().mean(dim=1, keepdim=True).repeat(1, 384)
        return SimpleNamespace(pooler_output=pooled)


class _RecordingHFModel(_DummyHFModel):
    def __init__(self):
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, input_values, return_dict=True):
        self.batch_sizes.append(int(input_values.shape[0]))
        return super().forward(input_values, return_dict=return_dict)


def _collate_audio(batch):
    return {"audio": [item["audio"] for item in batch]}


def test_hear_s11_meta_registered():
    meta = mteb.get_model_meta("matthewagi/HeAR-s1.1")

    assert meta.name == "matthewagi/HeAR-s1.1"
    assert meta.revision == "a5776bebff935a81c79720467ae1e10a4effe10e"
    assert meta.modalities == ["audio"]
    assert meta.embed_dim == 384
    assert meta.n_parameters == 22_140_288
    assert meta.license == "https://developers.google.com/health-ai-developer-foundations/terms"
    assert "Transformers" in meta.framework
    assert "safetensors" in meta.framework
    assert meta.loader is hear_s11_models.HeARS11AudioWrapper


def test_hear_s11_model_load_and_encode(monkeypatch):
    monkeypatch.setattr(hear_s11_models, "requires_audio_dependencies", lambda: None)
    monkeypatch.setattr(hear_s11_models, "requires_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        hear_s11_models,
        "AutoModel",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: _DummyHFModel()),
    )

    model = mteb.get_model("matthewagi/HeAR-s1.1", device="cpu")
    samples = [
        {
            "audio": {
                "array": np.ones(16000, dtype=np.float32),
                "sampling_rate": 16000,
            }
        },
        {
            "audio": {
                "array": np.zeros(16000, dtype=np.float32),
                "sampling_rate": 16000,
            }
        },
    ]
    loader = DataLoader(samples, batch_size=2, shuffle=False, collate_fn=_collate_audio)
    embeddings = model.encode(
        loader,
        task_metadata=SimpleNamespace(name="DummyAudioTask"),
        hf_split="test",
        hf_subset="default",
        batch_size=2,
        show_progress_bar=False,
    )

    assert embeddings.shape == (2, 384)
    assert embeddings.dtype == np.float32
    assert np.isfinite(embeddings).all()


def test_hear_s11_sliding_window_batches_across_items(monkeypatch):
    recording_model = _RecordingHFModel()

    monkeypatch.setattr(hear_s11_models, "requires_audio_dependencies", lambda: None)
    monkeypatch.setattr(hear_s11_models, "requires_package", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        hear_s11_models,
        "AutoModel",
        SimpleNamespace(from_pretrained=lambda *args, **kwargs: recording_model),
    )
    monkeypatch.setattr(
        hear_s11_models,
        "_prep_audio_sliding_windows",
        lambda *args, **kwargs: torch.arange(12, dtype=torch.float32).reshape(3, 4),
    )

    model = mteb.get_model("matthewagi/HeAR-s1.1", device="cpu")
    samples = [
        {
            "audio": {
                "array": np.ones(16000, dtype=np.float32),
                "sampling_rate": 16000,
            }
        },
        {
            "audio": {
                "array": np.zeros(16000, dtype=np.float32),
                "sampling_rate": 16000,
            }
        },
    ]
    loader = DataLoader(samples, batch_size=2, shuffle=False, collate_fn=_collate_audio)
    embeddings = model.encode(
        loader,
        task_metadata=SimpleNamespace(name="DummyAudioTask"),
        hf_split="test",
        hf_subset="default",
        batch_size=4,
        show_progress_bar=False,
    )

    assert embeddings.shape == (2, 384)
    assert recording_model.batch_sizes == [4, 2]
