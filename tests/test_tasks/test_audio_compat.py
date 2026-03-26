from __future__ import annotations

import torch

from mteb._audio import decode_audio, is_valid_audio_example
from mteb.tasks.classification.eng import vox_populi_accent_id
from mteb.tasks.classification.multilingual import vox_populi_language_id


class _FakeSamples:
    def __init__(self, data, sample_rate):
        self.data = data
        self.sample_rate = sample_rate


class _FakeAudioDecoder:
    def __init__(self, data, sample_rate):
        self._data = data
        self.sample_rate = sample_rate
        self.sampling_rate = sample_rate

    def get_all_samples(self):
        return _FakeSamples(self._data, self.sample_rate)


class _FakeDataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def filter(self, fn):
        return _FakeDataset([row for row in self.rows if fn(row)])

    def __len__(self):
        return len(self.rows)


def test_decode_audio_supports_get_all_samples():
    decoder = _FakeAudioDecoder(torch.ones(600), 16000)
    audio, sr, err = decode_audio(decoder)

    assert err is None
    assert sr == 16000
    assert audio is not None
    assert int(torch.as_tensor(audio).numel()) == 600


def test_is_valid_audio_example_supports_audio_decoder():
    valid = {"audio": _FakeAudioDecoder(torch.ones(600), 16000)}
    too_short = {"audio": _FakeAudioDecoder(torch.ones(100), 16000)}
    non_finite = {"audio": _FakeAudioDecoder(torch.tensor([float("nan")] * 600), 16000)}

    assert is_valid_audio_example(valid)
    assert not is_valid_audio_example(too_short)
    assert not is_valid_audio_example(non_finite)


def test_vox_populi_language_id_filters_audio_decoder(monkeypatch):
    task = vox_populi_language_id.VoxPopuliLanguageID()
    task.dataset = {
        "train": _FakeDataset(
            [
                {"audio": _FakeAudioDecoder(torch.ones(600), 16000)},
                {"audio": _FakeAudioDecoder(torch.ones(100), 16000)},
            ]
        )
    }
    monkeypatch.setattr(vox_populi_language_id, "DatasetDict", lambda data: data)

    task.dataset_transform()

    assert len(task.dataset["train"]) == 1


def test_vox_populi_accent_id_filters_audio_decoder(monkeypatch):
    task = vox_populi_accent_id.VoxPopuliAccentID()
    task.dataset = {
        "test": _FakeDataset(
            [
                {"audio": _FakeAudioDecoder(torch.ones(600), 16000)},
                {"audio": _FakeAudioDecoder(torch.ones(100), 16000)},
            ]
        )
    }
    monkeypatch.setattr(vox_populi_accent_id, "DatasetDict", lambda data: data)

    task.dataset_transform()

    assert len(task.dataset["train"]) == 1
    assert len(task.dataset["test"]) == 1
