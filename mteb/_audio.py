from __future__ import annotations

import io
from typing import Any

import numpy as np
import torch


def decode_audio(
    audio: Any,
    ex: dict[str, Any] | None = None,
    *,
    default_sr: int | None = None,
) -> tuple[Any | None, int | None, str | None]:
    if audio is None:
        return None, None, "audio_none"

    decode_err: str | None = None

    if not isinstance(audio, dict) and hasattr(audio, "as_py"):
        try:
            audio = audio.as_py()
        except Exception as exc:
            decode_err = f"audio_as_py:{type(exc).__name__}"

    if not isinstance(audio, dict) and hasattr(audio, "decode"):
        try:
            decoded = audio.decode()
            if isinstance(decoded, dict):
                audio = decoded
            elif isinstance(decoded, (tuple, list)) and len(decoded) == 2:
                arr, sr = decoded
                if sr is None:
                    sr = default_sr
                if sr is not None:
                    return arr, int(sr), None
            elif isinstance(decoded, (np.ndarray, torch.Tensor)):
                sr = getattr(audio, "sampling_rate", None) or getattr(audio, "sample_rate", None)
                if ex is not None and sr is None:
                    sr = ex.get("sampling_rate") or ex.get("sample_rate") or ex.get("sr") or ex.get("audio_sampling_rate")
                if sr is None:
                    sr = default_sr
                if sr is not None:
                    return decoded, int(sr), None
        except Exception as exc:
            decode_err = f"audio_decode_method:{type(exc).__name__}"

    if not isinstance(audio, dict) and hasattr(audio, "get_all_samples"):
        try:
            samples = audio.get_all_samples()
            data = getattr(samples, "data", None)
            sr = (
                getattr(samples, "sample_rate", None)
                or getattr(audio, "sampling_rate", None)
                or getattr(audio, "sample_rate", None)
            )
            if data is not None and sr is not None:
                return data, int(sr), None
        except Exception as exc:
            decode_err = f"audio_get_all_samples:{type(exc).__name__}"

    if not isinstance(audio, dict):
        arr = getattr(audio, "array", None)
        sr = getattr(audio, "sampling_rate", None) or getattr(audio, "sample_rate", None)
        if arr is not None and sr is not None:
            return arr, int(sr), None
        path_attr = getattr(audio, "path", None)
        bytes_attr = getattr(audio, "bytes", None)
        if any(v is not None for v in (arr, sr, path_attr, bytes_attr)):
            audio = {
                "array": arr,
                "sampling_rate": sr,
                "bytes": bytes_attr,
                "path": path_attr,
            }

    if not isinstance(audio, dict) and hasattr(audio, "keys") and hasattr(audio, "__getitem__"):
        try:
            audio = {k: audio[k] for k in audio.keys()}
        except Exception as exc:
            decode_err = f"audio_mapping:{type(exc).__name__}"

    if not isinstance(audio, dict) and hasattr(audio, "__dict__"):
        try:
            attrs = {
                "array": getattr(audio, "array", None),
                "sampling_rate": getattr(audio, "sampling_rate", None) or getattr(audio, "sample_rate", None),
                "bytes": getattr(audio, "bytes", None),
                "path": getattr(audio, "path", None),
            }
            if any(v is not None for v in attrs.values()):
                audio = attrs
        except Exception as exc:
            decode_err = f"audio_attrs:{type(exc).__name__}"

    if isinstance(audio, dict):
        arr = audio.get("array")
        sr = audio.get("sampling_rate")
        if arr is not None and sr is not None:
            return arr, int(sr), None
        if arr is not None and sr is None and default_sr is not None:
            return arr, int(default_sr), None

        data_bytes = audio.get("bytes")
        if data_bytes is not None:
            err = None
            try:
                import soundfile as sf
            except Exception:
                sf = None
            if sf is not None:
                try:
                    with io.BytesIO(data_bytes) as bio:
                        data, sr = sf.read(bio, dtype="float32", always_2d=False)
                    return data, int(sr), None
                except Exception:
                    err = "soundfile_bytes"
            try:
                import torchaudio

                with io.BytesIO(data_bytes) as bio:
                    data, sr = torchaudio.load(bio)
                if data.ndim == 2:
                    data = data.mean(dim=0)
                return data.numpy(), int(sr), None
            except Exception:
                return None, None, (err + "+torchaudio_bytes" if err else "torchaudio_bytes")

        path = audio.get("path")
        if path:
            err = None
            try:
                import soundfile as sf

                data, sr = sf.read(path, dtype="float32", always_2d=False)
                return data, int(sr), None
            except Exception:
                err = "soundfile_path"
            try:
                import torchaudio

                data, sr = torchaudio.load(path)
                if data.ndim == 2:
                    data = data.mean(dim=0)
                return data.numpy(), int(sr), None
            except Exception:
                return None, None, (err + "+torchaudio_path" if err else "torchaudio_path")

        if "array" in audio or "sampling_rate" in audio:
            return None, None, "audio_array_missing"
        if data_bytes is None and path is None:
            return None, None, "audio_missing_bytes_path"
        return None, None, "audio_decode_failed"

    if isinstance(audio, (tuple, list)) and len(audio) == 2:
        arr, sr = audio
        if sr is not None:
            return arr, int(sr), None

    if isinstance(audio, memoryview):
        audio = audio.tobytes()
    if isinstance(audio, (bytes, bytearray)):
        try:
            import soundfile as sf

            with io.BytesIO(audio) as bio:
                data, sr = sf.read(bio, dtype="float32", always_2d=False)
            return data, int(sr), None
        except Exception:
            return None, None, "soundfile_bytes"

    if isinstance(audio, (np.ndarray, torch.Tensor)):
        sr = None
        if ex is not None:
            sr = ex.get("sampling_rate") or ex.get("sample_rate") or ex.get("sr") or ex.get("audio_sampling_rate")
        if sr is None:
            sr = default_sr
        if sr is not None:
            return audio, int(sr), None

    if ex is not None:
        maybe_audio = ex.get("audio")
        if maybe_audio is not audio:
            return decode_audio(maybe_audio, None, default_sr=default_sr)

    return None, None, decode_err or f"unsupported_audio_type:{type(audio).__name__}"


def is_valid_audio_example(
    example: dict[str, Any],
    *,
    audio_key: str = "audio",
    min_samples: int = 500,
) -> bool:
    audio_item = example.get(audio_key)
    audio_arr, _sr, _err = decode_audio(audio_item, ex=example)
    if audio_arr is None:
        return False
    if isinstance(audio_arr, torch.Tensor):
        flat = audio_arr.detach().float().reshape(-1)
        if flat.numel() < min_samples:
            return False
        return bool(torch.isfinite(flat).all().item())

    flat_np = np.asarray(audio_arr, dtype=np.float32).reshape(-1)
    if flat_np.size < min_samples:
        return False
    return bool(np.isfinite(flat_np).all())
