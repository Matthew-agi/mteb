from __future__ import annotations

from contextlib import nullcontext
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm
from transformers import AutoModel

from mteb._audio import decode_audio
from mteb._requires_package import requires_audio_dependencies
from mteb._requires_package import requires_package
from mteb.models import ModelMeta
from mteb.models.abs_encoder import AbsEncoder
from mteb.models.model_meta import ScoringFunction

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

    from mteb import TaskMetadata
    from mteb.types import Array, BatchedInput, PromptType


DEFAULT_MODEL_ID = "matthewagi/HeAR-s1.1"
DEFAULT_REVISION = "a5776bebff935a81c79720467ae1e10a4effe10e"
DEFAULT_REFERENCE_URL = "https://huggingface.co/matthewagi/HeAR-s1.1"
DEFAULT_TARGET_SR = 16000
DEFAULT_CLIP_SECONDS = 2.0
DEFAULT_WINDOW_HOP_SECONDS = 2.0
DEFAULT_CROP = "center"
DEFAULT_WINDOW_POOL = "mean"
DEFAULT_EMBED_DIM = 384
DEFAULT_N_PARAMETERS = 22_140_288
DEFAULT_LICENSE_URL = (
    "https://developers.google.com/health-ai-developer-foundations/terms"
)


def _die(msg: str) -> "None":
    raise RuntimeError(msg)


def _pick_device(requested: str) -> tuple[str, torch.device]:
    if requested == "cpu":
        return "cpu", torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            _die("Requested CUDA, but torch.cuda.is_available() is False.")
        return "cuda", torch.device("cuda")
    if requested == "mps":
        if not getattr(torch.backends, "mps", None) or not torch.backends.mps.is_available():
            _die("Requested MPS, but torch.backends.mps.is_available() is False.")
        return "mps", torch.device("mps")
    if torch.cuda.is_available():
        return "cuda", torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps", torch.device("mps")
    return "cpu", torch.device("cpu")


def _configure_runtime(
    *,
    device_type: str,
    enable_tf32: bool,
    cudnn_benchmark: bool,
    matmul_precision: str,
) -> dict[str, Any]:
    config = {
        "tf32": False,
        "cudnn_benchmark": False,
        "matmul_precision": None,
    }
    if hasattr(torch, "set_float32_matmul_precision"):
        try:
            torch.set_float32_matmul_precision(matmul_precision)
            config["matmul_precision"] = matmul_precision
        except Exception:
            pass
    if device_type != "cuda":
        return config

    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        try:
            torch.backends.cuda.matmul.allow_tf32 = bool(enable_tf32)
            config["tf32"] = bool(enable_tf32)
        except Exception:
            pass
    if hasattr(torch.backends, "cudnn"):
        try:
            torch.backends.cudnn.allow_tf32 = bool(enable_tf32)
            config["tf32"] = bool(enable_tf32)
        except Exception:
            pass
        try:
            torch.backends.cudnn.benchmark = bool(cudnn_benchmark)
            config["cudnn_benchmark"] = bool(cudnn_benchmark)
        except Exception:
            pass
    return config


def _resolve_amp_dtype(
    requested: str,
    *,
    enabled: bool,
    device_type: str,
) -> tuple[torch.dtype | None, str]:
    if (not enabled) or device_type != "cuda":
        return None, "disabled"
    if requested == "auto":
        if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
            return torch.bfloat16, "bfloat16"
        return torch.float16, "float16"
    if requested == "bfloat16":
        return torch.bfloat16, "bfloat16"
    if requested == "float16":
        return torch.float16, "float16"
    _die(f"Unsupported amp_dtype: {requested}")


def _resample(audio: torch.Tensor, src_sr: int, dst_sr: int) -> torch.Tensor:
    if src_sr == dst_sr:
        return audio
    try:
        from scipy import signal
    except Exception as exc:
        _die(f"Resampling requires scipy: {exc}")
    new_len = int(round(audio.shape[0] * (dst_sr / src_sr)))
    resampled = signal.resample(audio.cpu().numpy(), new_len)
    return torch.from_numpy(resampled).float()


def _to_mono(audio: torch.Tensor) -> torch.Tensor:
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[1] > audio.shape[0]:
            return audio.mean(dim=0)
        return audio.mean(dim=1)
    return audio


def _crop_center(audio: torch.Tensor, clip_samples: int) -> torch.Tensor:
    if audio.numel() <= clip_samples:
        if audio.numel() < clip_samples:
            return F.pad(audio, (0, clip_samples - audio.numel()))
        return audio
    start = (audio.numel() - clip_samples) // 2
    return audio[start : start + clip_samples]


def _crop_peak(audio: torch.Tensor, clip_samples: int, sr: int) -> torch.Tensor:
    if audio.numel() <= clip_samples:
        if audio.numel() < clip_samples:
            return F.pad(audio, (0, clip_samples - audio.numel()))
        return audio
    win = max(1, int(sr * 0.05))
    hop = max(1, int(sr * 0.01))
    if audio.numel() < win:
        start = 0
    else:
        energy = F.avg_pool1d((audio * audio).view(1, 1, -1), kernel_size=win, stride=hop).view(-1)
        idx = int(torch.argmax(energy)) if energy.numel() else 0
        center = idx * hop + win // 2
        start = max(0, min(int(center - clip_samples // 2), audio.numel() - clip_samples))
    return audio[start : start + clip_samples]


def _prep_audio_clips(
    audio_arr: Any,
    sr: int,
    *,
    target_sr: int,
    clip_samples: int,
    crop: str,
    full_clip: bool,
) -> torch.Tensor:
    audio = torch.as_tensor(audio_arr).float()
    audio = _to_mono(audio)
    if full_clip:
        audio = _resample(audio, sr, target_sr)
        if audio.numel() <= 0:
            audio = F.pad(audio, (0, 1))
        return audio.unsqueeze(0)

    clip_src = int(round(float(clip_samples) * (float(sr) / float(target_sr))))
    clip_src = max(1, clip_src)
    if crop == "peak":
        segment = _crop_peak(audio, clip_src, sr)
    else:
        segment = _crop_center(audio, clip_src)
    segment = _resample(segment, sr, target_sr)
    if segment.numel() < clip_samples:
        segment = F.pad(segment, (0, clip_samples - segment.numel()))
    elif segment.numel() > clip_samples:
        segment = segment[:clip_samples]
    return segment.unsqueeze(0)


def _prep_audio_sliding_windows(
    audio_arr: Any,
    sr: int,
    *,
    target_sr: int,
    clip_samples: int,
    hop_samples: int,
) -> torch.Tensor:
    if clip_samples <= 0:
        raise ValueError("clip_samples must be > 0")
    if hop_samples <= 0:
        raise ValueError("hop_samples must be > 0")

    audio = torch.as_tensor(audio_arr).float()
    audio = _to_mono(audio)
    audio = _resample(audio, sr, target_sr)
    if audio.numel() <= 0:
        audio = F.pad(audio, (0, 1))
    if audio.numel() <= clip_samples:
        if audio.numel() < clip_samples:
            audio = F.pad(audio, (0, clip_samples - audio.numel()))
        return audio.unsqueeze(0)

    last_start = max(0, int(audio.numel()) - int(clip_samples))
    starts = list(range(0, last_start + 1, int(hop_samples)))
    if not starts:
        starts = [0]
    if starts[-1] != last_start:
        starts.append(last_start)
    return torch.stack([audio[start : start + clip_samples] for start in starts], dim=0)


class HeARS11AudioWrapper(AbsEncoder):
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        target_sr: int = DEFAULT_TARGET_SR,
        clip_seconds: float = DEFAULT_CLIP_SECONDS,
        crop: str = DEFAULT_CROP,
        full_clip: bool = False,
        sliding_window: bool = True,
        window_hop_seconds: float = DEFAULT_WINDOW_HOP_SECONDS,
        window_pool: str = DEFAULT_WINDOW_POOL,
        compile_enabled: bool = False,
        compile_mode: str = "default",
        compile_dynamic: bool = True,
        amp_enabled: bool = True,
        amp_dtype: str = "auto",
        enable_tf32: bool = True,
        cudnn_benchmark: bool = True,
        **_: Any,
    ):
        requires_audio_dependencies()
        requires_package(self, "timm", model_name, "pip install 'mteb[audio,timm]'")
        self.model_name = model_name
        self.revision = revision
        self.target_sr = int(target_sr)
        self.clip_seconds = float(clip_seconds)
        self.crop = str(crop)
        self.full_clip = bool(full_clip)
        self.sliding_window = bool(sliding_window)
        self.window_hop_seconds = float(window_hop_seconds)
        self.window_pool = str(window_pool)
        self.total_embeddings = 0

        if self.clip_seconds <= 0:
            _die("clip_seconds must be > 0.")
        if self.full_clip and self.sliding_window:
            _die("full_clip and sliding_window are mutually exclusive.")
        self.clip_samples = int(round(self.clip_seconds * self.target_sr))
        if self.clip_samples <= 0:
            _die("clip_seconds * target_sr must be at least 1 sample.")
        if self.sliding_window and self.window_hop_seconds <= 0:
            _die("window_hop_seconds must be > 0 when sliding_window is enabled.")
        self.window_hop_samples = int(round(self.window_hop_seconds * self.target_sr))
        if self.sliding_window and self.window_hop_samples <= 0:
            _die("window_hop_seconds * target_sr must be at least 1 sample.")
        if self.window_pool != "mean":
            _die(f"Unsupported window pool: {self.window_pool}")

        self.device_type, self.device = _pick_device(device)
        runtime_flags = _configure_runtime(
            device_type=self.device_type,
            enable_tf32=bool(enable_tf32),
            cudnn_benchmark=bool(cudnn_benchmark),
            matmul_precision="high",
        )
        self.amp_dtype, amp_label = _resolve_amp_dtype(
            amp_dtype,
            enabled=bool(amp_enabled),
            device_type=self.device_type,
        )
        self.runtime_config = {
            "tf32": bool(runtime_flags["tf32"]),
            "cudnn_benchmark": bool(runtime_flags["cudnn_benchmark"]),
            "matmul_precision": runtime_flags["matmul_precision"],
            "amp": self.amp_dtype is not None,
            "amp_dtype": amp_label,
        }

        load_kwargs: dict[str, Any] = {"trust_remote_code": True}
        if revision is not None:
            load_kwargs["revision"] = revision
        self.model = AutoModel.from_pretrained(model_name, **load_kwargs).eval().to(self.device)
        self.model.requires_grad_(False)
        if compile_enabled and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode=compile_mode, dynamic=bool(compile_dynamic))
            except Exception:
                pass

        embedding_dim = (
            getattr(getattr(self.model, "config", None), "pooler_output_size", None)
            or getattr(getattr(self.model, "config", None), "pooled_dim", None)
            or getattr(getattr(self.model, "config", None), "hidden_size", None)
        )
        if embedding_dim is None:
            _die("Could not determine embedding dimension from HeAR-s1.1 config.")
        self.embedding_dim = int(embedding_dim)

    def _extract_audio(self, audio_item: Any) -> tuple[Any, int]:
        arr, sr, err = decode_audio(audio_item, default_sr=self.target_sr)
        if arr is not None and sr is not None:
            return arr, int(sr)
        raise ValueError(
            f"Could not decode audio item of type {type(audio_item).__name__}"
            + (f" ({err})" if err else "")
        )

    def _embed_batch(self, batch: torch.Tensor) -> torch.Tensor:
        batch = batch.to(self.device)
        amp_ctx = (
            torch.autocast(device_type=self.device_type, dtype=self.amp_dtype)
            if self.amp_dtype is not None
            else nullcontext()
        )
        with amp_ctx, torch.inference_mode():
            output = self.model(input_values=batch, return_dict=True)
            embeddings = output.pooler_output
        return embeddings

    @staticmethod
    def _progress_total(inputs: DataLoader[BatchedInput]) -> int | None:
        dataset = getattr(inputs, "dataset", None)
        if dataset is None:
            return None
        try:
            return int(len(dataset))
        except Exception:
            return None

    @staticmethod
    def _progress_desc(task_metadata: TaskMetadata, hf_split: str, hf_subset: str) -> str:
        task_name = getattr(task_metadata, "name", None) or getattr(task_metadata, "task_name", None)
        parts = [str(task_name or "encode"), str(hf_split)]
        if hf_subset and hf_subset != "default":
            parts.append(str(hf_subset))
        return " | ".join(parts)

    def _make_progress_bar(
        self,
        *,
        inputs: DataLoader[BatchedInput],
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        enabled: bool,
    ) -> Any | None:
        if not enabled:
            return None
        bar = tqdm(
            total=self._progress_total(inputs),
            desc=self._progress_desc(task_metadata, hf_split, hf_subset),
            unit="embed",
            dynamic_ncols=True,
            leave=True,
            smoothing=0.1,
        )
        if self.total_embeddings:
            bar.set_postfix_str(f"cum={self.total_embeddings}", refresh=False)
        return bar

    def _update_progress(self, progress_bar: Any | None, amount: int) -> None:
        amount = int(amount)
        if amount <= 0:
            return
        self.total_embeddings += amount
        if progress_bar is None:
            return
        progress_bar.update(amount)
        progress_bar.set_postfix_str(f"cum={self.total_embeddings}", refresh=False)

    @staticmethod
    def _close_progress_bar(progress_bar: Any | None) -> None:
        if progress_bar is not None:
            progress_bar.close()

    def encode(
        self,
        inputs: DataLoader[BatchedInput],
        *,
        task_metadata: TaskMetadata,
        hf_split: str,
        hf_subset: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> Array:
        del prompt_type
        max_batch_size = int(kwargs.get("batch_size", 32))
        if max_batch_size <= 0:
            max_batch_size = 32
        show_progress = bool(kwargs.get("show_progress_bar", True))
        progress_bar = self._make_progress_bar(
            inputs=inputs,
            task_metadata=task_metadata,
            hf_split=hf_split,
            hf_subset=hf_subset,
            enabled=show_progress,
        )

        try:
            outputs: list[torch.Tensor] = []
            for batch in inputs:
                audio_items = batch.get("audio")
                if audio_items is None:
                    raise ValueError("Expected an audio batch, but the dataloader batch has no `audio` key.")

                if self.sliding_window:
                    for audio_item in audio_items:
                        arr, sr = self._extract_audio(audio_item)
                        clips = _prep_audio_sliding_windows(
                            arr,
                            sr,
                            target_sr=self.target_sr,
                            clip_samples=self.clip_samples,
                            hop_samples=self.window_hop_samples,
                        )
                        emb_sum: torch.Tensor | None = None
                        emb_count = 0
                        for start in range(0, int(clips.shape[0]), max_batch_size):
                            clip_batch = clips[start : start + max_batch_size]
                            emb = self._embed_batch(clip_batch)
                            batch_sum = emb.sum(dim=0)
                            emb_sum = batch_sum.clone() if emb_sum is None else emb_sum + batch_sum
                            emb_count += int(emb.shape[0])
                        if emb_sum is None or emb_count != int(clips.shape[0]):
                            raise RuntimeError(
                                "Failed to finalize sliding-window embedding: "
                                f"needed={int(clips.shape[0])} seen={emb_count}"
                            )
                        outputs.append((emb_sum / float(emb_count)).to(torch.float32))
                        self._update_progress(progress_bar, 1)
                    continue

                per_clip_audio: list[torch.Tensor] = []
                per_clip_owner: list[int] = []
                needed_per_owner: list[int] = []
                for owner, audio_item in enumerate(audio_items):
                    arr, sr = self._extract_audio(audio_item)
                    clips = _prep_audio_clips(
                        arr,
                        sr,
                        target_sr=self.target_sr,
                        clip_samples=self.clip_samples,
                        crop=self.crop,
                        full_clip=self.full_clip,
                    )
                    needed_per_owner.append(int(clips.shape[0]))
                    for clip in clips:
                        per_clip_audio.append(clip)
                        per_clip_owner.append(owner)

                sums: list[torch.Tensor | None] = [None] * len(audio_items)
                counts = [0] * len(audio_items)
                for start in range(0, len(per_clip_audio), max_batch_size):
                    stop = start + max_batch_size
                    clip_batch = torch.stack(per_clip_audio[start:stop], dim=0)
                    owners = per_clip_owner[start:stop]
                    emb = self._embed_batch(clip_batch)
                    for row, owner in zip(emb, owners):
                        sums[owner] = row.clone() if sums[owner] is None else sums[owner] + row
                        counts[owner] += 1

                for owner, needed in enumerate(needed_per_owner):
                    if needed <= 0 or counts[owner] != needed or sums[owner] is None:
                        raise RuntimeError(
                            f"Failed to finalize embedding for batch item {owner}: needed={needed} seen={counts[owner]}"
                        )
                    outputs.append((sums[owner] / float(needed)).to(torch.float32))
                self._update_progress(progress_bar, len(audio_items))

            if not outputs:
                return np.empty((0, self.embedding_dim), dtype=np.float32)
            return torch.stack(outputs, dim=0).cpu().numpy().astype(np.float32)
        finally:
            self._close_progress_bar(progress_bar)


hear_s11_audio = ModelMeta(
    loader=HeARS11AudioWrapper,
    loader_kwargs={
        "target_sr": DEFAULT_TARGET_SR,
        "clip_seconds": DEFAULT_CLIP_SECONDS,
        "crop": DEFAULT_CROP,
        "full_clip": False,
        "sliding_window": True,
        "window_hop_seconds": DEFAULT_WINDOW_HOP_SECONDS,
        "window_pool": DEFAULT_WINDOW_POOL,
        "compile_enabled": False,
        "compile_mode": "default",
        "compile_dynamic": True,
        "amp_enabled": True,
        "amp_dtype": "auto",
        "enable_tf32": True,
        "cudnn_benchmark": True,
    },
    name=DEFAULT_MODEL_ID,
    revision=DEFAULT_REVISION,
    release_date="2026-03-26",
    languages=None,
    n_parameters=DEFAULT_N_PARAMETERS,
    n_embedding_parameters=None,
    memory_usage_mb=None,
    max_tokens=None,
    embed_dim=DEFAULT_EMBED_DIM,
    license=DEFAULT_LICENSE_URL,
    open_weights=True,
    public_training_code=None,
    public_training_data=None,
    framework=["PyTorch", "Transformers", "safetensors"],
    reference=DEFAULT_REFERENCE_URL,
    similarity_fn_name=ScoringFunction.COSINE,
    use_instructions=False,
    training_datasets=set(),
    modalities=["audio"],
    model_type=["dense"],
    citation=None,
    contacts=None,
    experiment_kwargs=None,
)
