from __future__ import annotations

import io
import logging
import math
import os
import warnings
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import torch
from datasets import Audio, Dataset, Image
from torch.utils.data import DataLoader, default_collate

from mteb.types import (
    ConversationTurn,
    PromptType,
)
from mteb.types._encoder_io import AudioInputItem, VideoInputItem

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchcodec.decoders import VideoDecoder  # type: ignore[import-untyped]

    from mteb.abstasks.task_metadata import TaskMetadata
    from mteb.types import (
        BatchedInput,
        Conversation,
    )
    from mteb.types._encoder_io import (
        TextInput,
    )

logger = logging.getLogger(__name__)


def _estimate_audio_duration_seconds(audio: Any) -> float | None:
    if audio is None:
        return None

    if isinstance(audio, dict):
        array = audio.get("array")
        sampling_rate = audio.get("sampling_rate")
        if array is not None and sampling_rate:
            try:
                return float(np.asarray(array).reshape(-1).shape[0]) / float(sampling_rate)
            except Exception:
                pass

        data_bytes = audio.get("bytes")
        path = audio.get("path")
        try:
            import soundfile as sf
        except Exception:
            sf = None
        if sf is not None:
            if data_bytes is not None:
                try:
                    info = sf.info(io.BytesIO(data_bytes))
                    if info.samplerate:
                        return float(info.frames) / float(info.samplerate)
                except Exception:
                    pass
            if path and isinstance(path, str) and os.path.exists(path):
                try:
                    info = sf.info(path)
                    if info.samplerate:
                        return float(info.frames) / float(info.samplerate)
                except Exception:
                    pass
        if data_bytes is not None:
            # Coarse fallback when only compressed bytes are available.
            return max(1.0, float(len(data_bytes)) / 32000.0)

    array = getattr(audio, "array", None)
    sampling_rate = getattr(audio, "sampling_rate", None) or getattr(audio, "sample_rate", None)
    if array is not None and sampling_rate:
        try:
            return float(np.asarray(array).reshape(-1).shape[0]) / float(sampling_rate)
        except Exception:
            pass

    if hasattr(audio, "get_all_samples"):
        try:
            samples = audio.get_all_samples()
            data = getattr(samples, "data", None)
            sampling_rate = getattr(samples, "sample_rate", None)
            if data is not None and sampling_rate:
                return float(np.asarray(data).reshape(-1).shape[0]) / float(sampling_rate)
        except Exception:
            pass

    return None


def _estimate_audio_clip_cost(
    duration_seconds: float | None,
    *,
    clip_seconds: float,
    hop_seconds: float,
    full_clip: bool,
    sliding_window: bool,
) -> int:
    if duration_seconds is None or not math.isfinite(duration_seconds) or duration_seconds <= 0:
        return 1
    if full_clip:
        return max(1, int(math.ceil(duration_seconds / max(clip_seconds, 1e-6))))
    if not sliding_window:
        return 1
    return max(1, 1 + int(math.ceil(max(0.0, duration_seconds - clip_seconds) / max(hop_seconds, 1e-6))))


def _batch_indices_by_budget(
    costs: list[int],
    *,
    target_budget: int,
    max_batch_items: int,
) -> list[list[int]]:
    if target_budget <= 0:
        target_budget = 1
    if max_batch_items <= 0:
        max_batch_items = 1

    batches: list[list[int]] = []
    current: list[int] = []
    current_cost = 0
    for idx, raw_cost in enumerate(costs):
        cost = max(1, int(raw_cost))
        if current and (len(current) >= max_batch_items or current_cost + cost > target_budget):
            batches.append(current)
            current = []
            current_cost = 0
        current.append(idx)
        current_cost += cost
        if len(current) >= max_batch_items or current_cost >= target_budget:
            batches.append(current)
            current = []
            current_cost = 0
    if current:
        batches.append(current)
    return batches


def _resolve_audio_max_batch_items(
    costs: list[int],
    *,
    target_budget: int,
    max_batch_items: int | None,
) -> int:
    if max_batch_items is not None and int(max_batch_items) > 0:
        return int(max_batch_items)

    if target_budget <= 0:
        target_budget = 1
    if not costs:
        return 1

    cost_array = np.asarray([max(1, int(cost)) for cost in costs], dtype=np.int32)
    reference_cost = int(np.percentile(cost_array, 75))
    reference_cost = max(1, reference_cost)
    inferred = max(1, target_budget // reference_cost)
    # Keep a conservative hard cap so short-clip datasets do not overwhelm torchcodec.
    return max(4, min(64, inferred))


def _maybe_disable_audio_decoding(
    dataset: Dataset,
    *,
    audio_column_name: str,
) -> Dataset:
    feature = dataset.features.get(audio_column_name)
    if not isinstance(feature, Audio) or not getattr(feature, "decode", False):
        return dataset
    return dataset.cast_column(
        audio_column_name,
        Audio(
            sampling_rate=getattr(feature, "sampling_rate", None),
            mono=getattr(feature, "mono", True),
            decode=False,
            id=getattr(feature, "id", None),
        ),
    )


class _AudioBudgetBatchSampler:
    def __init__(
        self,
        dataset: Dataset,
        *,
        audio_column_name: str,
        target_clip_budget: int,
        max_batch_items: int | None,
        clip_seconds: float,
        hop_seconds: float,
        full_clip: bool,
        sliding_window: bool,
    ) -> None:
        costs: list[int] = []
        for audio in dataset[audio_column_name]:
            duration_seconds = _estimate_audio_duration_seconds(audio)
            costs.append(
                _estimate_audio_clip_cost(
                    duration_seconds,
                    clip_seconds=clip_seconds,
                    hop_seconds=hop_seconds,
                    full_clip=full_clip,
                    sliding_window=sliding_window,
                )
            )
        self.target_clip_budget = max(1, int(target_clip_budget))
        self.max_batch_items = _resolve_audio_max_batch_items(
            costs,
            target_budget=self.target_clip_budget,
            max_batch_items=max_batch_items,
        )
        self._batches = _batch_indices_by_budget(
            costs,
            target_budget=self.target_clip_budget,
            max_batch_items=self.max_batch_items,
        )

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches

    def __len__(self) -> int:
        return len(self._batches)


def _create_dataloader_from_texts(
    text: list[str],
    batch_size: int = 32,
    num_proc: int | None = None,
    **kwargs: Any,
) -> DataLoader[TextInput]:
    """Create a dataloader from a list of text.

    Args:
        text: A list of text to create a dataloader from.
        batch_size: Batch size for the dataloader.
        num_proc: Number of processes to use.
        kwargs: Not used, present catching extra arguments.

    Returns:
        A dataloader with the text.
    """
    dataset = Dataset.from_dict({"text": text})
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_proc if num_proc is not None and num_proc > 1 else 0,
    )


def _corpus_to_dict(
    row: dict[str, str],
) -> dict[str, str]:
    text = (
        (row["title"] + " " + row["text"]).strip()
        if "title" in row and len(row["title"]) > 0
        else row["text"].strip()
    )
    new_row = {
        "id": row["id"],
        "text": text,
        "body": row["text"],
    }
    # dataloaders can't handle None
    if "title" in row and row["title"] is not None and len(row["title"]) > 0:
        new_row["title"] = row["title"]
    return new_row


def _combine_queries_with_instruction_text(row: dict[str, str]) -> dict[str, str]:
    row["query"] = row["text"]

    if "instruction" in row and row["instruction"] is not None:
        row["text"] = row["query"] + " " + row["instruction"]
    else:
        row["text"] = row["query"]
    return row


def _convert_conv_history_to_query(
    row: dict[str, str | list[str] | Conversation],
) -> dict[str, str | Conversation]:
    """Convert a conversation history to a single query string.

    If row "conversation" is a list of strings, it will be joined with "; " and the role will be set to "user".
    If row "conversation" is a list of dictionaries, it will be converted to a string with the format "role: content; role: content; ...".

    Returns:
        The updated row with the "query" and "text" fields set to the conversation string, and the "conversation" field set to the list of ConversationTurn.
    """
    conversation = row["text"]
    # if it's a list of strings, just join them
    if isinstance(conversation, list) and isinstance(conversation[0], str):
        conversation_ = cast("list[str]", conversation)
        conv_str = "; ".join(conversation_)
        current_conversation = [
            ConversationTurn(role="user", content=message) for message in conversation_
        ]
        warnings.warn(
            "Conversations are a list of strings. Used 'user' role for all turns.",
            category=UserWarning,
        )
    # otherwise, it's a list of dictionaries, which we need to convert to strings
    elif isinstance(conversation, list) and isinstance(conversation[0], dict):
        conv = []
        current_conversation = []
        for i, turn in enumerate(conversation):
            error_msg = (
                "When converting conversations lists of dictionary to string, each turn in the conversation "
                + "must be a dictionary with 'role' and 'content' keys"
            )
            if not isinstance(turn, dict):
                raise ValueError(f"Turn {i} is not a dictionary. " + error_msg)

            # check for keys 'role' and 'content' in the dictionary, if not found, raise an error
            if "role" not in turn:
                raise ValueError("Key 'role' not found in the dictionary. " + error_msg)
            if "content" not in turn:
                raise ValueError(
                    "Key 'content' not found in the dictionary. " + error_msg
                )
            current_conversation.append(
                ConversationTurn(role=turn["role"], content=turn["content"])
            )
            conv.append(f"{turn['role']}: {turn['content']}")
        conv_str = "; ".join(conv)
    else:
        raise ValueError(
            "Conversations must be a list consisting of strings or dictionaries with 'role' and 'content' keys"
        )

    row["query"] = conv_str

    if "instruction" in row:
        conv_str = f"{row['instruction']} {conv_str}"

    row["text"] = conv_str
    row["conversation"] = current_conversation
    return cast("dict[str, str | list[ConversationTurn]]", row)


def _transform_image_to_rgb(
    image: Any, transform: Callable[[Any], Any] | None = None
) -> Any:
    """Convert image to RGB and apply a transformation (e.g. PILToTensor).

    Args:
        image: The input image, either a PIL image or a tensor.
        transform: An optional transformation function to apply to the image.

    Returns:
        The transformed image in RGB format.
    """
    # For PIL images: ensure RGB format.
    if hasattr(image, "mode") and image.mode != "RGB":
        image = image.convert("RGB")
    # For tensor images with 1 channel: repeat channels.
    elif isinstance(image, torch.Tensor) and image.shape[0] == 1:
        image = image.repeat(3, 1, 1)
    # Apply the additional transformation (e.g., conversion to tensor) if provided.
    if transform is not None:
        return transform(image)
    return image


def _convert_images_to_rgb(
    example: dict[str, Any],
    image_col_name: str = "image",
    transform: Callable[[Any], Any] | None = None,
) -> dict[str, Any]:
    if image_col_name not in example:
        return example
    example[image_col_name] = _transform_image_to_rgb(
        example[image_col_name], transform
    )
    return example


def _prepare_image_dataset(
    dataset: Dataset,
    image_column_name: str | None = None,
    transform: Callable[[Any], Any] | None = None,
    num_proc: int | None = None,
) -> Dataset:
    """Prepare the image dataset by converting images to RGB and applying transformations."""
    if (
        image_column_name
        and image_column_name in dataset.column_names
        and "image" not in dataset.column_names
    ):
        dataset = dataset.rename_column(image_column_name, "image")
    # don't process image if it's already in the correct format
    if isinstance(dataset.features["image"], Image):
        return dataset
    return dataset.map(
        _convert_images_to_rgb,
        fn_kwargs={"image_col_name": "image", "transform": transform},
        desc="Converting images to RGB",
        num_proc=num_proc,
    )


def _custom_collate_fn(batch: list[dict[str, Any]]) -> BatchedInput:
    """Custom collate function for DataLoader.

    - For the "image", "conversation" key, leave the images as a list (to avoid stacking errors).
    - For other keys, use the default collate.

    Args:
        batch: A list of dictionaries to collate.

    Returns:
        A collated dictionary.
    """
    collated = {}
    for key in batch[0]:
        if key in (
            "image",  # images can be with different sizes
            "conversation",  # conversations are lists of varying lengths
            "audio",  # audio can have different lengths
            "video",  # video VideoDecoder objects can't be collated
        ):
            collated[key] = [item[key] for item in batch]
        else:
            if any(item[key] is None for item in batch):
                raise ValueError(f"Found None in batch for key '{key}'")
            collated[key] = default_collate([item[key] for item in batch])
    return cast("BatchedInput", collated)


def _prepare_dataset(
    dataset: Dataset,
    task_metadata: TaskMetadata,
    prompt_type: PromptType | None = None,
    input_column: str | None = None,
    num_proc: int | None = None,
) -> Dataset:
    """Apply all modality-specific transformations to the dataset.

    Returns the transformed Dataset (no DataLoader wrapping).
    """
    modalities = task_metadata.get_modalities(prompt_type)

    if "text" in modalities:
        if prompt_type == PromptType.document:
            dataset = dataset.map(
                _corpus_to_dict,
                desc="Standardizing text corpus format",
                num_proc=num_proc,
            )
        elif prompt_type == PromptType.query:
            if isinstance(dataset["text"][0], list):
                dataset = dataset.map(
                    _convert_conv_history_to_query,
                    desc="Converting conversations to queries",
                    num_proc=num_proc,
                )
            else:
                dataset = dataset.map(
                    _combine_queries_with_instruction_text,
                    desc="Processing queries for dataloading",
                    num_proc=num_proc,
                )

    if "image" in modalities:
        dataset = _prepare_image_dataset(
            dataset,
            image_column_name=input_column if input_column else "image",
            num_proc=num_proc,
        )
    for modality in ("audio", "video"):
        if modality in modalities:
            if (
                input_column
                and input_column in dataset.column_names
                and modality not in dataset.column_names
            ):
                dataset = dataset.rename_column(input_column, modality)

    return dataset


def create_dataloader(
    dataset: Dataset,
    task_metadata: TaskMetadata,
    prompt_type: PromptType | None = None,
    input_column: str | None = None,
    batch_size: int = 32,
    num_proc: int | None = None,
    **kwargs: Any,
) -> DataLoader[BatchedInput]:
    """Create a dataloader from a dataset.

    If prompt_type is None, it will create a dataloader based on the modalities of the task.
    if prompt_type is provided, it will create a dataloader for the specified prompt type.

    Args:
        dataset: The dataset to create a dataloader from.
        task_metadata: The metadata of the task.
        prompt_type: The type of prompt to create a dataloader for. If None, it will be inferred from the task metadata.
        input_column: The column to use as input. If None, it will use the first column that matches the modality.
        batch_size: The batch size for the dataloader.
        num_proc: The number of processes to use for dataset processing.
        **kwargs: Additional arguments to pass to the dataloader creation functions.

    Returns:
        A dataloader for the dataset.
    """
    if (
        prompt_type is None
        and task_metadata.modalities == ["text"]
        and input_column is not None
    ):
        return _create_dataloader_from_texts(
            dataset[input_column],
            batch_size=batch_size,
        )

    prepared = _prepare_dataset(
        dataset,
        task_metadata,
        prompt_type=prompt_type,
        input_column=input_column,
        num_proc=num_proc,
    )

    modalities = task_metadata.get_modalities(prompt_type)
    audio_dynamic_batching = bool(kwargs.get("audio_dynamic_batching", False))
    if "audio" in modalities and audio_dynamic_batching:
        audio_column_name = "audio" if "audio" in prepared.column_names else input_column
        if audio_column_name is not None and audio_column_name in prepared.column_names:
            prepared = _maybe_disable_audio_decoding(
                prepared,
                audio_column_name=audio_column_name,
            )
            raw_max_batch_items = kwargs.get("audio_max_batch_items")
            max_batch_items = (
                int(raw_max_batch_items)
                if raw_max_batch_items is not None and int(raw_max_batch_items) > 0
                else None
            )
            target_clip_budget = int(kwargs.get("clip_batch_size", batch_size))
            clip_seconds = float(kwargs.get("audio_clip_seconds", 2.0))
            hop_seconds = float(kwargs.get("audio_window_hop_seconds", clip_seconds))
            full_clip = bool(kwargs.get("audio_full_clip", False))
            sliding_window = bool(kwargs.get("audio_sliding_window", False))
            batch_sampler = _AudioBudgetBatchSampler(
                prepared,
                audio_column_name=audio_column_name,
                target_clip_budget=target_clip_budget,
                max_batch_items=max_batch_items,
                clip_seconds=clip_seconds,
                hop_seconds=hop_seconds,
                full_clip=full_clip,
                sliding_window=sliding_window,
            )
            logger.info(
                "Audio dynamic batching enabled: target_clip_budget=%s max_batch_items=%s num_batches=%s",
                batch_sampler.target_clip_budget,
                batch_sampler.max_batch_items,
                len(batch_sampler),
            )
            return DataLoader(
                prepared,
                batch_sampler=batch_sampler,
                collate_fn=_custom_collate_fn,
                num_workers=num_proc if num_proc is not None and num_proc > 1 else 0,
            )

    return DataLoader(
        prepared,
        batch_size=batch_size,
        collate_fn=_custom_collate_fn,
        num_workers=num_proc if num_proc is not None and num_proc > 1 else 0,
        shuffle=False,
    )


class AudioCollator:
    """Collator for audio data that resamples audio to a target sampling rate and optionally truncates to a maximum number of samples."""

    def __init__(
        self,
        target_sampling_rate: int,
        max_samples: int | None = None,
    ) -> None:
        """Initialize the collator.

        Args:
            target_sampling_rate: The sampling rate to resample the audio to.
            max_samples: The maximum number of samples to keep for each audio. If None, no truncation is applied.
        """
        self.target_sampling_rate = target_sampling_rate
        self.max_samples = max_samples

    def __call__(self, inputs: list[dict[str, Any]]) -> BatchedInput:
        return self.resample_audios(
            inputs,
            target_sampling_rate=self.target_sampling_rate,
            max_samples=self.max_samples,
        )

    @staticmethod
    def resample_audios(
        inputs: list[dict[str, Any]],
        target_sampling_rate: int,
        max_samples: int | None = None,
    ) -> BatchedInput:
        """Resample a batch of audio inputs to a target sampling rate and optionally truncate to a maximum number of samples.

        Args:
            inputs: A list of dictionaries containing audio data under the "audio" key, where each audio is a dictionary with "array" and "sampling_rate" keys.
            target_sampling_rate: The sampling rate to resample the audio to.
            max_samples: The maximum number of samples to keep for each audio. If None, no truncation is applied.
        """
        collated_inputs = []
        for row in inputs:
            audio_array = AudioCollator.resample_audio(
                row,
                target_sampling_rate,
                max_samples,
            )
            row["audio"] = AudioInputItem(
                array=audio_array, sampling_rate=target_sampling_rate
            )
            collated_inputs.append(row)
        return cast(
            "BatchedInput",
            {
                key: [row[key] for row in collated_inputs]
                for key in collated_inputs[0].keys()
            },
        )

    @staticmethod
    def resample_audio(
        audio: dict[str, Any],
        target_sampling_rate: int,
        max_samples: int | None = None,
    ) -> np.typing.NDArray[np.floating]:
        """Resample an audio input to a target sampling rate and optionally truncate to a maximum number of samples.

        Args:
            audio: A list of dictionaries containing audio data under the "audio" key, where each audio is a dictionary with "array" and "sampling_rate" keys.
            target_sampling_rate: The sampling rate to resample the audio to.
            max_samples: The maximum number of samples to keep for each audio. If None, no truncation is applied.
        """
        import torchaudio

        audio = audio["audio"]
        if audio["sampling_rate"] != target_sampling_rate:
            logger.debug(
                f"Resampling audio from {audio['sampling_rate']} Hz to {target_sampling_rate} Hz."
            )
            resampler = torchaudio.transforms.Resample(
                orig_freq=audio["sampling_rate"],
                new_freq=target_sampling_rate,
            )
            audio_array = torch.from_numpy(audio["array"]).float()
            audio_array = resampler(audio_array)
            audio_array = audio_array.numpy()
        else:
            audio_array = audio["array"]

        # Convert to mono if needed
        if audio_array.ndim > 1 and audio_array.shape[0] > 1:
            audio_array = np.mean(audio_array, axis=0)

        if max_samples is not None:
            num_samples = audio_array.shape[-1]
            if num_samples > max_samples:
                audio_array = audio_array[..., :max_samples]
        return audio_array


class VideoCollator:
    """Collator for video data that optionally truncates to a maximum number of frames."""

    def __init__(
        self,
        max_frames: int,
        target_sampling_rate: int = 32_000,
        max_samples: int | None = None,
    ) -> None:
        """Initialize the collator.

        Args:
            max_frames: The maximum number of frames to keep for each video. If None, no truncation is applied.
            target_sampling_rate: The sampling rate to resample the audio to.
            max_samples: The maximum number of samples to keep for each audio. If None, no truncation is applied.
        """
        self.max_frames = max_frames
        self.audio_collator = AudioCollator(
            target_sampling_rate=target_sampling_rate, max_samples=max_samples
        )

    def __call__(self, inputs: list[dict[str, Any]]) -> BatchedInput:
        if "video" not in inputs[0]:
            return cast("BatchedInput", inputs)

        collated_inputs = []
        for row in inputs:
            videos = row.pop("video")
            video_inputs = []
            for video in videos:
                frames = self.resample_video(video["frames"], self.max_frames)
                audio = self.audio_collator.resample_audio(
                    video,
                    target_sampling_rate=self.audio_collator.target_sampling_rate,
                    max_samples=self.audio_collator.max_samples,
                )
                video_inputs.append(
                    VideoInputItem(
                        frames=frames,
                        audio=AudioInputItem(
                            array=audio,
                            sampling_rate=self.audio_collator.target_sampling_rate,
                        ),
                    )
                )
            row["video"] = video_inputs
            collated_inputs.append(row)

        return cast(
            "BatchedInput",
            {
                key: [row[key] for row in collated_inputs]
                for key in collated_inputs[0].keys()
            },
        )

    @staticmethod
    def resample_video(
        video: VideoDecoder,
        max_video_frames: int,
    ) -> torch.Tensor:
        """Resample a video input to a target number of frames.

        Args:
            video: A VideoDecoder object containing the video data.
            max_video_frames: The maximum number of frames to keep for each video. If None, no truncation is applied.
        """
        video_frames = video.metadata.num_frames
        frame_step = (
            max(1, video_frames // max_video_frames)
            if max_video_frames is not None
            else 1
        )
        selected_frames = (
            list(range(0, video_frames, frame_step))[:max_video_frames]
            if max_video_frames is not None
            else list(range(video_frames))
        )
        return video.get_frames_at(selected_frames).data
