#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import mteb
from mteb import ResultCache
from mteb.models.model_implementations.hear_s11_models import (
    DEFAULT_CLIP_SECONDS,
    DEFAULT_CROP,
    DEFAULT_MODEL_ID,
    DEFAULT_REFERENCE_URL,
    DEFAULT_REVISION,
    DEFAULT_TARGET_SR,
    DEFAULT_WINDOW_HOP_SECONDS,
    DEFAULT_WINDOW_POOL,
)


DEFAULT_BENCHMARK_NAME = "MAEB(beta, audio-only)"
MAEB_AUDIO_ONLY_TASKS: list[str] = [
    "JamAltArtistA2ARetrieval",
    "BeijingOpera",
    "BirdCLEF",
    "CREMA_D",
    "CommonLanguageAgeDetection",
    "GTZANGenre",
    "IEMOCAPGender",
    "MInDS14",
    "MridinghamTonic",
    "VoxCelebSA",
    "VoxPopuliLanguageID",
    "CREMA_DClustering",
    "VehicleSoundClustering",
    "VoxPopuliGenderClustering",
    "SIBFLEURS",
    "CREMADPairClassification",
    "NMSQAPairClassification",
    "VoxPopuliAccentPairClassification",
    "GTZANAudioReranking",
]
DEFAULT_MODEL_KWARGS: dict[str, Any] = {
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
}


def _parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _default_output_dir(*, model_tag: str, revision: str, benchmark_name: str) -> Path:
    safe_model = model_tag.replace("/", "__")
    safe_benchmark = (
        benchmark_name.replace(" ", "-")
        .replace("(", "")
        .replace(")", "")
        .replace(",", "")
        .replace("/", "-")
    )
    return Path.cwd() / "results" / f"{safe_model}-{safe_benchmark}-{revision}"


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Run MAEB(beta, audio-only) with matthewagi/HeAR-s1.1 through an MTEB fork."
    )
    ap.add_argument("--model-id", type=str, default=DEFAULT_MODEL_ID)
    ap.add_argument("--revision", type=str, default=DEFAULT_REVISION)
    ap.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    ap.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Internal clip/window batch size used inside the model encoder.",
    )
    ap.add_argument(
        "--item-batch-size",
        type=int,
        default=0,
        help="Override for maximum raw audio items per dataloader batch. <=0 means auto.",
    )
    ap.add_argument(
        "--auto-item-batch-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt audio item batches to an estimated clip/window budget derived from audio lengths.",
    )
    ap.add_argument("--target-sr", type=int, default=None)
    ap.add_argument("--clip-seconds", type=float, default=None)
    ap.add_argument("--crop", choices=["center", "peak"], default=None)
    ap.add_argument("--full-clip", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument(
        "--sliding-window",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override the model default sliding-window behavior.",
    )
    ap.add_argument("--window-hop-seconds", type=float, default=None)
    ap.add_argument("--window-pool", choices=["mean"], default=None)
    ap.add_argument("--compile", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--compile-mode", type=str, default=None)
    ap.add_argument("--compile-dynamic", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--amp-dtype", choices=["auto", "bfloat16", "float16"], default=None)
    ap.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--cudnn-benchmark", action=argparse.BooleanOptionalAction, default=None)
    ap.add_argument("--benchmark-name", type=str, default=DEFAULT_BENCHMARK_NAME)
    ap.add_argument("--tasks", type=str, default=None, help="Optional comma-separated task override.")
    ap.add_argument(
        "--prefer-benchmark-registry",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    ap.add_argument("--num-proc", type=int, default=0)
    ap.add_argument("--show-progress-bar", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--overwrite-strategy",
        choices=["always", "never", "only-missing", "only-cache"],
        default="always",
    )
    ap.add_argument("--output-dir", type=Path, default=None)
    return ap.parse_args()


def _resolve_tasks(
    *,
    benchmark_name: str,
    explicit_tasks: Sequence[str],
    prefer_benchmark_registry: bool,
) -> tuple[list[Any], str]:
    if prefer_benchmark_registry:
        for candidate in (benchmark_name, "MAEB(audio-only)"):
            try:
                benchmark = mteb.get_benchmark(candidate)
                tasks = list(benchmark.tasks)
                if tasks:
                    return tasks, candidate
            except Exception:
                pass

    tasks = list(mteb.get_tasks(tasks=list(explicit_tasks)))
    if not tasks:
        raise SystemExit("Could not resolve any MAEB audio-only tasks from MTEB.")
    return tasks, "explicit-task-list"

def _build_model_overrides(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    overrides: dict[str, Any] = {}
    if args.target_sr is not None:
        overrides["target_sr"] = int(args.target_sr)
    if args.clip_seconds is not None:
        overrides["clip_seconds"] = float(args.clip_seconds)
    if args.crop is not None:
        overrides["crop"] = str(args.crop)
    if args.full_clip is not None:
        overrides["full_clip"] = bool(args.full_clip)
    if args.sliding_window is not None:
        overrides["sliding_window"] = bool(args.sliding_window)
    if args.window_hop_seconds is not None:
        overrides["window_hop_seconds"] = float(args.window_hop_seconds)
    if args.window_pool is not None:
        overrides["window_pool"] = str(args.window_pool)
    if args.compile is not None:
        overrides["compile_enabled"] = bool(args.compile)
    if args.compile_mode is not None:
        overrides["compile_mode"] = str(args.compile_mode)
    if args.compile_dynamic is not None:
        overrides["compile_dynamic"] = bool(args.compile_dynamic)
    if args.amp is not None:
        overrides["amp_enabled"] = bool(args.amp)
    if args.amp_dtype is not None:
        overrides["amp_dtype"] = str(args.amp_dtype)
    if args.tf32 is not None:
        overrides["enable_tf32"] = bool(args.tf32)
    if args.cudnn_benchmark is not None:
        overrides["cudnn_benchmark"] = bool(args.cudnn_benchmark)

    if overrides.get("full_clip") is True and "sliding_window" not in overrides:
        overrides["sliding_window"] = False
    if overrides.get("sliding_window") is True and "full_clip" not in overrides:
        overrides["full_clip"] = False

    effective = dict(DEFAULT_MODEL_KWARGS)
    effective.update(overrides)
    if effective["full_clip"] and effective["sliding_window"]:
        raise SystemExit("--full-clip and --sliding-window are mutually exclusive.")
    return overrides, effective


def _mode_description(effective_model_kwargs: dict[str, Any]) -> str:
    if effective_model_kwargs["full_clip"]:
        return "full clip"
    if effective_model_kwargs["sliding_window"]:
        return (
            f"sliding window ({effective_model_kwargs['clip_seconds']:.2f}s / "
            f"{effective_model_kwargs['window_hop_seconds']:.2f}s {effective_model_kwargs['window_pool']})"
        )
    return f"{effective_model_kwargs['crop']} crop ({effective_model_kwargs['clip_seconds']:.2f}s)"


def _results_output_path(cache: ResultCache, task_name: str, model_meta: Any) -> Path:
    return cache.get_task_result_path(task_name=task_name, model_name=model_meta)


def _write_overall_results(
    *,
    result: Any,
    output_dir: Path,
    results_revision_dir: Path | None,
    benchmark_name: str,
    task_source: str,
) -> None:
    task_scores: dict[str, float] = {}
    for task_result in result.task_results:
        try:
            task_scores[task_result.task_name] = float(task_result.main_score)
        except Exception:
            continue
    mean_main_score = float(np.mean(list(task_scores.values()))) if task_scores else None
    summary = {
        "model_name": result.model_name,
        "revision": result.model_revision,
        "reference": DEFAULT_REFERENCE_URL,
        "benchmark_name": benchmark_name,
        "task_source": task_source,
        "task_count": len(task_scores),
        "mean_main_score": mean_main_score,
        "tasks": task_scores,
    }
    (output_dir / "overall_results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if results_revision_dir is not None:
        (results_revision_dir / "overall_results.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )


def main() -> None:
    args = _parse_args()
    model_overrides, effective_model_kwargs = _build_model_overrides(args)
    explicit_tasks = _parse_csv(args.tasks) or list(MAEB_AUDIO_ONLY_TASKS)
    item_batch_size_value = (
        int(args.item_batch_size)
        if int(args.item_batch_size) > 0
        else (1 if bool(args.auto_item_batch_size) else int(args.batch_size))
    )
    item_batch_size_override = (None if int(args.item_batch_size) <= 0 else int(args.item_batch_size))
    tasks, task_source = _resolve_tasks(
        benchmark_name=args.benchmark_name,
        explicit_tasks=explicit_tasks,
        prefer_benchmark_registry=bool(args.prefer_benchmark_registry),
    )

    model = mteb.get_model(
        args.model_id,
        revision=args.revision,
        device=args.device,
        **model_overrides,
    )

    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else _default_output_dir(
            model_tag=args.model_id,
            revision=args.revision,
            benchmark_name=args.benchmark_name,
        )
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    cache = ResultCache(cache_path=output_dir)

    run_config = {
        "model_id": args.model_id,
        "revision": args.revision,
        "device": args.device,
        "item_batch_size_override": item_batch_size_override,
        "item_batch_size_effective": int(item_batch_size_value),
        "clip_batch_size": int(args.batch_size),
        "auto_item_batch_size": bool(args.auto_item_batch_size),
        "benchmark_name": args.benchmark_name,
        "task_source": task_source,
        "tasks": [task.metadata.name for task in tasks],
        "num_proc": (None if args.num_proc <= 0 else int(args.num_proc)),
        "overwrite_strategy": args.overwrite_strategy,
        "show_progress_bar": bool(args.show_progress_bar),
        "model_overrides": model_overrides,
        "effective_model_kwargs": effective_model_kwargs,
        "experiment_name": getattr(model.mteb_model_meta, "experiment_name", None),
    }
    (output_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    print(f"Model: {args.model_id}", flush=True)
    print(f"Revision: {args.revision}", flush=True)
    print(f"Reference: {DEFAULT_REFERENCE_URL}", flush=True)
    print(f"Device: {getattr(model, 'device_type', args.device)}", flush=True)
    item_batch_label = "auto" if int(args.item_batch_size) <= 0 else str(int(args.item_batch_size))
    print(f"Dataloader item batch size (max): {item_batch_label}", flush=True)
    print(f"Auto item batch sizing: {int(bool(args.auto_item_batch_size))}", flush=True)
    print(f"Embedding clip batch size: {int(args.batch_size)}", flush=True)
    print(f"Mode: {_mode_description(effective_model_kwargs)}", flush=True)
    print(f"Tasks: {len(tasks)}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    if getattr(model.mteb_model_meta, "experiment_name", None):
        print(f"Experiment name: {model.mteb_model_meta.experiment_name}", flush=True)

    result = mteb.evaluate(
        model,
        tasks=tasks,
        cache=cache,
        overwrite_strategy=args.overwrite_strategy,
        show_progress_bar=bool(args.show_progress_bar),
        encode_kwargs={
            "batch_size": int(item_batch_size_value),
            "clip_batch_size": int(args.batch_size),
            "audio_dynamic_batching": bool(args.auto_item_batch_size),
            "audio_max_batch_items": item_batch_size_override,
            "audio_clip_seconds": float(effective_model_kwargs["clip_seconds"]),
            "audio_window_hop_seconds": float(effective_model_kwargs["window_hop_seconds"]),
            "audio_full_clip": bool(effective_model_kwargs["full_clip"]),
            "audio_sliding_window": bool(effective_model_kwargs["sliding_window"]),
            "show_progress_bar": bool(args.show_progress_bar),
        },
        num_proc=(None if args.num_proc <= 0 else int(args.num_proc)),
        public_only=False,
    )

    raw_result_path = output_dir / "model_result.json"
    result.to_disk(raw_result_path)

    results_revision_dir: Path | None = None
    if tasks:
        result_path = _results_output_path(cache, tasks[0].metadata.name, model.mteb_model_meta)
        results_revision_dir = result_path.parent

    _write_overall_results(
        result=result,
        output_dir=output_dir,
        results_revision_dir=results_revision_dir,
        benchmark_name=args.benchmark_name,
        task_source=task_source,
    )

    print(f"Saved raw model result: {raw_result_path}", flush=True)
    if results_revision_dir is not None:
        print(f"Saved results layout: {results_revision_dir}", flush=True)


if __name__ == "__main__":
    main()
