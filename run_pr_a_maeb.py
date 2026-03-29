#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import mteb
from mteb import ResultCache


DEFAULT_MODEL_ID = "matthewagi/HeAR-s1.1"
DEFAULT_REVISION = "a5776bebff935a81c79720467ae1e10a4effe10e"
DEFAULT_BENCHMARK_CANDIDATES = (
    "MAEB(beta, audio-only)",
    "MAEB(audio-only)",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attempt a full MAEB(audio-only) run with HeAR-s1.1 from an MTEB fork."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Internal clip/window batch size used inside the model encoder.",
    )
    parser.add_argument(
        "--item-batch-size",
        type=int,
        default=0,
        help="Override for maximum raw audio items per dataloader batch. <=0 means auto.",
    )
    parser.add_argument(
        "--auto-item-batch-size",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Adapt audio item batches to an estimated clip/window budget derived from audio lengths.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("hear_s11_pr_a_maeb"))
    return parser.parse_args()


def resolve_benchmark() -> tuple[object, str]:
    last_err: Exception | None = None
    for candidate in DEFAULT_BENCHMARK_CANDIDATES:
        try:
            return mteb.get_benchmark(candidate), candidate
        except Exception as exc:  # pragma: no cover - defensive for runtime only
            last_err = exc
    raise RuntimeError(
        "Could not resolve MAEB audio-only benchmark from MTEB registry."
    ) from last_err


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    item_batch_size_value = (
        int(args.item_batch_size)
        if int(args.item_batch_size) > 0
        else (1 if bool(args.auto_item_batch_size) else int(args.batch_size))
    )
    item_batch_size_override = (None if int(args.item_batch_size) <= 0 else int(args.item_batch_size))

    print("loading model...", flush=True)
    model = mteb.get_model(
        args.model_id,
        revision=args.revision,
        device=args.device,
    )
    print("model loaded", flush=True)

    print("loading benchmark...", flush=True)
    benchmark, benchmark_name = resolve_benchmark()
    tasks = list(benchmark.tasks)
    print(f"benchmark: {benchmark_name}", flush=True)
    print(f"tasks: {len(tasks)}", flush=True)

    cache = ResultCache(cache_path=args.output_dir)

    print("starting evaluation...", flush=True)
    result = mteb.evaluate(
        model,
        tasks=tasks,
        cache=cache,
        overwrite_strategy="always",
        show_progress_bar=True,
        encode_kwargs={
            "batch_size": int(item_batch_size_value),
            "clip_batch_size": int(args.batch_size),
            "audio_dynamic_batching": bool(args.auto_item_batch_size),
            "audio_max_batch_items": item_batch_size_override,
            "audio_clip_seconds": 2.0,
            "audio_window_hop_seconds": 2.0,
            "audio_full_clip": False,
            "audio_sliding_window": True,
            "show_progress_bar": True,
        },
        public_only=False,
    )

    raw_result_path = args.output_dir / "model_result.json"
    result.to_disk(raw_result_path)
    print(f"saved: {raw_result_path}", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
