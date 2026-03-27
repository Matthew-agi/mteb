from __future__ import annotations

from mteb._create_dataloaders import (
    _batch_indices_by_budget,
    _estimate_audio_clip_cost,
    _resolve_audio_max_batch_items,
)


def test_estimate_audio_clip_cost_sliding_window():
    assert (
        _estimate_audio_clip_cost(
            6.1,
            clip_seconds=2.0,
            hop_seconds=2.0,
            full_clip=False,
            sliding_window=True,
        )
        == 4
    )


def test_estimate_audio_clip_cost_fixed_crop():
    assert (
        _estimate_audio_clip_cost(
            30.0,
            clip_seconds=2.0,
            hop_seconds=2.0,
            full_clip=False,
            sliding_window=False,
        )
        == 1
    )


def test_batch_indices_by_budget_groups_by_cost():
    assert _batch_indices_by_budget([3, 3, 1, 1], target_budget=4, max_batch_items=4) == [
        [0],
        [1, 2],
        [3],
    ]


def test_resolve_audio_max_batch_items_auto_from_costs():
    assert _resolve_audio_max_batch_items([1, 1, 1, 1], target_budget=2048, max_batch_items=None) == 64
    assert _resolve_audio_max_batch_items([16, 16, 16, 16], target_budget=512, max_batch_items=None) == 32
    assert _resolve_audio_max_batch_items([1, 2, 3], target_budget=512, max_batch_items=7) == 7
