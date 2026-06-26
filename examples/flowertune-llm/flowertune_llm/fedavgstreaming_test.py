"""Tests for layerwise aggregation batching helpers."""

import sys
import types

import pytest
import torch

task_stub = types.ModuleType("flowertune_llm.task")
task_stub.state_dict_fingerprint = lambda state_dict: 0.0
sys.modules.setdefault("flowertune_llm.task", task_stub)

from flowertune_llm.fedavgstreaming import (
    _batch_entries_by_size,
    _build_layer_chunk_entries,
    _resolve_chunks_per_message,
)


def test_batch_entries_by_size_groups_multiple_layers() -> None:
    """Small layers should share a message until the byte budget is full."""
    state_dict = {
        "a": torch.zeros(2, dtype=torch.float32),
        "b": torch.zeros(2, dtype=torch.float32),
        "c": torch.zeros(3, dtype=torch.float32),
    }
    entries = _build_layer_chunk_entries(list(state_dict), state_dict, 64)

    batches = _batch_entries_by_size(
        entries,
        max_batch_bytes=16,
        max_chunks_per_message=0,
    )

    assert [[entry["layer_name"] for entry in batch] for batch in batches] == [
        ["a", "b"],
        ["c"],
    ]


def test_batch_entries_by_size_honors_explicit_chunk_cap() -> None:
    """The chunk cap is opt-in and can force one chunk per message."""
    state_dict = {
        "a": torch.zeros(2, dtype=torch.float32),
        "b": torch.zeros(2, dtype=torch.float32),
        "c": torch.zeros(2, dtype=torch.float32),
    }
    entries = _build_layer_chunk_entries(list(state_dict), state_dict, 64)

    batches = _batch_entries_by_size(
        entries,
        max_batch_bytes=64,
        max_chunks_per_message=1,
    )

    assert len(batches) == len(entries)
    assert all(len(batch) == 1 for batch in batches)


def test_deprecated_layers_per_message_is_ignored() -> None:
    """Old configs must not accidentally restore one-layer-per-message sends."""
    assert _resolve_chunks_per_message({"aggregation.layers-per-message": 1}) == 0


def test_negative_chunks_per_message_is_rejected() -> None:
    with pytest.raises(ValueError, match="chunks-per-message"):
        _resolve_chunks_per_message({"aggregation.chunks-per-message": -1})
