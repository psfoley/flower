# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
# ==============================================================================
"""Tests for compression pipelines."""

from __future__ import annotations

import numpy as np

from flwr.common.compression import (
    DeltaState,
    TurboQuantMSEPipeline,
    compress_arrayrecord,
    create_pipeline,
)
from flwr.common.constant import SType
from flwr.common.record import Array, ArrayRecord


def test_turboquant_mse_roundtrip_shape_dtype() -> None:
    """TurboQuant should preserve shape and dtype on decode."""
    rng = np.random.default_rng(7)
    array = rng.normal(size=(256, 32)).astype(np.float32)
    pipeline = TurboQuantMSEPipeline(n_bits=4, block_size=256)

    payload, metadata = pipeline.forward(array)
    decoded = pipeline.backward(payload, metadata)

    assert decoded.shape == array.shape
    assert decoded.dtype == array.dtype
    assert np.mean((decoded - array) ** 2) < np.mean(array**2)


def test_arrayrecord_compression_decodes_via_array_numpy() -> None:
    """Compressed Array stype should decode through Array.numpy."""
    rng = np.random.default_rng(8)
    array = rng.normal(size=(128,)).astype(np.float32)
    record = ArrayRecord({"x": Array(array)})
    pipeline = TurboQuantMSEPipeline(n_bits=4, block_size=128)

    compressed, stats = compress_arrayrecord(record, pipeline)
    decoded = compressed["x"].numpy()

    assert compressed["x"].stype == SType.COMPRESSED_PIPELINE
    assert stats.raw_bytes > stats.compressed_bytes
    assert decoded.shape == array.shape
    assert decoded.dtype == array.dtype


def test_turboquant_mse_cuda_flag_roundtrip() -> None:
    """CUDA flag should be optional and preserve CPU-compatible payloads."""
    rng = np.random.default_rng(9)
    array = rng.normal(size=(128,)).astype(np.float32)
    pipeline = create_pipeline(
        "turboquant_mse", n_bits=3, block_size=32, use_cuda=True
    )

    payload, metadata = pipeline.forward(array)
    decoded = pipeline.backward(payload, metadata)

    assert metadata["transformers"][0]["cuda_requested"] is True
    assert decoded.shape == array.shape
    assert decoded.dtype == array.dtype


def test_delta_state_extract_and_apply() -> None:
    """DeltaState should compute and apply ArrayRecord deltas."""
    base = ArrayRecord({"w": Array(np.array([1.0, 2.0], dtype=np.float32))})
    updated = ArrayRecord({"w": Array(np.array([1.5, 1.75], dtype=np.float32))})
    state = DeltaState.from_arrayrecord(base)

    delta = state.extract_delta(updated)
    restored = state.apply_delta(delta)

    np.testing.assert_allclose(
        delta["w"].numpy(), np.array([0.5, -0.25], dtype=np.float32)
    )
    np.testing.assert_allclose(restored["w"].numpy(), updated["w"].numpy())
