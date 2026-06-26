# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
# ==============================================================================
"""Tests compressed Array stype decoding."""

from __future__ import annotations

import numpy as np

from flwr.common.compression import TurboQuantMSEPipeline, compress_array
from flwr.common.record import Array


def test_array_numpy_decodes_compressed_pipeline_stype() -> None:
    """Array.numpy should decode compressed pipeline stype."""
    rng = np.random.default_rng(9)
    original = rng.normal(size=(64,)).astype(np.float32)
    compressed, _ = compress_array(
        Array(original), TurboQuantMSEPipeline(n_bits=4, block_size=64)
    )

    decoded = compressed.numpy()

    assert decoded.shape == original.shape
    assert decoded.dtype == original.dtype
