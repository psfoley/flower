# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
# ==============================================================================
"""Helpers for compressing Message API ArrayRecords."""

from __future__ import annotations

from dataclasses import dataclass

from flwr.common.constant import SType
from flwr.common.record import Array, ArrayRecord, RecordDict

from .envelope import CompressionEnvelope, encode_envelope
from .pipeline import TransformationPipeline


@dataclass(frozen=True)
class CompressionStats:
    """Compression byte counters."""

    raw_bytes: int
    compressed_bytes: int
    arrays: int

    @property
    def ratio(self) -> float:
        """Return raw/compressed ratio."""
        return self.raw_bytes / max(1, self.compressed_bytes)


def compress_array(
    array: Array, pipeline: TransformationPipeline
) -> tuple[Array, CompressionStats]:
    """Compress one Array into a compressed Array envelope."""
    if array.stype == SType.COMPRESSED_PIPELINE:
        return array, CompressionStats(len(array.data), len(array.data), 0)
    ndarray = array.numpy()
    payload, metadata = pipeline.forward(ndarray)
    envelope = CompressionEnvelope(
        pipeline_id=pipeline.pipeline_id,
        pipeline_params=pipeline.params,
        metadata=metadata,
        payload=payload,
    )
    encoded = encode_envelope(envelope)
    compressed = Array(
        dtype=array.dtype,
        shape=tuple(array.shape),
        stype=SType.COMPRESSED_PIPELINE,
        data=encoded,
    )
    return compressed, CompressionStats(len(array.data), len(encoded), 1)


def compress_arrayrecord(
    record: ArrayRecord, pipeline: TransformationPipeline
) -> tuple[ArrayRecord, CompressionStats]:
    """Compress all Arrays in an ArrayRecord."""
    out = ArrayRecord()
    raw = 0
    compressed = 0
    arrays = 0
    for key, array in record.items():
        out_array, stats = compress_array(array, pipeline)
        out[key] = out_array
        raw += stats.raw_bytes
        compressed += stats.compressed_bytes
        arrays += stats.arrays
    return out, CompressionStats(raw, compressed, arrays)


def compress_recorddict_arrayrecords(
    records: RecordDict, pipeline: TransformationPipeline
) -> tuple[int, int]:
    """Compress ArrayRecords in-place inside a RecordDict."""
    raw = 0
    compressed = 0
    for key, record in list(records.array_records.items()):
        compressed_record, stats = compress_arrayrecord(record, pipeline)
        records[key] = compressed_record
        raw += stats.raw_bytes
        compressed += stats.compressed_bytes
    return raw, compressed
