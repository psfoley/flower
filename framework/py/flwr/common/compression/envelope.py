# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Compressed Array envelope encoding."""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

COMPRESSION_ENVELOPE_VERSION = 1
_MAGIC = b"FLWR-COMP\0"
_HEADER_LEN = struct.Struct("!I")


@dataclass(frozen=True)
class CompressionEnvelope:
    """Flower-native compressed payload envelope."""

    pipeline_id: str
    pipeline_params: dict[str, Any]
    metadata: dict[str, Any]
    payload: bytes
    version: int = COMPRESSION_ENVELOPE_VERSION


def encode_envelope(envelope: CompressionEnvelope) -> bytes:
    """Encode a compression envelope to bytes."""
    header = {
        "version": envelope.version,
        "pipeline_id": envelope.pipeline_id,
        "pipeline_params": envelope.pipeline_params,
        "metadata": envelope.metadata,
    }
    header_bytes = json.dumps(header, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return (
        _MAGIC
        + _HEADER_LEN.pack(len(header_bytes))
        + header_bytes
        + envelope.payload
    )


def decode_envelope(data: bytes) -> CompressionEnvelope:
    """Decode a compression envelope from bytes."""
    if not data.startswith(_MAGIC):
        raise ValueError("Compressed Array payload has an invalid magic header")
    header_start = len(_MAGIC)
    header_len = _HEADER_LEN.unpack(data[header_start : header_start + 4])[0]
    payload_start = header_start + 4 + header_len
    header = json.loads(data[header_start + 4 : payload_start].decode("utf-8"))
    version = int(header["version"])
    if version != COMPRESSION_ENVELOPE_VERSION:
        raise ValueError(
            f"Unsupported compression envelope version {version}; "
            f"expected {COMPRESSION_ENVELOPE_VERSION}"
        )
    return CompressionEnvelope(
        pipeline_id=str(header["pipeline_id"]),
        pipeline_params=dict(header.get("pipeline_params", {})),
        metadata=dict(header.get("metadata", {})),
        payload=data[payload_start:],
        version=version,
    )
