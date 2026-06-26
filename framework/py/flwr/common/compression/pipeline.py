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
"""OpenFL-style compression pipeline interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from typing import Any

import numpy as np

from flwr.common.typing import NDArray


class Transformer(ABC):
    """Base transformer with OpenFL-compatible forward/backward methods."""

    @abstractmethod
    def forward(self, array: NDArray) -> tuple[bytes, dict[str, Any]]:
        """Compress or transform an array."""

    @abstractmethod
    def backward(self, payload: bytes, metadata: dict[str, Any]) -> NDArray:
        """Decompress or reverse a transformation."""


@dataclass
class TransformationPipeline:
    """Sequence of compression transformers."""

    pipeline_id: str
    transformers: list[Transformer]
    params: dict[str, Any]

    def forward(self, array: NDArray) -> tuple[bytes, dict[str, Any]]:
        """Compress an array using this pipeline."""
        if len(self.transformers) != 1:
            raise NotImplementedError("Only single-transformer pipelines are supported")
        payload, metadata = self.transformers[0].forward(array)
        return payload, {"transformers": [metadata]}

    def backward(self, payload: bytes, metadata: dict[str, Any]) -> NDArray:
        """Decompress an array using this pipeline."""
        if len(self.transformers) != 1:
            raise NotImplementedError("Only single-transformer pipelines are supported")
        transformer_metadata = metadata["transformers"][0]
        return self.transformers[0].backward(payload, transformer_metadata)


class NoCompressionTransformer(Transformer):
    """Lossless NumPy serialization transformer."""

    def forward(self, array: NDArray) -> tuple[bytes, dict[str, Any]]:
        """Serialize without compression."""
        buffer = BytesIO()
        np.save(buffer, array, allow_pickle=False)
        return buffer.getvalue(), {
            "dtype": str(array.dtype),
            "shape": tuple(array.shape),
        }

    def backward(self, payload: bytes, metadata: dict[str, Any]) -> NDArray:
        """Deserialize without compression."""
        del metadata
        return np.load(BytesIO(payload), allow_pickle=False)


class NoCompressionPipeline(TransformationPipeline):
    """No-op compression pipeline."""

    def __init__(self) -> None:
        super().__init__("none", [NoCompressionTransformer()], {})
