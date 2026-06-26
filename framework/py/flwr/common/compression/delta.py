# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
# ==============================================================================
"""Delta extraction utilities for ArrayRecords."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from flwr.common.record import Array, ArrayRecord
from flwr.common.typing import NDArray


@dataclass
class DeltaState:
    """Stateful helper for extracting and applying ArrayRecord deltas."""

    reference: dict[str, NDArray] = field(default_factory=dict)

    @classmethod
    def from_arrayrecord(cls, record: ArrayRecord) -> DeltaState:
        """Create state from an ArrayRecord."""
        return cls({key: value.numpy().copy() for key, value in record.items()})

    def extract_delta(self, record: ArrayRecord) -> ArrayRecord:
        """Return ``record - reference`` and update no state."""
        delta = ArrayRecord()
        for key, array in record.items():
            current = array.numpy()
            base = self.reference.get(key)
            if base is None:
                base = np.zeros_like(current)
            delta[key] = Array(current - base)
        return delta

    def apply_delta(self, delta: ArrayRecord) -> ArrayRecord:
        """Return ``reference + delta`` and update the reference."""
        updated = ArrayRecord()
        for key, array in delta.items():
            delta_array = array.numpy()
            base = self.reference.get(key)
            if base is None:
                base = np.zeros_like(delta_array)
            current = base + delta_array
            self.reference[key] = current.copy()
            updated[key] = Array(current)
        return updated

    def update(self, record: ArrayRecord) -> None:
        """Replace reference state with an ArrayRecord."""
        self.reference = {key: value.numpy().copy() for key, value in record.items()}


# Backward-compatible descriptive alias for API users.
ArrayRecordDelta = ArrayRecord
