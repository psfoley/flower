# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
# ==============================================================================
"""ServerApp strategy compression wrapper."""

from __future__ import annotations

from collections.abc import Iterable
from logging import INFO
from typing import Any

from flwr.common import ArrayRecord, ConfigRecord, Message, MetricRecord, log
from flwr.common.compression import compress_recorddict_arrayrecords, create_pipeline
from flwr.server import Grid

from .strategy import Strategy


class CompressionStrategy(Strategy):
    """Wrap a message-based Strategy and compress outbound ArrayRecords."""

    def __init__(
        self, strategy: Strategy, pipeline_name: str = "turboquant_mse", **params: Any
    ) -> None:
        self.strategy = strategy
        self.pipeline_name = pipeline_name
        self.params = params

    def _compress_messages(self, messages: Iterable[Message]) -> list[Message]:
        pipeline = create_pipeline(self.pipeline_name, **self.params)
        out = list(messages)
        raw_total = 0
        compressed_total = 0
        for msg in out:
            if msg.has_content() and msg.content.array_records:
                raw, compressed = compress_recorddict_arrayrecords(
                    msg.content, pipeline
                )
                raw_total += raw
                compressed_total += compressed
        if raw_total:
            log(
                INFO,
                "CompressionStrategy[%s]: compressed server messages %d -> %d bytes "
                "(ratio %.3f)",
                pipeline.pipeline_id,
                raw_total,
                compressed_total,
                raw_total / max(1, compressed_total),
            )
        return out

    def configure_train(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure compressed training messages."""
        return self._compress_messages(
            self.strategy.configure_train(server_round, arrays, config, grid)
        )

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[ArrayRecord | None, MetricRecord | None]:
        """Delegate train aggregation."""
        return self.strategy.aggregate_train(server_round, replies)

    def configure_evaluate(
        self, server_round: int, arrays: ArrayRecord, config: ConfigRecord, grid: Grid
    ) -> Iterable[Message]:
        """Configure compressed evaluation messages."""
        return self._compress_messages(
            self.strategy.configure_evaluate(server_round, arrays, config, grid)
        )

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> MetricRecord | None:
        """Delegate evaluation aggregation."""
        return self.strategy.aggregate_evaluate(server_round, replies)

    def summary(self) -> None:
        """Log wrapped strategy summary."""
        self.strategy.summary()
