# Copyright 2025 Flower Labs GmbH. All Rights Reserved.
# ==============================================================================
"""Compression mod for Message API ClientApps."""

from __future__ import annotations

from logging import INFO
from typing import Any

from flwr.clientapp.typing import ClientAppCallable
from flwr.common import ConfigRecord, Context, Message, log
from flwr.common.compression import compress_recorddict_arrayrecords, create_pipeline


class CompressionMod:
    """Compress outgoing client reply ArrayRecords.

    Parameters can be provided directly or through the incoming Message
    ConfigRecord using these keys:

    - ``compression-pipeline``: ``"turboquant_mse"`` or ``"none"``
    - ``compression-n-bits``: bit-width for TurboQuant MSE
    - ``compression-block-size``: block size for normalization metadata
    """

    def __init__(self, pipeline_name: str = "turboquant_mse", **params: Any) -> None:
        self.pipeline_name = pipeline_name
        self.params = params

    @staticmethod
    def _config(msg: Message) -> ConfigRecord | None:
        for record in msg.content.config_records.values():
            return record
        return None

    def _resolve(self, msg: Message):  # type: ignore[no-untyped-def]
        config = self._config(msg)
        name = self.pipeline_name
        params = dict(self.params)
        if config is not None:
            name = str(config.get("compression-pipeline", name))
            if "compression-n-bits" in config:
                params["n_bits"] = int(config["compression-n-bits"])
            if "compression-block-size" in config:
                params["block_size"] = int(config["compression-block-size"])
        return create_pipeline(name, **params)

    def __call__(
        self, msg: Message, ctxt: Context, call_next: ClientAppCallable
    ) -> Message:
        """Compress reply messages produced by the ClientApp."""
        out_msg = call_next(msg, ctxt)
        if not out_msg.has_content() or not out_msg.content.array_records:
            return out_msg
        pipeline = self._resolve(msg)
        raw, compressed = compress_recorddict_arrayrecords(out_msg.content, pipeline)
        if raw:
            log(
                INFO,
                "CompressionMod[%s]: compressed client reply %d -> %d bytes "
                "(ratio %.3f)",
                pipeline.pipeline_id,
                raw,
                compressed,
                raw / max(1, compressed),
            )
        return out_msg
