"""Tests for client-side layer artifact lifecycle."""

import os
import sys
import types

import torch
from flwr.app import ConfigRecord, Context, Message, RecordDict

omegaconf_stub = types.ModuleType("omegaconf")


class DictConfig(dict):
    """Minimal DictConfig stub for importing the client module."""

    def __getattr__(self, name: str):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


omegaconf_stub.DictConfig = DictConfig
sys.modules.setdefault("omegaconf", omegaconf_stub)

transformers_stub = types.ModuleType("transformers")
transformers_stub.AutoModelForCausalLM = object()
sys.modules.setdefault("transformers", transformers_stub)

from flowertune_llm.client_app import (  # noqa: E402
    STATE_LAYER_IDX,
    STATE_LAYER_NAMES,
    STATE_LAYER_PATHS,
    STATE_NUM_EXAMPLES,
    _persist_layer_files,
    train_comms,
)


def test_train_comms_cleans_layer_files_after_final_send(tmp_path) -> None:
    """Final layer-wise upload should not leave PT copies on disk."""
    context = Context(
        run_id=123,
        node_id=456,
        node_config={},
        state=RecordDict(),
        run_config={"aggregation.layer-write-dir": str(tmp_path)},
    )
    _persist_layer_files(
        context,
        {
            "layer.a": torch.tensor([1.0, 2.0]),
            "layer.b": torch.tensor([3.0, 4.0]),
        },
        ["layer.a", "layer.b"],
    )
    layer_paths = list(context.state[STATE_LAYER_PATHS]["paths"])
    assert all(os.path.exists(path) for path in layer_paths)

    message = Message(
        content=RecordDict({
            "config": ConfigRecord({
                "upload_layer_idxs": [0, 1],
                "upload_layer_names": ["layer.a", "layer.b"],
                "upload_chunk_starts": [0, 0],
                "upload_chunk_ends": [0, 0],
                "upload_is_last_chunk": [True, True],
            }),
        }),
        dst_node_id=1,
        message_type="train.layer_wise_communication",
    )

    reply = train_comms(message, context)

    assert reply.content["config"]["send_complete"]
    assert set(reply.content["arrays"].keys()) == {
        "layer.a::chunk_0_0",
        "layer.b::chunk_0_0",
    }
    assert all(not os.path.exists(path) for path in layer_paths)
    assert STATE_LAYER_NAMES not in context.state
    assert STATE_LAYER_PATHS not in context.state
    assert STATE_LAYER_IDX not in context.state
    assert STATE_NUM_EXAMPLES not in context.state
