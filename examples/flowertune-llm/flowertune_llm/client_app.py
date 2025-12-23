"""flowertune-llm: A Flower / FlowerTune app."""

import os
import warnings
import pickle

from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict, ConfigRecord
from flwr.clientapp import ClientApp
from flwr.common.config import unflatten_dict
from omegaconf import DictConfig
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from transformers import TrainingArguments
from trl import SFTTrainer

from flowertune_llm.dataset import (
    get_tokenizer_and_data_collator_and_propt_formatting,
    load_data,
    replace_keys,
)
from flowertune_llm.models import cosine_annealing, get_model

# Avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "true"
os.environ["RAY_DISABLE_DOCKER_CPU_WARNING"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)


# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    """Train the model on local data."""
    # Parse config
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    num_rounds = context.run_config["num-server-rounds"]
    cfg = DictConfig(replace_keys(unflatten_dict(context.run_config)))
    training_arguments = TrainingArguments(**cfg.train.training_arguments)

    # Let's get the client partition
    trainset = load_data(partition_id, num_partitions, cfg.dataset.name)
    (
        tokenizer,
        data_collator,
        formatting_prompts_func,
    ) = get_tokenizer_and_data_collator_and_propt_formatting(cfg.model.name)

    # Load the model and initialize it with the received weights
    model = get_model(cfg.model)
    #set_peft_model_state_dict(model, msg.content["arrays"].to_torch_state_dict())
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict(), strict=True)

    # Set learning rate for current round
    new_lr = cosine_annealing(
        msg.content["config"]["server-round"],
        num_rounds,
        cfg.train.learning_rate_max,
        cfg.train.learning_rate_min,
    )

    training_arguments.learning_rate = new_lr
    training_arguments.output_dir = msg.content["config"]["save_path"]

    # Uncomment this is sufficient resources exist for training
    #trainer = SFTTrainer(
    #    model=model,
    #    tokenizer=tokenizer,
    #    args=training_arguments,
    #    max_seq_length=cfg.train.seq_length,
    #    train_dataset=trainset,
    #    formatting_func=formatting_prompts_func,
    #    data_collator=data_collator,
    #)

    # Do local training
    #results = trainer.train()

    # Save model layers locally
    serialized_layer_paths = []
    model_dict = model.state_dict()
    os.makedirs("layers", exist_ok=True)
    for layer_name in model.state_dict():
        serialized_layer_path = f'layers/{msg.metadata.dst_node_id}_{layer_name}.pt'
        serialized_layer_paths.append(serialized_layer_path)
        with open(serialized_layer_path, 'wb') as file:
            pickle.dump({layer_name: model_dict[layer_name]}, file)

    metrics = {
        "train_loss": 0.0,
        "num-examples": len(trainset),
    }

    metric_record = MetricRecord(metrics)

    # A placeholder for arrays are sent. 
    # The actual layers will be sent in the train_comms function
    content = RecordDict({"arrays": ArrayRecord(), "metrics": metric_record})

    #Save layer paths so they can be deserialized for individual sending later
    context.state["serialized_layer_paths"] = ConfigRecord({"layer_paths": serialized_layer_paths})
    context.state["current_layer"] = MetricRecord({"idx":0})
    context.state["num_examples"] = MetricRecord({"num-examples": len(trainset)})

    return Message(content=content, reply_to=msg)

@app.train("layer_wise_communication")
def train_comms(msg: Message, context: Context):
    """Send the model layer by layer"""
    idx = context.state["current_layer"]["idx"]
    serialized_layer_paths = context.state["serialized_layer_paths"]["layer_paths"]
    send_complete = False
    if idx == (len(serialized_layer_paths) - 1):
        send_complete = True

    # Read model layer from disk
    serialized_layer_path = serialized_layer_paths[idx]
    with open(serialized_layer_path, 'rb') as file:
        model_dict = pickle.load(file)

    layer_name = list(model_dict.keys())[0]
    array = ArrayRecord({layer_name: model_dict[layer_name]})
    num_examples = context.state["num_examples"]
    content = RecordDict({"array": array, "status": ConfigRecord({"send_complete":send_complete}), "num_examples": num_examples})
    print(f'Sending layer {layer_name}...')
    context.state["current_layer"]["idx"] = idx + 1
    return Message(content=content, reply_to=msg)
    
