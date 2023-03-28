import subprocess
import sys
import os
import torch
from datasets import load_dataset
import argparse
from azureml.core.run import Run
from azureml.core.dataset import Dataset
from datasets import load_dataset, load_from_disk
from azureml.core.model import Model
from model import SentimentXLMRModel, TrainingLoopStructure
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from util import get_ml_client, get_latest_data_version


import azure.ai.ml._artifacts._artifact_utilities as artifact_utils

DATASET_PATH = "./dataset"

def main():
    pass

def download_dataset(
    ml_client,
    name: str,
    destination: str,
    version: str = None,
):
    if version is None:
        version = get_latest_data_version(name)

    data_info = ml_client.data.get(name=name, version=str(version))

    artifact_utils.download_artifact_from_aml_uri(
        uri=data_info.path,
        destination=destination,
        datastore_operation=ml_client.datastores
    )

def create_traced_model(tokenizer, model):

    dd = tokenizer(
        ["This is a test...", "Detta är ett test..."],
        return_tensors="pt",
        padding=True
    )

    model.eval()

    jit_model = torch.jit.trace(
        model,
        [
            dd["input_ids"].to(model._device),
            dd["attention_mask"].to(model._device)
        ]
    )

    return jit_model

if __name__ == "__main__":
    
    if not os.path.exists('artifacts'):
        os.makedirs('artifacts')

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--dataset',
        type=str,
        default=None,
        help="Id of the dataset that we want to train/val on"
    )
    parser.add_argument(
        '--download',
        action='store_true',
        default=False,
        help="are we going to download dataset"
    )
    parser.add_argument(
        '--dataset_version',
        type=str,
        default=None,
        help="version of the dataset"
    )
    parser.add_argument(
        '--lock_embedding',
        action='store_true',
        default=False,
        help="do we want to lock the embedding layer during training"
    )
    parser.add_argument(
        '--lock_first_n_layers', type=int, default=None,
        help="do we want to lock any of the n first layers during training"
    )
    parser.add_argument(
        '--lr', type=float, default=1e-5, help="learning rate"
    )
    parser.add_argument(
        '--batch_size', type=int, default=2, help="batch size"
    )
    parser.add_argument(
        '--iterations', type=int, default=100000, help="Number of training iterations"
    )

    args = parser.parse_args()

    ml_client = get_ml_client()

    dataset_path = f"artifacts/{args.dataset}"
    
    print("setting up dataset")

    print("cuda is avaiable", torch.cuda.is_available())

    if args.download:
        download_dataset(
            ml_client=ml_client,
            name=args.dataset,
            destination=dataset_path,
            version=args.dataset_version,
        )

    dataset_hg = load_from_disk(dataset_path)

    print("setting up model")
    model = SentimentXLMRModel(
        lock_embedding=args.lock_embedding,
        lock_first_n_layers=args.lock_first_n_layers,
    )

    tokenizer = model._tokenizer

    model.configure_optimizer(
        lr=args.lr,
    )

    dataloader_train = torch.utils.data.DataLoader(
        dataset_hg["train"],
        batch_size=args.batch_size,
        shuffle=True
    )
    dataloader_val = torch.utils.data.DataLoader(
        dataset_hg["validation"],
        batch_size=args.batch_size,
        shuffle=True
    )

    model.configure_training(
        experiment_name="run",
        train_dataloader=dataloader_train,
        val_dataloader=dataloader_val,
        train_freq=1,
        val_freq=10,
    )

    print("Training")
    train_config = TrainingLoopStructure(
        iterations=args.iterations,
        save_freq=int(args.iterations) / 10,
    )

    model.training_loop(train_config)

    print("Register model")
    file_model = Model(
        path=f"{model._model_folder}/latest.ckp",
        type=AssetTypes.CUSTOM_MODEL,
        name="xlmr_sentiment",
        description="XLMR trained on twitter sentiment dataset."
    )
    ml_client.models.create_or_update(file_model)

    print("Register the tokenizer")
    tokenizer.save_pretrained("artifacts/tokenizer")
    file_model = Model(
        path="artifacts/tokenizer",
        type=AssetTypes.CUSTOM_MODEL,
        name="xlmr_sentiment_tokenizer",
        description="XLMR tokenizer"
    )
    ml_client.models.create_or_update(file_model)
    
    print("Register a traced model")
    traced_model = create_traced_model(tokenizer, model)
    traced_model.save("artifacts/traced.pt")
    file_model = Model(
        path="artifacts/traced.pt",
        type=AssetTypes.CUSTOM_MODEL,
        name="xlmr_sentiment_traced",
        description="XLMR trained on twitter sentiment dataset. traced"
    )
    ml_client.models.create_or_update(file_model)

    