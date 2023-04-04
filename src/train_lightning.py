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
from model import SentimentXLMRModelLight
from azure.ai.ml.entities import Model
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import Data
from azure.ai.ml.constants import AssetTypes
from util import get_ml_client, get_latest_data_version, download_dataset, create_traced_model
import torchmetrics
from lightning.pytorch.cli import LightningCLI
import lightning.pytorch as pl
from lightning.pytorch.demos.boring_classes import DemoModel, BoringDataModule
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
import deepspeed

DATASET_PATH = "./dataset"

def main():
    pass

class SentimentDataModule(pl.LightningDataModule):
    
    DATASET_PATH = f"artifacts/twitter-sentiment"
    
    def __init__(
        self,
        dataset_name: str="twitter-sentiment",
        batch_size_train: int = 24,
        batch_size_val: int = 40,
        workers: int = 2,
        download: bool = False,
        subset_train: int = None,
        subset_validation: int = None,
    ):
        
        super().__init__()
        
        if download:
            download_dataset(
                ml_client=get_ml_client(),
                name=dataset_name,
                destination=SentimentDataModule.DATASET_PATH,
                version=1,
            )
        
        self.dataset_name = dataset_name
        self.workers = workers
        self.subset_train = subset_train
        self.subset_validation = subset_validation
        self.batch_size_train = batch_size_train
        self.batch_size_val = batch_size_val

    def setup(self, stage: str):
        
        self._dataset_hg = load_from_disk(SentimentDataModule.DATASET_PATH)

    def train_dataloader(self):
        
        dataset = self._dataset_hg["train"]
        
        if self.subset_train is not None:
            dataset = dataset.select(range(self.subset_train))
        
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size_train,
            shuffle=True,
            num_workers=self.workers,
        )

    def val_dataloader(self):
        
        dataset = self._dataset_hg["validation"]
        
        if self.subset_validation is not None:
            dataset = dataset.select(range(self.subset_validation))

        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size_val,
            shuffle=False,
            num_workers=self.workers,
        )

    def test_dataloader(self):
        None

    def predict_dataloader(self):
        None

    def teardown(self, stage: str):
        # Used to clean-up when the run is finished
        ...

def cli_main():

    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=10000,
    )
    
    checkpoint_callback_val = ModelCheckpoint(
        save_top_k=5,
        monitor="loss_val",
        mode="min",
        filename='{epoch}-{loss_val:.4f}',
    )

    cli = LightningCLI(
        save_config_overwrite=True,
        model_class=SentimentXLMRModelLight,
        trainer_class=pl.Trainer,
        datamodule_class=SentimentDataModule,
        run=False,
        trainer_defaults={
            "callbacks": [
                checkpoint_callback,
                checkpoint_callback_val
            ],
            "logger": TensorBoardLogger("artifacts", name="sentiment"),
            # "strategy": pl.strategies.DeepSpeedStrategy(logging_batch_size_per_gpu=13),
        }
    )
    
    return cli, checkpoint_callback_val

if __name__ == "__main__":

    cli, checkpoint_callback_val = cli_main()

    print(cli.model._model._encoder.print_trainable_parameters())

    cli.trainer.fit(cli.model, cli.datamodule)

    model = cli.model

    best_model_path = checkpoint_callback_val.best_model_path

    tokenizer = model._model._tokenizer

    ml_client = get_ml_client()
    
    print("Register model")
    file_model = Model(
        path=best_model_path,
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
    traced_model = create_traced_model(tokenizer, model._model) # Do it on the pt model
    traced_model.save("artifacts/traced.pt")
    file_model = Model(
        path="artifacts/traced.pt",
        type=AssetTypes.CUSTOM_MODEL,
        name="xlmr_sentiment_traced",
        description="XLMR trained on twitter sentiment dataset. traced"
    )
    ml_client.models.create_or_update(file_model)
    