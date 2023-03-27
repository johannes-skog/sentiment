from abc import abstractmethod
import os
from typing import List, Any,Dict
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from dataclasses import dataclass, field
from torch.utils.tensorboard import SummaryWriter


@dataclass
class TrainingStructure():

    TRAIN_DATASET: str = "train"
    VAL_DATASET: str = "val"

    optimizer: Any = None

    dataloaders: Dict = field(default_factory=dict)
    frequency: Dict = field(default_factory=dict)

    iters: Dict = field(default_factory=dict)

    writer: Any = None

    global_iteration: int = 0

@dataclass
class TrainingLoopStructure():

    current_iteration = None
    iterations: int = None
    save_freq: int = None

class NLPmodel(torch.nn.Module):

    def __init__(
        self,
        encoder: torch.nn.Module,
        tokenizer,
        model_folder: str = "models/ckp",
        device: str = None,
    ):

        super().__init__()

        self._encoder = encoder
        self._tokenizer = tokenizer
        self._model_folder = model_folder
        self._training_structure = TrainingStructure()

        self._device = (
            "cuda" if torch.cuda.is_available() else "cpu"
        ) if device is None else device

    def configure_optimizer(self, optimizer = torch.optim.Adam, **kwargs):

        # Initialize optimizer
        self._training_structure.optimizer = optimizer(self.parameters(), **kwargs)

    def configure_training(
        self,
        experiment_name: str,
        train_dataloader: torch.utils.data.DataLoader,
        val_dataloader: torch.utils.data.DataLoader,
        train_freq: int = 1,
        val_freq: int = 100,
    ):

        self._training_structure.dataloaders[
            TrainingStructure.TRAIN_DATASET
        ] = train_dataloader

        self._training_structure.iters[
            TrainingStructure.TRAIN_DATASET
        ] = iter(train_dataloader)

        self._training_structure.frequency[
            TrainingStructure.TRAIN_DATASET
        ] = train_freq

        self._training_structure.iters[
            TrainingStructure.VAL_DATASET
        ] = iter(val_dataloader)

        self._training_structure.dataloaders[
            TrainingStructure.VAL_DATASET
        ] = val_dataloader

        self._training_structure.frequency[
            TrainingStructure.VAL_DATASET
        ] = val_freq

        self._training_structure.writer = SummaryWriter(f"logs/{experiment_name}")

    def _get_batch(self, type: str):
        
        try: 
            batch = next(self._training_structure.iters[type])
        except StopIteration:
            self._training_structure.iters[type] = iter(
                self._training_structure.dataloaders[type]
            )
            batch = next(self._training_structure.iters[type])

        return batch

    @abstractmethod
    def loss(self):
        pass

    @abstractmethod
    def train_step(self,**kwargs):
        pass

    @abstractmethod
    def val_step(self, **kwargs):
        pass

    def log_scalar(self, tag: str, v: float):

        self._training_structure.writer.add_scalar(
            tag, v, self._training_structure.global_iteration,
        )

    def training_loop(self, conf: TrainingLoopStructure):

        self.to(self._device)

        assert conf.iterations is not None

        for i in range(conf.iterations):

            print(i)
            
            if i % self._training_structure.frequency[TrainingStructure.VAL_DATASET] == 0:
                with torch.no_grad():
                    self.val_step(self._get_batch(TrainingStructure.VAL_DATASET))

            if  i % self._training_structure.frequency[TrainingStructure.TRAIN_DATASET] == 0:
                self.train_step(self._get_batch(TrainingStructure.TRAIN_DATASET))

            if conf.save_freq is not None and i % conf.save_freq == 0:
                self.save("latest.ckp", meta={
                        "iterations": i,
                        "global_iteration": self._training_structure.global_iteration,
                        "conf": conf.__repr__(),
                    }
                )

            conf.current_iteration = i

            self._training_structure.global_iteration +=1 

    def save(self, name: str, meta: Dict = None):

        if not os.path.exists(self._model_folder):
            os.makedirs(self._model_folder)

        path = os.path.join(self._model_folder, name)

        torch.save({
            'model_state_dict': self.state_dict(),
            "meta_data": meta
        }, path)

        return path

    def load(self, name: str):

        path = os.path.join(self._model_folder, name)

        print(self._device)
        checkpoint = torch.load(path, map_location=torch.device(self._device))
        self.load_state_dict(checkpoint['model_state_dict'])
        print(checkpoint['meta_data'])

        self._training_structure.global_iteration = checkpoint['meta_data']["global_iteration"]

    def forward(self, **kwargs):
        
        return self._encoder.forward(**kwargs)

    @staticmethod
    def _lock_layer(layer: torch.nn.Module):
        for p in layer.parameters():
            p.requires_grad = False

    @staticmethod
    def _lock_layers(layers: List[torch.nn.Module]):
        for layer in layers:
            NLPmodel._lock_layer(layer)   


class SentimentXLMRModel(NLPmodel):
    
    def __init__(
        self,
        lock_embedding: bool = False,
        lock_first_n_layers: int = None,
        model_folder: str = "models/ckp/SentimentXLMRModel",
        device: str = None,
    ):

        super().__init__(
            encoder=AutoModelForMaskedLM.from_pretrained("xlm-roberta-base").roberta,
            tokenizer=AutoTokenizer.from_pretrained('xlm-roberta-base'),
            model_folder=model_folder,
            device=device,
        )
        
        hidden_dim = 768
        output_dim = 1

        self._final_embedder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
            ),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=hidden_dim,
            ),
            torch.nn.LayerNorm(hidden_dim),
            torch.nn.GELU(),
            torch.nn.Linear(
                in_features=hidden_dim,
                out_features=output_dim,
            ),
        )

        if lock_embedding:
            self._lock_layer(self._encoder.embeddings)
            
        if lock_first_n_layers is not None:
            self._lock_layers(
                self._encoder.encoder.layer[0:lock_first_n_layers]
            )

        self._loss_func = torch.nn.BCEWithLogitsLoss()

    def train_step(self, batch):

        self.train()
        self.zero_grad()

        y = self.forward(
            input_ids=batch["input_ids"].to(self._device),
            attention_mask=batch["attention_mask"].to(self._device)
        )

        l = self.loss(y=y, label=batch["label"].to(self._device))

        l.backward()

        self._training_structure.optimizer.step()
        
        self.log_scalar("Loss/train", l.item())
   
    def val_step(self, batch):

        self.eval()

        y = self.forward(
            input_ids=batch["input_ids"].to(self._device),
            attention_mask=batch["attention_mask"].to(self._device)
        )

        l = self.loss(y=y, label=batch["label"].to(self._device))

        self.log_scalar("Loss/val", l.item())
        
    def loss(self, y: torch.Tensor, label: torch.Tensor):

        return self._loss_func(y, label.view(y.shape).float())
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor = None):

        output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # pool the seq, take the representation for the first token
        hidden_state = output.last_hidden_state[:, 0, :]

        y = self._final_embedder(hidden_state)

        return y