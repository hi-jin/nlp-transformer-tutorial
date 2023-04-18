from typing import TypedDict

import pandas as pd

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import torchmetrics

from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset


class TrainConfig(TypedDict):
    dataset_path: str
    dataset_subset: str
    batch_size: int
    num_workers: int


class ModelConfig(TypedDict):
    pretrained_model_name: str
    num_classes: int
    lr: float


class TopicClassifier(pl.LightningModule):
    def __init__(
        self,
        config: ModelConfig,
        pretrained_model,
    ):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.pretrained_model = pretrained_model
        self.to_output = torch.nn.Linear(768, self.config["num_classes"])
        
        self.criterion = torch.nn.CrossEntropyLoss()
        
        self.val_pred = []
        self.val_label = []
        self.test_pred = []
        self.test_label = []
        
        self.f1 = torchmetrics.F1Score(task="multiclass", num_classes=self.config["num_classes"])
    
    def forward(
        self,
        input_ids,
        attention_mask,
        label = None,
    ):
        hidden_state = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )["last_hidden_state"][:, 0]

        logit = self.to_output(hidden_state)
        
        return logit

    def training_step(self, batch, batch_idx):
        logit = self(**batch)
        loss = self.criterion(logit, batch["label"])
        
        self.log("train_loss", loss, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logit = self(**batch)
        loss = self.criterion(logit, batch["label"])
        
        self.log("val_loss", loss, sync_dist=True)
        
        pred = logit.argmax(dim=-1)
        self.val_pred.extend(pred)
        self.val_label.extend(batch["label"])

    def on_validation_epoch_end(self) -> None:
        pred = torch.tensor(self.val_pred)
        label = torch.tensor(self.val_label)

        acc = torch.mean((pred == label).float())
        self.log("val_acc", acc, sync_dist=True)
        self.log("val_f1", self.f1(pred, label), sync_dist=True)
        
        self.val_pred.clear()
        self.val_label.clear()
    
    def test_step(self, batch, batch_idx):
        logit = self(**batch)
        loss = self.criterion(logit, batch["label"])
        
        self.log("test_loss", loss, sync_dist=True)
        
        pred = logit.argmax(dim=-1)
        self.test_pred.extend(pred)
        self.test_label.extend(batch["label"])
    
    def on_test_epoch_end(self) -> None:
        pred = torch.tensor(self.test_pred)
        label = torch.tensor(self.test_label)

        acc = torch.mean((pred == label).float())
        self.log("test_acc", acc, sync_dist=True)
        self.log("test_f1", self.f1(pred, label), sync_dist=True)
        
        pd.DataFrame({
            "pred": pred.tolist(),
            "label": label.tolist(),
        }).to_csv("test_result.csv")
        
        self.test_pred.clear()
        self.test_label.clear()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.config["lr"])


class Dataset(torch.utils.data.Dataset): # type: ignore
    def __init__(
        self,
        title,
        label,
    ):
        super().__init__()
        self.title = title
        self.label = label
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx: int):
        title = self.title[idx]
        label = self.label[idx]

        return title, label


class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        config: TrainConfig,
        tokenizer,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer

    def prepare_data(self):
        load_dataset(self.config["dataset_path"], self.config["dataset_subset"])
     
    def setup(self, stage = None):
        data = load_dataset(self.config["dataset_path"], self.config["dataset_subset"])
        train_data = data["train"] # type: ignore
        test_data = data["validation"] # type: ignore
        
        self.train_dataset, self.val_dataset = torch.utils.data.random_split( # type: ignore
            Dataset(train_data["title"], train_data["label"]), # type: ignore
            [0.8, 0.2],
        )
        self.test_dataset = Dataset(test_data["title"], test_data["label"]) # type: ignore

    def collate_fn(self, batch):
        title, label = zip(*batch)
        inputs = self.tokenizer(title, padding=True, return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "label": torch.tensor(label),
        }
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader( # type: ignore
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader( # type: ignore
            self.val_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return torch.utils.data.DataLoader( # type: ignore
            self.test_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=False,
        )


if __name__ == "__main__":
    train_config = TrainConfig(
        dataset_path="klue",
        dataset_subset="ynat",
        batch_size=64,
        num_workers=10,
    )
    
    model_config = ModelConfig(
        pretrained_model_name="bert-base-multilingual-cased",
        num_classes=7,
        lr=0.0001,
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_config["pretrained_model_name"])
    pretrained_model = AutoModel.from_pretrained(model_config["pretrained_model_name"])
    
    datamodule = Datamodule(train_config, tokenizer)
    classifier = TopicClassifier(model_config, pretrained_model)
    
    logger = WandbLogger(project="3-fine-tuning-bert")
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min"),
        ],
        accelerator="gpu",
        devices=2,
        strategy="ddp_find_unused_parameters_true",
    )
    
    # trainer.fit(classifier, datamodule)
    trainer.test(classifier, datamodule)
