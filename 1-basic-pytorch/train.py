from typing import TypedDict

import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torchvision.transforms import ToTensor
from datasets import load_dataset


class ModelConfig(TypedDict):
    lr: float


class TrainConfig(TypedDict):
    hg_dataset_name: str
    batch_size: int
    num_workers: int
    early_stopping_patience: int


class DigitDataset(Dataset):
    def __init__(self, image, label):
        self.image = image
        self.label = label
        self.transform = ToTensor()
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx: int):
        image = self.transform(self.image[idx])
        label = self.label[idx]
        
        return (
            image, # [1, 28, 28]
            label # scalar
        )


class DigitDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        config: TrainConfig,
    ):
        super().__init__()
        self.config = config
    
    def prepare_data(self):
        load_dataset(self.config["hg_dataset_name"])
    
    def setup(self, stage=None):
        data = load_dataset(self.config["hg_dataset_name"])
        train_data = data["train"] # type: ignore
        test_data = data["test"] # type: ignore
        
        self.train_dataset, self.val_dataset = random_split(
            DigitDataset(train_data["image"], train_data["label"]), # type: ignore
            [0.8, 0.2],
        )
        self.test_dataset = DigitDataset(test_data["image"], test_data["label"]) # type: ignore
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            shuffle=False,
        )


class DigitRecognizer(pl.LightningModule):
    def __init__(
        self,
        config: ModelConfig,
    ):
        super().__init__()
        self.save_hyperparameters(config)
        
        self.val_pred = []
        self.val_label = []
        
        self.test_pred = []
        self.test_label = []
        
        self.config = config
        
        self.criterion = nn.CrossEntropyLoss()
        
        self.conv1 = nn.Conv2d(1, 3, 5) # -> [3, 24, 24]
        self.conv2 = nn.Conv2d(3, 5, 5) # -> [5, 20, 20]
        self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(5 * 20 * 20, 512)
        self.fc2 = nn.Linear(512, 128)
        self.to_output = nn.Linear(128, 10)
    
    def forward(
        self,
        image,
        label = None,
    ):
        x = image
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        logit = self.to_output(x)
        
        return logit
    
    def training_step(self, batch, batch_idx):
        image, label = batch
        logit = self(image) # [batch_size, 10]
        loss = self.criterion(logit, label)
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        image, label = batch
        logit = self(image) # [batch_size, 10]
        loss = self.criterion(logit, label)
        self.log("val_loss", loss, sync_dist=True)
        
        pred = logit.argmax(dim=-1)
        
        self.val_pred.extend(pred)
        self.val_label.extend(label)
    
    def on_validation_epoch_end(self) -> None:
        val_pred = torch.tensor(self.val_pred)
        val_label = torch.tensor(self.val_label)
        
        self.log("val_acc", torch.mean((val_pred == val_label).float()), sync_dist=True)
        
        self.val_pred = []
        self.val_label = []
    
    def test_step(self, batch, batch_idx):
        image, label = batch
        logit = self(image) # [batch_size, 10]
        loss = self.criterion(logit, label)
        self.log("test_loss", loss, sync_dist=True)
        
        pred = logit.argmax(dim=-1)
        
        self.test_pred.extend(pred)
        self.test_label.extend(label)
    
    def on_test_epoch_end(self) -> None:
        test_pred = torch.tensor(self.test_pred)
        test_label = torch.tensor(self.test_label)
        
        self.log("test_acc", torch.mean((test_pred == test_label).float()), sync_dist=True)
        
        pd.DataFrame({
            "pred": test_pred.tolist(),
            "label": test_label.tolist()
        }).to_csv("test_result.csv")
        
        self.test_pred = []
        self.test_label = []
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.config["lr"])


if __name__ == "__main__":
    train_config = TrainConfig(
        hg_dataset_name="mnist",
        batch_size=128,
        num_workers=10,
        early_stopping_patience=3,
    )
    datamodule = DigitDatamodule(train_config)
    
    model_config = ModelConfig(lr=0.0001)
    model = DigitRecognizer.load_from_checkpoint(
        checkpoint_path="./1-basic-pytorch/4fib5a35/checkpoints/epoch=25-step=4888.ckpt",
        model_config=model_config
    )

    logger = WandbLogger(project="1-basic-pytorch")
    
    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=train_config["early_stopping_patience"], mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min"),
        ],
        accelerator="gpu",
        devices=2,
        strategy="ddp",
    )
    
    # trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
