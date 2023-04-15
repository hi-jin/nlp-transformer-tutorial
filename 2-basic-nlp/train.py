from typing import TypedDict, List, Union, Optional

import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from datasets import load_dataset


class TrainConfig(TypedDict):
    hg_dataset_path: str
    batch_size: int
    num_workers: int


class ModelConfig(TypedDict):
    max_seq_length: int
    num_classes: int
    lr: float


class MyTokenizer:
    def __init__(
        self,
    ):  
        self.pad_token_id = 0
        self.pad_token = "<pad>"

        self.unk_token_id = 1
        self.unk_token = "<unk>"
        
        self.__token_to_id = {self.pad_token: 0, self.unk_token: 1}
        self.__id_to_token = [self.pad_token, self.unk_token]
        
        self.vocab_size = 2
    
    def __call__(
        self,
        text: Union[str, List[str]],
        max_length: Optional[int] = None,
    ):
        return self.tokenize(text, max_length)
    
    def tokenize(
        self, 
        text: Union[str, List[str]], 
        max_length: Optional[int] = None, 
        add_new_tokens = False,
    ):  
        if type(text) is str:
            return torch.tensor(self.__tokenize_one_sentence(text, add_new_tokens), dtype=torch.long)
        else:
            if max_length is None:
                max_length = max(map(lambda sentence: len(self.__tokenize_one_sentence(sentence, add_new_tokens)), text))

            result = []
            for single_text in text:
                tokenized = self.__tokenize_one_sentence(single_text, add_new_tokens)
                if len(tokenized) > max_length:
                    tokenized = tokenized[:max_length]
                else:
                    pad_len = max_length - len(tokenized)
                    tokenized.extend([self.pad_token_id] * pad_len)
                result.append(tokenized)
            return torch.tensor(result)
    
    def __tokenize_one_sentence(self, text: str, add_new_tokens: bool):
        result = []
        for token in text.split():
            if token in self.__token_to_id:
                result.append(self.__token_to_id[token])
            else:
                if add_new_tokens:
                    new_id = self.vocab_size
                    self.__token_to_id[token] = new_id
                    self.__id_to_token.append(token)
                    self.vocab_size += 1
                    result.append(new_id)
                else:
                    result.append(self.unk_token_id)
        return result


class NewsClassifier(pl.LightningModule):
    def __init__(
        self,
        config: ModelConfig,
        tokenizer: MyTokenizer,
    ):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.tokenizer = tokenizer
        
        self.val_pred = []
        self.val_label = []
        self.test_pred = []
        self.test_label = []
        
        self.embedding = nn.Embedding(tokenizer.vocab_size, 512) # -> [batch_size, num_tokens, 512]
        self.fc1 = nn.Linear(512, 128)
        self.fc2 = nn.Linear(128, 64)
        self.flatten = nn.Flatten()
        self.fc3 = nn.Linear(self.config["max_seq_length"] * 64, 512)
        self.to_output = nn.Linear(512, self.config["num_classes"])
        
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, text, label = None):
        x = self.embedding(text)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.flatten(x)
        x = F.relu(self.fc3(x))
        logit = self.to_output(x)
        return logit
    
    def training_step(self, batch, batch_idx):
        logit = self(*batch)
        loss = self.criterion(logit, batch[1])
        self.log("train_loss", loss, sync_dist=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        logit = self(*batch)
        loss = self.criterion(logit, batch[1])
        self.log("val_loss", loss, sync_dist=True)
        self.val_pred.extend(logit.argmax(dim=-1))
        self.val_label.extend(batch[1])
    
    def on_validation_epoch_end(self) -> None:
        acc = torch.mean((torch.tensor(self.val_pred) == torch.tensor(self.val_label)).float())
        self.log("val_acc", acc)
        
        self.val_pred = []
        self.val_label = []
    
    def test_step(self, batch, batch_idx):
        logit = self(*batch)
        loss = self.criterion(logit, batch[1])
        self.log("test_loss", loss, sync_dist=True)
        self.test_pred.extend(logit.argmax(dim=-1))
        self.test_label.extend(batch[1])
    
    def on_test_epoch_end(self) -> None:
        test_pred = torch.tensor(self.test_pred)
        test_label = torch.tensor(self.test_label)
        acc = torch.mean((test_pred == test_label).float())
        self.log("test_acc", acc)

        pd.DataFrame({
            "pred": test_pred.tolist(),
            "label": test_label.tolist(),
        }).to_csv("test_result.csv")
        
        self.test_pred = []
        self.test_label = []

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.config["lr"])


class AGNewsDataset(Dataset):
    def __init__(
        self,
        text,
        label,
    ):
        super().__init__()
        self.text = text
        self.label = label
    
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        text = self.text[idx]
        label = self.label[idx]
        
        return text, label


class AGNewsDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        config: TrainConfig,
        tokenizer: MyTokenizer,
        max_seq_length: int,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
    
    def prepare_data(self):
        load_dataset(self.config["hg_dataset_path"])
    
    def setup(self, stage = None):
        data = load_dataset(self.config["hg_dataset_path"])
        
        train_data = data["train"] # type: ignore
        test_data = data["test"] # type: ignore

        self.train_dataset, self.val_dataset = random_split(
            dataset=AGNewsDataset(train_data["text"], train_data["label"]), # type: ignore
            lengths=[0.8, 0.2],
        )
        self.test_dataset = AGNewsDataset(test_data["text"], test_data["label"]) # type: ignore
    
    def collate_fn(self, batch):
        text, label = zip(*batch)
        text = self.tokenizer.tokenize(text, max_length=self.max_seq_length)
        return text, torch.tensor(label, dtype=torch.long)
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=True,
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=False,
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.config["num_workers"],
            collate_fn=self.collate_fn,
            shuffle=False,
        )


def initialize_tokenizer():
    def wrapper(tokenizer):
        def collate_fn_for_initialize(batch):
            text, label = zip(*batch)
            text = tokenizer.tokenize(text, add_new_tokens=True)
            return text, label
        return collate_fn_for_initialize
    
    tokenizer = MyTokenizer()
    train_data = load_dataset("ag_news")["train"] # type: ignore
    train_dataset = AGNewsDataset(train_data["text"], train_data["label"]) # type: ignore
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=128,
        collate_fn=wrapper(tokenizer),
    )
    
    for _ in train_dataloader:
        pass

    return tokenizer

if __name__ == "__main__":
    train_config = TrainConfig(
        hg_dataset_path="ag_news",
        batch_size=128,
        num_workers=10,
    )

    model_config = ModelConfig(
        max_seq_length=128,
        num_classes=4,
        lr=0.0001,
    )
    
    tokenizer = initialize_tokenizer()
    datamodule = AGNewsDatamodule(train_config, tokenizer, model_config["max_seq_length"])
    
    classifier = NewsClassifier.load_from_checkpoint(
        checkpoint_path="./2-basic-nlp/f88iaiob/checkpoints/epoch=18-step=7125.ckpt",
        model_config=model_config,
        tokenizer=tokenizer,
    )

    logger = WandbLogger(project="2-basic-nlp")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(monitor="val_loss"),
        ],
        accelerator="gpu",
        devices=2,
        strategy="ddp",
    )
    
    # trainer.fit(classifier, datamodule)
    trainer.test(classifier, datamodule)
