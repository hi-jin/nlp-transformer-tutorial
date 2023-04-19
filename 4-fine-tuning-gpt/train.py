from typing import TypedDict
import pandas as pd
import torch
import torchmetrics
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from transformers import AutoModel, PreTrainedTokenizerFast, BartModel
from datasets import load_dataset


class ModelConfig(TypedDict):
    lr: float
    hg_pretrained_model_name: str
    d_model: int
    vocab_size: int
    max_length: int


class TrainConfig(TypedDict):
    hg_dataset_path: str
    hg_dataset_subdir: str
    batch_size: int
    num_workers: int


class Model(pl.LightningModule):
    def __init__(
        self,
        config: ModelConfig,
        pretrained_model,
        tokenizer: PreTrainedTokenizerFast,
    ):
        super().__init__()
        self.save_hyperparameters(config)
        
        self.config = config
        self.pretrained_model = pretrained_model
        self.tokenizer = tokenizer
        
        self.to_output = torch.nn.Linear(config["d_model"], config["vocab_size"])
        
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id) # type: ignore
        self.bleu = torchmetrics.BLEUScore()
        
        self.val_pred_texts = []
        self.val_label_texts = []
        self.test_pred_texts = []
        self.test_label_texts = []
    
    def forward(self, input_ids, attention_mask, labels = None):
        output = self.pretrained_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        
        hidden_state = output["last_hidden_state"]
        logit = self.to_output(hidden_state) # [batch_size, seq_len, vocab_size]
        
        return logit
    
    def training_step(self, batch, batch_size):
        logit = self(**batch)
        labels = batch["labels"]

        loss = self.criterion(logit.permute(0, 2, 1), labels)
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_size):
        logit = self(**batch)
        labels = batch["labels"]

        loss = self.criterion(logit.permute(0, 2, 1), labels)
        self.log("val_loss", loss)
        
        pred_text = self.tokenizer.batch_decode(logit.argmax(dim=-1), skip_special_tokens=True)
        label_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        self.val_pred_texts.extend(pred_text)
        self.val_label_texts.extend(label_text)
        
    def on_validation_epoch_end(self) -> None:
        pred_text = self.val_pred_texts
        label_text = self.val_label_texts

        bleu_score = self.bleu(pred_text, label_text)
        self.log("val_bleu", bleu_score)
        
        self.val_pred_texts = []
        self.val_label_texts = []
    
    def test_step(self, batch, batch_idx):
        logit = self(**batch)
        labels = batch["labels"]

        loss = self.criterion(logit.permute(0, 2, 1), labels)
        self.log("test_loss", loss)
        
        pred_text = self.tokenizer.batch_decode(logit.argmax(dim=-1), skip_special_tokens=True)
        label_text = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        self.test_pred_texts.extend(pred_text)
        self.test_label_texts.extend(label_text)
    
    def on_test_epoch_end(self) -> None:
        pred_text = self.test_pred_texts
        label_text = self.test_label_texts

        bleu_score = self.bleu(pred_text, label_text)
        self.log("test_bleu", bleu_score)
        
        pd.DataFrame({
            "pred_text": pred_text,
            "label_text": label_text,
        }).to_csv("test_result.csv")
        
        self.test_pred_texts = []
        self.test_label_texts = []
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), self.config["lr"])


class Dataset(torch.utils.data.Dataset): # type: ignore
    def __init__(self, sentence1, sentence2):
        super().__init__()
        self.sentence1 = sentence1
        self.sentence2 = sentence2
    
    def __len__(self):
        return len(self.sentence1)
    
    def __getitem__(self, idx):
        sentence1 = self.sentence1[idx]
        sentence2 = self.sentence2[idx]

        return sentence1, sentence2


class Datamodule(pl.LightningDataModule):
    def __init__(
        self,
        config: TrainConfig,
        model_config: ModelConfig,
        tokenizer,
    ):
        super().__init__()
        self.config = config
        self.model_config = model_config
        self.tokenizer = tokenizer
    
    def prepare_data(self):
        load_dataset(
            path=self.config['hg_dataset_path'],
            name=self.config['hg_dataset_subdir'],
        )
    
    def setup(self, stage = None):
        data = load_dataset(
            path=self.config['hg_dataset_path'],
            name=self.config['hg_dataset_subdir'],
        )
        
        train_data = data["train"] # type: ignore
        test_data = data["validation"] # type: ignore

        self.train_dataset, self.val_dataset = torch.utils.data.random_split( # type: ignore
            dataset=Dataset(
                sentence1=train_data["sentence1"], # type: ignore
                sentence2=train_data["sentence2"], # type: ignore
            ),
            lengths=[0.8, 0.2],
        )
        self.test_dataset = Dataset(
            sentence1=test_data["sentence1"], # type: ignore
            sentence2=test_data["sentence2"], # type: ignore
        )
    
    def collate_fn(self, batch):
        sentence1, sentence2 = zip(*batch)

        inputs = self.tokenizer(sentence1, max_length=self.model_config["max_length"] ,padding="max_length", truncation=True, return_tensors="pt")
        outputs = self.tokenizer(sentence2, max_length=self.model_config["max_length"] , padding="max_length", truncation=True, return_tensors="pt")

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": outputs["input_ids"],
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
    model_config = ModelConfig(
        lr=0.0001,
        hg_pretrained_model_name="hyunwoongko/kobart",
        d_model=768,
        vocab_size=30000,
        max_length=64,
    )
    train_config = TrainConfig(
        hg_dataset_path="klue",
        hg_dataset_subdir="sts",
        batch_size=64,
        num_workers=10,
    )
    
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        model_config["hg_pretrained_model_name"],
    )
    pretrained_model = AutoModel.from_pretrained(model_config["hg_pretrained_model_name"])
    
    datamodule = Datamodule(train_config, model_config, tokenizer)
    model = Model(model_config, pretrained_model, tokenizer)

    logger = WandbLogger(project="4-fine-tuning-gpt")

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[
            EarlyStopping(monitor="val_loss", mode="min"),
            ModelCheckpoint(monitor="val_loss", mode="min"),
        ],
        accelerator="gpu",
        devices=2,
    )

    trainer.fit(model, datamodule)
    trainer.test(model, datamodule)
