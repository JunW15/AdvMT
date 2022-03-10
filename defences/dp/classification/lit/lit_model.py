from argparse import ArgumentParser
from datetime import datetime
import torch
import pytorch_lightning as pl
import datasets
from transformers import (
    DistilBertForSequenceClassification,
    BertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import f1_score
from opacus import PrivacyEngine


class IMDBTransformer(pl.LightningModule):

    def __init__(self,
                 num_labels: int,
                 learning_rate: float = 2e-5,
                 adam_epsilon: float = 1e-8,
                 warmup_steps: int = 0,
                 weight_decay: float = 0.0,
                 **kwargs
                 ):
        super().__init__()
        self.save_hyperparameters()

        self.model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        self.metric = datasets.load_metric(
            'glue',
            'sst2',
            experiment_id=datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        )
        trainable_layers = [
            # self.model.distilbert.transformer.layer[-1],
            self.model.pre_classifier,
            self.model.classifier
        ]

        for p in self.model.parameters():
            p.requires_grad = False

        for layer in trainable_layers:
            for p in layer.parameters():
                p.requires_grad = True

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        else:
            preds = None

        labels = batch["labels"]

        return {'loss': val_loss, "preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs) -> None:
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x['loss'] for x in outputs]).mean()
        self.log('val_loss', loss, prog_bar=True)
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)

    def test_step(self, batch, batch_idx):
        outputs = self(**batch)
        test_loss, logits = outputs[:2]

        if self.hparams.num_labels >= 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        else:
            preds = None

        labels = batch["labels"]

        return {"preds": preds, "labels": labels}

    def test_epoch_end(self, outputs):
        preds = torch.cat([x['preds'] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x['labels'] for x in outputs]).detach().cpu().numpy()
        f1 = f1_score(labels, preds, average='micro')
        self.log_dict(self.metric.compute(predictions=preds, references=labels), prog_bar=True)
        self.log_dict({'f1': f1})

    def setup(self, stage):
        if stage == 'fit':
            train_loader = self.train_dataloader()
            self.train_size = len(train_loader.dataset)
            self.total_steps = (
                    (self.train_size // (self.hparams.train_batch_size * max(1, self.hparams.gpus)))
                    // self.hparams.accumulate_grad_batches
                    * float(self.hparams.max_epochs)
            )
            if self.hparams.privacy:
                print('[Privacy] creating privacy engine ...')
                self.privacy_engine = PrivacyEngine(
                    module=self.model.cuda(),
                    batch_size=self.hparams.train_batch_size * 2,
                    sample_size=self.train_size,
                    alphas=[1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64)),
                    noise_multiplier=self.hparams.noise,
                    max_grad_norm=self.hparams.gnorm,
                )

    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""
        optimizer = AdamW(self.model.parameters(),
                          lr=self.hparams.learning_rate,
                          eps=self.hparams.adam_epsilon)

        if self.hparams.privacy:
            assert self.privacy_engine is not None
            print('[Privacy] attaching privacy engine to optimizer ...')
            self.privacy_engine.attach(optimizer)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=self.total_steps
        )
        scheduler = {
            'scheduler': scheduler,
            'interval': 'step',
            'frequency': 1
        }
        return [optimizer], [scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--learning_rate", default=2e-5, type=float)
        parser.add_argument("--adam_epsilon", default=1e-8, type=float)
        parser.add_argument("--warmup_steps", default=0, type=int)
        parser.add_argument("--weight_decay", default=0.0, type=float)
        parser.add_argument("--privacy", action='store_true')
        parser.add_argument("--noise", default=0.0, type=float)
        parser.add_argument("--gnorm", default=5.0, type=float)
        return parser
