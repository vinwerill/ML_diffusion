from typing import Any, Callable, List, Optional

import librosa
import plotly.graph_objs as go
import pytorch_lightning as pl
import torch
import torchaudio
import wandb
from audio_data_pytorch.utils import fractional_random_split
from audio_diffusion_pytorch import Sampler, Schedule #commented out AudioDiffusionModel
from einops import rearrange
from ema_pytorch import EMA
from pytorch_lightning import Callback, Trainer
from pytorch_lightning.loggers import WandbLogger
from torch import Tensor, nn
from torch.utils.data import DataLoader

""" Model """


class Model(pl.LightningModule):
    def __init__(
        self,
        lr: float,
        lr_beta1: float,
        lr_beta2: float,
        lr_eps: float,
        lr_weight_decay: float,
        ema_beta: float,
        ema_power: float,
        model: nn.Module,
    ):
        super().__init__()
        self.lr = lr
        self.lr_beta1 = lr_beta1
        self.lr_beta2 = lr_beta2
        self.lr_eps = lr_eps
        self.lr_weight_decay = lr_weight_decay
        self.ema_beta = ema_beta
        self.ema_power = ema_power
        self.model = model
        self.ema = EMA(beta=ema_beta, power=ema_power)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = nn.functional.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.lr,
            betas=(self.lr_beta1, self.lr_beta2),
            eps=self.lr_eps,
            weight_decay=self.lr_weight_decay,
        )
        return optimizer


""" Datamodule """


class Datamodule(pl.LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: Optional[str] = None):
        # Load and split data here
        pass

    def train_dataloader(self):
        # Return the training dataloader
        pass

    def val_dataloader(self):
        # Return the validation dataloader
        pass

    def test_dataloader(self):
        # Return the test dataloader
        pass


""" Callbacks """


def get_wandb_logger(trainer: Trainer) -> Optional[WandbLogger]:
    """Safely get Weights&Biases logger from Trainer."""
    for logger in trainer.loggers:
        if isinstance(logger, WandbLogger):
            return logger
    return None
