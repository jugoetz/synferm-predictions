from collections import defaultdict

import torch
from pytorch_lightning.callbacks import Callback
import wandb


class LogMetricsCallback(Callback):
    """PyTorch Lightning metric callback that logs trainer metrics to wandb on every validation epoch and accmulates them into a dictionary."""

    def __init__(self):
        super().__init__()
        self.metrics = defaultdict(list)

    def on_validation_epoch_end(self, trainer, pl_module):
        for k, v in trainer.callback_metrics.items():
            self.metrics[k].append(v)
        wandb.log(trainer.callback_metrics, step=trainer.current_epoch)

    def on_fit_end(self, trainer, pl_module):
        self.metrics = {k: torch.stack(v, dim=0) for k, v in self.metrics.items()}


class BestValLossEpochCallback(Callback):
    """Save the best epoch during training by val/loss"""

    def __init__(self):
        super().__init__()
        self.best_epoch = 0
        self.best_score = float("inf")

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.callback_metrics["val/loss"] < self.best_score:
            self.best_score = trainer.callback_metrics["val/loss"]
            self.best_epoch = trainer.current_epoch
