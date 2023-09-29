from copy import deepcopy

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torchmetrics as tm
from dgllife.model.gnn.gcn import GCN

from src.layer.ffn import FFN
from src.layer.mpnn import MPNNEncoder
from src.layer.pooling import (
    GlobalAttentionPooling,
    AvgPooling,
    SumPooling,
    MaxPooling,
    ConcatenateNodeEdgeSumPooling,
)
from src.layer.util import get_activation
from src.util.definitions import PRED_DIR


def load_model(hparams):
    if hparams["name"] == "D-MPNN":
        model = DMPNNModel(**hparams)
    elif hparams["name"] == "GCN":
        model = GCNModel(**hparams)
    elif hparams["name"] == "FFN":
        model = FFNModel(**hparams)
    elif hparams["name"] == "GraphAgnostic":
        model = GraphAgnosticModel(**hparams)
    else:
        raise ValueError(f"Model type {hparams['name']} not supported.")
    return model


def load_trained_model(model_type, checkpoint_path):
    if model_type == "D-MPNN":
        model = DMPNNModel.load_from_checkpoint(checkpoint_path)
    elif model_type == "GCN":
        model = GCNModel.load_from_checkpoint(checkpoint_path)
    elif model_type == "FFN":
        model = FFNModel.load_from_checkpoint(checkpoint_path)
    elif model_type == "GraphAgnostic":
        model = GraphAgnosticModel.load_from_checkpoint(checkpoint_path)
    else:
        raise ValueError(f"Model type {model_type} not supported.")
    return model


class Classifier(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.encoder = self.init_encoder()
        self.decoder = self.init_decoder()
        self.loss = self.init_loss()
        self.val_indices = []
        self.val_predictions = []
        self.test_indices = []
        self.test_predictions = []
        average = "micro" if self.hparams["training"]["task"] == "multiclass" else None
        auroc_average = (
            "macro" if self.hparams["training"]["task"] == "multiclass" else None
        )
        # TODO there is an upstream problem with torchmetrics
        #  the binary and multilabel accuracies assume they are receiving logits if any value is outside [0,1]
        #  otherwise they assume probabilities.
        #  Of course this is a stupid assumption.
        #  Hopefully soon fixed upstream, see https://github.com/Lightning-AI/torchmetrics/issues/1604
        #  and https://github.com/Lightning-AI/torchmetrics/pull/1676
        #  meantime "Prinzip Hoffnung"
        self.metrics = torch.nn.ModuleDict(
            {
                "accuracy": tm.Accuracy(
                    task=self.hparams["training"]["task"],
                    num_labels=kwargs["num_labels"],
                    num_classes=kwargs["num_labels"],
                    threshold=0.5,  # applies for binary and multilabel
                    average=average,
                ),
                "precision": tm.Precision(
                    task=self.hparams["training"]["task"],
                    num_labels=kwargs["num_labels"],
                    num_classes=kwargs["num_labels"],
                    threshold=0.5,  # applies for binary and multilabel
                    average=average,
                ),
                "recall": tm.Recall(
                    task=self.hparams["training"]["task"],
                    num_labels=kwargs["num_labels"],
                    num_classes=kwargs["num_labels"],
                    threshold=0.5,  # applies for binary and multilabel
                    average=average,
                ),
                "f1": tm.F1Score(
                    task=self.hparams["training"]["task"],
                    num_labels=kwargs["num_labels"],
                    num_classes=kwargs["num_labels"],
                    threshold=0.5,  # applies for binary and multilabel
                    average=average,
                ),
            }
        )

    def init_encoder(self):
        raise NotImplementedError("Child class must implement this method")

    def init_decoder(self):
        raise NotImplementedError("Child class must implement this method")

    def init_loss(self):
        if self.hparams["training"]["task"] in ["binary", "multilabel"]:
            return torch.nn.BCEWithLogitsLoss()
        elif self.hparams["training"]["task"] == "multiclass":
            return torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(
                f"'task' needs to be one of [binary, multiclass, multilabel]. Input: {self.hparams['training']['task']}"
            )

    def forward(self, x):
        raise NotImplementedError("Child class must implement this method")

    def _get_preds(self, batch):
        raise NotImplementedError("Child class must implement this method")

    def _get_preds_loss_metrics(self, batch, dataloader_idx):
        labels = batch[-1]
        indices = batch[0]
        X = batch[1:-1]
        # we need to transform the labels to class indices
        if self.hparams["training"]["task"] == "multiclass":
            # apply argmax to get class index
            targets = torch.argmax(labels, dim=1)
        else:
            # for binary and multilabel task, no transformations are needed
            targets = labels

        preds = self._get_preds(X)

        # calculate loss
        loss = self.loss(preds, targets)

        # calculate metrics
        metrics = {
            f"{dataloader_idx}/{k}": v(preds, targets) for k, v in self.metrics.items()
        }
        # in the multilabel case, we receive metrics for each label
        # we separate them
        if self.hparams["training"]["task"] == "multilabel":
            for k, v in deepcopy(metrics).items():
                if len(v) == self.hparams["num_labels"]:
                    for i, c in enumerate(self.hparams["target_names"]):
                        metrics[f"{k}_target_{c}"] = metrics[k][i]
                    del metrics[k]

        return indices, preds, loss, metrics

    def training_step(self, batch, batch_idx):
        indices, preds, loss, metrics = self._get_preds_loss_metrics(batch, "train")
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        indices, preds, loss, metrics = self._get_preds_loss_metrics(batch, "val")
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        self.val_indices.append(indices)
        if self.hparams["training"]["task"] == "multiclass":
            # for multiclass, we apply softmax to get class probabilities
            self.val_predictions.append(torch.softmax(preds, dim=1))
        else:
            # for multilabel and binary, we apply sigmoid to get the (label-wise) binary probability for the pos class
            self.val_predictions.append(torch.sigmoid(preds))
        return loss

    def test_step(self, batch, batch_idx):
        indices, preds, loss, metrics = self._get_preds_loss_metrics(batch, "test")
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log_dict(metrics, on_step=False, on_epoch=True)
        self.test_indices.append(indices)
        if self.hparams["training"]["task"] == "multiclass":
            # for multiclass, we apply softmax to get class probabilities
            self.test_predictions.append(torch.softmax(preds, dim=1))
        else:
            # for multilabel and binary, we apply sigmoid to get the (label-wise) binary probability for the pos class
            self.test_predictions.append(torch.sigmoid(preds))
        return loss

    def on_validation_epoch_end(self) -> None:
        filepath = PRED_DIR / self.hparams["run_id"] / "val_preds_last.csv"
        filepath.parent.mkdir(exist_ok=True, parents=True)
        ind = np.array([i for ind in self.val_indices for i in ind]).reshape(
            (-1, 1)
        )  # shape (n_samples, 1)
        preds = (
            torch.concatenate(self.val_predictions).cpu().numpy()
        )  # shape (n_samples, n_labels)
        columns = [
            "idx",
        ] + [f"pred_{n}" for n in range(preds.shape[1])]
        pd.DataFrame(np.hstack((ind, preds)), columns=columns).astype(
            {"idx": "int32"}
        ).to_csv(filepath, index=False)
        self.val_indices = []
        self.val_predictions = []

    def on_test_epoch_end(self) -> None:
        filepath = PRED_DIR / self.hparams["run_id"] / "test_preds_last.csv"
        filepath.parent.mkdir(exist_ok=True, parents=True)
        ind = np.array([i for ind in self.test_indices for i in ind]).reshape(
            (-1, 1)
        )  # shape (n_samples, 1)
        preds = (
            torch.concatenate(self.test_predictions).cpu().numpy()
        )  # shape (n_samples, n_labels)
        columns = [
            "idx",
        ] + [f"pred_{n}" for n in range(preds.shape[1])]
        pd.DataFrame(np.hstack((ind, preds)), columns=columns).astype(
            {"idx": "int32"}
        ).to_csv(filepath, index=False)
        self.test_indices = []
        self.test_predictions = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        X = batch[1:-1]  # remove indices and labels
        # TODO use the labelbinarizer inverse transform to obtain the actual predictions if that is desired
        return self._get_preds(X)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.parameters()),
            lr=self.hparams.optimizer["lr"],
            weight_decay=self.hparams.optimizer["weight_decay"],
        )

        # learning rate scheduler
        scheduler = self._config_lr_scheduler(optimizer)

        if scheduler is None:
            return optimizer
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": scheduler,
                "monitor": "val/loss",
            }

    def _config_lr_scheduler(self, optimizer):
        scheduler_name = self.hparams.optimizer["lr_scheduler"][
            "scheduler_name"
        ].lower()

        if scheduler_name == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode="max", factor=0.4, patience=50, verbose=True
            )
        elif scheduler_name == "exp_with_linear_warmup":
            lr_min = self.hparams.optimizer["lr_scheduler"]["lr_min"]
            lr_max = self.hparams.optimizer["lr"]
            start_factor = lr_min / lr_max
            end_factor = 1
            n_warmup_epochs = self.hparams.optimizer["lr_scheduler"]["lr_warmup_step"]

            gamma = start_factor ** (
                1 / (self.hparams.optimizer["lr_scheduler"]["epochs"] - n_warmup_epochs)
            )

            def lr_foo(epoch):
                if epoch <= self.hparams.optimizer["lr_scheduler"]["lr_warmup_step"]:
                    # warm up lr
                    lr_scale = start_factor + (
                        epoch
                        * (end_factor - start_factor / end_factor)
                        / n_warmup_epochs
                    )

                else:
                    lr_scale = gamma ** (epoch - n_warmup_epochs)

                return lr_scale

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_foo)

        elif scheduler_name == "none":
            scheduler = None
        else:
            raise ValueError(
                f"Not supported lr scheduler: {self.hparams.optimizer['lr_scheduler']}"
            )

        return scheduler


class DMPNNModel(Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_encoder(self):
        return MPNNEncoder(
            atom_feature_size=self.hparams["atom_feature_size"],
            bond_feature_size=self.hparams["bond_feature_size"],
            **self.hparams.encoder,
        )

    def init_decoder(self):
        return FFN(
            in_size=self.hparams.encoder["hidden_size"]
            + self.hparams["global_feature_size"],
            out_size=self.hparams["num_labels"],
            **self.hparams.decoder,
        )

    def forward(self, x):
        graph, global_features = x
        embedding = self.encoder(graph)
        if self.hparams["global_feature_size"] == 0:
            y = self.decoder(embedding)
        else:
            y = self.decoder(torch.cat((embedding, global_features), dim=1))

        return y

    def _get_preds(self, batch):
        y_hat = self(batch)
        return y_hat


class GCNModel(Classifier):
    """
    GCN model as implemented in DGL-Lifesci.

    Note that this model works on a molecular graph and that its forward method expects a BE-graph input.

    This class has a few hacky elements, due to some differences in the architecture of dgllife models and my own.
    Mainly:
        - dgllife encoder does not take care of pooling
        - dgllife encoder .forward() needs the feature vector to be passed separately
        - dgllife encoder expects a homogeneous graph
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pooling = self.init_pooling()

    def init_encoder(self):
        return GCN(
            in_feats=self.hparams["atom_feature_size"],
            hidden_feats=[
                self.hparams["encoder"]["hidden_size"]
                for _ in range(self.hparams["encoder"]["depth"])
            ],
            gnn_norm=["both" for _ in range(self.hparams["encoder"]["depth"])],
            activation=[
                get_activation(self.hparams["encoder"]["activation"])
                for _ in range(self.hparams["encoder"]["depth"])
            ],
            batchnorm=[False for _ in range(self.hparams["encoder"]["depth"])],
            dropout=[
                self.hparams["encoder"]["dropout_ratio"]
                for _ in range(self.hparams["encoder"]["depth"])
            ],
        )

    def init_pooling(self):
        if self.hparams["encoder"]["aggregation"] == "attention":
            return GlobalAttentionPooling(
                gate_nn=torch.nn.Linear(self.hparams["encoder"]["hidden_size"], 1),
                ntype="_N",
                feat="h_v",
                get_attention=True,
            )
        elif self.hparams["encoder"]["aggregation"] == "mean":
            return AvgPooling(ntype="_N", feat="h_v")
        elif self.hparams["encoder"]["aggregation"] == "sum":
            return SumPooling(ntype="_N", feat="h_v")
        elif self.hparams["encoder"]["aggregation"] == "max":
            return MaxPooling(ntype="_N", feat="h_v")
        else:
            raise ValueError(
                "Aggregation must be one of ['max', 'mean', 'sum', 'attention']"
            )

    def init_decoder(self):
        return FFN(
            in_size=self.hparams.encoder["hidden_size"]
            + self.hparams["global_feature_size"],
            out_size=self.hparams["num_labels"],
            **self.hparams.decoder,
        )

    def forward(self, x):
        graph, global_features = x
        embedding_before_pooling = self.encoder(graph, graph.ndata["x"])
        with graph.local_scope():
            graph.ndata["h_v"] = embedding_before_pooling
            if self.hparams["encoder"]["aggregation"] == "attention":
                embedding, attention = self.pooling(graph)
            else:
                embedding = self.pooling(graph)

        if self.hparams["global_feature_size"] == 0:
            y = self.decoder(embedding)
        else:
            y = self.decoder(torch.cat((embedding, global_features), dim=1))
        return y

    def _get_preds(self, batch):
        y_hat = self(batch)
        return y_hat


class FFNModel(Classifier):
    """
    FFN-only model.

    This expects a vectorized input, e.g., a fingerprint or a global feature vector.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_encoder(self):
        return None

    def init_decoder(self):
        return FFN(
            in_size=self.hparams["global_feature_size"],
            out_size=self.hparams["num_labels"],
            **self.hparams.decoder,
        )

    def forward(self, x):
        return self.decoder(x)

    def _get_preds(self, batch):
        _, global_features_batch = batch
        y_hat = self(global_features_batch)
        return y_hat


class GraphAgnosticModel(Classifier):
    """
    Baseline model that does not use graph connectivity information.
    This pools the node features and edge features separately, and concatenates them.

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init_encoder(self):
        return ConcatenateNodeEdgeSumPooling(
            ntype="_N", etype="_E", nfeat="x", efeat="e"
        )

    def init_decoder(self):
        return FFN(
            in_size=self.hparams["atom_feature_size"]
            + self.hparams["bond_feature_size"],
            out_size=self.hparams["num_labels"],
            **self.hparams.decoder,
        )

    def forward(self, x):
        graph, global_features = x
        embedding = self.encoder(graph)
        if self.hparams["global_feature_size"] == 0:
            y = self.decoder(embedding)
        else:
            y = self.decoder(torch.cat((embedding, global_features), dim=1))
        return y

    def _get_preds(self, batch):
        y_hat = self(batch)
        return y_hat
