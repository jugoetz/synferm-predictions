import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as tm

from dgllife.model.gnn.gcn import GCN
from torch import nn

from src.layer.mpnn import MPNNEncoder
from src.layer.ffn import FFN
from src.layer.pooling import (
    GlobalAttentionPooling,
    AvgPooling,
    SumPooling,
    MaxPooling,
    ConcatenateNodeEdgeSumPooling,
)
from src.layer.util import get_activation


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
        self.metrics = torch.nn.ModuleDict(
            {
                "accuracy": tm.Accuracy(),
                "AUROC": tm.AUROC(),
                "precision": tm.Precision(),
                "recall": tm.Recall(),
                "f1": tm.F1Score(),
            }
        )

    def init_encoder(self):
        raise NotImplementedError("Child class must implement this method")

    def init_decoder(self):
        raise NotImplementedError("Child class must implement this method")

    def forward(self, x):
        raise NotImplementedError("Child class must implement this method")

    def _get_preds(self, batch):
        raise NotImplementedError("Child class must implement this method")

    def _get_preds_loss_metrics(self, batch):
        y = batch[-1]
        y_hat = self._get_preds(batch)

        # calculate loss
        loss = self.calc_loss(y_hat, y)

        return y_hat, loss, {k: v(y_hat, y) for k, v in self.metrics.items()}

    def training_step(self, batch, batch_idx):
        preds, loss, metrics = self._get_preds_loss_metrics(batch)
        self.log(
            "train/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        for k, v in metrics.items():
            self.log(
                f"train/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        return loss

    def validation_step(self, batch, batch_idx):
        preds, loss, metrics = self._get_preds_loss_metrics(batch)
        self.log(
            "val/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        for k, v in metrics.items():
            self.log(
                f"val/{k}", v, on_step=False, on_epoch=True, prog_bar=False, logger=True
            )
        return loss

    def test_step(self, batch, batch_idx):
        preds, loss, metrics = self._get_preds_loss_metrics(batch)
        self.log(
            "test/loss", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        for k, v in metrics.items():
            self.log(
                f"test/{k}",
                v,
                on_step=False,
                on_epoch=True,
                prog_bar=False,
                logger=True,
            )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self._get_preds(batch)

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

    def calc_loss(self, preds, truth):
        if self.hparams.decoder["out_sigmoid"]:
            loss = F.binary_cross_entropy(
                preds,
                truth.to(torch.float),  # input label is int for metric purpose
                reduction="mean",
            )
        else:
            loss = F.binary_cross_entropy_with_logits(
                preds,
                truth.to(torch.float),  # input label is int for metric purpose
                reduction="mean",
            )
        return loss


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
            out_size=1,
            **self.hparams.decoder,
        )

    def forward(self, x):
        graph, global_features = x
        embedding = self.encoder(graph)
        if global_features is not None:
            y = self.decoder(torch.cat((embedding, global_features), dim=1))
        else:
            y = self.decoder(embedding)

        return y

    def _get_preds(self, batch):
        graph_batch, global_features_batch, y = batch
        y_hat = self((graph_batch, global_features_batch))

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
        self.init_pooling()

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
            self.pooling = GlobalAttentionPooling(
                gate_nn=nn.Linear(self.hparams["encoder"]["hidden_size"], 1),
                ntype="_N",
                feat="h_v",
                get_attention=True,
            )
        elif self.hparams["encoder"]["aggregation"] == "mean":
            self.pooling = AvgPooling(ntype="_N", feat="h_v")
        elif self.hparams["encoder"]["aggregation"] == "sum":
            self.pooling = SumPooling(ntype="_N", feat="h_v")
        elif self.hparams["encoder"]["aggregation"] == "max":
            self.pooling = MaxPooling(ntype="_N", feat="h_v")
        else:
            raise ValueError(
                "Aggregation must be one of ['max', 'mean', 'sum', 'attention']"
            )

    def init_decoder(self):
        return FFN(
            in_size=self.hparams.encoder["hidden_size"]
            + self.hparams["global_feature_size"],
            out_size=1,
            **self.hparams.decoder,
        )

    def forward(self, x):
        graph, global_features = x
        embedding = self.encoder(graph, graph.ndata["x"])
        with graph.local_scope():
            graph.ndata["h_v"] = embedding
            if self.hparams["encoder"]["aggregation"] == "attention":
                embedding_pooled, attention = self.pooling(graph)
            else:
                embedding_pooled = self.pooling(graph)

        if global_features is not None:
            y = self.decoder(torch.cat((embedding_pooled, global_features), dim=1))
        else:
            y = self.decoder(embedding_pooled)

        return y

    def _get_preds(self, batch):
        graph_batch, global_features_batch, y = batch
        y_hat = self((graph_batch, global_features_batch))

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
            out_size=1,
            **self.hparams.decoder,
        )

    def forward(self, x):
        return self.decoder(x)

    def _get_preds(self, batch):
        _, global_features_batch, _ = batch
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
            out_size=1,
            **self.hparams.decoder,
        )

    def forward(self, x):
        graph, global_features = x
        embedding = self.encoder(graph)
        if global_features is not None:
            y = self.decoder(torch.cat((embedding, global_features), dim=1))
        else:
            y = self.decoder(embedding)
        return y

    def _get_preds(self, batch):
        graph_batch, global_features_batch, _ = batch
        y_hat = self((graph_batch, global_features_batch))
        return y_hat
