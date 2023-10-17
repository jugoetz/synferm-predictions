from copy import deepcopy
from typing import Tuple, List

import pytorch_lightning as pl
import torch
import torchmetrics as tm
from dgllife.model.gnn.attentivefp import AttentiveFPGNN
from dgllife.model.gnn.gcn import GCN
from dgllife.model.gnn.graphsage import GraphSAGE

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
from src.model.metrics import BalancedAccuracy
from src.util.io import save_predictions


def load_model(hparams):
    if hparams["name"] == "D-MPNN":
        model = DMPNNModel(**hparams)
    elif hparams["name"] == "GCN":
        model = GCNModel(**hparams)
    elif hparams["name"] == "AttentiveFP":
        model = AttentiveFPModel(**hparams)
    elif hparams["name"] == "GraphSAGE":
        model = GraphSAGEModel(**hparams)
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
    elif model_type == "AttentiveFP":
        model = AttentiveFPModel.load_from_checkpoint(checkpoint_path)
    elif model_type == "GraphSAGE":
        model = GraphSAGEModel.load_from_checkpoint(checkpoint_path)
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

        average = (
            "micro" if self.hparams["training"]["task"] == "multiclass" else "none"
        )

        # TODO there is an upstream problem with torchmetrics
        #  the binary and multilabel accuracies assume they are receiving logits if any value is outside [0,1]
        #  otherwise they assume probabilities.
        #  Of course this is a stupid assumption.
        #  Hopefully soon fixed upstream, see https://github.com/Lightning-AI/torchmetrics/issues/1604
        #  and https://github.com/Lightning-AI/torchmetrics/pull/1676
        #  meantime it's either "Prinzip Hoffnung" or converting everything to probabilities to avoid the problem

        metrics = tm.MetricCollection(
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
                "avgPrecision": tm.AveragePrecision(
                    task=self.hparams["training"]["task"],
                    num_labels=kwargs["num_labels"],
                    num_classes=kwargs["num_labels"],
                    average=average,
                ),
                "balanced_accuracy": BalancedAccuracy(
                    task=self.hparams["training"]["task"],
                    num_labels=kwargs["num_labels"],
                    num_classes=kwargs["num_labels"],
                    threshold=0.5,  # applies for binary and multilabel
                    average=None,  # applies for multilabel
                ),
            }
        )

        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

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

    def _get_preds(self, batch) -> torch.FloatTensor:
        """
        This method returns prediction logits.
        Typically, it only wraps the forward() method of the model, but it can contain additional necessary logic to
        allow for a standard call signature.
        E.g. the FFN implementation will not pass the graphs to the forward method.
        """
        raise NotImplementedError("Child class must implement this method")

    def _get_preds_loss(
        self, batch, return_proba=False
    ) -> Tuple[List[int], torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
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

        # calculate loss from logits
        loss = self.loss(preds, targets)

        # convert logits to probabilities if desired
        if return_proba:
            if self.hparams["training"]["task"] == "multiclass":
                # for multiclass, we apply softmax to get class probabilities
                preds = torch.softmax(preds, dim=1)
            else:
                # for multilabel and binary, we apply sigmoid to get the (label-wise) binary probability for the pos class
                preds = torch.sigmoid(preds)

        return indices, preds, loss, targets

    def training_step(self, batch, batch_idx):
        indices, preds, loss, targets = self._get_preds_loss(batch, return_proba=True)
        self.train_metrics.update(preds, targets.int())
        self.log("train/loss", loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        indices, preds, loss, targets = self._get_preds_loss(batch, return_proba=True)
        self.val_metrics.update(preds, targets.int())
        self.log("val/loss", loss, on_step=False, on_epoch=True)
        self.val_indices.append(indices)
        self.val_predictions.append(preds)
        return loss

    def test_step(self, batch, batch_idx):
        indices, preds, loss, targets = self._get_preds_loss(batch, return_proba=True)
        self.test_metrics.update(preds, targets.int())
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.test_indices.append(indices)
        self.test_predictions.append(preds)
        return loss

    def on_train_epoch_start(self) -> None:
        # we do this on start instead of end so that the last state will be available in the trained model
        # reset metrics
        self.train_metrics.reset()

    def on_train_epoch_end(self) -> None:
        # log metrics
        metrics = self.train_metrics.compute()

        # in the multilabel case, we receive metrics for each label
        # we separate them
        if self.hparams["training"]["task"] == "multilabel":
            for k, v in deepcopy(metrics).items():
                if len(v) == self.hparams["num_labels"]:
                    for i, c in enumerate(self.hparams["target_names"]):
                        metrics[f"{k}_target_{c}"] = metrics[k][i]
                    metrics[f"{k}_macro"] = torch.mean(metrics[k])
                    del metrics[k]

        self.log_dict(metrics)

    def on_validation_epoch_start(self) -> None:
        # reset stored predictions
        # we do this on start instead of end so that the last state will be available in the trained model
        self.val_indices = []
        self.val_predictions = []
        # reset metrics
        self.val_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        # log metrics
        metrics = self.val_metrics.compute()

        # in the multilabel case, we receive metrics for each label
        # we separate them
        if self.hparams["training"]["task"] == "multilabel":
            for k, v in deepcopy(metrics).items():
                if len(v) == self.hparams["num_labels"]:
                    for i, c in enumerate(self.hparams["target_names"]):
                        metrics[f"{k}_target_{c}"] = metrics[k][i]
                    metrics[f"{k}_macro"] = torch.mean(metrics[k])
                    del metrics[k]

        self.log_dict(metrics)

        # save the predictions
        save_predictions(
            self.hparams["run_id"], self.val_indices, self.val_predictions, "val"
        )

    def on_test_epoch_start(self) -> None:
        # reset stored predictions
        # we do this on start instead of end so that the last state will be available in the trained model
        self.test_indices = []
        self.test_predictions = []
        # reset metrics
        self.test_metrics.reset()

    def on_test_epoch_end(self) -> None:
        # log metrics
        metrics = self.test_metrics.compute()

        # in the multilabel case, we receive metrics for each label
        # we separate them
        if self.hparams["training"]["task"] == "multilabel":
            for k, v in deepcopy(metrics).items():
                if len(v) == self.hparams["num_labels"]:
                    for i, c in enumerate(self.hparams["target_names"]):
                        metrics[f"{k}_target_{c}"] = metrics[k][i]
                    metrics[f"{k}_macro"] = torch.mean(metrics[k])
                    del metrics[k]

        self.log_dict(metrics)

        # save the predictions
        save_predictions(
            self.hparams["run_id"], self.test_indices, self.test_predictions, "test"
        )

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
        return self(batch)


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
        return self(batch)


class AttentiveFPModel(Classifier):
    """
    AttentiveFP model as implemented in DGL-Lifesci.

    Note that this model works on a molecular graph and that its forward method expects a BE-graph input.

    There are some differences in the architecture of dgllife models and my own.
    Mainly:
        - dgllife encoder does not take care of pooling
        - dgllife encoder .forward() needs the feature vector to be passed separately
        - dgllife encoder expects a homogeneous graph
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pooling = self.init_pooling()

    def init_encoder(self):
        return AttentiveFPGNN(
            node_feat_size=self.hparams["atom_feature_size"],
            edge_feat_size=self.hparams["bond_feature_size"],
            num_layers=self.hparams["encoder"]["depth"],
            graph_feat_size=self.hparams["encoder"]["hidden_size"],
            dropout=self.hparams["encoder"]["dropout_ratio"],
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
        embedding_before_pooling = self.encoder(
            graph, graph.ndata["x"], graph.edata["e"]
        )
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
        return self(batch)


class GraphSAGEModel(Classifier):
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
        return GraphSAGE(
            in_feats=self.hparams["atom_feature_size"],
            hidden_feats=[
                self.hparams["encoder"]["hidden_size"]
                for _ in range(self.hparams["encoder"]["depth"])
            ],
            activation=[
                get_activation(self.hparams["encoder"]["activation"])
                for _ in range(self.hparams["encoder"]["depth"])
            ],
            dropout=[
                self.hparams["encoder"]["dropout_ratio"]
                for _ in range(self.hparams["encoder"]["depth"])
            ],
            aggregator_type=["mean" for _ in range(self.hparams["encoder"]["depth"])],
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
        return self(batch)


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
        return self(global_features_batch)


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
        return self(batch)
