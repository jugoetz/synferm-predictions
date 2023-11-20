from copy import deepcopy
from unittest import TestCase
import unittest

import torch
from sklearn.metrics import balanced_accuracy_score
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader

from src.model.metrics import BalancedAccuracy
from src.evaluate import calculate_metrics
from src.model.classifier import DMPNNModel
from src.data.dataloader import SynFermDataset, collate_fn
from src.util.definitions import DATA_ROOT, LOG_DIR


class TestBinaryBalancedAccuracy(TestCase):
    def setUp(self):
        self.metric = BalancedAccuracy(task="binary")

    def test_basic(self):
        true = torch.tensor([0, 0, 1, 1])
        pred = torch.tensor([0.6, 0.4, 0.9, 0.9])
        value_sklearn = balanced_accuracy_score(true, pred > 0.5)
        value_metrics = self.metric(pred, true)
        # print(value_sklearn)
        # print(value_metrics)
        self.assertAlmostEqual(value_metrics.item(), value_sklearn)

    def test_works_with_cuda_tensor(self):
        if torch.cuda.is_available():
            true = torch.tensor([0, 0, 1, 1], device="cuda")
            pred = torch.tensor([0.6, 0.4, 0.9, 0.9], device="cuda")
            self.assertAlmostEqual(
                self.metric.to("cuda")(pred, true).item(),
                balanced_accuracy_score(true.cpu(), pred.cpu() > 0.5),
            )
        else:
            print("CUDA not found, skipping GPU test")

    def test_values_against_sklearn(self):
        for _ in range(1000):
            true = torch.randint(0, 2, (15,))
            pred = torch.rand((15,))

            value_sklearn = balanced_accuracy_score(true, pred > 0.5)
            value_metrics = self.metric(pred, true).item()

            with self.subTest(
                msg="May fail if y_true does not contain a class that is contained in y_pred (undefined case)",
                true=true.tolist(),
                pred=pred.tolist(),
            ):
                self.assertAlmostEqual(value_sklearn, value_metrics, places=4)


class TestMulticlassBalancedAccuracy(TestCase):
    def setUp(self):
        self.metric = BalancedAccuracy(task="multiclass", num_classes=4)

    def test_basic(self):
        true = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
        pred = torch.tensor(
            [
                [0.7, 0.1, 0.1, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.7, 0.1, 0.1],
            ]
        )
        value_sklearn = balanced_accuracy_score(true, torch.argmax(pred, dim=1))
        self.metric.update(pred, true)
        value_metrics = self.metric.compute()
        # print(value_sklearn)
        # print(value_metrics)
        self.assertTrue(
            torch.isclose(
                torch.tensor(value_sklearn, dtype=torch.float), value_metrics
            ).all()
        )

    def test_works_with_cuda_tensor(self):
        if torch.cuda.is_available():
            true = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device="cuda")
            pred = torch.tensor(
                [
                    [0.7, 0.1, 0.1, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.1, 0.1, 0.7, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.1, 0.1, 0.7, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                    [0.1, 0.1, 0.7, 0.1],
                    [0.1, 0.7, 0.1, 0.1],
                ],
                device="cuda",
            )
            self.assertAlmostEqual(
                self.metric.to("cuda")(pred, true).item(),
                balanced_accuracy_score(true.cpu(), torch.argmax(pred, dim=1).cpu()),
            )
        else:
            print("CUDA not found, skipping GPU test")

    def test_values_against_sklearn(self):
        for _ in range(1000):
            true = torch.randint(0, 4, (25,))
            pred = torch.rand((25, 4))
            value_sklearn = balanced_accuracy_score(true, torch.argmax(pred, dim=1))
            value_metrics = self.metric(pred, true).item()

            # skip if true does not have
            with self.subTest(
                msg="May fail if y_true does not contain a class that is contained in y_pred (undefined case)",
                true=true.tolist(),
                pred=pred.tolist(),
            ):
                self.assertAlmostEqual(value_sklearn, value_metrics, places=4)


class TestMultilabelBalancedAccuracy(TestCase):
    def setUp(self):
        self.metric = BalancedAccuracy(task="multilabel", num_labels=3)

    def test_basic(self):
        num_labels = 3
        true = torch.tensor([[0, 0, 1], [1, 0, 1], [0, 0, 0], [1, 1, 1]])
        pred = torch.tensor(
            [[0.1, 0.1, 0.9], [0.9, 0.9, 0.1], [0.1, 0.9, 0.1], [0.9, 0.9, 0.9]]
        )
        pred_thresh = (pred > 0.5).int()
        value_sklearn = [
            balanced_accuracy_score(true[:, i], pred_thresh[:, i])
            for i in range(num_labels)
        ]
        self.metric.update(pred, true)
        value_metrics = self.metric.compute()
        # print(value_sklearn)
        # print(value_metrics)
        self.assertTrue(
            torch.isclose(
                torch.tensor(value_sklearn, dtype=torch.float), value_metrics
            ).all()
        )

    def test_works_with_cuda_tensor(self):
        if torch.cuda.is_available():
            true = torch.tensor(
                [[0, 0, 1], [1, 0, 1], [0, 0, 0], [1, 1, 1]], device="cuda"
            )
            pred = torch.tensor(
                [[0.1, 0.1, 0.9], [0.9, 0.9, 0.1], [0.1, 0.9, 0.1], [0.9, 0.9, 0.9]],
                device="cuda",
            )
            pred_thresh = (pred > 0.5).int().cpu()
            value_sklearn = [
                balanced_accuracy_score(true[:, i].cpu(), pred_thresh[:, i])
                for i in range(3)
            ]

            self.assertTrue(
                torch.isclose(
                    torch.tensor(value_sklearn, dtype=torch.float),
                    self.metric.to("cuda")(pred, true).cpu(),
                ).all()
            )
        else:
            print("CUDA not found, skipping GPU test")

    def test_values_against_sklearn(self):
        for _ in range(1000):
            true = torch.randint(0, 2, (15, 3))
            pred = torch.rand((15, 3))
            pred_thresh = (pred > 0.5).int()
            value_sklearn = [
                balanced_accuracy_score(true[:, i], pred_thresh[:, i]) for i in range(3)
            ]
            value_metrics = self.metric(pred, true)

            # skip if true does not have
            with self.subTest(
                msg="May fail if y_true does not contain a class that is contained in y_pred (undefined case)",
                true=true.tolist(),
                pred=pred.tolist(),
            ):
                self.assertTrue(
                    torch.isclose(
                        torch.tensor(value_sklearn, dtype=torch.float), value_metrics
                    ).all()
                )


class TestEvaluate(TestCase):
    """
    Test whether the calculate_metrics function in evaluate.py delivers the same result as the torchmetrics metrics
    on the classifier.
    For this, we train a model and compare the metrics on its validation set on the last epoch.
    """

    def setUp(self) -> None:
        self.maxDiff = None
        data = SynFermDataset(
            name="synferm_dataset_2023-09-05_40018records.csv",
            raw_dir=DATA_ROOT,
            save_dir=(DATA_ROOT / "cache"),
            reaction=True,
            smiles_columns=["reaction_smiles_atom_mapped"],
            label_columns=["binary_A", "binary_B", "binary_C"],
            graph_type="bond_nodes",
            global_features=["None"],
            global_features_file="None",
            featurizers="custom",
            task="multilabel",
            force_reload=False,
        )
        train_dl = DataLoader(
            [data[i] for i in range(1024)],
            batch_size=128,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        val_dl = DataLoader(
            [data[i] for i in range(1024, 2048)],
            batch_size=128,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
        )
        model = DMPNNModel(
            atom_feature_size=data.atom_feature_size,
            bond_feature_size=data.bond_feature_size,
            global_feature_size=data.global_feature_size,
            num_labels=data.num_labels,
            encoder={
                "depth": 5,
                "hidden_size": 300,
                "dropout_ratio": 0.2,
                "aggregation": "sum",
            },
            decoder={"depth": 3, "hidden_size": 32, "dropout_ratio": 0.2},
            training={"task": "multilabel"},
            target_names=["binary_A", "binary_B", "binary_C"],
            optimizer={
                "lr": 0.0005,
                "weight_decay": 0,
                "lr_scheduler": {
                    "epochs": 100,
                    "lr_min": 0.00005,
                    "lr_warmup_step": 2,
                    "scheduler_name": "exp_with_linear_warmup",
                },
            },
            run_id="test1234",
        )
        trainer = Trainer(
            accelerator="gpu",
            max_epochs=10,
            log_every_n_steps=20,
            default_root_dir=LOG_DIR / "checkpoints",
            logger=False,
            enable_progress_bar=False,
        )

        trainer.fit(model=model, train_dataloaders=train_dl, val_dataloaders=val_dl)

        self.val_preds = torch.concatenate(model.val_predictions, dim=0).cpu()
        self.val_truth = torch.stack([i[3] for i in val_dl.dataset], dim=0).cpu()

        model_metrics = model.val_metrics.compute()
        # in the multilabel case, we receive metrics for each label
        # we separate them
        for k, v in deepcopy(model_metrics).items():
            if len(v) == model.hparams["num_labels"]:
                for i, c in enumerate(model.hparams["target_names"]):
                    model_metrics[f"{k}_target_{c}"] = model_metrics[k][i].item()
                model_metrics[f"{k}_macro"] = torch.mean(model_metrics[k]).item()
                del model_metrics[k]
        self.model_metrics = model_metrics

        self.eval_metrics = {
            f"val/{k}": v
            for k, v in calculate_metrics(
                self.val_truth,
                self.val_preds,
                "multilabel",
                ["binary_A", "binary_B", "binary_C"],
            ).items()
        }

    def test_model_metrics_and_eval_metrics_use_same_keys(self):
        """
        Test that we receive the same keys in both metric dicts.
        We skip keys with "loss", because loss is handled separately from metrics in the classifier implementation.
        """
        self.assertListEqual(
            sorted(self.model_metrics.keys()),
            sorted([k for k in self.eval_metrics.keys() if not "loss" in k]),
        )

    def test_model_metrics_equal_eval_metrics(self):
        for k in self.eval_metrics.keys():
            try:
                model_metric = self.model_metrics[k]
                with self.subTest(metric=k):
                    self.assertAlmostEqual(model_metric, self.eval_metrics[k], places=4)
            except KeyError:
                print(f"Skipped {k}, which is not contained in model metrics")


if __name__ == "__main__":
    unittest.main()