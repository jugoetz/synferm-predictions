from unittest import TestCase
import unittest

import torch
from sklearn.metrics import balanced_accuracy_score

from src.model.metrics import BalancedAccuracy


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


if __name__ == "__main__":
    unittest.main()
