from torchmetrics import Metric
import torch


class BalancedAccuracy:
    """Factory to deploy task-specific version of BalancedAccuracy"""

    def __new__(cls, *args, **kwargs):
        task = kwargs.get("task")
        if task == "binary":
            return BinaryBalancedAccuracy(*args, **kwargs)
        elif task == "multiclass":
            return MulticlassBalancedAccuracy(*args, **kwargs)
        elif task == "multilabel":
            return MultilabelBalancedAccuracy(*args, **kwargs)
        else:
            raise ValueError(f"Unknown task: {task}.")


class BinaryBalancedAccuracy(Metric):
    """
    Balanced accuracy is the average recall per class.
    In the binary case this is the arithmetic mean of sensitivity and specificity.
    Predictions are assigned the positive class if probability > `threshold`. The default `threshold` is 0.5.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(self, threshold=0.5, **kwargs):
        super().__init__()
        self.threshold = threshold
        self.add_state("tp", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(1), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(1), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds (torch.Tensor): IntTensor of shape (n_samples) if classes are given or FloatTensor of shape (n_samples) if probabilities are given.
        target (torch.Tensor): IntTensor of shape (n_samples)
        """
        if preds.is_floating_point():
            preds = (preds > self.threshold).int()
        assert preds.shape == target.shape

        self.tp += torch.sum((preds == target) & (target == 1), dim=0)
        self.fp += torch.sum((preds != target) & (target == 0), dim=0)
        self.fn += torch.sum((preds != target) & (target == 1), dim=0)
        self.tn += torch.sum((preds == target) & (target == 0), dim=0)

    def compute(self):
        # n.b. the metric is not properly defined if (tp + tn) == 0.
        # This case is handled differently by sklearn.metrics and torchmetrics
        # sklearn excludes classes where this happens before averaging, torchmetrics sets them to 0.
        # This leads to different aggregate results. We follow the torchmetrics convention

        return torch.nan_to_num(
            0.5
            * (
                self.tp.float() / (self.tp + self.fn)
                + self.tn.float() / (self.tn + self.fp)
            )
        )


class MulticlassBalancedAccuracy(Metric):
    """
    Balanced accuracy is the average recall per class.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(self, num_classes: int, **kwargs):
        super().__init__()
        self.num_classes = num_classes
        self.add_state("tp", default=torch.zeros(num_classes), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_classes), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds (torch.Tensor): IntTensor of shape (n_samples) if classes are given or FloatTensor of shape (n_samples, n_classes) if probabilities are given.
        target (torch.Tensor): IntTensor of shape (n_samples)
        """
        if preds.is_floating_point():
            preds = torch.argmax(preds, dim=1)
        assert preds.shape == target.shape

        for class_idx in range(self.num_classes):
            self.tp[class_idx] += torch.sum(
                (preds == target) & (target == class_idx), dim=0
            )
            self.fn[class_idx] += torch.sum(
                (preds != target) & (target == class_idx), dim=0
            )

    def compute(self):
        class_wise = torch.nan_to_num(self.tp.float() / (self.tp + self.fn))

        # n.b. the metric is not properly defined if (tp + tn) == 0.
        # This case is handled differently by sklearn.metrics and torchmetrics
        # sklearn excludes classes where this happens before averaging, torchmetrics sets them to 0.
        # This leads to different aggregate results. We follow the torchmetrics convention
        return torch.mean(class_wise)


class MultilabelBalancedAccuracy(Metric):
    """
    Balanced accuracy is the average recall per class (for each label).
    If average="none" (or None), balanced accuracy is returned for each label.
    If average="macro", the mean of balanced accuracy over all labels is returned.

    Predictions are assigned the positive class if probability > `threshold`. The default `threshold` is 0.5.
    """

    is_differentiable = False
    higher_is_better = True
    full_state_update = True

    def __init__(self, num_labels: int, threshold=0.5, average="none", **kwargs):
        super().__init__()
        self.num_labels = num_labels
        self.threshold = threshold
        self.average = average
        self.add_state("tp", default=torch.zeros(num_labels), dist_reduce_fx="sum")
        self.add_state("fp", default=torch.zeros(num_labels), dist_reduce_fx="sum")
        self.add_state("fn", default=torch.zeros(num_labels), dist_reduce_fx="sum")
        self.add_state("tn", default=torch.zeros(num_labels), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        """
        preds (torch.Tensor): IntTensor of shape (n_samples, n_labels) if classes are given or FloatTensor of shape (n_samples, n_labels) if probabilities are given.
        target (torch.Tensor): IntTensor of shape (n_samples, n_labels)
        """
        if preds.is_floating_point():
            preds = (preds > self.threshold).int()
        assert preds.shape == target.shape
        assert preds.shape[1] == self.num_labels

        self.tp += torch.sum((preds == target) & (target == 1), dim=0)
        self.fp += torch.sum((preds != target) & (target == 0), dim=0)
        self.fn += torch.sum((preds != target) & (target == 1), dim=0)
        self.tn += torch.sum((preds == target) & (target == 0), dim=0)

    def compute(self):
        class_wise = torch.nan_to_num(
            0.5
            * (
                self.tp.float() / (self.tp + self.fn)
                + self.tn.float() / (self.tn + self.fp)
            )
        )
        # n.b. the metric is not properly defined if (tp + tn) == 0.
        # This case is handled differently by sklearn.metrics and torchmetrics
        # sklearn excludes classes where this happens before averaging, torchmetrics sets them to 0.
        # This leads to different aggregate results. We follow the torchmetrics convention
        if self.average == "none" or self.average is None:
            return class_wise
        elif self.average == "macro":
            return torch.mean(class_wise)
        else:
            raise ValueError("average should be one of {'none', None, 'macro'}")
