import pytorch_lightning as pl
import torch


def predict(model, data, hparams, return_proba=False):
    """
    Predicts on given data with given model.

    Args:
        model (Classifier): Trained model
        data (torch.utils.data.DataLoader): DataLoader with data for prediction.

    Returns:
        torch.tensor: Predicted labels
    """
    model.eval()
    trainer = pl.Trainer(
        accelerator=hparams["accelerator"], logger=False, max_epochs=-1
    )
    probs = torch.cat(trainer.predict(model, data))
    if return_proba:
        return probs
    else:
        # obtain labels from probabilities at 0.5 threshold
        labels = (probs > 0.5).int()
        return labels
