import pickle

import wandb
import pytorch_lightning as pl

from src.evaluate import calculate_metrics
from src.model.callbacks import LogMetricsCallback, BestValLossEpochCallback
from src.model.classifier import load_model
from src.model.sklearnmodels import load_sklearn_model
from src.util.definitions import LOG_DIR, CKPT_DIR
from src.util.logging import generate_run_id, concatenate_to_dict_keys


def train(
    train_dl,
    val_dl,
    hparams,
    test_dls=None,
    run_id=None,
    run_group=None,
    return_metrics=False,
    tags=None,
    job_type=None,
):
    """
    Trains a model on given data with one set of hyperparameters. Training, validation, and, optionally, test metrics
    (as specified in the model class) are logged to wandb.

    Args:
        train_dl (torch.utils.data.DataLoader): Dataloader with training data.
        val_dl (torch.utils.data.DataLoader): Dataloader with validation data.
        hparams (dict): Model hyperparameters.
        test_dls (optional, dict): Dictionary of dataloaders with test data. If given, test metrics will be returned.
        run_id (optional, str): Unique id to identify the run. If None, will generate an ID containing the current
            datetime. Defaults to None.
        run_group (optional, str): Name to identify the run group. Default None.
        return_metrics (bool, optional): Whether to return train and val metrics. Defaults to False.
        tags (list, optional): List of tags to add to the run in wandb. Defaults to None.
        job_type (str, optional): Type of job for wandb. Defaults to None.

    Returns:
        str: run_id that identifies the run/model
        dict: Dictionary of training and validation (+ optionally test) metrics.
            Only returned if return_metrics is True.
    """
    # generate run_id if None is passed
    if not run_id:
        run_id = generate_run_id()

    # set run group name
    if not run_group:
        run_group = "single_run"

    # set up trainer

    checkpoint_callback_last = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor="epoch",
        mode="max",
        dirpath=CKPT_DIR / run_id,
        filename="last-epoch{epoch:02d}-val_loss{val/loss:.2f}",
        auto_insert_metric_name=False,
    )
    metrics_callback = LogMetricsCallback()

    trainer = pl.Trainer(
        max_epochs=hparams["training"]["max_epochs"],
        log_every_n_steps=1,
        default_root_dir=LOG_DIR / "checkpoints",
        accelerator=hparams["accelerator"],
        callbacks=[checkpoint_callback_last, metrics_callback],
    )

    wandb.init(
        reinit=True,
        project="slap-gnn",
        name=run_id,
        group=run_group,
        config=hparams,
        tags=tags,
        job_type=job_type,
    )

    model = load_model(hparams)

    # run training
    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    # get the metrics for all epochs
    metrics = metrics_callback.metrics

    # optionally, run test
    if test_dls:
        for test_name, test_dl in test_dls.items():
            trainer.test(
                model, test_dl, ckpt_path=checkpoint_callback_last.best_model_path
            )
            for k, v in trainer.logged_metrics.items():
                if k.startswith("test"):
                    metrics[k.replace("test", test_name)] = v

    # for the return metrics, we want to return the metrics for the last epoch not all epochs
    metrics = {
        k: v[-1] if v.dim() == 1 else v for k, v in metrics.items()
    }  # some metrics are only a single value so we don't want to index them

    # log the best metrics to wandb
    wandb.log(metrics)
    wandb.finish()

    if return_metrics:
        return run_id, metrics
    else:
        return run_id


def train_sklearn(
    train,
    val,
    hparams,
    test=None,
    run_id=None,
    run_group=None,
    return_metrics=False,
    tags=None,
    job_type=None,
):
    """
    Trains a sklearn model on a given data split with one set of hyperparameters. By default, returns the evaluation
    metrics on the validation set.

    Args:
        train (torch.utils.data.DataLoader): Training data
        val: (torch.utils.data.DataLoader): Validation data
        hparams (dict): Model hyperparameters
        test: (Union[DataLoader, Dict[torch.utils.data.DataLoader]], optional): Test data. If data is given, test metrics will be returned.
        run_id (optional, str): Unique id to identify the run. If None, will generate an ID containing the current datetime.
            Defaults to None.
        run_group (optional, str): Id to identify the run group. Default None.
        return_metrics (bool, optional): Whether to return train and val metrics. Defaults to False.
        tags (list, optional): List of tags to add to the run in wandb. Defaults to None.
        job_type (str, optional): Type of job for wandb. Defaults to None.

    Returns:
        str: run_id that identifies the run/model
        dict: Dictionary of training and validation (+ optionally test) metrics.
            Only returned if return_metrics is True.
    """
    # generate run_id if None is passed
    if not run_id:
        run_id = generate_run_id()

    # set run group name
    if not run_group:
        run_group = "single_run"

    wandb.init(
        reinit=True,
        project="slap-gnn",
        name=run_id,
        group=run_group,
        config=hparams,
        tags=tags,
        job_type=job_type,
    )

    # initialize model
    model = load_sklearn_model(hparams)

    # get training and validation data
    train_graphs, train_global_features, train_labels = map(list, zip(*train))
    val_graphs, val_global_features, val_labels = map(list, zip(*val))

    # run training
    model.fit(train_global_features, train_labels)

    # evaluate on training set
    train_pred = model.predict_proba(train_global_features)
    train_metrics = concatenate_to_dict_keys(
        calculate_metrics(train_labels, train_pred, pred_proba=True), prefix="train/"
    )

    # evaluate on validation set
    val_pred = model.predict_proba(val_global_features)
    val_metrics = concatenate_to_dict_keys(
        calculate_metrics(val_labels, val_pred, pred_proba=True), prefix="val/"
    )

    # logging metrics
    metrics = {}
    metrics.update(train_metrics)
    metrics.update(val_metrics)

    # save model
    model_path = CKPT_DIR / run_id / "model.pkl"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    # optionally, run test set
    if test:
        test_metrics = {}
        for k, v in test.items():
            test_graphs, test_global_features, test_labels = map(list, zip(*v))
            test_pred = model.predict_proba(test_global_features)
            test_metrics.update(
                concatenate_to_dict_keys(
                    calculate_metrics(test_labels, test_pred, pred_proba=True), f"{k}/"
                )
            )

        # add to logging metrics
        metrics.update(test_metrics)

    wandb.log(metrics)
    wandb.finish()

    if return_metrics:
        return run_id, metrics
    else:
        return run_id



    
