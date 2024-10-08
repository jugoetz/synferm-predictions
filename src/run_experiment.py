from torch.utils.data import DataLoader

from src.data.dataloader import (
    SynFermDataset,
    GraphLessSynFermDataset,
    collate_fn,
    graphless_collate_fn,
)
from src.util.definitions import LOG_DIR
from src.util.io import walk_split_directory
from src.util.logging import generate_run_id
from src.cross_validation import cross_validate_sklearn, cross_validate
from src.predict import predict
from src.model.classifier import load_trained_model
from src.hyperopt import optimize_hyperparameters_bayes


def run_training(args, hparams):
    """
    Handles training and hyperparameter optimization.
    """

    # based on the model type, we will use different data set classes and training functions
    # the major advantage is that for the models that do not require graph data,
    # we can skip the expensive graph building
    if hparams["name"] in [
        "D-MPNN",
        "GCN",
        "AttentiveFP",
        "GraphSAGE",
        "GraphAgnostic",
    ]:
        hparams["model_type"] = "torch_graph"  # graph data set + torch training
    elif hparams["name"] in ["FFN"]:
        hparams[
            "model_type"
        ] = "torch_nongraph"  # data set without graphs + torch training
    elif hparams["name"] in ["LogisticRegression", "XGB"]:
        hparams["model_type"] = "sklearn"  # data set without graphs + sklearn training
    else:
        raise ValueError(f"Unknown model type {hparams['name']}")

    if hparams["model_type"] == "torch_graph":
        # load data
        data = SynFermDataset(
            name=args.data_path.name,
            raw_dir=args.data_path.parent,
            save_dir=(args.data_path.parent / "cache"),
            reaction=hparams["encoder"]["reaction"],
            smiles_columns=args.smiles_columns,
            label_columns=args.label_columns,
            graph_type=hparams["encoder"]["graph_type"],
            global_features=hparams["decoder"]["global_features"],
            global_features_file=hparams["decoder"]["global_features_file"],
            featurizers=hparams["encoder"]["featurizers"],
            task=hparams["training"]["task"],
            force_reload=args.force_reload,
        )
    elif hparams["model_type"] in ["torch_nongraph", "sklearn"]:
        data = GraphLessSynFermDataset(
            name=args.data_path.name,
            raw_dir=args.data_path.parent,
            save_dir=(args.data_path.parent / "cache"),
            smiles_columns=args.smiles_columns,
            label_columns=args.label_columns,
            global_features=hparams["decoder"]["global_features"],
            global_features_file=hparams["decoder"]["global_features_file"],
            task=hparams["training"]["task"],
            force_reload=args.force_reload,
        )
    else:
        raise ValueError(f"Unknown model type {hparams['model_type']}")

    # update config with data processing specifics
    hparams["atom_feature_size"] = data.atom_feature_size
    hparams["bond_feature_size"] = data.bond_feature_size
    hparams["global_feature_size"] = data.global_feature_size
    hparams["num_labels"] = data.num_labels
    hparams["label_binarizer"] = data.label_binarizer
    hparams["target_names"] = args.label_columns
    hparams["data_hash_key"] = data.hash
    if data.global_featurizer_state_dict_path:  # only in case of OHE
        hparams["global_featurizer_state_dict_path"] = str(
            data.global_featurizer_state_dict_path
        )  # update with generated path
        print(
            f"OneHotEncoder state dict saved to {data.global_featurizer_state_dict_path}"
        )

    # define split index files
    if args.split_indices:
        split_files = walk_split_directory(args.split_indices)
        strategy = "predefined"
    elif args.cv > 1:
        strategy = "KFold"
        split_files = None
    elif args.train_size:
        strategy = "random"
        split_files = None
    else:
        raise ValueError(
            "One of `--split-indices`, `--cv`, or `--train-size` must be given."
        )

    job_type = args.overwrite_job_type
    # run hyperparameter optimization if requested
    if args.hparam_optimization:
        if job_type is None:
            job_type = "hparam_optimization"
        # run bayesian hparam optimization
        hparams = optimize_hyperparameters_bayes(
            data=data,
            hparams=hparams,
            hparam_config_path=args.hparam_config_path,
            cv_parameters={
                "strategy": strategy,
                "split_files": split_files,
                "n_folds": args.cv,
                "train_size": args.train_size,
                "tags": args.tags,
                "job_type": job_type,
            },
            n_iter=args.hparam_n_iter,
        )

    # run cross-validation with preconfigured or optimized hparams
    if job_type == "hparam_optimization":
        job_type = "hparam_best"
    elif job_type is None:
        job_type = "training"

    if hparams["model_type"].startswith("torch"):
        aggregate_metrics, fold_metrics = cross_validate(
            data,
            hparams,
            strategy=strategy,
            n_folds=args.cv,
            train_size=args.train_size,
            split_files=split_files,
            return_fold_metrics=True,
            run_test=args.run_test,
            tags=args.tags,
            job_type=job_type,
        )
    elif hparams["model_type"].startswith("sklearn"):
        aggregate_metrics, fold_metrics = cross_validate_sklearn(
            data,
            hparams,
            strategy=strategy,
            n_folds=args.cv,
            train_size=args.train_size,
            split_files=split_files,
            return_fold_metrics=True,
            run_test=args.run_test,
            tags=args.tags,
            job_type=job_type,
        )

    if args.hparam_optimization:
        print(f"Optimized hyperparameters: {hparams}")
    print("Metrics aggregated over all splits:")
    for k, v in aggregate_metrics.items():
        print(f"{k}: {v}")

    return


def run_prediction(args, hparams):
    """
    Handles prediction from a trained model.
    """

    if hparams["model_type"] == "torch_graph":
        # load data
        data = SynFermDataset(
            name=args.data_path.name,
            raw_dir=args.data_path.parent,
            reaction=hparams["encoder"]["reaction"],
            smiles_columns=args.smiles_columns,
            label_columns=None,
            graph_type=hparams["encoder"]["graph_type"],
            global_features=hparams["decoder"]["global_features"],
            featurizers=hparams["encoder"]["featurizers"],
            task=hparams["training"]["task"],
        )

        collate = collate_fn
    elif hparams["model_type"] in ["torch_nongraph", "sklearn"]:
        data = GraphLessSynFermDataset(
            name=args.data_path.name,
            raw_dir=args.data_path.parent,
            smiles_columns=args.smiles_columns,
            label_columns=None,
            global_features=hparams["decoder"]["global_features"],
            global_features_file=hparams["decoder"]["global_features_file"],
            task=hparams["training"]["task"],
        )
        collate = graphless_collate_fn
    else:
        raise ValueError(f"Unknown model type {hparams['model_type']}")

    # instantiate DataLoader
    dl = DataLoader(data, batch_size=32, shuffle=False, collate_fn=collate)

    # load trained model
    model = load_trained_model(hparams["name"], args.model_path)
    model.modules()
    # predict
    predictions = predict(model, dl, hparams, return_proba=args.return_probabilities)

    # save predictions to text file
    pred_file = (
        LOG_DIR
        / "predictions"
        / f"{args.model_path.parent.name}_{generate_run_id()}"
        / "predictions.txt"
    )
    if not pred_file.parent.exists():
        pred_file.parent.mkdir(parents=True)
    with open(pred_file, "w") as f:
        f.write("Prediction\n")
        for i in predictions.tolist():
            f.write(str(i) + "\n")

    print("Predicted values:")
    print(predictions)

    return
