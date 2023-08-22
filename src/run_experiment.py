from torch.utils.data import DataLoader

from src.data.dataloader import SLAPDataset, collate_fn
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
    # load data
    data = SLAPDataset(
        name=args.data_path.name,
        raw_dir=args.data_path.parent,
        reaction=hparams["encoder"]["reaction"],
        smiles_columns=(args.smiles_column,),
        label_column=args.label_column,
        graph_type=hparams["encoder"]["graph_type"],
        global_features=hparams["decoder"]["global_features"],
        global_features_file=hparams["decoder"]["global_features_file"],
        featurizers=hparams["encoder"]["featurizers"],
    )
    data.process()

    # update config with data processing specifics
    hparams["atom_feature_size"] = data.atom_feature_size
    hparams["bond_feature_size"] = data.bond_feature_size
    hparams["global_feature_size"] = data.global_feature_size
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

    job_type = None
    # run hyperparameter optimization if requested
    if args.hparam_optimization:
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
    if job_type is "hparam_optimization":
        job_type = "hparam_best"
    else:
        job_type = "training"

    if hparams["name"] in ["D-MPNN", "GCN", "GraphAgnostic", "FFN"]:
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
    elif hparams["name"] in ["LogisticRegression", "XGB"]:
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
    else:
        raise ValueError(f"Unknown model type {hparams['name']}")
    if args.hparam_optimization:
        print(f"Optimized hyperparameters: {hparams}")
    print(f"Aggregate metrics:")
    for k, v in aggregate_metrics.items():
        print(f"{k}: {v}")

    return


def run_prediction(args, hparams):
    """
    Handles prediction from a trained model.
    """

    # load data
    data = SLAPDataset(
        name=args.data_path.name,
        raw_dir=args.data_path.parent,
        reaction=hparams["encoder"]["reaction"],
        smiles_columns=(args.smiles_column,),
        label_column=None,
        graph_type=hparams["encoder"]["graph_type"],
        global_features=hparams["decoder"]["global_features"],
        featurizers=hparams["encoder"]["featurizers"],
    )

    # instantiate DataLoader
    dl = DataLoader(data, batch_size=32, shuffle=False, collate_fn=collate_fn)

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
