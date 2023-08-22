import argparse
import pathlib

from src.run_experiment import run_training, run_prediction
from src.util.configuration import get_config


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    # parent parser for training and prediction
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument(
        "--config", type=pathlib.Path, help="Path to config file", required=True
    )
    parent_parser.add_argument(
        "--data-path",
        type=pathlib.Path,
        required=True,
        help="Path to the data. For training, this should contain labels. For prediction, labels will be ignored.",
    )
    parent_parser.add_argument(
        "--smiles-column",
        type=str,
        default="SMILES",
        help="Name of the column containing SMILES strings to use as input.",
    )
    parent_parser.add_argument(
        "--tags",
        type=str,
        nargs="+",
        default=None,
        help="List of tags to add to the run in wandb.",
    )

    parent_parser.add_argument(
        "--global-features",
        type=str,
        nargs="+",
        choices=["RDKit", "FP", "OHE", "fromFile", "None"],
        help="Which global features to add. Multiple options can be given, separated by a space.",
    )

    parent_parser.add_argument(
        "--global-features-file",
        type=pathlib.Path,
        help="Path to file containing global features. Required if --global-features is set to 'fromFile'.",
    )

    # train parser
    train_parser = subparsers.add_parser("train", parents=[parent_parser])
    train_parser.set_defaults(func=run_training)

    train_parser.add_argument(
        "--label-column",
        type=str,
        default="targets",
        help="Name of the column containing ground truth labels.",
    )
    train_parser.add_argument(
        "--hparam-optimization",
        action="store_true",
        help="Run hyperparameter optimization",
    )
    train_parser.add_argument(
        "--hparam-config-path",
        type=str,
        help="Path to hyperparameter search config file",
    )
    train_parser.add_argument(
        "--hparam-n-iter",
        type=int,
        help="Number of trials for hyperparameter optimization",
    )
    train_parser.add_argument(
        "--cv",
        type=int,
        help="Number of CV folds. If > 1, a random CV split is done. Ignored if --split-indices is provided.",
        default=0,
    )
    train_parser.add_argument(
        "--train-size",
        type=float,
        help="Fraction of data to use for training. If --cv > 1 is given or --split-indices is given, this is ignored.",
        default=0.9,
    )
    train_parser.add_argument(
        "--split-indices",
        type=pathlib.Path,
        help="""Path to directory containing data set splits.
        Expects a directory of csv-files following the naming convention fold[i]_{train,val,test}[_optionalsuffix].csv.
        For each fold, one train and one val set need to be given. An optional arbitrary number of test sets can be
        given. Multiple test sets are distinguished by the optional suffix.
        Fold indices start at 0 and increase monotonically.
        """,
    )
    train_parser.add_argument(
        "--run-test",
        action="store_true",
        help="Run test set evaluation after training. Only takes an effect if `--split-indices` is given and test set "
        "is present in the split directory.",
    )

    # predict parser
    predict_parser = subparsers.add_parser("predict", parents=[parent_parser])
    predict_parser.add_argument(
        "--model-path",
        required=True,
        type=pathlib.Path,
        help="Path to the model checkpoint to be used for inference.",
    )
    predict_parser.add_argument(
        "--return-probabilities",
        action="store_true",
        help="Return class probabilities instead of class predictions",
    )

    predict_parser.set_defaults(func=run_prediction)

    # parse command-line arguments
    args = parser.parse_args()

    # parse configuration file
    hparams = get_config(args.config)

    # overwrite some typically changed hparams with command line arguments
    # slightly unintuitive: Checking for None works here because argparse will parse "None" as a string, which makes it not None
    if args.global_features:
        hparams["decoder"]["global_features"] = args.global_features
    if args.global_features_file:
        hparams["decoder"]["global_features_file"] = args.global_features_file

    args.func(args, hparams)
