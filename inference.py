import argparse
import pathlib
import pickle as pkl
import warnings

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.model.classifier import load_trained_model
from src.util.definitions import TRAINED_MODEL_DIR
from src.data.dataloader import GraphLessSynFermDataset, graphless_collate_fn


def run_ffn_models(model_paths, dataloader):
    """
    Run the FFN models and return predictions

    Args:
        model_paths (list): List of pathlib.Path containing an FFN model checkpoint
        dataloader (torch.utils.data.DataLoader): Dataloader supplying the datapoints the models should be applied to.
            A datapoint is a 4-tuple (idx, graph, global_features, label) such as the one returned from a SynFermDataset

    Returns:
        np.array: Predicted labels, aggregated over models by majority voting
    """
    preds_folds = []
    trainer = pl.Trainer(accelerator="auto", logger=False, max_epochs=-1)

    for model_path in model_paths:
        # prepare
        model = load_trained_model("FFN", model_path)
        model.eval()
        # predict
        probs_0D = torch.sigmoid(torch.concat(trainer.predict(model, dataloader)))
        # load decision thresholds
        with open(model_path.parent / f"{model_path.parent.name}.txt", "r") as f:
            thresholds = [float(i) for i in f.readlines()]
        # apply thresholds
        preds_folds.append(
            torch.stack(
                [torch.where(probs_0D[:, i] > thresholds[i], 1, 0) for i in range(3)],
                dim=1,
            )
            .detach()
            .numpy()
        )

    # get final pred (majority vote)
    preds = np.where(np.sum(preds_folds, axis=0) >= len(preds_folds) / 2, 1, 0)
    return preds


def run_xgb_models(model_paths, dataset):
    """
    Run the XGB models and return predictions

    Args:
        model_paths (list): List of pathlib.Path containing an XGB model pickle file
        dataset (list): List of datapoints the models should be applied to.
            A datapoint is a 4-tuple (idx, graph, global_features, label) such as the one returned from a SynFermDataset

    Returns:
        np.array: Predicted labels, aggregated over models by majority voting
    """
    idx, _, global_features, _ = map(list, zip(*dataset))
    preds_folds = []
    for model_path in model_paths:
        # load the trained model
        with open(model_path, "rb") as f:
            model = pkl.load(f)
        # predict
        probs = np.stack(
            [y[:, 1] for y in model.predict_proba(global_features)], axis=1
        )
        # load decision thresholds
        with open(model_path.parent / f"{model_path.parent.name}.txt", "r") as f:
            thresholds = [float(i) for i in f.readlines()]
        # apply thresholds
        preds_folds.append(
            np.stack(
                [np.where(probs[:, i] > thresholds[i], 1, 0) for i in range(3)], axis=1
            )
        )
    # get final pred (majority vote)
    preds = np.where(np.sum(preds_folds, axis=0) >= len(preds_folds) / 2, 1, 0)
    return preds


def main(product_file, output_file, smiles_cols):
    # trained model paths
    model_0D = [
        TRAINED_MODEL_DIR
        / f"2024-04-23-114842_552891_fold0"
        / "last-epoch56-val_loss0.18.ckpt",
        TRAINED_MODEL_DIR
        / f"2024-04-23-114842_552891_fold1"
        / "last-epoch50-val_loss0.19.ckpt",
        TRAINED_MODEL_DIR
        / f"2024-04-23-114842_552891_fold2"
        / "last-epoch45-val_loss0.21.ckpt",
        TRAINED_MODEL_DIR
        / f"2024-04-23-114842_552891_fold3"
        / "last-epoch49-val_loss0.20.ckpt",
        TRAINED_MODEL_DIR
        / f"2024-04-23-114842_552891_fold4"
        / "last-epoch46-val_loss0.21.ckpt",
        TRAINED_MODEL_DIR
        / f"2024-04-23-114842_552891_fold5"
        / "last-epoch48-val_loss0.19.ckpt",
        TRAINED_MODEL_DIR
        / f"2024-04-23-114842_552891_fold6"
        / "last-epoch51-val_loss0.19.ckpt",
        TRAINED_MODEL_DIR
        / f"2024-04-23-114842_552891_fold7"
        / "last-epoch53-val_loss0.20.ckpt",
        TRAINED_MODEL_DIR
        / f"2024-04-23-114842_552891_fold8"
        / "last-epoch53-val_loss0.20.ckpt",
    ]
    model_1D_I = [
        TRAINED_MODEL_DIR / f"2024-04-21-005536_932721_fold{i}" / "model.pkl"
        for i in range(3)
    ]  # XGB
    model_1D_M = [
        TRAINED_MODEL_DIR / f"2024-04-21-005536_932721_fold{i}" / "model.pkl"
        for i in range(3, 6)
    ]  # XGB
    model_1D_T = [
        TRAINED_MODEL_DIR / f"2024-04-21-005536_932721_fold{i}" / "model.pkl"
        for i in range(6, 9)
    ]  # XGB
    model_2D_IM = [
        TRAINED_MODEL_DIR / f"2024-04-20-225117_229650_fold{i}" / "model.pkl"
        for i in range(3)
    ]  # XGB
    model_2D_IT = [
        TRAINED_MODEL_DIR / f"2024-04-20-225117_229650_fold{i}" / "model.pkl"
        for i in range(6, 9)
    ]  # XGB
    model_2D_MT = [
        TRAINED_MODEL_DIR / f"2024-04-20-225117_229650_fold{i}" / "model.pkl"
        for i in range(3, 6)
    ]  # XGB
    model_3D = [
        TRAINED_MODEL_DIR / f"2024-04-20-142305_520992_fold{i}" / "model.pkl"
        for i in range(9)
    ]  # XGB

    # path to the OneHotEncoder state for model_0D
    ohe_state_dict = TRAINED_MODEL_DIR / "OHE_state_dict_rQsvApbyvgOdwgpW.json"

    # set file paths
    raw_dir = product_file.parent
    filename_base = product_file.name.split(".csv")[0]
    if output_file is None:
        output_file = raw_dir / f"{filename_base}_predictions.csv"
    log_file = output_file.with_suffix(".log")

    # TODO if we want to accept --product inputs, we need to do the conversion to reactants at this point

    # load data
    data = GraphLessSynFermDataset(
        name=product_file.name,
        raw_dir=raw_dir,
        global_features=["OHE_silent"],
        global_featurizer_state_dict_path=ohe_state_dict,
        smiles_columns=smiles_cols,
        label_columns=None,
        task="multilabel",
        force_reload=True,
    )

    # for the 1D/2D/3D models we will need fingerprints
    data_fp = GraphLessSynFermDataset(
        name=product_file.name,
        raw_dir=raw_dir,
        global_features=["FP"],
        smiles_columns=smiles_cols,
        label_columns=None,
        task="multilabel",
        force_reload=True,
    )

    # reference for which model we apply, based on one-hot-encoding for each reactant
    model_domains = {
        (True, True, True): "0D",
        (True, True, False): "1D_T",
        (True, False, True): "1D_M",
        (False, True, True): "1D_I",
        (True, False, False): "2D_MT",
        (False, True, False): "2D_IT",
        (False, False, True): "2D_IM",
        (False, False, False): "3D",
    }

    # determine which model to apply for each data points
    models_to_apply = pd.DataFrame(
        [model_domains[data.known_one_hot_encodings(i)] for i in range(len(data))],
        columns=["dim"],
    )

    (
        preds_0D,
        preds_1D_I,
        preds_1D_M,
        preds_1D_T,
        preds_2D_IM,
        preds_2D_IT,
        preds_2D_MT,
        preds_3D,
    ) = 8 * (None,)

    # 0D model
    if len(models_to_apply.loc[models_to_apply["dim"] == "0D"]) > 0:
        # prepare
        dl = DataLoader(
            [
                d
                for i, d in enumerate(data)
                if i in (models_to_apply.loc[models_to_apply["dim"] == "0D"]).index
            ],
            collate_fn=graphless_collate_fn,
            num_workers=0,
        )
        # predict
        preds_0D = run_ffn_models(model_0D, dl)

    # 1D_I models
    if len(models_to_apply.loc[models_to_apply["dim"] == "1D_I"]) > 0:
        dataset = [
            d
            for i, d in enumerate(data_fp)
            if i in (models_to_apply.loc[models_to_apply["dim"] == "1D_I"]).index
        ]
        preds_1D_I = run_xgb_models(model_1D_I, dataset)

    # 1D_M models
    if len(models_to_apply.loc[models_to_apply["dim"] == "1D_M"]) > 0:
        dataset = [
            d
            for i, d in enumerate(data_fp)
            if i in (models_to_apply.loc[models_to_apply["dim"] == "1D_M"]).index
        ]
        preds_1D_M = run_xgb_models(model_1D_M, dataset)

    # 1D_T models
    if len(models_to_apply.loc[models_to_apply["dim"] == "1D_T"]) > 0:
        dataset = [
            d
            for i, d in enumerate(data_fp)
            if i in (models_to_apply.loc[models_to_apply["dim"] == "1D_T"]).index
        ]
        preds_1D_T = run_xgb_models(model_1D_T, dataset)

    # 2D_IM models
    if len(models_to_apply.loc[models_to_apply["dim"] == "2D_IM"]) > 0:
        dataset = [
            d
            for i, d in enumerate(data_fp)
            if i in (models_to_apply.loc[models_to_apply["dim"] == "2D_IM"]).index
        ]
        preds_2D_IM = run_xgb_models(model_2D_IM, dataset)

    # 2D_IT models
    if len(models_to_apply.loc[models_to_apply["dim"] == "2D_IT"]) > 0:
        dataset = [
            d
            for i, d in enumerate(data_fp)
            if i in (models_to_apply.loc[models_to_apply["dim"] == "2D_IT"]).index
        ]
        preds_2D_IT = run_xgb_models(model_2D_IT, dataset)

    # 2D_MT models
    if len(models_to_apply.loc[models_to_apply["dim"] == "2D_MT"]) > 0:
        dataset = [
            d
            for i, d in enumerate(data_fp)
            if i in (models_to_apply.loc[models_to_apply["dim"] == "2D_MT"]).index
        ]
        preds_2D_MT = run_xgb_models(model_2D_MT, dataset)

    # 3D models
    if len(models_to_apply.loc[models_to_apply["dim"] == "3D"]) > 0:
        dataset = [
            d
            for i, d in enumerate(data_fp)
            if i in (models_to_apply.loc[models_to_apply["dim"] == "3D"]).index
        ]
        preds_3D = run_xgb_models(model_3D, dataset)

    # assemble outputs
    results = models_to_apply.copy()
    results[["pred_A", "pred_B", "pred_C"]] = -1  # placeholder
    if preds_0D is not None:
        results.loc[results["dim"] == "0D", ["pred_A", "pred_B", "pred_C"]] = preds_0D
    if preds_1D_I is not None:
        results.loc[
            results["dim"] == "1D_I", ["pred_A", "pred_B", "pred_C"]
        ] = preds_1D_I
    if preds_1D_M is not None:
        results.loc[
            results["dim"] == "1D_M", ["pred_A", "pred_B", "pred_C"]
        ] = preds_1D_M
    if preds_1D_T is not None:
        results.loc[
            results["dim"] == "1D_T", ["pred_A", "pred_B", "pred_C"]
        ] = preds_1D_T
    if preds_2D_IM is not None:
        results.loc[
            results["dim"] == "2D_IM", ["pred_A", "pred_B", "pred_C"]
        ] = preds_2D_IM
    if preds_2D_IT is not None:
        results.loc[
            results["dim"] == "2D_IT", ["pred_A", "pred_B", "pred_C"]
        ] = preds_2D_IT
    if preds_2D_MT is not None:
        results.loc[
            results["dim"] == "2D_MT", ["pred_A", "pred_B", "pred_C"]
        ] = preds_2D_MT
    if preds_3D is not None:
        results.loc[results["dim"] == "3D", ["pred_A", "pred_B", "pred_C"]] = preds_3D

    # safety net
    if results.eq(-1).any().any():
        warnings.warn(
            "A prediction could not be made for all data points. Missing predictions are marked as '-1' in output file. Logged statistics will be inaccurate."
        )

    print(results)

    # write dataset statistics for control to log file (+ optionally print)
    log_output = f"Predicted for {len(data)} reactant combinations\n"
    for split in ["0D", "1D_I", "1D_M", "1D_T", "2D_IM", "2D_IT", "2D_MT", "3D"]:
        if len(results.loc[results["dim"] == split]) > 0:
            log_output += f"{split} data: {len(results.loc[results['dim'] == split])}, thereof {results.loc[results['dim'] == split, 'pred_A'].sum()} predicted to form product A\n"

    with open(log_file, "w") as file:
        file.write(log_output)

    # write df to output file
    results.to_csv(output_file, index=False)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-file",
        type=pathlib.Path,
        help="Path to a CSV file containing SMILES of Synthetic Fermentation reactants in columns named 'initiator', 'monomer', 'terminator'.",
        required=True,
    )
    parser.add_argument(
        "--products",
        help="If passed, expects the CSV file to have one column 'product' containing the SMILES for product A instead of reactants. Note: Currently not implemented.",
        action="store_true",
    )
    parser.add_argument(
        "-o",
        "--output-file",
        type=pathlib.Path,
        help="Path to a CSV file to save the results",
        required=False,
        default=None,
    )
    parser.add_argument(
        "--smiles-columns",
        nargs="*",
        help="Headers of the columns containing SMILES strings of reactants (in the order I, M, T) (or product if '--products' is passed). The default headers are 'initiator', 'monomer', 'terminator' for reactants and 'product' for products",
        required=False,
        default=None,
    )

    args = parser.parse_args()
    if args.smiles_columns:
        smiles_columns = args.smiles_columns
    else:
        if args.products:
            smiles_columns = ["product"]
        else:
            smiles_columns = ["initiator", "monomer", "terminator"]
    main(args.input_file, args.output_file, smiles_columns)
