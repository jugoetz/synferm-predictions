import argparse
import pathlib
import statistics

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from src.model.classifier import load_trained_model
from src.util.definitions import TRAINED_MODEL_DIR, LOG_DIR
from src.data.dataloader import SLAPProductDataset, collate_fn


def import_valid_smiles_from_vl(
    raw_dir: pathlib.Path, filename: str, valid_idx_file: pathlib.Path = None
):
    """Import smiles from a csv file and filter by values in the `valid` column of a second csv file"""
    smiles_df = pd.read_csv(raw_dir / filename)
    if valid_idx_file is None:
        return smiles_df
    else:
        indices_arr = pd.read_csv(valid_idx_file)["valid"].values
        return smiles_df[indices_arr]


def main(product_file, valid_idx_file, output_file, is_reaction, verbose=False):
    use_validation_data = True
    # paths to the best models
    if use_validation_data:
        # the next three are trained with full data, including validation plate data
        model_0D = TRAINED_MODEL_DIR / "2023-03-06-105610_484882" / "best.ckpt"  # FFN
        model_1D = (
            TRAINED_MODEL_DIR / "2023-03-06-112027_188465" / "best.ckpt"
        )  # D-MPNN
        model_2D = (
            TRAINED_MODEL_DIR / "2023-03-06-112721_778803" / "best.ckpt"
        )  # D-MPNN
        # path to the OneHotEncoder state for model_0D
        ohe_state_dict = (
            LOG_DIR / "OHE_state_dict_bhTczANzKRRqIgUR.json"
        )  # with validation plate data

    else:
        # the next three are trained without using validation plate data
        model_0D = TRAINED_MODEL_DIR / "2022-12-16-144509_863758" / "best.ckpt"  # FFN
        model_1D = (
            TRAINED_MODEL_DIR / "2022-12-16-145840_448790" / "best.ckpt"
        )  # D-MPNN
        model_2D = (
            TRAINED_MODEL_DIR / "2022-12-06-115456_273019" / "best.ckpt"
        )  # D-MPNN
        # path to the OneHotEncoder state for model_0D
        ohe_state_dict = (
            LOG_DIR / "OHE_state_dict_FqIDTIsCHoURGQcv.json"
        )  # without validation plate data

    # import data
    raw_dir = product_file.parent
    # remove the .csv extension AND any other extensions behind it (e.g. remove .csv.bz2 or csv.gz)
    filename_base = product_file.name.split(".csv")[0]
    df = import_valid_smiles_from_vl(
        raw_dir, product_file.name, valid_idx_file=valid_idx_file
    )

    data = SLAPProductDataset(
        smiles=df["smiles"].values.tolist(),
        is_reaction=is_reaction,
        use_validation=use_validation_data,
    )

    # Process data. This includes generating reaction graphs and takes some time.
    data.process(
        {
            "dataset_0D": dict(
                reaction=True,
                global_features=[
                    "OHE",
                ],
                global_featurizer_state_dict_path=ohe_state_dict,
                graph_type="bond_edges",
                featurizers="custom",
            ),
            "dataset_1D_slap": dict(
                reaction=True,
                global_features=None,
                graph_type="bond_nodes",
                featurizers="custom",
            ),
            "dataset_1D_aldehyde": dict(
                reaction=True,
                global_features=None,
                graph_type="bond_nodes",
                featurizers="custom",
            ),
            "dataset_2D": dict(
                reaction=True,
                global_features=None,
                graph_type="bond_nodes",
                featurizers="custom",
            ),
        }
    )

    # run all the predictions

    if data.dataset_0D:
        # load the trained model if it is not loaded
        if isinstance(model_0D, pathlib.Path):
            model_0D = load_trained_model("FFN", model_0D)
            model_0D.eval()
        trainer = pl.Trainer(accelerator="gpu", logger=False, max_epochs=-1)
        dl = DataLoader(data.dataset_0D, collate_fn=collate_fn)
        probabilities_0D = torch.concat(trainer.predict(model_0D, dl))
        predictions_0D = (probabilities_0D > 0.5).numpy().astype(float)

    if data.dataset_1D_aldehyde:
        # load the trained model if it is not loaded
        if isinstance(model_1D, pathlib.Path):
            model_1D = load_trained_model("D-MPNN", model_1D)
            model_1D.eval()
        trainer = pl.Trainer(accelerator="gpu", logger=False, max_epochs=-1)
        dl = DataLoader(data.dataset_1D_aldehyde, collate_fn=collate_fn)
        probabilities_1D_aldehyde = torch.concat(trainer.predict(model_1D, dl))
        predictions_1D_aldehyde = (
            (probabilities_1D_aldehyde > 0.5).numpy().astype(float)
        )

    if data.dataset_1D_slap:
        # load the trained model if it is not loaded
        if isinstance(model_1D, pathlib.Path):
            model_1D = load_trained_model("D-MPNN", model_1D)
            model_1D.eval()
        trainer = pl.Trainer(accelerator="gpu", logger=False, max_epochs=-1)
        dl = DataLoader(data.dataset_1D_slap, collate_fn=collate_fn)
        probabilities_1D_slap = torch.concat(trainer.predict(model_1D, dl))
        predictions_1D_slap = (probabilities_1D_slap > 0.5).numpy().astype(float)

    if data.dataset_2D:
        # load the trained model if it is not loaded
        if isinstance(model_2D, pathlib.Path):
            model_2D = load_trained_model("D-MPNN", model_2D)
            model_2D.eval()
        trainer = pl.Trainer(accelerator="gpu", logger=False, max_epochs=-1)
        dl = DataLoader(data.dataset_2D, collate_fn=collate_fn)
        probabilities_2D = torch.concat(trainer.predict(model_2D, dl))
        predictions_2D = (probabilities_2D > 0.5).numpy().astype(float)

    # assemble outputs
    predictions = np.full(len(data.reactions), np.nan, dtype=float)

    predictions[data.idx_known] = [
        statistics.mean(data.known_outcomes[i]) for i in data.idx_known
    ]  # for known reaction we add the average reaction outcome
    try:
        predictions[data.idx_0D] = predictions_0D
    except NameError:
        pass
    try:
        predictions[data.idx_1D_slap] = predictions_1D_slap
    except NameError:
        pass
    try:
        predictions[data.idx_1D_aldehyde] = predictions_1D_aldehyde
    except NameError:
        pass
    try:
        predictions[data.idx_2D] = predictions_2D
    except NameError:
        pass

    # check if we have not predicted for anything
    # this should be only the reactions in data.invalid_idxs
    rxn_idxs_no_pred = np.argwhere(np.isnan(predictions)).flatten()

    rxn_idxs_invalid = [data.product_idxs.index(i) for i in data.invalid_idxs]

    if set(rxn_idxs_no_pred) != set(rxn_idxs_invalid):
        raise RuntimeError(
            f"We have not predicted for some reactions that are not invalid. (reactions at index {rxn_idxs_no_pred.difference(rxn_idxs_invalid)})"
        )

    # convert the 1D- product_idxs to the directionally reverse 2D indices
    arr = np.full((len(data.smiles), 2), fill_value=-1)
    last_idx = -1
    for i, idx in enumerate(data.product_idxs):
        if idx == last_idx:
            arr[idx, 1] = i
        else:
            last_idx = idx
            arr[idx, 0] = i

    confidence_dict = {
        "known": 0,
        "0D": 1,
        "1D_SLAP": 2,
        "1D_aldehyde": 2,
        "2D_similar": 3,
        "2D_dissimilar": 4,
    }

    # translate problem type to integer
    rxn_problem_types = list(map(confidence_dict.get, data.problem_type))

    # we add a nonsense value to the end of each of these lists so that indexing with -1 will return the nonsense value
    reactions_augmented = data.reactions + [""]
    predictions_augmented = list(predictions) + [np.nan]
    rxn_problem_types_augmented = rxn_problem_types + [99]

    # obtain individual new columns for output df
    df["rxn1_smiles"] = [data.reactions[i] for i in arr[:, 0]]

    df["rxn1_predictions"] = [predictions[i] for i in arr[:, 0]]

    df["rxn1_confidence"] = [rxn_problem_types[i] for i in arr[:, 0]]

    df["rxn2_smiles"] = [reactions_augmented[i] for i in arr[:, 1]]

    df["rxn2_predictions"] = [predictions_augmented[i] for i in arr[:, 1]]

    df["rxn2_confidence"] = [rxn_problem_types_augmented[i] for i in arr[:, 1]]

    # write dataset statistics for control to log file (+ optionally print)
    log_output = f"""{len(data.reactions)} reactions generated from {len(data.smiles)} input SMILES
Known reactions: {(sum(x is not None for x in data.known_outcomes))}
"""
    if data.dataset_0D:
        log_output += f"0D reactions: {len(data.dataset_0D)}, thereof {np.count_nonzero(predictions_0D)} predicted positive\n"
    else:
        log_output += "0D reactions: 0\n"
    if data.dataset_1D_aldehyde:
        log_output += f"1D reactions with unknown aldehyde: {len(data.dataset_1D_aldehyde)}, thereof {np.count_nonzero(predictions_1D_aldehyde)} predicted positive\n"
    else:
        log_output += "1D reactions with unknown aldehyde: 0\n"
    if data.dataset_1D_slap:
        log_output += f"1D reactions with unknown SLAP reagent: {len(data.dataset_1D_slap)}, thereof {np.count_nonzero(predictions_1D_slap)} predicted positive\n"
    else:
        log_output += "1D reactions with unknown SLAP reagent: 0\n"
    if data.dataset_2D:
        log_output += f"2D reactions: {len(data.dataset_2D)}, thereof {np.count_nonzero(predictions_2D)} predicted positive\n"
    else:
        log_output += "2D reactions: 0\n"

    if output_file is None:
        output_file = raw_dir / f"{filename_base}_reaction_prediction.csv"
        log_file = raw_dir / f"{filename_base}_reaction_prediction.log"
    else:
        log_file = output_file.with_suffix(".log")

    with open(log_file, "w") as file:
        file.write(log_output)
    if verbose:
        print(log_output)

    # write df to output file
    df.to_csv(output_file, index=False)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--product-file",
        type=pathlib.Path,
        help="Path to a CSV file containing SMILES of SLAP products",
        required=True,
    )
    parser.add_argument(
        "--reaction",
        help="Whether input is a reaction or not",
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
        "--valid-idx-file",
        type=pathlib.Path,
        help="Path to a CSV file containing indices used to filter products",
        required=False,
        default=None,
    )

    args = parser.parse_args()
    main(args.product_file, args.valid_idx_file, args.output_file, args.reaction)
