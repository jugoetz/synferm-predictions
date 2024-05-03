import os
import pathlib
import string
import random
from typing import List, Tuple, Optional, Union, Sequence

import dgl
import numpy as np
from dgl.data import DGLDataset
import torch
import pandas as pd
from dgllife.utils.featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from rdkit import Chem
from sklearn.preprocessing import LabelBinarizer

from src.data.grapher import build_cgr, build_mol_graph
from src.data.featurizers import (
    ChempropAtomFeaturizer,
    ChempropBondFeaturizer,
    SynFermAtomFeaturizer,
    SynFermBondFeaturizer,
    RDKit2DGlobalFeaturizer,
    RDKitMorganFingerprinter,
    OneHotEncoder,
    FromFileFeaturizer,
)
from src.data.util import SLAPReactionGenerator, SLAPReactionSimilarityCalculator
from src.util.definitions import DATA_ROOT, LOG_DIR
from src.util.rdkit_util import canonicalize_smiles, standardize_building_block


def collate_fn(
    batch: List[Tuple[int, dgl.DGLGraph, torch.tensor, torch.tensor]]
) -> Tuple[List[int], dgl.DGLGraph, Optional[torch.tensor], torch.tensor]:
    """Collate a list of samples into a batch, i.e. into a single tuple representing the entire batch"""

    indices, graphs, global_features, labels = map(list, zip(*batch))

    batched_graphs = dgl.batch(graphs)

    out_feat = None
    out_labels = None
    elem_feat = global_features[0]
    elem_labels = labels[0]
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel_feat = sum(x.numel() for x in global_features)
        storage = elem_feat._typed_storage()._new_shared(
            numel_feat, device=elem_feat.device
        )
        out_feat = elem_feat.new(storage).resize_(
            len(global_features), *list(elem_feat.size())
        )

        numel_labels = sum(x.numel() for x in labels)
        storage = elem_labels._typed_storage()._new_shared(
            numel_labels, device=elem_labels.device
        )
        out_labels = elem_labels.new(storage).resize_(
            len(labels), *list(elem_labels.size())
        )

    batched_global_features = torch.stack(global_features, dim=0, out=out_feat)
    batched_labels = torch.stack(labels, dim=0, out=out_labels)

    return indices, batched_graphs, batched_global_features, batched_labels


def graphless_collate_fn(
    batch: List[Tuple[int, torch.tensor, torch.tensor, torch.tensor]]
) -> Tuple[List[int], torch.tensor, Optional[torch.tensor], torch.tensor]:
    """
    Collate a list of samples into a batch, i.e. into a single tuple representing the entire batch.
    This implementation is for the GraphLessSynFermDataset specifically.
    """

    indices, graphs, global_features, labels = map(list, zip(*batch))

    batched_graphs = torch.empty((len(graphs),))  # gives a

    out_feat = None
    out_labels = None
    elem_feat = global_features[0]
    elem_labels = labels[0]
    if torch.utils.data.get_worker_info() is not None:
        # If we're in a background process, concatenate directly into a
        # shared memory tensor to avoid an extra copy
        numel_feat = sum(x.numel() for x in global_features)
        storage = elem_feat._typed_storage()._new_shared(
            numel_feat, device=elem_feat.device
        )
        out_feat = elem_feat.new(storage).resize_(
            len(global_features), *list(elem_feat.size())
        )

        numel_labels = sum(x.numel() for x in labels)
        storage = elem_labels._typed_storage()._new_shared(
            numel_labels, device=elem_labels.device
        )
        out_labels = elem_labels.new(storage).resize_(
            len(labels), *list(elem_labels.size())
        )

    batched_global_features = torch.stack(global_features, dim=0, out=out_feat)
    batched_labels = torch.stack(labels, dim=0, out=out_labels)

    return indices, batched_graphs, batched_global_features, batched_labels


class SynFermDataset(DGLDataset):
    """
    Synthetic Fermentation Dataset

    Can load a set of data points containing either a reactionSMILES or a SMILES of a single molecule and one or more
    columns containing labels.

    After processing, the data set will contain a featurized graph encoding of the molecule or reaction.
    If reaction=True, the encoding will be a condensed graph of reaction (CGR).
    """

    def __init__(
        self,
        name: str,
        raw_dir: Union[str, os.PathLike] = None,
        url: str = None,
        reaction=False,
        global_features: Union[str, List[str]] = None,
        global_features_file: Union[str, os.PathLike] = None,
        global_featurizer_state_dict_path: Union[str, os.PathLike] = None,
        graph_type: str = "bond_edges",
        featurizers: str = "dgllife",
        smiles_columns: Sequence = ["SMILES"],
        label_columns: Optional[List[str]] = ["label"],
        task: str = "binary",
        save_dir: Union[str, os.PathLike] = None,
        force_reload=False,
        verbose=True,
    ):
        """
        Args:
            name: File name of the dataset
            raw_dir: Directory containing data. Default None
            url: Url to fetch data from. Default None
            reaction: Whether data is a reaction. If True, data will be loaded as CGR. Default False.
            global_features: Which global features to add.
                Options: {"RDKit", "FP", "OHE", "fromFile", None} and combinations thereof.
                If a string is passed, any way to combine the features is allowed (e.g. "RDKit_FP", "RDKit+FP",...)
                If a list is passed, only the strings list in options as list members will have an effect.
                Passing and empty list, or a list with unrecognized options is equivalent to passing None.
                Default None.
            global_features_file: Path to file containing global features.
                Only used with global_features=fromFile. Default None.
            global_featurizer_state_dict_path: Path to state dict of featurizer to use for global features (only for OHE).
            graph_type: Type of graph to use. If "bond_edges", graphs are formed as molecular graphs (nodes are
                    atoms and edges are bonds). These graphs are homogeneous. If "bond_nodes", bond-node graphs will be
                    formed (both atoms and bonds are nodes, edges represent their connectivity).
                    Options: {"bond_edges", "bond_nodes"}. Default "bond_edges".
            featurizers: Featurizers to use for atom and bond featurization. Options: {"dgllife", "chemprop", "custom"}.
                Default "dgllife".
            smiles_columns: Headers of columns in data file that contain SMILES strings
            label_columns: Header of the columns in data file that contain the labels
            task: Kind of prediction task. Options: {binary, multiclass, multilabel}
            save_dir: Directory to save the processed data set. If None, `raw_dir` is used. Default None.
            force_reload: Reload data set, ignoring cache. Default False.
                If OHE is in global_features, reload will always be forced.
            verbose: Whether to provide verbose output
        """

        if global_features is None:
            global_features = []

        self.reaction = reaction
        self.label_columns = label_columns
        self.smiles_columns = smiles_columns
        self.graph_type = graph_type
        self.task = task
        self.label_binarizer = LabelBinarizer()
        self.global_features = []
        self.global_featurizers = []
        self.graphs = []
        self.labels = []
        if isinstance(global_featurizer_state_dict_path, str):
            self.global_featurizer_state_dict_path = pathlib.Path(
                global_featurizer_state_dict_path
            )

        else:
            self.global_featurizer_state_dict_path = global_featurizer_state_dict_path

        # featurizer to obtain atom and bond features
        if featurizers == "dgllife":
            self.atom_featurizer = CanonicalAtomFeaturizer(atom_data_field="x")
            self.bond_featurizer = CanonicalBondFeaturizer(bond_data_field="e")
        elif featurizers == "chemprop":
            self.atom_featurizer = ChempropAtomFeaturizer(atom_data_field="x")
            self.bond_featurizer = ChempropBondFeaturizer(bond_data_field="e")
        elif featurizers == "custom":
            self.atom_featurizer = SynFermAtomFeaturizer(atom_data_field="x")
            self.bond_featurizer = SynFermBondFeaturizer(bond_data_field="e")
        else:
            raise ValueError(f"Unexpected value '{featurizers}' for 'featurizers'")

        # global featurizers
        if "RDKit" in global_features:
            self.global_featurizers.append(RDKit2DGlobalFeaturizer(normalize=True))
        if "FP" in global_features:
            self.global_featurizers.append(
                RDKitMorganFingerprinter(radius=6, n_bits=1024)
            )
        if "OHE" in global_features:
            self.global_featurizers.append(OneHotEncoder())
            # if we use OHE, we cannot rely on cached data sets
            force_reload = True
        if "fromFile" in global_features:
            self.global_featurizers.append(
                FromFileFeaturizer(filename=global_features_file)
            )

        hash_key = (
            self.reaction,
            self.label_columns,
            self.smiles_columns,
            self.graph_type,
            self.task,
            global_features,
            featurizers,
        )

        super(SynFermDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
            hash_key=hash_key,
        )

    def save(self):
        dgl.data.utils.save_graphs(
            os.path.join(self.save_path, "graphs.bin"), self.graphs
        )
        dgl.data.utils.save_info(
            os.path.join(self.save_path, "info.bin"),
            {
                "global_features": self.global_features,
                "labels": self.labels,
                "label_binarizer": self.label_binarizer,
            },
        )

    def load(self):
        self.graphs, _ = dgl.data.utils.load_graphs(
            os.path.join(self.save_path, "graphs.bin")
        )
        info = dgl.data.utils.load_info(os.path.join(self.save_path, "info.bin"))
        self.global_features = info["global_features"]
        self.labels = info["labels"]
        self.label_binarizer = info["label_binarizer"]

    def has_cache(self):
        return os.path.exists(self.save_path)

    @property
    def save_path(self):
        return os.path.join(self.save_dir, self.name.removesuffix(".csv") + self.hash)

    def process(self):
        """
        Read reactionSMILES or SMILES data from csv file and generate CGR or molecular graph, respectively.
        If the reaction argument was True during __init__, we expect reactionSMILES and produce the condensed graph
        of reaction (CGR). Else, we expect a single SMILES and produce the molecular graph.
        Further, calculates global features and reads labels.
        """

        csv_data = pd.read_csv(self.raw_path)
        smiles = csv_data[
            list(self.smiles_columns)
        ]  # shape (n_samples, n_smiles_columns)
        if self.reaction:
            # check input
            if smiles.shape[1] != 1:
                raise ValueError(
                    "Cannot have more than one reactionSMILES per record. Make sure smiles_columns has exactly one item."
                )

            # pull out reactant SMILES from reactionSMILES
            reactants = (
                smiles.iloc[:, 0]
                .str.split(">>", expand=True)
                .iloc[:, 0]
                .str.split(".", expand=True)
                .applymap(
                    canonicalize_smiles
                )  # n.b. using standardize_building_blocks might kill graph construction
            )  # shape (n_samples, n_reactants)

            # generate CGR from reactionSMILES
            self.graphs = [
                build_cgr(
                    s,
                    self.atom_featurizer,
                    self.bond_featurizer,
                    mode="reac_diff",
                    graph_type=self.graph_type,
                )
                for s in smiles.iloc[:, 0]
            ]  # list with shape (n_samples, )

        else:
            # generate molecular graphs from molecule SMILES
            self.graphs = [
                [
                    build_mol_graph(
                        s,
                        self.atom_featurizer,
                        self.bond_featurizer,
                        graph_type=self.graph_type,
                    )
                    for s in smi
                ]
                for i, smi in smiles.iterrows()
            ]  # nested list with shape (n_samples, n_smiles_columns)

        # set input for the global featurizer(s)
        global_featurizer_input = (reactants if self.reaction else smiles).applymap(
            standardize_building_block, return_smiles=True
        )

        # apply global featurizers
        global_features = []
        for global_featurizer in self.global_featurizers:
            # OneHotEncoder needs setup
            if isinstance(global_featurizer, OneHotEncoder):
                # use state if given
                if self.global_featurizer_state_dict_path:
                    # load given state dict
                    global_featurizer.load_state_dict(
                        self.global_featurizer_state_dict_path
                    )
                else:  # if no state given
                    # sanity check: OneHotEncoder should not have been initialized before
                    assert global_featurizer.n_dimensions == 0
                    # initialize encoder with the list(s) of SMILES to encode
                    for _, smi in global_featurizer_input.items():
                        global_featurizer.add_dimension(smi.to_list())

                    # save OHE state (for inference)
                    self.global_featurizer_state_dict_path = LOG_DIR / (
                        "OHE_state_dict_"
                        + "".join(random.choices(string.ascii_letters, k=16))
                        + ".json"
                    )
                    global_featurizer.save_state_dict(
                        self.global_featurizer_state_dict_path
                    )

            # calculate global features for all inputs
            # result has shape (n_samples, len_global_feature)
            # where len_global_feature is n_smiles_columns (or n_reactants) x global_featurizer.feat_size
            global_features.append(
                np.stack(
                    [
                        global_featurizer.process(*smi)
                        for _, smi in global_featurizer_input.iterrows()
                    ]
                )
            )

        if len(global_features) == 0:
            # if no global features are requested, provide uninitialized array of shape (n_samples, n_smiles_columns)
            self.global_features = torch.empty(smiles.shape, dtype=torch.float32)

        else:
            # assemble global features into shape (n_samples, len_global_features)
            # where len_global_features is the sum over the lengths of all global features
            self.global_features = torch.tensor(
                np.concatenate(global_features, axis=1), dtype=torch.float32
            )

        if self.label_columns:
            # we apply the label binarizer regardless of task.
            # for binary task, it will encode the labels if they are given as strings, else leave them untouched
            # for multiclass task, it will one-hot encode the labels
            # for multilabel task, it will leave the labels untouched
            # combination of multiclass and multilabel is not supported
            labels = csv_data[self.label_columns].to_numpy()
            self.labels = torch.tensor(
                self.label_binarizer.fit_transform(labels), dtype=torch.float32
            )
            # the loss from torch needs float input.
            # The metrics from torchmetrics will need int input, so we have to convert that later.

        else:
            # if no labels are given (for inference), provide uninitialized tensor of shape (n_samples, 1)
            self.labels = torch.empty((smiles.shape[0], 1), dtype=torch.float32)

        # little safety net
        assert len(self.graphs) == len(self.labels) == len(self.global_features)

    def __getitem__(
        self, idx: int
    ) -> Tuple[int, dgl.DGLGraph, torch.tensor, torch.tensor]:
        """Get graph, global features and label(s) by index.
         We also return the index to enable later matching predictions to records.

        Args:
            idx (int): Item index

        Returns:
            (dgl.DGLGraph, tensor, tensor)
        """
        return idx, self.graphs[idx], self.global_features[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.graphs)

    @property
    def atom_feature_size(self) -> int:
        n_atom_features = self.atom_featurizer.feat_size()
        if self.reaction:
            return 2 * n_atom_features  # CGR has 2 x features
        else:
            return n_atom_features

    @property
    def bond_feature_size(self) -> int:
        n_bond_features = self.bond_featurizer.feat_size()
        if self.reaction:
            return 2 * n_bond_features  # CGR has 2 x features
        else:
            return n_bond_features

    @property
    def feature_size(self) -> int:
        return self.atom_feature_size + self.bond_feature_size

    @property
    def num_labels(self) -> int:
        """
        Returns the number of labels/targets.
        In the multiclass task, this equals the number of classes.
        """
        return len(self.labels[0])

    @property
    def global_feature_size(self) -> int:
        n_global_features = 0
        for global_featurizer in self.global_featurizers:
            if isinstance(global_featurizer, OneHotEncoder):
                # (OHE which always incorporates all inputs in feat size)
                n_global_features += global_featurizer.feat_size
            else:
                if self.reaction:
                    # for 3 reactants we have 3 x features
                    n_global_features += 3 * global_featurizer.feat_size
                else:
                    # for SMILES inputs, multiply with number of SMILES
                    n_global_features += global_featurizer.feat_size * len(
                        self.smiles_columns
                    )
        return n_global_features


class GraphLessSynFermDataset(DGLDataset):
    """
    Synthetic Fermentation Dataset without graphs.
    Useful for less overhead when using only FFN method.

    Can load a set of data points containing SMILES of one or more molecules (e.g. product or reactants) and one or more
    columns containing labels.

    Global features can be added if they derive from the input SMILES, not the reaction graph.

    The data set will contain a list of None values in lieu of the graph to be directly interchangeable with the
    "normal" SynFermDataset.
    """

    def __init__(
        self,
        name: str,
        raw_dir: Union[str, os.PathLike] = None,
        url: str = None,
        global_features: Union[str, List[str]] = None,
        global_features_file: Union[str, os.PathLike] = None,
        global_featurizer_state_dict_path: Union[str, os.PathLike] = None,
        smiles_columns: Sequence = ["SMILES"],
        label_columns: Optional[List[str]] = ["label"],
        task: str = "binary",
        save_dir: Union[str, os.PathLike] = None,
        force_reload=False,
        verbose=True,
    ):
        """
        Args:
            name: File name of the dataset
            raw_dir: Directory containing data. Default None
            url: Url to fetch data from. Default None
            global_features: Which global features to add.
                Options: {"RDKit", "FP", "OHE", "fromFile", None} and combinations thereof.
                If a string is passed, any way to combine the features is allowed (e.g. "RDKit_FP", "RDKit+FP",...)
                If a list is passed, only the strings list in options as list members will have an effect.
                Passing and empty list, or a list with unrecognized options is equivalent to passing None.
                Default None.
            global_features_file: Path to file containing global features.
                Only used with global_features=fromFile. Default None.
            global_featurizer_state_dict_path: Path to state dict of featurizer to use for global features (only for OHE).
            smiles_columns: Headers of columns in data file that contain SMILES strings
            label_columns: Header of the columns in data file that contain the labels
            task: Kind of prediction task. Options: {binary, multiclass, multilabel}
            save_dir: Directory to save the processed data set. If None, `raw_dir` is used. Default None.
            force_reload: Reload data set, ignoring cache. Default False.
                If OHE is in global_features, reload will always be forced.
            verbose: Whether to provide verbose output
        """

        if global_features is None:
            global_features = []

        self.label_columns = label_columns
        self.smiles_columns = smiles_columns
        self.task = task
        self.label_binarizer = LabelBinarizer()
        self.global_features = []
        self.global_featurizers = []
        self.global_features_sizes = []
        self.graphs = []
        self.labels = []
        if isinstance(global_featurizer_state_dict_path, str):
            self.global_featurizer_state_dict_path = pathlib.Path(
                global_featurizer_state_dict_path
            )

        else:
            self.global_featurizer_state_dict_path = global_featurizer_state_dict_path

        # global featurizers
        if "RDKit" in global_features:
            self.global_featurizers.append(RDKit2DGlobalFeaturizer(normalize=True))
        if "FP" in global_features:
            self.global_featurizers.append(
                RDKitMorganFingerprinter(radius=6, n_bits=1024)
            )
        if "OHE" in global_features:
            self.global_featurizers.append(OneHotEncoder())
            # if we use OHE, we cannot rely on cached data sets
            force_reload = True
        if "OHE_silent" in global_features:
            self.global_featurizers.append(OneHotEncoder(unknown_molecule="silent"))
            # if we use OHE, we cannot rely on cached data sets
            force_reload = True
        if "fromFile" in global_features:
            self.global_featurizers.append(
                FromFileFeaturizer(filename=global_features_file)
            )

        hash_key = (
            self.label_columns,
            self.smiles_columns,
            self.task,
            global_features,
        )

        super(GraphLessSynFermDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
            hash_key=hash_key,
        )

    def save(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        dgl.data.utils.save_info(
            os.path.join(self.save_path, "info.bin"),
            {
                "global_features": self.global_features,
                "labels": self.labels,
                "label_binarizer": self.label_binarizer,
            },
        )

    def load(self):
        info = dgl.data.utils.load_info(os.path.join(self.save_path, "info.bin"))
        self.global_features = info["global_features"]
        self.labels = info["labels"]
        self.label_binarizer = info["label_binarizer"]
        self.graphs = [None for _ in self.global_features]

    def has_cache(self):
        return os.path.exists(self.save_path)

    @property
    def save_path(self):
        return os.path.join(self.save_dir, self.name.removesuffix(".csv") + self.hash)

    def process(self):
        """
        Read SMILES data from csv file and generate CGR or molecular graph, respectively.
        Calculates global features and read labels.
        """

        csv_data = pd.read_csv(self.raw_path)
        smiles = csv_data[
            list(self.smiles_columns)
        ]  # shape (n_samples, n_smiles_columns)
        if smiles.shape[1] == 3:
            # sanitize inputs
            reactants = smiles.applymap(
                standardize_building_block, return_smiles=True
            )  # shape (n_samples, n_reactants)
            # placeholder for graphs
            self.graphs = [
                None for _ in smiles.iloc[:, 0]
            ]  # list with shape (n_samples,)
        elif smiles.shape[1] == 1:
            raise NotImplementedError(
                "Predicting from products directly is currently not implemented. Pass reactants instead."
            )
        else:
            raise ValueError(
                "We expect one (product) or three (I, M, T) SMILES per record. Make sure smiles_columns fits that."
            )

        # apply global featurizers
        global_features = []
        for global_featurizer in self.global_featurizers:
            # OneHotEncoder needs setup
            if isinstance(global_featurizer, OneHotEncoder):
                # use state if given
                if self.global_featurizer_state_dict_path:
                    # load given state dict
                    global_featurizer.load_state_dict(
                        self.global_featurizer_state_dict_path
                    )
                else:  # if no state given
                    # sanity check: OneHotEncoder should not have been initialized before
                    assert global_featurizer.n_dimensions == 0
                    # initialize encoder with the list(s) of SMILES to encode
                    for _, smi in reactants.items():
                        global_featurizer.add_dimension(smi.to_list())

                    # save OHE state (for inference)
                    self.global_featurizer_state_dict_path = LOG_DIR / (
                        "OHE_state_dict_"
                        + "".join(random.choices(string.ascii_letters, k=16))
                        + ".json"
                    )
                    global_featurizer.save_state_dict(
                        self.global_featurizer_state_dict_path
                    )
                for i, size in enumerate(global_featurizer.feat_size_by_dimension):
                    self.global_features_sizes.append((f"OHE_{i}", size))
            else:
                self.global_features_sizes.append(
                    (global_featurizer.name, global_featurizer.feat_size)
                )

            # calculate global features for all inputs
            # result has shape (n_samples, len_global_feature)
            # where len_global_feature is n_smiles_columns (or n_reactants) x global_featurizer.feat_size
            global_features.append(
                np.stack(
                    [global_featurizer.process(*smi) for _, smi in reactants.iterrows()]
                )
            )

        if len(global_features) == 0:
            # if no global features are requested, provide uninitialized array of shape (n_samples, n_smiles_columns)
            self.global_features = torch.empty(smiles.shape, dtype=torch.float32)

        else:
            # assemble global features into shape (n_samples, len_global_features)
            # where len_global_features is the sum over the lengths of all global features
            self.global_features = torch.tensor(
                np.concatenate(global_features, axis=1), dtype=torch.float32
            )

        if self.label_columns:
            # we apply the label binarizer regardless of task.
            # for binary task, it will encode the labels if they are given as strings, else leave them untouched
            # for multiclass task, it will one-hot encode the labels
            # for multilabel task, it will leave the labels untouched
            # combination of multiclass and multilabel is not supported
            labels = csv_data[self.label_columns].to_numpy()
            self.labels = torch.tensor(
                self.label_binarizer.fit_transform(labels), dtype=torch.float32
            )
            # the loss from torch needs float input.
            # The metrics from torchmetrics will need int input, so we have to convert that later.

        else:
            # if no labels are given (for inference), provide uninitialized tensor of shape (n_samples, 1)
            self.labels = torch.empty((smiles.shape[0], 1), dtype=torch.float32)

        # little safety net
        assert len(self.graphs) == len(self.labels) == len(self.global_features)

    def __getitem__(self, idx: int) -> Tuple[int, None, torch.tensor, torch.tensor]:
        """Get graph, global features and label(s) by index.
         We also return the index to enable later matching predictions to records.

        Args:
            idx (int): Item index

        Returns:
            (dgl.DGLGraph, tensor, tensor)
        """
        return idx, self.graphs[idx], self.global_features[idx], self.labels[idx]

    def __len__(self) -> int:
        return len(self.graphs)

    @property
    def atom_feature_size(self) -> int:
        return 0

    @property
    def bond_feature_size(self) -> int:
        return 0

    @property
    def feature_size(self) -> int:
        return self.atom_feature_size + self.bond_feature_size

    @property
    def num_labels(self) -> int:
        """
        Returns the number of labels/targets.
        In the multiclass task, this equals the number of classes.
        """
        return len(self.labels[0])

    @property
    def global_feature_size(self) -> int:
        n_global_features = 0
        for global_featurizer in self.global_featurizers:
            if isinstance(global_featurizer, OneHotEncoder):
                # (OHE which always incorporates all inputs in feat size)
                n_global_features += global_featurizer.feat_size
            else:
                n_global_features += global_featurizer.feat_size * len(
                    self.smiles_columns
                )
        return n_global_features

    def known_one_hot_encodings(self, idx):
        """
        Returns which reactants were recognized by the one-hot encoder.

        Args:
            idx: Index of the datapoint

        Returns:
            tuple: 3-tuple of bools. True if the reactant was recognized by the OHEncoder. Order I, M, T.

        Raises:
            RuntimeError: If the dataset does not contain OHE features.
        """
        if not any(
            [
                isinstance(global_featurizer, OneHotEncoder)
                for global_featurizer in self.global_featurizers
            ]
        ):
            raise RuntimeError(
                "This dataset was not initialized with OneHotEncoder as global featurizer."
            )

        # get the indices
        counter = 0
        for size in self.global_features_sizes:
            if size[0] == "OHE_0":
                idx_i = [counter, counter + size[1]]
            if size[0] == "OHE_1":
                idx_m = [counter, counter + size[1]]
            if size[0] == "OHE_2":
                idx_t = [counter, counter + size[1]]
            counter += size[1]

        # see if there are on bits in the respective range
        i = bool(sum(self.global_features[idx, idx_i[0] : idx_i[1]]))
        m = bool(sum(self.global_features[idx, idx_m[0] : idx_m[1]]))
        t = bool(sum(self.global_features[idx, idx_t[0] : idx_t[1]]))

        return i, m, t
