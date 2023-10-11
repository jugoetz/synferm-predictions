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
from src.util.rdkit_util import canonicalize_smiles


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
                .applymap(canonicalize_smiles)
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
        global_featurizer_input = reactants if self.reaction else smiles

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


class SynFermProductDataset:
    """
    A wrapper around SynFermDataset that simplifies loading a dataset for inference.

    Where the SynfermDataset expects reactionSMILES as inputs which can take some effort to produce, this wrapper expects
    only the SMILES corresponding to the product. From the product SMILES, it infers the reactionSMILES leading to
    the product (this is a bijective mapping). It further annotates the SMILES with information about how
    close it is to the data used to train the model.
    """

    dummy_reactants = [
        [Chem.MolFromSmiles("C"), Chem.MolFromSmiles("C")],
    ]

    def __init__(
        self,
        smiles: Optional[List[str]] = None,
        file_path: Optional[os.PathLike] = None,
        file_smiles_column: str = "SMILES",
        is_reaction: bool = False,
        use_validation: bool = False,
    ):
        """
        At least one of smiles and file_path has to be given. If both are given, the contents are concatenated.
        Args:
            smiles (list): Products of the SynFerm reaction, given as SMILES.
            file_path (os.PathLike): Path to csv file containing products of the SynFerm reaction, given as SMILES.
            file_smiles_column (str): Header of the column containing SMILES. Defaults to "SMILES"
            is_reaction (bool): Whether the SMILES are reaction SMILES (or product SMILES). Defaults to False.
                If reactionSMILES are given, we skip generating reactions and only use the given reaction.
            use_validation (bool): Whether the validation plate data was used in training. This changes which reactions
             will be considered similar to training data. Defaults to False.
        """
        raise NotImplementedError("needs to be adapted from SLAP to SynFerm if needed.")
        # load the SMILES
        self.dataset_0D = None
        self.dataset_1D_aldehyde = None
        self.dataset_1D_slap = None
        self.dataset_2D = None
        self.smiles = []
        if smiles is None and file_path is None:
            raise ValueError("At least one of 'SMILES' and 'file_path' must be given.")
        if smiles:
            self.smiles.extend(smiles)
        if file_path:
            csv_data = pd.read_csv(file_path)
            self.smiles.extend(csv_data[file_smiles_column])

        # generate the reactions
        self.reactions = []
        self.reactants = []
        self.product_types = []
        self.product_idxs = []
        self.invalid_idxs = []
        self.problem_type = []
        self.known_outcomes = []
        # initialize the reaction generator with training data
        reaction_generator = SLAPReactionGenerator()
        if use_validation:
            dataset_path = DATA_ROOT / "reactionSMILESunbalanced_LCMS_2022-08-25.csv"
        else:
            dataset_path = (
                DATA_ROOT
                / "reactionSMILESunbalanced_LCMS_2022-08-25_without_validation.csv"
            )
        reaction_generator.initialize_dataset(
            dataset_path,
            use_cache=True,
        )
        # initialize the similarity calculator
        slaps, aldehydes = reaction_generator.get_reactants_in_dataset()
        similarity_calculator = SLAPReactionSimilarityCalculator(
            slap_reactants=slaps, aldehyde_reactants=aldehydes
        )

        if is_reaction:
            # arrange reactions
            for i, smi in enumerate(self.smiles):
                try:
                    reactants, products = smi.split(">>")
                    slap_reactant, aldehyde_reactant = reactants.split(".")
                    slap_reactant = Chem.MolFromSmiles(
                        canonicalize_smiles(slap_reactant)
                    )
                    aldehyde_reactant = Chem.MolFromSmiles(
                        canonicalize_smiles(aldehyde_reactant)
                    )

                    reactions = [
                        smi,
                    ]
                    reactants = [[slap_reactant, aldehyde_reactant]]
                    product_type = [
                        "user-defined reactionSMILES",
                    ]
                except RuntimeError as e:
                    # if something goes wrong, we use a dummy reaction (to not fuck up the indexing)
                    reactions = [
                        "[C:1].[C:2]>>[C:1][C:2]",
                    ]
                    reactants = self.dummy_reactants
                    product_type = [
                        "dummy",
                    ]
                    self.invalid_idxs.append(i)
                    print(
                        f"WARNING: Could not generate reaction for product with index {i}. Using dummy reaction.\n"
                        f"Error leading to this warning: {e}"
                    )

                self.reactions.extend(reactions)
                self.reactants.extend(reactants)
                self.product_types.extend(product_type)
                self.product_idxs.extend([i for _ in reactions])
            print(
                f"INFO: {len(self.smiles)} reactionSMILES were read. For {len(self.invalid_idxs)} reactionSMILES, "
                f"something went wrong and a dummy reaction was used."
            )

        else:
            # generate the reactions
            for i, smi in enumerate(self.smiles):
                try:
                    (
                        reactions,
                        reactants,
                        product_type,
                    ) = reaction_generator.generate_reactions_for_product(
                        product=smi,
                        return_additional_info=True,
                        return_reaction_smiles=True,
                    )
                except RuntimeError as e:
                    # if we can't generate a reaction, we use a dummy reaction (to not fuck up the indexing)
                    reactions = [
                        "[C:1].[C:2]>>[C:1][C:2]",
                    ]
                    reactants = self.dummy_reactants
                    product_type = [
                        "dummy",
                    ]
                    self.invalid_idxs.append(i)
                    print(
                        f"WARNING: Could not generate reaction for product with index {i}. Using dummy reaction.\n"
                        f"Error leading to this warning: {e}"
                    )

                self.reactions.extend(reactions)
                self.reactants.extend(reactants)
                self.product_types.extend(product_type)
                self.product_idxs.extend([i for _ in reactions])

            print(
                f"INFO: {len(self.smiles)} SMILES were read. For {len(self.invalid_idxs)} SMILES, no valid SLAP reaction could be generated."
            )

        # n.b. the following indices index self.reactants, not self.smiles
        self.idx_known = []
        self.idx_0D = []
        self.idx_1D_aldehyde = []
        self.idx_1D_slap = []
        self.idx_2D_similar = []
        self.idx_2D_dissimilar = []

        # determine the relation to the dataset
        for i, (reactant_pair, product_type) in enumerate(
            zip(self.reactants, self.product_types)
        ):
            if (
                product_type == "dummy"
            ):  # dummy reaction indicates invalid reaction, do not process further
                self.problem_type.append("invalid")
                self.known_outcomes.append(None)
                continue
            else:
                (
                    slap_in_dataset,
                    aldehyde_in_dataset,
                    reaction_in_dataset,
                    reaction_outcomes,
                ) = reaction_generator.reactants_in_dataset(
                    reactant_pair,
                    form_slap_reagent=False,
                    use_cache=True,
                )
                if reaction_in_dataset:
                    self.problem_type.append("known")
                    self.idx_known.append(i)
                elif slap_in_dataset and aldehyde_in_dataset:
                    self.problem_type.append("0D")
                    self.idx_0D.append(i)
                elif slap_in_dataset:  # the SLAP reagent is in training data
                    self.problem_type.append("1D_aldehyde")
                    self.idx_1D_aldehyde.append(i)
                elif aldehyde_in_dataset:  # the aldehyde is in training data
                    self.problem_type.append("1D_SLAP")
                    self.idx_1D_slap.append(i)
                else:
                    # check if the reaction is similar to reactions in the training data
                    if similarity_calculator.is_similar(
                        reactants=reactant_pair,
                        slap_similarity_threshold=0.74,
                        aldehyde_similarity_threshold=0.42,
                    ):
                        self.problem_type.append("2D_similar")
                        self.idx_2D_similar.append(i)
                    else:
                        self.problem_type.append("2D_dissimilar")
                        self.idx_2D_dissimilar.append(i)

            self.known_outcomes.append(reaction_outcomes)  # will be None if not "known"

        # make sure indices are used in canonical order across the class
        self.idx_1D = self.idx_1D_slap + self.idx_1D_aldehyde
        self.idx_2D = self.idx_2D_similar + self.idx_2D_dissimilar

    def process(self, processing_kwargs):
        """
        Process the reactionSMILES to obtain SLAPDatasets.
        After this, the object will expose the following attributes:
            - dataset_0D: SLAPDataset containing the 0D reactions.
            - dataset_1D_slap: SLAPDataset containing the 1D reactions with known aldehyde.
            - dataset_1D_aldehyde: SLAPDataset containing the 1D reactions with known SLAP reagent.
            - dataset_2D: SLAPDataset containing the 2D reactions.
        Note that the "known" reactions are not included in any of these datasets. Their known outcomes are stored in
        self.known_outcomes.
        Depending on the input data, some of these datasets may be empty. In this case, the corresponding attribute will
        be None.

        Args:
            processing_kwargs: A dictionary of keyword arguments passed to the SLAPDataset constructor.
                kwargs are given separately for each dataset. Hence, a dictionary of the form
                {"dataset_0D": kwargs_0D,...} is expected.
        """

        # all reactions with 0D problem type go into the same dataset
        reactions_0D = [self.reactions[i] for i in self.idx_0D]
        if len(reactions_0D) > 0:
            try:
                self.dataset_0D = SynFermDataset(
                    name="0D", **processing_kwargs["dataset_0D"]
                )
            except KeyError:
                self.dataset_0D = SynFermDataset(name="0D")
            self.dataset_0D.process(reactions_0D)
        # all reactions with 1D_aldehyde problem type go into the same dataset
        reactions_1D_aldehyde = [self.reactions[i] for i in self.idx_1D_aldehyde]
        if len(reactions_1D_aldehyde) > 0:
            try:
                self.dataset_1D_aldehyde = SynFermDataset(
                    name="1D_aldehyde", **processing_kwargs["dataset_1D_aldehyde"]
                )
            except KeyError:
                self.dataset_1D_aldehyde = SynFermDataset(name="1D_aldehyde")
            self.dataset_1D_aldehyde.process(reactions_1D_aldehyde)
        # all reactions with 1D_SLAP problem type go into the same dataset
        reactions_1D_slap = [self.reactions[i] for i in self.idx_1D_slap]
        if len(reactions_1D_slap) > 0:
            try:
                self.dataset_1D_slap = SynFermDataset(
                    name="1D_SLAP", **processing_kwargs["dataset_1D_slap"]
                )
            except KeyError:
                self.dataset_1D_slap = SynFermDataset(name="1D_SLAP")
            self.dataset_1D_slap.process([self.reactions[i] for i in self.idx_1D_slap])
        # all reactions with 2D problem type go into the same dataset
        reactions_2D = [self.reactions[i] for i in self.idx_2D]
        if len(reactions_2D) > 0:
            try:
                self.dataset_2D = SynFermDataset(
                    name="2D", **processing_kwargs["dataset_2D"]
                )
            except KeyError:
                self.dataset_2D = SynFermDataset(name="2D")
            self.dataset_2D.process(reactions_2D)
