import os
import pathlib
import string
import random
from typing import List, Tuple, Optional, Union

import dgl
import numpy as np
from dgl.data import DGLDataset
import torch
import pandas as pd
from dgllife.utils.featurizers import CanonicalAtomFeaturizer, CanonicalBondFeaturizer
from rdkit import Chem

from src.data.grapher import build_cgr, build_mol_graph
from src.data.featurizers import (
    ChempropAtomFeaturizer,
    ChempropBondFeaturizer,
    SLAPAtomFeaturizer,
    SLAPBondFeaturizer,
    RDKit2DGlobalFeaturizer,
    RDKitMorganFingerprinter,
    OneHotEncoder,
    FromFileFeaturizer,
)
from src.data.util import SLAPReactionGenerator, SLAPReactionSimilarityCalculator
from src.util.definitions import DATA_ROOT, LOG_DIR
from src.util.rdkit_util import canonicalize_smiles


def collate_fn(
    batch: List[Tuple[dgl.DGLGraph, list, torch.tensor]]
) -> Tuple[dgl.DGLGraph, Optional[torch.tensor], torch.tensor]:
    """Collate a list of samples into a batch, i.e. into a single tuple representing the entire batch"""

    graphs, global_features, labels = map(list, zip(*batch))

    batched_graphs = dgl.batch(graphs)
    if len(global_features[0]) == 0:
        batched_global_features = None
    else:
        batched_global_features = torch.tensor(global_features, dtype=torch.float32)

    if labels[0] is None:
        batched_labels = None
    else:
        batched_labels = torch.tensor(labels)

    return batched_graphs, batched_global_features, batched_labels


class SLAPDataset(DGLDataset):
    """
    SLAP Dataset

    Can load a set of data points containing either a reactionSMILES or a SMILES of a single molecule and one
    column containing labels.

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
        smiles_columns: tuple = ("SMILES",),
        label_column: Optional[str] = "label",
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
            label_column: Header of the column in data file that contains the labels
            save_dir: Directory to save the processed data set. If None, `raw_dir` is used. Default None.
            force_reload: Reload data set, ignoring cache. Default False.
            verbose: Whether to provide verbose output
        """

        if global_features is None:
            global_features = []

        self.reaction = reaction
        self.label_column = label_column
        self.smiles_columns = smiles_columns
        self.graph_type = graph_type  # whether to form BE- or BN-graph
        self.global_features = (
            []
        )  # container for global features e.g. rdkit or fingerprints
        self.global_featurizers = []  # container for global featurizers
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
            self.atom_featurizer = SLAPAtomFeaturizer(atom_data_field="x")
            self.bond_featurizer = SLAPBondFeaturizer(bond_data_field="e")
        else:
            raise ValueError("Unexpected value for 'featurizers'")

        # global featurizer(s)
        if "RDKit" in global_features:
            self.global_featurizers.append(RDKit2DGlobalFeaturizer(normalize=True))
        if "FP" in global_features:
            self.global_featurizers.append(
                RDKitMorganFingerprinter(radius=6, n_bits=1024)
            )
        if "OHE" in global_features:
            self.global_featurizers.append(OneHotEncoder())
        if "fromFile" in global_features:
            self.global_featurizers.append(
                FromFileFeaturizer(filename=global_features_file)
            )

        super(SLAPDataset, self).__init__(
            name=name,
            url=url,
            raw_dir=raw_dir,
            save_dir=save_dir,
            force_reload=force_reload,
            verbose=verbose,
        )

    def _load(self):
        # Parent class calls this method during initialization. We don't want a call to process() here, so we remove
        # this functionality. We lose caching with this, but don't need it here anyway
        return

    def process(self, smiles: Optional[List[str]] = None):
        """
        Read reactionSMILES/SMILES data from csv file or optional argument and generate molecular graph / CGR.
        If the reaction argument was True during __init__, we expect reactionSMILES and produce the condensed graph
        of reaction (CGR). Else, we expect a single SMILES (product or intermediate) and produce the molecular graph.

        Args:
            smiles (list): A flat list of reactionSMILES or SMILES. If given, files specified during __init__ are
                ignored. Defaults to None.
        """
        csv_data = None
        if not smiles:
            csv_data = pd.read_csv(self.raw_path)
            smiles = [csv_data[s] for s in self.smiles_columns]

            # Currently, we don't support having multiple inputs per data point
            if len(smiles) > 1:
                raise NotImplementedError("Multi-input prediction is not implemented.")
            # ...which allows us to do this:
            smiles = smiles[0]

        # clear previous data
        self.graphs = []
        self.global_features = [np.array(()) for _ in smiles]
        self.labels = []

        if self.reaction:
            self.graphs = [
                build_cgr(
                    s,
                    self.atom_featurizer,
                    self.bond_featurizer,
                    mode="reac_diff",
                    graph_type=self.graph_type,
                )
                for s in smiles
            ]

        else:
            self.graphs = [
                build_mol_graph(
                    s,
                    self.atom_featurizer,
                    self.bond_featurizer,
                    graph_type=self.graph_type,
                )
                for s in smiles
            ]

        if len(self.global_featurizers) > 0:
            if self.reaction:
                # if it is a reaction, we featurize for both reactants, then concatenate
                for global_featurizer in self.global_featurizers:
                    if isinstance(global_featurizer, OneHotEncoder):
                        # for OHE, we need to set up the encoder with the list(s) of smiles it should encode
                        if self.global_featurizer_state_dict_path:
                            # if we already have a state dict, we can load it
                            global_featurizer.load_state_dict(
                                self.global_featurizer_state_dict_path
                            )
                        else:
                            # sanity check: the OneHotEncoder should not have been initialized before
                            assert global_featurizer.n_dimensions == 0
                            # else, we need to set up the encoder with the list(s) of smiles it should encode
                            smiles_reactant1 = [s.split(".")[0] for s in smiles]
                            smiles_reactant2 = [
                                s.split(">>")[0].split(".")[1] for s in smiles
                            ]
                            global_featurizer.add_dimension(smiles_reactant1)
                            global_featurizer.add_dimension(smiles_reactant2)
                            # and save the state of the encoder, to use it later for inference
                            self.global_featurizer_state_dict_path = LOG_DIR / (
                                "OHE_state_dict_"
                                + "".join(random.choices(string.ascii_letters, k=16))
                                + ".json"
                            )
                            global_featurizer.save_state_dict(
                                self.global_featurizer_state_dict_path
                            )

                    self.global_features = [
                        np.concatenate(i)
                        for i in zip(
                            self.global_features,
                            [
                                global_featurizer.process(*s.split(">>")[0].split("."))
                                for s in smiles
                            ],
                        )
                    ]

            else:
                # if instead we get a single molecule, we just featurize for that
                for global_featurizer in self.global_featurizers:
                    if isinstance(global_featurizer, OneHotEncoder):
                        if self.global_featurizer_state_dict_path:
                            # if we already have a state dict, we can load it
                            global_featurizer.load_state_dict(
                                self.global_featurizer_state_dict_path
                            )
                        else:
                            # else, we need to set up the encoder with the list(s) of smiles it should encode
                            global_featurizer.add_dimension(smiles)
                            # and save the state of the encoder, to use it later for inference
                            self.global_featurizer_state_dict_path = (
                                LOG_DIR / "OHE_state_dict_"
                                + "".join(random.choices(string.ascii_letters, k=16))
                                + ".json"
                            )
                            global_featurizer.save_state_dict(
                                self.global_featurizer_state_dict_path
                            )

                    self.global_features = [
                        np.concatenate(i)
                        for i in zip(
                            self.global_features,
                            [global_featurizer.process(s) for s in smiles],
                        )
                    ]

        if self.label_column is not None and csv_data is not None:
            self.labels = csv_data[self.label_column].values.tolist()
        else:
            # allow having no labels, e.g. for inference
            self.labels = [None for _ in smiles]

        # little safety net
        assert len(self.graphs) == len(self.labels) == len(self.global_features)

    def __getitem__(self, idx):
        """Get graph and label by index

        Args:
            idx (int): Item index

        Returns:
            (dgl.DGLGraph, Tensor)
        """
        return self.graphs[idx], self.global_features[idx], self.labels[idx]

    def __len__(self):
        """Number of graphs in the dataset"""
        return len(self.graphs)

    @property
    def atom_feature_size(self):
        n_atom_features = self.atom_featurizer.feat_size()
        if self.reaction:
            return 2 * n_atom_features  # CGR has 2 x features
        else:
            return n_atom_features

    @property
    def bond_feature_size(self):
        n_bond_features = self.bond_featurizer.feat_size()
        if self.reaction:
            return 2 * n_bond_features  # CGR has 2 x features
        else:
            return n_bond_features

    @property
    def feature_size(self):
        return self.atom_feature_size + self.bond_feature_size

    @property
    def global_feature_size(self):
        n_global_features = 0
        for global_featurizer in self.global_featurizers:
            if self.reaction and not isinstance(global_featurizer, OneHotEncoder):
                # for 2 reactants we have 2 x features (except for the OHE which always encorporates all inputs in feat size)
                n_global_features += 2 * global_featurizer.feat_size
            else:
                n_global_features += global_featurizer.feat_size
        return n_global_features


class SLAPProductDataset:
    """
    A wrapper around SLAPDataset that simplifies loading a dataset for inference.

    Where the SLAPDataset expects reactionSMILES as inputs which can take some effort to produce, this wrapper expects
    only the SMILES corresponding to the product. From the product SMILES, it infers possible reactionSMILES leading to
    the product (this can be two different reactionSMILES). It further annotates the SMILES with information about how
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
            smiles (list): Products of the SLAP reaction, given as SMILES.
            file_path (os.PathLike): Path to csv file containing products of the SLAP reaction, given as SMILES.
            file_smiles_column (str): Header of the column containing SMILES. Defaults to "SMILES"
            is_reaction (bool): Whether the SMILES are reaction SMILES (or product SMILES). Defaults to False.
                If reactionSMILES are given, we skip generating reactions and only use the given reaction.
            use_validation (bool): Whether the validation plate data was used in training. This changes which reactions
             will be considered similar to training data. Defaults to False.
        """
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
                self.dataset_0D = SLAPDataset(
                    name="0D", **processing_kwargs["dataset_0D"]
                )
            except KeyError:
                self.dataset_0D = SLAPDataset(name="0D")
            self.dataset_0D.process(reactions_0D)
        # all reactions with 1D_aldehyde problem type go into the same dataset
        reactions_1D_aldehyde = [self.reactions[i] for i in self.idx_1D_aldehyde]
        if len(reactions_1D_aldehyde) > 0:
            try:
                self.dataset_1D_aldehyde = SLAPDataset(
                    name="1D_aldehyde", **processing_kwargs["dataset_1D_aldehyde"]
                )
            except KeyError:
                self.dataset_1D_aldehyde = SLAPDataset(name="1D_aldehyde")
            self.dataset_1D_aldehyde.process(reactions_1D_aldehyde)
        # all reactions with 1D_SLAP problem type go into the same dataset
        reactions_1D_slap = [self.reactions[i] for i in self.idx_1D_slap]
        if len(reactions_1D_slap) > 0:
            try:
                self.dataset_1D_slap = SLAPDataset(
                    name="1D_SLAP", **processing_kwargs["dataset_1D_slap"]
                )
            except KeyError:
                self.dataset_1D_slap = SLAPDataset(name="1D_SLAP")
            self.dataset_1D_slap.process([self.reactions[i] for i in self.idx_1D_slap])
        # all reactions with 2D problem type go into the same dataset
        reactions_2D = [self.reactions[i] for i in self.idx_2D]
        if len(reactions_2D) > 0:
            try:
                self.dataset_2D = SLAPDataset(
                    name="2D", **processing_kwargs["dataset_2D"]
                )
            except KeyError:
                self.dataset_2D = SLAPDataset(name="2D")
            self.dataset_2D.process(reactions_2D)
