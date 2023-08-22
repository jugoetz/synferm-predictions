import json
import os
from functools import partial
from typing import Union

import numpy as np
import torch
from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
from dgllife.utils.featurizers import (
    BaseAtomFeaturizer,
    BaseBondFeaturizer,
    ConcatFeaturizer,
    atom_type_one_hot,
    atom_total_degree_one_hot,
    atom_formal_charge_one_hot,
    atom_chiral_tag_one_hot,
    atom_total_num_H_one_hot,
    atom_hybridization_one_hot,
    atom_is_aromatic,
    atom_is_in_ring,
    atom_mass,
    bond_type_one_hot,
    bond_is_conjugated,
    bond_is_in_ring,
    bond_stereo_one_hot,
)

from rdkit import Chem
from rdkit.Chem import Mol, MolFromSmiles
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from rdkit.DataStructs import ConvertToNumpyArray

from src.util.rdkit_util import canonicalize_smiles


class SLAPAtomFeaturizer(BaseAtomFeaturizer):
    """An atom featurizer tailored to SLAP chemistry.

    The atom features are:

    * **One hot encoding of the atom type**. 12 elements ["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Br", "I",] and "unknown" are supported.
    * **One hot encoding of the atom degree**. The supported possibilities are ``0 - 5``.
    * **One hot encoding of the formal charge of the atom**. [-1, 0, 1] and "unknown" are supported.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities are ``0 - 4``.
    * **One hot encoding of the atom hybridization**. The supported possibilities are
      ``S``, ``SP``, ``SP2``, ``SP3`` and "unknown".
    * **Whether the atom is aromatic**.
    * **Whether the atom is in a ring**.

    In total, the feature vector has length 35.

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.
    """

    def __init__(self, atom_data_field="h"):
        allowable_atoms = [
            "H",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Si",
            "P",
            "S",
            "Cl",
            "Br",
            "I",
        ]

        featurizer_funcs = {
            atom_data_field: ConcatFeaturizer(
                [
                    partial(
                        atom_type_one_hot,
                        allowable_set=allowable_atoms,
                        encode_unknown=True,
                    ),
                    partial(
                        atom_total_degree_one_hot,
                        allowable_set=list(range(6)),
                        encode_unknown=False,
                    ),
                    partial(
                        atom_formal_charge_one_hot,
                        allowable_set=[-1, 0, 1],
                        encode_unknown=True,
                    ),
                    partial(
                        atom_total_num_H_one_hot,
                        allowable_set=list(range(5)),
                        encode_unknown=False,
                    ),
                    partial(
                        atom_hybridization_one_hot,
                        allowable_set=[
                            Chem.rdchem.HybridizationType.S,
                            Chem.rdchem.HybridizationType.SP,
                            Chem.rdchem.HybridizationType.SP2,
                            Chem.rdchem.HybridizationType.SP3,
                        ],
                        encode_unknown=True,
                    ),
                    atom_is_aromatic,
                    atom_is_in_ring,
                ]
            )
        }

        super().__init__(featurizer_funcs=featurizer_funcs)


class SLAPBondFeaturizer(BaseBondFeaturizer):
    """A bond featurizer tailored to SLAP chemistry.

    Actually, this turned out to be identical to dgllife's canonical bond featurizer.
    We still keep it to be safe in case they change it.

    The bond features are:
    * **One hot encoding of the bond type**. The supported bond types are
      ``SINGLE``, ``DOUBLE``, ``TRIPLE``, ``AROMATIC``.
    * **Whether the bond is conjugated.**.
    * **Whether the bond is in a ring.**
    * **One hot encoding of the stereo configuration of a bond**. The supported bond stereo
      configurations include ``STEREONONE``, ``STEREOANY``, ``STEREOZ``, ``STEREOE``,
      ``STEREOCIS``, ``STEREOTRANS``.

    In total, the feature vector has length 12.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**
    """

    def __init__(self, bond_data_field="e"):
        featurizer_funcs = {
            bond_data_field: ConcatFeaturizer(
                [
                    bond_type_one_hot,
                    bond_is_conjugated,
                    bond_is_in_ring,
                    bond_stereo_one_hot,
                ]
            )
        }

        super().__init__(featurizer_funcs=featurizer_funcs, self_loop=False)


class ChempropAtomFeaturizer(BaseAtomFeaturizer):
    """A DGLLife implementation of the atom featurizer used in Chemprop.

    The atom features include:

    * **One hot encoding of the atom type by atomic number**. Atomic numbers 1 - 100 are supported.
    * **One hot encoding of the atom degree**. The supported possibilities
      include ``0 - 6``.
    * **Formal charge of the atom**.
    * **One hot encoding of the atom hybridization**. The supported possibilities include
      ``SP``, ``SP2``, ``SP3``, ``SP3D``, ``SP3D2``.
    * **Whether the atom is aromatic**.
    * **One hot encoding of the number of total Hs on the atom**. The supported possibilities
      include ``0 - 4``.
    * **Mass of the atom**. Divided by 100, not onehot encoded.

    **We assume the resulting DGLGraph will not contain any virtual nodes.**

    Parameters
    ----------
    atom_data_field : str
        Name for storing atom features in DGLGraphs, default to 'h'.

    """

    def __init__(self, atom_data_field="h"):
        allowable_atoms = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
            "Po",
            "At",
            "Rn",
            "Fr",
            "Ra",
            "Ac",
            "Th",
            "Pa",
            "U",
            "Np",
            "Pu",
            "Am",
            "Cm",
            "Bk",
            "Cf",
            "Es",
            "Fm",
        ]

        super(ChempropAtomFeaturizer, self).__init__(
            featurizer_funcs={
                atom_data_field: ConcatFeaturizer(
                    [
                        partial(
                            atom_type_one_hot,
                            allowable_set=allowable_atoms,
                            encode_unknown=True,
                        ),
                        partial(
                            atom_total_degree_one_hot,
                            allowable_set=list(range(6)),
                            encode_unknown=True,
                        ),
                        partial(
                            atom_formal_charge_one_hot,
                            allowable_set=[-1, -2, 1, 2, 0],
                            encode_unknown=True,
                        ),
                        partial(atom_chiral_tag_one_hot, encode_unknown=True),
                        # note that this encode_unknown=True does not make sense as the chiral tags already cover this case. But we follow the ref implementation.
                        partial(
                            atom_total_num_H_one_hot,
                            allowable_set=list(range(5)),
                            encode_unknown=True,
                        ),
                        partial(atom_hybridization_one_hot, encode_unknown=True),
                        atom_is_aromatic,
                        atom_mass,
                    ]
                )
            }
        )


class ChempropBondFeaturizer(BaseBondFeaturizer):
    """A DGLLife implementation of the bond featurizer used in Chemprop.

    The bond features include:
    * A zero if the bond is not None. This seems really useless, but we follow the ref implementation
    * **One hot encoding of the bond type**. The supported bond types are
      ``SINGLE``, ``DOUBLE``, ``TRIPLE``, ``AROMATIC``.
    * **Whether the bond is conjugated.**.
    * **Whether the bond is in a ring of any size.**
    * **One hot encoding of the stereo configuration of a bond**. The supported bond stereo
      configurations include ``STEREONONE``, ``STEREOANY``, ``STEREOZ``, ``STEREOE``,
      ``STEREOCIS``, ``STEREOTRANS``.

    **We assume the resulting DGLGraph will be created with :func:`smiles_to_bigraph` without
    self loops.**

    Parameters
    ----------
    bond_data_field : str
        Name for storing bond features in DGLGraphs, default to ``'e'``.


    See Also
    --------
    BaseBondFeaturizer
    WeaveEdgeFeaturizer
    PretrainBondFeaturizer
    AttentiveFPBondFeaturizer
    PAGTNEdgeFeaturizer
    """

    def __init__(self, bond_data_field="e"):
        super(ChempropBondFeaturizer, self).__init__(
            featurizer_funcs={
                bond_data_field: ConcatFeaturizer(
                    [
                        lambda bond: [0],
                        bond_type_one_hot,
                        bond_is_conjugated,
                        bond_is_in_ring,
                        partial(bond_stereo_one_hot, encode_unknown=True)
                        # encode_unknown seems unnecessary as one of the options is STEREONONE. But we still follow the ref implementation.
                    ]
                )
            },
            self_loop=False,
        )


class RDKitMorganFingerprinter:
    """
    Molecule featurization with Morgan fingerprint
    """

    def __init__(self, radius=3, n_bits=1024, **kwargs):
        self.radius = radius
        self.n_bits = n_bits

    def process(self, *smiles):
        """
        Process one or more SMILES and return a one-hot vector.
        Outputs will be concatenated into one array of length feat_size x number_of_smiles.
        """
        arrays = []
        for smi in smiles:
            mol = MolFromSmiles(smi)
            fp = GetMorganFingerprintAsBitVect(
                mol, radius=self.radius, nBits=self.n_bits
            )
            arr = np.zeros(self.n_bits)
            ConvertToNumpyArray(fp, arr)
            arrays.append(arr)
        return np.concatenate(arrays, axis=0)

    @property
    def feat_size(self) -> int:
        return self.n_bits


class OneHotEncoder:
    """
    Molecule featurization with one-hot vector
    """

    def __init__(self):
        self.classes = {}

    def add_dimension(self, smiles: list):
        """
        Initialize a new dimension in the encoder.

        Expects a list of SMILES strings that are all options for the new dimension.
        SMILES will be canonicalized before being added to the encoder.

        Args:
            smiles (list): List of SMILES strings.
        """
        new_dimension = len(self.classes.keys())
        class_values = {}
        for smi in smiles:
            # canonicalize
            canonical_smiles = canonicalize_smiles(smi)
            # check if the smiles is already in the dictionary
            if canonical_smiles not in class_values.keys():
                # if not, add
                class_values[canonical_smiles] = len(class_values.keys())
            self.classes[new_dimension] = class_values
        return

    def process(self, *smiles):
        """
        Process one or more smiles and return a one-hot vector.
        Expects a number of input smiles equal to the number of dimensions in the encoder (.n_dimensions)
        """
        # check if the encoder has been initialized
        if self.n_dimensions == 0:
            raise RuntimeError(
                "OneHotEncoder must be initialized by calling add_dimension() at least once before processing."
            )
        # check if the number of smiles is equal to the number of dimensions
        if len(smiles) != self.n_dimensions:
            raise ValueError(
                "Expected {} smiles, got {}".format(self.n_dimensions, len(smiles))
            )
        feat_sizes = self.feat_size_by_dimension
        one_hot_vector = np.zeros(sum(feat_sizes))

        for dimension, smi in enumerate(smiles):
            # canonicalize the smiles
            canonical_smiles = canonicalize_smiles(smi)
            # get the one-hot index
            one_hot_index = self.classes[dimension][canonical_smiles]
            # get the one-hot vector
            one_hot_vector[one_hot_index + sum(feat_sizes[:dimension])] = 1

        return one_hot_vector

    def save_state_dict(self, path):
        """Save the state of the encoder to a JSON file."""
        # essentially save self.classes
        # there is a catch here: json serializes int dict keys to strings. We take care of that in the load_state_dict
        # method
        with open(path, "w") as f:
            json.dump(self.classes, f)
        return

    def load_state_dict(self, path):
        """Load the state of the encoder from a JSON file."""
        # essentially load self.classes
        # there is a catch here: json serializes int dict keys to strings. We interpret the keys as ints here.
        with open(path, "r") as f:
            state_dict_with_string_keys = json.load(f)
        self.classes = {int(k): v for k, v in state_dict_with_string_keys.items()}
        return

    @property
    def n_dimensions(self) -> int:
        return len(self.classes.keys())

    @property
    def feat_size(self) -> int:
        return sum(self.feat_size_by_dimension)

    @property
    def feat_size_by_dimension(self) -> list:
        return [len(v.keys()) for v in self.classes.values()]


class RDKit2DGlobalFeaturizer:
    """
    Molecule featurization with RDKit 2D features. Uses descriptastorus (https://github.com/bp-kelley/descriptastorus).
    """

    def __init__(self, normalize: bool = True, **kwargs):
        """
        Args:
            normalize (bool): If True, normalize feature values. The normalization uses a CDF fitted on a NIBR compound
                                catalogue. Default True.
        """
        if normalize:
            self.features_generator = MakeGenerator(("rdkit2dnormalized",))
        else:
            self.features_generator = MakeGenerator(("rdkit2d",))
        # we only need to evaluate feat_size once as the generator is not intended for change during runtime
        self._feat_size = (
            len(self.features_generator.process("C")) - 1
        )  # -1 for the initial 'True' that we do not return

    def process(self, *smiles):
        feat = np.zeros((len(smiles) * self.feat_size), dtype="float32")
        for i, smi in enumerate(smiles):
            features = self.features_generator.process(smi)
            if features is None:  # fail
                raise ValueError(
                    f"ERROR: could not generate rdkit features for SMILES '{smi}'"
                )
            else:
                feat[i * self.feat_size : (i + 1) * self.feat_size] = features[
                    1:
                ]  # do not return the initial 'True'.
        return feat

    @property
    def feat_size(self) -> int:
        return self._feat_size


def dummy_atom_featurizer(m: Mol):
    """For testing. Featurizes every atom with its index"""
    feats = [[a.GetIdx()] for a in m.GetAtoms()]
    return {"x": torch.FloatTensor(feats)}


def dummy_bond_featurizer(m: Mol):
    """For testing. Featurizes every bond with its index"""
    feats = [[b.GetIdx()] for b in m.GetBonds() for _ in range(2)]
    return {"e": torch.FloatTensor(feats)}


class FromFileFeaturizer:
    """
    Featurizer that loads features from a file. The file should be a JSON file with the following format:
    {
        "smiles1": [feature1, feature2, ...],
        "smiles2": [feature1, feature2, ...],
        ...
    }
    """

    def __init__(self, filename: Union[str, os.PathLike], **kwargs):
        """
        Args:
            filename (str or os.Pathlike): Path to the file containing the features.
        """
        self.filename = filename
        self._features = None

    def process(self, *smiles):
        """
        Process one or more SMILES and return the features.
        """
        if self._features is None:
            with open(self.filename, "r") as f:
                self._features = json.load(f)

        return np.array(
            [self._features[canonicalize_smiles(smi)] for smi in smiles]
        ).flatten()

    @property
    def feat_size(self) -> int:
        if self._features is None:
            with open(self.filename, "r") as f:
                self._features = json.load(f)
        return len(list(self._features.values())[0])
