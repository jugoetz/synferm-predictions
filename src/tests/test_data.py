import os
from unittest import TestCase

import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

from src.data.dataloader import SynFermDataset
from src.data.featurizers import OneHotEncoder
from src.util.rdkit_util import canonicalize_smiles
from src.util.definitions import DATA_ROOT


class TestDataCaching(TestCase):
    def setUp(self) -> None:
        self.data = SynFermDataset(
            name="example_sf.csv",
            raw_dir=DATA_ROOT,
            save_dir=(DATA_ROOT / "cache"),
            reaction=True,
            smiles_columns=["reaction_smiles_atom_mapped"],
            label_columns=["binary_A"],
            graph_type="bond_edges",
            global_features=None,
            global_features_file=None,
            featurizers="custom",
            task="binary",
            force_reload=True,  # this will be cached freshly
        )

        self.cached_data = SynFermDataset(
            name="example_sf.csv",
            raw_dir=DATA_ROOT,
            save_dir=(DATA_ROOT / "cache"),
            reaction=True,
            smiles_columns=["reaction_smiles_atom_mapped"],
            label_columns=["binary_A"],
            graph_type="bond_edges",
            global_features=None,
            global_features_file=None,
            featurizers="custom",
            task="binary",
            force_reload=False,  # this will be loaded from cache
        )

        assert (
            self.data.hash == self.cached_data.hash
        )  # assert that hashes actually align

    def test_reload_gives_same_data_length(self):
        self.assertEqual(len(self.data.graphs), len(self.cached_data.graphs))

    def test_reload_gives_same_global_features(self):
        self.assertTrue(
            torch.all(self.data.global_features == self.cached_data.global_features)
        )


class TestOneHotEncoder(TestCase):
    def setUp(self):
        self.encoder = OneHotEncoder()
        self.smiles = ["C", "CC", "CCC", "CCCC"]
        self.encoder.add_dimension(self.smiles)
        self.encoder.add_dimension(self.smiles)
        self.encoder.add_dimension(self.smiles)

    def test_save_state_dict(self):
        """Test that a save/load roundtrip works"""
        self.encoder.save_state_dict("test_save_state_dict.json")
        encoder_from_file = OneHotEncoder()
        encoder_from_file.load_state_dict("test_save_state_dict.json")
        os.remove("test_save_state_dict.json")
        self.assertEqual(self.encoder.n_dimensions, encoder_from_file.n_dimensions)

    def test_process(self):
        for i, smi in enumerate(self.smiles):
            with self.subTest(smi=smi):
                ohe = self.encoder.process(smi, smi, smi)
                self.assertTrue(
                    np.all(
                        ohe.nonzero()
                        == np.array([i, i + len(self.smiles), i + 2 * len(self.smiles)])
                    )
                )

    def test_process_after_save_load(self):
        self.encoder.save_state_dict("test_process_after_save_load.json")
        encoder_from_file = OneHotEncoder()
        encoder_from_file.load_state_dict("test_process_after_save_load.json")
        os.remove("test_process_after_save_load.json")
        for i, smi in enumerate(self.smiles):
            with self.subTest(smi=smi):
                ohe = encoder_from_file.process(smi, smi, smi)
                self.assertTrue(
                    np.all(
                        ohe.nonzero()
                        == np.array([i, i + len(self.smiles), i + 2 * len(self.smiles)])
                    )
                )
